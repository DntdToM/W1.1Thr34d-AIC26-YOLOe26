"""Multimodal vector embedding generation module supporting SigLIP 2 and BGE-M3."""

import os
import logging
from typing import List, Union, Dict, Any, Optional
import yaml
import torch
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EmbeddingGenerator")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class EmbeddingGenerator:
    """Multimodal vector embedding generator for image features (SigLIP 2) and dense text (BGE-M3)."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        models_cfg = self.config.get("models", {})
        
        vision_cfg = models_cfg.get("vision_embedding", {})
        text_cfg = models_cfg.get("text_embedding", {})

        self.vision_model_name = vision_cfg.get("name", "google/siglip-base-patch16-224")
        self.text_model_name = text_cfg.get("name", "BAAI/bge-m3")
        self.use_fp16 = self.config.get("preprocessing", {}).get("use_fp16", True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if (torch.cuda.is_available() and self.use_fp16) else torch.float32
        
        logger.info(f"EmbeddingGenerator initialized on device '{self.device}' (fp16={self.torch_dtype == torch.float16}).")

        self.vision_model = None
        self.vision_processor = None
        self.text_model = None
        self.text_tokenizer = None

        self._init_vision_model()
        self._init_text_model()

    def _init_vision_model(self):
        """Initialize SigLIP 2 vision model architecture."""
        if self.vision_model is None:
            try:
                from transformers import AutoImageProcessor, AutoModel
                logger.info(f"Loading SigLIP 2 model from '{self.vision_model_name}' (dtype={self.torch_dtype})...")
                self.vision_processor = AutoImageProcessor.from_pretrained(self.vision_model_name)
                self.vision_model = AutoModel.from_pretrained(
                    self.vision_model_name,
                    torch_dtype=self.torch_dtype
                ).to(self.device)
                self.vision_model.eval()
                logger.info("SigLIP 2 model loaded successfully.")
            except Exception as e:
                logger.error(f"SigLIP 2 model initialization failed for '{self.vision_model_name}': {e}")
                raise e

    def _init_text_model(self):
        """Initialize BAAI/bge-m3 text embedding model."""
        if self.text_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading BGE-M3 text model from '{self.text_model_name}'...")
                model_kwargs = {"torch_dtype": self.torch_dtype} if torch.cuda.is_available() else {}
                self.text_model = SentenceTransformer(
                    self.text_model_name,
                    device=str(self.device),
                    model_kwargs=model_kwargs
                )
                logger.info("BGE-M3 model loaded successfully via SentenceTransformer.")
            except Exception:
                logger.info("Falling back to HuggingFace AutoModel for BGE-M3...")
                from transformers import AutoTokenizer, AutoModel
                self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
                self.text_model = AutoModel.from_pretrained(
                    self.text_model_name,
                    torch_dtype=self.torch_dtype
                ).to(self.device)
                self.text_model.eval()

    def get_image_embeddings_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 64) -> np.ndarray:
        """Extract L2-normalized image embeddings batch [N, 768] using SigLIP 2."""
        if not images:
            return np.empty((0, 768), dtype=np.float32)

        self._init_vision_model()

        if len(images) > batch_size:
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                chunk = images[i:i + batch_size]
                chunk_emb = self._embed_image_chunk(chunk)
                all_embeddings.append(chunk_emb)
                logger.info(f"Image embedding batch {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size} processed ({len(chunk)} images)")
            return np.concatenate(all_embeddings, axis=0)
        
        return self._embed_image_chunk(images)

    def _embed_image_chunk(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """Embed an image chunk into a numpy array [N, 768]."""
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                if os.path.exists(img):
                    loaded_images.append(Image.open(img).convert("RGB"))
                else:
                    logger.warning(f"Image file path not found: {img}. Utilizing blank image placeholder.")
                    loaded_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            elif isinstance(img, Image.Image):
                loaded_images.append(img.convert("RGB"))

        try:
            inputs = self.vision_processor(images=loaded_images, return_tensors="pt")
            inputs = {
                k: (v.to(device=self.device, dtype=self.torch_dtype) if torch.is_floating_point(v) else v.to(self.device))
                for k, v in inputs.items()
            }

            with torch.no_grad():
                if hasattr(self.vision_model, "get_image_features"):
                    out = getattr(self.vision_model, "get_image_features")(**inputs)
                else:
                    out = self.vision_model(**inputs)

                if hasattr(out, "image_embeds"):
                    feats = getattr(out, "image_embeds")
                elif hasattr(out, "pooler_output"):
                    feats = getattr(out, "pooler_output")
                elif isinstance(out, torch.Tensor):
                    feats = out
                else:
                    feats = out[0]

                feats = feats / feats.norm(dim=-1, keepdim=True)

            embeddings = feats.float().cpu().numpy().astype(np.float32)
            return embeddings
        except Exception as e:
            logger.error(f"Image batch embedding extraction error: {e}")
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_image_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Extract a single image vector embedding [768,]."""
        batch_res = self.get_image_embeddings_batch([image])
        return batch_res[0] if len(batch_res) > 0 else np.zeros((768,), dtype=np.float32)

    def get_text_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Extract L2-normalized dense text embeddings batch [N, 1024] using BGE-M3."""
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        self._init_text_model()

        try:
            if hasattr(self.text_model, "encode"):
                embeddings = self.text_model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                res_array = np.array(embeddings, dtype=np.float32)
            else:
                inputs = self.text_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    out = self.text_model(**inputs)
                    token_embeddings = out[0]
                    attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
                    
                res_array = pooled.float().cpu().numpy().astype(np.float32)
            return res_array
        except Exception as e:
            logger.error(f"Text batch embedding extraction error: {e}")
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Extract a single text vector embedding [1024,]."""
        batch_res = self.get_text_embeddings_batch([text])
        return batch_res[0] if len(batch_res) > 0 else np.zeros((1024,), dtype=np.float32)


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    dummy_img = Image.new("RGB", (224, 224), (255, 0, 0))
    img_emb = generator.get_image_embedding(dummy_img)
    print(f"Image Embedding Shape: {img_emb.shape}, Dtype: {img_emb.dtype}")

    text_emb = generator.get_text_embedding("Smart video retrieval system")
    print(f"Text Embedding Shape: {text_emb.shape}, Dtype: {text_emb.dtype}")
