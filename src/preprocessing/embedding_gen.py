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
        """Initialize BAAI/bge-m3 text embedding model for Dense and Sparse."""
        if self.text_model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
                logger.info(f"Loading BGEM3FlagModel from '{self.text_model_name}'...")
                self.text_model = BGEM3FlagModel(
                    self.text_model_name,
                    use_fp16=self.use_fp16 and torch.cuda.is_available()
                )
                self.use_flag_embedding = True
                logger.info("BGEM3FlagModel loaded successfully (Dense + Sparse).")
            except Exception as e:
                logger.warning(f"FlagEmbedding not available ({e}). Falling back to SentenceTransformer (Dense only).")
                self.use_flag_embedding = False
                from sentence_transformers import SentenceTransformer
                model_kwargs = {"torch_dtype": self.torch_dtype} if torch.cuda.is_available() else {}
                self.text_model = SentenceTransformer(
                    self.text_model_name,
                    device=str(self.device),
                    model_kwargs=model_kwargs
                )
                logger.info("BGE-M3 model loaded successfully via SentenceTransformer.")

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

    def get_text_embeddings_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Extract Dense [N, 1024] and Sparse text embeddings using BGE-M3."""
        if not texts:
            import scipy.sparse as sp
            return {"dense": np.empty((0, 1024), dtype=np.float32), "sparse": sp.csr_matrix((0, 250002), dtype=np.float32)}

        self._init_text_model()

        try:
            if getattr(self, "use_flag_embedding", False):
                out = self.text_model.encode(
                    texts,
                    batch_size=32,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False
                )
                dense_vecs = out['dense_vecs']
                lexical_weights = out['lexical_weights'] # List of Dict[str, float]
                
                import scipy.sparse as sp
                rows, cols, data = [], [], []
                vocab_size = 250002 # Standard BGE-M3 XLM-R vocab size
                for r_idx, lex_dict in enumerate(lexical_weights):
                    for k, v in lex_dict.items():
                        rows.append(r_idx)
                        cols.append(int(k))
                        data.append(float(v))
                sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(texts), vocab_size), dtype=np.float32)
                
                return {"dense": dense_vecs, "sparse": sparse_matrix}
            else:
                # Fallback to SentenceTransformer
                embeddings = self.text_model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                res_array = np.array(embeddings, dtype=np.float32)
                
                import scipy.sparse as sp
                sparse_matrix = sp.csr_matrix((len(texts), 250002), dtype=np.float32)
                
                return {"dense": res_array, "sparse": sparse_matrix}
                
        except Exception as e:
            logger.error(f"Text batch embedding extraction error: {e}")
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_text_embedding(self, text: str) -> Dict[str, Any]:
        """Extract a single text vector embedding (Dense + Sparse)."""
        batch_res = self.get_text_embeddings_batch([text])
        if batch_res["dense"].shape[0] > 0:
            return {"dense": batch_res["dense"][0], "sparse": batch_res["sparse"][0]}
        else:
            import scipy.sparse as sp
            return {"dense": np.zeros((1024,), dtype=np.float32), "sparse": sp.csr_matrix((1, 250002), dtype=np.float32)[0]}


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    dummy_img = Image.new("RGB", (224, 224), (255, 0, 0))
    img_emb = generator.get_image_embedding(dummy_img)
    print(f"Image Embedding Shape: {img_emb.shape}, Dtype: {img_emb.dtype}")

    text_emb = generator.get_text_embedding("Smart video retrieval system")
    print(f"Text Embedding Shape: {text_emb.shape}, Dtype: {text_emb.dtype}")
