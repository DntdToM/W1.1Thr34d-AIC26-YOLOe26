"""Hybrid Vector Database for Dual FAISS GPU Indices and PyTorch SpMM."""

import os
import glob
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
import yaml
import scipy.sparse as sp
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HybridVectorDB")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class HybridVectorDB:
    """Manages Vision (Dense), Text (Dense), and Text (Sparse) indices."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.db_dir = self.config.get("paths", {}).get("faiss_index_path", "processed_data/2_embeddings").replace("faiss_index.bin", "")
        self.db_dir = self.db_dir if os.path.isdir(self.db_dir) else os.path.dirname(self.db_dir)
        
        self.vision_index_path = os.path.join(self.db_dir, "vision_faiss.bin")
        self.text_dense_index_path = os.path.join(self.db_dir, "text_dense_faiss.bin")
        self.text_sparse_matrix_path = os.path.join(self.db_dir, "text_sparse.npz")
        self.mapping_path = os.path.join(self.db_dir, "faiss_mapping.json")

        self.vision_index = None
        self.text_dense_index = None
        self.text_sparse_matrix = None
        self.mapping: List[Dict[str, Any]] = []

        self.res = None
        if faiss.get_num_gpus() > 0:
            self.res = faiss.StandardGpuResources()
            logger.info("FAISS GPU Resources initialized.")

    def _create_index(self, dimension: int, ntotal: int) -> faiss.Index:
        """Create Auto-Scaling FAISS Index (GPU if available)."""
        if ntotal > 1500000:
            nlist = int(4 * np.sqrt(ntotal))
            index = faiss.index_factory(dimension, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Created IndexIVFFlat for {ntotal} items (nlist={nlist})")
        else:
            index = faiss.IndexFlatIP(dimension)
            logger.info(f"Created IndexFlatIP for {ntotal} items")

        if self.res:
            try:
                index = faiss.index_cpu_to_gpu(self.res, 0, index)
            except Exception as e:
                logger.warning(f"Could not move index to GPU: {e}")
        return index

    def build_from_data(self, vision_vectors: np.ndarray, text_dense_vectors: np.ndarray, text_sparse_matrix: sp.csr_matrix, metadata_list: List[Dict[str, Any]]):
        """Train and add vectors to indices."""
        ntotal = len(vision_vectors)
        if ntotal == 0:
            return

        # 1. Vision Index
        self.vision_index = self._create_index(vision_vectors.shape[1], ntotal)
        vision_vecs = vision_vectors.astype(np.float32)
        norms = np.linalg.norm(vision_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vision_vecs = vision_vecs / norms
        if not self.vision_index.is_trained:
            self.vision_index.train(vision_vecs)
        self.vision_index.add(vision_vecs)

        # 2. Text Dense Index
        self.text_dense_index = self._create_index(text_dense_vectors.shape[1], ntotal)
        text_vecs = text_dense_vectors.astype(np.float32)
        norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        text_vecs = text_vecs / norms
        if not self.text_dense_index.is_trained:
            self.text_dense_index.train(text_vecs)
        self.text_dense_index.add(text_vecs)

        # 3. Text Sparse Matrix
        self.text_sparse_matrix = text_sparse_matrix
        
        # 4. Metadata
        self.mapping = metadata_list

    def save_indices(self):
        """Persist indices and metadata to disk."""
        os.makedirs(self.db_dir, exist_ok=True)
        
        if self.vision_index:
            cpu_idx = faiss.index_gpu_to_cpu(self.vision_index) if self.res else self.vision_index
            faiss.write_index(cpu_idx, self.vision_index_path)
            
        if self.text_dense_index:
            cpu_idx = faiss.index_gpu_to_cpu(self.text_dense_index) if self.res else self.text_dense_index
            faiss.write_index(cpu_idx, self.text_dense_index_path)
            
        if self.text_sparse_matrix is not None:
            sp.save_npz(self.text_sparse_matrix_path, self.text_sparse_matrix)
            
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Hybrid indices saved to '{self.db_dir}'")

    def load_indices(self):
        """Load indices and metadata from disk."""
        if os.path.exists(self.vision_index_path):
            cpu_idx = faiss.read_index(self.vision_index_path)
            self.vision_index = faiss.index_cpu_to_gpu(self.res, 0, cpu_idx) if self.res else cpu_idx
            
        if os.path.exists(self.text_dense_index_path):
            cpu_idx = faiss.read_index(self.text_dense_index_path)
            self.text_dense_index = faiss.index_cpu_to_gpu(self.res, 0, cpu_idx) if self.res else cpu_idx
            
        if os.path.exists(self.text_sparse_matrix_path):
            self.text_sparse_matrix = sp.load_npz(self.text_sparse_matrix_path)
            
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
                
        logger.info(f"Hybrid indices loaded. Mapping size: {len(self.mapping)}")

    def get_sparse_tensor(self) -> torch.Tensor:
        """Get the text sparse matrix as a PyTorch sparse tensor."""
        if self.text_sparse_matrix is None:
            return None
        
        coo = self.text_sparse_matrix.tocoo()
        indices = np.vstack((coo.row, coo.col))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.sparse_coo_tensor(
            indices,
            coo.data,
            size=coo.shape,
            dtype=torch.float32
        ).to(device)
        return tensor

def build_global_index(
    embeddings_dir: str = "processed_data/2_embeddings",
    metadata_dir: str = "processed_data/3_metadata"
) -> HybridVectorDB:
    logger.info("=== STARTING HYBRID GLOBAL INDEXING ===")
    
    # We will gather vision (.npy), text_dense (.npy) and text_sparse (.npz)
    img_emb_files = sorted(glob.glob(os.path.join(embeddings_dir, "*_img_emb.npy")))
    
    if not img_emb_files:
        logger.warning(f"No *_img_emb.npy files detected in '{embeddings_dir}'.")
        return HybridVectorDB()

    all_vision = []
    all_text_dense = []
    all_text_sparse = []
    all_metadata = []

    for img_file in img_emb_files:
        video_name = os.path.basename(img_file).replace("_img_emb.npy", "")
        text_dense_file = os.path.join(embeddings_dir, f"{video_name}_text_emb.npy")
        text_sparse_file = os.path.join(embeddings_dir, f"{video_name}_text_sparse.npz")
        meta_json_path = os.path.join(metadata_dir, f"{video_name}_metadata.json")

        if not (os.path.exists(text_dense_file) and os.path.exists(text_sparse_file) and os.path.exists(meta_json_path)):
            logger.warning(f"Missing required embedding or metadata for '{video_name}'. Skipping.")
            continue

        try:
            v_vecs = np.load(img_file).astype(np.float32)
            t_vecs = np.load(text_dense_file).astype(np.float32)
            t_sparse = sp.load_npz(text_sparse_file)
            
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
                
            keyframes = meta_json.get("keyframes", [])
            
            if len(v_vecs) != len(keyframes):
                continue
                
            all_vision.append(v_vecs)
            all_text_dense.append(t_vecs)
            all_text_sparse.append(t_sparse)
            
            for kf in keyframes:
                all_metadata.append({
                    "video_id": video_name,
                    "shot_id": kf.get("shot_id", 0),
                    "frame_type": kf.get("frame_type", "common"),
                    "frame_idx": kf.get("frame_idx", 0),
                    "timestamp_ms": kf.get("timestamp_ms", 0),
                    "frame_path": kf.get("saved_path", ""),
                    "ocr_text": kf.get("ocr_fixed") or kf.get("ocr_raw", ""),
                    "asr_text": kf.get("asr_text", ""),
                    "detected_objects": ", ".join(kf.get("objects", [])) if isinstance(kf.get("objects"), list) else str(kf.get("objects", "")),
                    "context_summary": kf.get("context_summary", "")
                })

        except Exception as e:
            logger.error(f"Error processing video '{video_name}': {e}")

    if not all_vision:
        return HybridVectorDB()

    stacked_vision = np.vstack(all_vision)
    stacked_text_dense = np.vstack(all_text_dense)
    stacked_text_sparse = sp.vstack(all_text_sparse)

    db = HybridVectorDB()
    db.build_from_data(stacked_vision, stacked_text_dense, stacked_text_sparse, all_metadata)
    db.save_indices()
    
    logger.info("=== HYBRID GLOBAL INDEXING COMPLETE ===")
    return db

if __name__ == "__main__":
    vdb = build_global_index()
    if vdb.vision_index:
        print(f"Global Index Total Vectors: {vdb.vision_index.ntotal}")
