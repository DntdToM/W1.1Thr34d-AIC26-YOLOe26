"""
FAISS Vector Index Management & Global Indexing Script
Quản lý chỉ mục Vector Search độ trễ mili-giây và quét/gộp toàn bộ các file .npy embeddings ra faiss_index.bin.
"""

import os
import glob
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VectorDB")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class VectorDB:
    """
    FAISS Index Wrapper phục vụ truy xuất vector siêu tốc.
    """

    def __init__(self, dimension: int = 768, metric_type: str = "COSINE", config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        faiss_cfg = self.config.get("faiss", {})
        
        self.dimension = faiss_cfg.get("dimension", dimension)
        self.index_path = self.config.get("paths", {}).get("faiss_index_path", "processed_data/2_embeddings/faiss_index.bin")
        self.mapping_path = os.path.join(os.path.dirname(self.index_path), "faiss_mapping.json")

        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for Cosine Similarity on normalized vectors
        self.mapping: List[Dict[str, Any]] = []

        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.load_index(self.index_path, self.mapping_path)

    def add_vectors(self, vectors: np.ndarray, metadata_list: Optional[List[Dict[str, Any]]] = None):
        """Thêm danh sách vector và metadata vào index."""
        if vectors is None or vectors.size == 0:
            return

        # Ensure float32 and L2 normalization
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        if not self.index.is_trained:
            self.index.train(vectors)
            
        self.index.add(vectors)

        if metadata_list:
            self.mapping.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Truy vấn Top-K kết quả tương đồng nhất."""
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
            
        k = min(top_k, self.index.ntotal) if self.index.ntotal > 0 else top_k
        if self.index.ntotal == 0:
            return np.array([]), np.array([])
            
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]

    def search_with_metadata(self, query_vector: np.ndarray, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Truy vấn Top-K kết quả và khớp trực tiếp với metadata mapping.
        Trả về danh sách candidates dictionary kèm điểm số `score`.
        """
        distances, indices = self.search(query_vector, top_k=top_k)
        results = []
        
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.mapping):
                continue
            meta = dict(self.mapping[idx])
            meta["score"] = float(dist)
            results.append(meta)

        return results

    def save_index(self, index_path: Optional[str] = None, mapping_path: Optional[str] = None):
        """Lưu index ra file bin và mapping ra file json."""
        idx_p = index_path or self.index_path
        map_p = mapping_path or self.mapping_path
        
        os.makedirs(os.path.dirname(idx_p), exist_ok=True)
        faiss.write_index(self.index, idx_p)
        
        with open(map_p, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Đã lưu FAISS Index ({self.index.ntotal} vectors) tại: {idx_p}")

    def load_index(self, index_path: Optional[str] = None, mapping_path: Optional[str] = None):
        """Đọc index từ file bin và mapping từ file json."""
        idx_p = index_path or self.index_path
        map_p = mapping_path or self.mapping_path
        
        if os.path.exists(idx_p):
            self.index = faiss.read_index(idx_p)
            logger.info(f"Đã nạp FAISS Index từ {idx_p} ({self.index.ntotal} vectors).")
            
        if os.path.exists(map_p):
            with open(map_p, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
            logger.info(f"Đã nạp FAISS Mapping từ {map_p} ({len(self.mapping)} entries).")


def build_global_index(
    embeddings_dir: str = "processed_data/2_embeddings",
    metadata_dir: str = "processed_data/3_metadata",
    index_output_path: str = "processed_data/2_embeddings/faiss_index.bin"
) -> VectorDB:
    """
    KỊCH BẢN TỔNG HỢP FAISS GLOBAL INDEXING (100% Complete Implementation):
    - Quét toàn bộ file *_img_emb.npy trong thư mục embeddings_dir.
    - Gom nhóm ma trận bằng np.vstack().
    - Khớp chi tiết metadata từng keyframe từ các file *_metadata.json trong metadata_dir.
    - Đưa vào faiss.IndexFlatIP và lưu ra faiss_index.bin.
    """
    logger.info("=== BẮT ĐẦU TẠO FAISS GLOBAL INDEX ===")
    img_emb_files = sorted(glob.glob(os.path.join(embeddings_dir, "*_img_emb.npy")))

    if not img_emb_files:
        logger.warning(f"Không tìm thấy file *_img_emb.npy nào trong '{embeddings_dir}'.")
        vdb = VectorDB(dimension=768)
        return vdb

    all_vectors = []
    all_metadata = []

    for emb_file in img_emb_files:
        video_name = os.path.basename(emb_file).replace("_img_emb.npy", "")
        meta_json_path = os.path.join(metadata_dir, f"{video_name}_metadata.json")

        if not os.path.exists(meta_json_path):
            logger.warning(f"Không tìm thấy file metadata {meta_json_path} cho {video_name}. Bỏ qua.")
            continue

        try:
            vecs = np.load(emb_file).astype(np.float32)
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
                
            keyframes = meta_json.get("keyframes", [])
            
            if len(vecs) != len(keyframes):
                logger.warning(f"Mismatch số lượng vector ({len(vecs)}) và keyframes ({len(keyframes)}) cho {video_name}.")

            all_vectors.append(vecs)
            
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
            logger.error(f"Lỗi khi xử lý file embedding {emb_file}: {e}")

    if not all_vectors:
        logger.warning("Không gom được vector nào hợp lệ.")
        return VectorDB(dimension=768)

    # Gộp toàn bộ vector bằng np.vstack()
    stacked_vectors = np.vstack(all_vectors)
    dimension = stacked_vectors.shape[1]

    logger.info(f"Đã gom tổng cộng {stacked_vectors.shape[0]} vectors (Dimension={dimension}). Nạp vào FAISS IndexFlatIP...")

    vector_db = VectorDB(dimension=dimension)
    vector_db.add_vectors(stacked_vectors, all_metadata)
    
    mapping_path = os.path.join(os.path.dirname(index_output_path), "faiss_mapping.json")
    vector_db.save_index(index_output_path, mapping_path)
    
    logger.info("=== HOÀN THÀNH TẠO FAISS GLOBAL INDEX ===")
    return vector_db


if __name__ == "__main__":
    vdb = build_global_index()
    print(f"Global Index Total Vectors: {vdb.index.ntotal}")
