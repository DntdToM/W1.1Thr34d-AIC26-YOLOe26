"""
Late-Fusion Scoring Engine + Min-Max Normalization
Tính tổng điểm kết hợp chuẩn hóa: 
S_total = w_image * Norm(Sim_Image) + w_audio * Norm(Sim_Audio) + w_meta * Norm(Sim_Metadata)
"""

from typing import List, Dict, Any
import numpy as np


def min_max_normalize(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Chuẩn hóa Min-Max Scaling mảng điểm Similarity về đoạn [0, 1]:
    S_norm = (S - S_min) / (S_max - S_min + eps)
    Đảm bảo phân bố điểm của cả 3 phương thức (Image, Audio, Metadata) đồng nhất
    trước khi nhân trọng số (0.7 / 0.2 / 0.1).
    """
    if scores is None or scores.size == 0:
        return np.array([], dtype=np.float32)
    
    scores = np.asarray(scores, dtype=np.float32)
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    
    range_diff = s_max - s_min
    if range_diff < eps:
        return np.ones_like(scores, dtype=np.float32)
        
    return (scores - s_min) / (range_diff + eps)


class FusionEngine:
    """
    Late-Fusion Score Integrator tích hợp Min-Max Normalization.
    """

    def __init__(self, w_image: float = 0.7, w_audio: float = 0.2, w_meta: float = 0.1):
        self.w_image = w_image
        self.w_audio = w_audio
        self.w_meta = w_meta

    def compute_fusion_score(
        self, 
        image_sims: np.ndarray, 
        audio_sims: np.ndarray, 
        meta_sims: np.ndarray
    ) -> np.ndarray:
        """
        Tính điểm kết hợp trọng số từ 3 luồng dữ liệu đã qua Min-Max Scaling.
        """
        norm_img = min_max_normalize(image_sims)
        norm_aud = min_max_normalize(audio_sims)
        norm_meta = min_max_normalize(meta_sims)

        total_score = (
            self.w_image * norm_img +
            self.w_audio * norm_aud +
            self.w_meta * norm_meta
        )
        return total_score


if __name__ == "__main__":
    fusion = FusionEngine(0.7, 0.2, 0.1)
    img_scores = np.array([0.92, 0.85, 0.60])
    aud_scores = np.array([12.5, 5.0, 1.0])  # Khác dải biên độ
    meta_scores = np.array([0.1, 0.05, 0.0])

    final_scores = fusion.compute_fusion_score(img_scores, aud_scores, meta_scores)
    print("Mảng điểm Fusion đã qua Min-Max Normalization:", final_scores)
