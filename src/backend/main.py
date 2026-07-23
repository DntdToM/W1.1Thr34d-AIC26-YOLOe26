"""
FastAPI Server for Online Retrieval (Phase 2 Microservice Endpoint)
Cung cấp API phục vụ truy vấn siêu tốc độ trễ mili-giây.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FastAPIServer")

app = FastAPI(
    title="AIC 2026 Multimedia Retrieval System API",
    version="1.0.0",
    description="High-performance Local-first Multimodal Retrieval Engine"
)


class WeightConfig(BaseModel):
    image: float = Field(0.7, ge=0.0, le=1.0)
    audio: float = Field(0.2, ge=0.0, le=1.0)
    meta: float = Field(0.1, ge=0.0, le=1.0)


class SearchRequestPayload(BaseModel):
    query: str
    top_k: int = 100
    weights: Optional[WeightConfig] = WeightConfig()


@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "AIC2026 Multimedia Retrieval Engine",
        "version": "1.0.0"
    }


@app.post("/api/search")
def search(payload: SearchRequestPayload):
    """
    Online Retrieval Pipeline (FastAPI Endpoint Skeleton):
    1. Query Expansion (Qwen2.5 LLM Planner)
    2. FAISS Vector Similarity Search (SigLIP 2 & BGE-M3)
    3. Late Fusion Scoring (Min-Max Normalized Weights)
    4. Dynamic Scene-Aware Reranking (TransNet V2 Event Clips)
    """
    logger.info(f"Nhận yêu cầu tìm kiếm: '{payload.query}' (Weights: {payload.weights})")

    try:
        # STEP 1: Query Expansion (Khung gọi Agent)
        # from src.agent.llm_planner import QueryExpansionAgent
        # expansion_agent = QueryExpansionAgent()
        # expanded_queries = expansion_agent.expand_query(payload.query)

        # STEP 2: FAISS Cosine Vector Search (Khung gọi VectorDB)
        # from src.database.vector_db import VectorDB
        # vector_db = VectorDB()
        # raw_candidates = vector_db.search_with_metadata(query_vec, top_k=payload.top_k)

        # STEP 3: Late Fusion Scoring (Khung gọi FusionEngine)
        # from src.backend.fusion_engine import FusionEngine
        # fusion = FusionEngine(w_image=payload.weights.image, w_audio=payload.weights.audio, w_meta=payload.weights.meta)
        # fused_scores = fusion.compute_fusion_score(img_sims, aud_sims, meta_sims)

        # STEP 4: Dynamic Scene-Aware Reranking (Khung gọi Reranker)
        # from src.backend.reranker import DynamicSceneAwareReranker
        # reranker = DynamicSceneAwareReranker()
        # final_clips = reranker.rerank_and_group_clips(raw_candidates)

        # MOCK DATA GIẢ LẬP KẾT QUẢ CHO STREAMLIT UI TEST NGAY LẬP TỨC
        mock_clips = [
            {
                "clip_id": "walking_tour_720p_clip_10_to_12",
                "video_id": "pov_walkingtour_720p",
                "start_shot_id": 10,
                "end_shot_id": 12,
                "start_time_str": "00:00:15.000",
                "end_time_str": "00:00:27.500",
                "duration_sec": 12.5,
                "score": 0.9452,
                "context_summary": "Người quay phim đang đi bộ dọc theo con phố sầm uất tại Hà Nội, đi qua cửa hàng cà phê và biển hiệu quảng cáo màu đỏ.",
                "keyframes_count": 3,
                "keyframes": [
                    {
                        "frame_path": "processed_data/1_frames/pov_walkingtour_720p_shot_0010_sharpest.jpg",
                        "shot_id": 10,
                        "frame_type": "sharpest",
                        "timestamp_str": "00:00:15.000",
                        "score": 0.9610,
                        "objects": "person, chair, table, coffee cup",
                        "ocr_text": "Cửa hàng Cà phê Phố",
                        "asr_text": "Chào mừng bạn đến với chuyến đi bộ chiều nay"
                    },
                    {
                        "frame_path": "processed_data/1_frames/pov_walkingtour_720p_shot_0011_common.jpg",
                        "shot_id": 11,
                        "frame_type": "common",
                        "timestamp_str": "00:00:21.000",
                        "score": 0.9294,
                        "objects": "car, bicycle, street sign",
                        "ocr_text": "Đường Lê Lợi - Khuyến mãi 20%",
                        "asr_text": "Thời tiết hôm nay rất đẹp"
                    }
                ]
            },
            {
                "clip_id": "test_transnet_clip_1_to_2",
                "video_id": "test_transnet",
                "start_shot_id": 1,
                "end_shot_id": 2,
                "start_time_str": "00:00:01.200",
                "end_time_str": "00:00:06.800",
                "duration_sec": 5.6,
                "score": 0.8812,
                "context_summary": "Cảnh quay thử nghiệm ghi lại giao thông và người đi bộ trên vỉa hè.",
                "keyframes_count": 2,
                "keyframes": [
                    {
                        "frame_path": "processed_data/1_frames/test_transnet_shot_0001_sharpest.jpg",
                        "shot_id": 1,
                        "frame_type": "sharpest",
                        "timestamp_str": "00:00:01.200",
                        "score": 0.8950,
                        "objects": "car, bus",
                        "ocr_text": "THỬ NGHIỆM",
                        "asr_text": ""
                    }
                ]
            }
        ]

        return mock_clips

    except Exception as e:
        logger.error(f"Lỗi xử lý yêu cầu tìm kiếm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
