"""
Reranking & Dynamic Scene-Aware Clipping Module
- Dynamic Scene-Aware Clipping: Gom nhóm linh hoạt các khung hình theo Shot ID của TransNet V2.
- Event Sequence Merging: Tự động gộp các shots liền kề cùng video nếu khoảng cách timestamp <= max_gap_ms (đọc từ config.yaml).
- Clip Scoring: Chấm điểm cụm bằng Max-Pooling + Average-Pooling.
- Trả về danh sách Clips chuẩn hóa cho Giao diện UI và Tác vụ TRAKE.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import yaml
from src.database.metadata_db import MetadataDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DynamicReranker")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def format_timestamp_ms(ms: int) -> str:
    """Format milliseconds sang hh:mm:ss.mmm"""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class DynamicSceneAwareReranker:
    """
    Reranker với cơ chế Dynamic Scene-Aware Clipping:
    Không sử dụng cắt đoạn cứng T giây. Gom nhóm dựa trên Shot ID TransNet V2 và gộp các shot liền kề trong max_gap_ms (đọc từ config.yaml).
    """

    def __init__(self, config_path: str = "config.yaml", db_path: Optional[str] = None):
        self.config_path = config_path
        self.config = load_config(config_path)
        
        if db_path is None:
            db_path = self.config.get("paths", {}).get("sqlite_db_path", "processed_data/3_metadata/metadata.db")
            
        self.metadata_db = MetadataDB(db_path=db_path)
        self.default_max_gap_ms = self.config.get("late_fusion", {}).get("max_gap_ms", 3000)
        self.default_top_k = self.config.get("late_fusion", {}).get("top_k_rerank", 30)

    def rerank_and_group_clips(
        self,
        candidates: List[Dict[str, Any]],
        query: Optional[str] = None,
        max_gap_ms: Optional[int] = None,
        top_k_clips: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        1. Nhận Input từ FAISS Top 100 frames (chứa frame_path hoặc id và score).
        2. Truy vấn SQLite metadata_db để lấy shot_id, video_id, timestamp_ms, etc.
        3. Gom nhóm theo video_id và shot_id.
        4. Tự động gộp thêm các shots liền kề nếu khoảng cách timestamp <= max_gap_ms (đọc từ config.yaml).
        5. Chấm điểm theo cụm bằng Max-Pooling + Average-Pooling Fusion.
        6. Áp dụng Temporal Reranking: Nhân 2.0x điểm số nếu query chứa từ khóa thời gian ('trước khi', 'sau khi', 'rồi') và timestamp_A < timestamp_B.
        7. Trả về danh sách các Clips (TRAKE-ready).
        """
        if not candidates:
            return []

        if max_gap_ms is None:
            max_gap_ms = self.default_max_gap_ms
        if top_k_clips is None:
            top_k_clips = self.default_top_k

        # Kiểm tra xem truy vấn có chứa từ khóa thời gian hay không
        temporal_keywords = ["trước khi", "sau khi", "rồi", "sau đó", "trước đó", "bắt đầu", "kết thúc", "sau cùng"]
        is_temporal_query = False
        if query and isinstance(query, str):
            query_lower = query.lower()
            is_temporal_query = any(kw in query_lower for kw in temporal_keywords)

        # 1. Truy vấn metadata bổ sung cho từng candidate frame từ SQLite DB
        enriched_candidates = []
        for cand in candidates:
            frame_path = cand.get("frame_path") or cand.get("saved_path") or ""
            score = float(cand.get("score", 0.0))

            meta = None
            if frame_path:
                meta = self.metadata_db.get_by_frame_path(frame_path)
            elif "id" in cand:
                meta = self.metadata_db.get_by_id(cand["id"])

            if meta is None:
                # Fallback từ thông tin có sẵn trong candidate
                meta = {
                    "video_id": cand.get("video_name") or cand.get("video_id", "unknown"),
                    "shot_id": cand.get("shot_id", 0),
                    "frame_type": cand.get("frame_type", "common"),
                    "frame_idx": cand.get("frame_idx", 0),
                    "timestamp_ms": cand.get("timestamp_ms", 0),
                    "frame_path": frame_path,
                    "ocr_text": cand.get("ocr_fixed") or cand.get("ocr_raw", ""),
                    "asr_text": cand.get("asr_text", ""),
                    "detected_objects": ", ".join(cand.get("objects", [])) if isinstance(cand.get("objects"), list) else str(cand.get("objects", "")),
                    "context_summary": cand.get("context_summary", "")
                }

            meta["score"] = score
            enriched_candidates.append(meta)

        # 2. Gom nhóm theo video_id
        video_groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in enriched_candidates:
            v_id = item["video_id"]
            if v_id not in video_groups:
                video_groups[v_id] = []
            video_groups[v_id].append(item)

        final_clips = []

        # 3. Gom nhóm theo Shot ID và gộp các shot liền kề trong max_gap_ms cho từng Video
        for v_id, frames in video_groups.items():
            sorted_frames = sorted(frames, key=lambda x: x["timestamp_ms"])

            shot_clusters: Dict[int, List[Dict[str, Any]]] = {}
            for f in sorted_frames:
                s_id = f["shot_id"]
                if s_id not in shot_clusters:
                    shot_clusters[s_id] = []
                shot_clusters[s_id].append(f)

            sorted_shot_ids = sorted(shot_clusters.keys())

            merged_events: List[List[Dict[str, Any]]] = []
            current_event: List[Dict[str, Any]] = []

            for s_id in sorted_shot_ids:
                shot_frames = shot_clusters[s_id]
                if not current_event:
                    current_event.extend(shot_frames)
                else:
                    prev_end_ts = max(f["timestamp_ms"] for f in current_event)
                    curr_start_ts = min(f["timestamp_ms"] for f in shot_frames)

                    if (curr_start_ts - prev_end_ts) <= max_gap_ms:
                        current_event.extend(shot_frames)
                    else:
                        merged_events.append(current_event)
                        current_event = list(shot_frames)

            if current_event:
                merged_events.append(current_event)

            # 4. Chấm điểm cho từng Clip (Clip Scoring) & Tạo Cấu trúc Clip TRAKE
            for event_idx, event_frames in enumerate(merged_events, start=1):
                scores = [f["score"] for f in event_frames]
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)

                clip_score = max_score * 0.7 + avg_score * 0.3

                # Temporal Reranking Logic:
                # Nếu là Temporal Query ("trước khi", "sau khi", "rồi") và thứ tự timestamp_A < timestamp_B -> Thưởng 2.0x điểm
                if len(event_frames) > 1:
                    sorted_by_ts = sorted(event_frames, key=lambda x: x["timestamp_ms"])
                    if sorted_by_ts == event_frames:
                        if is_temporal_query:
                            clip_score *= 2.0  # Thưởng gấp đôi điểm cho đúng thứ tự thời gian theo truy vấn
                        else:
                            clip_score *= 1.15  # Thưởng 15% cho clip có thứ tự tự nhiên khi query thường

                min_ts = min(f["timestamp_ms"] for f in event_frames)
                max_ts = max(f["timestamp_ms"] for f in event_frames)
                shot_ids = list(set(f["shot_id"] for f in event_frames))

                summaries = [f["context_summary"] for f in event_frames if f.get("context_summary")]
                clip_summary = summaries[0] if summaries else ""

                clip_obj = {
                    "clip_id": f"{v_id}_clip_{min(shot_ids)}_to_{max(shot_ids)}",
                    "video_id": v_id,
                    "start_shot_id": min(shot_ids),
                    "end_shot_id": max(shot_ids),
                    "shot_ids": sorted(shot_ids),
                    "start_time_ms": min_ts,
                    "end_time_ms": max_ts,
                    "start_time_str": format_timestamp_ms(min_ts),
                    "end_time_str": format_timestamp_ms(max_ts),
                    "duration_sec": round((max_ts - min_ts) / 1000.0, 2),
                    "score": round(float(clip_score), 4),
                    "max_frame_score": round(float(max_score), 4),
                    "context_summary": clip_summary,
                    "keyframes_count": len(event_frames),
                    "keyframes": [
                        {
                            "frame_path": f.get("frame_path"),
                            "shot_id": f.get("shot_id"),
                            "frame_type": f.get("frame_type"),
                            "timestamp_ms": f.get("timestamp_ms"),
                            "timestamp_str": format_timestamp_ms(f.get("timestamp_ms", 0)),
                            "score": round(float(f.get("score", 0.0)), 4),
                            "objects": f.get("detected_objects"),
                            "ocr_text": f.get("ocr_text"),
                            "asr_text": f.get("asr_text")
                        }
                        for f in sorted(event_frames, key=lambda x: x["score"], reverse=True)
                    ]
                }
                final_clips.append(clip_obj)

        final_clips = sorted(final_clips, key=lambda x: x["score"], reverse=True)
        return final_clips[:top_k_clips]


CrossEncoderReranker = DynamicSceneAwareReranker


if __name__ == "__main__":
    reranker = DynamicSceneAwareReranker()
    print("DynamicSceneAwareReranker khởi tạo thành công với config.yaml!")
