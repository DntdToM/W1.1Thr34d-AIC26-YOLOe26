"""
Reranking & Dynamic Scene-Aware Clipping Module
- Dynamic Scene-Aware Clipping: Gom nhom linh hoat cac khung hinh theo Shot ID cua TransNet V2.
- Event Sequence Merging: Tu dong gop cac shots lien ke cung video neu khoang cach timestamp <= max_gap_ms.
- Temporal Intersection Reranking: Doi chieu thoi gian giua cac sub-query de xu ly truy van tuan tu.
- Clip Scoring: Cham diem cum bang Max-Pooling + Average-Pooling.
- Tra ve danh sach Clips chuan hoa cho Giao dien UI va Tac vu TRAKE.
"""

import os
import math
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

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
    Reranker voi co che Dynamic Scene-Aware Clipping va Temporal Intersection.
    Gom nhom dua tren Shot ID TransNet V2, gop cac shot lien ke trong max_gap_ms,
    va ho tro doi chieu thoi gian tuan tu cho truy van da buoc.
    """

    def __init__(self, config_path: str = "config.yaml", db_path: Optional[str] = None):
        self.config_path = config_path
        self.config = load_config(config_path)

        if db_path is None:
            db_path = self.config.get("paths", {}).get("sqlite_db_path", "processed_data/3_metadata/metadata.db")

        self.metadata_db = MetadataDB(db_path=db_path)
        self.default_max_gap_ms = self.config.get("late_fusion", {}).get("max_gap_ms", 3000)
        self.default_top_k = self.config.get("late_fusion", {}).get("top_k_rerank", 30)

        temporal_cfg = self.config.get("temporal_reasoning", {})
        self.default_max_temporal_gap_ms = temporal_cfg.get("max_temporal_gap_sec", 30) * 1000

    def _enrich_candidate(self, cand: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Truy van metadata bo sung cho mot candidate frame tu SQLite DB."""
        frame_path = cand.get("frame_path") or cand.get("saved_path") or ""
        score = float(cand.get("score", 0.0))

        meta = None
        if frame_path:
            meta = self.metadata_db.get_by_frame_path(frame_path)
        elif "id" in cand:
            meta = self.metadata_db.get_by_id(cand["id"])

        if meta is None:
            meta = {
                "video_id": cand.get("video_name") or cand.get("video_id", "unknown"),
                "shot_id": cand.get("shot_id", 0),
                "frame_type": cand.get("frame_type", "common"),
                "frame_idx": cand.get("frame_idx", 0),
                "timestamp_ms": cand.get("timestamp_ms", 0),
                "frame_path": frame_path,
                "ocr_text": cand.get("ocr_fixed") or cand.get("ocr_raw", ""),
                "asr_text": cand.get("asr_text", ""),
                "detected_objects": (
                    ", ".join(cand.get("objects", []))
                    if isinstance(cand.get("objects"), list)
                    else str(cand.get("objects", ""))
                ),
                "context_summary": cand.get("context_summary", ""),
            }

        meta["score"] = score
        return meta

    def _build_clip_object(
        self, v_id: str, event_frames: List[Dict[str, Any]], clip_score: float
    ) -> Dict[str, Any]:
        """Xay dung cau truc Clip chuan tu danh sach frames va diem so."""
        min_ts = min(f["timestamp_ms"] for f in event_frames)
        max_ts = max(f["timestamp_ms"] for f in event_frames)
        shot_ids = sorted(set(f["shot_id"] for f in event_frames))

        summaries = [f["context_summary"] for f in event_frames if f.get("context_summary")]
        clip_summary = summaries[0] if summaries else ""

        return {
            "clip_id": f"{v_id}_clip_{shot_ids[0]}_to_{shot_ids[-1]}",
            "video_id": v_id,
            "start_shot_id": shot_ids[0],
            "end_shot_id": shot_ids[-1],
            "shot_ids": shot_ids,
            "start_time_ms": min_ts,
            "end_time_ms": max_ts,
            "start_time_str": format_timestamp_ms(min_ts),
            "end_time_str": format_timestamp_ms(max_ts),
            "duration_sec": round((max_ts - min_ts) / 1000.0, 2),
            "score": round(float(clip_score), 4),
            "max_frame_score": round(float(max(f["score"] for f in event_frames)), 4),
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
                    "asr_text": f.get("asr_text"),
                }
                for f in sorted(event_frames, key=lambda x: x["score"], reverse=True)
            ],
        }

    def rerank_and_group_clips(
        self,
        candidates: List[Dict[str, Any]],
        max_gap_ms: Optional[int] = None,
        top_k_clips: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pipeline reranking chuan cho truy van don (khong co yeu to thoi gian tuan tu):
        1. Enrichment metadata tu SQLite.
        2. Gom nhom theo video_id va shot_id.
        3. Gop shots lien ke trong max_gap_ms.
        4. Cham diem Clip bang Max-Pooling (0.7) + Average-Pooling (0.3).
        """
        if not candidates:
            return []

        if max_gap_ms is None:
            max_gap_ms = self.default_max_gap_ms
        if top_k_clips is None:
            top_k_clips = self.default_top_k

        enriched_candidates = []
        for cand in candidates:
            meta = self._enrich_candidate(cand)
            if meta is not None:
                enriched_candidates.append(meta)

        video_groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in enriched_candidates:
            v_id = item["video_id"]
            if v_id not in video_groups:
                video_groups[v_id] = []
            video_groups[v_id].append(item)

        final_clips = []

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

            for event_frames in merged_events:
                scores = [f["score"] for f in event_frames]
                clip_score = max(scores) * 0.7 + (sum(scores) / len(scores)) * 0.3

                clip_obj = self._build_clip_object(v_id, event_frames, clip_score)
                final_clips.append(clip_obj)

        final_clips = sorted(final_clips, key=lambda x: x["score"], reverse=True)
        return final_clips[:top_k_clips]

    def temporal_rerank(
        self,
        sub_results: List[List[Dict[str, Any]]],
        temporal_plan: Dict[str, Any],
        max_temporal_gap_ms: Optional[int] = None,
        top_k_clips: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Temporal Intersection Reranking cho truy van tuan tu da buoc.

        Thuat toan:
        1. Gom nhom tat ca candidates theo video_id (HashMap, O(N)).
        2. Voi moi video_id co du tat ca cac step:
           - Sap xep candidates cua moi step theo timestamp_ms.
           - Two-Pointer verify: T_{step_i} < T_{step_i+1} va delta_T <= max_gap.
        3. Tinh temporal_score = mean(matched_scores) * proximity_bonus.
        4. Tra ve danh sach clips da rerank.

        Args:
            sub_results: Danh sach ket qua tung step, moi phan tu la List[Dict] tu FAISS.
            temporal_plan: Output tu TemporalQueryDecomposer.decompose().
            max_temporal_gap_ms: Nguong khoang cach thoi gian toi da (ms) giua 2 step lien ke.
            top_k_clips: So luong clip toi da tra ve.
        """
        if max_temporal_gap_ms is None:
            max_temporal_gap_ms = self.default_max_temporal_gap_ms
        if top_k_clips is None:
            top_k_clips = self.default_top_k

        num_steps = len(sub_results)
        if num_steps < 2:
            if sub_results:
                return self.rerank_and_group_clips(sub_results[0], top_k_clips=top_k_clips)
            return []

        # 1. Enrich va gom nhom theo video_id -> step_index -> [candidates]
        video_step_map: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for step_idx, step_candidates in enumerate(sub_results):
            for cand in step_candidates:
                meta = self._enrich_candidate(cand)
                if meta is None:
                    continue
                meta["_step_idx"] = step_idx
                video_step_map[meta["video_id"]][step_idx].append(meta)

        # Sap xep candidates trong moi step theo timestamp_ms
        for v_id in video_step_map:
            for step_idx in video_step_map[v_id]:
                video_step_map[v_id][step_idx].sort(key=lambda x: x["timestamp_ms"])

        # 2. Tim cac chuoi thoi gian hop le cho moi video
        temporal_matches: List[Dict[str, Any]] = []

        for v_id, step_data in video_step_map.items():
            # Chi xu ly video co du tat ca cac step
            if len(step_data) < num_steps:
                continue

            has_all_steps = all(step_idx in step_data for step_idx in range(num_steps))
            if not has_all_steps:
                continue

            # Tim cac chuoi T_0 < T_1 < ... < T_{K-1} thoa man delta_T constraints
            valid_chains = self._find_valid_temporal_chains(
                step_data, num_steps, max_temporal_gap_ms
            )

            for chain in valid_chains:
                all_frames = []
                total_score = 0.0
                total_delta_t = 0

                for i, frame in enumerate(chain):
                    all_frames.append(frame)
                    total_score += frame["score"]
                    if i > 0:
                        total_delta_t += frame["timestamp_ms"] - chain[i - 1]["timestamp_ms"]

                mean_score = total_score / len(chain)
                avg_delta_t = total_delta_t / (len(chain) - 1) if len(chain) > 1 else 0

                # proximity_bonus: delta_T cang nho, bonus cang cao.
                # Chuan hoa theo max_temporal_gap_ms de bonus nam trong [1.0, 1.2].
                proximity_bonus = 1.0 + 0.2 * math.exp(
                    -avg_delta_t / max(max_temporal_gap_ms * 0.5, 1)
                )
                temporal_score = mean_score * proximity_bonus

                temporal_matches.append({
                    "video_id": v_id,
                    "temporal_score": temporal_score,
                    "mean_score": mean_score,
                    "proximity_bonus": round(proximity_bonus, 4),
                    "avg_delta_t_ms": round(avg_delta_t, 1),
                    "chain": chain,
                    "all_frames": all_frames,
                })

        if not temporal_matches:
            logger.warning(
                "Temporal rerank: no valid temporal chains found. "
                "Falling back to standard reranking on step 0 results."
            )
            return self.rerank_and_group_clips(sub_results[0], top_k_clips=top_k_clips)

        # 3. Sap xep theo temporal_score giam dan
        temporal_matches.sort(key=lambda x: x["temporal_score"], reverse=True)

        # 4. Xay dung clip objects tu cac temporal matches
        final_clips = []
        for match in temporal_matches[:top_k_clips]:
            clip_obj = self._build_clip_object(
                match["video_id"], match["all_frames"], match["temporal_score"]
            )
            clip_obj["temporal_metadata"] = {
                "is_temporal_match": True,
                "mean_sub_score": round(match["mean_score"], 4),
                "proximity_bonus": match["proximity_bonus"],
                "avg_delta_t_ms": match["avg_delta_t_ms"],
                "num_steps_matched": len(match["chain"]),
            }
            final_clips.append(clip_obj)

        logger.info(
            f"Temporal rerank: {len(temporal_matches)} valid chains found, "
            f"returning top {len(final_clips)} clips."
        )
        return final_clips

    def _find_valid_temporal_chains(
        self,
        step_data: Dict[int, List[Dict[str, Any]]],
        num_steps: int,
        max_gap_ms: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        Tim tat ca cac chuoi thoi gian hop le qua cac step bang Two-Pointer.
        Dieu kien: T_{i} < T_{i+1} va 0 < T_{i+1} - T_{i} <= max_gap_ms.

        Voi 2 steps: Two-Pointer tren 2 mang da sap xep. O(N + M).
        Voi K steps: Mo rong tuan tu, xay dung chuoi tu step 0 den step K-1.
        """
        if num_steps == 0:
            return []

        # Khoi tao voi tat ca candidates cua step 0
        partial_chains: List[List[Dict[str, Any]]] = [
            [frame] for frame in step_data[0]
        ]

        for step_idx in range(1, num_steps):
            next_step_frames = step_data[step_idx]
            extended_chains: List[List[Dict[str, Any]]] = []

            for chain in partial_chains:
                last_ts = chain[-1]["timestamp_ms"]
                # Two-Pointer: tim cac frame hop le trong next_step_frames
                # next_step_frames da duoc sap xep theo timestamp_ms
                for candidate in next_step_frames:
                    delta_t = candidate["timestamp_ms"] - last_ts
                    if delta_t <= 0:
                        continue
                    if delta_t > max_gap_ms:
                        break  # Da sap xep, cac frame sau chi lon hon
                    extended_chains.append(chain + [candidate])

            partial_chains = extended_chains

            if not partial_chains:
                return []

            # Gioi han so luong chains de tranh bung no to hop
            # Giu lai top chains theo tong diem cao nhat
            if len(partial_chains) > 500:
                partial_chains.sort(
                    key=lambda c: sum(f["score"] for f in c), reverse=True
                )
                partial_chains = partial_chains[:500]

        return partial_chains


if __name__ == "__main__":
    reranker = DynamicSceneAwareReranker()
    print("DynamicSceneAwareReranker khoi tao thanh cong voi config.yaml!")
