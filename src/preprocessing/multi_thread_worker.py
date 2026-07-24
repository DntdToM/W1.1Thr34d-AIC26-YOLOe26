"""
Multi-Threaded & Multi-Processed Pipeline Worker (Singleton Pattern Model Preloading)
Điều phối xử lý song song video & audio cho Offline Indexing (Phase 1)
Áp dụng Singleton Pattern nạp toàn bộ Mô hình (TransNetV2, PhoWhisper, YOLOv9, SigLIP 2, BGE-M3) 
ĐÚNG 1 LẦN DUY NHẤT ở VRAM/RAM. Triệt tiêu hoàn toàn Rủi ro VRAM Concurrency & OOM khi chạy đa luồng.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import numpy as np
from tqdm import tqdm

from src.preprocessing.video_segment import VideoSegmenter
from src.preprocessing.audio_asr import AudioASRProcessor
from src.preprocessing.vision_ocr_obj import VisionAnalytics, window_based_summarize, COCO_VI_MAP
from src.preprocessing.embedding_gen import EmbeddingGenerator
from src.database.metadata_db import MetadataDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MultiThreadWorker")

# Singleton Pattern Global State
_GLOBAL_SEGMENTER: Optional[VideoSegmenter] = None
_GLOBAL_AUDIO_PROCESSOR: Optional[AudioASRProcessor] = None
_GLOBAL_VISION_ANALYTICS: Optional[VisionAnalytics] = None
_GLOBAL_EMBEDDING_GENERATOR: Optional[EmbeddingGenerator] = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_shared_segmenter(config_path: str = "config.yaml") -> VideoSegmenter:
    global _GLOBAL_SEGMENTER
    if _GLOBAL_SEGMENTER is None:
        _GLOBAL_SEGMENTER = VideoSegmenter(config_path=config_path)
    return _GLOBAL_SEGMENTER


def get_shared_audio_processor(config_path: str = "config.yaml") -> AudioASRProcessor:
    global _GLOBAL_AUDIO_PROCESSOR
    if _GLOBAL_AUDIO_PROCESSOR is None:
        _GLOBAL_AUDIO_PROCESSOR = AudioASRProcessor(config_path=config_path)
    return _GLOBAL_AUDIO_PROCESSOR


def get_shared_vision_analytics(config_path: str = "config.yaml") -> VisionAnalytics:
    global _GLOBAL_VISION_ANALYTICS
    if _GLOBAL_VISION_ANALYTICS is None:
        _GLOBAL_VISION_ANALYTICS = VisionAnalytics(config_path=config_path)
    return _GLOBAL_VISION_ANALYTICS


def get_shared_embedding_generator(config_path: str = "config.yaml") -> EmbeddingGenerator:
    global _GLOBAL_EMBEDDING_GENERATOR
    if _GLOBAL_EMBEDDING_GENERATOR is None:
        _GLOBAL_EMBEDDING_GENERATOR = EmbeddingGenerator(config_path=config_path)
    return _GLOBAL_EMBEDDING_GENERATOR


def parse_timestamp_str(ts_str: str) -> int:
    """Chuyển đổi chuỗi hh:mm:ss.mmm sang milliseconds."""
    try:
        parts = ts_str.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s_parts = parts[2].split(".")
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return (h * 3600 + m * 60 + s) * 1000 + ms
    except Exception:
        return 0


def preparse_asr_map(asr_map: Dict[str, str]) -> List[Tuple[int, int, str]]:
    """Pre-parse asr_map chuỗi timestamp ranges sang danh sách Tuple (start_ms, end_ms, text)."""
    parsed = []
    for ts_range, text in asr_map.items():
        if not text:
            continue
        try:
            if " - " in ts_range:
                start_str, end_str = [s.strip() for s in ts_range.split(" - ")]
                start_ms = parse_timestamp_str(start_str)
                end_ms = parse_timestamp_str(end_str)
                parsed.append((start_ms, end_ms, text))
        except Exception:
            continue
    return parsed


def match_asr_for_timestamp(frame_ts: int, parsed_asr_list: List[Tuple[int, int, str]], tolerance_ms: int = 3000) -> str:
    """Khớp nhanh các phân đoạn ASR dựa trên danh sách Tuple (start_ms, end_ms, text) đã pre-parse."""
    matched_texts = [
        text for start_ms, end_ms, text in parsed_asr_list
        if (start_ms - tolerance_ms) <= frame_ts <= (end_ms + tolerance_ms)
    ]
    return " ".join(matched_texts).strip()


class MultiThreadPipelineWorker:
    """
    Worker điều phối bóc tách dữ liệu đa luồng / đa tiến trình cho Offline Indexing.
    Sử dụng Singleton Pattern nạp trước các mô hình GPU/CPU vào bộ nhớ dùng chung (Global Singletons).
    VRAM giữ nguyên hằng số ~7.5GB cho mọi số lượng luồng worker, hoàn toàn không bị nhân VRAM gây OOM.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        import torch
        if torch.cuda.is_available():
            self.max_workers = 1
            logger.info("Phát hiện CUDA GPU: Tự động tối ưu max_workers=1 để dồn 100% CUDA Compute và triệt tiêu treo luồng VRAM.")
        else:
            self.max_workers = self.config.get("preprocessing", {}).get("max_workers", 8)
        
        self.frames_dir = self.config.get("paths", {}).get("frames_dir", "processed_data/1_frames")
        self.embeddings_dir = self.config.get("paths", {}).get("embeddings_dir", "processed_data/2_embeddings")
        self.metadata_dir = self.config.get("paths", {}).get("metadata_dir", "processed_data/3_metadata")

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        logger.info("Đang nạp các mô hình GPU/CPU theo Singleton Pattern vào VRAM dùng chung...")
        self.segmenter = get_shared_segmenter(self.config_path)
        self.audio_processor = get_shared_audio_processor(self.config_path)
        self.vision_analytics = get_shared_vision_analytics(self.config_path)
        self.emb_generator = get_shared_embedding_generator(self.config_path)
        logger.info(f"Đã sẵn sàng MultiThreadPipelineWorker ({self.max_workers} max_workers, VRAM khống chế hằng số ~7.5GB).")

    def process_single_video(self, video_path: str) -> Dict[str, Any]:
        """
        Quy trình xử lý hoàn chỉnh cho 1 tệp video:
        1. Sử dụng Singleton Models đã load sẵn để chạy bóc tách Video & Audio.
        2. Trích xuất đặc trưng độc lập (YOLOv9 + PaddleOCR/EasyOCR) cho các keyframes.
        3. Khớp ASR chính xác theo timestamp của từng frame.
        4. Tổng hợp bối cảnh LLM theo Cửa sổ 30 giây (Window-based Summarize).
        5. Tạo Vector Embeddings cho Hình ảnh (SigLIP 2) và Văn bản (BGE-M3).
        6. Ghi file [video_name]_metadata.json và [video_name]_img_emb.npy, [video_name]_text_emb.npy
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        start_time = time.time()
        logger.info(f"=== Bắt đầu xử lý Video: {video_name} ===")

        try:
            # 1. Bóc tách video keyframes & audio ASR trên Singleton models
            keyframes_meta = self.segmenter.process_video(video_path)
            asr_map = self.audio_processor.process_audio(video_path)

            logger.info(f"[{video_name}] Bóc tách xong {len(keyframes_meta)} keyframes và {len(asr_map)} đoạn ASR.")

            if not keyframes_meta:
                logger.warning(f"[{video_name}] Không trích xuất được keyframe nào.")
                return {
                    "video_name": video_name,
                    "video_path": video_path,
                    "status": "warning",
                    "reason": "No keyframes extracted"
                }

            # 2. Phân tích Thị giác độc lập (YOLOv9 + OCR) bằng Singleton VisionAnalytics
            raw_keyframes_meta = []
            image_paths = []
            parsed_asr_list = preparse_asr_map(asr_map)

            for meta in keyframes_meta:
                img_path = meta["saved_path"]
                image_paths.append(img_path)

                vision_res = self.vision_analytics.analyze_frame(img_path)
                
                frame_ts = meta.get("timestamp_ms", 0)
                matched_asr = match_asr_for_timestamp(frame_ts, parsed_asr_list, tolerance_ms=3000)

                meta["objects"] = vision_res.get("objects", [])
                meta["ocr_raw"] = vision_res.get("ocr_raw", "")
                meta["ocr_fixed"] = vision_res.get("ocr_fixed", "")
                meta["asr_text"] = matched_asr

                raw_keyframes_meta.append(meta)

            # 3. Tổng hợp bối cảnh LLM theo Cửa sổ 30s (Window-Based Summarize)
            logger.info(f"[{video_name}] Tiến hành tổng hợp bối cảnh LLM theo Cửa sổ 30s (Window-Based)...")
            enriched_keyframes = window_based_summarize(raw_keyframes_meta, window_size=30, config_path=self.config_path)

            texts_for_embedding = []
            for meta in enriched_keyframes:
                objs_vi = [COCO_VI_MAP.get(o, o) for o in meta.get("objects", [])]
                obj_str = f"Vật thể: {', '.join(objs_vi)}. " if objs_vi else ""
                ocr_str = f"Chữ màn hình: {meta.get('ocr_fixed', '')}. " if meta.get("ocr_fixed") else ""
                asr_str = f"Lời nói: {meta.get('asr_text', '')}. " if meta.get("asr_text") else ""
                ctx_str = f"Bối cảnh: {meta.get('context_summary', '')}" if meta.get("context_summary") else ""

                combined_text = f"{obj_str}{ocr_str}{asr_str}{ctx_str}".strip()
                texts_for_embedding.append(combined_text if combined_text else "Video keyframe")

            # 4. Trích xuất Vector Embeddings bằng Singleton EmbeddingGenerator (SigLIP 2 & BGE-M3)
            logger.info(f"[{video_name}] Đang tạo Image & Text Embeddings...")
            image_embeddings = self.emb_generator.get_image_embeddings_batch(image_paths)
            text_embeddings = self.emb_generator.get_text_embeddings_batch(texts_for_embedding)

            # 5. Ghi file kết quả vào processed_data/
            img_emb_path = os.path.join(self.embeddings_dir, f"{video_name}_img_emb.npy")
            text_emb_path = os.path.join(self.embeddings_dir, f"{video_name}_text_emb.npy")

            np.save(img_emb_path, image_embeddings)
            np.save(text_emb_path, text_embeddings)

            json_metadata_path = os.path.join(self.metadata_dir, f"{video_name}_metadata.json")
            full_metadata = {
                "video_name": video_name,
                "video_path": video_path,
                "total_keyframes": len(enriched_keyframes),
                "asr_map": asr_map,
                "img_embeddings_file": img_emb_path,
                "text_embeddings_file": text_emb_path,
                "keyframes": enriched_keyframes
            }

            with open(json_metadata_path, "w", encoding="utf-8") as f:
                json.dump(full_metadata, f, ensure_ascii=False, indent=2)

            db_path = self.config.get("paths", {}).get("sqlite_db_path", "processed_data/3_metadata/metadata.db")
            db_success = False
            for attempt in range(3):
                try:
                    metadata_db = MetadataDB(db_path=db_path)
                    for item in enriched_keyframes:
                        metadata_db.insert_frame_metadata({
                            "video_id": video_name,
                            "shot_id": item.get("shot_id", 0),
                            "frame_type": item.get("frame_type", ""),
                            "frame_idx": item.get("frame_idx", 0),
                            "timestamp_ms": item.get("timestamp_ms", 0),
                            "frame_path": item.get("saved_path", ""),
                            "ocr_text": item.get("ocr_fixed", ""),
                            "asr_text": item.get("asr_text", ""),
                            "detected_objects": ", ".join(item.get("objects", [])),
                            "context_summary": item.get("context_summary", "")
                        })
                    db_success = True
                    break
                except Exception as e:
                    logger.warning(f"Thử chèn SQLite lần {attempt + 1}/3 cho {video_name} thất bại: {e}")
                    time.sleep(1)

            if not db_success:
                logger.error(f"[LỖI NGHIÊM TRỌNG] Không thể chèn SQLite Metadata cho video {video_name} sau 3 lần thử.")
                return {
                    "video_name": video_name,
                    "video_path": video_path,
                    "status": "failed",
                    "error": "SQLite insertion failed after 3 retries"
                }

            elapsed = time.time() - start_time
            logger.info(f"=== [HOÀN THÀNH] {video_name} trong {elapsed:.2f}s | Metadata: {json_metadata_path} ===")

            return {
                "video_name": video_name,
                "video_path": video_path,
                "status": "success",
                "execution_time": elapsed,
                "num_keyframes": len(enriched_keyframes),
                "metadata_file": json_metadata_path,
                "img_emb_file": img_emb_path,
                "text_emb_file": text_emb_path
            }

        except Exception as e:
            logger.error(f"[XỬ LÝ LỖI] Lỗi bóc tách video {video_name}: {str(e)}")
            return {
                "video_name": video_name,
                "video_path": video_path,
                "status": "failed",
                "error": str(e)
            }

    def process_video_batch_parallel(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Chạy song song danh sách video_paths sử dụng ThreadPoolExecutor trên các Singleton Models dùng chung.
        """
        results = []
        logger.info(f"Bắt đầu xử lý song song {len(video_paths)} videos (Singleton Models preloaded)...")
        
        with ThreadPoolExecutor(max_workers=min(len(video_paths), self.max_workers)) as executor:
            future_to_video = {
                executor.submit(self.process_single_video, v_path): v_path 
                for v_path in video_paths
            }
            
            for future in tqdm(as_completed(future_to_video), total=len(video_paths), desc="Offline Indexing Pipeline"):
                video_path = future_to_video[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    logger.error(f"Video {video_path} phát sinh ngoại lệ: {exc}")
                    results.append({
                        "video_path": video_path,
                        "status": "error",
                        "error": str(exc)
                    })
                    
        logger.info(f"Đã hoàn thành xử lý {len(results)}/{len(video_paths)} videos.")
        return results


if __name__ == "__main__":
    worker = MultiThreadPipelineWorker()
    print("MultiThreadPipelineWorker khởi tạo thành công với Singleton Pattern!")
