"""Multi-threaded pipeline worker for feature extraction and multimodal indexing."""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import numpy as np
from tqdm import tqdm

from src.preprocessing.video_segment import VideoSegmenter
from src.preprocessing.audio_asr import AudioASRProcessor
from src.preprocessing.vision_ocr_obj import VisionAnalytics, window_based_summarize
from src.preprocessing.embedding_gen import EmbeddingGenerator
from src.database.metadata_db import MetadataDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MultiThreadWorker")

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
    """Parse timestamp string formatted as hh:mm:ss.mmm into milliseconds."""
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
    """Pre-parse ASR timestamp ranges into a list of (start_ms, end_ms, text) tuples."""
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
    """Match ASR segments overlapping frame timestamp within tolerance bounds."""
    matched_texts = [
        text for start_ms, end_ms, text in parsed_asr_list
        if (start_ms - tolerance_ms) <= frame_ts <= (end_ms + tolerance_ms)
    ]
    return " ".join(matched_texts).strip()


class MultiThreadPipelineWorker:
    """Pipeline execution worker supporting concurrent video feature extraction."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        import torch
        if torch.cuda.is_available():
            self.max_workers = 1
            logger.info("CUDA GPU detected: Constraining max_workers to 1 to optimize CUDA compute efficiency.")
        else:
            self.max_workers = self.config.get("preprocessing", {}).get("max_workers", 8)
        
        self.frames_dir = self.config.get("paths", {}).get("frames_dir", "processed_data/1_frames")
        self.embeddings_dir = self.config.get("paths", {}).get("embeddings_dir", "processed_data/2_embeddings")
        self.metadata_dir = self.config.get("paths", {}).get("metadata_dir", "processed_data/3_metadata")

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        logger.info("Pre-loading shared model singletons...")
        self.segmenter = get_shared_segmenter(self.config_path)
        self.audio_processor = get_shared_audio_processor(self.config_path)
        self.vision_analytics = get_shared_vision_analytics(self.config_path)
        self.emb_generator = get_shared_embedding_generator(self.config_path)
        logger.info(f"MultiThreadPipelineWorker initialized (max_workers={self.max_workers}).")

    def process_single_video(self, video_path: str) -> Dict[str, Any]:
        """Process a single video file through segmentation, ASR, OCR, object detection, LLM context, and embedding generation."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        start_time = time.time()
        logger.info(f"Processing video: {video_name}")

        try:
            keyframes_meta = self.segmenter.process_video(video_path)
            audio_results = self.audio_processor.process_audio(video_path)
            asr_map = audio_results.get("asr_map", {})
            audio_event_map = audio_results.get("audio_event_map", {})

            logger.info(f"[{video_name}] Extracted {len(keyframes_meta)} keyframes, {len(asr_map)} ASR segments, and {len(audio_event_map)} audio events.")

            if not keyframes_meta:
                logger.warning(f"[{video_name}] Keyframe extraction yielded 0 frames.")
                return {
                    "video_name": video_name,
                    "video_path": video_path,
                    "status": "warning",
                    "reason": "No keyframes extracted"
                }

            raw_keyframes_meta = []
            image_paths = []
            parsed_asr_list = preparse_asr_map(asr_map)
            parsed_event_list = preparse_asr_map(audio_event_map)

            for meta in keyframes_meta:
                img_path = meta["saved_path"]
                image_paths.append(img_path)

                vision_res = self.vision_analytics.analyze_frame(img_path)
                
                frame_ts = meta.get("timestamp_ms", 0)
                matched_asr = match_asr_for_timestamp(frame_ts, parsed_asr_list, tolerance_ms=3000)
                matched_event = match_asr_for_timestamp(frame_ts, parsed_event_list, tolerance_ms=3000)

                meta["objects"] = vision_res.get("objects", [])
                meta["ocr_raw"] = vision_res.get("ocr_raw", "")
                meta["ocr_fixed"] = vision_res.get("ocr_fixed", "")
                meta["asr_text"] = matched_asr
                meta["audio_event"] = matched_event

                raw_keyframes_meta.append(meta)

            logger.info(f"[{video_name}] Generating window-based LLM context summaries...")
            enriched_keyframes = window_based_summarize(raw_keyframes_meta, window_size=30, config_path=self.config_path)

            texts_for_embedding = []
            for meta in enriched_keyframes:
                objs_raw = meta.get("objects", [])
                obj_str = ', '.join(objs_raw) if objs_raw else 'Không có'
                ocr_str = meta.get('ocr_fixed', '').strip() or 'Không có'
                
                audio_str_parts = []
                if meta.get("asr_text"):
                    audio_str_parts.append(meta.get("asr_text"))
                if meta.get("audio_event"):
                    audio_str_parts.append(meta.get("audio_event"))
                audio_str = ' | '.join(audio_str_parts) if audio_str_parts else 'Không có'
                
                ctx_str = meta.get('context_summary', '').strip() or 'Không có'

                combined_text = f"Ngữ cảnh: {ctx_str}. Chi tiết: Vật thể: {obj_str}. Chữ: {ocr_str}. Âm thanh: {audio_str}."
                texts_for_embedding.append(combined_text)

            logger.info(f"[{video_name}] Generating visual and text embeddings...")
            image_embeddings = self.emb_generator.get_image_embeddings_batch(image_paths)
            text_emb_result = self.emb_generator.get_text_embeddings_batch(texts_for_embedding)
            
            if isinstance(text_emb_result, dict):
                text_dense = text_emb_result["dense"]
                text_sparse = text_emb_result["sparse"]
            else:
                text_dense = text_emb_result
                import scipy.sparse as sp
                text_sparse = sp.csr_matrix((len(texts_for_embedding), 250002), dtype=np.float32)

            img_emb_path = os.path.join(self.embeddings_dir, f"{video_name}_img_emb.npy")
            text_emb_path = os.path.join(self.embeddings_dir, f"{video_name}_text_emb.npy")
            text_sparse_path = os.path.join(self.embeddings_dir, f"{video_name}_text_sparse.npz")

            np.save(img_emb_path, image_embeddings)
            np.save(text_emb_path, text_dense)
            
            import scipy.sparse as sp
            sp.save_npz(text_sparse_path, text_sparse)

            json_metadata_path = os.path.join(self.metadata_dir, f"{video_name}_metadata.json")
            full_metadata = {
                "video_name": video_name,
                "video_path": video_path,
                "total_keyframes": len(enriched_keyframes),
                "asr_map": asr_map,
                "audio_event_map": audio_event_map,
                "img_embeddings_file": img_emb_path,
                "text_embeddings_file": text_emb_path,
                "text_sparse_file": text_sparse_path,
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
                            "asr_text": f"{item.get('asr_text', '')} | {item.get('audio_event', '')}".strip(" |"),
                            "detected_objects": ", ".join(item.get("objects", [])),
                            "context_summary": item.get("context_summary", "")
                        })
                    db_success = True
                    break
                except Exception as e:
                    logger.warning(f"SQLite insertion attempt {attempt + 1}/3 failed for {video_name}: {e}")
                    time.sleep(1)

            if not db_success:
                logger.error(f"SQLite metadata insertion failed for {video_name} after 3 retries.")
                return {
                    "video_name": video_name,
                    "video_path": video_path,
                    "status": "failed",
                    "error": "SQLite insertion failed after 3 retries"
                }

            elapsed = time.time() - start_time
            logger.info(f"Completed video processing: {video_name} in {elapsed:.2f}s | Metadata: {json_metadata_path}")

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
            logger.error(f"Processing failed for video {video_name}: {str(e)}")
            return {
                "video_name": video_name,
                "video_path": video_path,
                "status": "failed",
                "error": str(e)
            }

    def process_video_batch_parallel(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of video paths concurrently using shared model singletons."""
        results = []
        logger.info(f"Initiating batch processing for {len(video_paths)} videos...")
        
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
                    logger.error(f"Execution exception for video {video_path}: {exc}")
                    results.append({
                        "video_path": video_path,
                        "status": "error",
                        "error": str(exc)
                    })
                    
        logger.info(f"Batch processing completed: {len(results)}/{len(video_paths)} videos.")
        return results


if __name__ == "__main__":
    worker = MultiThreadPipelineWorker()
    print("MultiThreadPipelineWorker initialized successfully.")
