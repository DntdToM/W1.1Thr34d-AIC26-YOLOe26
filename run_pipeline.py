"""Offline Indexing Pipeline Controller (Phase 1 Entry Point).

Orchestrates automatic video ingestion, feature extraction (Keyframes, Audio ASR, OCR, Object Detection,
LLM Window Context), multimodal embedding generation, and global FAISS index construction.
"""

import os
import glob
import logging
import yaml
from src.preprocessing.multi_thread_worker import MultiThreadPipelineWorker
from src.database.vector_db import build_global_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PipelineController")


def load_config(config_path: str = "config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def scan_video_files(official_dir: str, dummy_dir: str) -> list:
    """Recursively scan video files from official and dummy directories."""
    video_extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov")
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(official_dir, ext)))
        video_files.extend(glob.glob(os.path.join(official_dir, "**", ext), recursive=True))

    if video_files:
        logger.info(f"Discovered {len(video_files)} official video files in '{official_dir}'.")
        return sorted(list(set(video_files)))

    logger.warning(f"No video files found in '{official_dir}'. Scanning fallback directory '{dummy_dir}'...")
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(dummy_dir, ext)))
        video_files.extend(glob.glob(os.path.join(dummy_dir, "**", ext), recursive=True))
    
    if video_files:
        logger.info(f"Discovered {len(video_files)} test video files in '{dummy_dir}'.")
        return sorted(list(set(video_files)))

    logger.warning(f"No video files detected in '{official_dir}' or '{dummy_dir}'.")
    return []


def main():
    logger.info("==========================================================")
    logger.info("=== STARTING OFFLINE INDEXING PIPELINE (PHASE 1) ===")
    logger.info("==========================================================")
    
    config = load_config("config.yaml")
    paths = config.get("paths", {})
    official_dir = paths.get("official_videos_dir", "data/official_videos")
    dummy_dir = paths.get("dummy_videos_dir", "data/dummy_videos")

    video_files = scan_video_files(official_dir, dummy_dir)
    
    if not video_files:
        logger.error("Pipeline execution aborted: No input video files found.")
        return

    worker = MultiThreadPipelineWorker(config_path="config.yaml")

    logger.info(f"Dispatching {len(video_files)} video files to MultiThreadPipelineWorker...")
    results = worker.process_video_batch_parallel(video_files)

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") in ("failed", "error")]

    logger.info("==========================================================")
    logger.info("=== PIPELINE EXECUTION SUMMARY ===")
    logger.info(f"- Total processed videos: {len(results)}")
    logger.info(f"- Successful: {len(successful)}")
    logger.info(f"- Failed: {len(failed)}")
    
    if successful:
        logger.info("Output artifacts saved to:")
        logger.info("  + Keyframes:    processed_data/1_frames/")
        logger.info("  + Embeddings:   processed_data/2_embeddings/")
        logger.info("  + Metadata:     processed_data/3_metadata/")

        logger.info("----------------------------------------------------------")
        logger.info("Initiating global FAISS index construction...")
        vdb = build_global_index()
        logger.info(f"Global FAISS index construction complete. Total vectors indexed: {vdb.index.ntotal}")
    
    logger.info("==========================================================")


if __name__ == "__main__":
    main()
