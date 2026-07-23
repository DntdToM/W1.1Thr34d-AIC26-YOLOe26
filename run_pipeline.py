"""
Offline Indexing Pipeline Controller (Phase 1 Entry Point)
Nút thắt chạy tự động quá trình Offline Indexing toàn bộ dữ liệu video.
Tự động quét video, bóc tách Keyframes/Audio/OCR/Objects/LLM Context, sinh Embeddings và tự động xây dựng FAISS Global Index.
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
    """
    Quét đệ quy toàn bộ video trong data/official_videos/ và data/dummy_videos/.
    Hỗ trợ quét video nằm trong các thư mục con giải nén từ file ZIP.
    """
    video_extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov")
    video_files = []
    
    # 1. Ưu tiên quét đệ quy trong official_videos
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(official_dir, ext)))
        video_files.extend(glob.glob(os.path.join(official_dir, "**", ext), recursive=True))

    if video_files:
        logger.info(f"Phát hiện {len(video_files)} video chính thức trong '{official_dir}'.")
        return sorted(list(set(video_files)))

    # 2. Fallback quét đệ quy trong dummy_videos nếu chưa có video chính thức
    logger.warning(f"Chưa có video nào trong '{official_dir}'. Đang tìm kiếm trong '{dummy_dir}'...")
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(dummy_dir, ext)))
        video_files.extend(glob.glob(os.path.join(dummy_dir, "**", ext), recursive=True))
    
    if video_files:
        logger.info(f"Phát hiện {len(video_files)} video chạy thử nghiệm trong '{dummy_dir}'.")
        return sorted(list(set(video_files)))

    logger.warning("Không tìm thấy tệp video nào trong cả 2 thư mục data/official_videos/ và data/dummy_videos/.")
    return []


def main():
    logger.info("==========================================================")
    logger.info("=== BẮT ĐẦU QUY TRÌNH OFFLINE INDEXING (PHASE 1 PIPELINE) ===")
    logger.info("==========================================================")
    
    config = load_config("config.yaml")
    paths = config.get("paths", {})
    official_dir = paths.get("official_videos_dir", "data/official_videos")
    dummy_dir = paths.get("dummy_videos_dir", "data/dummy_videos")

    # 1. Quét danh sách file video
    video_files = scan_video_files(official_dir, dummy_dir)
    
    if not video_files:
        logger.error("Không có video để xử lý. Vui lòng thêm video vào data/official_videos/ hoặc data/dummy_videos/.")
        return

    # 2. Khởi tạo bộ điều phối đa luồng MultiThreadPipelineWorker (Singleton Models Preloaded)
    worker = MultiThreadPipelineWorker(config_path="config.yaml")

    # 3. Kích hoạt bóc tách song song và sinh metadata + embeddings
    logger.info(f"Đang đẩy {len(video_files)} video vào MultiThreadWorker...")
    results = worker.process_video_batch_parallel(video_files)

    # 4. Tổng hợp báo cáo kết quả
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") in ("failed", "error")]

    logger.info("==========================================================")
    logger.info("=== BÁO CÁO KẾT QUẢ OFFLINE INDEXING (PHASE 1) ===")
    logger.info(f"- Tổng số video đã xử lý: {len(results)}")
    logger.info(f"- Thành công: {len(successful)}")
    logger.info(f"- Thất bại/Lỗi: {len(failed)}")
    
    if successful:
        logger.info("Các file kết quả được lưu tại:")
        logger.info("  + Frames:       processed_data/1_frames/")
        logger.info("  + Embeddings:   processed_data/2_embeddings/")
        logger.info("  + Metadata:     processed_data/3_metadata/")

        # 5. Tự động sinh FAISS Global Index cho toàn bộ dữ liệu vừa trích xuất
        logger.info("----------------------------------------------------------")
        logger.info("Đang tự động khởi chạy FAISS Global Indexing...")
        vdb = build_global_index()
        logger.info(f"Đã hoàn thành tạo FAISS Global Index với tổng cộng {vdb.index.ntotal} vectors!")
    
    logger.info("==========================================================")


if __name__ == "__main__":
    main()
