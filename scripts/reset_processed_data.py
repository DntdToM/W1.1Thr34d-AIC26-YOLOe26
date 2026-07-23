"""
Reset Processed Data Script (Cross-Platform Python / Windows / Linux)
Dọn dẹp sạch sẽ các thư mục processed_data/1_frames, 2_embeddings, 3_metadata để chạy lại pipeline từ đầu.
"""

import os
import sys
import shutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_FOLDERS = [
    "processed_data/1_frames",
    "processed_data/2_embeddings",
    "processed_data/3_metadata"
]


def reset_processed_data():
    print("==========================================================")
    print("=== RESET PIPELINE DATA (processed_data/ 1_, 2_, 3_) ===")
    print("==========================================================")

    for folder in TARGET_FOLDERS:
        full_path = os.path.join(PROJECT_ROOT, folder)
        if os.path.exists(full_path):
            print(f"Cleaning directory: {folder} ...")
            for item in os.listdir(full_path):
                if item == ".gitkeep":
                    continue
                item_path = os.path.join(full_path, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"  Warning removing {item_path}: {e}")

    print("==========================================================")
    print("Successfully cleaned all old data in processed_data!")
    print("System is ready for a fresh pipeline run.")
    print("==========================================================")


if __name__ == "__main__":
    reset_processed_data()
