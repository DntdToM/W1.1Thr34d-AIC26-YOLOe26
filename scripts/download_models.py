"""
Script Tải Toàn Bộ Weights Mô Hình về Workspace Local cho AIC 2026 Retrieval
Giúp hệ thống chạy 100% Offline (Local-First) không cần kết nối Internet hay HuggingFace Hub.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoProcessor, AutoModel, AutoModelForSpeechSeq2Seq
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

models_to_download = [
    {
        "name": "SigLIP 2 Vision Embedding",
        "repo_id": "google/siglip-base-patch16-224",
        "save_dir": os.path.join(MODELS_DIR, "siglip-base-patch16-224"),
        "type": "transformers"
    },
    {
        "name": "BGE-M3 Text Embedding",
        "repo_id": "BAAI/bge-m3",
        "save_dir": os.path.join(MODELS_DIR, "bge-m3"),
        "type": "sentence_transformers"
    },
    {
        "name": "PhoWhisper Small ASR",
        "repo_id": "vinai/phowhisper-small",
        "save_dir": os.path.join(MODELS_DIR, "phowhisper-small"),
        "type": "phowhisper"
    }
]


def download_all_models():
    print("==========================================================")
    print("=== BẮT ĐẦU TẢI WEIGHTS MÔ HÌNH VỀ WORKSPACE (LOCAL-FIRST) ===")
    print("==========================================================")

    for item in models_to_download:
        name = item["name"]
        repo_id = item["repo_id"]
        save_dir = item["save_dir"]
        m_type = item["type"]

        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 2:
            print(f"\n[✓] Mô hình '{name}' đã có sẵn tại '{save_dir}'. Bỏ qua tải lại.")
            continue

        print(f"\n[+] Đang tải mô hình '{name}' ({repo_id}) về '{save_dir}'...")

        try:
            if m_type == "transformers":
                processor = AutoProcessor.from_pretrained(repo_id)
                model = AutoModel.from_pretrained(repo_id)
                processor.save_pretrained(save_dir)
                model.save_pretrained(save_dir)

            elif m_type == "sentence_transformers":
                st_model = SentenceTransformer(repo_id)
                st_model.save(save_dir)

            elif m_type == "phowhisper":
                processor = AutoProcessor.from_pretrained(repo_id)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id)
                processor.save_pretrained(save_dir)
                model.save_pretrained(save_dir)

            print(f"  --> ✅ Tải thành công mô hình '{name}'!")

        except Exception as e:
            print(f"  --> ❌ Lỗi khi tải mô hình '{name}': {e}")

    print("\n==========================================================")
    print("=== HOÀN THÀNH TẢI TẤT CẢ MÔ HÌNH VỀ WORKSPACE! ===")
    print("==========================================================")


if __name__ == "__main__":
    download_all_models()
