"""
Script Tải Toàn Bộ Weights Mô Hình về Workspace Local cho AIC 2026 Retrieval
Giúp hệ thống chạy 100% Offline (Local-First) không cần kết nối Internet hay HuggingFace Hub.
Bao gồm: SigLIP 2, BGE-M3, PhoWhisper, TransNetV2, Silero VAD, PaddleOCR.
"""

import os
import sys
import urllib.request
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoProcessor, AutoModel, AutoModelForSpeechSeq2Seq
from sentence_transformers import SentenceTransformer
import torch

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
    },
    {
        "name": "Silero VAD",
        "repo_id": "snakers4/silero-vad",
        "save_dir": os.path.join(MODELS_DIR, "torch_hub"),
        "type": "torch_hub"
    },
    {
        "name": "PaddleOCR Models (vi, en)",
        "repo_id": "paddleocr",
        "save_dir": os.path.join(MODELS_DIR, "paddleocr"),
        "type": "paddleocr"
    }
]

def download_all_models():
    print("==========================================================")
    print("=== BẮT ĐẦU TẢI TRƯỚC TOÀN BỘ WEIGHTS MÔ HÌNH VỀ WORKSPACE ===")
    print("==========================================================")

    results = {}
    all_success = True

    for item in models_to_download:
        name = item["name"]
        repo_id = item["repo_id"]
        save_dir = item["save_dir"]
        m_type = item["type"]

        print(f"\n[+] Đang xử lý: '{name}'...")
        
        try:
            if m_type == "transformers":
                if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 2:
                    print(f"  --> Đã có sẵn tại '{save_dir}'. Bỏ qua tải lại.")
                else:
                    processor = AutoProcessor.from_pretrained(repo_id)
                    model = AutoModel.from_pretrained(repo_id)
                    processor.save_pretrained(save_dir)
                    model.save_pretrained(save_dir)
                    print(f"  --> [SUCCESS] Tải thành công mô hình '{name}'!")
                results[name] = True

            elif m_type == "sentence_transformers":
                if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 2:
                    print(f"  --> Đã có sẵn tại '{save_dir}'. Bỏ qua tải lại.")
                else:
                    st_model = SentenceTransformer(repo_id)
                    st_model.save(save_dir)
                    print(f"  --> [SUCCESS] Tải thành công mô hình '{name}'!")
                results[name] = True

            elif m_type == "phowhisper":
                if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 2:
                    print(f"  --> Đã có sẵn tại '{save_dir}'. Bỏ qua tải lại.")
                else:
                    processor = AutoProcessor.from_pretrained(repo_id)
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id)
                    processor.save_pretrained(save_dir)
                    model.save_pretrained(save_dir)
                    print(f"  --> [SUCCESS] Tải thành công mô hình '{name}'!")
                results[name] = True
                
            elif m_type == "url":
                os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                if os.path.exists(save_dir) and os.path.getsize(save_dir) > 1000000:
                    print(f"  --> Đã có sẵn tại '{save_dir}'. Bỏ qua tải lại.")
                else:
                    print(f"  --> Downloading from {repo_id}...")
                    urllib.request.urlretrieve(repo_id, save_dir)
                    print(f"  --> [SUCCESS] Tải thành công '{name}'!")
                results[name] = True

            elif m_type == "torch_hub":
                torch.hub.set_dir(save_dir)
                model = torch.hub.load(repo_id, 'silero_vad', force_reload=False)
                print(f"  --> [SUCCESS] Tải thành công '{name}' vào '{save_dir}'!")
                results[name] = True

            elif m_type == "paddleocr":
                from paddleocr import PaddleOCR
                os.environ['MODULE_BASE_DIR'] = save_dir
                ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=False, show_log=False)
                print(f"  --> ✅ Tải thành công '{name}'!")
                results[name] = True

        except Exception as e:
            print(f"  --> [ERROR] LỖI khi tải mô hình '{name}': {e}")
            traceback.print_exc()
            results[name] = False
            all_success = False

    print("\n==========================================================")
    print("=== BẢNG KIỂM TRA TRẠNG THÁI TẢI (CHECKLIST) ===")
    print("==========================================================")
    for name, success in results.items():
        status = "[SUCCESS] THÀNH CÔNG" if success else "[FAILED] THẤT BẠI"
        print(f"{status:15} : {name}")

    print("==========================================================")
    if all_success:
        print("\n[SUCCESS] TẤT CẢ MODEL ĐÃ SẴN SÀNG! CÓ THỂ CHẠY PIPELINE NGAY BÂY GIỜ.")
        sys.exit(0)
    else:
        print("\n[FATAL] CÓ LỖI XẢY RA! Vui lòng kiểm tra lại mạng. CHƯƠNG TRÌNH DỪNG LẠI (CRASH).")
        sys.exit(1)


if __name__ == "__main__":
    download_all_models()
