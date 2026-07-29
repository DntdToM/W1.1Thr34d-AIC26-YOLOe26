======================================================
PHASE 1: OFFLINE INDEXING (TRÍCH XUẤT DỮ LIỆU SẠCH TUYỆT ĐỐI)
======================================================
[Video Thô (mp4, mkv)]
│
▼
[VideoSegmenter & InfoShot]
├──► TransNetV2 (Cắt Shot)
└──► InfoShot (Chọn frame nét nhất bằng Laplacian)
│
▼
[Stillness Pruning (Lọc tĩnh)] ◄ TỐI ƯU CẮT GIẢM RÁC
└──► Tính Cosine Similarity bằng SigLIP 2 giữa các frame liên tiếp.
└──► NẾU Sim > 0.95 (cảnh không đổi) ─► Bỏ qua, giảm 50% khối lượng index.
│
├─────────────────────────────────────────────┐
▼ (Luồng Hình ảnh) ▼ (Luồng Âm thanh)
[Vision Analytics Parallel] [Audio Router] ◄ BẮT "ÂM THANH MÙ"
├──► YOLOE-26 (Bắt vật thể, không dùng NMS) ├──► Silero VAD check tiếng người
└──► PaddleOCR (Đọc chữ) │
│ ├──► CÓ tiếng: PhoWhisper (ASR)
│ └──► KHÔNG tiếng: CLAP (Tiếng mưa, động cơ...)
│ │
└──────────────────────┬──────────────────────┘
│
▼
[Data Aggregation & Denoising] (Gom nhóm theo cửa sổ 30s)
└──► Giao thoa không gian (Spatial Intersection):
Xóa sổ các cụm từ OCR vô nghĩa nếu nó nằm trơ trọi, không gắn với Bounding Box của YOLOE-26.
└──► Dữ liệu thô sau lọc: { "vật_thể": [...], "ocr": [...], "âm_thanh": "..." }
│
▼
[Context Summarization]
└──► Groq API (Llama 3.1 8B). Ép Prompt xuất đúng 1 câu mô tả tuyến tính, không ảo giác.
│
┌──────────────────────┴──────────────────────┐
▼ ▼
[SigLIP 2 Vision Embedding] [BGE-M3 Text Embedding]
└──► L2 Normalize (768D) └──► Input: Văn bản tóm tắt + OCR sạch
│ └──► L2 Normalize (1024D)
└──────────────────────┬──────────────────────┘
│
======================================================
PHASE 2: ONLINE RETRIEVAL (VŨ KHÍ TỐC ĐỘ TRÊN BÀN THI ĐẤU)
======================================================
▼
[LƯU TRỮ CHUYÊN BIỆT: CHIA ĐỂ TRỊ (SPLIT ARCHITECTURE)]
(Loại bỏ Qdrant/Milvus để ép xung tối đa trên Server)
│
┌──────────────────────┴──────────────────────┐
▼ (Tải lên VRAM GPU) ▼ (Tải lên RAM CPU)
[DUAL FAISS GPU INDEX] [IN-MEMORY DICTIONARY]
├──► faiss.GpuIndexFlatIP (768D) └──► Dictionary/Pickle Hash Map siêu tốc
│ (Chứa 100% Vector Vision) └──► Ánh xạ O(1) để xử lý logic:
│ {
├──► faiss.GpuIndexFlatIP (1024D) "frame_123": {
(Chứa 100% Vector Text) "video_id": "vid_01",
"timestamp_ms": 15000,
"objects": ["xe tải", "chó"],
"ocr_text": "quán phở 24h"
}
}
│
======================================================
KHI NHẬN QUERY TỪ HỆ THỐNG CHẤM ĐIỂM (REAL-TIME EXECUTION)
======================================================
▼
[Query: "Xe tải chạy qua quán phở 24h, có tiếng chó sủa"]

Bước 1 (Trích xuất): Bắn Query qua SigLIP (768D) và BGE-M3 (1024D).
│
Bước 2 (Tìm Vector): Quét song song trên GPU qua 2 index FAISS.
Lấy thẳng Top 2000 frame_id có Cosine lớn nhất.
(Thời gian: < 15 mili-giây).
│
Bước 3 (Lọc Cứng): Vác 2000 frame_id sang In-Memory Dictionary (trên RAM).
Chạy code Python loại trừ ngay các frame không chứa "xe tải" hoặc "phở".
│
Bước 4 (Reranking): Chạy thuật toán Two-Pointer Temporal trên tập frame đã lọc siêu nhỏ.
Áp dụng proximity_bonus để nối các shot liên tiếp.
│
Bước 5 (Output): Đóng gói JSON Top 100 trả về Server BTC.
