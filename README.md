# 🎬 AIC 2026 - Multimodal Video Retrieval System (Phase 1 Offline Indexing)

> **Hệ thống Trích xuất Đặc trưng & Đánh chỉ mục Video Đa phương tiện Tốc độ cao**  
> Dự án được tối ưu hóa cho cuộc thi **Ho Chi Minh City AI Challenge (AIC 2026)**, sử dụng mô hình thị giác mở, nhận diện đối tượng, đọc chữ OCR, bóc tách âm thanh ASR và tổng hợp ngữ cảnh LLM theo cửa sổ thời gian.

---

## 📌 1. Tổng quan Kiến trúc Hệ thống (System Overview)

Hệ thống được thiết kế theo mô hình **Pipeline Khép kín (Offline Indexing - Phase 1)** với khả năng quản lý tài nguyên GPU/RAM theo chuẩn **Singleton Pattern**, loại bỏ hoàn toàn các lỗi xung đột CUDA/C++, tràn bộ nhớ VRAM và giới hạn Rate-Limit API.

```
                                  +-----------------------+
                                  |   Raw Input Videos    |
                                  +-----------+-----------+
                                              |
                                              v
                              +---------------+---------------+
                              |  Video & Audio Preprocessing  |
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +------------+------------+                       +------------+------------+
        | TransNetV2 + InfoShot   |                       |    Silero VAD + Whisper   |
        | Keyframe Extraction     |                       |    Speech-to-Text (ASR)   |
        +------------+------------+                       +------------+------------+
                     |                                                 |
                     v                                                 v
        +------------+------------+                       +------------+------------+
        | YOLOv9 + EasyOCR/Paddle |                       |  Timestamp Alignment Map |
        | Objects & Text OCR      |                       +------------+------------+
        +------------+------------+                                    |
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +---------------+---------------+
                              |    Window-Based Summarize     |
                              |   (30s-90s Window Batching)   |
                              +---------------+---------------+
                                              |
                                              v
                              +---------------+---------------+
                              |   Groq Multi-Keys Rotation    |
                              |  (Llama 3.1 8B) / Gemini API  |
                              +---------------+---------------+
                                              |
                                              v
                              +---------------+---------------+
                              |  Multimodal Feature Embedding |
                              | (SigLIP 2 Vision + BGE-M3 Text|
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +------------+------------+                       +------------+------------+
        |    FAISS Vector Index   |                       | SQLite Structured DB &    |
        |    (indexFlatIP 768D)   |                       | JSON Frame Metadata       |
        +-------------------------+                       +-------------------------+
```

---

## 🛠️ 2. Danh mục Công nghệ & Mô hình (Tech Stack & Models)

### 👁️ **Thị giác & Nhận diện (Computer Vision)**
* **TransNetV2 (PyTorch CUDA):** Phát hiện ngưỡng cắt cảnh (Shot Boundaries) chính xác mức từng khung hình.
* **InfoShot (Histogram & Laplacian Entropy):** Trích xuất kép 2 Keyframes cho mỗi Shot:
  * `common`: Khung hình cắt cảnh chuẩn cho visual embedding.
  * `sharpest`: Khung hình nét nhất phục vụ đọc chữ OCR.
* **SigLIP 2 (`google/siglip-base-patch16-224`):** Mô hình thị giác mở (Open-Vocabulary Zero-Shot Vision-Language) trích xuất Vector Embedding 768D.
* **YOLOv9-small (`yolov9c.pt`):** Nhận diện đối tượng phổ biến (80 lớp COCO) với từ điển ánh xạ tự động sang Tiếng Việt.

### 📜 **Đọc chữ & Âm thanh (OCR & ASR)**
* **EasyOCR & PaddleOCR (GPU Accelerated):** Trích xuất văn bản tiếng Việt/tiếng Anh trên khung hình video.
* **Silero VAD & PhoWhisper / Whisper-small:** Bóc tách tiếng nói, lọc khoảng lặng và chuyển thể lời thoại ASR chính xác theo mốc thời gian (timestamps).

### 🧠 **Trí tuệ nhân tạo Ngôn ngữ (LLM & Embeddings)**
* **Groq API (Llama 3.1 8B Instant):** Xoay vòng đa khóa (Multi-Key Rotation) với tốc độ > 800 tokens/sec.
* **Google Gemini Flash API (`gemini-flash-latest`):** Fallback tự động khi Groq kịch hạn mức 429.
* **BGE-M3 (`BAAI/bge-m3`):** Trích xuất Dense Text Embedding (1024D) đa ngữ chuẩn hóa L2 Cosine Similarity.

### 💾 **Lưu trữ & Đánh chỉ mục (Indexing & Storage)**
* **FAISS (`IndexFlatIP`):** Đánh chỉ mục Vector không gian 768D tìm kiếm cực nhanh trên GPU/CPU.
* **SQLite (`metadata.db`) & JSON Metadata:** Quản lý siêu dữ liệu có cấu trúc.

---

## 🔄 3. Quy trình Xử lý Toàn diện (End-to-End Workflow)

1. **Phân đoạn Video (Shot Segmentation):**  
   Video đầu vào được phân tích qua TransNetV2 để tách thành các Shots. InfoShot chọn ra 2 frames (`common` & `sharpest`) cho mỗi phân cảnh.
2. **Trích xuất Đặc trưng Thô (Raw Analytics):**  
   YOLOv9 quét vật thể, EasyOCR đọc chữ màn hình, Silero VAD + Whisper bóc tách lời thoại ASR.
3. **Gom cụm Cửa sổ & Tổng hợp Ngữ cảnh LLM (Window-Based Summarization):**  
   Dữ liệu được gom thành từng cửa sổ 30s–90s. LLM chỉ được gọi **ĐÚNG 1 LẦN per batch** (giảm 98% chi phí API), áp dụng prompt chuyên gia 60 từ để sinh ra đoạn văn xuôi bối cảnh hoàn chỉnh, bảo tồn nguyên vẹn tên riêng và từ gốc tiếng Anh.
4. **Tạo Vector Embedding Đa thức (Multimodal Embedding):**  
   * SigLIP 2 tạo Vector hình ảnh 768D từ Keyframes.
   * BGE-M3 tạo Vector văn bản 1024D từ chuỗi kết hợp `[Vật thể + Chữ OCR + Lời nói ASR + Bối cảnh LLM]`.
5. **Đánh chỉ mục FAISS & Lưu trữ Metadata:**  
   Tất cả vector được chuẩn hóa L2 và lưu vào `faiss_index.bin`. Metadata được ghi đồng thời vào file `.json` và database SQLite `metadata.db`.

---

## ⚡ 4. Hướng dẫn Khởi chạy trên Kaggle GPU

### 1. Chuẩn bị API Keys & Kaggle Secrets
Nạp các Secret sau vào Kaggle Notebook (**Add-ons** ➔ **Secrets**):
* `GROQ_API_KEY`: Chuỗi các Groq Key phân cách bởi dấu phẩy (`gsk_key1,gsk_key2,gsk_key3,gsk_key4`).
* `GEMINI_API_KEY`: Gemini Flash API Key.
* `GITHUB_TOKEN`: Personal Access Token (PAT) nếu Repo để ở chế độ Private.

### 2. Thực thi Notebook `kaggle_executor.ipynb`

* **Cell 1:** Clone / Pull mã nguồn mới nhất từ GitHub.
```python
!git clone https://github.com/DntdToM/aic2026_retrieval_project.git
%cd aic2026_retrieval_project
```
* **Cell 2 & 3:** Cài đặt phụ thuộc hệ thống & thư viện Python.
```bash
!apt-get update && apt-get install -y ffmpeg libsm6 libxext6
!pip install -r requirements.txt
```
* **Cell 4:** Nạp API Keys & Tải Weights mô hình.
* **Cell 5:** Khởi chạy Offline Indexing Pipeline.
```bash
!python run_pipeline.py
```
* **Cell 6:** Nén kết quả thành `output_data.zip` để tải về.

---

## 📁 5. Cấu trúc Thư mục Kết quả (Output Structure)

```text
processed_data/
├── 1_frames/                # Thư mục lưu các file ảnh Keyframes (.jpg)
├── 2_embeddings/            # Thư mục lưu Vector Embeddings (.npy) & FAISS Index (.bin)
│   ├── video_name_img_emb.npy
│   ├── video_name_text_emb.npy
│   └── faiss_index.bin
└── 3_metadata/              # Thư mục lưu Metadata (.json) & Database SQLite (.db)
    ├── video_name_metadata.json
    └── metadata.db
```

---

## 🛡️ 6. Giấy phép & Đóng góp (License)

Dự án thuộc bản quyền đội ngũ phát triển **AIC 2026 Retrieval Team**. Nghiêm cấm sao chép hoặc thương mại hóa khi chưa có sự đồng ý.
