# AIC 2026 - Multimodal Video Retrieval System (SOTA Hybrid Architecture)

> **Hệ thống Trích xuất Đặc trưng & Đánh chỉ mục Video Đa phương tiện Khép kín**
> Dự án được tối ưu hóa cho cuộc thi **Ho Chi Minh City AI Challenge (AIC 2026)**, sử dụng kiến trúc lai (Hybrid Architecture) kết hợp Dense Semantic Vector và Sparse Lexical Matrix, tích hợp Canonical Lexicon Layer chống nhiễu từ vựng tiếng Việt.

---

## 1. Tổng quan Kiến trúc Hệ thống (System Overview)

Hệ thống được thiết kế theo mô hình **Split Architecture** tách bạch hoàn toàn quy trình lập chỉ mục ngoại tuyến (Offline Indexing) và truy xuất trực tuyến (Online Retrieval). Loại bỏ ảo tưởng tối ưu hóa, kiến trúc này tập trung vào sự tinh gọn, ổn định của C++ (FAISS) và sức mạnh xử lý ma trận của PyTorch (SpMM).

```text
                                  +-----------------------+
                                  |   Raw Input Videos    |
                                  +-----------+-----------+
                                              |
                                              v
                              +---------------+---------------+
                              |  Video & Audio Preprocessing  |
                              |  (TransNetV2 + InfoShot)      |
                              +---------------+---------------+
                                              |
                                              v
                              +---------------+---------------+
                              | HSV 2D Hist Stillness Pruning |
                              | (Loại bỏ khung hình tĩnh)     |
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +------------+------------+                       +------------+------------+
        | YOLOE-26 + PaddleOCR    |                       |   Silero VAD Routing    |
        | Object & Text Detection |                       | PhoWhisper(ASR) / CLAP  |
        +------------+------------+                       +------------+------------+
                     |                                                 |
                     v                                                 v
        +------------+------------+                       +------------+------------+
        | Spatial Intersection    |                       |     Audio Aggregation   |
        | Denoise OCR by BBoxes   |                       |  (Lời thoại & Tiếng động)|
        +------------+------------+                       +------------+------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +---------------+---------------+
                              |    Window-Based Summarize     |
                              | (Groq API Llama 3.1 8B 30s)   |
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
        |  FAISS Dual Dense Index |                       |  SpMM Sparse Matrix &   |
        | (Vision 768D / Text 1024D|                      |  Canonical Lexicon JSON |
        +-------------------------+                       +-------------------------+
```

---

## 2. Danh mục Công nghệ & Mô hình (Tech Stack & Models)

### Thị giác & Nhận diện (Computer Vision)
* **TransNetV2 & InfoShot:** Phân đoạn Shot và chọn 2 frames (`common` và `sharpest`) theo Laplacian Entropy.
* **HSV 2D Histogram Pruning:** Loại bỏ cực nhanh các frame tĩnh (stillness pruning) cắt giảm 50% gánh nặng embedding.
* **SigLIP 2 (`google/siglip-base-patch16-224`):** Trích xuất Vector Hình ảnh L2-Normalized (768D).
* **YOLOE-26 (`confidence 0.05`, có NMS):** Nhận diện đối tượng thô Open-Vocabulary.
* **PaddleOCR:** Trích xuất văn bản trên khung hình (được lọc thông qua Spatial Intersection với YOLO BBoxes).

### Âm thanh (Audio & Speech)
* **Silero VAD Router:** Phân luồng âm thanh tự động.
* **PhoWhisper (ASR):** Bóc tách lời thoại tiếng Việt nếu có tiếng người.
* **CLAP:** Nhận diện âm thanh môi trường (tiếng mưa, tiếng động cơ) nếu không có tiếng người.

### Ngôn ngữ & Embedding (NLP)
* **Groq API (Llama 3.1 8B Instant):** Sinh mô tả tuyến tính (Semantic Summary) cho các cửa sổ thời gian.
* **BGE-M3 (`BAAI/bge-m3`):** Sinh Dual Embeddings (Dense Vector 1024D + Sparse Lexical Weights 250002D) hỗ trợ Hybrid Retrieval.
* **Canonical Lexicon Parser:** Bộ máy chuẩn hóa khái niệm Tiếng Việt O(1), triệt tiêu lỗi Word Boundary bằng kỹ thuật string padding.

### Chỉ mục & Truy xuất (Indexing & Retrieval)
* **FAISS (GPU IndexFlatIP / IVFFlat):** Xử lý Dense Vectors trên GPU.
* **PyTorch Sparse Matrix Multiplication (SpMM):** Tính Lexical Score trực tiếp trên GPU thay vì CPU BM25.
* **Hybrid Late Fusion:** $S_{final} = 0.5 \cdot S_{vision} + 0.3 \cdot S_{semantic} + 0.2 \cdot S_{lexical}$.
* **Temporal Reranking:** Thuật toán trượt cửa sổ 3-giây để cộng điểm (proximity bonus) cho các shot kề nhau.

---

## 3. Quy trình Triển khai (Phases Breakdown)

### Phase 1: Offline Indexing (Trích xuất Sạch tuyệt đối)
1. Cắt video thành các Shot và trích xuất Frame tĩnh (Pruning).
2. YOLOE-26 và PaddleOCR trích xuất Raw Labels. Lọc OCR rác thông qua thuật toán giao thoa (Intersection).
3. Gom dữ liệu vào khung cửa sổ (Window). LLM tóm tắt bối cảnh phục vụ duy nhất cho Semantic Retrieval.
4. SigLIP 2 tạo Vector Vision (768D), BGE-M3 tạo Dense Vector (1024D) và ghi Sparse Lexical Tensor (.npz).

### Phase 2: Online Retrieval (Vũ khí Tốc độ)
1. **Canonicalization:** Truy vấn đầu vào được xử lý bởi Lexicon Parser, đệm thêm các từ vựng gốc (Canonical IDs) để tạo `sparse_input`.
2. **Dual FAISS Query:** Tìm Top-K frame có Cosine Similarity lớn nhất từ Vision và Text Dense Index.
3. **SpMM Lexical Matching:** Trọng số Sparse của câu query được nhân ma trận (Multiply) với toàn bộ Database Sparse Tensor siêu tốc trên GPU.
4. **Late Fusion & Reranking:** Hợp nhất điểm 3 luồng và chạy Temporal Reranking trả về top 100 kết quả chuẩn xác nhất.

---

## 4. Hướng dẫn Khởi chạy trên Kaggle GPU

### 1. Chuẩn bị API Keys & Kaggle Secrets
Nạp các Secret sau vào Kaggle Notebook (**Add-ons** -> **Secrets**):
* `GROQ_API_KEY`: Chuỗi các Groq Key phân cách bởi dấu phẩy.
* `GEMINI_API_KEY`: Gemini Flash API Key (dự phòng).
* `GITHUB_TOKEN`: Personal Access Token (PAT).

### 2. Thực thi Pipeline
* Clone mã nguồn:
```bash
git clone https://github.com/DntdToM/W1.1Thr34d-AIC26-YOLOe26.git
cd W1.1Thr34d-AIC26-YOLOe26
```
* Khởi chạy Offline Indexing:
```bash
python run_pipeline.py
```
* Khởi chạy Online Retrieval (Testing):
```bash
python src/backend/fast_retrieval.py
```

---

## 5. Cấu trúc Output (Processed Data)

```text
processed_data/
├── 1_frames/                # Khung hình ảnh (.jpg)
├── 2_embeddings/            # FAISS Index (.bin), Sparse Tensors (.npz)
└── 3_metadata/              # Metadata JSON & Canonical Lexicon Cache
```

---

## 6. Giấy phép & Đóng góp (License)

Dự án thuộc bản quyền đội ngũ phát triển **W1.1Thr34d**. Nghiêm cấm sao chép hoặc thương mại hóa khi chưa có sự đồng ý.
