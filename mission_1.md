Mình đã chỉnh lại prompt theo hướng **Production Specification** hơn. Mình giữ nguyên ý tưởng của bạn, chỉ bổ sung các phần còn thiếu:

- Đổi `related_terms` → `controlled_variants` để tránh hiểu sai semantic.
- Thêm `source` metadata.
- Thêm `stable identifier policy`.
- Thêm rule parser không được "thông minh quá".
- Thêm regression test cho version stability.
- Làm rõ Dense/Sparse pipeline.
- Thêm yêu cầu không phá API hiện tại.
- Thêm khả năng thay Trie/Aho-Corasick.
- Sửa một vài câu để AI coding assistant ít tự diễn giải sai.

---

```markdown
# Production Specification: Canonical Lexicon Layer for Multimedia Retrieval (AIC2026)

Bạn là một Senior System Architect và Search Engineer.

Nhiệm vụ của bạn là xây dựng cơ chế **Canonicalization Layer** (Chuẩn hóa khái niệm) cho hệ thống Multimedia Retrieval (AIC2026).

## Mục tiêu cốt lõi

Thiết lập một parser chuẩn hóa từ vựng:

- Tốc độ cực cao, chạy hoàn toàn offline/in-memory.
- Deterministic (cùng input luôn cho cùng output).
- Tách biệt hoàn toàn với Semantic Inference.
- Không phụ thuộc LLM.
- Sử dụng Canonical Lexicon làm nguồn sự thật duy nhất (Single Source of Truth).
- Đảm bảo tính nhất quán dữ liệu dài hạn cho Index, Metadata và Retrieval.

---

# Bước 1: Dọn dẹp Phase 1 (Data Ingestion)

## File:
```

src/preprocessing/vision_ocr_obj.py

```

Yêu cầu:

- Gỡ bỏ hoàn toàn logic gọi LLM để dịch hoặc chuẩn hóa nhãn vật thể.
- Không dùng Groq/Gemini/LLM cho object normalization.
- Giữ nguyên raw label mà vision detector trả về.

Ví dụ:

Detector output:

```

car
dog
traffic light

```

Phải được lưu nguyên dạng.

Không được biến đổi:

```

car -> ô tô
dog -> chó

```

Phase 1 chỉ có nhiệm vụ:

```

Detection
|
v
Raw Labels
|
v
Index Storage

```

Không canonicalize ở bước này.

Format chuẩn:

```

Ngữ cảnh: {summary}.
Vật thể: {raw_detector_labels}.
Chữ: {ocr}.
Âm thanh: {audio}.

```

---

# Bước 2: Tạo Canonical Lexicon (Versioning + Feature Flag)

## File:

```

src/utils/canonical_lexicon.json

```

hoặc:

```

src/utils/canonical_lexicon.yaml

```

Không sử dụng tên ontology.

Lý do:

Module này chỉ thực hiện:

```

surface form
|
v
canonical id

````

Không chứa hierarchy knowledge graph.

---

## Schema bắt buộc:

```json
{
  "metadata": {
    "version": "1.0.0",
    "updated_at": "2026-07-27"
  },

  "entities": {

    "car": {
      "enabled": true,

      "source": [
        "COCO",
        "YOLOE",
        "manual"
      ],

      "aliases": {
        "vi": [
          "ô tô",
          "xe hơi",
          "xe bốn bánh"
        ],
        "en": [
          "car",
          "automobile"
        ]
      },

      "controlled_variants": {
        "vi": [
          "xe con",
          "xe 4 chỗ"
        ],
        "en": [
          "sedan",
          "hatchback"
        ]
      }
    },


    "suv": {
      "enabled": true,

      "source": [
        "YOLOE",
        "manual"
      ],

      "aliases": {
        "vi": [
          "xe thể thao đa dụng",
          "xe gầm cao"
        ],
        "en": [
          "suv"
        ]
      }
    }
  }
}
````

---

## Quy tắc Semantic bắt buộc:

### aliases

Chỉ chứa:

- từ đồng nghĩa trực tiếp.
- các cách gọi khác nhau của cùng một khái niệm.

Ví dụ:

```
car
ô tô
xe hơi
automobile
```

được phép map:

```
car
```

---

### controlled_variants

Chỉ được map về Canonical ID khi:

- Detector hiện tại KHÔNG phân biệt được variant đó.
- Việc gộp không làm mất thông tin retrieval.

Ví dụ:

Nếu detector chỉ có:

```
car
```

thì:

```
sedan -> car
```

có thể chấp nhận.

Nhưng nếu detector hỗ trợ:

```
car
sedan
suv
```

thì:

```
suv
```

phải là Canonical ID riêng.

Không được:

```
suv -> car
```

---

# Bước 3: Viết Canonical Lexicon Parser

## File:

```
src/utils/lexicon_parser.py
```

---

## Ràng buộc tối thượng:

Parser CHỈ làm:

```
Canonicalization
```

Không được làm:

- Translation.
- Stemming.
- Lemmatization.
- Query Expansion.
- Semantic Search.
- Synonym Generation.
- LLM inference.

Semantic understanding thuộc về:

```
BGE-M3 Dense Encoder
```

---

# Normalization

Viết:

```python
normalize_text(text)
```

Bắt buộc:

- Unicode NFC normalization.
- lowercase.
- strip.
- collapse whitespace.

Áp dụng cho:

- Query.
- Alias trong lexicon.
- Controlled variants.

---

# Loading & Cache

Viết:

```python
load_lexicon(file_path)
```

Yêu cầu:

- Load đúng 1 lần khi server startup.
- Không đọc file mỗi request.
- Có thể cache bằng singleton hoặc module-level cache.

Bỏ qua:

- metadata.
- node có:

```json
"enabled": false
```

---

# Internal Representation

Flatten thành cấu trúc tối ưu:

Ví dụ:

```
{
 "xe hơi": "car",
 "ô tô": "car",
 "car": "car",
 "sedan": "car"
}
```

Không hardcode.

Toàn bộ rule phải đến từ file lexicon.

---

# Extraction API

Viết:

```python
extract_canonical_terms(
    query,
    lexicon_cache
)
```

Output:

```python
[
 "car",
 "person"
]
```

---

# Matching Algorithm

Yêu cầu:

## Longest Match First

Ví dụ:

Query:

```
máy bay không người lái
```

Nếu lexicon có:

```
máy bay
máy bay không người lái
```

Phải chọn:

```
drone
```

Không được trả:

```
airplane
drone
```

---

## Overlap Handling

Sau khi match cụm dài:

- đánh dấu span.
- không cho cụm ngắn match đè lên.

---

## Boundary Safety

Không dùng:

```python
if key in query
```

Cấm substring matching.

Ví dụ:

Không được match:

```
car
```

trong:

```
carpet
```

Cần hỗ trợ Unicode-aware matching.

---

## Future Algorithm Replacement

Public API phải độc lập implementation.

Hiện tại có thể dùng:

- Regex.
- Sorted matching.

Nhưng tương lai có thể thay bằng:

- Trie.
- Aho-Corasick.

Không được làm thay đổi:

```python
extract_canonical_terms()
```

---

# Bước 4: Retrieval Pipeline

## File:

```
src/backend/fast_retrieval.py
```

---

Input:

```python
original_query
```

Normalize:

```python
normalized_query = normalize_text(original_query)
```

Extract:

```python
canonical_terms = extract_canonical_terms(
    normalized_query,
    lexicon_cache
)
```

---

# BGE-M3 Processing

Không ép toàn bộ input thành một chuỗi semantic giả.

Tách:

## Dense Input

Dùng cho semantic understanding:

```python
dense_input = original_query
```

Giữ nguyên ngôn ngữ tự nhiên.

---

## Sparse Input

Dùng cho lexical matching:

```python
sparse_input = (
    original_query +
    " " +
    " ".join(canonical_terms)
)
```

Mục tiêu:

Tăng lexical weight chính xác cho:

- BGE-M3 Sparse Vector.
- BM25.
- PyTorch SpMM.

---

Sau đó tiếp tục:

```
Dense Vector
      |
      v
FAISS Vision/Text


Sparse Vector
      |
      v
SpMM/BM25


Late Fusion
```

---

# Bước 5: Unit Test Bắt Buộc

## File:

```
tests/test_lexicon_parser.py
```

Dùng pytest.

---

Test cases:

## Feature Flag

Đảm bảo:

```
enabled:false
```

không được load.

---

## Exact Alias

Ví dụ:

```
xe hơi
```

ra:

```
car
```

---

## Controlled Variant

Ví dụ:

```
sedan
```

ra:

```
car
```

khi variant được enable.

---

## Longest Match

Ví dụ:

Input:

```
máy bay không người lái
```

Output:

```
drone
```

Không thêm:

```
airplane
```

---

## Boundary

Test:

```
car
```

không match:

```
carpet
```

---

## Unicode

Test:

- NFC/NFD Vietnamese characters.
- Multiple spaces.
- Mixed uppercase/lowercase.

---

## Stability Regression

Đảm bảo cùng alias luôn trả cùng Canonical ID.

Ví dụ:

```
xe hơi -> car
```

sau nhiều lần reload.

---

# Bước 6: Critical Architecture Rules

## Stable Identifier Policy

Canonical ID là immutable.

Ví dụ:

```
car
```

đã production:

Không được đổi thành:

```
automobile
```

Không được tái sử dụng:

```
car = motorcycle
```

Nếu thay đổi semantic:

Tạo ID mới.

Mục tiêu:

Bảo vệ:

- Existing metadata.
- Existing vector index.
- Historical retrieval.

---

## Backward Compatibility

Không thay đổi:

- Public function signature.
- Import path.
- Existing module interface.

Nếu bắt buộc thay đổi:

Phải giữ adapter layer.

---

## No Hardcoding

Cấm:

```python
{
 "xe hơi": "car"
}
```

trong source code.

Mọi mapping phải nằm trong:

```
canonical_lexicon.json
```

---

## Final Requirement

Code phải:

- Type hint đầy đủ.
- Có comment rõ ràng.
- Không over-engineer.
- Tối ưu cho inference latency.
- Dễ thay thế thuật toán matching trong tương lai.

```

---

Bản này mình nghĩ là phiên bản nên dùng. Điểm quan trọng nhất là nó khóa được "AI coding assistant tự sáng tạo quá mức" — thứ hay phá những hệ thống retrieval kiểu này. Nó buộc AI hiểu rằng đây là **một lớp deterministic indexing/retrieval infrastructure**, không phải một bài toán NLP cần "thông minh".
```
