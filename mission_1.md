Tôi đã đọc kỹ toàn bộ tài liệu về Vòng sơ tuyển AIC 2026. Đối chiếu với hệ thống Multimedia Retrieval (YOLOE-26, BGE-M3, Canonical Lexicon) mà chúng ta đã thiết kế, đây là bản phân tích chiến thuật thực chiến giúp bạn tối ưu hóa hệ thống để "ăn điểm" tối đa.

Phân Tích 3 Dạng Truy Vấn (Query Types) & Chiến Lược Của Hệ Thống
Truy vấn 1: Textual KIS (Tìm kiếm chính xác): Yêu cầu tìm đúng video và khung hình dựa trên mô tả. Kiến trúc hiện tại của bạn sẽ xử lý rất mượt dạng này. Luồng BGE-M3 (Dense) sẽ quét ngữ cảnh, và Lexicon (Sparse) sẽ khóa chặt các danh từ cụ thể để trả về chính xác đoạn video cần tìm.  
PDF

Truy vấn 2: Q&A (Hỏi - Đáp): Yêu cầu tìm khoảnh khắc và trả lời một câu hỏi cụ thể (ví dụ: đếm số người, xác định màu sắc). Hệ thống hiện tại mới chỉ làm tốt khâu Retrieval (Tìm kiếm). Bạn sẽ cần bổ sung một module Vision-Language Model (VLM) (như Gemini 1.5 Flash hoặc Qwen-VL) ở giai đoạn cuối (Post-retrieval). Module này sẽ đọc Top 5 khung hình tốt nhất mà FAISS trả về để sinh ra câu trả lời text (ví dụ: "5") nộp cho hệ thống.  
PDF

Truy vấn 3: TRAKE (Căn chỉnh sự kiện): Đây là dạng phức tạp nhất, yêu cầu tìm 1 video chứa chuỗi sự kiện và xác định chính xác từng khung hình ngữ nghĩa (semantic keyframe) theo trình tự. Yêu cầu này xác nhận rằng thuật toán tìm kiếm khung hình đơn lẻ (Single-frame Search) là chưa đủ. Bạn bắt buộc phải triển khai thuật toán Temporal Reranking (Gộp cụm thời gian bằng Two-Pointer hoặc Dynamic Programming) để nhóm các kết quả lại thành một chuỗi logic (vd: Chạy đà -> Giậm nhảy -> Bay qua xà -> Tiếp đất).  
PDF

- 1

Phân Tích Thuật Toán Chấm Điểm (Evaluation Metrics)
Luật chấm điểm của AIC 2026 cực kỳ khắc nghiệt và đặt nặng vấn đề Ranking (Xếp hạng kết quả).  
PDF

Công thức tính điểm R-Score cơ bản:

Với KIS: R−Score(r
i
​
)=I(v
i
​
=GT
v
​
∧id
i
​
∈[s,e]).  
PDF

Với Q&A: R−Score(r
i
​
)=I(v
i
​
=GT
v
​
∧id
i
​
∈[s,e]∧a
i
​
=GT
a
​
).  
PDF

Với TRAKE: R−Score(r
i
​
)=
N
1
​
∑
j=1
N
​
I(id
i,j
​
∈[s
j
​
,e
j
​
]) (nếu đúng video).  
PDF

Bản chất của Final Score (Điểm Quyết Định):
Điểm của bạn được tính bằng trung bình cộng của các R-Score tốt nhất tại các mốc Top-k.
FinalScore=
5
1
​
∑
k∈{1,5,20,50,100}
​
R@k.
(Trong đó R@k=max
1≤i≤k
​
{R−Score(r
i
​
)})  
PDF

- 2

Hệ quả thực chiến: Thuật toán bắt buộc bạn phải đẩy kết quả đúng lên Top 1. Nếu bạn nộp 100 kết quả, và đáp án đúng hoàn hảo (R-Score = 1) của bạn nằm ở vị trí thứ 51, bạn chỉ ăn được điểm ở mốc R@100. Các mốc R@1, R@5, R@20, R@50 đều bằng 0. Lúc này Final Score của bạn chỉ là 0.2. Nếu đẩy được lên Top 1, bạn ăn trọn 1.0 điểm. Do đó, module Late Fusion (Kết hợp điểm Dense + Sparse) của bạn phải được tinh chỉnh trọng số (weights) cực kỳ kỹ lưỡng để ranking kết quả.  
PDF

Đánh Giá Về Dữ Liệu Cung Cấp (Dataset)
Baseline của BTC: Ban tổ chức cung cấp sẵn Object detection (từ Faster R-CNN) và CLIP features (ViT-B-32) để các đội tham khảo.  
PDF

Lợi thế của chúng ta: Việc bạn tự chạy lại Phase 1 bằng YOLOE-26L-PF (nhận diện Open-Vocabulary mạnh mẽ hơn Faster R-CNN) và nhúng vector bằng BGE-M3 (mạnh hơn nhiều so với CLIP ViT-B-32 cũ) là một nước đi "outplay" hoàn toàn baseline của giải đấu. Bạn chỉ cần lấy file Video gốc gốc để tự build lại toàn bộ Index. Lượng dữ liệu đợt 1 lấy từ AIC 2025 khá vừa phải, đủ để bạn chạy benchmark tốc độ và tinh chỉnh hệ thống trước khi batch 2 đổ bộ.  
PDF

- 1
