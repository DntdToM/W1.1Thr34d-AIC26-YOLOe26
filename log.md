2026-07-24 03:02:45,313 - INFO - Loading faiss with AVX512 support.
2026-07-24 03:02:45,313 - INFO - Could not load library with AVX512 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx512'")
2026-07-24 03:02:45,313 - INFO - Loading faiss with AVX2 support.
2026-07-24 03:02:45,313 - INFO - Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-24 03:02:45,313 - INFO - Loading faiss.
2026-07-24 03:02:45,339 - INFO - Successfully loaded faiss.
2026-07-24 03:02:45,345 - INFO - ==========================================================
2026-07-24 03:02:45,345 - INFO - === BẮT ĐẦU QUY TRÌNH OFFLINE INDEXING (PHASE 1 PIPELINE) ===
2026-07-24 03:02:45,345 - INFO - ==========================================================
2026-07-24 03:02:45,350 - INFO - Phát hiện 4 video chính thức trong 'data/official_videos'.
2026-07-24 03:02:45,599 - INFO - Phát hiện CUDA GPU: Tự động tối ưu max_workers=1 để dồn 100% CUDA Compute và triệt tiêu treo luồng VRAM.
2026-07-24 03:02:45,600 - INFO - Đang nạp các mô hình GPU/CPU theo Singleton Pattern vào VRAM dùng chung...
2026-07-24 03:02:45,606 - INFO - Local weights 'src/preprocessing/weights/transnetv2-pytorch-weights.pth' chưa khởi tạo, nạp gói TransNetV2 PyTorch mặc định...
2026-07-24 03:02:47,157 - INFO - Đã khởi tạo mô hình TransNetV2 PyTorch (CUDA GPU) thành công.
2026-07-24 03:02:47,268 - INFO - Sử dụng bundled ffmpeg binary: /usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9c.pt to 'yolov9c.pt': 100% ━━━━━━━━━━━━ 49.4MB 155.5MB/s 0.3s
2026-07-24 03:02:48,613 - INFO - Đã nạp mô hình YOLOv9 từ yolov9c.pt thành công.
2026-07-24 03:02:52,099 - INFO - Đã nạp EasyOCR (languages=['vi', 'en'], gpu=True) thành công.
2026-07-24 03:02:52,099 - INFO - PaddleOCR chưa có sẵn: No module named 'paddleocr'
2026-07-24 03:02:52,104 - INFO - Khởi tạo EmbeddingGenerator trên thiết bị: cuda (FP16=True)
2026-07-24 03:02:56,485 - INFO - Đang nạp mô hình SigLIP 2 (dtype=torch.float16) từ 'models/siglip-base-patch16-224'...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 408/408 [00:00<00:00, 911.67it/s, Materializing param=v
2026-07-24 03:02:57,476 - INFO - Đã nạp mô hình SigLIP 2 thành công.
2026-07-24 03:02:57,864 - INFO - TensorFlow version 2.20.0 available.
2026-07-24 03:02:57,865 - INFO - JAX version 0.7.2 available.
2026-07-24 03:03:00,574 - INFO - Đang nạp mô hình BGE-M3 từ 'models/bge-m3'...
2026-07-24 03:03:00,577 - INFO - Loading SentenceTransformer model from models/bge-m3.
Loading weights: 100%|█| 391/391 [00:00<00:00, 811.36it/s, Materializing param=p
2026-07-24 03:03:04,510 - INFO - HTTP Request: GET https://huggingface.co/api/models/models/bge-m3 "HTTP/1.1 401 Unauthorized"
2026-07-24 03:03:04,838 - INFO - Đã nạp mô hình BGE-M3 qua SentenceTransformer thành công.
2026-07-24 03:03:04,838 - INFO - Đã sẵn sàng MultiThreadPipelineWorker (1 max_workers, VRAM khống chế hằng số ~7.5GB).
2026-07-24 03:03:04,838 - INFO - Đang đẩy 4 video vào MultiThreadWorker...
2026-07-24 03:03:04,838 - INFO - Bắt đầu xử lý song song 4 videos (Singleton Models preloaded)...
2026-07-24 03:03:04,839 - INFO - === Bắt đầu xử lý Video: 1_news_60s_720p ===
2026-07-24 03:03:04,839 - INFO - Đang bóc tách video với TransNet V2 & InfoShot: data/official_videos/dummy_videos/1_news_60s_720p.mp4
Offline Indexing Pipeline:   0%|                          | 0/4 [00:00<?, ?it/s]2026-07-24 03:03:25,986 - INFO - TransNet V2 phát hiện 95 shots cho video 1_news_60s_720p.mp4
2026-07-24 03:05:08,451 - INFO - InfoShot đã lưu 190 keyframes vào processed_data/1_frames
/usr/local/lib/python3.12/dist-packages/torch/hub.py:247: UserWarning: You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to load(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or load(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
  _check_repo_is_trusted(
Downloading: "https://github.com/snakers4/silero-vad/zipball/master" to /root/.cache/torch/hub/master.zip
2026-07-24 03:05:10,741 - INFO - Đã nạp mô hình Silero VAD thành công.
2026-07-24 03:05:16,924 - INFO - Silero VAD phát hiện 61 phân đoạn giọng nói.
`torch_dtype` is deprecated! Use `dtype` instead!

Loading weights:   0%|                                  | 0/479 [00:00<?, ?it/s]