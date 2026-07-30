Waiting for OCR Microservice to be ready...
[WARNING] OCR Microservice might not be ready yet.
2026-07-30 09:54:43,334 - INFO - Loading faiss with AVX512 support.
2026-07-30 09:54:43,335 - INFO - Could not load library with AVX512 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx512'")
2026-07-30 09:54:43,335 - INFO - Loading faiss with AVX2 support.
2026-07-30 09:54:43,335 - INFO - Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-30 09:54:43,335 - INFO - Loading faiss.
2026-07-30 09:54:43,370 - INFO - Successfully loaded faiss.
2026-07-30 09:54:43,375 - INFO - ==========================================================
2026-07-30 09:54:43,375 - INFO - === STARTING OFFLINE INDEXING PIPELINE (PHASE 1) ===
2026-07-30 09:54:43,375 - INFO - ==========================================================
2026-07-30 09:54:43,381 - INFO - Discovered 4 official video files in 'data/official_videos'.
2026-07-30 09:54:43,616 - INFO - CUDA GPU detected: Constraining max_workers to 1 to optimize CUDA compute efficiency.
2026-07-30 09:54:43,617 - INFO - Pre-loading shared model singletons...
2026-07-30 09:54:43,623 - INFO - Local weights file 'src/preprocessing/weights/transnetv2-pytorch-weights.pth' not found. Initializing default TransNetV2 PyTorch model...
2026-07-30 09:54:45,169 - INFO - TransNetV2 model initialized successfully on CUDA GPU.
2026-07-30 09:54:45,179 - INFO - Using bundled ffmpeg binary at '/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
2026-07-30 09:54:45,942 - INFO - YOLOE-26 model loaded successfully from 'yoloe-26l-seg-pf.pt'.
2026-07-30 09:54:45,945 - INFO - PaddleOCR Microservice unavailable (is setup_ocr_server.sh running?): HTTPConnectionPool(host='localhost', port=5050): Max retries exceeded with url: /health (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7d0ec49dbe60>: Failed to establish a new connection: [Errno 111] Connection refused'))
2026-07-30 09:54:45,950 - INFO - EmbeddingGenerator initialized on device 'cuda' (fp16=True).
2026-07-30 09:54:50,799 - INFO - Loading SigLIP 2 model from 'models/siglip-base-patch16-224' (dtype=torch.float16)...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 408/408 [00:00<00:00, 891.20it/s, Materializing param=v
2026-07-30 09:54:51,583 - INFO - SigLIP 2 model loaded successfully.
2026-07-30 09:54:51,584 - WARNING - FlagEmbedding not available (No module named 'FlagEmbedding'). Falling back to SentenceTransformer (Dense only).
2026-07-30 09:54:52,156 - INFO - TensorFlow version 2.20.0 available.
2026-07-30 09:54:52,158 - INFO - JAX version 0.7.2 available.
2026-07-30 09:54:54,812 - INFO - Loading SentenceTransformer model from models/bge-m3.
Loading weights: 100%|█| 391/391 [00:00<00:00, 628.70it/s, Materializing param=p
2026-07-30 09:54:58,322 - INFO - HTTP Request: GET https://huggingface.co/api/models/models/bge-m3 "HTTP/1.1 401 Unauthorized"
2026-07-30 09:54:58,651 - INFO - BGE-M3 model loaded successfully via SentenceTransformer.
2026-07-30 09:54:58,651 - INFO - MultiThreadPipelineWorker initialized (max_workers=1).
2026-07-30 09:54:58,651 - INFO - Dispatching 4 video files to MultiThreadPipelineWorker...
2026-07-30 09:54:58,651 - INFO - Initiating batch processing for 4 videos...
2026-07-30 09:54:58,652 - INFO - Processing video: 1_news_60s_720p
2026-07-30 09:54:58,652 - INFO - Segmenting video: data/official_videos/dummy_videos/1_news_60s_720p.mp4
Offline Indexing Pipeline: 0%| | 0/4 [00:00<?, ?it/s]2026-07-30 09:55:18,902 - INFO - TransNetV2 detected 95 shots in video '1_news_60s_720p.mp4'.
2026-07-30 09:55:22,846 - INFO - Shot 3 sharpest frame pruned (sim=0.9994)
2026-07-30 09:55:24,671 - INFO - Shot 4 sharpest frame pruned (sim=0.9996)
2026-07-30 09:55:25,942 - INFO - Shot 5 sharpest frame pruned (sim=0.9894)
2026-07-30 09:55:26,741 - INFO - Shot 6 sharpest frame pruned (sim=0.9999)
2026-07-30 09:55:27,481 - INFO - Shot 7 sharpest frame pruned (sim=0.9987)
2026-07-30 09:55:28,847 - INFO - Shot 8 sharpest frame pruned (sim=0.9958)
2026-07-30 09:55:30,085 - INFO - Shot 9 sharpest frame pruned (sim=0.9869)
2026-07-30 09:55:30,889 - INFO - Shot 10 sharpest frame pruned (sim=0.9981)
2026-07-30 09:55:32,017 - INFO - Shot 11 sharpest frame pruned (sim=0.9990)
2026-07-30 09:55:33,927 - INFO - Shot 12 sharpest frame pruned (sim=0.9925)
2026-07-30 09:55:35,203 - INFO - Shot 13 sharpest frame pruned (sim=0.9926)
2026-07-30 09:55:37,864 - INFO - Shot 15 sharpest frame pruned (sim=0.9881)
2026-07-30 09:55:39,189 - INFO - Shot 16 sharpest frame pruned (sim=0.9954)
2026-07-30 09:55:39,901 - INFO - Shot 17 sharpest frame pruned (sim=0.9928)
2026-07-30 09:55:40,871 - INFO - Shot 18 sharpest frame pruned (sim=0.9939)
2026-07-30 09:55:41,901 - INFO - Shot 19 sharpest frame pruned (sim=0.9958)
2026-07-30 09:55:42,758 - INFO - Shot 20 sharpest frame pruned (sim=0.9981)
2026-07-30 09:55:46,230 - INFO - Shot 22 sharpest frame pruned (sim=0.9947)
2026-07-30 09:55:48,114 - INFO - Shot 24 sharpest frame pruned (sim=0.9994)
2026-07-30 09:55:49,004 - INFO - Shot 25 sharpest frame pruned (sim=0.9992)
2026-07-30 09:55:50,435 - INFO - Shot 26 sharpest frame pruned (sim=0.9834)
2026-07-30 09:55:52,624 - INFO - Shot 28 sharpest frame pruned (sim=0.9927)
2026-07-30 09:55:58,252 - INFO - Shot 32 sharpest frame pruned (sim=0.9950)
2026-07-30 09:55:59,312 - INFO - Shot 33 sharpest frame pruned (sim=0.9998)
2026-07-30 09:56:00,066 - INFO - Shot 34 sharpest frame pruned (sim=0.9840)
2026-07-30 09:56:00,713 - INFO - Shot 35 sharpest frame pruned (sim=0.9997)
2026-07-30 09:56:04,216 - INFO - Shot 38 sharpest frame pruned (sim=0.9924)
2026-07-30 09:56:05,854 - INFO - Shot 40 sharpest frame pruned (sim=0.9867)
2026-07-30 09:56:11,676 - INFO - Shot 45 sharpest frame pruned (sim=0.9961)
2026-07-30 09:56:15,639 - INFO - Shot 49 sharpest frame pruned (sim=0.9886)
2026-07-30 09:56:16,709 - INFO - Shot 50 sharpest frame pruned (sim=0.9868)
2026-07-30 09:56:17,547 - INFO - Shot 51 sharpest frame pruned (sim=0.9998)
2026-07-30 09:56:18,640 - INFO - Shot 52 sharpest frame pruned (sim=0.9984)
2026-07-30 09:56:22,462 - INFO - Shot 55 sharpest frame pruned (sim=0.9905)
2026-07-30 09:56:32,786 - INFO - Shot 65 sharpest frame pruned (sim=0.9833)
2026-07-30 09:56:34,514 - INFO - Shot 66 sharpest frame pruned (sim=0.9879)
2026-07-30 09:56:35,857 - INFO - Shot 67 sharpest frame pruned (sim=0.9956)
2026-07-30 09:56:38,074 - INFO - Shot 69 sharpest frame pruned (sim=0.9946)
2026-07-30 09:56:39,030 - INFO - Shot 70 sharpest frame pruned (sim=0.9862)
2026-07-30 09:56:39,742 - INFO - Shot 71 sharpest frame pruned (sim=0.9981)
2026-07-30 09:56:40,955 - INFO - Shot 72 sharpest frame pruned (sim=0.9939)
2026-07-30 09:56:42,832 - INFO - Shot 74 sharpest frame pruned (sim=0.9827)
2026-07-30 09:56:45,271 - INFO - Shot 76 sharpest frame pruned (sim=0.9977)
2026-07-30 09:56:50,984 - INFO - Shot 80 sharpest frame pruned (sim=0.9827)
2026-07-30 09:56:52,324 - INFO - Shot 81 sharpest frame pruned (sim=0.9887)
2026-07-30 09:56:53,599 - INFO - Shot 82 sharpest frame pruned (sim=0.9870)
2026-07-30 09:56:54,782 - INFO - Shot 83 sharpest frame pruned (sim=0.9995)
2026-07-30 09:56:57,796 - INFO - Shot 86 sharpest frame pruned (sim=0.9964)
2026-07-30 09:56:58,633 - INFO - Shot 87 sharpest frame pruned (sim=0.9944)
2026-07-30 09:56:59,613 - INFO - Shot 88 sharpest frame pruned (sim=0.9939)
2026-07-30 09:57:01,125 - INFO - Shot 90 sharpest frame pruned (sim=0.9969)
2026-07-30 09:57:07,690 - INFO - Extracted 139 keyframe artifacts to 'processed_data/1_frames'.
`torch_dtype` is deprecated! Use `dtype` instead!

Loading weights: 0%| | 0/479 [00:00<?, ?it/s]
Loading weights: 0%| | 1/479 [00:00<00:00, 8456.26it/s, Materializing param=mo
Loading weights: 0%| | 1/479 [00:00<00:00, 2666.44it/s, Materializing param=mo
Loading weights: 0%| | 2/479 [00:00<00:00, 851.46it/s, Materializing param=mod
Loading weights: 0%| | 2/479 [00:00<00:00, 669.86it/s, Materializing param=mod
Loading weights: 1%| | 3/479 [00:00<00:00, 813.27it/s, Materializing param=mod
Loading weights: 1%| | 3/479 [00:00<00:00, 716.24it/s, Materializing param=mod
Loading weights: 1%| | 4/479 [00:00<00:00, 766.75it/s, Materializing param=mod
Loading weights: 1%| | 4/479 [00:00<00:00, 691.39it/s, Materializing param=mod
Loading weights: 1%| | 5/479 [00:00<00:00, 622.34it/s, Materializing param=mod
Loading weights: 1%| | 5/479 [00:00<00:00, 584.88it/s, Materializing param=mod
Loading weights: 1%| | 6/479 [00:00<00:00, 665.64it/s, Materializing param=mod
Loading weights: 1%| | 6/479 [00:00<00:00, 596.70it/s, Materializing param=mod
Loading weights: 1%| | 7/479 [00:00<00:00, 651.39it/s, Materializing param=mod
Loading weights: 1%| | 7/479 [00:00<00:00, 630.34it/s, Materializing param=mod
Loading weights: 2%| | 8/479 [00:00<00:00, 650.63it/s, Materializing param=mod
Loading weights: 2%| | 8/479 [00:00<00:00, 621.57it/s, Materializing param=mod
Loading weights: 2%| | 9/479 [00:00<00:00, 653.86it/s, Materializing param=mod
Loading weights: 2%| | 9/479 [00:00<00:00, 621.61it/s, Materializing param=mod
Loading weights: 2%| | 10/479 [00:00<00:00, 635.31it/s, Materializing param=mo
Loading weights: 2%| | 10/479 [00:00<00:00, 607.47it/s, Materializing param=mo
Loading weights: 2%| | 11/479 [00:00<00:00, 632.74it/s, Materializing param=mo
Loading weights: 2%| | 11/479 [00:00<00:00, 591.82it/s, Materializing param=mo
Loading weights: 3%| | 12/479 [00:00<00:00, 613.02it/s, Materializing param=mo
Loading weights: 3%| | 12/479 [00:00<00:00, 589.05it/s, Materializing param=mo
Loading weights: 3%| | 13/479 [00:00<00:00, 608.79it/s, Materializing param=mo
Loading weights: 3%| | 13/479 [00:00<00:00, 575.32it/s, Materializing param=mo
Loading weights: 3%| | 14/479 [00:00<00:00, 591.62it/s, Materializing param=mo
Loading weights: 3%| | 14/479 [00:00<00:00, 572.12it/s, Materializing param=mo
Loading weights: 3%| | 15/479 [00:00<00:00, 596.13it/s, Materializing param=mo
Loading weights: 3%| | 15/479 [00:00<00:00, 583.35it/s, Materializing param=mo
Loading weights: 3%| | 16/479 [00:00<00:00, 608.43it/s, Materializing param=mo
Loading weights: 3%| | 16/479 [00:00<00:00, 597.60it/s, Materializing param=mo
Loading weights: 4%| | 17/479 [00:00<00:00, 590.10it/s, Materializing param=mo
Loading weights: 4%| | 17/479 [00:00<00:00, 533.17it/s, Materializing param=mo
Loading weights: 4%| | 18/479 [00:00<00:00, 556.93it/s, Materializing param=mo
Loading weights: 4%| | 18/479 [00:00<00:00, 551.25it/s, Materializing param=mo
Loading weights: 4%| | 19/479 [00:00<00:00, 562.74it/s, Materializing param=mo
Loading weights: 4%| | 19/479 [00:00<00:00, 555.92it/s, Materializing param=mo
Loading weights: 4%| | 20/479 [00:00<00:00, 572.14it/s, Materializing param=mo
Loading weights: 4%| | 20/479 [00:00<00:00, 542.23it/s, Materializing param=mo
Loading weights: 4%| | 21/479 [00:00<00:00, 549.40it/s, Materializing param=mo
Loading weights: 4%| | 21/479 [00:00<00:00, 544.69it/s, Materializing param=mo
Loading weights: 5%| | 22/479 [00:00<00:00, 563.48it/s, Materializing param=mo
Loading weights: 5%| | 22/479 [00:00<00:00, 552.70it/s, Materializing param=mo
Loading weights: 5%| | 23/479 [00:00<00:00, 570.78it/s, Materializing param=mo
Loading weights: 5%| | 23/479 [00:00<00:00, 559.90it/s, Materializing param=mo
Loading weights: 5%| | 24/479 [00:00<00:00, 576.38it/s, Materializing param=mo
Loading weights: 5%| | 24/479 [00:00<00:00, 573.07it/s, Materializing param=mo
Loading weights: 5%| | 25/479 [00:00<00:00, 587.79it/s, Materializing param=mo
Loading weights: 5%| | 25/479 [00:00<00:00, 580.68it/s, Materializing param=mo
Loading weights: 5%| | 26/479 [00:00<00:00, 595.22it/s, Materializing param=mo
Loading weights: 5%| | 26/479 [00:00<00:00, 579.41it/s, Materializing param=mo
Loading weights: 6%| | 27/479 [00:00<00:00, 595.08it/s, Materializing param=mo
Loading weights: 6%| | 27/479 [00:00<00:00, 577.97it/s, Materializing param=mo
Loading weights: 6%| | 28/479 [00:00<00:00, 589.88it/s, Materializing param=mo
Loading weights: 6%| | 28/479 [00:00<00:00, 584.96it/s, Materializing param=mo
Loading weights: 6%| | 29/479 [00:00<00:00, 590.13it/s, Materializing param=mo
Loading weights: 6%| | 29/479 [00:00<00:00, 573.71it/s, Materializing param=mo
Loading weights: 6%| | 30/479 [00:00<00:00, 583.50it/s, Materializing param=mo
Loading weights: 6%| | 30/479 [00:00<00:00, 574.73it/s, Materializing param=mo
Loading weights: 6%| | 31/479 [00:00<00:00, 584.31it/s, Materializing param=mo
Loading weights: 6%| | 31/479 [00:00<00:00, 578.30it/s, Materializing param=mo
Loading weights: 7%| | 32/479 [00:00<00:00, 582.69it/s, Materializing param=mo
Loading weights: 7%| | 32/479 [00:00<00:00, 572.22it/s, Materializing param=mo
Loading weights: 7%| | 33/479 [00:00<00:00, 578.66it/s, Materializing param=mo
Loading weights: 7%| | 33/479 [00:00<00:00, 567.94it/s, Materializing param=mo
Loading weights: 7%| | 34/479 [00:00<00:00, 573.00it/s, Materializing param=mo
Loading weights: 7%| | 34/479 [00:00<00:00, 563.90it/s, Materializing param=mo
Loading weights: 7%| | 35/479 [00:00<00:00, 570.33it/s, Materializing param=mo
Loading weights: 7%| | 35/479 [00:00<00:00, 561.04it/s, Materializing param=mo
Loading weights: 8%| | 36/479 [00:00<00:00, 567.84it/s, Materializing param=mo
Loading weights: 8%| | 36/479 [00:00<00:00, 560.05it/s, Materializing param=mo
Loading weights: 8%| | 37/479 [00:00<00:00, 567.68it/s, Materializing param=mo
Loading weights: 8%| | 37/479 [00:00<00:00, 561.73it/s, Materializing param=mo
Loading weights: 8%| | 38/479 [00:00<00:00, 571.04it/s, Materializing param=mo
Loading weights: 8%| | 38/479 [00:00<00:00, 569.05it/s, Materializing param=mo
Loading weights: 8%| | 39/479 [00:00<00:00, 580.66it/s, Materializing param=mo
Loading weights: 8%| | 39/479 [00:00<00:00, 571.71it/s, Materializing param=mo
Loading weights: 8%| | 40/479 [00:00<00:00, 577.25it/s, Materializing param=mo
Loading weights: 8%| | 40/479 [00:00<00:00, 571.92it/s, Materializing param=mo
Loading weights: 9%| | 41/479 [00:00<00:00, 578.57it/s, Materializing param=mo
Loading weights: 9%| | 41/479 [00:00<00:00, 573.58it/s, Materializing param=mo
Loading weights: 9%| | 42/479 [00:00<00:00, 557.34it/s, Materializing param=mo
Loading weights: 9%| | 42/479 [00:00<00:00, 550.95it/s, Materializing param=mo
Loading weights: 9%| | 43/479 [00:00<00:00, 556.96it/s, Materializing param=mo
Loading weights: 9%| | 43/479 [00:00<00:00, 555.05it/s, Materializing param=mo
Loading weights: 9%| | 44/479 [00:00<00:00, 559.50it/s, Materializing param=mo
Loading weights: 9%| | 44/479 [00:00<00:00, 556.05it/s, Materializing param=mo
Loading weights: 9%| | 45/479 [00:00<00:00, 562.48it/s, Materializing param=mo
Loading weights: 9%| | 45/479 [00:00<00:00, 557.59it/s, Materializing param=mo
Loading weights: 10%| | 46/479 [00:00<00:00, 560.68it/s, Materializing param=mo
Loading weights: 10%| | 46/479 [00:00<00:00, 558.90it/s, Materializing param=mo
Loading weights: 10%| | 47/479 [00:00<00:00, 562.31it/s, Materializing param=mo
Loading weights: 10%| | 47/479 [00:00<00:00, 560.49it/s, Materializing param=mo
Loading weights: 10%| | 48/479 [00:00<00:00, 567.12it/s, Materializing param=mo
Loading weights: 10%| | 48/479 [00:00<00:00, 560.48it/s, Materializing param=mo
Loading weights: 10%| | 49/479 [00:00<00:00, 562.82it/s, Materializing param=mo
Loading weights: 10%| | 49/479 [00:00<00:00, 558.76it/s, Materializing param=mo
Loading weights: 10%| | 50/479 [00:00<00:00, 566.33it/s, Materializing param=mo
Loading weights: 10%| | 50/479 [00:00<00:00, 564.25it/s, Materializing param=mo
Loading weights: 11%| | 51/479 [00:00<00:00, 569.28it/s, Materializing param=mo
Loading weights: 11%| | 51/479 [00:00<00:00, 563.30it/s, Materializing param=mo
Loading weights: 11%| | 52/479 [00:00<00:00, 567.68it/s, Materializing param=mo
Loading weights: 11%| | 52/479 [00:00<00:00, 564.23it/s, Materializing param=mo
Loading weights: 11%| | 53/479 [00:00<00:00, 569.74it/s, Materializing param=mo
Loading weights: 11%| | 53/479 [00:00<00:00, 566.00it/s, Materializing param=mo
Loading weights: 11%| | 54/479 [00:00<00:00, 567.83it/s, Materializing param=mo
Loading weights: 11%| | 54/479 [00:00<00:00, 565.35it/s, Materializing param=mo
Loading weights: 11%| | 55/479 [00:00<00:00, 572.76it/s, Materializing param=mo
Loading weights: 11%| | 55/479 [00:00<00:00, 571.23it/s, Materializing param=mo
Loading weights: 12%| | 56/479 [00:00<00:00, 579.10it/s, Materializing param=mo
Loading weights: 12%| | 56/479 [00:00<00:00, 572.35it/s, Materializing param=mo
Loading weights: 12%| | 57/479 [00:00<00:00, 578.13it/s, Materializing param=mo
Loading weights: 12%| | 57/479 [00:00<00:00, 566.75it/s, Materializing param=mo
Loading weights: 12%| | 58/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 12%| | 58/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 12%| | 58/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 12%| | 59/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 12%| | 59/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 60/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 60/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 61/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 61/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 62/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 62/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 63/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 63/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 64/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 13%|▏| 64/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 65/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 65/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 66/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 66/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 67/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 67/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 68/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 68/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 69/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 14%|▏| 69/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 70/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 70/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 71/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 71/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 72/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 72/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 73/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 73/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 74/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 15%|▏| 74/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 75/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 75/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 76/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 76/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 77/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 77/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 78/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 78/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 79/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 16%|▏| 79/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 17%|▏| 80/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 17%|▏| 80/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 17%|▏| 81/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 17%|▏| 81/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 17%|▏| 82/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 17%|▏| 82/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 17%|▏| 83/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 17%|▏| 83/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 84/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 84/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 85/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 85/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 86/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 86/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 87/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 87/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 88/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 18%|▏| 88/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 89/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 89/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 90/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 90/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 91/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 91/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 92/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 92/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 93/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 19%|▏| 93/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 94/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 94/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 95/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 95/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 96/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 96/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 97/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 97/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 98/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 20%|▏| 98/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 21%|▏| 99/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 21%|▏| 99/479 [00:00<00:00, 573.54it/s, Materializing param=mo
Loading weights: 21%|▏| 100/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 21%|▏| 100/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 21%|▏| 101/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 21%|▏| 101/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 21%|▏| 102/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 21%|▏| 102/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 103/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 103/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 104/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 104/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 105/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 105/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 106/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 106/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 107/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 22%|▏| 107/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 108/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 108/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 109/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 109/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 110/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 110/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 111/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 111/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 112/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 23%|▏| 112/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 24%|▏| 113/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 24%|▏| 113/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 24%|▏| 114/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 24%|▏| 114/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 24%|▏| 115/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 24%|▏| 115/479 [00:00<00:00, 573.54it/s, Materializing param=m
Loading weights: 24%|▏| 116/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 24%|▏| 116/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 24%|▏| 116/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 24%|▏| 117/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 24%|▏| 117/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▏| 118/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▏| 118/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▏| 119/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▏| 119/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▎| 120/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▎| 120/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▎| 121/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▎| 121/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▎| 122/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 25%|▎| 122/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 26%|▎| 123/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 26%|▎| 123/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 26%|▎| 124/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 26%|▎| 124/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 26%|▎| 125/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 26%|▎| 125/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 26%|▎| 126/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 26%|▎| 126/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 127/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 127/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 128/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 128/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 129/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 129/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 130/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 130/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 131/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 27%|▎| 131/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 132/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 132/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 133/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 133/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 134/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 134/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 135/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 135/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 136/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 28%|▎| 136/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 137/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 137/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 138/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 138/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 139/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 139/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 140/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 140/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 141/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 29%|▎| 141/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 142/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 142/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 143/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 143/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 144/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 144/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 145/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 145/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 146/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 30%|▎| 146/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 31%|▎| 147/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 31%|▎| 147/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 31%|▎| 148/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 31%|▎| 148/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 31%|▎| 149/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 31%|▎| 149/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 31%|▎| 150/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 31%|▎| 150/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 151/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 151/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 152/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 152/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 153/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 153/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 154/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 154/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 155/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 32%|▎| 155/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 156/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 156/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 157/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 157/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 158/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 158/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 159/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 159/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 160/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 33%|▎| 160/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 161/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 161/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 162/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 162/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 163/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 163/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 164/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 164/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 165/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 34%|▎| 165/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 166/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 166/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 167/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 167/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 168/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 168/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 169/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 169/479 [00:00<00:00, 528.28it/s, Materializing param=m
Loading weights: 35%|▎| 170/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 35%|▎| 170/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 35%|▎| 170/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 36%|▎| 171/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 36%|▎| 171/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 36%|▎| 172/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 36%|▎| 172/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 36%|▎| 173/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 36%|▎| 173/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 36%|▎| 174/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 36%|▎| 174/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 175/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 175/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 176/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 176/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 177/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 177/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 178/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 178/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 179/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 37%|▎| 179/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 180/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 180/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 181/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 181/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 182/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 182/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 183/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 183/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 184/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 38%|▍| 184/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 185/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 185/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 186/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 186/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 187/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 187/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 188/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 188/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 189/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 39%|▍| 189/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 40%|▍| 190/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 40%|▍| 190/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 40%|▍| 191/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 40%|▍| 191/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 40%|▍| 192/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 40%|▍| 192/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 40%|▍| 193/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 40%|▍| 193/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 194/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 194/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 195/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 195/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 196/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 196/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 197/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 197/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 198/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 41%|▍| 198/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 199/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 199/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 200/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 200/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 201/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 201/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 202/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 202/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 203/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 42%|▍| 203/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 204/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 204/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 205/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 205/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 206/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 206/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 207/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 207/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 208/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 43%|▍| 208/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 209/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 209/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 210/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 210/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 211/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 211/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 212/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 212/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 213/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 44%|▍| 213/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 45%|▍| 214/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 45%|▍| 214/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 45%|▍| 215/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 45%|▍| 215/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 45%|▍| 216/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 45%|▍| 216/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 45%|▍| 217/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 45%|▍| 217/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 218/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 218/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 219/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 219/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 220/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 220/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 221/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 221/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 222/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 46%|▍| 222/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 223/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 223/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 224/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 224/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 225/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 225/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 226/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 226/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 227/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 47%|▍| 227/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 228/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 228/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 229/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 229/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 230/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 230/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 231/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 231/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 232/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 48%|▍| 232/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 233/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 233/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 234/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 234/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 235/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 235/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 236/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 236/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 237/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 49%|▍| 237/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 50%|▍| 238/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 50%|▍| 238/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 50%|▍| 239/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 50%|▍| 239/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 50%|▌| 240/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 50%|▌| 240/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 50%|▌| 241/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 50%|▌| 241/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 242/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 242/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 243/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 243/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 244/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 244/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 245/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 245/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 246/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 51%|▌| 246/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 247/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 247/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 248/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 248/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 249/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 249/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 250/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 250/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 251/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 52%|▌| 251/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 252/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 252/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 253/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 253/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 254/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 254/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 255/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 255/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 256/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 53%|▌| 256/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 257/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 257/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 258/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 258/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 259/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 259/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 260/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 260/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 261/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 54%|▌| 261/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 55%|▌| 262/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 55%|▌| 262/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 55%|▌| 263/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 55%|▌| 263/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 55%|▌| 264/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 55%|▌| 264/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 55%|▌| 265/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 55%|▌| 265/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 266/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 266/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 267/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 267/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 268/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 268/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 269/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 269/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 270/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 56%|▌| 270/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 271/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 271/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 272/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 272/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 273/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 273/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 274/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 274/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 275/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 57%|▌| 275/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 276/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 276/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 277/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 277/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 278/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 278/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 279/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 279/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 280/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 58%|▌| 280/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 281/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 281/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 282/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 282/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 283/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 283/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 284/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 284/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 285/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 59%|▌| 285/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 60%|▌| 286/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 60%|▌| 286/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 60%|▌| 287/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 60%|▌| 287/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 60%|▌| 288/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 60%|▌| 288/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 60%|▌| 289/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 60%|▌| 289/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 290/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 290/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 291/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 291/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 292/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 292/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 293/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 293/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 294/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 61%|▌| 294/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 295/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 295/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 296/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 296/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 297/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 297/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 298/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 298/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 299/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 62%|▌| 299/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 300/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 300/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 301/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 301/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 302/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 302/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 303/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 303/479 [00:00<00:00, 453.91it/s, Materializing param=m
Loading weights: 63%|▋| 304/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 63%|▋| 304/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 63%|▋| 304/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 64%|▋| 305/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 64%|▋| 305/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 64%|▋| 306/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 64%|▋| 306/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 64%|▋| 307/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 64%|▋| 307/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 64%|▋| 308/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 64%|▋| 308/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 309/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 309/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 310/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 310/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 311/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 311/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 312/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 312/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 313/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 65%|▋| 313/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 314/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 314/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 315/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 315/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 316/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 316/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 317/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 317/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 318/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 66%|▋| 318/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 319/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 319/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 320/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 320/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 321/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 321/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 322/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 322/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 323/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 67%|▋| 323/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 324/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 324/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 325/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 325/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 326/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 326/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 327/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 327/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 328/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 68%|▋| 328/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 69%|▋| 329/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 69%|▋| 329/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 69%|▋| 330/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 69%|▋| 330/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 69%|▋| 331/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 69%|▋| 331/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 69%|▋| 332/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 69%|▋| 332/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 333/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 333/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 334/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 334/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 335/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 335/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 336/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 336/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 337/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 70%|▋| 337/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 338/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 338/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 339/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 339/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 340/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 340/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 341/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 341/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 342/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 71%|▋| 342/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 343/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 343/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 344/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 344/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 345/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 345/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 346/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 346/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 347/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 72%|▋| 347/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 348/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 348/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 349/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 349/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 350/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 350/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 351/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 351/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 352/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 73%|▋| 352/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 74%|▋| 353/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 74%|▋| 353/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 74%|▋| 354/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 74%|▋| 354/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 74%|▋| 355/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 74%|▋| 355/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 74%|▋| 356/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 74%|▋| 356/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▋| 357/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▋| 357/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▋| 358/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▋| 358/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▋| 359/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▋| 359/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▊| 360/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▊| 360/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▊| 361/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 75%|▊| 361/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 362/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 362/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 363/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 363/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 364/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 364/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 365/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 365/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 366/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 76%|▊| 366/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 367/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 367/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 368/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 368/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 369/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 369/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 370/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 370/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 371/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 77%|▊| 371/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 372/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 372/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 373/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 373/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 374/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 374/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 375/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 375/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 376/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 78%|▊| 376/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 79%|▊| 377/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 79%|▊| 377/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 79%|▊| 378/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 79%|▊| 378/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 79%|▊| 379/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 79%|▊| 379/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 79%|▊| 380/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 79%|▊| 380/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 381/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 381/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 382/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 382/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 383/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 383/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 384/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 384/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 385/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 80%|▊| 385/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 386/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 386/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 387/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 387/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 388/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 388/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 389/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 389/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 390/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 81%|▊| 390/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 391/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 391/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 392/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 392/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 393/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 393/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 394/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 394/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 395/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 82%|▊| 395/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 83%|▊| 396/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 83%|▊| 396/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 83%|▊| 397/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 83%|▊| 397/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 83%|▊| 398/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 83%|▊| 398/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 83%|▊| 399/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 83%|▊| 399/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 400/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 400/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 401/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 401/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 402/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 402/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 403/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 403/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 404/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 84%|▊| 404/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 405/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 405/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 406/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 406/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 407/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 407/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 408/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 408/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 409/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 85%|▊| 409/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 410/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 410/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 411/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 411/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 412/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 412/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 413/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 413/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 414/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 86%|▊| 414/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 415/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 415/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 416/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 416/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 417/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 417/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 418/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 418/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 419/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 87%|▊| 419/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 88%|▉| 420/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 88%|▉| 420/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 88%|▉| 421/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 88%|▉| 421/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 88%|▉| 422/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 88%|▉| 422/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 88%|▉| 423/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 88%|▉| 423/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 424/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 424/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 425/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 425/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 426/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 426/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 427/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 427/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 428/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 89%|▉| 428/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 429/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 429/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 430/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 430/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 431/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 431/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 432/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 432/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 433/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 90%|▉| 433/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 434/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 434/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 435/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 435/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 436/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 436/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 437/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 437/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 438/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 91%|▉| 438/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 439/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 439/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 440/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 440/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 441/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 441/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 442/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 442/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 443/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 92%|▉| 443/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 93%|▉| 444/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 93%|▉| 444/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 93%|▉| 445/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 93%|▉| 445/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 93%|▉| 446/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 93%|▉| 446/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 93%|▉| 447/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 93%|▉| 447/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 448/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 448/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 449/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 449/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 450/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 450/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 451/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 451/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 452/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 94%|▉| 452/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 453/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 453/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 454/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 454/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 455/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 455/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 456/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 456/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 457/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 95%|▉| 457/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 458/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 458/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 459/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 459/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 460/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 460/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 461/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 461/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 462/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 96%|▉| 462/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 463/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 463/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 464/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 464/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 465/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 465/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 466/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 466/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 467/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 97%|▉| 467/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 98%|▉| 468/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 98%|▉| 468/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 98%|▉| 469/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 98%|▉| 469/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 98%|▉| 470/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 98%|▉| 470/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 98%|▉| 471/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 98%|▉| 471/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 472/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 472/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 473/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 473/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 474/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 474/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 475/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 475/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 476/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 99%|▉| 476/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 100%|▉| 477/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 100%|▉| 477/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 100%|▉| 478/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 100%|▉| 478/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 100%|█| 479/479 [00:00<00:00, 759.71it/s, Materializing param=m
Loading weights: 100%|█| 479/479 [00:00<00:00, 866.19it/s, Materializing param=m
2026-07-30 09:57:09,567 - INFO - PhoWhisper model 'models/phowhisper-small' loaded successfully (dtype=torch.float16).
2026-07-30 09:57:09,724 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-07-30 09:57:09,724 - WARNING - Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-07-30 09:57:09,740 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/config.json "HTTP/1.1 200 OK"
2026-07-30 09:57:09,822 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 09:57:09,838 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/config.json "HTTP/1.1 200 OK"
2026-07-30 09:57:09,900 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
2026-07-30 09:57:09,965 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused "HTTP/1.1 200 OK"
2026-07-30 09:57:10,047 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/commits/main "HTTP/1.1 200 OK"

Loading weights: 0%| | 0/447 [00:00<?, ?it/s]
Loading weights: 0%| | 1/447 [00:00<00:00, 2102.41it/s, Materializing param=au
Loading weights: 0%| | 1/447 [00:00<00:00, 692.70it/s, Materializing param=aud
Loading weights: 0%| | 2/447 [00:00<00:00, 759.08it/s, Materializing param=aud
Loading weights: 0%| | 2/447 [00:00<00:00, 488.33it/s, Materializing param=aud
Loading weights: 1%| | 3/447 [00:00<00:00, 577.86it/s, Materializing param=aud
Loading weights: 1%| | 3/447 [00:00<00:01, 361.97it/s, Materializing param=aud
Loading weights: 1%| | 4/447 [00:00<00:01, 436.53it/s, Materializing param=aud
Loading weights: 1%| | 4/447 [00:00<00:01, 418.31it/s, Materializing param=aud
Loading weights: 1%| | 5/447 [00:00<00:01, 422.99it/s, Materializing param=aud
Loading weights: 1%| | 5/447 [00:00<00:01, 388.20it/s, Materializing param=aud
Loading weights: 1%| | 6/447 [00:00<00:01, 437.13it/s, Materializing param=aud
Loading weights: 1%| | 6/447 [00:00<00:01, 409.91it/s, Materializing param=aud
Loading weights: 2%| | 7/447 [00:00<00:01, 413.37it/s, Materializing param=aud
Loading weights: 2%| | 7/447 [00:00<00:01, 407.51it/s, Materializing param=aud
Loading weights: 2%| | 8/447 [00:00<00:00, 456.31it/s, Materializing param=aud
Loading weights: 2%| | 8/447 [00:00<00:00, 450.52it/s, Materializing param=aud
Loading weights: 2%| | 9/447 [00:00<00:00, 497.20it/s, Materializing param=aud
Loading weights: 2%| | 9/447 [00:00<00:00, 490.79it/s, Materializing param=aud
Loading weights: 2%| | 10/447 [00:00<00:00, 535.05it/s, Materializing param=au
Loading weights: 2%| | 10/447 [00:00<00:00, 529.56it/s, Materializing param=au
Loading weights: 2%| | 11/447 [00:00<00:00, 571.04it/s, Materializing param=au
Loading weights: 2%| | 11/447 [00:00<00:00, 564.92it/s, Materializing param=au
Loading weights: 3%| | 12/447 [00:00<00:00, 606.50it/s, Materializing param=au
Loading weights: 3%| | 12/447 [00:00<00:00, 598.94it/s, Materializing param=au
Loading weights: 3%| | 13/447 [00:00<00:00, 636.85it/s, Materializing param=au
Loading weights: 3%| | 13/447 [00:00<00:00, 630.05it/s, Materializing param=au
Loading weights: 3%| | 14/447 [00:00<00:00, 667.75it/s, Materializing param=au
Loading weights: 3%| | 14/447 [00:00<00:00, 661.26it/s, Materializing param=au
Loading weights: 3%| | 15/447 [00:00<00:00, 698.26it/s, Materializing param=au
Loading weights: 3%| | 15/447 [00:00<00:00, 691.53it/s, Materializing param=au
Loading weights: 4%| | 16/447 [00:00<00:00, 725.69it/s, Materializing param=au
Loading weights: 4%| | 16/447 [00:00<00:00, 718.66it/s, Materializing param=au
Loading weights: 4%| | 17/447 [00:00<00:00, 751.67it/s, Materializing param=au
Loading weights: 4%| | 17/447 [00:00<00:00, 744.33it/s, Materializing param=au
Loading weights: 4%| | 18/447 [00:00<00:00, 776.08it/s, Materializing param=au
Loading weights: 4%| | 18/447 [00:00<00:00, 768.19it/s, Materializing param=au
Loading weights: 4%| | 19/447 [00:00<00:00, 798.46it/s, Materializing param=au
Loading weights: 4%| | 19/447 [00:00<00:00, 791.63it/s, Materializing param=au
Loading weights: 4%| | 20/447 [00:00<00:00, 822.49it/s, Materializing param=au
Loading weights: 4%| | 20/447 [00:00<00:00, 814.86it/s, Materializing param=au
Loading weights: 5%| | 21/447 [00:00<00:00, 843.01it/s, Materializing param=au
Loading weights: 5%| | 21/447 [00:00<00:00, 835.55it/s, Materializing param=au
Loading weights: 5%| | 22/447 [00:00<00:00, 863.28it/s, Materializing param=au
Loading weights: 5%| | 22/447 [00:00<00:00, 855.90it/s, Materializing param=au
Loading weights: 5%| | 23/447 [00:00<00:00, 882.69it/s, Materializing param=au
Loading weights: 5%| | 23/447 [00:00<00:00, 874.16it/s, Materializing param=au
Loading weights: 5%| | 24/447 [00:00<00:00, 899.80it/s, Materializing param=au
Loading weights: 5%| | 24/447 [00:00<00:00, 892.78it/s, Materializing param=au
Loading weights: 6%| | 25/447 [00:00<00:00, 917.04it/s, Materializing param=au
Loading weights: 6%| | 25/447 [00:00<00:00, 912.22it/s, Materializing param=au
Loading weights: 6%| | 26/447 [00:00<00:00, 932.90it/s, Materializing param=au
Loading weights: 6%| | 26/447 [00:00<00:00, 924.10it/s, Materializing param=au
Loading weights: 6%| | 27/447 [00:00<00:00, 942.86it/s, Materializing param=au
Loading weights: 6%| | 27/447 [00:00<00:00, 934.41it/s, Materializing param=au
Loading weights: 6%| | 28/447 [00:00<00:00, 955.06it/s, Materializing param=au
Loading weights: 6%| | 28/447 [00:00<00:00, 947.12it/s, Materializing param=au
Loading weights: 6%| | 29/447 [00:00<00:00, 969.42it/s, Materializing param=au
Loading weights: 6%| | 29/447 [00:00<00:00, 962.25it/s, Materializing param=au
Loading weights: 7%| | 30/447 [00:00<00:00, 982.14it/s, Materializing param=au
Loading weights: 7%| | 30/447 [00:00<00:00, 974.54it/s, Materializing param=au
Loading weights: 7%| | 31/447 [00:00<00:00, 993.90it/s, Materializing param=au
Loading weights: 7%| | 31/447 [00:00<00:00, 986.90it/s, Materializing param=au
Loading weights: 7%| | 32/447 [00:00<00:00, 1007.14it/s, Materializing param=a
Loading weights: 7%| | 32/447 [00:00<00:00, 1000.21it/s, Materializing param=a
Loading weights: 7%| | 33/447 [00:00<00:00, 1020.69it/s, Materializing param=a
Loading weights: 7%| | 33/447 [00:00<00:00, 1013.59it/s, Materializing param=a
Loading weights: 8%| | 34/447 [00:00<00:00, 1032.61it/s, Materializing param=a
Loading weights: 8%| | 34/447 [00:00<00:00, 1024.98it/s, Materializing param=a
Loading weights: 8%| | 35/447 [00:00<00:00, 1042.67it/s, Materializing param=a
Loading weights: 8%| | 35/447 [00:00<00:00, 1036.17it/s, Materializing param=a
Loading weights: 8%| | 36/447 [00:00<00:00, 1055.71it/s, Materializing param=a
Loading weights: 8%| | 36/447 [00:00<00:00, 1048.34it/s, Materializing param=a
Loading weights: 8%| | 37/447 [00:00<00:00, 1067.03it/s, Materializing param=a
Loading weights: 8%| | 37/447 [00:00<00:00, 1061.21it/s, Materializing param=a
Loading weights: 9%| | 38/447 [00:00<00:00, 1079.43it/s, Materializing param=a
Loading weights: 9%| | 38/447 [00:00<00:00, 1073.38it/s, Materializing param=a
Loading weights: 9%| | 39/447 [00:00<00:00, 1090.02it/s, Materializing param=a
Loading weights: 9%| | 39/447 [00:00<00:00, 1083.47it/s, Materializing param=a
Loading weights: 9%| | 40/447 [00:00<00:00, 1100.06it/s, Materializing param=a
Loading weights: 9%| | 40/447 [00:00<00:00, 1093.77it/s, Materializing param=a
Loading weights: 9%| | 41/447 [00:00<00:00, 1111.96it/s, Materializing param=a
Loading weights: 9%| | 41/447 [00:00<00:00, 1105.31it/s, Materializing param=a
Loading weights: 9%| | 42/447 [00:00<00:00, 1121.31it/s, Materializing param=a
Loading weights: 9%| | 42/447 [00:00<00:00, 1115.04it/s, Materializing param=a
Loading weights: 10%| | 43/447 [00:00<00:00, 1132.56it/s, Materializing param=a
Loading weights: 10%| | 43/447 [00:00<00:00, 1124.25it/s, Materializing param=a
Loading weights: 10%| | 44/447 [00:00<00:00, 1141.29it/s, Materializing param=a
Loading weights: 10%| | 44/447 [00:00<00:00, 1134.75it/s, Materializing param=a
Loading weights: 10%| | 45/447 [00:00<00:00, 1149.83it/s, Materializing param=a
Loading weights: 10%| | 45/447 [00:00<00:00, 1142.46it/s, Materializing param=a
Loading weights: 10%| | 46/447 [00:00<00:00, 1156.67it/s, Materializing param=a
Loading weights: 10%| | 46/447 [00:00<00:00, 1150.21it/s, Materializing param=a
Loading weights: 11%| | 47/447 [00:00<00:00, 1165.19it/s, Materializing param=a
Loading weights: 11%| | 47/447 [00:00<00:00, 1158.80it/s, Materializing param=a
Loading weights: 11%| | 48/447 [00:00<00:00, 1173.33it/s, Materializing param=a
Loading weights: 11%| | 48/447 [00:00<00:00, 1164.07it/s, Materializing param=a
Loading weights: 11%| | 49/447 [00:00<00:00, 1180.33it/s, Materializing param=a
Loading weights: 11%| | 49/447 [00:00<00:00, 1176.09it/s, Materializing param=a
Loading weights: 11%| | 50/447 [00:00<00:00, 1193.58it/s, Materializing param=a
Loading weights: 11%| | 50/447 [00:00<00:00, 1189.51it/s, Materializing param=a
Loading weights: 11%| | 51/447 [00:00<00:00, 1204.89it/s, Materializing param=a
Loading weights: 11%| | 51/447 [00:00<00:00, 1200.97it/s, Materializing param=a
Loading weights: 12%| | 52/447 [00:00<00:00, 1218.59it/s, Materializing param=a
Loading weights: 12%| | 52/447 [00:00<00:00, 1214.77it/s, Materializing param=a
Loading weights: 12%| | 53/447 [00:00<00:00, 1232.86it/s, Materializing param=a
Loading weights: 12%| | 53/447 [00:00<00:00, 1229.26it/s, Materializing param=a
Loading weights: 12%| | 54/447 [00:00<00:00, 1244.91it/s, Materializing param=a
Loading weights: 12%| | 54/447 [00:00<00:00, 1240.94it/s, Materializing param=a
Loading weights: 12%| | 55/447 [00:00<00:00, 1258.17it/s, Materializing param=a
Loading weights: 12%| | 55/447 [00:00<00:00, 1254.48it/s, Materializing param=a
Loading weights: 13%|▏| 56/447 [00:00<00:00, 1271.80it/s, Materializing param=a
Loading weights: 13%|▏| 56/447 [00:00<00:00, 1266.43it/s, Materializing param=a
Loading weights: 13%|▏| 57/447 [00:00<00:00, 1279.39it/s, Materializing param=a
Loading weights: 13%|▏| 57/447 [00:00<00:00, 1273.53it/s, Materializing param=a
Loading weights: 13%|▏| 58/447 [00:00<00:00, 1286.34it/s, Materializing param=a
Loading weights: 13%|▏| 58/447 [00:00<00:00, 1278.20it/s, Materializing param=a
Loading weights: 13%|▏| 59/447 [00:00<00:00, 1289.98it/s, Materializing param=a
Loading weights: 13%|▏| 59/447 [00:00<00:00, 1283.73it/s, Materializing param=a
Loading weights: 13%|▏| 60/447 [00:00<00:00, 1295.20it/s, Materializing param=a
Loading weights: 13%|▏| 60/447 [00:00<00:00, 1288.70it/s, Materializing param=a
Loading weights: 14%|▏| 61/447 [00:00<00:00, 1301.15it/s, Materializing param=a
Loading weights: 14%|▏| 61/447 [00:00<00:00, 1295.58it/s, Materializing param=a
Loading weights: 14%|▏| 62/447 [00:00<00:00, 1305.81it/s, Materializing param=a
Loading weights: 14%|▏| 62/447 [00:00<00:00, 1300.24it/s, Materializing param=a
Loading weights: 14%|▏| 63/447 [00:00<00:00, 1312.15it/s, Materializing param=a
Loading weights: 14%|▏| 63/447 [00:00<00:00, 1306.00it/s, Materializing param=a
Loading weights: 14%|▏| 64/447 [00:00<00:00, 1317.42it/s, Materializing param=a
Loading weights: 14%|▏| 64/447 [00:00<00:00, 1311.82it/s, Materializing param=a
Loading weights: 15%|▏| 65/447 [00:00<00:00, 1323.39it/s, Materializing param=a
Loading weights: 15%|▏| 65/447 [00:00<00:00, 1316.96it/s, Materializing param=a
Loading weights: 15%|▏| 66/447 [00:00<00:00, 1328.82it/s, Materializing param=a
Loading weights: 15%|▏| 66/447 [00:00<00:00, 1324.93it/s, Materializing param=a
Loading weights: 15%|▏| 67/447 [00:00<00:00, 1339.39it/s, Materializing param=a
Loading weights: 15%|▏| 67/447 [00:00<00:00, 1334.33it/s, Materializing param=a
Loading weights: 15%|▏| 68/447 [00:00<00:00, 1347.75it/s, Materializing param=a
Loading weights: 15%|▏| 68/447 [00:00<00:00, 1343.92it/s, Materializing param=a
Loading weights: 15%|▏| 69/447 [00:00<00:00, 1358.14it/s, Materializing param=a
Loading weights: 15%|▏| 69/447 [00:00<00:00, 1354.49it/s, Materializing param=a
Loading weights: 16%|▏| 70/447 [00:00<00:00, 1368.43it/s, Materializing param=a
Loading weights: 16%|▏| 70/447 [00:00<00:00, 1364.76it/s, Materializing param=a
Loading weights: 16%|▏| 71/447 [00:00<00:00, 1378.92it/s, Materializing param=a
Loading weights: 16%|▏| 71/447 [00:00<00:00, 1375.44it/s, Materializing param=a
Loading weights: 16%|▏| 72/447 [00:00<00:00, 1389.55it/s, Materializing param=a
Loading weights: 16%|▏| 72/447 [00:00<00:00, 1386.05it/s, Materializing param=a
Loading weights: 16%|▏| 73/447 [00:00<00:00, 1398.85it/s, Materializing param=a
Loading weights: 16%|▏| 73/447 [00:00<00:00, 1394.78it/s, Materializing param=a
Loading weights: 17%|▏| 74/447 [00:00<00:00, 1407.98it/s, Materializing param=a
Loading weights: 17%|▏| 74/447 [00:00<00:00, 1404.31it/s, Materializing param=a
Loading weights: 17%|▏| 75/447 [00:00<00:00, 1417.91it/s, Materializing param=a
Loading weights: 17%|▏| 75/447 [00:00<00:00, 1414.41it/s, Materializing param=a
Loading weights: 17%|▏| 76/447 [00:00<00:00, 1427.26it/s, Materializing param=a
Loading weights: 17%|▏| 76/447 [00:00<00:00, 1423.65it/s, Materializing param=a
Loading weights: 17%|▏| 77/447 [00:00<00:00, 1436.86it/s, Materializing param=a
Loading weights: 17%|▏| 77/447 [00:00<00:00, 1433.33it/s, Materializing param=a
Loading weights: 17%|▏| 78/447 [00:00<00:00, 1446.73it/s, Materializing param=a
Loading weights: 17%|▏| 78/447 [00:00<00:00, 1443.20it/s, Materializing param=a
Loading weights: 18%|▏| 79/447 [00:00<00:00, 1454.49it/s, Materializing param=a
Loading weights: 18%|▏| 79/447 [00:00<00:00, 1450.76it/s, Materializing param=a
Loading weights: 18%|▏| 80/447 [00:00<00:00, 1463.53it/s, Materializing param=a
Loading weights: 18%|▏| 80/447 [00:00<00:00, 1460.01it/s, Materializing param=a
Loading weights: 18%|▏| 81/447 [00:00<00:00, 1473.07it/s, Materializing param=a
Loading weights: 18%|▏| 81/447 [00:00<00:00, 1469.60it/s, Materializing param=a
Loading weights: 18%|▏| 82/447 [00:00<00:00, 1480.54it/s, Materializing param=a
Loading weights: 18%|▏| 82/447 [00:00<00:00, 1476.80it/s, Materializing param=a
Loading weights: 19%|▏| 83/447 [00:00<00:00, 1489.54it/s, Materializing param=a
Loading weights: 19%|▏| 83/447 [00:00<00:00, 1485.96it/s, Materializing param=a
Loading weights: 19%|▏| 84/447 [00:00<00:00, 1498.63it/s, Materializing param=a
Loading weights: 19%|▏| 84/447 [00:00<00:00, 1493.79it/s, Materializing param=a
Loading weights: 19%|▏| 85/447 [00:00<00:00, 1505.65it/s, Materializing param=a
Loading weights: 19%|▏| 85/447 [00:00<00:00, 1501.84it/s, Materializing param=a
Loading weights: 19%|▏| 86/447 [00:00<00:00, 1514.00it/s, Materializing param=a
Loading weights: 19%|▏| 86/447 [00:00<00:00, 1510.47it/s, Materializing param=a
Loading weights: 19%|▏| 87/447 [00:00<00:00, 1522.67it/s, Materializing param=a
Loading weights: 19%|▏| 87/447 [00:00<00:00, 1518.69it/s, Materializing param=a
Loading weights: 20%|▏| 88/447 [00:00<00:00, 1530.60it/s, Materializing param=a
Loading weights: 20%|▏| 88/447 [00:00<00:00, 1527.01it/s, Materializing param=a
Loading weights: 20%|▏| 89/447 [00:00<00:00, 1538.95it/s, Materializing param=a
Loading weights: 20%|▏| 89/447 [00:00<00:00, 1535.45it/s, Materializing param=a
Loading weights: 20%|▏| 90/447 [00:00<00:00, 1546.00it/s, Materializing param=a
Loading weights: 20%|▏| 90/447 [00:00<00:00, 1542.31it/s, Materializing param=a
Loading weights: 20%|▏| 91/447 [00:00<00:00, 1553.58it/s, Materializing param=a
Loading weights: 20%|▏| 91/447 [00:00<00:00, 1549.37it/s, Materializing param=a
Loading weights: 21%|▏| 92/447 [00:00<00:00, 1560.94it/s, Materializing param=a
Loading weights: 21%|▏| 92/447 [00:00<00:00, 1557.46it/s, Materializing param=a
Loading weights: 21%|▏| 93/447 [00:00<00:00, 1567.37it/s, Materializing param=a
Loading weights: 21%|▏| 93/447 [00:00<00:00, 1563.56it/s, Materializing param=a
Loading weights: 21%|▏| 94/447 [00:00<00:00, 1575.00it/s, Materializing param=a
Loading weights: 21%|▏| 94/447 [00:00<00:00, 1571.53it/s, Materializing param=a
Loading weights: 21%|▏| 95/447 [00:00<00:00, 1583.11it/s, Materializing param=a
Loading weights: 21%|▏| 95/447 [00:00<00:00, 1579.69it/s, Materializing param=a
Loading weights: 21%|▏| 96/447 [00:00<00:00, 1590.80it/s, Materializing param=a
Loading weights: 21%|▏| 96/447 [00:00<00:00, 1587.35it/s, Materializing param=a
Loading weights: 22%|▏| 97/447 [00:00<00:00, 1598.78it/s, Materializing param=a
Loading weights: 22%|▏| 97/447 [00:00<00:00, 1595.44it/s, Materializing param=a
Loading weights: 22%|▏| 98/447 [00:00<00:00, 1606.86it/s, Materializing param=a
Loading weights: 22%|▏| 98/447 [00:00<00:00, 1603.50it/s, Materializing param=a
Loading weights: 22%|▏| 99/447 [00:00<00:00, 1614.34it/s, Materializing param=a
Loading weights: 22%|▏| 99/447 [00:00<00:00, 1610.93it/s, Materializing param=a
Loading weights: 22%|▏| 100/447 [00:00<00:00, 1622.01it/s, Materializing param=
Loading weights: 22%|▏| 100/447 [00:00<00:00, 1618.68it/s, Materializing param=
Loading weights: 23%|▏| 101/447 [00:00<00:00, 1629.81it/s, Materializing param=
Loading weights: 23%|▏| 101/447 [00:00<00:00, 1626.46it/s, Materializing param=
Loading weights: 23%|▏| 102/447 [00:00<00:00, 1635.46it/s, Materializing param=
Loading weights: 23%|▏| 102/447 [00:00<00:00, 1631.73it/s, Materializing param=
Loading weights: 23%|▏| 103/447 [00:00<00:00, 1642.33it/s, Materializing param=
Loading weights: 23%|▏| 103/447 [00:00<00:00, 1638.85it/s, Materializing param=
Loading weights: 23%|▏| 104/447 [00:00<00:00, 1649.61it/s, Materializing param=
Loading weights: 23%|▏| 104/447 [00:00<00:00, 1644.73it/s, Materializing param=
Loading weights: 23%|▏| 105/447 [00:00<00:00, 1654.72it/s, Materializing param=
Loading weights: 23%|▏| 105/447 [00:00<00:00, 1651.05it/s, Materializing param=
Loading weights: 24%|▏| 106/447 [00:00<00:00, 1661.52it/s, Materializing param=
Loading weights: 24%|▏| 106/447 [00:00<00:00, 1658.05it/s, Materializing param=
Loading weights: 24%|▏| 107/447 [00:00<00:00, 1668.60it/s, Materializing param=
Loading weights: 24%|▏| 107/447 [00:00<00:00, 1663.51it/s, Materializing param=
Loading weights: 24%|▏| 108/447 [00:00<00:00, 1673.10it/s, Materializing param=
Loading weights: 24%|▏| 108/447 [00:00<00:00, 1669.49it/s, Materializing param=
Loading weights: 24%|▏| 109/447 [00:00<00:00, 1679.73it/s, Materializing param=
Loading weights: 24%|▏| 109/447 [00:00<00:00, 1676.28it/s, Materializing param=
Loading weights: 25%|▏| 110/447 [00:00<00:00, 1686.36it/s, Materializing param=
Loading weights: 25%|▏| 110/447 [00:00<00:00, 1683.04it/s, Materializing param=
Loading weights: 25%|▏| 111/447 [00:00<00:00, 1693.07it/s, Materializing param=
Loading weights: 25%|▏| 111/447 [00:00<00:00, 1689.65it/s, Materializing param=
Loading weights: 25%|▎| 112/447 [00:00<00:00, 1699.70it/s, Materializing param=
Loading weights: 25%|▎| 112/447 [00:00<00:00, 1696.36it/s, Materializing param=
Loading weights: 25%|▎| 113/447 [00:00<00:00, 1705.98it/s, Materializing param=
Loading weights: 25%|▎| 113/447 [00:00<00:00, 1702.67it/s, Materializing param=
Loading weights: 26%|▎| 114/447 [00:00<00:00, 1712.57it/s, Materializing param=
Loading weights: 26%|▎| 114/447 [00:00<00:00, 1709.29it/s, Materializing param=
Loading weights: 26%|▎| 115/447 [00:00<00:00, 1719.41it/s, Materializing param=
Loading weights: 26%|▎| 115/447 [00:00<00:00, 1716.15it/s, Materializing param=
Loading weights: 26%|▎| 116/447 [00:00<00:00, 1725.88it/s, Materializing param=
Loading weights: 26%|▎| 116/447 [00:00<00:00, 1722.58it/s, Materializing param=
Loading weights: 26%|▎| 117/447 [00:00<00:00, 1732.38it/s, Materializing param=
Loading weights: 26%|▎| 117/447 [00:00<00:00, 1729.01it/s, Materializing param=
Loading weights: 26%|▎| 118/447 [00:00<00:00, 1738.82it/s, Materializing param=
Loading weights: 26%|▎| 118/447 [00:00<00:00, 1735.52it/s, Materializing param=
Loading weights: 27%|▎| 119/447 [00:00<00:00, 1744.98it/s, Materializing param=
Loading weights: 27%|▎| 119/447 [00:00<00:00, 1741.67it/s, Materializing param=
Loading weights: 27%|▎| 120/447 [00:00<00:00, 1751.20it/s, Materializing param=
Loading weights: 27%|▎| 120/447 [00:00<00:00, 1747.85it/s, Materializing param=
Loading weights: 27%|▎| 121/447 [00:00<00:00, 1757.41it/s, Materializing param=
Loading weights: 27%|▎| 121/447 [00:00<00:00, 1754.04it/s, Materializing param=
Loading weights: 27%|▎| 122/447 [00:00<00:00, 1762.27it/s, Materializing param=
Loading weights: 27%|▎| 122/447 [00:00<00:00, 1758.84it/s, Materializing param=
Loading weights: 28%|▎| 123/447 [00:00<00:00, 1767.70it/s, Materializing param=
Loading weights: 28%|▎| 123/447 [00:00<00:00, 1764.36it/s, Materializing param=
Loading weights: 28%|▎| 124/447 [00:00<00:00, 1773.62it/s, Materializing param=
Loading weights: 28%|▎| 124/447 [00:00<00:00, 1770.29it/s, Materializing param=
Loading weights: 28%|▎| 125/447 [00:00<00:00, 1779.16it/s, Materializing param=
Loading weights: 28%|▎| 125/447 [00:00<00:00, 1775.80it/s, Materializing param=
Loading weights: 28%|▎| 126/447 [00:00<00:00, 1784.47it/s, Materializing param=
Loading weights: 28%|▎| 126/447 [00:00<00:00, 1780.86it/s, Materializing param=
Loading weights: 28%|▎| 127/447 [00:00<00:00, 1788.08it/s, Materializing param=
Loading weights: 28%|▎| 127/447 [00:00<00:00, 1781.66it/s, Materializing param=
Loading weights: 29%|▎| 128/447 [00:00<00:00, 1786.60it/s, Materializing param=
2026-07-30 09:57:10,147 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/discussions?p=0 "HTTP/1.1 200 OK"
Loading weights: 29%|▎| 128/447 [00:00<00:00, 1757.75it/s, Materializing param=
Loading weights: 29%|▎| 129/447 [00:00<00:00, 1751.02it/s, Materializing param=
Loading weights: 29%|▎| 129/447 [00:00<00:00, 1715.73it/s, Materializing param=
Loading weights: 29%|▎| 130/447 [00:00<00:00, 1721.45it/s, Materializing param=
Loading weights: 29%|▎| 130/447 [00:00<00:00, 1717.71it/s, Materializing param=
Loading weights: 29%|▎| 131/447 [00:00<00:00, 1725.95it/s, Materializing param=
Loading weights: 29%|▎| 131/447 [00:00<00:00, 1722.82it/s, Materializing param=
Loading weights: 30%|▎| 132/447 [00:00<00:00, 1730.98it/s, Materializing param=
Loading weights: 30%|▎| 132/447 [00:00<00:00, 1727.92it/s, Materializing param=
Loading weights: 30%|▎| 133/447 [00:00<00:00, 1736.17it/s, Materializing param=
Loading weights: 30%|▎| 133/447 [00:00<00:00, 1733.05it/s, Materializing param=
Loading weights: 30%|▎| 134/447 [00:00<00:00, 1741.50it/s, Materializing param=
Loading weights: 30%|▎| 134/447 [00:00<00:00, 1738.49it/s, Materializing param=
Loading weights: 30%|▎| 135/447 [00:00<00:00, 1745.20it/s, Materializing param=
Loading weights: 30%|▎| 135/447 [00:00<00:00, 1741.34it/s, Materializing param=
Loading weights: 30%|▎| 136/447 [00:00<00:00, 1749.13it/s, Materializing param=
Loading weights: 30%|▎| 136/447 [00:00<00:00, 1744.90it/s, Materializing param=
Loading weights: 31%|▎| 137/447 [00:00<00:00, 1748.95it/s, Materializing param=
Loading weights: 31%|▎| 137/447 [00:00<00:00, 1743.03it/s, Materializing param=
Loading weights: 31%|▎| 138/447 [00:00<00:00, 1746.76it/s, Materializing param=
Loading weights: 31%|▎| 138/447 [00:00<00:00, 1740.11it/s, Materializing param=
Loading weights: 31%|▎| 139/447 [00:00<00:00, 1744.47it/s, Materializing param=
Loading weights: 31%|▎| 139/447 [00:00<00:00, 1738.84it/s, Materializing param=
Loading weights: 31%|▎| 140/447 [00:00<00:00, 1742.45it/s, Materializing param=
Loading weights: 31%|▎| 140/447 [00:00<00:00, 1737.96it/s, Materializing param=
Loading weights: 32%|▎| 141/447 [00:00<00:00, 1744.26it/s, Materializing param=
Loading weights: 32%|▎| 141/447 [00:00<00:00, 1740.96it/s, Materializing param=
Loading weights: 32%|▎| 142/447 [00:00<00:00, 1746.80it/s, Materializing param=
Loading weights: 32%|▎| 142/447 [00:00<00:00, 1742.30it/s, Materializing param=
Loading weights: 32%|▎| 143/447 [00:00<00:00, 1746.84it/s, Materializing param=
Loading weights: 32%|▎| 143/447 [00:00<00:00, 1741.83it/s, Materializing param=
Loading weights: 32%|▎| 144/447 [00:00<00:00, 1745.20it/s, Materializing param=
Loading weights: 32%|▎| 144/447 [00:00<00:00, 1740.38it/s, Materializing param=
Loading weights: 32%|▎| 145/447 [00:00<00:00, 1745.12it/s, Materializing param=
Loading weights: 32%|▎| 145/447 [00:00<00:00, 1739.08it/s, Materializing param=
Loading weights: 33%|▎| 146/447 [00:00<00:00, 1744.19it/s, Materializing param=
Loading weights: 33%|▎| 146/447 [00:00<00:00, 1739.88it/s, Materializing param=
Loading weights: 33%|▎| 147/447 [00:00<00:00, 1744.21it/s, Materializing param=
Loading weights: 33%|▎| 147/447 [00:00<00:00, 1739.70it/s, Materializing param=
Loading weights: 33%|▎| 148/447 [00:00<00:00, 1744.62it/s, Materializing param=
Loading weights: 33%|▎| 148/447 [00:00<00:00, 1740.35it/s, Materializing param=
Loading weights: 33%|▎| 149/447 [00:00<00:00, 1744.99it/s, Materializing param=
Loading weights: 33%|▎| 149/447 [00:00<00:00, 1740.61it/s, Materializing param=
Loading weights: 34%|▎| 150/447 [00:00<00:00, 1744.96it/s, Materializing param=
Loading weights: 34%|▎| 150/447 [00:00<00:00, 1740.27it/s, Materializing param=
Loading weights: 34%|▎| 151/447 [00:00<00:00, 1744.76it/s, Materializing param=
Loading weights: 34%|▎| 151/447 [00:00<00:00, 1740.32it/s, Materializing param=
Loading weights: 34%|▎| 152/447 [00:00<00:00, 1745.01it/s, Materializing param=
Loading weights: 34%|▎| 152/447 [00:00<00:00, 1739.60it/s, Materializing param=
Loading weights: 34%|▎| 153/447 [00:00<00:00, 1744.39it/s, Materializing param=
Loading weights: 34%|▎| 153/447 [00:00<00:00, 1740.22it/s, Materializing param=
Loading weights: 34%|▎| 154/447 [00:00<00:00, 1744.45it/s, Materializing param=
Loading weights: 34%|▎| 154/447 [00:00<00:00, 1740.29it/s, Materializing param=
Loading weights: 35%|▎| 155/447 [00:00<00:00, 1744.43it/s, Materializing param=
Loading weights: 35%|▎| 155/447 [00:00<00:00, 1739.82it/s, Materializing param=
Loading weights: 35%|▎| 156/447 [00:00<00:00, 1742.68it/s, Materializing param=
Loading weights: 35%|▎| 156/447 [00:00<00:00, 1738.17it/s, Materializing param=
Loading weights: 35%|▎| 157/447 [00:00<00:00, 1742.55it/s, Materializing param=
Loading weights: 35%|▎| 157/447 [00:00<00:00, 1737.07it/s, Materializing param=
Loading weights: 35%|▎| 158/447 [00:00<00:00, 1740.21it/s, Materializing param=
Loading weights: 35%|▎| 158/447 [00:00<00:00, 1734.34it/s, Materializing param=
Loading weights: 36%|▎| 159/447 [00:00<00:00, 1737.20it/s, Materializing param=
Loading weights: 36%|▎| 159/447 [00:00<00:00, 1732.47it/s, Materializing param=
Loading weights: 36%|▎| 160/447 [00:00<00:00, 1736.50it/s, Materializing param=
Loading weights: 36%|▎| 160/447 [00:00<00:00, 1731.74it/s, Materializing param=
Loading weights: 36%|▎| 161/447 [00:00<00:00, 1735.75it/s, Materializing param=
Loading weights: 36%|▎| 161/447 [00:00<00:00, 1731.59it/s, Materializing param=
Loading weights: 36%|▎| 162/447 [00:00<00:00, 1735.52it/s, Materializing param=
Loading weights: 36%|▎| 162/447 [00:00<00:00, 1731.08it/s, Materializing param=
Loading weights: 36%|▎| 163/447 [00:00<00:00, 1735.23it/s, Materializing param=
Loading weights: 36%|▎| 163/447 [00:00<00:00, 1730.83it/s, Materializing param=
Loading weights: 37%|▎| 164/447 [00:00<00:00, 1735.39it/s, Materializing param=
Loading weights: 37%|▎| 164/447 [00:00<00:00, 1731.59it/s, Materializing param=
Loading weights: 37%|▎| 165/447 [00:00<00:00, 1735.96it/s, Materializing param=
Loading weights: 37%|▎| 165/447 [00:00<00:00, 1731.64it/s, Materializing param=
Loading weights: 37%|▎| 166/447 [00:00<00:00, 1735.70it/s, Materializing param=
Loading weights: 37%|▎| 166/447 [00:00<00:00, 1731.89it/s, Materializing param=
Loading weights: 37%|▎| 167/447 [00:00<00:00, 1736.13it/s, Materializing param=
Loading weights: 37%|▎| 167/447 [00:00<00:00, 1732.35it/s, Materializing param=
Loading weights: 38%|▍| 168/447 [00:00<00:00, 1736.61it/s, Materializing param=
Loading weights: 38%|▍| 168/447 [00:00<00:00, 1733.08it/s, Materializing param=
Loading weights: 38%|▍| 169/447 [00:00<00:00, 1737.32it/s, Materializing param=
Loading weights: 38%|▍| 169/447 [00:00<00:00, 1733.68it/s, Materializing param=
Loading weights: 38%|▍| 170/447 [00:00<00:00, 1737.85it/s, Materializing param=
Loading weights: 38%|▍| 170/447 [00:00<00:00, 1734.21it/s, Materializing param=
Loading weights: 38%|▍| 171/447 [00:00<00:00, 1737.98it/s, Materializing param=
Loading weights: 38%|▍| 171/447 [00:00<00:00, 1731.77it/s, Materializing param=
Loading weights: 38%|▍| 172/447 [00:00<00:00, 1736.42it/s, Materializing param=
Loading weights: 38%|▍| 172/447 [00:00<00:00, 1733.00it/s, Materializing param=
Loading weights: 39%|▍| 173/447 [00:00<00:00, 1739.04it/s, Materializing param=
Loading weights: 39%|▍| 173/447 [00:00<00:00, 1736.71it/s, Materializing param=
Loading weights: 39%|▍| 174/447 [00:00<00:00, 1743.16it/s, Materializing param=
Loading weights: 39%|▍| 174/447 [00:00<00:00, 1740.85it/s, Materializing param=
Loading weights: 39%|▍| 175/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 39%|▍| 175/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 39%|▍| 175/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 39%|▍| 176/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 39%|▍| 176/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 177/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 177/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 178/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 178/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 179/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 179/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 180/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 180/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 181/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 40%|▍| 181/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 41%|▍| 182/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 41%|▍| 182/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 41%|▍| 183/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 41%|▍| 183/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 41%|▍| 184/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 41%|▍| 184/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 41%|▍| 185/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 41%|▍| 185/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 42%|▍| 186/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 42%|▍| 186/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 42%|▍| 187/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 42%|▍| 187/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 42%|▍| 188/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 42%|▍| 188/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 42%|▍| 189/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 42%|▍| 189/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 190/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 190/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 191/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 191/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 192/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 192/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 193/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 193/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 194/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 43%|▍| 194/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 44%|▍| 195/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 44%|▍| 195/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 44%|▍| 196/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 44%|▍| 196/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 44%|▍| 197/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 44%|▍| 197/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 44%|▍| 198/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 44%|▍| 198/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 199/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 199/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 200/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 200/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 201/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 201/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 202/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 202/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 203/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 45%|▍| 203/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 46%|▍| 204/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 46%|▍| 204/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 46%|▍| 205/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 46%|▍| 205/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 46%|▍| 206/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 46%|▍| 206/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 46%|▍| 207/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 46%|▍| 207/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 208/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 208/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 209/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 209/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 210/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 210/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 211/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 211/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 212/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 47%|▍| 212/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 48%|▍| 213/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 48%|▍| 213/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 48%|▍| 214/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 48%|▍| 214/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 48%|▍| 215/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 48%|▍| 215/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 48%|▍| 216/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 48%|▍| 216/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 217/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 217/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 218/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 218/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 219/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 219/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 220/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 220/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 221/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 49%|▍| 221/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 50%|▍| 222/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 50%|▍| 222/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 50%|▍| 223/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 50%|▍| 223/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 50%|▌| 224/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 50%|▌| 224/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 50%|▌| 225/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 50%|▌| 225/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 226/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 226/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 227/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 227/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 228/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 228/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 229/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 229/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 230/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 51%|▌| 230/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 52%|▌| 231/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 52%|▌| 231/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 52%|▌| 232/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 52%|▌| 232/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 52%|▌| 233/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 52%|▌| 233/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 52%|▌| 234/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 52%|▌| 234/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 235/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 235/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 236/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 236/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 237/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 237/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 238/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 238/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 239/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 53%|▌| 239/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 54%|▌| 240/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 54%|▌| 240/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 54%|▌| 241/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 54%|▌| 241/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 54%|▌| 242/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 54%|▌| 242/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 54%|▌| 243/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 54%|▌| 243/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 244/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 244/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 245/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 245/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 246/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 246/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 247/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 247/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 248/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 55%|▌| 248/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 56%|▌| 249/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 56%|▌| 249/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 56%|▌| 250/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 56%|▌| 250/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 56%|▌| 251/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 56%|▌| 251/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 56%|▌| 252/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 56%|▌| 252/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 253/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 253/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 254/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 254/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 255/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 255/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 256/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 256/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 257/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 57%|▌| 257/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 58%|▌| 258/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 58%|▌| 258/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 58%|▌| 259/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 58%|▌| 259/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 58%|▌| 260/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 58%|▌| 260/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 58%|▌| 261/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 58%|▌| 261/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 59%|▌| 262/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 59%|▌| 262/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 59%|▌| 263/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 59%|▌| 263/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 59%|▌| 264/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 59%|▌| 264/447 [00:00<00:00, 1747.87it/s, Materializing param=2026-07-30 09:57:10,228 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/commits/refs%2Fpr%2F3 "HTTP/1.1 200 OK"

Loading weights: 59%|▌| 265/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 59%|▌| 265/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 266/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 266/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 267/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 267/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 268/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 268/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 269/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 269/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 270/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 60%|▌| 270/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 61%|▌| 271/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 61%|▌| 271/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 61%|▌| 272/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 61%|▌| 272/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 61%|▌| 273/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 61%|▌| 273/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 61%|▌| 274/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 61%|▌| 274/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 275/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 275/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 276/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 276/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 277/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 277/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 278/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 278/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 279/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 62%|▌| 279/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 63%|▋| 280/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 63%|▋| 280/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 63%|▋| 281/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 63%|▋| 281/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 63%|▋| 282/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 63%|▋| 282/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 63%|▋| 283/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 63%|▋| 283/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 284/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 284/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 285/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 285/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 286/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 286/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 287/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 287/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 288/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 64%|▋| 288/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 65%|▋| 289/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 65%|▋| 289/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 65%|▋| 290/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 65%|▋| 290/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 65%|▋| 291/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 65%|▋| 291/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 65%|▋| 292/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 65%|▋| 292/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 293/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 293/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 294/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 294/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 295/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 295/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 296/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 296/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 297/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 66%|▋| 297/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 67%|▋| 298/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 67%|▋| 298/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 67%|▋| 299/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 67%|▋| 299/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 67%|▋| 300/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 67%|▋| 300/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 67%|▋| 301/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 67%|▋| 301/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 302/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 302/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 303/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 303/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 304/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 304/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 305/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 305/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 306/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 68%|▋| 306/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 69%|▋| 307/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 69%|▋| 307/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 69%|▋| 308/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 69%|▋| 308/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 69%|▋| 309/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 69%|▋| 309/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 69%|▋| 310/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 69%|▋| 310/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 311/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 311/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 312/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 312/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 313/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 313/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 314/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 314/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 315/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 70%|▋| 315/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 71%|▋| 316/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 71%|▋| 316/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 71%|▋| 317/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 71%|▋| 317/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 71%|▋| 318/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 71%|▋| 318/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 71%|▋| 319/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 71%|▋| 319/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 320/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 320/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 321/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 321/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 322/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 322/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 323/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 323/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 324/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 72%|▋| 324/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 73%|▋| 325/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 73%|▋| 325/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 73%|▋| 326/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 73%|▋| 326/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 73%|▋| 327/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 73%|▋| 327/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 73%|▋| 328/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 73%|▋| 328/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 329/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 329/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 330/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 330/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 331/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 331/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 332/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 332/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 333/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 74%|▋| 333/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 75%|▋| 334/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 75%|▋| 334/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 75%|▋| 335/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 75%|▋| 335/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 75%|▊| 336/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 75%|▊| 336/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 75%|▊| 337/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 75%|▊| 337/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 76%|▊| 338/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 76%|▊| 338/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 76%|▊| 339/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 76%|▊| 339/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 76%|▊| 340/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 76%|▊| 340/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 76%|▊| 341/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 76%|▊| 341/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 342/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 342/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 343/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 343/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 344/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 344/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 345/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 345/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 346/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 77%|▊| 346/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 78%|▊| 347/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 78%|▊| 347/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 78%|▊| 348/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 78%|▊| 348/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 78%|▊| 349/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 78%|▊| 349/447 [00:00<00:00, 1747.87it/s, Materializing param=
Loading weights: 78%|▊| 350/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 78%|▊| 350/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 78%|▊| 350/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 351/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 351/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 352/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 352/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 353/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 353/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 354/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 354/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 355/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 79%|▊| 355/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 80%|▊| 356/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 80%|▊| 356/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 80%|▊| 357/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 80%|▊| 357/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 80%|▊| 358/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 80%|▊| 358/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 80%|▊| 359/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 80%|▊| 359/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 360/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 360/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 361/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 361/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 362/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 362/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 363/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 363/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 364/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 81%|▊| 364/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 82%|▊| 365/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 82%|▊| 365/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 82%|▊| 366/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 82%|▊| 366/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 82%|▊| 367/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 82%|▊| 367/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 82%|▊| 368/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 82%|▊| 368/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 369/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 369/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 370/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 370/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 371/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 371/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 372/447 [00:00<00:00, 1706.75it/s, Materializing param=2026-07-30 09:57:10,290 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/refs%2Fpr%2F3/model.safetensors.index.json "HTTP/1.1 404 Not Found"

Loading weights: 83%|▊| 372/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 373/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 83%|▊| 373/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 84%|▊| 374/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 84%|▊| 374/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 84%|▊| 375/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 84%|▊| 375/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 84%|▊| 376/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 84%|▊| 376/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 84%|▊| 377/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 84%|▊| 377/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 378/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 378/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 379/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 379/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 380/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 380/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 381/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 381/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 382/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 85%|▊| 382/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 86%|▊| 383/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 86%|▊| 383/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 86%|▊| 384/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 86%|▊| 384/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 86%|▊| 385/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 86%|▊| 385/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 86%|▊| 386/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 86%|▊| 386/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 387/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 387/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 388/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 388/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 389/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 389/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 390/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 390/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 391/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 87%|▊| 391/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 88%|▉| 392/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 88%|▉| 392/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 88%|▉| 393/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 88%|▉| 393/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 88%|▉| 394/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 88%|▉| 394/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 88%|▉| 395/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 88%|▉| 395/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 396/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 396/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 397/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 397/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 398/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 398/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 399/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 399/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 400/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 89%|▉| 400/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 90%|▉| 401/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 90%|▉| 401/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 90%|▉| 402/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 90%|▉| 402/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 90%|▉| 403/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 90%|▉| 403/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 90%|▉| 404/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 90%|▉| 404/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 405/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 405/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 406/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 406/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 407/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 407/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 408/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 408/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 409/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 91%|▉| 409/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 92%|▉| 410/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 92%|▉| 410/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 92%|▉| 411/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 92%|▉| 411/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 92%|▉| 412/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 92%|▉| 412/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 92%|▉| 413/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 92%|▉| 413/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 93%|▉| 414/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 93%|▉| 414/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 93%|▉| 415/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 93%|▉| 415/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 93%|▉| 416/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 93%|▉| 416/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 93%|▉| 417/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 93%|▉| 417/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 418/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 418/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 419/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 419/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 420/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 420/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 421/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 421/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 422/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 94%|▉| 422/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 95%|▉| 423/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 95%|▉| 423/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 95%|▉| 424/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 95%|▉| 424/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 95%|▉| 425/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 95%|▉| 425/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 95%|▉| 426/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 95%|▉| 426/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 427/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 427/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 428/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 428/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 429/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 429/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 430/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 430/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 431/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 96%|▉| 431/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 97%|▉| 432/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 97%|▉| 432/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 97%|▉| 433/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 97%|▉| 433/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 97%|▉| 434/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 97%|▉| 434/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 97%|▉| 435/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 97%|▉| 435/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 436/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 436/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 437/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 437/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 438/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 438/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 439/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 439/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 440/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 98%|▉| 440/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 99%|▉| 441/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 99%|▉| 441/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 99%|▉| 442/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 99%|▉| 442/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 99%|▉| 443/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 99%|▉| 443/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 99%|▉| 444/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 99%|▉| 444/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 100%|▉| 445/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 100%|▉| 445/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 100%|▉| 446/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 100%|▉| 446/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 100%|█| 447/447 [00:00<00:00, 1706.75it/s, Materializing param=
Loading weights: 100%|█| 447/447 [00:00<00:00, 1716.77it/s, Materializing param=
2026-07-30 09:57:10,409 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/refs%2Fpr%2F3/model.safetensors "HTTP/1.1 302 Found"
2026-07-30 09:57:10,457 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-07-30 09:57:10,519 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-07-30 09:57:10,818 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-07-30 09:57:10,876 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 09:57:10,892 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/preprocessor_config.json "HTTP/1.1 200 OK"
2026-07-30 09:57:10,964 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-07-30 09:57:11,042 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 09:57:11,058 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/preprocessor_config.json "HTTP/1.1 200 OK"
2026-07-30 09:57:11,120 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-07-30 09:57:11,180 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 09:57:11,195 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/preprocessor_config.json "HTTP/1.1 200 OK"
2026-07-30 09:57:11,420 - INFO - CLAP zero-shot model loaded successfully.
Using cache found in /root/.cache/torch/hub/snakers4_silero-vad_master
2026-07-30 09:57:11,618 - INFO - Silero VAD model loaded successfully.
2026-07-30 09:57:17,300 - INFO - Silero VAD detected 61 speech segments.
2026-07-30 09:57:17,316 - WARNING - Transcription failed for audio interval 5378-12222ms: 'num_frames'
2026-07-30 09:57:17,333 - WARNING - Transcription failed for audio interval 12802-16062ms: 'num_frames'
2026-07-30 09:57:17,349 - WARNING - Transcription failed for audio interval 19906-21086ms: 'num_frames'
2026-07-30 09:57:17,362 - WARNING - Transcription failed for audio interval 21218-23678ms: 'num_frames'
2026-07-30 09:57:17,374 - WARNING - Transcription failed for audio interval 24578-29150ms: 'num_frames'
2026-07-30 09:57:17,387 - WARNING - Transcription failed for audio interval 30114-34462ms: 'num_frames'
2026-07-30 09:57:17,398 - WARNING - Transcription failed for audio interval 35170-40798ms: 'num_frames'
2026-07-30 09:57:17,411 - WARNING - Transcription failed for audio interval 43810-48190ms: 'num_frames'
2026-07-30 09:57:17,424 - WARNING - Transcription failed for audio interval 48418-52062ms: 'num_frames'
2026-07-30 09:57:17,437 - WARNING - Transcription failed for audio interval 52674-59614ms: 'num_frames'
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
2026-07-30 09:57:17,450 - WARNING - Transcription failed for audio interval 59810-64861ms: 'num_frames'
2026-07-30 09:57:17,462 - WARNING - Transcription failed for audio interval 66946-74270ms: 'num_frames'
2026-07-30 09:57:17,474 - WARNING - Transcription failed for audio interval 74530-81694ms: 'num_frames'
2026-07-30 09:57:17,486 - WARNING - Transcription failed for audio interval 82082-91038ms: 'num_frames'
2026-07-30 09:57:17,498 - WARNING - Transcription failed for audio interval 91522-92286ms: 'num_frames'
2026-07-30 09:57:17,511 - WARNING - Transcription failed for audio interval 92514-97214ms: 'num_frames'
2026-07-30 09:57:17,524 - WARNING - Transcription failed for audio interval 97314-101246ms: 'num_frames'
2026-07-30 09:57:17,536 - WARNING - Transcription failed for audio interval 101378-102878ms: 'num_frames'
2026-07-30 09:57:17,548 - WARNING - Transcription failed for audio interval 103650-107966ms: 'num_frames'
2026-07-30 09:57:17,560 - WARNING - Transcription failed for audio interval 108258-110558ms: 'num_frames'
2026-07-30 09:57:17,572 - WARNING - Transcription failed for audio interval 110882-112510ms: 'num_frames'
2026-07-30 09:57:17,584 - WARNING - Transcription failed for audio interval 112674-113982ms: 'num_frames'
2026-07-30 09:57:17,597 - WARNING - Transcription failed for audio interval 114242-118110ms: 'num_frames'
2026-07-30 09:57:17,610 - WARNING - Transcription failed for audio interval 118562-127710ms: 'num_frames'
2026-07-30 09:57:17,622 - WARNING - Transcription failed for audio interval 129922-133150ms: 'num_frames'
2026-07-30 09:57:17,635 - WARNING - Transcription failed for audio interval 133474-136222ms: 'num_frames'
2026-07-30 09:57:17,647 - WARNING - Transcription failed for audio interval 136354-142814ms: 'num_frames'
2026-07-30 09:57:17,660 - WARNING - Transcription failed for audio interval 142914-144670ms: 'num_frames'
2026-07-30 09:57:17,672 - WARNING - Transcription failed for audio interval 145474-147454ms: 'num_frames'
2026-07-30 09:57:17,686 - WARNING - Transcription failed for audio interval 147554-153854ms: 'num_frames'
2026-07-30 09:57:17,699 - WARNING - Transcription failed for audio interval 155778-160510ms: 'num_frames'
2026-07-30 09:57:17,711 - WARNING - Transcription failed for audio interval 160610-164318ms: 'num_frames'
2026-07-30 09:57:17,724 - WARNING - Transcription failed for audio interval 164546-170462ms: 'num_frames'
2026-07-30 09:57:17,737 - WARNING - Transcription failed for audio interval 170786-171102ms: 'num_frames'
2026-07-30 09:57:17,749 - WARNING - Transcription failed for audio interval 171554-173438ms: 'num_frames'
2026-07-30 09:57:17,761 - WARNING - Transcription failed for audio interval 173890-176222ms: 'num_frames'
2026-07-30 09:57:17,774 - WARNING - Transcription failed for audio interval 176354-177790ms: 'num_frames'
2026-07-30 09:57:17,786 - WARNING - Transcription failed for audio interval 178050-180638ms: 'num_frames'
2026-07-30 09:57:17,798 - WARNING - Transcription failed for audio interval 181986-183838ms: 'num_frames'
2026-07-30 09:57:17,810 - WARNING - Transcription failed for audio interval 183938-185502ms: 'num_frames'
2026-07-30 09:57:17,822 - WARNING - Transcription failed for audio interval 185602-191358ms: 'num_frames'
2026-07-30 09:57:17,835 - WARNING - Transcription failed for audio interval 191490-200926ms: 'num_frames'
2026-07-30 09:57:17,848 - WARNING - Transcription failed for audio interval 201090-208350ms: 'num_frames'
2026-07-30 09:57:17,861 - WARNING - Transcription failed for audio interval 208738-212030ms: 'num_frames'
2026-07-30 09:57:17,874 - WARNING - Transcription failed for audio interval 212322-216478ms: 'num_frames'
2026-07-30 09:57:17,887 - WARNING - Transcription failed for audio interval 216706-219486ms: 'num_frames'
2026-07-30 09:57:17,899 - WARNING - Transcription failed for audio interval 221282-231774ms: 'num_frames'
2026-07-30 09:57:17,911 - WARNING - Transcription failed for audio interval 233346-236894ms: 'num_frames'
2026-07-30 09:57:17,924 - WARNING - Transcription failed for audio interval 236994-242718ms: 'num_frames'
2026-07-30 09:57:17,936 - WARNING - Transcription failed for audio interval 242978-247550ms: 'num_frames'
2026-07-30 09:57:17,948 - WARNING - Transcription failed for audio interval 247970-251166ms: 'num_frames'
2026-07-30 09:57:17,960 - WARNING - Transcription failed for audio interval 251266-256829ms: 'num_frames'
2026-07-30 09:57:17,972 - WARNING - Transcription failed for audio interval 257218-265150ms: 'num_frames'
2026-07-30 09:57:17,985 - WARNING - Transcription failed for audio interval 265410-269278ms: 'num_frames'
2026-07-30 09:57:17,998 - WARNING - Transcription failed for audio interval 269602-277726ms: 'num_frames'
2026-07-30 09:57:18,010 - WARNING - Transcription failed for audio interval 278882-279198ms: 'num_frames'
2026-07-30 09:57:18,022 - WARNING - Transcription failed for audio interval 280002-281758ms: 'num_frames'
2026-07-30 09:57:18,035 - WARNING - Transcription failed for audio interval 281922-284510ms: 'num_frames'
2026-07-30 09:57:18,047 - WARNING - Transcription failed for audio interval 286786-287134ms: 'num_frames'
2026-07-30 09:57:18,059 - WARNING - Transcription failed for audio interval 290242-297438ms: 'num_frames'
2026-07-30 09:57:18,072 - WARNING - Transcription failed for audio interval 297570-300419ms: 'num_frames'
2026-07-30 09:57:25,694 - INFO - Audio processing completed: 0 speech segments, 53 audio events.
2026-07-30 09:57:25,698 - INFO - [1_news_60s_720p] Extracted 139 keyframes, 0 ASR segments, and 53 audio events.
2026-07-30 09:57:31,273 - INFO - [1_news_60s_720p] Generating window-based LLM context summaries...
2026-07-30 09:57:31,279 - INFO - Summarizing LLM context batch 1/4 [0s - 90s] (40 frames)...
2026-07-30 09:57:31,661 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 09:57:31,676 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 09:57:31,676 - INFO - Summarizing LLM context batch 2/4 [90s - 180s] (47 frames)...
2026-07-30 09:57:32,373 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 09:57:32,374 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 09:57:32,374 - INFO - Summarizing LLM context batch 3/4 [180s - 270s] (36 frames)...
2026-07-30 09:57:32,638 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 09:57:32,639 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 09:57:32,639 - INFO - Summarizing LLM context batch 4/4 [270s - 300s] (16 frames)...
2026-07-30 09:57:32,929 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 09:57:32,930 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 09:57:32,931 - INFO - [1_news_60s_720p] Generating visual and text embeddings...
2026-07-30 09:57:34,488 - INFO - Image embedding batch 1/3 processed (64 images)
2026-07-30 09:57:35,740 - INFO - Image embedding batch 2/3 processed (64 images)
2026-07-30 09:57:35,903 - INFO - Image embedding batch 3/3 processed (11 images)
2026-07-30 09:57:36,830 - INFO - Completed video processing: 1_news_60s_720p in 158.18s | Metadata: processed_data/3_metadata/1_news_60s_720p_metadata.json
2026-07-30 09:57:36,830 - INFO - Processing video: 2_news_60s_720p
Offline Indexing Pipeline: 25%|████▎ | 1/4 [02:38<07:54, 158.18s/it]2026-07-30 09:57:36,831 - INFO - Segmenting video: data/official_videos/dummy_videos/2_news_60s_720p.mp4
2026-07-30 09:57:55,941 - INFO - TransNetV2 detected 63 shots in video '2_news_60s_720p.mp4'.
2026-07-30 09:57:56,892 - INFO - Shot 1 sharpest frame pruned (sim=0.9966)
2026-07-30 09:57:58,156 - INFO - Shot 2 sharpest frame pruned (sim=0.9897)
2026-07-30 09:57:59,673 - INFO - Shot 3 sharpest frame pruned (sim=0.9823)
2026-07-30 09:58:01,429 - INFO - Shot 4 sharpest frame pruned (sim=0.9891)
2026-07-30 09:58:02,871 - INFO - Shot 5 sharpest frame pruned (sim=0.9863)
2026-07-30 09:58:11,546 - INFO - Shot 12 sharpest frame pruned (sim=0.9961)
2026-07-30 09:58:20,111 - INFO - Shot 18 sharpest frame pruned (sim=0.9920)
2026-07-30 09:58:30,142 - INFO - Shot 25 sharpest frame pruned (sim=0.9806)
2026-07-30 09:58:31,202 - INFO - Shot 26 sharpest frame pruned (sim=0.9917)
2026-07-30 09:58:40,365 - INFO - Shot 32 sharpest frame pruned (sim=0.9825)
2026-07-30 09:59:06,077 - INFO - Shot 49 sharpest frame pruned (sim=0.9846)
2026-07-30 09:59:10,775 - INFO - Shot 52 sharpest frame pruned (sim=0.9920)
2026-07-30 09:59:28,732 - INFO - Shot 62 sharpest frame pruned (sim=0.9802)
2026-07-30 09:59:29,778 - INFO - Extracted 113 keyframe artifacts to 'processed_data/1_frames'.
2026-07-30 09:59:35,286 - INFO - Silero VAD detected 50 speech segments.
2026-07-30 09:59:35,301 - WARNING - Transcription failed for audio interval 2-9310ms: 'num_frames'
2026-07-30 09:59:35,315 - WARNING - Transcription failed for audio interval 9698-15870ms: 'num_frames'
2026-07-30 09:59:35,327 - WARNING - Transcription failed for audio interval 16001-17854ms: 'num_frames'
2026-07-30 09:59:35,339 - WARNING - Transcription failed for audio interval 17954-23326ms: 'num_frames'
2026-07-30 09:59:35,352 - WARNING - Transcription failed for audio interval 23554-24990ms: 'num_frames'
2026-07-30 09:59:35,365 - WARNING - Transcription failed for audio interval 25122-30526ms: 'num_frames'
2026-07-30 09:59:35,377 - WARNING - Transcription failed for audio interval 32738-36958ms: 'num_frames'
2026-07-30 09:59:35,390 - WARNING - Transcription failed for audio interval 37090-38590ms: 'num_frames'
2026-07-30 09:59:35,402 - WARNING - Transcription failed for audio interval 38850-42782ms: 'num_frames'
2026-07-30 09:59:35,415 - WARNING - Transcription failed for audio interval 42946-45918ms: 'num_frames'
2026-07-30 09:59:35,427 - WARNING - Transcription failed for audio interval 47586-53790ms: 'num_frames'
2026-07-30 09:59:35,440 - WARNING - Transcription failed for audio interval 54050-57854ms: 'num_frames'
2026-07-30 09:59:35,453 - WARNING - Transcription failed for audio interval 57954-62366ms: 'num_frames'
2026-07-30 09:59:35,466 - WARNING - Transcription failed for audio interval 62530-71358ms: 'num_frames'
2026-07-30 09:59:35,478 - WARNING - Transcription failed for audio interval 71874-75262ms: 'num_frames'
2026-07-30 09:59:35,490 - WARNING - Transcription failed for audio interval 75394-78302ms: 'num_frames'
2026-07-30 09:59:35,504 - WARNING - Transcription failed for audio interval 78530-80894ms: 'num_frames'
2026-07-30 09:59:35,518 - WARNING - Transcription failed for audio interval 82178-88510ms: 'num_frames'
2026-07-30 09:59:35,532 - WARNING - Transcription failed for audio interval 88610-90014ms: 'num_frames'
2026-07-30 09:59:35,547 - WARNING - Transcription failed for audio interval 90306-94494ms: 'num_frames'
2026-07-30 09:59:35,560 - WARNING - Transcription failed for audio interval 94594-96574ms: 'num_frames'
2026-07-30 09:59:35,574 - WARNING - Transcription failed for audio interval 98594-115614ms: 'num_frames'
2026-07-30 09:59:35,588 - WARNING - Transcription failed for audio interval 116002-122974ms: 'num_frames'
2026-07-30 09:59:35,601 - WARNING - Transcription failed for audio interval 123362-129086ms: 'num_frames'
2026-07-30 09:59:35,614 - WARNING - Transcription failed for audio interval 129666-138814ms: 'num_frames'
2026-07-30 09:59:35,626 - WARNING - Transcription failed for audio interval 140994-146942ms: 'num_frames'
2026-07-30 09:59:35,639 - WARNING - Transcription failed for audio interval 147138-150430ms: 'num_frames'
2026-07-30 09:59:35,652 - WARNING - Transcription failed for audio interval 151874-161150ms: 'num_frames'
2026-07-30 09:59:35,665 - WARNING - Transcription failed for audio interval 161314-167966ms: 'num_frames'
2026-07-30 09:59:35,679 - WARNING - Transcription failed for audio interval 168290-172798ms: 'num_frames'
2026-07-30 09:59:35,691 - WARNING - Transcription failed for audio interval 172898-174238ms: 'num_frames'
2026-07-30 09:59:35,704 - WARNING - Transcription failed for audio interval 174402-176222ms: 'num_frames'
2026-07-30 09:59:35,717 - WARNING - Transcription failed for audio interval 176642-181982ms: 'num_frames'
2026-07-30 09:59:35,729 - WARNING - Transcription failed for audio interval 182146-183838ms: 'num_frames'
2026-07-30 09:59:35,741 - WARNING - Transcription failed for audio interval 183970-186398ms: 'num_frames'
2026-07-30 09:59:35,753 - WARNING - Transcription failed for audio interval 186786-192222ms: 'num_frames'
2026-07-30 09:59:35,765 - WARNING - Transcription failed for audio interval 194018-195710ms: 'num_frames'
2026-07-30 09:59:35,778 - WARNING - Transcription failed for audio interval 195906-201854ms: 'num_frames'
2026-07-30 09:59:35,791 - WARNING - Transcription failed for audio interval 203938-214654ms: 'num_frames'
2026-07-30 09:59:35,803 - WARNING - Transcription failed for audio interval 214818-225598ms: 'num_frames'
2026-07-30 09:59:35,817 - WARNING - Transcription failed for audio interval 225858-236350ms: 'num_frames'
2026-07-30 09:59:35,830 - WARNING - Transcription failed for audio interval 236482-241246ms: 'num_frames'
2026-07-30 09:59:35,842 - WARNING - Transcription failed for audio interval 242338-242686ms: 'num_frames'
2026-07-30 09:59:35,855 - WARNING - Transcription failed for audio interval 243618-246366ms: 'num_frames'
2026-07-30 09:59:35,868 - WARNING - Transcription failed for audio interval 246818-250526ms: 'num_frames'
2026-07-30 09:59:35,881 - WARNING - Transcription failed for audio interval 253058-257118ms: 'num_frames'
2026-07-30 09:59:35,893 - WARNING - Transcription failed for audio interval 262850-264222ms: 'num_frames'
2026-07-30 09:59:35,905 - WARNING - Transcription failed for audio interval 264354-270078ms: 'num_frames'
2026-07-30 09:59:35,917 - WARNING - Transcription failed for audio interval 270338-274206ms: 'num_frames'
2026-07-30 09:59:35,930 - WARNING - Transcription failed for audio interval 276386-277548ms: 'num_frames'
2026-07-30 09:59:42,432 - INFO - Audio processing completed: 0 speech segments, 33 audio events.
2026-07-30 09:59:42,434 - INFO - [2_news_60s_720p] Extracted 113 keyframes, 0 ASR segments, and 33 audio events.
2026-07-30 09:59:46,669 - INFO - [2_news_60s_720p] Generating window-based LLM context summaries...
2026-07-30 09:59:46,674 - INFO - Summarizing LLM context batch 1/4 [0s - 90s] (33 frames)...
2026-07-30 09:59:46,962 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 09:59:46,963 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 09:59:46,963 - INFO - Summarizing LLM context batch 2/4 [90s - 180s] (37 frames)...
2026-07-30 09:59:47,336 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 09:59:47,337 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 09:59:47,337 - INFO - Summarizing LLM context batch 3/4 [180s - 270s] (41 frames)...
2026-07-30 09:59:47,726 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 09:59:47,727 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 09:59:47,727 - INFO - Summarizing LLM context batch 4/4 [270s - 300s] (2 frames)...
2026-07-30 09:59:47,984 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 09:59:47,986 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 09:59:47,986 - INFO - [2_news_60s_720p] Generating visual and text embeddings...
2026-07-30 09:59:49,276 - INFO - Image embedding batch 1/2 processed (64 images)
2026-07-30 09:59:50,145 - INFO - Image embedding batch 2/2 processed (49 images)
2026-07-30 09:59:50,938 - INFO - Completed video processing: 2_news_60s_720p in 134.11s | Metadata: processed_data/3_metadata/2_news_60s_720p_metadata.json
2026-07-30 09:59:50,939 - INFO - Processing video: pov_walkingtour_720p
Offline Indexing Pipeline: 50%|████████▌ | 2/4 [04:52<04:48, 144.02s/it]2026-07-30 09:59:50,939 - INFO - Segmenting video: data/official_videos/dummy_videos/pov_walkingtour_720p.mp4
2026-07-30 10:00:29,482 - INFO - TransNetV2 detected 48 shots in video 'pov_walkingtour_720p.mp4'.
2026-07-30 10:00:29,559 - INFO - Shot 1 sharpest frame pruned (sim=1.0000)
2026-07-30 10:00:30,248 - INFO - Shot 2 sharpest frame pruned (sim=0.9920)
2026-07-30 10:00:32,401 - INFO - Shot 3 sharpest frame pruned (sim=0.9941)
2026-07-30 10:00:33,423 - INFO - Shot 4 sharpest frame pruned (sim=0.9919)
2026-07-30 10:00:35,155 - INFO - Shot 5 sharpest frame pruned (sim=0.9960)
2026-07-30 10:00:48,846 - INFO - Shot 13 sharpest frame pruned (sim=0.9904)
2026-07-30 10:00:54,692 - INFO - Shot 17 sharpest frame pruned (sim=0.9977)
2026-07-30 10:01:07,271 - INFO - Shot 25 sharpest frame pruned (sim=0.9856)
2026-07-30 10:01:15,861 - INFO - Shot 31 sharpest frame pruned (sim=0.9973)
2026-07-30 10:01:22,014 - INFO - Shot 35 sharpest frame pruned (sim=0.9949)
2026-07-30 10:01:28,378 - INFO - Shot 39 sharpest frame pruned (sim=0.9960)
2026-07-30 10:01:29,829 - INFO - Shot 40 sharpest frame pruned (sim=0.9971)
2026-07-30 10:01:31,768 - INFO - Shot 41 sharpest frame pruned (sim=0.9980)
2026-07-30 10:01:33,147 - INFO - Shot 42 sharpest frame pruned (sim=0.9987)
2026-07-30 10:01:35,050 - INFO - Shot 43 sharpest frame pruned (sim=0.9974)
2026-07-30 10:01:36,403 - INFO - Shot 44 sharpest frame pruned (sim=0.9985)
2026-07-30 10:01:38,169 - INFO - Shot 45 sharpest frame pruned (sim=0.9874)
2026-07-30 10:01:41,444 - INFO - Shot 47 sharpest frame pruned (sim=0.9805)
2026-07-30 10:01:43,482 - INFO - Extracted 78 keyframe artifacts to 'processed_data/1_frames'.
2026-07-30 10:01:53,327 - INFO - Silero VAD detected 4 speech segments.
2026-07-30 10:01:53,343 - WARNING - Transcription failed for audio interval 181506-181918ms: 'num_frames'
2026-07-30 10:01:53,357 - WARNING - Transcription failed for audio interval 204898-205246ms: 'num_frames'
2026-07-30 10:01:53,371 - WARNING - Transcription failed for audio interval 205474-206142ms: 'num_frames'
2026-07-30 10:01:53,383 - WARNING - Transcription failed for audio interval 260098-260541ms: 'num_frames'
2026-07-30 10:02:05,049 - INFO - Audio processing completed: 0 speech segments, 92 audio events.
2026-07-30 10:02:05,053 - INFO - [pov_walkingtour_720p] Extracted 78 keyframes, 0 ASR segments, and 92 audio events.
2026-07-30 10:02:08,271 - INFO - [pov_walkingtour_720p] Generating window-based LLM context summaries...
2026-07-30 10:02:08,276 - INFO - Summarizing LLM context batch 1/5 [0s - 90s] (21 frames)...
2026-07-30 10:02:08,907 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 10:02:08,908 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 10:02:08,908 - INFO - Summarizing LLM context batch 2/5 [90s - 210s] (15 frames)...
2026-07-30 10:02:09,329 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 10:02:09,329 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 10:02:09,330 - INFO - Summarizing LLM context batch 3/5 [210s - 300s] (23 frames)...
2026-07-30 10:02:09,628 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 10:02:09,629 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 10:02:09,629 - INFO - Summarizing LLM context batch 4/5 [300s - 390s] (14 frames)...
2026-07-30 10:02:10,068 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 10:02:10,069 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 10:02:10,069 - INFO - Summarizing LLM context batch 5/5 [420s - 480s] (5 frames)...
2026-07-30 10:02:10,373 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 10:02:10,374 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 10:02:10,375 - INFO - [pov_walkingtour_720p] Generating visual and text embeddings...
2026-07-30 10:02:11,706 - INFO - Image embedding batch 1/2 processed (64 images)
2026-07-30 10:02:11,933 - INFO - Image embedding batch 2/2 processed (14 images)
2026-07-30 10:02:12,463 - INFO - Completed video processing: pov_walkingtour_720p in 141.52s | Metadata: processed_data/3_metadata/pov_walkingtour_720p_metadata.json
2026-07-30 10:02:12,464 - INFO - Processing video: test_transnet
Offline Indexing Pipeline: 75%|████████████▊ | 3/4 [07:13<02:22, 142.88s/it]2026-07-30 10:02:12,464 - INFO - Segmenting video: data/official_videos/dummy_videos/test_transnet.mp4
2026-07-30 10:02:12,495 - INFO - TransNetV2 detected 2 shots in video 'test_transnet.mp4'.
2026-07-30 10:02:12,510 - INFO - Shot 1 sharpest frame pruned (sim=1.0000)
2026-07-30 10:02:12,529 - INFO - Shot 2 sharpest frame pruned (sim=1.0000)
2026-07-30 10:02:12,530 - INFO - Extracted 2 keyframe artifacts to 'processed_data/1_frames'.
2026-07-30 10:02:12,535 - WARNING - Audio extraction failed for video 'data/official_videos/dummy_videos/test_transnet.mp4' (stream may be absent): Command '['/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2', '-y', '-i', 'data/official_videos/dummy_videos/test_transnet.mp4', '-ac', '1', '-ar', '16000', '-f', 'wav', '/tmp/tmpmr3sy6lc.wav']' returned non-zero exit status 234.
2026-07-30 10:02:12,535 - INFO - Video file 'data/official_videos/dummy_videos/test_transnet.mp4' contains no audio track or empty audio stream.
2026-07-30 10:02:12,535 - INFO - [test_transnet] Extracted 2 keyframes, 0 ASR segments, and 0 audio events.
2026-07-30 10:02:12,652 - INFO - [test_transnet] Generating window-based LLM context summaries...
2026-07-30 10:02:12,658 - INFO - Summarizing LLM context batch 1/1 [0s - 30s] (2 frames)...
2026-07-30 10:02:13,031 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-30 10:02:13,032 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-30 10:02:13,033 - INFO - [test_transnet] Generating visual and text embeddings...
2026-07-30 10:02:13,107 - INFO - Completed video processing: test_transnet in 0.64s | Metadata: processed_data/3_metadata/test_transnet_metadata.json
Offline Indexing Pipeline: 100%|█████████████████| 4/4 [07:14<00:00, 108.61s/it]
2026-07-30 10:02:13,108 - INFO - Batch processing completed: 4/4 videos.
2026-07-30 10:02:13,108 - INFO - ==========================================================
2026-07-30 10:02:13,108 - INFO - === PIPELINE EXECUTION SUMMARY ===
2026-07-30 10:02:13,108 - INFO - - Total processed videos: 4
2026-07-30 10:02:13,108 - INFO - - Successful: 4
2026-07-30 10:02:13,109 - INFO - - Failed: 0
2026-07-30 10:02:13,109 - INFO - Output artifacts saved to:
2026-07-30 10:02:13,109 - INFO - + Keyframes: processed_data/1_frames/
2026-07-30 10:02:13,109 - INFO - + Embeddings: processed_data/2_embeddings/
2026-07-30 10:02:13,109 - INFO - + Metadata: processed_data/3_metadata/
2026-07-30 10:02:13,109 - INFO - ----------------------------------------------------------
2026-07-30 10:02:13,109 - INFO - Initiating global FAISS index construction...
2026-07-30 10:02:13,109 - INFO - === STARTING HYBRID GLOBAL INDEXING ===
2026-07-30 10:02:13,135 - INFO - FAISS GPU Resources initialized.
2026-07-30 10:02:13,135 - INFO - Created IndexFlatIP for 332 items
2026-07-30 10:02:13,301 - INFO - Created IndexFlatIP for 332 items
2026-07-30 10:02:13,314 - INFO - Hybrid indices saved to 'processed_data/2_embeddings/'
2026-07-30 10:02:13,315 - INFO - === HYBRID GLOBAL INDEXING COMPLETE ===
Traceback (most recent call last):
File "/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/run_pipeline.py", line 96, in <module>
main()
File "/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/run_pipeline.py", line 90, in main
logger.info(f"Global FAISS index construction complete. Total vectors indexed: {vdb.index.ntotal}")
^^^^^^^^^
AttributeError: 'HybridVectorDB' object has no attribute 'index'
