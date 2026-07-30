Extracting videos_dataset.zip into data/official_videos...
Dataset extraction complete.
Downloading model weights to local models/ directory...
==========================================================
=== BẮT ĐẦU TẢI TRƯỚC TOÀN BỘ WEIGHTS MÔ HÌNH VỀ WORKSPACE ===
==========================================================

[+] Đang xử lý: 'SigLIP 2 Vision Embedding'...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
preprocessor_config.json: 100%|████████████████| 368/368 [00:00<00:00, 1.51MB/s]
The image processor of type `SiglipImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.
config.json: 100%|█████████████████████████████| 432/432 [00:00<00:00, 2.56MB/s]
tokenizer_config.json: 100%|███████████████████| 711/711 [00:00<00:00, 4.42MB/s]
spiece.model: 100%|███████████████████████████| 798k/798k [00:00<00:00, 931kB/s]
special_tokens_map.json: 100%|█████████████████| 409/409 [00:00<00:00, 1.89MB/s]
tokenizer.json: 2.40MB [00:00, 49.1MB/s]
model.safetensors: 100%|██████████████████████| 813M/813M [00:04<00:00, 176MB/s]
Loading weights: 100%|█| 408/408 [00:00<00:00, 1168.68it/s, Materializing param=
Writing model shards: 100%|███████████████████████| 1/1 [00:01<00:00, 1.33s/it]
--> [SUCCESS] Tải thành công mô hình 'SigLIP 2 Vision Embedding'!

[+] Đang xử lý: 'BGE-M3 Text Embedding'...
modules.json: 100%|████████████████████████████| 349/349 [00:00<00:00, 1.77MB/s]
config_sentence_transformers.json: 100%|████████| 123/123 [00:00<00:00, 784kB/s]
README.md: 15.8kB [00:00, 37.1MB/s]
sentence_bert_config.json: 100%|██████████████| 54.0/54.0 [00:00<00:00, 237kB/s]
config.json: 100%|█████████████████████████████| 687/687 [00:00<00:00, 2.93MB/s]
pytorch_model.bin: 100%|████████████████████| 2.27G/2.27G [00:14<00:00, 153MB/s]
Loading weights: 100%|█| 391/391 [00:00<00:00, 2109.39it/s, Materializing param=
model.safetensors: 0%| | 0.00/2.27G [00:00<?, ?B/s]
tokenizer_config.json: 100%|███████████████████| 444/444 [00:00<00:00, 1.58MB/s]
model.safetensors: 0%| | 0.00/2.27G [00:00<?, ?B/s]
model.safetensors: 0%| | 0.00/2.27G [00:00<?, ?B/s]
sentencepiece.bpe.model: 100%|█████████████| 5.07M/5.07M [00:00<00:00, 23.2MB/s]
model.safetensors: 0%| | 0.00/2.27G [00:01<?, ?B/s]
model.safetensors: 0%| | 0.00/2.27G [00:01<?, ?B/s]
model.safetensors: 0%| | 0.00/2.27G [00:01<?, ?B/s]
tokenizer.json: 100%|██████████████████████| 17.1M/17.1M [00:00<00:00, 40.0MB/s]
model.safetensors: 12%|██▎ | 268M/2.27G [00:01<00:01, 1.34GB/s]
special_tokens_map.json: 100%|█████████████████| 964/964 [00:00<00:00, 2.26MB/s]
model.safetensors: 100%|████████████████████| 2.27G/2.27G [00:05<00:00, 568MB/s]
config.json: 100%|██████████████████████████████| 191/191 [00:00<00:00, 614kB/s]

model.safetensors: 100%|████████████████████| 2.27G/2.27G [00:13<00:00, 174MB/s]

Writing model shards: 100%|███████████████████████| 1/1 [00:05<00:00, 5.03s/it]
--> [SUCCESS] Tải thành công mô hình 'BGE-M3 Text Embedding'!

[+] Đang xử lý: 'PhoWhisper Small ASR'...
preprocessor_config.json: 100%|████████████████| 339/339 [00:00<00:00, 1.26MB/s]
config.json: 1.33kB [00:00, 5.46MB/s]
tokenizer_config.json: 100%|███████████████████| 805/805 [00:00<00:00, 4.91MB/s]
vocab.json: 836kB [00:00, 23.7MB/s]
tokenizer.json: 2.20MB [00:00, 126MB/s]
merges.txt: 494kB [00:00, 97.4MB/s]
normalizer.json: 52.7kB [00:00, 79.7MB/s]
added_tokens.json: 2.08kB [00:00, 6.52MB/s]
special_tokens_map.json: 2.08kB [00:00, 5.17MB/s]
pytorch_model.bin: 100%|██████████████████████| 967M/967M [00:05<00:00, 172MB/s]
Loading weights: 100%|█| 480/480 [00:00<00:00, 2263.28it/s, Materializing param=
The tied weights mapping and config for this model specifies to tie model.decoder.embed_tokens.weight to proj_out.weight, but both are present in the checkpoints, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning
generation_config.json: 3.83kB [00:00, 12.1MB/s]
Writing model shards: 100%|███████████████████████| 1/1 [00:01<00:00, 1.66s/it]
--> [SUCCESS] Tải thành công mô hình 'PhoWhisper Small ASR'!

[+] Đang xử lý: 'Silero VAD'...
/usr/local/lib/python3.12/dist-packages/torch/hub.py:247: UserWarning: You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to load(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or load(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
\_check_repo_is_trusted(
Downloading: "https://github.com/snakers4/silero-vad/zipball/master" to /kaggle/working/W1.1Thr34d-AIC26-YOLOe26/models/torch_hub/master.zip
--> [SUCCESS] Tải thành công 'Silero VAD' vào '/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/models/torch_hub'!

[+] Đang xử lý: 'PaddleOCR Models (vi, en)'...
/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/scripts/download_models.py:123: DeprecationWarning: The parameter `use_angle_cls` has been deprecated and will be removed in the future. Please use `use_textline_orientation` instead.
ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=False)
--> [ERROR] LỖI khi tải mô hình 'PaddleOCR Models (vi, en)': Unknown argument: use_gpu
Traceback (most recent call last):
File "/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/scripts/download_models.py", line 123, in download_all_models
ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=False)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/paddleocr/\_pipelines/ocr.py", line 173, in **init**
super().**init**(\*\*base_params)
File "/usr/local/lib/python3.12/dist-packages/paddleocr/\_pipelines/base.py", line 63, in **init**
self.\_common_args = parse_common_args(
^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/paddleocr/\_common_args.py", line 54, in parse_common_args
raise ValueError(f"Unknown argument: {name}")
ValueError: Unknown argument: use_gpu

==========================================================
=== BẢNG KIỂM TRA TRẠNG THÁI TẢI (CHECKLIST) ===
==========================================================
[SUCCESS] THÀNH CÔNG : SigLIP 2 Vision Embedding
[SUCCESS] THÀNH CÔNG : BGE-M3 Text Embedding
[SUCCESS] THÀNH CÔNG : PhoWhisper Small ASR
[SUCCESS] THÀNH CÔNG : Silero VAD
[FAILED] THẤT BẠI : PaddleOCR Models (vi, en)
==========================================================

[FATAL] CÓ LỖI XẢY RA! Vui lòng kiểm tra lại mạng. CHƯƠNG TRÌNH DỪNG LẠI (CRASH).
Downloading YOLOE-26L-PF...

2026-07-30 08:16:17,693 - INFO - Loading faiss with AVX512 support.
2026-07-30 08:16:17,694 - INFO - Could not load library with AVX512 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx512'")
2026-07-30 08:16:17,694 - INFO - Loading faiss with AVX2 support.
2026-07-30 08:16:17,694 - INFO - Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-30 08:16:17,694 - INFO - Loading faiss.
2026-07-30 08:16:17,721 - INFO - Successfully loaded faiss.
2026-07-30 08:16:17,726 - INFO - ==========================================================
2026-07-30 08:16:17,726 - INFO - === STARTING OFFLINE INDEXING PIPELINE (PHASE 1) ===
2026-07-30 08:16:17,726 - INFO - ==========================================================
2026-07-30 08:16:17,732 - INFO - Discovered 4 official video files in 'data/official_videos'.
2026-07-30 08:16:17,984 - INFO - CUDA GPU detected: Constraining max_workers to 1 to optimize CUDA compute efficiency.
2026-07-30 08:16:17,984 - INFO - Pre-loading shared model singletons...
2026-07-30 08:16:17,990 - INFO - Local weights file 'src/preprocessing/weights/transnetv2-pytorch-weights.pth' not found. Initializing default TransNetV2 PyTorch model...
2026-07-30 08:16:19,685 - INFO - TransNetV2 model initialized successfully on CUDA GPU.
2026-07-30 08:16:19,880 - INFO - Using bundled ffmpeg binary at '/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
Creating new Ultralytics Settings v0.0.6 file ✅
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
2026-07-30 08:16:20,760 - INFO - YOLOv9 model loaded successfully from 'yoloe-26l-seg-pf.pt'.
/usr/local/lib/python3.12/dist-packages/paddle/utils/cpp_extension/extension_utils.py:712: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
warnings.warn(warning_message)
Creating model: ('PP-LCNet_x1_0_doc_ori', None, None)
Checking connectivity to the model hosters, this may take a while. To bypass this check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` to `True`.
Using official model (PP-LCNet_x1_0_doc_ori), the model files will be automatically downloaded and saved in `/root/.paddlex/official_models/PP-LCNet_x1_0_doc_ori`.
2026-07-30 08:16:35,473 - INFO - HTTP Request: GET https://huggingface.co/api/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/revision/main "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 6 files: 0%| | 0/6 [00:00<?, ?it/s]2026-07-30 08:16:35,550 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori/resolve/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/inference.pdiparams "HTTP/1.1 302 Found"
2026-07-30 08:16:35,592 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori/resolve/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:35,608 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/config.json "HTTP/1.1 200 OK"
2026-07-30 08:16:35,610 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori/resolve/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/README.md "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:35,617 - INFO - HTTP Request: GET https://huggingface.co/api/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/xet-read-token/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8 "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 0.00/6.75M [00:00<?, ?B/s]2026-07-30 08:16:35,618 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori/resolve/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/inference.yml "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:35,620 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori/resolve/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/.gitattributes "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:35,625 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/config.json "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 2.56k/6.75M [00:00<06:35, 17.1kB/s]2026-07-30 08:16:35,626 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/README.md "HTTP/1.1 200 OK"
2026-07-30 08:16:35,633 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori/resolve/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/inference.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:35,634 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/inference.yml "HTTP/1.1 200 OK"
2026-07-30 08:16:35,638 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/.gitattributes "HTTP/1.1 200 OK"
2026-07-30 08:16:35,646 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/README.md "HTTP/1.1 200 OK"
2026-07-30 08:16:35,652 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/inference.json "HTTP/1.1 200 OK"
2026-07-30 08:16:35,653 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/inference.yml "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 9.41k/6.75M [00:00<06:34, 17.1kB/s]2026-07-30 08:16:35,654 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/.gitattributes "HTTP/1.1 200 OK"

Fetching 6 files: 17%|████▌ | 1/6 [00:00<00:00, 5.65it/s]2026-07-30 08:16:35,674 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori/d3b95a6dff5fe8a94f2748e12b61cb26818a0df8/inference.json "HTTP/1.1 200 OK"
Downloading (incomplete total...): 6.87MB [00:00, 15.9MB/s]  
Fetching 6 files: 100%|███████████████████████████| 6/6 [00:00<00:00, 10.57it/s]
Download complete: : 6.87MB [00:00, 15.9MB/s] Creating model: ('UVDoc', None, None)
Using official model (UVDoc), the model files will be automatically downloaded and saved in `/root/.paddlex/official_models/UVDoc`.
2026-07-30 08:16:36,486 - INFO - HTTP Request: GET https://huggingface.co/api/models/PaddlePaddle/UVDoc/revision/main "HTTP/1.1 200 OK"

Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 6 files: 0%| | 0/6 [00:00<?, ?it/s]2026-07-30 08:16:36,557 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/UVDoc/resolve/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:36,559 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/UVDoc/resolve/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/inference.pdiparams "HTTP/1.1 302 Found"
2026-07-30 08:16:36,560 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/UVDoc/resolve/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/inference.yml "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:36,572 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/config.json "HTTP/1.1 200 OK"
2026-07-30 08:16:36,576 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/inference.yml "HTTP/1.1 200 OK"
2026-07-30 08:16:36,587 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/config.json "HTTP/1.1 200 OK"
2026-07-30 08:16:36,592 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/inference.yml "HTTP/1.1 200 OK"

Downloading (incomplete total...): 1.49kB [00:00, 14.3kB/s]
Downloading (incomplete total...): 1.82kB [00:00, 17.4kB/s]2026-07-30 08:16:36,606 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/UVDoc/resolve/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/README.md "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:36,607 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/UVDoc/resolve/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/inference.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:36,620 - INFO - HTTP Request: GET https://huggingface.co/api/models/PaddlePaddle/UVDoc/xet-read-token/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3 "HTTP/1.1 200 OK"

Downloading (incomplete total...): 0%| | 1.82k/32.1M [00:00<30:42, 17.4kB/s]2026-07-30 08:16:36,623 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/inference.json "HTTP/1.1 200 OK"
2026-07-30 08:16:36,623 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/README.md "HTTP/1.1 200 OK"
2026-07-30 08:16:36,642 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/inference.json "HTTP/1.1 200 OK"
2026-07-30 08:16:36,642 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/README.md "HTTP/1.1 200 OK"
2026-07-30 08:16:36,656 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/UVDoc/resolve/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/.gitattributes "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:36,670 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/.gitattributes "HTTP/1.1 200 OK"
2026-07-30 08:16:36,686 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/UVDoc/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3/.gitattributes "HTTP/1.1 200 OK"

Fetching 6 files: 17%|████▌ | 1/6 [00:00<00:00, 5.17it/s]
Downloading (incomplete total...): 1%| | 199k/32.1M [00:00<00:50, 630kB/s]
Downloading (incomplete total...): 32.3MB [00:00, 39.3MB/s]

Fetching 6 files: 100%|███████████████████████████| 6/6 [00:00<00:00, 6.38it/s]

Download complete: : 32.3MB [00:00, 39.3MB/s] Creating model: ('PP-OCRv6_medium_det', None, None)
Using official model (PP-OCRv6_medium_det), the model files will be automatically downloaded and saved in `/root/.paddlex/official_models/PP-OCRv6_medium_det`.
2026-07-30 08:16:37,629 - INFO - HTTP Request: GET https://huggingface.co/api/models/PaddlePaddle/PP-OCRv6_medium_det/revision/main "HTTP/1.1 200 OK"

Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 5 files: 0%| | 0/5 [00:00<?, ?it/s]2026-07-30 08:16:37,714 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det/resolve/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/inference.yml "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:37,729 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_det/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/inference.yml "HTTP/1.1 200 OK"
2026-07-30 08:16:37,745 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_det/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/inference.yml "HTTP/1.1 200 OK"

Downloading (incomplete total...): 0%| | 0.00/886 [00:00<?, ?B/s]

Downloading (incomplete total...): 100%|███████| 886/886 [00:00<00:00, 7.70kB/s]2026-07-30 08:16:37,752 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det/resolve/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/inference.pdiparams "HTTP/1.1 302 Found"
2026-07-30 08:16:37,802 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det/resolve/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/inference.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:37,803 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det/resolve/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/.gitattributes "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:37,804 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det/resolve/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/README.md "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:37,811 - INFO - HTTP Request: GET https://huggingface.co/api/models/PaddlePaddle/PP-OCRv6_medium_det/xet-read-token/8e0f56fb2ef86b461d99cfc7ac5c137738985f61 "HTTP/1.1 200 OK"

Downloading (incomplete total...): 0%| | 886/62.0M [00:00<2:14:06, 7.70kB/s]2026-07-30 08:16:37,817 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_det/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/inference.json "HTTP/1.1 200 OK"
2026-07-30 08:16:37,819 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_det/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/.gitattributes "HTTP/1.1 200 OK"
2026-07-30 08:16:37,820 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_det/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/README.md "HTTP/1.1 200 OK"
2026-07-30 08:16:37,833 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_det/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/inference.json "HTTP/1.1 200 OK"
2026-07-30 08:16:37,837 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_det/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/.gitattributes "HTTP/1.1 200 OK"
2026-07-30 08:16:37,838 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_det/8e0f56fb2ef86b461d99cfc7ac5c137738985f61/README.md "HTTP/1.1 200 OK"

Fetching 5 files: 20%|█████▍ | 1/5 [00:00<00:00, 4.83it/s]

Downloading (incomplete total...): 1%| | 338k/62.0M [00:00<01:05, 939kB/s]

Downloading (incomplete total...): 62.3MB [00:00, 73.0MB/s]

Fetching 5 files: 100%|███████████████████████████| 5/5 [00:01<00:00, 4.19it/s]

Download complete: : 62.3MB [00:01, 73.0MB/s] Creating model: ('PP-OCRv6_medium_rec', None, None)
Using official model (PP-OCRv6_medium_rec), the model files will be automatically downloaded and saved in `/root/.paddlex/official_models/PP-OCRv6_medium_rec`.
2026-07-30 08:16:39,249 - INFO - HTTP Request: GET https://huggingface.co/api/models/PaddlePaddle/PP-OCRv6_medium_rec/revision/main "HTTP/1.1 200 OK"

Downloading (incomplete total...): 0.00B [00:00, ?B/s]

Fetching 5 files: 0%| | 0/5 [00:00<?, ?it/s]2026-07-30 08:16:39,318 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec/resolve/e5a92bcbc5cc1b494628e458d267778f0704fd7c/.gitattributes "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:39,332 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_rec/e5a92bcbc5cc1b494628e458d267778f0704fd7c/.gitattributes "HTTP/1.1 200 OK"
2026-07-30 08:16:39,348 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_rec/e5a92bcbc5cc1b494628e458d267778f0704fd7c/.gitattributes "HTTP/1.1 200 OK"
2026-07-30 08:16:39,368 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec/resolve/e5a92bcbc5cc1b494628e458d267778f0704fd7c/README.md "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:39,384 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_rec/e5a92bcbc5cc1b494628e458d267778f0704fd7c/README.md "HTTP/1.1 200 OK"
2026-07-30 08:16:39,400 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_rec/e5a92bcbc5cc1b494628e458d267778f0704fd7c/README.md "HTTP/1.1 200 OK"

Downloading (incomplete total...): 25.0kB [00:00, 167kB/s]

Fetching 5 files: 40%|██████████▊ | 2/5 [00:00<00:00, 13.61it/s]2026-07-30 08:16:39,419 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec/resolve/e5a92bcbc5cc1b494628e458d267778f0704fd7c/inference.json "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:39,424 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec/resolve/e5a92bcbc5cc1b494628e458d267778f0704fd7c/inference.pdiparams "HTTP/1.1 302 Found"
2026-07-30 08:16:39,433 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_rec/e5a92bcbc5cc1b494628e458d267778f0704fd7c/inference.json "HTTP/1.1 200 OK"
2026-07-30 08:16:39,440 - INFO - HTTP Request: HEAD https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec/resolve/e5a92bcbc5cc1b494628e458d267778f0704fd7c/inference.yml "HTTP/1.1 307 Temporary Redirect"
2026-07-30 08:16:39,448 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_rec/e5a92bcbc5cc1b494628e458d267778f0704fd7c/inference.json "HTTP/1.1 200 OK"
2026-07-30 08:16:39,454 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_rec/e5a92bcbc5cc1b494628e458d267778f0704fd7c/inference.yml "HTTP/1.1 200 OK"
2026-07-30 08:16:39,469 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/PaddlePaddle/PP-OCRv6_medium_rec/e5a92bcbc5cc1b494628e458d267778f0704fd7c/inference.yml "HTTP/1.1 200 OK"
2026-07-30 08:16:39,481 - INFO - HTTP Request: GET https://huggingface.co/api/models/PaddlePaddle/PP-OCRv6_medium_rec/xet-read-token/e5a92bcbc5cc1b494628e458d267778f0704fd7c "HTTP/1.1 200 OK"

Downloading (incomplete total...): 1%| | 397k/76.5M [00:00<07:36, 167kB/s]

Downloading (incomplete total...): 1%| | 397k/76.5M [00:00<01:17, 977kB/s]

Downloading (incomplete total...): 97%|██▉| 74.1M/76.5M [00:01<00:00, 84.2MB/s]

Fetching 5 files: 100%|███████████████████████████| 5/5 [00:01<00:00, 4.03it/s]

Download complete: : 32.3MB [00:04, 7.94MB/s]  
Download complete: : 62.3MB [00:02, 21.4MB/s]
Download complete: : 76.9MB [00:01, 59.2MB/s]
Download complete: : 6.87MB [00:05, 1.24MB/s]
2026-07-30 08:16:41,644 - INFO - PaddleOCR engine initialized (lang=vi).
2026-07-30 08:16:41,649 - INFO - EmbeddingGenerator initialized on device 'cuda' (fp16=True).
2026-07-30 08:16:41,667 - INFO - Loading SigLIP 2 model from 'models/siglip-base-patch16-224' (dtype=torch.float16)...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█| 408/408 [00:00<00:00, 844.59it/s, Materializing param=v
2026-07-30 08:16:42,450 - INFO - SigLIP 2 model loaded successfully.
2026-07-30 08:16:42,450 - WARNING - FlagEmbedding not available (No module named 'FlagEmbedding'). Falling back to SentenceTransformer (Dense only).
2026-07-30 08:16:42,924 - INFO - TensorFlow version 2.20.0 available.
2026-07-30 08:16:42,925 - INFO - JAX version 0.7.2 available.
2026-07-30 08:16:45,951 - INFO - Loading SentenceTransformer model from models/bge-m3.
Loading weights: 100%|█| 391/391 [00:00<00:00, 653.78it/s, Materializing param=p
2026-07-30 08:16:49,656 - INFO - HTTP Request: GET https://huggingface.co/api/models/models/bge-m3 "HTTP/1.1 401 Unauthorized"
2026-07-30 08:16:49,988 - INFO - BGE-M3 model loaded successfully via SentenceTransformer.
2026-07-30 08:16:49,988 - INFO - MultiThreadPipelineWorker initialized (max_workers=1).
2026-07-30 08:16:49,988 - INFO - Dispatching 4 video files to MultiThreadPipelineWorker...
2026-07-30 08:16:49,988 - INFO - Initiating batch processing for 4 videos...
2026-07-30 08:16:49,989 - INFO - Processing video: 1_news_60s_720p
2026-07-30 08:16:49,989 - INFO - Segmenting video: data/official_videos/dummy_videos/1_news_60s_720p.mp4
Offline Indexing Pipeline: 0%| | 0/4 [00:00<?, ?it/s]2026-07-30 08:17:12,337 - INFO - TransNetV2 detected 95 shots in video '1_news_60s_720p.mp4'.
2026-07-30 08:17:16,461 - INFO - Shot 3 sharpest frame pruned (sim=0.9994)
2026-07-30 08:17:18,246 - INFO - Shot 4 sharpest frame pruned (sim=0.9996)
2026-07-30 08:17:19,548 - INFO - Shot 5 sharpest frame pruned (sim=0.9894)
2026-07-30 08:17:20,368 - INFO - Shot 6 sharpest frame pruned (sim=0.9999)
2026-07-30 08:17:21,102 - INFO - Shot 7 sharpest frame pruned (sim=0.9987)
2026-07-30 08:17:22,451 - INFO - Shot 8 sharpest frame pruned (sim=0.9958)
2026-07-30 08:17:23,680 - INFO - Shot 9 sharpest frame pruned (sim=0.9869)
2026-07-30 08:17:24,464 - INFO - Shot 10 sharpest frame pruned (sim=0.9981)
2026-07-30 08:17:25,546 - INFO - Shot 11 sharpest frame pruned (sim=0.9990)
2026-07-30 08:17:27,392 - INFO - Shot 12 sharpest frame pruned (sim=0.9925)
2026-07-30 08:17:28,682 - INFO - Shot 13 sharpest frame pruned (sim=0.9926)
2026-07-30 08:17:31,271 - INFO - Shot 15 sharpest frame pruned (sim=0.9881)
2026-07-30 08:17:32,514 - INFO - Shot 16 sharpest frame pruned (sim=0.9954)
2026-07-30 08:17:33,202 - INFO - Shot 17 sharpest frame pruned (sim=0.9928)
2026-07-30 08:17:34,192 - INFO - Shot 18 sharpest frame pruned (sim=0.9939)
2026-07-30 08:17:35,286 - INFO - Shot 19 sharpest frame pruned (sim=0.9958)
2026-07-30 08:17:36,146 - INFO - Shot 20 sharpest frame pruned (sim=0.9981)
2026-07-30 08:17:39,633 - INFO - Shot 22 sharpest frame pruned (sim=0.9947)
2026-07-30 08:17:41,459 - INFO - Shot 24 sharpest frame pruned (sim=0.9994)
2026-07-30 08:17:42,314 - INFO - Shot 25 sharpest frame pruned (sim=0.9992)
2026-07-30 08:17:43,763 - INFO - Shot 26 sharpest frame pruned (sim=0.9834)
2026-07-30 08:17:46,011 - INFO - Shot 28 sharpest frame pruned (sim=0.9927)
