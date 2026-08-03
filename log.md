2026-07-31 02:05:47,220 - INFO - Loading faiss with AVX2 support.
2026-07-31 02:05:47,220 - INFO - Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-31 02:05:47,220 - INFO - Loading faiss.
2026-07-31 02:05:47,361 - INFO - Successfully loaded faiss.
2026-07-31 02:05:47,370 - INFO - ==========================================================
2026-07-31 02:05:47,370 - INFO - === STARTING OFFLINE INDEXING PIPELINE (PHASE 1) ===
2026-07-31 02:05:47,370 - INFO - ==========================================================
2026-07-31 02:05:47,376 - INFO - Discovered 4 official video files in 'data/official_videos'.
2026-07-31 02:05:47,380 - INFO - Pre-loading shared model singletons...
2026-07-31 02:05:47,385 - INFO - Loading TransNetV2 model from local weights: 'src/preprocessing/weights/transnetv2-pytorch-weights.pth'
2026-07-31 02:05:48,622 - INFO - TransNetV2 model initialized successfully on CPU.
2026-07-31 02:05:48,777 - INFO - Using bundled ffmpeg binary at '/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
Creating new Ultralytics Settings v0.0.6 file ✅
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
2026-07-31 02:05:49,626 - INFO - YOLOE-26 model loaded successfully from 'yoloe-26l-seg-pf.pt'.
2026-07-31 02:05:49,630 - INFO - EmbeddingGenerator initialized on device 'cpu' (fp16=False).
2026-07-31 02:05:54,614 - INFO - Loading SigLIP 2 model from 'models/siglip-base-patch16-224' (dtype=torch.float32)...
Loading weights: 100%|█| 408/408 [00:00<00:00, 2218.45it/s, Materializing param=
2026-07-31 02:05:54,960 - INFO - SigLIP 2 model loaded successfully.
2026-07-31 02:05:55,085 - INFO - TensorFlow version 2.20.0 available.
2026-07-31 02:05:55,086 - INFO - JAX version 0.7.2 available.
2026-07-31 02:05:55,587 - INFO - Loading BGEM3FlagModel from 'BAAI/bge-m3'...
2026-07-31 02:05:55,877 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:05:56,073 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/config.json "HTTP/1.1 200 OK"
2026-07-31 02:05:56,271 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/config.json "HTTP/1.1 200 OK"
config.json: 100%|█████████████████████████████| 687/687 [00:00<00:00, 2.26MB/s]
2026-07-31 02:05:56,480 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:05:56,680 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/tokenizer_config.json "HTTP/1.1 200 OK"
2026-07-31 02:05:56,876 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/tokenizer_config.json "HTTP/1.1 200 OK"
tokenizer_config.json: 100%|███████████████████| 444/444 [00:00<00:00, 2.54MB/s]
2026-07-31 02:05:57,080 - INFO - HTTP Request: GET https://huggingface.co/api/models/BAAI/bge-m3/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-07-31 02:05:57,288 - INFO - HTTP Request: GET https://huggingface.co/api/models/BAAI/bge-m3/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-07-31 02:05:57,495 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/sentencepiece.bpe.model "HTTP/1.1 302 Found"
2026-07-31 02:05:57,699 - INFO - HTTP Request: GET https://huggingface.co/api/models/BAAI/bge-m3/xet-read-token/5617a9f61b028005a4858fdac845db406aefb181 "HTTP/1.1 200 OK"
sentencepiece.bpe.model: 100%|█████████████| 5.07M/5.07M [00:01<00:00, 3.12MB/s]
2026-07-31 02:05:59,540 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json "HTTP/1.1 302 Found"
tokenizer.json: 100%|██████████████████████| 17.1M/17.1M [00:01<00:00, 16.9MB/s]
2026-07-31 02:06:00,755 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/added_tokens.json "HTTP/1.1 404 Not Found"
2026-07-31 02:06:00,960 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/special_tokens_map.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:01,219 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/special_tokens_map.json "HTTP/1.1 200 OK"
2026-07-31 02:06:01,412 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/special_tokens_map.json "HTTP/1.1 200 OK"
special_tokens_map.json: 100%|█████████████████| 964/964 [00:00<00:00, 4.75MB/s]
2026-07-31 02:06:01,616 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/chat_template.jinja "HTTP/1.1 404 Not Found"
2026-07-31 02:06:03,942 - INFO - HTTP Request: GET https://huggingface.co/api/models/BAAI/bge-m3 "HTTP/1.1 200 OK"
2026-07-31 02:06:04,197 - INFO - HTTP Request: GET https://huggingface.co/api/models/BAAI/bge-m3/revision/main "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 30 files: 0%| | 0/30 [00:00<?, ?it/s]2026-07-31 02:06:04,444 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/.gitattributes "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:04,461 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/imgs/long.jpg "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:04,462 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling/config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:04,465 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/README.md "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:04,467 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/imgs/.DS_Store "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:04,468 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/config_sentence_transformers.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:04,475 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/imgs/bm25.jpg "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:04,509 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/colbert_linear.pt "HTTP/1.1 302 Found"
Downloading (incomplete total...): 0%| | 0.00/2.10M [00:00<?, ?B/s]2026-07-31 02:06:04,654 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling%2Fconfig.json "HTTP/1.1 200 OK"
2026-07-31 02:06:04,682 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/.gitattributes "HTTP/1.1 200 OK"
2026-07-31 02:06:04,693 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Flong.jpg "HTTP/1.1 200 OK"
2026-07-31 02:06:04,699 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/README.md "HTTP/1.1 200 OK"
2026-07-31 02:06:04,700 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2F.DS_Store "HTTP/1.1 200 OK"
2026-07-31 02:06:04,704 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fbm25.jpg "HTTP/1.1 200 OK"
2026-07-31 02:06:04,705 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/config_sentence_transformers.json "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 0.00/2.10M [00:00<?, ?B/s]2026-07-31 02:06:04,877 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/.gitattributes "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 1.63k/2.10M [00:00<03:22, 10.4kB/s]
Fetching 30 files: 3%|▊ | 1/30 [00:00<00:19, 1.49it/s]2026-07-31 02:06:04,883 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling%2Fconfig.json "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 1.63k/2.10M [00:00<03:22, 10.4kB/s]2026-07-31 02:06:04,884 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Flong.jpg "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 1.82k/2.59M [00:00<04:09, 10.4kB/s]2026-07-31 02:06:04,902 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/config_sentence_transformers.json "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 1.82k/2.59M [00:00<04:09, 10.4kB/s]2026-07-31 02:06:04,932 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2F.DS_Store "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 1.94k/2.59M [00:00<04:10, 10.4kB/s]2026-07-31 02:06:04,940 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/README.md "HTTP/1.1 200 OK"
2026-07-31 02:06:04,943 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fbm25.jpg "HTTP/1.1 200 OK"
Downloading (incomplete total...): 24%|▉ | 641k/2.72M [00:00<00:00, 2.47MB/s]2026-07-31 02:06:05,122 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/imgs/miracl.jpg "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,125 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/imgs/mkqa.jpg "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,144 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/imgs/nqa.jpg "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,181 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/imgs/others.webp "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,196 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/long.jpg "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,221 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/modules.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,268 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/onnx/Constant_7_attr__value "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,318 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fmiracl.jpg "HTTP/1.1 200 OK"
2026-07-31 02:06:05,342 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fnqa.jpg "HTTP/1.1 200 OK"
2026-07-31 02:06:05,359 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fmkqa.jpg "HTTP/1.1 200 OK"
2026-07-31 02:06:05,374 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fothers.webp "HTTP/1.1 200 OK"
2026-07-31 02:06:05,390 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/long.jpg "HTTP/1.1 200 OK"
2026-07-31 02:06:05,457 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/modules.json "HTTP/1.1 200 OK"
2026-07-31 02:06:05,516 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fmiracl.jpg "HTTP/1.1 200 OK"
Downloading (incomplete total...): 37%|█ | 1.22M/3.30M [00:01<00:01, 1.49MB/s]2026-07-31 02:06:05,574 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fnqa.jpg "HTTP/1.1 200 OK"
Downloading (incomplete total...): 35%|█ | 1.22M/3.46M [00:01<00:01, 1.49MB/s]2026-07-31 02:06:05,585 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fmkqa.jpg "HTTP/1.1 200 OK"
Downloading (incomplete total...): 30%|▉ | 1.22M/4.07M [00:01<00:01, 1.49MB/s]2026-07-31 02:06:05,614 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/imgs%2Fothers.webp "HTTP/1.1 200 OK"
Downloading (incomplete total...): 30%|▉ | 1.22M/4.09M [00:01<00:01, 1.49MB/s]2026-07-31 02:06:05,620 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/long.jpg "HTTP/1.1 200 OK"
Downloading (incomplete total...): 29%|▉ | 1.24M/4.22M [00:01<00:01, 1.49MB/s]2026-07-31 02:06:05,639 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/onnx%2FConstant_7_attr__value "HTTP/1.1 200 OK"
Downloading (incomplete total...): 33%|▉ | 1.40M/4.22M [00:01<00:01, 1.50MB/s]2026-07-31 02:06:05,689 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/modules.json "HTTP/1.1 200 OK"
Downloading (incomplete total...): 33%|▉ | 1.40M/4.22M [00:01<00:01, 1.50MB/s]
Fetching 30 files: 13%|███▎ | 4/30 [00:01<00:09, 2.82it/s]2026-07-31 02:06:05,775 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/onnx/config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,864 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/onnx/model.onnx "HTTP/1.1 302 Found"
Downloading (incomplete total...): 86%|██▌| 4.23M/4.94M [00:01<00:00, 1.50MB/s]2026-07-31 02:06:05,867 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/onnx%2FConstant_7_attr__value "HTTP/1.1 200 OK"
Downloading (incomplete total...): 86%|██▌| 4.30M/5.01M [00:01<00:00, 5.40MB/s]
Fetching 30 files: 53%|████████████▊ | 16/30 [00:01<00:01, 13.48it/s]2026-07-31 02:06:05,889 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/onnx/model.onnx_data "HTTP/1.1 302 Found"
Downloading (incomplete total...): 0%| | 4.30M/2.27G [00:01<06:59, 5.40MB/s]2026-07-31 02:06:05,932 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/onnx/sentencepiece.bpe.model "HTTP/1.1 302 Found"
2026-07-31 02:06:05,949 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/onnx/special_tokens_map.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,963 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/onnx/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:05,996 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/onnx%2Fconfig.json "HTTP/1.1 200 OK"
2026-07-31 02:06:06,014 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/onnx/tokenizer.json "HTTP/1.1 302 Found"
Downloading (incomplete total...): 0%| | 4.30M/2.29G [00:01<07:03, 5.40MB/s]2026-07-31 02:06:06,122 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/pytorch_model.bin "HTTP/1.1 302 Found"
Downloading (incomplete total...): 0%| | 4.30M/4.56G [00:01<14:03, 5.40MB/s]2026-07-31 02:06:06,161 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/sentence_bert_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:06:06,173 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/onnx%2Fspecial_tokens_map.json "HTTP/1.1 200 OK"
2026-07-31 02:06:06,187 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/onnx%2Ftokenizer_config.json "HTTP/1.1 200 OK"
2026-07-31 02:06:06,220 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/onnx%2Fconfig.json "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0%| | 5.02M/4.56G [00:02<20:11, 3.76MB/s]2026-07-31 02:06:06,380 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/sentence_bert_config.json "HTTP/1.1 200 OK"
2026-07-31 02:06:06,398 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/onnx%2Ftokenizer_config.json "HTTP/1.1 200 OK"
2026-07-31 02:06:06,410 - INFO - HTTP Request: HEAD https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/sparse_linear.pt "HTTP/1.1 302 Found"
Downloading (incomplete total...): 0%| | 5.03M/4.56G [00:02<20:11, 3.76MB/s]2026-07-31 02:06:06,587 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/sentence_bert_config.json "HTTP/1.1 200 OK"
Downloading (incomplete total...): 4.56GB [00:13, 563MB/s]  
Downloading (incomplete total...): 4.56GB [00:30, 563MB/s]0:20<00:01, 13.48it/s]
Fetching 30 files: 100%|████████████████████████| 30/30 [00:31<00:00, 1.06s/it]
Download complete: : 4.56GB [00:31, 563MB/s]  
Loading weights: 0%| | 0/391 [00:00<?, ?it/s]
Loading weights: 0%| | 1/391 [00:00<00:00, 8305.55it/s, Materializing param=em
Loading weights: 0%| | 1/391 [00:00<00:00, 3151.24it/s, Materializing param=em
Loading weights: 1%| | 2/391 [00:00<00:00, 2855.21it/s, Materializing param=em
Loading weights: 1%| | 2/391 [00:00<00:00, 2304.56it/s, Materializing param=em
Loading weights: 1%| | 3/391 [00:00<00:00, 479.81it/s, Materializing param=emb
Loading weights: 1%| | 3/391 [00:00<00:00, 463.77it/s, Materializing param=emb
Loading weights: 1%| | 4/391 [00:00<00:00, 599.12it/s, Materializing param=emb
Loading weights: 1%| | 4/391 [00:00<00:00, 588.55it/s, Materializing param=emb
Loading weights: 1%| | 5/391 [00:00<00:00, 716.80it/s, Materializing param=emb
Loading weights: 1%| | 5/391 [00:00<00:00, 705.76it/s, Materializing param=emb
Loading weights: 2%| | 6/391 [00:00<00:00, 827.50it/s, Materializing param=enc
Loading weights: 2%| | 6/391 [00:00<00:00, 810.26it/s, Materializing param=enc
Loading weights: 2%| | 7/391 [00:00<00:00, 922.23it/s, Materializing param=enc
Loading weights: 2%| | 7/391 [00:00<00:00, 913.08it/s, Materializing param=enc
Loading weights: 2%| | 8/391 [00:00<00:00, 1028.11it/s, Materializing param=en
Loading weights: 2%| | 8/391 [00:00<00:00, 1016.43it/s, Materializing param=en
Loading weights: 2%| | 9/391 [00:00<00:00, 1126.26it/s, Materializing param=en
Loading weights: 2%| | 9/391 [00:00<00:00, 1115.97it/s, Materializing param=en
Loading weights: 3%| | 10/391 [00:00<00:00, 1220.66it/s, Materializing param=e
Loading weights: 3%| | 10/391 [00:00<00:00, 1209.64it/s, Materializing param=e
Loading weights: 3%| | 11/391 [00:00<00:00, 1301.77it/s, Materializing param=e
Loading weights: 3%| | 11/391 [00:00<00:00, 1288.86it/s, Materializing param=e
Loading weights: 3%| | 12/391 [00:00<00:00, 1374.77it/s, Materializing param=e
Loading weights: 3%| | 12/391 [00:00<00:00, 1361.75it/s, Materializing param=e
Loading weights: 3%| | 13/391 [00:00<00:00, 1445.05it/s, Materializing param=e
Loading weights: 3%| | 13/391 [00:00<00:00, 1426.04it/s, Materializing param=e
Loading weights: 4%| | 14/391 [00:00<00:00, 1506.27it/s, Materializing param=e
Loading weights: 4%| | 14/391 [00:00<00:00, 1484.90it/s, Materializing param=e
Loading weights: 4%| | 15/391 [00:00<00:00, 1560.96it/s, Materializing param=e
Loading weights: 4%| | 15/391 [00:00<00:00, 1542.48it/s, Materializing param=e
Loading weights: 4%| | 16/391 [00:00<00:00, 1613.66it/s, Materializing param=e
Loading weights: 4%| | 16/391 [00:00<00:00, 1594.38it/s, Materializing param=e
Loading weights: 4%| | 17/391 [00:00<00:00, 1664.29it/s, Materializing param=e
Loading weights: 4%| | 17/391 [00:00<00:00, 1643.20it/s, Materializing param=e
Loading weights: 5%| | 18/391 [00:00<00:00, 1708.36it/s, Materializing param=e
Loading weights: 5%| | 18/391 [00:00<00:00, 1689.89it/s, Materializing param=e
Loading weights: 5%| | 19/391 [00:00<00:00, 1753.55it/s, Materializing param=e
Loading weights: 5%| | 19/391 [00:00<00:00, 1735.56it/s, Materializing param=e
Loading weights: 5%| | 20/391 [00:00<00:00, 1796.28it/s, Materializing param=e
Loading weights: 5%| | 20/391 [00:00<00:00, 1775.48it/s, Materializing param=e
Loading weights: 5%| | 21/391 [00:00<00:00, 1827.32it/s, Materializing param=e
Loading weights: 5%| | 21/391 [00:00<00:00, 1809.97it/s, Materializing param=e
Loading weights: 6%| | 22/391 [00:00<00:00, 1869.31it/s, Materializing param=e
Loading weights: 6%| | 22/391 [00:00<00:00, 1851.79it/s, Materializing param=e
Loading weights: 6%| | 23/391 [00:00<00:00, 1901.05it/s, Materializing param=e
Loading weights: 6%| | 23/391 [00:00<00:00, 1875.70it/s, Materializing param=e
Loading weights: 6%| | 24/391 [00:00<00:00, 1919.15it/s, Materializing param=e
Loading weights: 6%| | 24/391 [00:00<00:00, 1897.77it/s, Materializing param=e
Loading weights: 6%| | 25/391 [00:00<00:00, 1943.68it/s, Materializing param=e
Loading weights: 6%| | 25/391 [00:00<00:00, 1926.61it/s, Materializing param=e
Loading weights: 7%| | 26/391 [00:00<00:00, 1972.93it/s, Materializing param=e
Loading weights: 7%| | 26/391 [00:00<00:00, 1955.21it/s, Materializing param=e
Loading weights: 7%| | 27/391 [00:00<00:00, 1993.95it/s, Materializing param=e
Loading weights: 7%| | 27/391 [00:00<00:00, 1975.41it/s, Materializing param=e
Loading weights: 7%| | 28/391 [00:00<00:00, 2018.29it/s, Materializing param=e
Loading weights: 7%| | 28/391 [00:00<00:00, 2001.30it/s, Materializing param=e
Loading weights: 7%| | 29/391 [00:00<00:00, 2029.61it/s, Materializing param=e
Loading weights: 7%| | 29/391 [00:00<00:00, 2009.43it/s, Materializing param=e
Loading weights: 8%| | 30/391 [00:00<00:00, 2057.11it/s, Materializing param=e
Loading weights: 8%| | 30/391 [00:00<00:00, 2047.73it/s, Materializing param=e
Loading weights: 8%| | 31/391 [00:00<00:00, 2089.47it/s, Materializing param=e
Loading weights: 8%| | 31/391 [00:00<00:00, 2078.01it/s, Materializing param=e
Loading weights: 8%| | 32/391 [00:00<00:00, 2128.11it/s, Materializing param=e
Loading weights: 8%| | 32/391 [00:00<00:00, 2118.40it/s, Materializing param=e
Loading weights: 8%| | 33/391 [00:00<00:00, 2169.60it/s, Materializing param=e
Loading weights: 8%| | 33/391 [00:00<00:00, 2160.90it/s, Materializing param=e
Loading weights: 9%| | 34/391 [00:00<00:00, 2201.16it/s, Materializing param=e
Loading weights: 9%| | 34/391 [00:00<00:00, 2189.73it/s, Materializing param=e
Loading weights: 9%| | 35/391 [00:00<00:00, 2233.05it/s, Materializing param=e
Loading weights: 9%| | 35/391 [00:00<00:00, 2216.83it/s, Materializing param=e
Loading weights: 9%| | 36/391 [00:00<00:00, 2255.74it/s, Materializing param=e
Loading weights: 9%| | 36/391 [00:00<00:00, 2240.85it/s, Materializing param=e
Loading weights: 9%| | 37/391 [00:00<00:00, 2279.65it/s, Materializing param=e
Loading weights: 9%| | 37/391 [00:00<00:00, 2260.85it/s, Materializing param=e
Loading weights: 10%| | 38/391 [00:00<00:00, 2298.41it/s, Materializing param=e
Loading weights: 10%| | 38/391 [00:00<00:00, 2283.66it/s, Materializing param=e
Loading weights: 10%| | 39/391 [00:00<00:00, 2319.73it/s, Materializing param=e
Loading weights: 10%| | 39/391 [00:00<00:00, 2304.47it/s, Materializing param=e
Loading weights: 10%| | 40/391 [00:00<00:00, 2338.97it/s, Materializing param=e
Loading weights: 10%| | 40/391 [00:00<00:00, 2324.23it/s, Materializing param=e
Loading weights: 10%| | 41/391 [00:00<00:00, 2354.90it/s, Materializing param=e
Loading weights: 10%| | 41/391 [00:00<00:00, 2340.19it/s, Materializing param=e
Loading weights: 11%| | 42/391 [00:00<00:00, 2374.55it/s, Materializing param=e
Loading weights: 11%| | 42/391 [00:00<00:00, 2360.30it/s, Materializing param=e
Loading weights: 11%| | 43/391 [00:00<00:00, 2393.75it/s, Materializing param=e
Loading weights: 11%| | 43/391 [00:00<00:00, 2379.83it/s, Materializing param=e
Loading weights: 11%| | 44/391 [00:00<00:00, 2412.95it/s, Materializing param=e
Loading weights: 11%| | 44/391 [00:00<00:00, 2395.59it/s, Materializing param=e
Loading weights: 12%| | 45/391 [00:00<00:00, 2423.49it/s, Materializing param=e
Loading weights: 12%| | 45/391 [00:00<00:00, 2406.86it/s, Materializing param=e
Loading weights: 12%| | 46/391 [00:00<00:00, 2433.66it/s, Materializing param=e
Loading weights: 12%| | 46/391 [00:00<00:00, 2419.68it/s, Materializing param=e
Loading weights: 12%| | 47/391 [00:00<00:00, 2448.21it/s, Materializing param=e
Loading weights: 12%| | 47/391 [00:00<00:00, 2434.57it/s, Materializing param=e
Loading weights: 12%| | 48/391 [00:00<00:00, 2458.20it/s, Materializing param=e
Loading weights: 12%| | 48/391 [00:00<00:00, 2442.45it/s, Materializing param=e
Loading weights: 13%|▏| 49/391 [00:00<00:00, 2470.98it/s, Materializing param=e
Loading weights: 13%|▏| 49/391 [00:00<00:00, 2457.30it/s, Materializing param=e
Loading weights: 13%|▏| 50/391 [00:00<00:00, 2482.78it/s, Materializing param=e
Loading weights: 13%|▏| 50/391 [00:00<00:00, 2469.21it/s, Materializing param=e
Loading weights: 13%|▏| 51/391 [00:00<00:00, 2491.72it/s, Materializing param=e
Loading weights: 13%|▏| 51/391 [00:00<00:00, 2475.49it/s, Materializing param=e
Loading weights: 13%|▏| 52/391 [00:00<00:00, 2502.88it/s, Materializing param=e
Loading weights: 13%|▏| 52/391 [00:00<00:00, 2491.33it/s, Materializing param=e
Loading weights: 14%|▏| 53/391 [00:00<00:00, 2523.36it/s, Materializing param=e
Loading weights: 14%|▏| 53/391 [00:00<00:00, 2514.37it/s, Materializing param=e
Loading weights: 14%|▏| 54/391 [00:00<00:00, 2542.89it/s, Materializing param=e
Loading weights: 14%|▏| 54/391 [00:00<00:00, 2528.69it/s, Materializing param=e
Loading weights: 14%|▏| 55/391 [00:00<00:00, 2551.90it/s, Materializing param=e
Loading weights: 14%|▏| 55/391 [00:00<00:00, 2532.51it/s, Materializing param=e
Loading weights: 14%|▏| 56/391 [00:00<00:00, 2557.92it/s, Materializing param=e
Loading weights: 14%|▏| 56/391 [00:00<00:00, 2546.00it/s, Materializing param=e
Loading weights: 15%|▏| 57/391 [00:00<00:00, 2571.61it/s, Materializing param=e
Loading weights: 15%|▏| 57/391 [00:00<00:00, 2559.78it/s, Materializing param=e
Loading weights: 15%|▏| 58/391 [00:00<00:00, 2582.56it/s, Materializing param=e
Loading weights: 15%|▏| 58/391 [00:00<00:00, 2570.66it/s, Materializing param=e
Loading weights: 15%|▏| 59/391 [00:00<00:00, 2595.87it/s, Materializing param=e
Loading weights: 15%|▏| 59/391 [00:00<00:00, 2584.13it/s, Materializing param=e
Loading weights: 15%|▏| 60/391 [00:00<00:00, 2609.21it/s, Materializing param=e
Loading weights: 15%|▏| 60/391 [00:00<00:00, 2597.68it/s, Materializing param=e
Loading weights: 16%|▏| 61/391 [00:00<00:00, 2622.81it/s, Materializing param=e
Loading weights: 16%|▏| 61/391 [00:00<00:00, 2609.06it/s, Materializing param=e
Loading weights: 16%|▏| 62/391 [00:00<00:00, 2632.67it/s, Materializing param=e
Loading weights: 16%|▏| 62/391 [00:00<00:00, 2620.91it/s, Materializing param=e
Loading weights: 16%|▏| 63/391 [00:00<00:00, 2644.87it/s, Materializing param=e
Loading weights: 16%|▏| 63/391 [00:00<00:00, 2633.04it/s, Materializing param=e
Loading weights: 16%|▏| 64/391 [00:00<00:00, 2656.86it/s, Materializing param=e
Loading weights: 16%|▏| 64/391 [00:00<00:00, 2646.09it/s, Materializing param=e
Loading weights: 17%|▏| 65/391 [00:00<00:00, 2668.79it/s, Materializing param=e
Loading weights: 17%|▏| 65/391 [00:00<00:00, 2656.54it/s, Materializing param=e
Loading weights: 17%|▏| 66/391 [00:00<00:00, 2680.33it/s, Materializing param=e
Loading weights: 17%|▏| 66/391 [00:00<00:00, 2669.32it/s, Materializing param=e
Loading weights: 17%|▏| 67/391 [00:00<00:00, 2691.80it/s, Materializing param=e
Loading weights: 17%|▏| 67/391 [00:00<00:00, 2680.07it/s, Materializing param=e
Loading weights: 17%|▏| 68/391 [00:00<00:00, 2701.98it/s, Materializing param=e
Loading weights: 17%|▏| 68/391 [00:00<00:00, 2691.80it/s, Materializing param=e
Loading weights: 18%|▏| 69/391 [00:00<00:00, 2712.85it/s, Materializing param=e
Loading weights: 18%|▏| 69/391 [00:00<00:00, 2701.56it/s, Materializing param=e
Loading weights: 18%|▏| 70/391 [00:00<00:00, 2724.56it/s, Materializing param=e
Loading weights: 18%|▏| 70/391 [00:00<00:00, 2712.63it/s, Materializing param=e
Loading weights: 18%|▏| 71/391 [00:00<00:00, 2733.50it/s, Materializing param=e
Loading weights: 18%|▏| 71/391 [00:00<00:00, 2722.33it/s, Materializing param=e
Loading weights: 18%|▏| 72/391 [00:00<00:00, 2743.09it/s, Materializing param=e
Loading weights: 18%|▏| 72/391 [00:00<00:00, 2725.12it/s, Materializing param=e
Loading weights: 19%|▏| 73/391 [00:00<00:00, 2739.00it/s, Materializing param=e
Loading weights: 19%|▏| 73/391 [00:00<00:00, 2728.43it/s, Materializing param=e
Loading weights: 19%|▏| 74/391 [00:00<00:00, 2745.23it/s, Materializing param=e
Loading weights: 19%|▏| 74/391 [00:00<00:00, 2735.41it/s, Materializing param=e
Loading weights: 19%|▏| 75/391 [00:00<00:00, 2756.10it/s, Materializing param=e
Loading weights: 19%|▏| 75/391 [00:00<00:00, 2745.44it/s, Materializing param=e
Loading weights: 19%|▏| 76/391 [00:00<00:00, 2757.76it/s, Materializing param=e
Loading weights: 19%|▏| 76/391 [00:00<00:00, 2745.32it/s, Materializing param=e
Loading weights: 20%|▏| 77/391 [00:00<00:00, 2764.82it/s, Materializing param=e
Loading weights: 20%|▏| 77/391 [00:00<00:00, 2752.75it/s, Materializing param=e
Loading weights: 20%|▏| 78/391 [00:00<00:00, 2771.50it/s, Materializing param=e
Loading weights: 20%|▏| 78/391 [00:00<00:00, 2761.48it/s, Materializing param=e
Loading weights: 20%|▏| 79/391 [00:00<00:00, 2778.55it/s, Materializing param=e
Loading weights: 20%|▏| 79/391 [00:00<00:00, 2768.98it/s, Materializing param=e
Loading weights: 20%|▏| 80/391 [00:00<00:00, 2788.53it/s, Materializing param=e
Loading weights: 20%|▏| 80/391 [00:00<00:00, 2778.44it/s, Materializing param=e
Loading weights: 21%|▏| 81/391 [00:00<00:00, 2798.25it/s, Materializing param=e
Loading weights: 21%|▏| 81/391 [00:00<00:00, 2791.19it/s, Materializing param=e
Loading weights: 21%|▏| 82/391 [00:00<00:00, 2814.62it/s, Materializing param=e
Loading weights: 21%|▏| 82/391 [00:00<00:00, 2806.93it/s, Materializing param=e
Loading weights: 21%|▏| 83/391 [00:00<00:00, 2818.16it/s, Materializing param=e
Loading weights: 21%|▏| 83/391 [00:00<00:00, 2810.02it/s, Materializing param=e
Loading weights: 21%|▏| 84/391 [00:00<00:00, 2831.51it/s, Materializing param=e
Loading weights: 21%|▏| 84/391 [00:00<00:00, 2824.45it/s, Materializing param=e
Loading weights: 22%|▏| 85/391 [00:00<00:00, 2847.46it/s, Materializing param=e
Loading weights: 22%|▏| 85/391 [00:00<00:00, 2840.56it/s, Materializing param=e
Loading weights: 22%|▏| 86/391 [00:00<00:00, 2863.01it/s, Materializing param=e
Loading weights: 22%|▏| 86/391 [00:00<00:00, 2856.27it/s, Materializing param=e
Loading weights: 22%|▏| 87/391 [00:00<00:00, 2878.66it/s, Materializing param=e
Loading weights: 22%|▏| 87/391 [00:00<00:00, 2872.13it/s, Materializing param=e
Loading weights: 23%|▏| 88/391 [00:00<00:00, 2886.47it/s, Materializing param=e
Loading weights: 23%|▏| 88/391 [00:00<00:00, 2879.27it/s, Materializing param=e
Loading weights: 23%|▏| 89/391 [00:00<00:00, 2901.26it/s, Materializing param=e
Loading weights: 23%|▏| 89/391 [00:00<00:00, 2892.98it/s, Materializing param=e
Loading weights: 23%|▏| 90/391 [00:00<00:00, 2909.55it/s, Materializing param=e
Loading weights: 23%|▏| 90/391 [00:00<00:00, 2900.07it/s, Materializing param=e
Loading weights: 23%|▏| 91/391 [00:00<00:00, 2917.48it/s, Materializing param=e
Loading weights: 23%|▏| 91/391 [00:00<00:00, 2909.62it/s, Materializing param=e
Loading weights: 24%|▏| 92/391 [00:00<00:00, 2927.14it/s, Materializing param=e
Loading weights: 24%|▏| 92/391 [00:00<00:00, 2919.90it/s, Materializing param=e
Loading weights: 24%|▏| 93/391 [00:00<00:00, 2933.30it/s, Materializing param=e
Loading weights: 24%|▏| 93/391 [00:00<00:00, 2922.82it/s, Materializing param=e
Loading weights: 24%|▏| 94/391 [00:00<00:00, 2936.12it/s, Materializing param=e
Loading weights: 24%|▏| 94/391 [00:00<00:00, 2923.75it/s, Materializing param=e
Loading weights: 24%|▏| 95/391 [00:00<00:00, 2935.87it/s, Materializing param=e
Loading weights: 24%|▏| 95/391 [00:00<00:00, 2925.65it/s, Materializing param=e
Loading weights: 25%|▏| 96/391 [00:00<00:00, 2940.51it/s, Materializing param=e
Loading weights: 25%|▏| 96/391 [00:00<00:00, 2929.20it/s, Materializing param=e
Loading weights: 25%|▏| 97/391 [00:00<00:00, 2942.18it/s, Materializing param=e
Loading weights: 25%|▏| 97/391 [00:00<00:00, 2931.81it/s, Materializing param=e
Loading weights: 25%|▎| 98/391 [00:00<00:00, 2945.99it/s, Materializing param=e
Loading weights: 25%|▎| 98/391 [00:00<00:00, 2933.67it/s, Materializing param=e
Loading weights: 25%|▎| 99/391 [00:00<00:00, 2948.11it/s, Materializing param=e
Loading weights: 25%|▎| 99/391 [00:00<00:00, 2938.60it/s, Materializing param=e
Loading weights: 26%|▎| 100/391 [00:00<00:00, 2949.89it/s, Materializing param=
Loading weights: 26%|▎| 100/391 [00:00<00:00, 2938.94it/s, Materializing param=
Loading weights: 26%|▎| 101/391 [00:00<00:00, 2951.55it/s, Materializing param=
Loading weights: 26%|▎| 101/391 [00:00<00:00, 2941.02it/s, Materializing param=
Loading weights: 26%|▎| 102/391 [00:00<00:00, 2955.25it/s, Materializing param=
Loading weights: 26%|▎| 102/391 [00:00<00:00, 2945.09it/s, Materializing param=
Loading weights: 26%|▎| 103/391 [00:00<00:00, 2959.04it/s, Materializing param=
Loading weights: 26%|▎| 103/391 [00:00<00:00, 2949.22it/s, Materializing param=
Loading weights: 27%|▎| 104/391 [00:00<00:00, 2962.86it/s, Materializing param=
Loading weights: 27%|▎| 104/391 [00:00<00:00, 2953.60it/s, Materializing param=
Loading weights: 27%|▎| 105/391 [00:00<00:00, 2965.29it/s, Materializing param=
Loading weights: 27%|▎| 105/391 [00:00<00:00, 2955.70it/s, Materializing param=
Loading weights: 27%|▎| 106/391 [00:00<00:00, 2969.00it/s, Materializing param=
Loading weights: 27%|▎| 106/391 [00:00<00:00, 2959.83it/s, Materializing param=
Loading weights: 27%|▎| 107/391 [00:00<00:00, 2973.70it/s, Materializing param=
Loading weights: 27%|▎| 107/391 [00:00<00:00, 2964.88it/s, Materializing param=
Loading weights: 28%|▎| 108/391 [00:00<00:00, 2978.73it/s, Materializing param=
Loading weights: 28%|▎| 108/391 [00:00<00:00, 2967.43it/s, Materializing param=
Loading weights: 28%|▎| 109/391 [00:00<00:00, 2980.62it/s, Materializing param=
Loading weights: 28%|▎| 109/391 [00:00<00:00, 2971.61it/s, Materializing param=
Loading weights: 28%|▎| 110/391 [00:00<00:00, 2984.96it/s, Materializing param=
Loading weights: 28%|▎| 110/391 [00:00<00:00, 2975.36it/s, Materializing param=
Loading weights: 28%|▎| 111/391 [00:00<00:00, 2988.55it/s, Materializing param=
Loading weights: 28%|▎| 111/391 [00:00<00:00, 2979.81it/s, Materializing param=
Loading weights: 29%|▎| 112/391 [00:00<00:00, 2990.04it/s, Materializing param=
Loading weights: 29%|▎| 112/391 [00:00<00:00, 2981.37it/s, Materializing param=
Loading weights: 29%|▎| 113/391 [00:00<00:00, 2994.44it/s, Materializing param=
Loading weights: 29%|▎| 113/391 [00:00<00:00, 2985.72it/s, Materializing param=
Loading weights: 29%|▎| 114/391 [00:00<00:00, 2998.17it/s, Materializing param=
Loading weights: 29%|▎| 114/391 [00:00<00:00, 2989.88it/s, Materializing param=
Loading weights: 29%|▎| 115/391 [00:00<00:00, 2996.96it/s, Materializing param=
Loading weights: 29%|▎| 115/391 [00:00<00:00, 2988.51it/s, Materializing param=
Loading weights: 30%|▎| 116/391 [00:00<00:00, 3001.12it/s, Materializing param=
Loading weights: 30%|▎| 116/391 [00:00<00:00, 2992.82it/s, Materializing param=
Loading weights: 30%|▎| 117/391 [00:00<00:00, 3005.53it/s, Materializing param=
Loading weights: 30%|▎| 117/391 [00:00<00:00, 2997.63it/s, Materializing param=
Loading weights: 30%|▎| 118/391 [00:00<00:00, 3010.49it/s, Materializing param=
Loading weights: 30%|▎| 118/391 [00:00<00:00, 3002.42it/s, Materializing param=
Loading weights: 30%|▎| 119/391 [00:00<00:00, 3009.95it/s, Materializing param=
Loading weights: 30%|▎| 119/391 [00:00<00:00, 3000.25it/s, Materializing param=
Loading weights: 31%|▎| 120/391 [00:00<00:00, 3011.87it/s, Materializing param=
Loading weights: 31%|▎| 120/391 [00:00<00:00, 3003.46it/s, Materializing param=
Loading weights: 31%|▎| 121/391 [00:00<00:00, 3015.57it/s, Materializing param=
Loading weights: 31%|▎| 121/391 [00:00<00:00, 3004.96it/s, Materializing param=
Loading weights: 31%|▎| 122/391 [00:00<00:00, 3014.25it/s, Materializing param=
Loading weights: 31%|▎| 122/391 [00:00<00:00, 3006.16it/s, Materializing param=
Loading weights: 31%|▎| 123/391 [00:00<00:00, 3018.10it/s, Materializing param=
Loading weights: 31%|▎| 123/391 [00:00<00:00, 3009.58it/s, Materializing param=
Loading weights: 32%|▎| 124/391 [00:00<00:00, 3023.13it/s, Materializing param=
Loading weights: 32%|▎| 124/391 [00:00<00:00, 3017.87it/s, Materializing param=
Loading weights: 32%|▎| 125/391 [00:00<00:00, 3033.62it/s, Materializing param=
Loading weights: 32%|▎| 125/391 [00:00<00:00, 3027.87it/s, Materializing param=
Loading weights: 32%|▎| 126/391 [00:00<00:00, 3038.56it/s, Materializing param=
Loading weights: 32%|▎| 126/391 [00:00<00:00, 3032.71it/s, Materializing param=
Loading weights: 32%|▎| 127/391 [00:00<00:00, 3048.20it/s, Materializing param=
Loading weights: 32%|▎| 127/391 [00:00<00:00, 3042.82it/s, Materializing param=
Loading weights: 33%|▎| 128/391 [00:00<00:00, 3058.47it/s, Materializing param=
Loading weights: 33%|▎| 128/391 [00:00<00:00, 3053.13it/s, Materializing param=
Loading weights: 33%|▎| 129/391 [00:00<00:00, 3067.59it/s, Materializing param=
Loading weights: 33%|▎| 129/391 [00:00<00:00, 3062.14it/s, Materializing param=
Loading weights: 33%|▎| 130/391 [00:00<00:00, 3077.57it/s, Materializing param=
Loading weights: 33%|▎| 130/391 [00:00<00:00, 3071.95it/s, Materializing param=
Loading weights: 34%|▎| 131/391 [00:00<00:00, 3080.90it/s, Materializing param=
Loading weights: 34%|▎| 131/391 [00:00<00:00, 3072.94it/s, Materializing param=
Loading weights: 34%|▎| 132/391 [00:00<00:00, 3084.24it/s, Materializing param=
Loading weights: 34%|▎| 132/391 [00:00<00:00, 3076.37it/s, Materializing param=
Loading weights: 34%|▎| 133/391 [00:00<00:00, 3086.90it/s, Materializing param=
Loading weights: 34%|▎| 133/391 [00:00<00:00, 3079.38it/s, Materializing param=
Loading weights: 34%|▎| 134/391 [00:00<00:00, 3087.94it/s, Materializing param=
Loading weights: 34%|▎| 134/391 [00:00<00:00, 3081.56it/s, Materializing param=
Loading weights: 35%|▎| 135/391 [00:00<00:00, 3095.28it/s, Materializing param=
Loading weights: 35%|▎| 135/391 [00:00<00:00, 3089.77it/s, Materializing param=
Loading weights: 35%|▎| 136/391 [00:00<00:00, 3104.30it/s, Materializing param=
Loading weights: 35%|▎| 136/391 [00:00<00:00, 3098.82it/s, Materializing param=
Loading weights: 35%|▎| 137/391 [00:00<00:00, 3110.81it/s, Materializing param=
Loading weights: 35%|▎| 137/391 [00:00<00:00, 3102.50it/s, Materializing param=
Loading weights: 35%|▎| 138/391 [00:00<00:00, 3110.40it/s, Materializing param=
Loading weights: 35%|▎| 138/391 [00:00<00:00, 3102.84it/s, Materializing param=
Loading weights: 36%|▎| 139/391 [00:00<00:00, 3113.21it/s, Materializing param=
Loading weights: 36%|▎| 139/391 [00:00<00:00, 3105.65it/s, Materializing param=
Loading weights: 36%|▎| 140/391 [00:00<00:00, 3116.29it/s, Materializing param=
Loading weights: 36%|▎| 140/391 [00:00<00:00, 3108.83it/s, Materializing param=
Loading weights: 36%|▎| 141/391 [00:00<00:00, 3119.12it/s, Materializing param=
Loading weights: 36%|▎| 141/391 [00:00<00:00, 3111.71it/s, Materializing param=
Loading weights: 36%|▎| 142/391 [00:00<00:00, 3120.06it/s, Materializing param=
Loading weights: 36%|▎| 142/391 [00:00<00:00, 3112.56it/s, Materializing param=
Loading weights: 37%|▎| 143/391 [00:00<00:00, 3123.10it/s, Materializing param=
Loading weights: 37%|▎| 143/391 [00:00<00:00, 3115.66it/s, Materializing param=
Loading weights: 37%|▎| 144/391 [00:00<00:00, 3126.11it/s, Materializing param=
Loading weights: 37%|▎| 144/391 [00:00<00:00, 3118.36it/s, Materializing param=
Loading weights: 37%|▎| 145/391 [00:00<00:00, 3124.30it/s, Materializing param=
Loading weights: 37%|▎| 145/391 [00:00<00:00, 3118.68it/s, Materializing param=
Loading weights: 37%|▎| 146/391 [00:00<00:00, 3128.46it/s, Materializing param=
Loading weights: 37%|▎| 146/391 [00:00<00:00, 3121.53it/s, Materializing param=
Loading weights: 38%|▍| 147/391 [00:00<00:00, 3131.11it/s, Materializing param=
Loading weights: 38%|▍| 147/391 [00:00<00:00, 3123.56it/s, Materializing param=
Loading weights: 38%|▍| 148/391 [00:00<00:00, 3133.33it/s, Materializing param=
Loading weights: 38%|▍| 148/391 [00:00<00:00, 3124.19it/s, Materializing param=
Loading weights: 38%|▍| 149/391 [00:00<00:00, 3133.23it/s, Materializing param=
Loading weights: 38%|▍| 149/391 [00:00<00:00, 3126.02it/s, Materializing param=
Loading weights: 38%|▍| 150/391 [00:00<00:00, 3136.15it/s, Materializing param=
Loading weights: 38%|▍| 150/391 [00:00<00:00, 3127.11it/s, Materializing param=
Loading weights: 39%|▍| 151/391 [00:00<00:00, 3135.69it/s, Materializing param=
Loading weights: 39%|▍| 151/391 [00:00<00:00, 3128.55it/s, Materializing param=
Loading weights: 39%|▍| 152/391 [00:00<00:00, 3136.39it/s, Materializing param=
Loading weights: 39%|▍| 152/391 [00:00<00:00, 3129.34it/s, Materializing param=
Loading weights: 39%|▍| 153/391 [00:00<00:00, 3138.88it/s, Materializing param=
Loading weights: 39%|▍| 153/391 [00:00<00:00, 3131.85it/s, Materializing param=
Loading weights: 39%|▍| 154/391 [00:00<00:00, 3141.59it/s, Materializing param=
Loading weights: 39%|▍| 154/391 [00:00<00:00, 3134.51it/s, Materializing param=
Loading weights: 40%|▍| 155/391 [00:00<00:00, 3140.91it/s, Materializing param=
Loading weights: 40%|▍| 155/391 [00:00<00:00, 3133.12it/s, Materializing param=
Loading weights: 40%|▍| 156/391 [00:00<00:00, 3142.21it/s, Materializing param=
Loading weights: 40%|▍| 156/391 [00:00<00:00, 3135.37it/s, Materializing param=
Loading weights: 40%|▍| 157/391 [00:00<00:00, 3142.31it/s, Materializing param=
Loading weights: 40%|▍| 157/391 [00:00<00:00, 3135.43it/s, Materializing param=
Loading weights: 40%|▍| 158/391 [00:00<00:00, 3143.14it/s, Materializing param=
Loading weights: 40%|▍| 158/391 [00:00<00:00, 3134.33it/s, Materializing param=
Loading weights: 41%|▍| 159/391 [00:00<00:00, 3142.20it/s, Materializing param=
Loading weights: 41%|▍| 159/391 [00:00<00:00, 3132.49it/s, Materializing param=
Loading weights: 41%|▍| 160/391 [00:00<00:00, 3139.49it/s, Materializing param=
Loading weights: 41%|▍| 160/391 [00:00<00:00, 3132.80it/s, Materializing param=
Loading weights: 41%|▍| 161/391 [00:00<00:00, 3142.02it/s, Materializing param=
Loading weights: 41%|▍| 161/391 [00:00<00:00, 3133.11it/s, Materializing param=
Loading weights: 41%|▍| 162/391 [00:00<00:00, 3141.67it/s, Materializing param=
Loading weights: 41%|▍| 162/391 [00:00<00:00, 3135.16it/s, Materializing param=
Loading weights: 42%|▍| 163/391 [00:00<00:00, 3144.39it/s, Materializing param=
Loading weights: 42%|▍| 163/391 [00:00<00:00, 3138.45it/s, Materializing param=
Loading weights: 42%|▍| 164/391 [00:00<00:00, 3147.34it/s, Materializing param=
Loading weights: 42%|▍| 164/391 [00:00<00:00, 3140.84it/s, Materializing param=
Loading weights: 42%|▍| 165/391 [00:00<00:00, 3148.85it/s, Materializing param=
Loading weights: 42%|▍| 165/391 [00:00<00:00, 3141.84it/s, Materializing param=
Loading weights: 42%|▍| 166/391 [00:00<00:00, 3150.03it/s, Materializing param=
Loading weights: 42%|▍| 166/391 [00:00<00:00, 3144.48it/s, Materializing param=
Loading weights: 43%|▍| 167/391 [00:00<00:00, 3154.55it/s, Materializing param=
Loading weights: 43%|▍| 167/391 [00:00<00:00, 3149.95it/s, Materializing param=
Loading weights: 43%|▍| 168/391 [00:00<00:00, 3161.48it/s, Materializing param=
Loading weights: 43%|▍| 168/391 [00:00<00:00, 3156.30it/s, Materializing param=
Loading weights: 43%|▍| 169/391 [00:00<00:00, 3163.89it/s, Materializing param=
Loading weights: 43%|▍| 169/391 [00:00<00:00, 3159.16it/s, Materializing param=
Loading weights: 43%|▍| 170/391 [00:00<00:00, 3170.89it/s, Materializing param=
Loading weights: 43%|▍| 170/391 [00:00<00:00, 3166.67it/s, Materializing param=
Loading weights: 44%|▍| 171/391 [00:00<00:00, 3178.69it/s, Materializing param=
Loading weights: 44%|▍| 171/391 [00:00<00:00, 3174.44it/s, Materializing param=
Loading weights: 44%|▍| 172/391 [00:00<00:00, 3186.47it/s, Materializing param=
Loading weights: 44%|▍| 172/391 [00:00<00:00, 3182.32it/s, Materializing param=
Loading weights: 44%|▍| 173/391 [00:00<00:00, 3193.14it/s, Materializing param=
Loading weights: 44%|▍| 173/391 [00:00<00:00, 3188.98it/s, Materializing param=
Loading weights: 45%|▍| 174/391 [00:00<00:00, 3199.19it/s, Materializing param=
Loading weights: 45%|▍| 174/391 [00:00<00:00, 3194.75it/s, Materializing param=
Loading weights: 45%|▍| 175/391 [00:00<00:00, 3206.15it/s, Materializing param=
Loading weights: 45%|▍| 175/391 [00:00<00:00, 3201.44it/s, Materializing param=
Loading weights: 45%|▍| 176/391 [00:00<00:00, 3209.29it/s, Materializing param=
Loading weights: 45%|▍| 176/391 [00:00<00:00, 3202.54it/s, Materializing param=
Loading weights: 45%|▍| 177/391 [00:00<00:00, 3212.15it/s, Materializing param=
Loading weights: 45%|▍| 177/391 [00:00<00:00, 3207.98it/s, Materializing param=
Loading weights: 46%|▍| 178/391 [00:00<00:00, 3219.74it/s, Materializing param=
Loading weights: 46%|▍| 178/391 [00:00<00:00, 3213.02it/s, Materializing param=
Loading weights: 46%|▍| 179/391 [00:00<00:00, 3223.51it/s, Materializing param=
Loading weights: 46%|▍| 179/391 [00:00<00:00, 3219.51it/s, Materializing param=
Loading weights: 46%|▍| 180/391 [00:00<00:00, 3225.82it/s, Materializing param=
Loading weights: 46%|▍| 180/391 [00:00<00:00, 3220.48it/s, Materializing param=
Loading weights: 46%|▍| 181/391 [00:00<00:00, 3230.53it/s, Materializing param=
Loading weights: 46%|▍| 181/391 [00:00<00:00, 3226.25it/s, Materializing param=
Loading weights: 47%|▍| 182/391 [00:00<00:00, 3234.74it/s, Materializing param=
Loading weights: 47%|▍| 182/391 [00:00<00:00, 3228.04it/s, Materializing param=
Loading weights: 47%|▍| 183/391 [00:00<00:00, 3235.60it/s, Materializing param=
Loading weights: 47%|▍| 183/391 [00:00<00:00, 3229.54it/s, Materializing param=
Loading weights: 47%|▍| 184/391 [00:00<00:00, 3236.41it/s, Materializing param=
Loading weights: 47%|▍| 184/391 [00:00<00:00, 3228.74it/s, Materializing param=
Loading weights: 47%|▍| 185/391 [00:00<00:00, 3234.38it/s, Materializing param=
Loading weights: 47%|▍| 185/391 [00:00<00:00, 3225.73it/s, Materializing param=
Loading weights: 48%|▍| 186/391 [00:00<00:00, 3230.61it/s, Materializing param=
Loading weights: 48%|▍| 186/391 [00:00<00:00, 3223.47it/s, Materializing param=
Loading weights: 48%|▍| 187/391 [00:00<00:00, 3231.28it/s, Materializing param=
Loading weights: 48%|▍| 187/391 [00:00<00:00, 3227.10it/s, Materializing param=
Loading weights: 48%|▍| 188/391 [00:00<00:00, 3237.69it/s, Materializing param=
Loading weights: 48%|▍| 188/391 [00:00<00:00, 3233.69it/s, Materializing param=
Loading weights: 48%|▍| 189/391 [00:00<00:00, 3244.45it/s, Materializing param=
Loading weights: 48%|▍| 189/391 [00:00<00:00, 3238.98it/s, Materializing param=
Loading weights: 49%|▍| 190/391 [00:00<00:00, 3247.11it/s, Materializing param=
Loading weights: 49%|▍| 190/391 [00:00<00:00, 3242.77it/s, Materializing param=
Loading weights: 49%|▍| 191/391 [00:00<00:00, 3248.54it/s, Materializing param=
Loading weights: 49%|▍| 191/391 [00:00<00:00, 3240.13it/s, Materializing param=
Loading weights: 49%|▍| 192/391 [00:00<00:00, 3242.65it/s, Materializing param=
Loading weights: 49%|▍| 192/391 [00:00<00:00, 3231.71it/s, Materializing param=
Loading weights: 49%|▍| 193/391 [00:00<00:00, 3231.41it/s, Materializing param=
Loading weights: 49%|▍| 193/391 [00:00<00:00, 3222.16it/s, Materializing param=
Loading weights: 50%|▍| 194/391 [00:00<00:00, 3227.69it/s, Materializing param=
Loading weights: 50%|▍| 194/391 [00:00<00:00, 3221.89it/s, Materializing param=
Loading weights: 50%|▍| 195/391 [00:00<00:00, 3224.43it/s, Materializing param=
Loading weights: 50%|▍| 195/391 [00:00<00:00, 3217.34it/s, Materializing param=
Loading weights: 50%|▌| 196/391 [00:00<00:00, 3218.96it/s, Materializing param=
Loading weights: 50%|▌| 196/391 [00:00<00:00, 3208.33it/s, Materializing param=
Loading weights: 50%|▌| 197/391 [00:00<00:00, 3209.36it/s, Materializing param=
Loading weights: 50%|▌| 197/391 [00:00<00:00, 3199.34it/s, Materializing param=
Loading weights: 51%|▌| 198/391 [00:00<00:00, 3202.24it/s, Materializing param=
Loading weights: 51%|▌| 198/391 [00:00<00:00, 3192.58it/s, Materializing param=
Loading weights: 51%|▌| 199/391 [00:00<00:00, 3198.34it/s, Materializing param=
Loading weights: 51%|▌| 199/391 [00:00<00:00, 3191.45it/s, Materializing param=
Loading weights: 51%|▌| 200/391 [00:00<00:00, 3196.73it/s, Materializing param=
Loading weights: 51%|▌| 200/391 [00:00<00:00, 3191.04it/s, Materializing param=
Loading weights: 51%|▌| 201/391 [00:00<00:00, 3196.77it/s, Materializing param=
Loading weights: 51%|▌| 201/391 [00:00<00:00, 3189.67it/s, Materializing param=
Loading weights: 52%|▌| 202/391 [00:00<00:00, 3194.43it/s, Materializing param=
Loading weights: 52%|▌| 202/391 [00:00<00:00, 3186.37it/s, Materializing param=
Loading weights: 52%|▌| 203/391 [00:00<00:00, 3191.45it/s, Materializing param=
Loading weights: 52%|▌| 203/391 [00:00<00:00, 3183.36it/s, Materializing param=
Loading weights: 52%|▌| 204/391 [00:00<00:00, 3186.37it/s, Materializing param=
Loading weights: 52%|▌| 204/391 [00:00<00:00, 3179.66it/s, Materializing param=
Loading weights: 52%|▌| 205/391 [00:00<00:00, 3184.23it/s, Materializing param=
Loading weights: 52%|▌| 205/391 [00:00<00:00, 3178.31it/s, Materializing param=
Loading weights: 53%|▌| 206/391 [00:00<00:00, 3183.98it/s, Materializing param=
Loading weights: 53%|▌| 206/391 [00:00<00:00, 3177.90it/s, Materializing param=
Loading weights: 53%|▌| 207/391 [00:00<00:00, 3181.53it/s, Materializing param=
Loading weights: 53%|▌| 207/391 [00:00<00:00, 3174.99it/s, Materializing param=
Loading weights: 53%|▌| 208/391 [00:00<00:00, 3179.81it/s, Materializing param=
Loading weights: 53%|▌| 208/391 [00:00<00:00, 3173.43it/s, Materializing param=
Loading weights: 53%|▌| 209/391 [00:00<00:00, 3178.13it/s, Materializing param=
Loading weights: 53%|▌| 209/391 [00:00<00:00, 3172.35it/s, Materializing param=
Loading weights: 54%|▌| 210/391 [00:00<00:00, 3169.97it/s, Materializing param=
Loading weights: 54%|▌| 210/391 [00:00<00:00, 3156.45it/s, Materializing param=
Loading weights: 54%|▌| 211/391 [00:00<00:00, 3161.72it/s, Materializing param=
Loading weights: 54%|▌| 211/391 [00:00<00:00, 3155.71it/s, Materializing param=
Loading weights: 54%|▌| 212/391 [00:00<00:00, 3163.02it/s, Materializing param=
Loading weights: 54%|▌| 212/391 [00:00<00:00, 3157.67it/s, Materializing param=
Loading weights: 54%|▌| 213/391 [00:00<00:00, 3164.10it/s, Materializing param=
Loading weights: 54%|▌| 213/391 [00:00<00:00, 3157.25it/s, Materializing param=
Loading weights: 55%|▌| 214/391 [00:00<00:00, 3163.76it/s, Materializing param=
Loading weights: 55%|▌| 214/391 [00:00<00:00, 3158.72it/s, Materializing param=
Loading weights: 55%|▌| 215/391 [00:00<00:00, 3163.27it/s, Materializing param=
Loading weights: 55%|▌| 215/391 [00:00<00:00, 3158.05it/s, Materializing param=
Loading weights: 55%|▌| 216/391 [00:00<00:00, 3164.29it/s, Materializing param=
Loading weights: 55%|▌| 216/391 [00:00<00:00, 3157.35it/s, Materializing param=
Loading weights: 55%|▌| 217/391 [00:00<00:00, 3163.47it/s, Materializing param=
Loading weights: 55%|▌| 217/391 [00:00<00:00, 3159.50it/s, Materializing param=
Loading weights: 56%|▌| 218/391 [00:00<00:00, 3166.10it/s, Materializing param=
Loading weights: 56%|▌| 218/391 [00:00<00:00, 3160.60it/s, Materializing param=
Loading weights: 56%|▌| 219/391 [00:00<00:00, 3166.87it/s, Materializing param=
Loading weights: 56%|▌| 219/391 [00:00<00:00, 3161.47it/s, Materializing param=
Loading weights: 56%|▌| 220/391 [00:00<00:00, 3166.21it/s, Materializing param=
Loading weights: 56%|▌| 220/391 [00:00<00:00, 3162.08it/s, Materializing param=
Loading weights: 57%|▌| 221/391 [00:00<00:00, 3168.28it/s, Materializing param=
Loading weights: 57%|▌| 221/391 [00:00<00:00, 3163.01it/s, Materializing param=
Loading weights: 57%|▌| 222/391 [00:00<00:00, 3169.69it/s, Materializing param=
Loading weights: 57%|▌| 222/391 [00:00<00:00, 3164.44it/s, Materializing param=
Loading weights: 57%|▌| 223/391 [00:00<00:00, 3169.98it/s, Materializing param=
Loading weights: 57%|▌| 223/391 [00:00<00:00, 3164.23it/s, Materializing param=
Loading weights: 57%|▌| 224/391 [00:00<00:00, 3169.06it/s, Materializing param=
Loading weights: 57%|▌| 224/391 [00:00<00:00, 3162.80it/s, Materializing param=
Loading weights: 58%|▌| 225/391 [00:00<00:00, 3162.84it/s, Materializing param=
Loading weights: 58%|▌| 225/391 [00:00<00:00, 3148.83it/s, Materializing param=
Loading weights: 58%|▌| 226/391 [00:00<00:00, 3134.13it/s, Materializing param=
Loading weights: 58%|▌| 226/391 [00:00<00:00, 3127.40it/s, Materializing param=
Loading weights: 58%|▌| 227/391 [00:00<00:00, 3122.34it/s, Materializing param=
Loading weights: 58%|▌| 227/391 [00:00<00:00, 3114.12it/s, Materializing param=
Loading weights: 58%|▌| 228/391 [00:00<00:00, 3112.16it/s, Materializing param=
Loading weights: 58%|▌| 228/391 [00:00<00:00, 3095.93it/s, Materializing param=
Loading weights: 59%|▌| 229/391 [00:00<00:00, 3100.83it/s, Materializing param=
Loading weights: 59%|▌| 229/391 [00:00<00:00, 3094.63it/s, Materializing param=
Loading weights: 59%|▌| 230/391 [00:00<00:00, 3098.64it/s, Materializing param=
Loading weights: 59%|▌| 230/391 [00:00<00:00, 3092.52it/s, Materializing param=
Loading weights: 59%|▌| 231/391 [00:00<00:00, 3098.18it/s, Materializing param=
Loading weights: 59%|▌| 231/391 [00:00<00:00, 3093.34it/s, Materializing param=
Loading weights: 59%|▌| 232/391 [00:00<00:00, 3100.21it/s, Materializing param=
Loading weights: 59%|▌| 232/391 [00:00<00:00, 3094.96it/s, Materializing param=
Loading weights: 60%|▌| 233/391 [00:00<00:00, 3099.25it/s, Materializing param=
Loading weights: 60%|▌| 233/391 [00:00<00:00, 3092.19it/s, Materializing param=
Loading weights: 60%|▌| 234/391 [00:00<00:00, 3092.79it/s, Materializing param=
Loading weights: 60%|▌| 234/391 [00:00<00:00, 3086.44it/s, Materializing param=
Loading weights: 60%|▌| 235/391 [00:00<00:00, 3088.52it/s, Materializing param=
Loading weights: 60%|▌| 235/391 [00:00<00:00, 3082.68it/s, Materializing param=
Loading weights: 60%|▌| 236/391 [00:00<00:00, 3084.87it/s, Materializing param=
Loading weights: 60%|▌| 236/391 [00:00<00:00, 3079.79it/s, Materializing param=
Loading weights: 61%|▌| 237/391 [00:00<00:00, 3085.81it/s, Materializing param=
Loading weights: 61%|▌| 237/391 [00:00<00:00, 3081.29it/s, Materializing param=
Loading weights: 61%|▌| 238/391 [00:00<00:00, 3087.22it/s, Materializing param=
Loading weights: 61%|▌| 238/391 [00:00<00:00, 3083.78it/s, Materializing param=
Loading weights: 61%|▌| 239/391 [00:00<00:00, 3086.33it/s, Materializing param=
Loading weights: 61%|▌| 239/391 [00:00<00:00, 3082.28it/s, Materializing param=
Loading weights: 61%|▌| 240/391 [00:00<00:00, 3087.40it/s, Materializing param=
Loading weights: 61%|▌| 240/391 [00:00<00:00, 3082.47it/s, Materializing param=
Loading weights: 62%|▌| 241/391 [00:00<00:00, 3087.69it/s, Materializing param=
Loading weights: 62%|▌| 241/391 [00:00<00:00, 3082.21it/s, Materializing param=
Loading weights: 62%|▌| 242/391 [00:00<00:00, 3086.66it/s, Materializing param=
Loading weights: 62%|▌| 242/391 [00:00<00:00, 3081.23it/s, Materializing param=
Loading weights: 62%|▌| 243/391 [00:00<00:00, 3085.77it/s, Materializing param=
Loading weights: 62%|▌| 243/391 [00:00<00:00, 3081.87it/s, Materializing param=
Loading weights: 62%|▌| 244/391 [00:00<00:00, 3087.06it/s, Materializing param=
Loading weights: 62%|▌| 244/391 [00:00<00:00, 3081.92it/s, Materializing param=
Loading weights: 63%|▋| 245/391 [00:00<00:00, 3085.23it/s, Materializing param=
Loading weights: 63%|▋| 245/391 [00:00<00:00, 3081.63it/s, Materializing param=
Loading weights: 63%|▋| 246/391 [00:00<00:00, 3087.70it/s, Materializing param=
Loading weights: 63%|▋| 246/391 [00:00<00:00, 3084.45it/s, Materializing param=
Loading weights: 63%|▋| 247/391 [00:00<00:00, 3089.55it/s, Materializing param=
Loading weights: 63%|▋| 247/391 [00:00<00:00, 3085.14it/s, Materializing param=
Loading weights: 63%|▋| 248/391 [00:00<00:00, 3091.66it/s, Materializing param=
Loading weights: 63%|▋| 248/391 [00:00<00:00, 3086.33it/s, Materializing param=
Loading weights: 64%|▋| 249/391 [00:00<00:00, 3091.44it/s, Materializing param=
Loading weights: 64%|▋| 249/391 [00:00<00:00, 3087.67it/s, Materializing param=
Loading weights: 64%|▋| 250/391 [00:00<00:00, 3094.57it/s, Materializing param=
Loading weights: 64%|▋| 250/391 [00:00<00:00, 3092.04it/s, Materializing param=
Loading weights: 64%|▋| 251/391 [00:00<00:00, 3099.20it/s, Materializing param=
Loading weights: 64%|▋| 251/391 [00:00<00:00, 3095.38it/s, Materializing param=
Loading weights: 64%|▋| 252/391 [00:00<00:00, 3101.81it/s, Materializing param=
Loading weights: 64%|▋| 252/391 [00:00<00:00, 3098.01it/s, Materializing param=
Loading weights: 65%|▋| 253/391 [00:00<00:00, 3103.45it/s, Materializing param=
Loading weights: 65%|▋| 253/391 [00:00<00:00, 3099.05it/s, Materializing param=
Loading weights: 65%|▋| 254/391 [00:00<00:00, 3104.85it/s, Materializing param=
Loading weights: 65%|▋| 254/391 [00:00<00:00, 3100.48it/s, Materializing param=
Loading weights: 65%|▋| 255/391 [00:00<00:00, 3106.77it/s, Materializing param=
Loading weights: 65%|▋| 255/391 [00:00<00:00, 3103.19it/s, Materializing param=
Loading weights: 65%|▋| 256/391 [00:00<00:00, 3108.70it/s, Materializing param=
Loading weights: 65%|▋| 256/391 [00:00<00:00, 3103.97it/s, Materializing param=
Loading weights: 66%|▋| 257/391 [00:00<00:00, 3109.88it/s, Materializing param=
Loading weights: 66%|▋| 257/391 [00:00<00:00, 3105.56it/s, Materializing param=
Loading weights: 66%|▋| 258/391 [00:00<00:00, 3111.93it/s, Materializing param=
Loading weights: 66%|▋| 258/391 [00:00<00:00, 3108.31it/s, Materializing param=
Loading weights: 66%|▋| 259/391 [00:00<00:00, 3113.92it/s, Materializing param=
Loading weights: 66%|▋| 259/391 [00:00<00:00, 3110.54it/s, Materializing param=
Loading weights: 66%|▋| 260/391 [00:00<00:00, 3115.15it/s, Materializing param=
Loading weights: 66%|▋| 260/391 [00:00<00:00, 3110.84it/s, Materializing param=
Loading weights: 67%|▋| 261/391 [00:00<00:00, 3116.91it/s, Materializing param=
Loading weights: 67%|▋| 261/391 [00:00<00:00, 3112.87it/s, Materializing param=
Loading weights: 67%|▋| 262/391 [00:00<00:00, 3117.87it/s, Materializing param=
Loading weights: 67%|▋| 262/391 [00:00<00:00, 3114.24it/s, Materializing param=
Loading weights: 67%|▋| 263/391 [00:00<00:00, 3118.27it/s, Materializing param=
Loading weights: 67%|▋| 263/391 [00:00<00:00, 3113.69it/s, Materializing param=
Loading weights: 68%|▋| 264/391 [00:00<00:00, 3119.71it/s, Materializing param=
Loading weights: 68%|▋| 264/391 [00:00<00:00, 3116.46it/s, Materializing param=
Loading weights: 68%|▋| 265/391 [00:00<00:00, 3121.91it/s, Materializing param=
Loading weights: 68%|▋| 265/391 [00:00<00:00, 3118.21it/s, Materializing param=
Loading weights: 68%|▋| 266/391 [00:00<00:00, 3122.19it/s, Materializing param=
Loading weights: 68%|▋| 266/391 [00:00<00:00, 3117.79it/s, Materializing param=
Loading weights: 68%|▋| 267/391 [00:00<00:00, 3121.39it/s, Materializing param=
Loading weights: 68%|▋| 267/391 [00:00<00:00, 3116.91it/s, Materializing param=
Loading weights: 69%|▋| 268/391 [00:00<00:00, 3122.86it/s, Materializing param=
Loading weights: 69%|▋| 268/391 [00:00<00:00, 3119.46it/s, Materializing param=
Loading weights: 69%|▋| 269/391 [00:00<00:00, 3124.69it/s, Materializing param=
Loading weights: 69%|▋| 269/391 [00:00<00:00, 3121.67it/s, Materializing param=
Loading weights: 69%|▋| 270/391 [00:00<00:00, 3125.80it/s, Materializing param=
Loading weights: 69%|▋| 270/391 [00:00<00:00, 3121.52it/s, Materializing param=
Loading weights: 69%|▋| 271/391 [00:00<00:00, 3127.52it/s, Materializing param=
Loading weights: 69%|▋| 271/391 [00:00<00:00, 3124.27it/s, Materializing param=
Loading weights: 70%|▋| 272/391 [00:00<00:00, 3129.73it/s, Materializing param=
Loading weights: 70%|▋| 272/391 [00:00<00:00, 3126.29it/s, Materializing param=
Loading weights: 70%|▋| 273/391 [00:00<00:00, 3131.88it/s, Materializing param=
Loading weights: 70%|▋| 273/391 [00:00<00:00, 3127.85it/s, Materializing param=
Loading weights: 70%|▋| 274/391 [00:00<00:00, 3132.01it/s, Materializing param=
Loading weights: 70%|▋| 274/391 [00:00<00:00, 3129.08it/s, Materializing param=
Loading weights: 70%|▋| 275/391 [00:00<00:00, 3134.13it/s, Materializing param=
Loading weights: 70%|▋| 275/391 [00:00<00:00, 3130.25it/s, Materializing param=
Loading weights: 71%|▋| 276/391 [00:00<00:00, 3135.22it/s, Materializing param=
Loading weights: 71%|▋| 276/391 [00:00<00:00, 3131.14it/s, Materializing param=
Loading weights: 71%|▋| 277/391 [00:00<00:00, 3137.28it/s, Materializing param=
Loading weights: 71%|▋| 277/391 [00:00<00:00, 3132.53it/s, Materializing param=
Loading weights: 71%|▋| 278/391 [00:00<00:00, 3138.43it/s, Materializing param=
Loading weights: 71%|▋| 278/391 [00:00<00:00, 3135.26it/s, Materializing param=
Loading weights: 71%|▋| 279/391 [00:00<00:00, 3139.92it/s, Materializing param=
Loading weights: 71%|▋| 279/391 [00:00<00:00, 3136.85it/s, Materializing param=
Loading weights: 72%|▋| 280/391 [00:00<00:00, 3141.66it/s, Materializing param=
Loading weights: 72%|▋| 280/391 [00:00<00:00, 3137.78it/s, Materializing param=
Loading weights: 72%|▋| 281/391 [00:00<00:00, 3141.85it/s, Materializing param=
Loading weights: 72%|▋| 281/391 [00:00<00:00, 3138.87it/s, Materializing param=
Loading weights: 72%|▋| 282/391 [00:00<00:00, 3143.91it/s, Materializing param=
Loading weights: 72%|▋| 282/391 [00:00<00:00, 3140.02it/s, Materializing param=
Loading weights: 72%|▋| 283/391 [00:00<00:00, 3144.81it/s, Materializing param=
Loading weights: 72%|▋| 283/391 [00:00<00:00, 3140.78it/s, Materializing param=
Loading weights: 73%|▋| 284/391 [00:00<00:00, 3146.62it/s, Materializing param=
Loading weights: 73%|▋| 284/391 [00:00<00:00, 3141.68it/s, Materializing param=
Loading weights: 73%|▋| 285/391 [00:00<00:00, 3144.27it/s, Materializing param=
Loading weights: 73%|▋| 285/391 [00:00<00:00, 3140.52it/s, Materializing param=
Loading weights: 73%|▋| 286/391 [00:00<00:00, 3145.15it/s, Materializing param=
Loading weights: 73%|▋| 286/391 [00:00<00:00, 3141.00it/s, Materializing param=
Loading weights: 73%|▋| 287/391 [00:00<00:00, 3145.59it/s, Materializing param=
Loading weights: 73%|▋| 287/391 [00:00<00:00, 3141.71it/s, Materializing param=
Loading weights: 74%|▋| 288/391 [00:00<00:00, 3144.39it/s, Materializing param=
Loading weights: 74%|▋| 288/391 [00:00<00:00, 3140.32it/s, Materializing param=
Loading weights: 74%|▋| 289/391 [00:00<00:00, 3145.76it/s, Materializing param=
Loading weights: 74%|▋| 289/391 [00:00<00:00, 3142.68it/s, Materializing param=
Loading weights: 74%|▋| 290/391 [00:00<00:00, 3147.52it/s, Materializing param=
Loading weights: 74%|▋| 290/391 [00:00<00:00, 3144.12it/s, Materializing param=
Loading weights: 74%|▋| 291/391 [00:00<00:00, 3146.80it/s, Materializing param=
Loading weights: 74%|▋| 291/391 [00:00<00:00, 3142.99it/s, Materializing param=
Loading weights: 75%|▋| 292/391 [00:00<00:00, 3147.23it/s, Materializing param=
Loading weights: 75%|▋| 292/391 [00:00<00:00, 3143.03it/s, Materializing param=
Loading weights: 75%|▋| 293/391 [00:00<00:00, 3146.79it/s, Materializing param=
Loading weights: 75%|▋| 293/391 [00:00<00:00, 3142.83it/s, Materializing param=
Loading weights: 75%|▊| 294/391 [00:00<00:00, 3146.20it/s, Materializing param=
Loading weights: 75%|▊| 294/391 [00:00<00:00, 3141.54it/s, Materializing param=
Loading weights: 75%|▊| 295/391 [00:00<00:00, 3145.37it/s, Materializing param=
Loading weights: 75%|▊| 295/391 [00:00<00:00, 3141.04it/s, Materializing param=
Loading weights: 76%|▊| 296/391 [00:00<00:00, 3146.35it/s, Materializing param=
Loading weights: 76%|▊| 296/391 [00:00<00:00, 3142.83it/s, Materializing param=
Loading weights: 76%|▊| 297/391 [00:00<00:00, 3146.44it/s, Materializing param=
Loading weights: 76%|▊| 297/391 [00:00<00:00, 3142.28it/s, Materializing param=
Loading weights: 76%|▊| 298/391 [00:00<00:00, 3145.96it/s, Materializing param=
Loading weights: 76%|▊| 298/391 [00:00<00:00, 3141.57it/s, Materializing param=
Loading weights: 76%|▊| 299/391 [00:00<00:00, 3144.87it/s, Materializing param=
Loading weights: 76%|▊| 299/391 [00:00<00:00, 3139.88it/s, Materializing param=
Loading weights: 77%|▊| 300/391 [00:00<00:00, 3142.19it/s, Materializing param=
Loading weights: 77%|▊| 300/391 [00:00<00:00, 3139.35it/s, Materializing param=
Loading weights: 77%|▊| 301/391 [00:00<00:00, 3143.95it/s, Materializing param=
Loading weights: 77%|▊| 301/391 [00:00<00:00, 3139.46it/s, Materializing param=
Loading weights: 77%|▊| 302/391 [00:00<00:00, 3144.40it/s, Materializing param=
Loading weights: 77%|▊| 302/391 [00:00<00:00, 3141.34it/s, Materializing param=
Loading weights: 77%|▊| 303/391 [00:00<00:00, 3145.82it/s, Materializing param=
Loading weights: 77%|▊| 303/391 [00:00<00:00, 3141.23it/s, Materializing param=
Loading weights: 78%|▊| 304/391 [00:00<00:00, 3145.83it/s, Materializing param=
Loading weights: 78%|▊| 304/391 [00:00<00:00, 3142.18it/s, Materializing param=
Loading weights: 78%|▊| 305/391 [00:00<00:00, 3146.70it/s, Materializing param=
Loading weights: 78%|▊| 305/391 [00:00<00:00, 3143.88it/s, Materializing param=
Loading weights: 78%|▊| 306/391 [00:00<00:00, 3148.71it/s, Materializing param=
Loading weights: 78%|▊| 306/391 [00:00<00:00, 3145.64it/s, Materializing param=
Loading weights: 79%|▊| 307/391 [00:00<00:00, 3149.01it/s, Materializing param=
Loading weights: 79%|▊| 307/391 [00:00<00:00, 3145.42it/s, Materializing param=
Loading weights: 79%|▊| 308/391 [00:00<00:00, 3150.44it/s, Materializing param=
Loading weights: 79%|▊| 308/391 [00:00<00:00, 3147.78it/s, Materializing param=
Loading weights: 79%|▊| 309/391 [00:00<00:00, 3152.45it/s, Materializing param=
Loading weights: 79%|▊| 309/391 [00:00<00:00, 3149.14it/s, Materializing param=
Loading weights: 79%|▊| 310/391 [00:00<00:00, 3153.93it/s, Materializing param=
Loading weights: 79%|▊| 310/391 [00:00<00:00, 3148.98it/s, Materializing param=
Loading weights: 80%|▊| 311/391 [00:00<00:00, 3153.52it/s, Materializing param=
Loading weights: 80%|▊| 311/391 [00:00<00:00, 3150.60it/s, Materializing param=
Loading weights: 80%|▊| 312/391 [00:00<00:00, 3155.27it/s, Materializing param=
Loading weights: 80%|▊| 312/391 [00:00<00:00, 3152.62it/s, Materializing param=
Loading weights: 80%|▊| 313/391 [00:00<00:00, 3157.18it/s, Materializing param=
Loading weights: 80%|▊| 313/391 [00:00<00:00, 3153.87it/s, Materializing param=
Loading weights: 80%|▊| 314/391 [00:00<00:00, 3157.06it/s, Materializing param=
Loading weights: 80%|▊| 314/391 [00:00<00:00, 3153.22it/s, Materializing param=
Loading weights: 81%|▊| 315/391 [00:00<00:00, 3158.38it/s, Materializing param=
Loading weights: 81%|▊| 315/391 [00:00<00:00, 3155.40it/s, Materializing param=
Loading weights: 81%|▊| 316/391 [00:00<00:00, 3159.84it/s, Materializing param=
Loading weights: 81%|▊| 316/391 [00:00<00:00, 3157.20it/s, Materializing param=
Loading weights: 81%|▊| 317/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 81%|▊| 317/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 81%|▊| 317/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 81%|▊| 318/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 81%|▊| 318/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 82%|▊| 319/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 82%|▊| 319/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 82%|▊| 320/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 82%|▊| 320/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 82%|▊| 321/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 82%|▊| 321/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 82%|▊| 322/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 82%|▊| 322/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 83%|▊| 323/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 83%|▊| 323/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 83%|▊| 324/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 83%|▊| 324/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 83%|▊| 325/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 83%|▊| 325/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 83%|▊| 326/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 83%|▊| 326/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 84%|▊| 327/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 84%|▊| 327/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 84%|▊| 328/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 84%|▊| 328/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 84%|▊| 329/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 84%|▊| 329/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 84%|▊| 330/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 84%|▊| 330/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 85%|▊| 331/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 85%|▊| 331/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 85%|▊| 332/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 85%|▊| 332/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 85%|▊| 333/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 85%|▊| 333/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 85%|▊| 334/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 85%|▊| 334/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 86%|▊| 335/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 86%|▊| 335/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 86%|▊| 336/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 86%|▊| 336/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 86%|▊| 337/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 86%|▊| 337/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 86%|▊| 338/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 86%|▊| 338/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 87%|▊| 339/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 87%|▊| 339/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 87%|▊| 340/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 87%|▊| 340/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 87%|▊| 341/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 87%|▊| 341/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 87%|▊| 342/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 87%|▊| 342/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 88%|▉| 343/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 88%|▉| 343/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 88%|▉| 344/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 88%|▉| 344/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 88%|▉| 345/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 88%|▉| 345/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 88%|▉| 346/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 88%|▉| 346/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 89%|▉| 347/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 89%|▉| 347/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 89%|▉| 348/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 89%|▉| 348/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 89%|▉| 349/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 89%|▉| 349/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 90%|▉| 350/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 90%|▉| 350/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 90%|▉| 351/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 90%|▉| 351/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 90%|▉| 352/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 90%|▉| 352/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 90%|▉| 353/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 90%|▉| 353/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 91%|▉| 354/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 91%|▉| 354/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 91%|▉| 355/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 91%|▉| 355/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 91%|▉| 356/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 91%|▉| 356/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 91%|▉| 357/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 91%|▉| 357/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 92%|▉| 358/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 92%|▉| 358/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 92%|▉| 359/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 92%|▉| 359/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 92%|▉| 360/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 92%|▉| 360/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 92%|▉| 361/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 92%|▉| 361/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 93%|▉| 362/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 93%|▉| 362/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 93%|▉| 363/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 93%|▉| 363/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 93%|▉| 364/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 93%|▉| 364/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 93%|▉| 365/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 93%|▉| 365/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 94%|▉| 366/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 94%|▉| 366/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 94%|▉| 367/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 94%|▉| 367/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 94%|▉| 368/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 94%|▉| 368/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 94%|▉| 369/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 94%|▉| 369/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 95%|▉| 370/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 95%|▉| 370/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 95%|▉| 371/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 95%|▉| 371/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 95%|▉| 372/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 95%|▉| 372/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 95%|▉| 373/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 95%|▉| 373/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 96%|▉| 374/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 96%|▉| 374/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 96%|▉| 375/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 96%|▉| 375/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 96%|▉| 376/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 96%|▉| 376/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 96%|▉| 377/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 96%|▉| 377/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 97%|▉| 378/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 97%|▉| 378/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 97%|▉| 379/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 97%|▉| 379/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 97%|▉| 380/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 97%|▉| 380/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 97%|▉| 381/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 97%|▉| 381/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 98%|▉| 382/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 98%|▉| 382/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 98%|▉| 383/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 98%|▉| 383/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 98%|▉| 384/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 98%|▉| 384/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 98%|▉| 385/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 98%|▉| 385/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 99%|▉| 386/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 99%|▉| 386/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 99%|▉| 387/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 99%|▉| 387/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 99%|▉| 388/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 99%|▉| 388/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 99%|▉| 389/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 99%|▉| 389/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 100%|▉| 390/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 100%|▉| 390/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 100%|█| 391/391 [00:00<00:00, 3162.28it/s, Materializing param=
Loading weights: 100%|█| 391/391 [00:00<00:00, 3083.06it/s, Materializing param=
2026-07-31 02:06:36,213 - INFO - loading existing colbert_linear and sparse_linear---------
2026-07-31 02:06:36,215 - INFO - BGEM3FlagModel loaded successfully (Dense + Sparse).
2026-07-31 02:06:36,215 - INFO - MultiThreadPipelineWorker initialized (max_workers=8).
2026-07-31 02:06:36,215 - INFO - Dispatching 4 video files to MultiThreadPipelineWorker...
2026-07-31 02:06:36,215 - INFO - Initiating batch processing for 4 videos...
2026-07-31 02:06:36,216 - INFO - Processing video: 1_news_60s_720p
2026-07-31 02:06:36,216 - INFO - Segmenting video: data/official_videos/dummy_videos/1_news_60s_720p.mp4
2026-07-31 02:06:36,216 - INFO - Processing video: 2_news_60s_720p
2026-07-31 02:06:36,216 - INFO - Segmenting video: data/official_videos/dummy_videos/2_news_60s_720p.mp4
2026-07-31 02:06:36,217 - INFO - Processing video: pov_walkingtour_720p
2026-07-31 02:06:36,217 - INFO - Segmenting video: data/official_videos/dummy_videos/pov_walkingtour_720p.mp4
2026-07-31 02:06:36,217 - INFO - Processing video: test_transnet
2026-07-31 02:06:36,218 - INFO - Segmenting video: data/official_videos/dummy_videos/test_transnet.mp4

Offline Indexing Pipeline: 0%| | 0/4 [00:00<?, ?it/s]2026-07-31 02:06:39,304 - INFO - TransNetV2 detected 2 shots in video 'test_transnet.mp4'.
2026-07-31 02:06:39,417 - INFO - Shot 1 sharpest frame pruned (sim=1.0000)
2026-07-31 02:06:39,458 - INFO - Shot 2 sharpest frame pruned (sim=1.0000)
2026-07-31 02:06:39,458 - INFO - Extracted 2 keyframe artifacts to 'processed_data/1_frames'.
2026-07-31 02:06:39,582 - WARNING - Audio extraction failed for video 'data/official_videos/dummy_videos/test_transnet.mp4' (stream may be absent): Command '['/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2', '-y', '-i', 'data/official_videos/dummy_videos/test_transnet.mp4', '-ac', '1', '-ar', '16000', '-f', 'wav', '/tmp/tmpxwk8stmb.wav']' returned non-zero exit status 234.
2026-07-31 02:06:39,582 - INFO - Video file 'data/official_videos/dummy_videos/test_transnet.mp4' contains no audio track or empty audio stream.
2026-07-31 02:06:39,582 - INFO - [test_transnet] Extracted 2 keyframes, 0 ASR segments, and 0 audio events.
2026-07-31 02:06:39,582 - INFO - [test_transnet] Running YOLO object detection and Conda Offline OCR batch processing...
2026-07-31 02:06:58,267 - INFO - Đang gọi Conda xử lý OCR cho thư mục temp_ocr_test_transnet...
[OCR_ENV] Đang khởi tạo PaddleOCR...
/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddle/base/framework.py:688: UserWarning: You are using GPU version Paddle, but your CUDA device is not set properly. CPU device will be used by default.
warnings.warn(
Creating model: ('PP-LCNet_x1_0_doc_ori', None, None)
Checking connectivity to the model hosters, this may take a while. To bypass this check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` to `True`.
Using official model (PP-LCNet_x1_0_doc_ori), the model files will be automatically downloaded and saved in `/root/.paddlex/official_models/PP-LCNet_x1_0_doc_ori`.

Fetching 6 files: 0%| | 0/6 [00:00<?, ?it/s]
Fetching 6 files: 17%|█▋ | 1/6 [00:00<00:03, 1.61it/s]
Fetching 6 files: 100%|██████████| 6/6 [00:02<00:00, 2.40it/s]
Fetching 6 files: 100%|██████████| 6/6 [00:02<00:00, 2.40it/s]
Traceback (most recent call last):
File "/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/scripts/offline_ocr_batch.py", line 47, in <module>
main()
File "/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/scripts/offline_ocr_batch.py", line 18, in main
ocr = PaddleOCR(use_textline_orientation=True, lang='vi')
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddleocr/\_pipelines/ocr.py", line 173, in **init**
super().**init**(**base_params)
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddleocr/\_pipelines/base.py", line 67, in **init**
self.paddlex_pipeline = self.\_create_paddlex_pipeline()
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddleocr/\_pipelines/base.py", line 105, in \_create_paddlex_pipeline
return create_pipeline(config=self.\_merged_paddlex_config, **kwargs)
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/**init**.py", line 169, in create_pipeline
pipeline = BasePipeline.get(pipeline_name)(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/utils/deps.py", line 208, in \_wrapper
return old_init_func(self, *args, \*\*kwargs)
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/\_parallel.py", line 135, in **init**
self.\_pipeline = self.\_create_internal_pipeline(config, self.device)
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/\_parallel.py", line 190, in \_create_internal_pipeline
pipeline = self.\_pipeline_cls(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/ocr/pipeline.py", line 85, in **init**
self.doc_preprocessor_pipeline = self.create_pipeline(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/base.py", line 181, in create_pipeline
return create_pipeline(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/**init**.py", line 169, in create_pipeline
pipeline = BasePipeline.get(pipeline_name)(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/utils/deps.py", line 208, in \_wrapper
return old_init_func(self, *args, **kwargs)
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/\_parallel.py", line 135, in **init**
self.\_pipeline = self.\_create_internal_pipeline(config, self.device)
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/\_parallel.py", line 190, in \_create_internal_pipeline
pipeline = self.\_pipeline_cls(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/doc_preprocessor/pipeline.py", line 78, in **init**
self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/pipelines/base.py", line 143, in create_model
return create_predictor(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/models/**init**.py", line 470, in create_predictor
create_kwargs = \_build_predictor_kwargs(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/models/**init**.py", line 339, in \_build_predictor_kwargs
runner = \_build_predictor_runner(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/models/**init**.py", line 312, in \_build_predictor_runner
return inference_engine.build_runner(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/models/engines/\_base.py", line 190, in build_runner
runner = runner_builder(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/models/engines/paddle.py", line 116, in runner_builder
return PaddleStaticRunner(
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/utils/deps.py", line 158, in \_wrapper
return old_init_func(self, \*args, **kwargs)
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/models/runners/paddle_static/runner.py", line 286, in **init**
self.predictor = self.\_create()
File "/opt/conda/envs/ocr_env/lib/python3.9/site-packages/paddlex/inference/models/runners/paddle_static/runner.py", line 495, in \_create
config.set_optimization_level(3)
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
ERROR conda.cli.main_run:execute(148): `conda run python scripts/offline_ocr_batch.py --input_dir temp_ocr_test_transnet --output_file ocr_results_test_transnet.json` failed. (See above for error)
2026-07-31 02:07:20,804 - ERROR - [LỖI] Tiến trình OCR qua Conda thất bại: Command '['/opt/conda/bin/conda', 'run', '-n', 'ocr_env', 'python', 'scripts/offline_ocr_batch.py', '--input_dir', 'temp_ocr_test_transnet', '--output_file', 'ocr_results_test_transnet.json']' returned non-zero exit status 1.
2026-07-31 02:07:20,805 - INFO - [test_transnet] Generating window-based LLM context summaries...
2026-07-31 02:07:20,810 - INFO - Summarizing LLM context batch 1/1 [0s - 30s] (2 frames)...
2026-07-31 02:07:21,703 - INFO - HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-31 02:07:21,771 - INFO - Groq SDK call successful (llama-3.1-8b-instant).
2026-07-31 02:07:21,772 - INFO - [test_transnet] Generating visual and text embeddings...
2026-07-31 02:07:28,581 - INFO - Completed video processing: test_transnet in 52.36s | Metadata: processed_data/3_metadata/test_transnet_metadata.json

Offline Indexing Pipeline: 25%|████▌ | 1/4 [00:52<02:37, 52.36s/it]2026-07-31 02:09:38,379 - INFO - TransNetV2 detected 63 shots in video '2_news_60s_720p.mp4'.
2026-07-31 02:09:40,000 - INFO - Shot 1 sharpest frame pruned (sim=0.9966)
2026-07-31 02:09:42,354 - INFO - Shot 2 sharpest frame pruned (sim=0.9897)
2026-07-31 02:09:45,015 - INFO - Shot 3 sharpest frame pruned (sim=0.9823)
2026-07-31 02:09:48,163 - INFO - Shot 4 sharpest frame pruned (sim=0.9891)
2026-07-31 02:09:49,904 - INFO - TransNetV2 detected 95 shots in video '1_news_60s_720p.mp4'.
2026-07-31 02:09:50,692 - INFO - Shot 5 sharpest frame pruned (sim=0.9863)
2026-07-31 02:09:57,083 - INFO - Shot 3 sharpest frame pruned (sim=0.9994)
2026-07-31 02:10:00,318 - INFO - Shot 4 sharpest frame pruned (sim=0.9996)
2026-07-31 02:10:02,506 - INFO - Shot 5 sharpest frame pruned (sim=0.9894)
2026-07-31 02:10:03,835 - INFO - Shot 6 sharpest frame pruned (sim=0.9999)
2026-07-31 02:10:05,094 - INFO - Shot 7 sharpest frame pruned (sim=0.9987)
2026-07-31 02:10:06,221 - INFO - Shot 12 sharpest frame pruned (sim=0.9961)
2026-07-31 02:10:07,330 - INFO - Shot 8 sharpest frame pruned (sim=0.9958)
2026-07-31 02:10:09,378 - INFO - Shot 9 sharpest frame pruned (sim=0.9869)
2026-07-31 02:10:10,821 - INFO - Shot 10 sharpest frame pruned (sim=0.9981)
2026-07-31 02:10:12,536 - INFO - Shot 11 sharpest frame pruned (sim=0.9990)
2026-07-31 02:10:15,720 - INFO - Shot 12 sharpest frame pruned (sim=0.9925)
2026-07-31 02:10:17,969 - INFO - Shot 13 sharpest frame pruned (sim=0.9926)
2026-07-31 02:10:21,335 - INFO - Shot 18 sharpest frame pruned (sim=0.9920)
2026-07-31 02:10:21,918 - INFO - Shot 15 sharpest frame pruned (sim=0.9881)
2026-07-31 02:10:23,856 - INFO - Shot 16 sharpest frame pruned (sim=0.9954)
2026-07-31 02:10:25,145 - INFO - Shot 17 sharpest frame pruned (sim=0.9928)
2026-07-31 02:10:26,819 - INFO - Shot 18 sharpest frame pruned (sim=0.9939)
2026-07-31 02:10:28,493 - INFO - Shot 19 sharpest frame pruned (sim=0.9958)
2026-07-31 02:10:30,004 - INFO - Shot 20 sharpest frame pruned (sim=0.9981)
2026-07-31 02:10:36,195 - INFO - Shot 22 sharpest frame pruned (sim=0.9947)
2026-07-31 02:10:39,382 - INFO - Shot 25 sharpest frame pruned (sim=0.9806)
2026-07-31 02:10:39,436 - INFO - Shot 24 sharpest frame pruned (sim=0.9994)
2026-07-31 02:10:40,896 - INFO - Shot 25 sharpest frame pruned (sim=0.9992)
2026-07-31 02:10:41,251 - INFO - Shot 26 sharpest frame pruned (sim=0.9917)
2026-07-31 02:10:43,354 - INFO - Shot 26 sharpest frame pruned (sim=0.9834)
2026-07-31 02:10:47,298 - INFO - Shot 28 sharpest frame pruned (sim=0.9927)
2026-07-31 02:10:57,411 - INFO - Shot 32 sharpest frame pruned (sim=0.9950)
2026-07-31 02:10:57,703 - INFO - Shot 32 sharpest frame pruned (sim=0.9825)
2026-07-31 02:10:59,120 - INFO - Shot 33 sharpest frame pruned (sim=0.9998)
2026-07-31 02:11:00,406 - INFO - Shot 34 sharpest frame pruned (sim=0.9840)
2026-07-31 02:11:01,482 - INFO - Shot 35 sharpest frame pruned (sim=0.9997)
2026-07-31 02:11:07,737 - INFO - Shot 38 sharpest frame pruned (sim=0.9924)
2026-07-31 02:11:10,708 - INFO - Shot 40 sharpest frame pruned (sim=0.9867)
2026-07-31 02:11:20,845 - INFO - Shot 45 sharpest frame pruned (sim=0.9961)
2026-07-31 02:11:28,029 - INFO - Shot 49 sharpest frame pruned (sim=0.9886)
2026-07-31 02:11:29,963 - INFO - Shot 50 sharpest frame pruned (sim=0.9868)
2026-07-31 02:11:31,383 - INFO - Shot 51 sharpest frame pruned (sim=0.9998)
2026-07-31 02:11:33,473 - INFO - Shot 52 sharpest frame pruned (sim=0.9984)
2026-07-31 02:11:40,436 - INFO - Shot 55 sharpest frame pruned (sim=0.9905)
2026-07-31 02:11:44,166 - INFO - Shot 49 sharpest frame pruned (sim=0.9846)
2026-07-31 02:11:52,419 - INFO - Shot 52 sharpest frame pruned (sim=0.9920)
2026-07-31 02:11:57,690 - INFO - Shot 65 sharpest frame pruned (sim=0.9833)
2026-07-31 02:12:00,918 - INFO - Shot 66 sharpest frame pruned (sim=0.9879)
2026-07-31 02:12:03,651 - INFO - Shot 67 sharpest frame pruned (sim=0.9956)
2026-07-31 02:12:07,715 - INFO - Shot 69 sharpest frame pruned (sim=0.9946)
2026-07-31 02:12:09,297 - INFO - Shot 70 sharpest frame pruned (sim=0.9862)
2026-07-31 02:12:10,586 - INFO - Shot 71 sharpest frame pruned (sim=0.9981)
2026-07-31 02:12:12,728 - INFO - Shot 72 sharpest frame pruned (sim=0.9939)
2026-07-31 02:12:16,144 - INFO - Shot 74 sharpest frame pruned (sim=0.9827)
2026-07-31 02:12:20,876 - INFO - Shot 76 sharpest frame pruned (sim=0.9977)
2026-07-31 02:12:23,147 - INFO - Shot 62 sharpest frame pruned (sim=0.9802)
2026-07-31 02:12:25,169 - INFO - Extracted 113 keyframe artifacts to 'processed_data/1_frames'.
`torch_dtype` is deprecated! Use `dtype` instead!

Loading weights: 0%| | 0/479 [00:00<?, ?it/s]

Loading weights: 0%| | 1/479 [00:00<00:00, 9383.23it/s, Materializing param=mo

Loading weights: 0%| | 1/479 [00:00<00:03, 119.91it/s, Materializing param=mod

Loading weights: 0%| | 2/479 [00:00<00:10, 46.28it/s, Materializing param=mode

Loading weights: 0%| | 2/479 [00:00<00:10, 46.07it/s, Materializing param=mode

Loading weights: 1%| | 3/479 [00:00<00:06, 68.68it/s, Materializing param=mode

Loading weights: 1%| | 3/479 [00:00<00:06, 68.48it/s, Materializing param=mode

Loading weights: 1%| | 4/479 [00:00<00:05, 90.85it/s, Materializing param=mode

Loading weights: 1%| | 4/479 [00:00<00:05, 90.59it/s, Materializing param=mode

Loading weights: 1%| | 5/479 [00:00<00:04, 112.77it/s, Materializing param=mod

Loading weights: 1%| | 5/479 [00:00<00:04, 112.47it/s, Materializing param=mod

Loading weights: 1%| | 6/479 [00:00<00:03, 134.36it/s, Materializing param=mod

Loading weights: 1%| | 6/479 [00:00<00:03, 134.00it/s, Materializing param=mod

Loading weights: 1%| | 7/479 [00:00<00:03, 155.56it/s, Materializing param=mod

Loading weights: 1%| | 7/479 [00:00<00:03, 155.12it/s, Materializing param=mod

Loading weights: 2%| | 8/479 [00:00<00:03, 136.28it/s, Materializing param=mod

Loading weights: 2%| | 8/479 [00:00<00:04, 112.28it/s, Materializing param=mod

Loading weights: 2%| | 9/479 [00:00<00:03, 125.55it/s, Materializing param=mod

Loading weights: 2%| | 9/479 [00:00<00:03, 125.31it/s, Materializing param=mod

Loading weights: 2%| | 10/479 [00:00<00:03, 138.78it/s, Materializing param=mo

Loading weights: 2%| | 10/479 [00:00<00:03, 138.52it/s, Materializing param=mo

Loading weights: 2%| | 11/479 [00:00<00:03, 151.92it/s, Materializing param=mo

Loading weights: 2%| | 11/479 [00:00<00:03, 151.65it/s, Materializing param=mo

Loading weights: 3%| | 12/479 [00:00<00:02, 164.94it/s, Materializing param=mo

Loading weights: 3%| | 12/479 [00:00<00:02, 164.60it/s, Materializing param=mo

Loading weights: 3%| | 13/479 [00:00<00:02, 177.78it/s, Materializing param=mo

Loading weights: 3%| | 13/479 [00:00<00:02, 177.44it/s, Materializing param=mo

Loading weights: 3%| | 14/479 [00:00<00:02, 190.56it/s, Materializing param=mo

Loading weights: 3%| | 14/479 [00:00<00:02, 190.25it/s, Materializing param=mo

Loading weights: 3%| | 15/479 [00:00<00:02, 203.29it/s, Materializing param=mo

Loading weights: 3%| | 15/479 [00:00<00:02, 174.57it/s, Materializing param=mo

Loading weights: 3%| | 16/479 [00:00<00:02, 185.36it/s, Materializing param=mo

Loading weights: 3%| | 16/479 [00:00<00:02, 185.06it/s, Materializing param=mo

Loading weights: 4%| | 17/479 [00:00<00:02, 195.57it/s, Materializing param=mo

Loading weights: 4%| | 17/479 [00:00<00:02, 168.64it/s, Materializing param=mo

Loading weights: 4%| | 18/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 4%| | 18/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 4%| | 18/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 4%| | 19/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 4%| | 19/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 4%| | 20/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 4%| | 20/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 4%| | 21/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 4%| | 21/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 22/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 22/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 23/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 23/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 24/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 24/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 25/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 25/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 26/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 5%| | 26/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 27/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 27/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 28/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 28/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 29/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 29/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 30/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 30/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 31/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 6%| | 31/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 7%| | 32/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 7%| | 32/479 [00:00<00:03, 143.91it/s, Materializing param=mo

Loading weights: 7%| | 33/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 7%| | 33/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 7%| | 33/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 7%| | 34/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 7%| | 34/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 7%| | 35/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 7%| | 35/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 36/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 36/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 37/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 37/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 38/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 38/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 39/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 39/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 40/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 8%| | 40/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 41/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 41/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 42/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 42/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 43/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 43/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 44/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 44/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 45/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 9%| | 45/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 10%| | 46/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 10%| | 46/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 10%| | 47/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 10%| | 47/479 [00:00<00:03, 128.28it/s, Materializing param=mo

Loading weights: 10%| | 48/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 10%| | 48/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 10%| | 48/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 10%| | 49/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 10%| | 49/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 10%| | 50/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 10%| | 50/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 51/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 51/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 52/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 52/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 53/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 53/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 54/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 54/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 55/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 11%| | 55/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 12%| | 56/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 12%| | 56/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 12%| | 57/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 12%| | 57/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 12%| | 58/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 12%| | 58/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 12%| | 59/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 12%| | 59/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 13%|▏| 60/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 13%|▏| 60/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 13%|▏| 61/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 13%|▏| 61/479 [00:00<00:03, 134.87it/s, Materializing param=mo

Loading weights: 13%|▏| 62/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 13%|▏| 62/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 13%|▏| 62/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 13%|▏| 63/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 13%|▏| 63/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 13%|▏| 64/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 13%|▏| 64/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 65/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 65/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 66/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 66/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 67/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 67/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 68/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 68/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 69/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 14%|▏| 69/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 70/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 70/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 71/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 71/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 72/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 72/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 73/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 73/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 74/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 15%|▏| 74/479 [00:00<00:03, 128.75it/s, Materializing param=mo

Loading weights: 16%|▏| 75/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 75/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 75/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 76/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 76/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 77/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 77/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 78/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 78/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 79/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 16%|▏| 79/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 17%|▏| 80/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 17%|▏| 80/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 17%|▏| 81/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 17%|▏| 81/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 17%|▏| 82/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 17%|▏| 82/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 17%|▏| 83/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 17%|▏| 83/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 84/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 84/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 85/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 85/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 86/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 86/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 87/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 87/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 88/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 18%|▏| 88/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 89/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 89/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 90/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 90/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 91/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 91/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 92/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 92/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 93/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 19%|▏| 93/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 94/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 94/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 95/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 95/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 96/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 96/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 97/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 97/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 98/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 20%|▏| 98/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 21%|▏| 99/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 21%|▏| 99/479 [00:00<00:03, 120.66it/s, Materializing param=mo

Loading weights: 21%|▏| 100/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 21%|▏| 100/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 21%|▏| 101/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 21%|▏| 101/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 21%|▏| 102/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 21%|▏| 102/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 103/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 103/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 104/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 104/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 105/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 105/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 106/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 106/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 107/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 22%|▏| 107/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 108/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 108/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 109/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 109/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 110/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 110/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 111/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 111/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 112/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 23%|▏| 112/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 113/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 113/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 114/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 114/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 115/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 115/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 116/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 116/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 117/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 24%|▏| 117/479 [00:00<00:03, 120.66it/s, Materializing param=m

Loading weights: 25%|▏| 118/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▏| 118/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▏| 119/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▏| 119/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▎| 120/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▎| 120/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▎| 121/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▎| 121/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▎| 122/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 25%|▎| 122/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 26%|▎| 123/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 26%|▎| 123/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 26%|▎| 124/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 26%|▎| 124/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 26%|▎| 125/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 26%|▎| 125/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 26%|▎| 126/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 26%|▎| 126/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 127/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 127/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 128/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 128/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 129/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 129/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 130/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 130/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 131/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 27%|▎| 131/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 132/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 132/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 133/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 133/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 134/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 134/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 135/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 135/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 136/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 28%|▎| 136/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 137/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 137/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 138/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 138/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 139/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 139/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 140/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 140/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 141/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 29%|▎| 141/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 142/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 142/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 143/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 143/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 144/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 144/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 145/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 145/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 146/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 30%|▎| 146/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 31%|▎| 147/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 31%|▎| 147/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 31%|▎| 148/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 31%|▎| 148/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 31%|▎| 149/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 31%|▎| 149/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 31%|▎| 150/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 31%|▎| 150/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 151/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 151/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 152/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 152/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 153/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 153/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 154/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 154/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 155/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 32%|▎| 155/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 156/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 156/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 157/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 157/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 158/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 158/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 159/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 159/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 160/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 33%|▎| 160/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 161/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 161/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 162/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 162/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 163/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 163/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 164/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 164/479 [00:00<00:02, 120.66it/s, Materializing param=m

Loading weights: 34%|▎| 165/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 34%|▎| 165/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 34%|▎| 165/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 166/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 166/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 167/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 167/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 168/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 168/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 169/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 169/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 170/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 35%|▎| 170/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 36%|▎| 171/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 36%|▎| 171/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 36%|▎| 172/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 36%|▎| 172/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 36%|▎| 173/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 36%|▎| 173/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 36%|▎| 174/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 36%|▎| 174/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 175/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 175/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 176/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 176/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 177/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 177/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 178/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 178/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 179/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 37%|▎| 179/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 180/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 180/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 181/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 181/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 182/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 182/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 183/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 183/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 184/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 38%|▍| 184/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 185/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 185/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 186/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 186/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 187/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 187/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 188/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 188/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 189/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 39%|▍| 189/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 40%|▍| 190/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 40%|▍| 190/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 40%|▍| 191/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 40%|▍| 191/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 40%|▍| 192/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 40%|▍| 192/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 40%|▍| 193/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 40%|▍| 193/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 194/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 194/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 195/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 195/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 196/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 196/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 197/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 197/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 198/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 41%|▍| 198/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 199/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 199/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 200/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 200/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 201/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 201/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 202/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 202/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 203/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 42%|▍| 203/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 204/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 204/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 205/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 205/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 206/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 206/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 207/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 207/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 208/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 43%|▍| 208/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 209/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 209/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 210/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 210/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 211/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 211/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 212/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 212/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 213/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 44%|▍| 213/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 45%|▍| 214/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 45%|▍| 214/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 45%|▍| 215/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 45%|▍| 215/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 45%|▍| 216/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 45%|▍| 216/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 45%|▍| 217/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 45%|▍| 217/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 218/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 218/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 219/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 219/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 220/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 220/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 221/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 221/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 222/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 46%|▍| 222/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 223/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 223/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 224/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 224/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 225/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 225/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 226/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 226/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 227/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 47%|▍| 227/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 228/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 228/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 229/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 229/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 230/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 230/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 231/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 231/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 232/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 48%|▍| 232/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 233/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 233/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 234/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 234/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 235/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 235/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 236/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 236/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 237/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 49%|▍| 237/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 50%|▍| 238/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 50%|▍| 238/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 50%|▍| 239/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 50%|▍| 239/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 50%|▌| 240/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 50%|▌| 240/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 50%|▌| 241/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 50%|▌| 241/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 242/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 242/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 243/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 243/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 244/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 244/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 245/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 245/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 246/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 51%|▌| 246/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 247/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 247/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 248/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 248/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 249/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 249/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 250/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 250/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 251/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 52%|▌| 251/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 252/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 252/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 253/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 253/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 254/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 254/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 255/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 255/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 256/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 53%|▌| 256/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 257/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 257/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 258/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 258/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 259/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 259/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 260/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 260/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 261/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 54%|▌| 261/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 55%|▌| 262/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 55%|▌| 262/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 55%|▌| 263/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 55%|▌| 263/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 55%|▌| 264/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 55%|▌| 264/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 55%|▌| 265/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 55%|▌| 265/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 266/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 266/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 267/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 267/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 268/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 268/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 269/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 269/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 270/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 56%|▌| 270/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 271/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 271/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 272/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 272/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 273/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 273/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 274/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 274/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 275/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 57%|▌| 275/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 276/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 276/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 277/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 277/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 278/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 278/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 279/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 279/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 280/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 58%|▌| 280/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 281/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 281/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 282/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 282/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 283/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 283/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 284/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 284/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 285/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 59%|▌| 285/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 60%|▌| 286/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 60%|▌| 286/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 60%|▌| 287/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 60%|▌| 287/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 60%|▌| 288/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 60%|▌| 288/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 60%|▌| 289/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 60%|▌| 289/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 290/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 290/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 291/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 291/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 292/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 292/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 293/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 293/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 294/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 61%|▌| 294/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 295/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 295/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 296/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 296/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 297/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 297/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 298/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 298/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 299/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 62%|▌| 299/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 300/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 300/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 301/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 301/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 302/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 302/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 303/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 303/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 304/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 63%|▋| 304/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 64%|▋| 305/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 64%|▋| 305/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 64%|▋| 306/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 64%|▋| 306/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 64%|▋| 307/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 64%|▋| 307/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 64%|▋| 308/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 64%|▋| 308/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 309/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 309/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 310/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 310/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 311/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 311/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 312/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 312/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 313/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 65%|▋| 313/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 314/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 314/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 315/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 315/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 316/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 316/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 317/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 317/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 318/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 66%|▋| 318/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 319/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 319/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 320/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 320/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 321/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 321/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 322/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 322/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 323/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 67%|▋| 323/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 324/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 324/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 325/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 325/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 326/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 326/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 327/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 327/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 328/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 68%|▋| 328/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 69%|▋| 329/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 69%|▋| 329/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 69%|▋| 330/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 69%|▋| 330/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 69%|▋| 331/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 69%|▋| 331/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 69%|▋| 332/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 69%|▋| 332/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 333/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 333/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 334/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 334/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 335/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 335/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 336/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 336/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 337/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 70%|▋| 337/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 338/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 338/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 339/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 339/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 340/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 340/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 341/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 341/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 342/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 71%|▋| 342/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 343/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 343/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 344/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 344/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 345/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 345/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 346/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 346/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 347/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 72%|▋| 347/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 348/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 348/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 349/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 349/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 350/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 350/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 351/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 351/479 [00:00<00:00, 355.95it/s, Materializing param=m

Loading weights: 73%|▋| 352/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 73%|▋| 352/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 73%|▋| 352/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 74%|▋| 353/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 74%|▋| 353/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 74%|▋| 354/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 74%|▋| 354/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 74%|▋| 355/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 74%|▋| 355/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 74%|▋| 356/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 74%|▋| 356/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▋| 357/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▋| 357/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▋| 358/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▋| 358/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▋| 359/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▋| 359/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▊| 360/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▊| 360/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▊| 361/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 75%|▊| 361/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 362/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 362/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 363/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 363/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 364/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 364/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 365/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 365/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 366/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 76%|▊| 366/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 367/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 367/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 368/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 368/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 369/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 369/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 370/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 370/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 371/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 77%|▊| 371/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 372/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 372/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 373/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 373/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 374/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 374/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 375/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 375/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 376/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 78%|▊| 376/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 79%|▊| 377/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 79%|▊| 377/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 79%|▊| 378/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 79%|▊| 378/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 79%|▊| 379/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 79%|▊| 379/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 79%|▊| 380/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 79%|▊| 380/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 381/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 381/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 382/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 382/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 383/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 383/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 384/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 384/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 385/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 80%|▊| 385/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 386/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 386/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 387/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 387/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 388/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 388/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 389/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 389/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 390/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 81%|▊| 390/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 391/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 391/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 392/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 392/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 393/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 393/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 394/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 394/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 395/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 82%|▊| 395/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 83%|▊| 396/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 83%|▊| 396/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 83%|▊| 397/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 83%|▊| 397/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 83%|▊| 398/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 83%|▊| 398/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 83%|▊| 399/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 83%|▊| 399/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 400/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 400/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 401/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 401/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 402/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 402/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 403/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 403/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 404/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 84%|▊| 404/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 405/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 405/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 406/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 406/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 407/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 407/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 408/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 408/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 409/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 85%|▊| 409/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 410/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 410/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 411/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 411/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 412/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 412/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 413/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 413/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 414/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 86%|▊| 414/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 415/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 415/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 416/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 416/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 417/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 417/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 418/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 418/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 419/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 87%|▊| 419/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 88%|▉| 420/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 88%|▉| 420/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 88%|▉| 421/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 88%|▉| 421/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 88%|▉| 422/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 88%|▉| 422/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 88%|▉| 423/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 88%|▉| 423/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 424/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 424/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 425/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 425/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 426/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 426/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 427/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 427/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 428/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 89%|▉| 428/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 429/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 429/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 430/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 430/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 431/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 431/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 432/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 432/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 433/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 90%|▉| 433/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 434/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 434/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 435/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 435/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 436/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 436/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 437/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 437/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 438/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 91%|▉| 438/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 439/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 439/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 440/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 440/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 441/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 441/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 442/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 442/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 443/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 92%|▉| 443/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 93%|▉| 444/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 93%|▉| 444/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 93%|▉| 445/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 93%|▉| 445/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 93%|▉| 446/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 93%|▉| 446/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 93%|▉| 447/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 93%|▉| 447/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 448/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 448/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 449/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 449/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 450/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 450/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 451/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 451/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 452/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 94%|▉| 452/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 453/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 453/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 454/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 454/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 455/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 455/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 456/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 456/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 457/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 95%|▉| 457/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 458/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 458/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 459/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 459/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 460/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 460/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 461/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 461/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 462/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 96%|▉| 462/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 463/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 463/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 464/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 464/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 465/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 465/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 466/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 466/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 467/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 97%|▉| 467/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 98%|▉| 468/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 98%|▉| 468/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 98%|▉| 469/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 98%|▉| 469/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 98%|▉| 470/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 98%|▉| 470/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 98%|▉| 471/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 98%|▉| 471/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 472/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 472/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 473/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 473/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 474/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 474/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 475/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 475/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 476/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 99%|▉| 476/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 100%|▉| 477/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 100%|▉| 477/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 100%|▉| 478/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 100%|▉| 478/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 100%|█| 479/479 [00:00<00:00, 812.50it/s, Materializing param=m

Loading weights: 100%|█| 479/479 [00:00<00:00, 552.88it/s, Materializing param=m
2026-07-31 02:12:28,603 - INFO - PhoWhisper model 'models/phowhisper-small' loaded successfully (dtype=torch.float32).
2026-07-31 02:12:28,828 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:29,024 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/config.json "HTTP/1.1 200 OK"
2026-07-31 02:12:29,224 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/config.json "HTTP/1.1 200 OK"

config.json: 5.39kB [00:00, 8.32MB/s]A
2026-07-31 02:12:29,444 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/adapter_config.json "HTTP/1.1 404 Not Found"
2026-07-31 02:12:29,759 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:29,768 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/config.json "HTTP/1.1 200 OK"
2026-07-31 02:12:29,970 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
2026-07-31 02:12:30,001 - INFO - Shot 80 sharpest frame pruned (sim=0.9827)
2026-07-31 02:12:30,178 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-07-31 02:12:30,387 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/pytorch_model.bin "HTTP/1.1 302 Found"
2026-07-31 02:12:30,598 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/xet-read-token/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a "HTTP/1.1 200 OK"

pytorch_model.bin: 0%| | 0.00/615M [00:00<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:00<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:00<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:00<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:00<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:01<?, ?B/s]2026-07-31 02:12:31,700 - INFO - Shot 81 sharpest frame pruned (sim=0.9887)

pytorch_model.bin: 0%| | 0.00/615M [00:01<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:01<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:01<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:01<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:02<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:02<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:02<?, ?B/s]

pytorch_model.bin: 0%| | 0.00/615M [00:02<?, ?B/s]2026-07-31 02:12:33,371 - INFO - Shot 82 sharpest frame pruned (sim=0.9870)

pytorch_model.bin: 1%|▎ | 8.36M/615M [00:02<00:14, 41.9MB/s]

pytorch_model.bin: 2%|▍ | 14.3M/615M [00:03<00:37, 15.9MB/s]

pytorch_model.bin: 3%|▌ | 19.1M/615M [00:03<00:32, 18.2MB/s]2026-07-31 02:12:34,770 - INFO - TransNetV2 detected 48 shots in video 'pov_walkingtour_720p.mp4'.

pytorch_model.bin: 8%|█▌ | 49.8M/615M [00:04<00:15, 36.1MB/s]2026-07-31 02:12:34,870 - INFO - Shot 1 sharpest frame pruned (sim=1.0000)
2026-07-31 02:12:35,042 - INFO - Shot 83 sharpest frame pruned (sim=0.9995)

pytorch_model.bin: 12%|██▍ | 73.3M/615M [00:04<00:14, 37.5MB/s]2026-07-31 02:12:35,932 - INFO - Shot 2 sharpest frame pruned (sim=0.9920)

pytorch_model.bin: 18%|███▊ | 111M/615M [00:05<00:10, 47.3MB/s]

pytorch_model.bin: 29%|██████ | 177M/615M [00:06<00:06, 69.6MB/s]

pytorch_model.bin: 35%|███████▍ | 216M/615M [00:06<00:05, 68.0MB/s]

pytorch_model.bin: 40%|████████▎ | 243M/615M [00:07<00:05, 67.7MB/s]

pytorch_model.bin: 42%|████████▋ | 256M/615M [00:07<00:05, 67.3MB/s]

pytorch_model.bin: 48%|██████████ | 293M/615M [00:07<00:04, 74.7MB/s]

pytorch_model.bin: 51%|██████████▋ | 313M/615M [00:07<00:03, 79.6MB/s]

pytorch_model.bin: 62%|████████████▉ | 380M/615M [00:08<00:02, 81.6MB/s]

pytorch_model.bin: 74%|████████████████▎ | 457M/615M [00:08<00:01, 127MB/s]2026-07-31 02:12:39,529 - INFO - Shot 86 sharpest frame pruned (sim=0.9964)

pytorch_model.bin: 98%|█████████████████████▋| 605M/615M [00:09<00:00, 237MB/s]2026-07-31 02:12:39,822 - INFO - Shot 3 sharpest frame pruned (sim=0.9941)
2026-07-31 02:12:40,616 - INFO - Shot 87 sharpest frame pruned (sim=0.9944)
pytorch_model.bin: 100%|█████████████████████| 615M/615M [00:10<00:00, 60.1MB/s]
2026-07-31 02:12:41,058 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
2026-07-31 02:12:41,284 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused "HTTP/1.1 200 OK"

Loading weights: 0%| | 0/447 [00:00<?, ?it/s]

Loading weights: 0%| | 1/447 [00:00<00:00, 19418.07it/s, Materializing param=a

Loading weights: 0%| | 1/447 [00:00<00:00, 4629.47it/s, Materializing param=au

Loading weights: 0%| | 2/447 [00:00<00:00, 472.01it/s, Materializing param=aud

Loading weights: 0%| | 2/447 [00:00<00:01, 442.55it/s, Materializing param=aud

Loading weights: 1%| | 3/447 [00:00<00:00, 615.09it/s, Materializing param=aud

Loading weights: 1%| | 3/447 [00:00<00:00, 599.04it/s, Materializing param=aud

Loading weights: 1%| | 4/447 [00:00<00:00, 767.24it/s, Materializing param=aud

Loading weights: 1%| | 4/447 [00:00<00:00, 719.62it/s, Materializing param=aud

Loading weights: 1%| | 5/447 [00:00<00:00, 858.08it/s, Materializing param=aud

Loading weights: 1%| | 5/447 [00:00<00:00, 831.18it/s, Materializing param=aud

Loading weights: 1%| | 6/447 [00:00<00:00, 941.52it/s, Materializing param=aud

Loading weights: 1%| | 6/447 [00:00<00:00, 916.85it/s, Materializing param=aud

Loading weights: 2%| | 7/447 [00:00<00:00, 1022.64it/s, Materializing param=au

Loading weights: 2%| | 7/447 [00:00<00:00, 994.62it/s, Materializing param=aud

Loading weights: 2%| | 8/447 [00:00<00:00, 1088.19it/s, Materializing param=au

Loading weights: 2%| | 8/447 [00:00<00:00, 1058.13it/s, Materializing param=au

Loading weights: 2%| | 9/447 [00:00<00:00, 1137.25it/s, Materializing param=au

Loading weights: 2%| | 9/447 [00:00<00:00, 1122.04it/s, Materializing param=au

Loading weights: 2%| | 10/447 [00:00<00:00, 1201.77it/s, Materializing param=a

Loading weights: 2%| | 10/447 [00:00<00:00, 1171.76it/s, Materializing param=a

Loading weights: 2%| | 11/447 [00:00<00:00, 1242.02it/s, Materializing param=a

Loading weights: 2%| | 11/447 [00:00<00:00, 1216.29it/s, Materializing param=a

Loading weights: 3%| | 12/447 [00:00<00:00, 1275.06it/s, Materializing param=a

Loading weights: 3%| | 12/447 [00:00<00:00, 1249.61it/s, Materializing param=a

Loading weights: 3%| | 13/447 [00:00<00:00, 1317.34it/s, Materializing param=a

Loading weights: 3%| | 13/447 [00:00<00:00, 1291.04it/s, Materializing param=a

Loading weights: 3%| | 14/447 [00:00<00:00, 1350.51it/s, Materializing param=a

Loading weights: 3%| | 14/447 [00:00<00:00, 1319.82it/s, Materializing param=a

Loading weights: 3%| | 15/447 [00:00<00:00, 1366.76it/s, Materializing param=a

Loading weights: 3%| | 15/447 [00:00<00:00, 1339.09it/s, Materializing param=a

Loading weights: 4%| | 16/447 [00:00<00:00, 1374.45it/s, Materializing param=a

Loading weights: 4%| | 16/447 [00:00<00:00, 1358.70it/s, Materializing param=a

Loading weights: 4%| | 17/447 [00:00<00:00, 1419.11it/s, Materializing param=a

Loading weights: 4%| | 17/447 [00:00<00:00, 1406.01it/s, Materializing param=a

Loading weights: 4%| | 18/447 [00:00<00:00, 1367.04it/s, Materializing param=a

Loading weights: 4%| | 18/447 [00:00<00:00, 1345.33it/s, Materializing param=a

Loading weights: 4%| | 19/447 [00:00<00:00, 1389.59it/s, Materializing param=a

Loading weights: 4%| | 19/447 [00:00<00:00, 1368.73it/s, Materializing param=a

Loading weights: 4%| | 20/447 [00:00<00:00, 1411.25it/s, Materializing param=a

Loading weights: 4%| | 20/447 [00:00<00:00, 1391.68it/s, Materializing param=a

Loading weights: 5%| | 21/447 [00:00<00:00, 1422.76it/s, Materializing param=a

Loading weights: 5%| | 21/447 [00:00<00:00, 1404.21it/s, Materializing param=a

Loading weights: 5%| | 22/447 [00:00<00:00, 1443.33it/s, Materializing param=a

Loading weights: 5%| | 22/447 [00:00<00:00, 1425.62it/s, Materializing param=a

Loading weights: 5%| | 23/447 [00:00<00:00, 1462.45it/s, Materializing param=a

Loading weights: 5%| | 23/447 [00:00<00:00, 1443.22it/s, Materializing param=a

Loading weights: 5%| | 24/447 [00:00<00:00, 1470.55it/s, Materializing param=a

Loading weights: 5%| | 24/447 [00:00<00:00, 1452.34it/s, Materializing param=a

Loading weights: 6%| | 25/447 [00:00<00:00, 1483.49it/s, Materializing param=a

Loading weights: 6%| | 25/447 [00:00<00:00, 1466.97it/s, Materializing param=a

Loading weights: 6%| | 26/447 [00:00<00:00, 1493.04it/s, Materializing param=a

Loading weights: 6%| | 26/447 [00:00<00:00, 1475.11it/s, Materializing param=a

Loading weights: 6%| | 27/447 [00:00<00:00, 1504.63it/s, Materializing param=a

Loading weights: 6%| | 27/447 [00:00<00:00, 1488.22it/s, Materializing param=a

Loading weights: 6%| | 28/447 [00:00<00:00, 1513.06it/s, Materializing param=a

Loading weights: 6%| | 28/447 [00:00<00:00, 1496.65it/s, Materializing param=a

Loading weights: 6%| | 29/447 [00:00<00:00, 1513.91it/s, Materializing param=a

Loading weights: 6%| | 29/447 [00:00<00:00, 1446.47it/s, Materializing param=a

Loading weights: 7%| | 30/447 [00:00<00:00, 1455.98it/s, Materializing param=a

Loading weights: 7%| | 30/447 [00:00<00:00, 1443.94it/s, Materializing param=a

Loading weights: 7%| | 31/447 [00:00<00:00, 1474.21it/s, Materializing param=a

Loading weights: 7%| | 31/447 [00:00<00:00, 1463.88it/s, Materializing param=a

Loading weights: 7%| | 32/447 [00:00<00:00, 1493.23it/s, Materializing param=a

Loading weights: 7%| | 32/447 [00:00<00:00, 1483.59it/s, Materializing param=a

Loading weights: 7%| | 33/447 [00:00<00:00, 1513.43it/s, Materializing param=a

Loading weights: 7%| | 33/447 [00:00<00:00, 1503.60it/s, Materializing param=a

Loading weights: 8%| | 34/447 [00:00<00:00, 1532.92it/s, Materializing param=a

Loading weights: 8%| | 34/447 [00:00<00:00, 1522.53it/s, Materializing param=a

Loading weights: 8%| | 35/447 [00:00<00:00, 1551.03it/s, Materializing param=a

Loading weights: 8%| | 35/447 [00:00<00:00, 1541.60it/s, Materializing param=a

Loading weights: 8%| | 36/447 [00:00<00:00, 1568.70it/s, Materializing param=a

Loading weights: 8%| | 36/447 [00:00<00:00, 1559.58it/s, Materializing param=a

Loading weights: 8%| | 37/447 [00:00<00:00, 1584.87it/s, Materializing param=a

Loading weights: 8%| | 37/447 [00:00<00:00, 1575.11it/s, Materializing param=a

Loading weights: 9%| | 38/447 [00:00<00:00, 1602.62it/s, Materializing param=a

Loading weights: 9%| | 38/447 [00:00<00:00, 1593.01it/s, Materializing param=a

Loading weights: 9%| | 39/447 [00:00<00:00, 1619.50it/s, Materializing param=a

Loading weights: 9%| | 39/447 [00:00<00:00, 1610.40it/s, Materializing param=a

Loading weights: 9%| | 40/447 [00:00<00:00, 1636.19it/s, Materializing param=a

Loading weights: 9%| | 40/447 [00:00<00:00, 1627.24it/s, Materializing param=a

Loading weights: 9%| | 41/447 [00:00<00:00, 1652.62it/s, Materializing param=a

Loading weights: 9%| | 41/447 [00:00<00:00, 1643.91it/s, Materializing param=a

Loading weights: 9%| | 42/447 [00:00<00:00, 1669.94it/s, Materializing param=a

Loading weights: 9%| | 42/447 [00:00<00:00, 1659.87it/s, Materializing param=a

Loading weights: 10%| | 43/447 [00:00<00:00, 1684.99it/s, Materializing param=a

Loading weights: 10%| | 43/447 [00:00<00:00, 1676.43it/s, Materializing param=a

Loading weights: 10%| | 44/447 [00:00<00:00, 1701.29it/s, Materializing param=a

Loading weights: 10%| | 44/447 [00:00<00:00, 1692.83it/s, Materializing param=a

Loading weights: 10%| | 45/447 [00:00<00:00, 1716.74it/s, Materializing param=a

Loading weights: 10%| | 45/447 [00:00<00:00, 1707.84it/s, Materializing param=a

Loading weights: 10%| | 46/447 [00:00<00:00, 1730.14it/s, Materializing param=a

Loading weights: 10%| | 46/447 [00:00<00:00, 1721.23it/s, Materializing param=a

Loading weights: 11%| | 47/447 [00:00<00:00, 1743.95it/s, Materializing param=a

Loading weights: 11%| | 47/447 [00:00<00:00, 1735.29it/s, Materializing param=a

Loading weights: 11%| | 48/447 [00:00<00:00, 1756.53it/s, Materializing param=a

Loading weights: 11%| | 48/447 [00:00<00:00, 1747.98it/s, Materializing param=a

Loading weights: 11%| | 49/447 [00:00<00:00, 1770.50it/s, Materializing param=a

Loading weights: 11%| | 49/447 [00:00<00:00, 1760.82it/s, Materializing param=a

Loading weights: 11%| | 50/447 [00:00<00:00, 1781.99it/s, Materializing param=a

Loading weights: 11%| | 50/447 [00:00<00:00, 1772.20it/s, Materializing param=a

Loading weights: 11%| | 51/447 [00:00<00:00, 1793.61it/s, Materializing param=a

Loading weights: 11%| | 51/447 [00:00<00:00, 1784.81it/s, Materializing param=a

Loading weights: 12%| | 52/447 [00:00<00:00, 1805.60it/s, Materializing param=a

Loading weights: 12%| | 52/447 [00:00<00:00, 1797.10it/s, Materializing param=a

Loading weights: 12%| | 53/447 [00:00<00:00, 1818.20it/s, Materializing param=a

Loading weights: 12%| | 53/447 [00:00<00:00, 1808.73it/s, Materializing param=a

Loading weights: 12%| | 54/447 [00:00<00:00, 1829.10it/s, Materializing param=a

Loading weights: 12%| | 54/447 [00:00<00:00, 1820.39it/s, Materializing param=a

Loading weights: 12%| | 55/447 [00:00<00:00, 1838.86it/s, Materializing param=a

Loading weights: 12%| | 55/447 [00:00<00:00, 1829.93it/s, Materializing param=a

Loading weights: 13%|▏| 56/447 [00:00<00:00, 1847.57it/s, Materializing param=a

Loading weights: 13%|▏| 56/447 [00:00<00:00, 1838.66it/s, Materializing param=a

Loading weights: 13%|▏| 57/447 [00:00<00:00, 1856.98it/s, Materializing param=a

Loading weights: 13%|▏| 57/447 [00:00<00:00, 1847.51it/s, Materializing param=a

Loading weights: 13%|▏| 58/447 [00:00<00:00, 1865.45it/s, Materializing param=a

Loading weights: 13%|▏| 58/447 [00:00<00:00, 1856.13it/s, Materializing param=a

Loading weights: 13%|▏| 59/447 [00:00<00:00, 1874.05it/s, Materializing param=a

Loading weights: 13%|▏| 59/447 [00:00<00:00, 1865.58it/s, Materializing param=a

Loading weights: 13%|▏| 60/447 [00:00<00:00, 1880.60it/s, Materializing param=a

Loading weights: 13%|▏| 60/447 [00:00<00:00, 1872.14it/s, Materializing param=a

Loading weights: 14%|▏| 61/447 [00:00<00:00, 1888.61it/s, Materializing param=a

Loading weights: 14%|▏| 61/447 [00:00<00:00, 1880.09it/s, Materializing param=a

Loading weights: 14%|▏| 62/447 [00:00<00:00, 1897.13it/s, Materializing param=a

Loading weights: 14%|▏| 62/447 [00:00<00:00, 1888.13it/s, Materializing param=a

Loading weights: 14%|▏| 63/447 [00:00<00:00, 1901.81it/s, Materializing param=a

Loading weights: 14%|▏| 63/447 [00:00<00:00, 1886.22it/s, Materializing param=a

Loading weights: 14%|▏| 64/447 [00:00<00:00, 1895.69it/s, Materializing param=a

Loading weights: 14%|▏| 64/447 [00:00<00:00, 1883.90it/s, Materializing param=a

Loading weights: 15%|▏| 65/447 [00:00<00:00, 1892.61it/s, Materializing param=a

Loading weights: 15%|▏| 65/447 [00:00<00:00, 1881.32it/s, Materializing param=a

Loading weights: 15%|▏| 66/447 [00:00<00:00, 1889.39it/s, Materializing param=a

Loading weights: 15%|▏| 66/447 [00:00<00:00, 1878.54it/s, Materializing param=a

Loading weights: 15%|▏| 67/447 [00:00<00:00, 1886.71it/s, Materializing param=a

Loading weights: 15%|▏| 67/447 [00:00<00:00, 1876.28it/s, Materializing param=a

Loading weights: 15%|▏| 68/447 [00:00<00:00, 1884.76it/s, Materializing param=a

Loading weights: 15%|▏| 68/447 [00:00<00:00, 1871.34it/s, Materializing param=a

Loading weights: 15%|▏| 69/447 [00:00<00:00, 1883.28it/s, Materializing param=a

Loading weights: 15%|▏| 69/447 [00:00<00:00, 1868.93it/s, Materializing param=a

Loading weights: 16%|▏| 70/447 [00:00<00:00, 1878.33it/s, Materializing param=a

Loading weights: 16%|▏| 70/447 [00:00<00:00, 1867.56it/s, Materializing param=a

Loading weights: 16%|▏| 71/447 [00:00<00:00, 1875.59it/s, Materializing param=a

Loading weights: 16%|▏| 71/447 [00:00<00:00, 1865.14it/s, Materializing param=a

Loading weights: 16%|▏| 72/447 [00:00<00:00, 1875.38it/s, Materializing param=a

Loading weights: 16%|▏| 72/447 [00:00<00:00, 1866.10it/s, Materializing param=a

Loading weights: 16%|▏| 73/447 [00:00<00:00, 1874.52it/s, Materializing param=a

Loading weights: 16%|▏| 73/447 [00:00<00:00, 1865.12it/s, Materializing param=a

Loading weights: 17%|▏| 74/447 [00:00<00:00, 1875.10it/s, Materializing param=a

Loading weights: 17%|▏| 74/447 [00:00<00:00, 1863.59it/s, Materializing param=a

Loading weights: 17%|▏| 75/447 [00:00<00:00, 1873.47it/s, Materializing param=a

Loading weights: 17%|▏| 75/447 [00:00<00:00, 1863.85it/s, Materializing param=a

Loading weights: 17%|▏| 76/447 [00:00<00:00, 1870.64it/s, Materializing param=a

Loading weights: 17%|▏| 76/447 [00:00<00:00, 1860.73it/s, Materializing param=a

Loading weights: 17%|▏| 77/447 [00:00<00:00, 1871.51it/s, Materializing param=a

Loading weights: 17%|▏| 77/447 [00:00<00:00, 1859.38it/s, Materializing param=a

Loading weights: 17%|▏| 78/447 [00:00<00:00, 1870.21it/s, Materializing param=a

Loading weights: 17%|▏| 78/447 [00:00<00:00, 1860.80it/s, Materializing param=a

Loading weights: 18%|▏| 79/447 [00:00<00:00, 1868.17it/s, Materializing param=a

Loading weights: 18%|▏| 79/447 [00:00<00:00, 1860.40it/s, Materializing param=a

Loading weights: 18%|▏| 80/447 [00:00<00:00, 1868.15it/s, Materializing param=a

Loading weights: 18%|▏| 80/447 [00:00<00:00, 1859.43it/s, Materializing param=a

Loading weights: 18%|▏| 81/447 [00:00<00:00, 1868.60it/s, Materializing param=a

Loading weights: 18%|▏| 81/447 [00:00<00:00, 1859.82it/s, Materializing param=a

Loading weights: 18%|▏| 82/447 [00:00<00:00, 1869.73it/s, Materializing param=a

Loading weights: 18%|▏| 82/447 [00:00<00:00, 1860.36it/s, Materializing param=a

Loading weights: 19%|▏| 83/447 [00:00<00:00, 1870.53it/s, Materializing param=a

Loading weights: 19%|▏| 83/447 [00:00<00:00, 1859.51it/s, Materializing param=a

Loading weights: 19%|▏| 84/447 [00:00<00:00, 1869.15it/s, Materializing param=a

Loading weights: 19%|▏| 84/447 [00:00<00:00, 1860.59it/s, Materializing param=a

Loading weights: 19%|▏| 85/447 [00:00<00:00, 1865.82it/s, Materializing param=a

Loading weights: 19%|▏| 85/447 [00:00<00:00, 1856.90it/s, Materializing param=a

Loading weights: 19%|▏| 86/447 [00:00<00:00, 1866.23it/s, Materializing param=a

Loading weights: 19%|▏| 86/447 [00:00<00:00, 1855.37it/s, Materializing param=a

Loading weights: 19%|▏| 87/447 [00:00<00:00, 1861.86it/s, Materializing param=a

Loading weights: 19%|▏| 87/447 [00:00<00:00, 1853.33it/s, Materializing param=a

Loading weights: 20%|▏| 88/447 [00:00<00:00, 1859.97it/s, Materializing param=a

Loading weights: 20%|▏| 88/447 [00:00<00:00, 1851.77it/s, Materializing param=a

Loading weights: 20%|▏| 89/447 [00:00<00:00, 1860.28it/s, Materializing param=a

Loading weights: 20%|▏| 89/447 [00:00<00:00, 1852.40it/s, Materializing param=a

Loading weights: 20%|▏| 90/447 [00:00<00:00, 1860.95it/s, Materializing param=a

Loading weights: 20%|▏| 90/447 [00:00<00:00, 1851.03it/s, Materializing param=a

Loading weights: 20%|▏| 91/447 [00:00<00:00, 1859.45it/s, Materializing param=a

Loading weights: 20%|▏| 91/447 [00:00<00:00, 1848.61it/s, Materializing param=a

Loading weights: 21%|▏| 92/447 [00:00<00:00, 1856.09it/s, Materializing param=a

Loading weights: 21%|▏| 92/447 [00:00<00:00, 1848.30it/s, Materializing param=a

Loading weights: 21%|▏| 93/447 [00:00<00:00, 1854.71it/s, Materializing param=a

Loading weights: 21%|▏| 93/447 [00:00<00:00, 1846.62it/s, Materializing param=a

Loading weights: 21%|▏| 94/447 [00:00<00:00, 1855.23it/s, Materializing param=a

Loading weights: 21%|▏| 94/447 [00:00<00:00, 1845.69it/s, Materializing param=a

Loading weights: 21%|▏| 95/447 [00:00<00:00, 1851.87it/s, Materializing param=a

Loading weights: 21%|▏| 95/447 [00:00<00:00, 1844.76it/s, Materializing param=a

Loading weights: 21%|▏| 96/447 [00:00<00:00, 1852.74it/s, Materializing param=a

Loading weights: 21%|▏| 96/447 [00:00<00:00, 1846.03it/s, Materializing param=a

Loading weights: 22%|▏| 97/447 [00:00<00:00, 1852.46it/s, Materializing param=a

Loading weights: 22%|▏| 97/447 [00:00<00:00, 1845.43it/s, Materializing param=a

Loading weights: 22%|▏| 98/447 [00:00<00:00, 1854.02it/s, Materializing param=a

Loading weights: 22%|▏| 98/447 [00:00<00:00, 1847.43it/s, Materializing param=a

Loading weights: 22%|▏| 99/447 [00:00<00:00, 1856.20it/s, Materializing param=a

Loading weights: 22%|▏| 99/447 [00:00<00:00, 1848.94it/s, Materializing param=a

Loading weights: 22%|▏| 100/447 [00:00<00:00, 1855.04it/s, Materializing param=

Loading weights: 22%|▏| 100/447 [00:00<00:00, 1847.93it/s, Materializing param=

Loading weights: 23%|▏| 101/447 [00:00<00:00, 1854.21it/s, Materializing param=

Loading weights: 23%|▏| 101/447 [00:00<00:00, 1847.42it/s, Materializing param=

Loading weights: 23%|▏| 102/447 [00:00<00:00, 1854.85it/s, Materializing param=

Loading weights: 23%|▏| 102/447 [00:00<00:00, 1847.51it/s, Materializing param=

Loading weights: 23%|▏| 103/447 [00:00<00:00, 1852.07it/s, Materializing param=

Loading weights: 23%|▏| 103/447 [00:00<00:00, 1845.04it/s, Materializing param=

Loading weights: 23%|▏| 104/447 [00:00<00:00, 1850.53it/s, Materializing param=

Loading weights: 23%|▏| 104/447 [00:00<00:00, 1844.17it/s, Materializing param=

Loading weights: 23%|▏| 105/447 [00:00<00:00, 1851.98it/s, Materializing param=

Loading weights: 23%|▏| 105/447 [00:00<00:00, 1843.63it/s, Materializing param=

Loading weights: 24%|▏| 106/447 [00:00<00:00, 1851.43it/s, Materializing param=

Loading weights: 24%|▏| 106/447 [00:00<00:00, 1845.05it/s, Materializing param=

Loading weights: 24%|▏| 107/447 [00:00<00:00, 1853.05it/s, Materializing param=

Loading weights: 24%|▏| 107/447 [00:00<00:00, 1845.09it/s, Materializing param=

Loading weights: 24%|▏| 108/447 [00:00<00:00, 1852.05it/s, Materializing param=

Loading weights: 24%|▏| 108/447 [00:00<00:00, 1845.55it/s, Materializing param=

Loading weights: 24%|▏| 109/447 [00:00<00:00, 1853.01it/s, Materializing param=

Loading weights: 24%|▏| 109/447 [00:00<00:00, 1842.15it/s, Materializing param=

Loading weights: 25%|▏| 110/447 [00:00<00:00, 1846.76it/s, Materializing param=

Loading weights: 25%|▏| 110/447 [00:00<00:00, 1840.08it/s, Materializing param=

Loading weights: 25%|▏| 111/447 [00:00<00:00, 1847.06it/s, Materializing param=

Loading weights: 25%|▏| 111/447 [00:00<00:00, 1839.98it/s, Materializing param=

Loading weights: 25%|▎| 112/447 [00:00<00:00, 1845.03it/s, Materializing param=

Loading weights: 25%|▎| 112/447 [00:00<00:00, 1838.44it/s, Materializing param=

Loading weights: 25%|▎| 113/447 [00:00<00:00, 1842.02it/s, Materializing param=

Loading weights: 25%|▎| 113/447 [00:00<00:00, 1835.59it/s, Materializing param=

Loading weights: 26%|▎| 114/447 [00:00<00:00, 1840.08it/s, Materializing param=

Loading weights: 26%|▎| 114/447 [00:00<00:00, 1834.02it/s, Materializing param=

Loading weights: 26%|▎| 115/447 [00:00<00:00, 1839.02it/s, Materializing param=

Loading weights: 26%|▎| 115/447 [00:00<00:00, 1832.92it/s, Materializing param=

Loading weights: 26%|▎| 116/447 [00:00<00:00, 1839.22it/s, Materializing param=

Loading weights: 26%|▎| 116/447 [00:00<00:00, 1801.37it/s, Materializing param=

Loading weights: 26%|▎| 117/447 [00:00<00:00, 1807.68it/s, Materializing param=

Loading weights: 26%|▎| 117/447 [00:00<00:00, 1803.09it/s, Materializing param=

Loading weights: 26%|▎| 118/447 [00:00<00:00, 1806.65it/s, Materializing param=

Loading weights: 26%|▎| 118/447 [00:00<00:00, 1801.07it/s, Materializing param=

Loading weights: 27%|▎| 119/447 [00:00<00:00, 1806.27it/s, Materializing param=

Loading weights: 27%|▎| 119/447 [00:00<00:00, 1800.90it/s, Materializing param=

Loading weights: 27%|▎| 120/447 [00:00<00:00, 1806.30it/s, Materializing param=

Loading weights: 27%|▎| 120/447 [00:00<00:00, 1800.70it/s, Materializing param=

Loading weights: 27%|▎| 121/447 [00:00<00:00, 1805.36it/s, Materializing param=

Loading weights: 27%|▎| 121/447 [00:00<00:00, 1799.43it/s, Materializing param=

Loading weights: 27%|▎| 122/447 [00:00<00:00, 1804.68it/s, Materializing param=

Loading weights: 27%|▎| 122/447 [00:00<00:00, 1799.05it/s, Materializing param=

Loading weights: 28%|▎| 123/447 [00:00<00:00, 1805.85it/s, Materializing param=

Loading weights: 28%|▎| 123/447 [00:00<00:00, 1798.91it/s, Materializing param=

Loading weights: 28%|▎| 124/447 [00:00<00:00, 1805.55it/s, Materializing param=

Loading weights: 28%|▎| 124/447 [00:00<00:00, 1798.75it/s, Materializing param=

Loading weights: 28%|▎| 125/447 [00:00<00:00, 1804.94it/s, Materializing param=

Loading weights: 28%|▎| 125/447 [00:00<00:00, 1798.28it/s, Materializing param=

Loading weights: 28%|▎| 126/447 [00:00<00:00, 1714.53it/s, Materializing param=

Loading weights: 28%|▎| 126/447 [00:00<00:00, 1709.88it/s, Materializing param=

Loading weights: 28%|▎| 127/447 [00:00<00:00, 1717.25it/s, Materializing param=

Loading weights: 28%|▎| 127/447 [00:00<00:00, 1712.46it/s, Materializing param=

Loading weights: 29%|▎| 128/447 [00:00<00:00, 1704.83it/s, Materializing param=

Loading weights: 29%|▎| 128/447 [00:00<00:00, 1699.87it/s, Materializing param=

Loading weights: 29%|▎| 129/447 [00:00<00:00, 1674.20it/s, Materializing param=

Loading weights: 29%|▎| 129/447 [00:00<00:00, 1670.27it/s, Materializing param=

Loading weights: 29%|▎| 130/447 [00:00<00:00, 1677.20it/s, Materializing param=

Loading weights: 29%|▎| 130/447 [00:00<00:00, 1672.99it/s, Materializing param=

Loading weights: 29%|▎| 131/447 [00:00<00:00, 1677.73it/s, Materializing param=

Loading weights: 29%|▎| 131/447 [00:00<00:00, 1673.23it/s, Materializing param=

Loading weights: 30%|▎| 132/447 [00:00<00:00, 1671.07it/s, Materializing param=

Loading weights: 30%|▎| 132/447 [00:00<00:00, 1657.60it/s, Materializing param=

Loading weights: 30%|▎| 133/447 [00:00<00:00, 1658.88it/s, Materializing param=

Loading weights: 30%|▎| 133/447 [00:00<00:00, 1646.51it/s, Materializing param=

Loading weights: 30%|▎| 134/447 [00:00<00:00, 1636.06it/s, Materializing param=

Loading weights: 30%|▎| 134/447 [00:00<00:00, 1633.00it/s, Materializing param=

Loading weights: 30%|▎| 135/447 [00:00<00:00, 1638.00it/s, Materializing param=

Loading weights: 30%|▎| 135/447 [00:00<00:00, 1633.84it/s, Materializing param=

Loading weights: 30%|▎| 136/447 [00:00<00:00, 1639.72it/s, Materializing param=

Loading weights: 30%|▎| 136/447 [00:00<00:00, 1634.44it/s, Materializing param=

Loading weights: 31%|▎| 137/447 [00:00<00:00, 1638.97it/s, Materializing param=

Loading weights: 31%|▎| 137/447 [00:00<00:00, 1634.70it/s, Materializing param=

Loading weights: 31%|▎| 138/447 [00:00<00:00, 1640.91it/s, Materializing param=

Loading weights: 31%|▎| 138/447 [00:00<00:00, 1635.41it/s, Materializing param=

Loading weights: 31%|▎| 139/447 [00:00<00:00, 1639.92it/s, Materializing param=

Loading weights: 31%|▎| 139/447 [00:00<00:00, 1635.73it/s, Materializing param=

Loading weights: 31%|▎| 140/447 [00:00<00:00, 1640.19it/s, Materializing param=

Loading weights: 31%|▎| 140/447 [00:00<00:00, 1636.33it/s, Materializing param=

Loading weights: 32%|▎| 141/447 [00:00<00:00, 1639.99it/s, Materializing param=

Loading weights: 32%|▎| 141/447 [00:00<00:00, 1635.81it/s, Materializing param=

Loading weights: 32%|▎| 142/447 [00:00<00:00, 1635.74it/s, Materializing param=

Loading weights: 32%|▎| 142/447 [00:00<00:00, 1631.81it/s, Materializing param=

Loading weights: 32%|▎| 143/447 [00:00<00:00, 1636.44it/s, Materializing param=

Loading weights: 32%|▎| 143/447 [00:00<00:00, 1630.19it/s, Materializing param=

Loading weights: 32%|▎| 144/447 [00:00<00:00, 1634.64it/s, Materializing param=

Loading weights: 32%|▎| 144/447 [00:00<00:00, 1630.65it/s, Materializing param=

Loading weights: 32%|▎| 145/447 [00:00<00:00, 1635.24it/s, Materializing param=

Loading weights: 32%|▎| 145/447 [00:00<00:00, 1631.29it/s, Materializing param=

Loading weights: 33%|▎| 146/447 [00:00<00:00, 1635.92it/s, Materializing param=

Loading weights: 33%|▎| 146/447 [00:00<00:00, 1632.09it/s, Materializing param=

Loading weights: 33%|▎| 147/447 [00:00<00:00, 1636.33it/s, Materializing param=

Loading weights: 33%|▎| 147/447 [00:00<00:00, 1632.63it/s, Materializing param=

Loading weights: 33%|▎| 148/447 [00:00<00:00, 1636.88it/s, Materializing param=

Loading weights: 33%|▎| 148/447 [00:00<00:00, 1633.25it/s, Materializing param=

Loading weights: 33%|▎| 149/447 [00:00<00:00, 1637.28it/s, Materializing param=

Loading weights: 33%|▎| 149/447 [00:00<00:00, 1633.43it/s, Materializing param=

Loading weights: 34%|▎| 150/447 [00:00<00:00, 1637.99it/s, Materializing param=

Loading weights: 34%|▎| 150/447 [00:00<00:00, 1634.23it/s, Materializing param=

Loading weights: 34%|▎| 151/447 [00:00<00:00, 1638.66it/s, Materializing param=

Loading weights: 34%|▎| 151/447 [00:00<00:00, 1634.97it/s, Materializing param=

Loading weights: 34%|▎| 152/447 [00:00<00:00, 1639.54it/s, Materializing param=

Loading weights: 34%|▎| 152/447 [00:00<00:00, 1634.81it/s, Materializing param=

Loading weights: 34%|▎| 153/447 [00:00<00:00, 1639.22it/s, Materializing param=

Loading weights: 34%|▎| 153/447 [00:00<00:00, 1635.55it/s, Materializing param=

Loading weights: 34%|▎| 154/447 [00:00<00:00, 1639.59it/s, Materializing param=

Loading weights: 34%|▎| 154/447 [00:00<00:00, 1635.91it/s, Materializing param=

Loading weights: 35%|▎| 155/447 [00:00<00:00, 1641.07it/s, Materializing param=

Loading weights: 35%|▎| 155/447 [00:00<00:00, 1636.39it/s, Materializing param=

Loading weights: 35%|▎| 156/447 [00:00<00:00, 1640.40it/s, Materializing param=

Loading weights: 35%|▎| 156/447 [00:00<00:00, 1636.43it/s, Materializing param=

Loading weights: 35%|▎| 157/447 [00:00<00:00, 1640.70it/s, Materializing param=

Loading weights: 35%|▎| 157/447 [00:00<00:00, 1635.24it/s, Materializing param=

Loading weights: 35%|▎| 158/447 [00:00<00:00, 1639.04it/s, Materializing param=

Loading weights: 35%|▎| 158/447 [00:00<00:00, 1635.57it/s, Materializing param=

Loading weights: 36%|▎| 159/447 [00:00<00:00, 1640.33it/s, Materializing param=

Loading weights: 36%|▎| 159/447 [00:00<00:00, 1636.91it/s, Materializing param=

Loading weights: 36%|▎| 160/447 [00:00<00:00, 1640.64it/s, Materializing param=

Loading weights: 36%|▎| 160/447 [00:00<00:00, 1636.93it/s, Materializing param=

Loading weights: 36%|▎| 161/447 [00:00<00:00, 1641.91it/s, Materializing param=

Loading weights: 36%|▎| 161/447 [00:00<00:00, 1637.48it/s, Materializing param=

Loading weights: 36%|▎| 162/447 [00:00<00:00, 1642.73it/s, Materializing param=

Loading weights: 36%|▎| 162/447 [00:00<00:00, 1639.27it/s, Materializing param=

Loading weights: 36%|▎| 163/447 [00:00<00:00, 1644.49it/s, Materializing param=

Loading weights: 36%|▎| 163/447 [00:00<00:00, 1640.11it/s, Materializing param=

Loading weights: 37%|▎| 164/447 [00:00<00:00, 1645.38it/s, Materializing param=

Loading weights: 37%|▎| 164/447 [00:00<00:00, 1642.01it/s, Materializing param=

Loading weights: 37%|▎| 165/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 37%|▎| 165/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 37%|▎| 165/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 37%|▎| 166/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 37%|▎| 166/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 37%|▎| 167/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 37%|▎| 167/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 168/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 168/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 169/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 169/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 170/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 170/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 171/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 171/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 172/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 38%|▍| 172/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 39%|▍| 173/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 39%|▍| 173/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 39%|▍| 174/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 39%|▍| 174/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 39%|▍| 175/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 39%|▍| 175/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 39%|▍| 176/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 39%|▍| 176/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 177/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 177/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 178/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 178/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 179/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 179/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 180/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 180/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 181/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 40%|▍| 181/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 41%|▍| 182/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 41%|▍| 182/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 41%|▍| 183/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 41%|▍| 183/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 41%|▍| 184/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 41%|▍| 184/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 41%|▍| 185/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 41%|▍| 185/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 42%|▍| 186/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 42%|▍| 186/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 42%|▍| 187/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 42%|▍| 187/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 42%|▍| 188/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 42%|▍| 188/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 42%|▍| 189/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 42%|▍| 189/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 190/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 190/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 191/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 191/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 192/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 192/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 193/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 193/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 194/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 43%|▍| 194/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 44%|▍| 195/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 44%|▍| 195/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 44%|▍| 196/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 44%|▍| 196/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 44%|▍| 197/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 44%|▍| 197/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 44%|▍| 198/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 44%|▍| 198/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 199/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 199/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 200/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 200/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 201/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 201/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 202/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 202/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 203/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 45%|▍| 203/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 46%|▍| 204/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 46%|▍| 204/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 46%|▍| 205/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 46%|▍| 205/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 46%|▍| 206/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 46%|▍| 206/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 46%|▍| 207/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 46%|▍| 207/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 208/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 208/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 209/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 209/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 210/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 210/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 211/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 211/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 212/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 47%|▍| 212/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 48%|▍| 213/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 48%|▍| 213/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 48%|▍| 214/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 48%|▍| 214/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 48%|▍| 215/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 48%|▍| 215/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 48%|▍| 216/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 48%|▍| 216/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 217/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 217/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 218/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 218/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 219/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 219/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 220/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 220/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 221/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 49%|▍| 221/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 50%|▍| 222/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 50%|▍| 222/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 50%|▍| 223/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 50%|▍| 223/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 50%|▌| 224/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 50%|▌| 224/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 50%|▌| 225/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 50%|▌| 225/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 226/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 226/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 227/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 227/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 228/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 228/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 229/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 229/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 230/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 51%|▌| 230/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 52%|▌| 231/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 52%|▌| 231/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 52%|▌| 232/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 52%|▌| 232/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 52%|▌| 233/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 52%|▌| 233/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 52%|▌| 234/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 52%|▌| 234/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 235/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 235/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 236/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 236/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 237/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 237/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 238/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 238/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 239/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 53%|▌| 239/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 54%|▌| 240/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 54%|▌| 240/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 54%|▌| 241/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 54%|▌| 241/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 54%|▌| 242/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 54%|▌| 242/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 54%|▌| 243/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 54%|▌| 243/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 244/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 244/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 245/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 245/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 246/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 246/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 247/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 247/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 248/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 55%|▌| 248/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 56%|▌| 249/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 56%|▌| 249/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 56%|▌| 250/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 56%|▌| 250/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 56%|▌| 251/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 56%|▌| 251/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 56%|▌| 252/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 56%|▌| 252/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 253/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 253/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 254/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 254/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 255/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 255/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 256/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 256/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 257/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 57%|▌| 257/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 58%|▌| 258/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 58%|▌| 258/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 58%|▌| 259/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 58%|▌| 259/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 58%|▌| 260/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 58%|▌| 260/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 58%|▌| 261/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 58%|▌| 261/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 59%|▌| 262/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 59%|▌| 262/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 59%|▌| 263/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 59%|▌| 263/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 59%|▌| 264/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 59%|▌| 264/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 59%|▌| 265/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 59%|▌| 265/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 266/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 266/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 267/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 267/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 268/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 268/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 269/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 269/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 270/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 60%|▌| 270/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 61%|▌| 271/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 61%|▌| 271/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 61%|▌| 272/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 61%|▌| 272/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 61%|▌| 273/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 61%|▌| 273/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 61%|▌| 274/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 61%|▌| 274/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 275/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 275/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 276/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 276/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 277/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 277/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 278/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 278/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 279/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 62%|▌| 279/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 63%|▋| 280/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 63%|▋| 280/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 63%|▋| 281/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 63%|▋| 281/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 63%|▋| 282/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 63%|▋| 282/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 63%|▋| 283/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 63%|▋| 283/447 [00:00<00:00, 1647.87it/s, Materializing param=

2026-07-31 02:12:41,499 - INFO - Shot 4 sharpest frame pruned (sim=0.9919)
Loading weights: 64%|▋| 284/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 284/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 285/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 285/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 286/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 286/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 287/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 287/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 288/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 64%|▋| 288/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 65%|▋| 289/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 65%|▋| 289/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 65%|▋| 290/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 65%|▋| 290/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 65%|▋| 291/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 65%|▋| 291/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 65%|▋| 292/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 65%|▋| 292/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 293/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 293/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 294/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 294/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 295/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 295/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 296/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 296/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 297/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 66%|▋| 297/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 67%|▋| 298/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 67%|▋| 298/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 67%|▋| 299/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 67%|▋| 299/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 67%|▋| 300/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 67%|▋| 300/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 67%|▋| 301/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 67%|▋| 301/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 302/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 302/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 303/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 303/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 304/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 304/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 305/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 305/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 306/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 68%|▋| 306/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 69%|▋| 307/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 69%|▋| 307/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 69%|▋| 308/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 69%|▋| 308/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 69%|▋| 309/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 69%|▋| 309/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 69%|▋| 310/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 69%|▋| 310/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 311/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 311/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 312/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 312/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 313/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 313/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 314/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 314/447 [00:00<00:00, 1647.87it/s, Materializing param=2026-07-31 02:12:41,517 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/commits/main "HTTP/1.1 200 OK"

Loading weights: 70%|▋| 315/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 70%|▋| 315/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 71%|▋| 316/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 71%|▋| 316/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 71%|▋| 317/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 71%|▋| 317/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 71%|▋| 318/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 71%|▋| 318/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 71%|▋| 319/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 71%|▋| 319/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 320/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 320/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 321/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 321/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 322/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 322/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 323/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 323/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 324/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 72%|▋| 324/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 73%|▋| 325/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 73%|▋| 325/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 73%|▋| 326/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 73%|▋| 326/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 73%|▋| 327/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 73%|▋| 327/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 73%|▋| 328/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 73%|▋| 328/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 329/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 329/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 330/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 330/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 331/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 331/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 332/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 332/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 333/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 74%|▋| 333/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 75%|▋| 334/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 75%|▋| 334/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 75%|▋| 335/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 75%|▋| 335/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 75%|▊| 336/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 75%|▊| 336/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 75%|▊| 337/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 75%|▊| 337/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 76%|▊| 338/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 76%|▊| 338/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 76%|▊| 339/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 76%|▊| 339/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 76%|▊| 340/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 76%|▊| 340/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 76%|▊| 341/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 76%|▊| 341/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 342/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 342/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 343/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 343/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 344/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 344/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 345/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 345/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 346/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 77%|▊| 346/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 78%|▊| 347/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 78%|▊| 347/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 78%|▊| 348/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 78%|▊| 348/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 78%|▊| 349/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 78%|▊| 349/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 78%|▊| 350/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 78%|▊| 350/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 351/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 351/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 352/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 352/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 353/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 353/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 354/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 354/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 355/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 79%|▊| 355/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 80%|▊| 356/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 80%|▊| 356/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 80%|▊| 357/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 80%|▊| 357/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 80%|▊| 358/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 80%|▊| 358/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 80%|▊| 359/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 80%|▊| 359/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 360/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 360/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 361/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 361/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 362/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 362/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 363/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 363/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 364/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 81%|▊| 364/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 82%|▊| 365/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 82%|▊| 365/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 82%|▊| 366/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 82%|▊| 366/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 82%|▊| 367/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 82%|▊| 367/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 82%|▊| 368/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 82%|▊| 368/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 83%|▊| 369/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 83%|▊| 369/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 83%|▊| 370/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 83%|▊| 370/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 83%|▊| 371/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 83%|▊| 371/447 [00:00<00:00, 1647.87it/s, Materializing param=

Loading weights: 83%|▊| 372/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 83%|▊| 372/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 83%|▊| 372/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 83%|▊| 373/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 83%|▊| 373/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 84%|▊| 374/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 84%|▊| 374/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 84%|▊| 375/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 84%|▊| 375/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 84%|▊| 376/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 84%|▊| 376/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 84%|▊| 377/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 84%|▊| 377/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 378/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 378/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 379/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 379/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 380/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 380/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 381/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 381/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 382/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 85%|▊| 382/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 86%|▊| 383/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 86%|▊| 383/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 86%|▊| 384/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 86%|▊| 384/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 86%|▊| 385/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 86%|▊| 385/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 86%|▊| 386/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 86%|▊| 386/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 387/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 387/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 388/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 388/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 389/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 389/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 390/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 390/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 391/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 87%|▊| 391/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 88%|▉| 392/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 88%|▉| 392/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 88%|▉| 393/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 88%|▉| 393/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 88%|▉| 394/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 88%|▉| 394/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 88%|▉| 395/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 88%|▉| 395/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 396/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 396/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 397/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 397/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 398/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 398/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 399/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 399/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 400/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 89%|▉| 400/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 90%|▉| 401/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 90%|▉| 401/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 90%|▉| 402/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 90%|▉| 402/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 90%|▉| 403/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 90%|▉| 403/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 90%|▉| 404/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 90%|▉| 404/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 405/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 405/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 406/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 406/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 407/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 407/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 408/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 408/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 409/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 91%|▉| 409/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 92%|▉| 410/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 92%|▉| 410/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 92%|▉| 411/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 92%|▉| 411/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 92%|▉| 412/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 92%|▉| 412/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 92%|▉| 413/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 92%|▉| 413/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 93%|▉| 414/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 93%|▉| 414/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 93%|▉| 415/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 93%|▉| 415/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 93%|▉| 416/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 93%|▉| 416/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 93%|▉| 417/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 93%|▉| 417/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 418/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 418/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 419/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 419/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 420/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 420/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 421/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 421/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 422/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 94%|▉| 422/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 95%|▉| 423/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 95%|▉| 423/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 95%|▉| 424/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 95%|▉| 424/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 95%|▉| 425/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 95%|▉| 425/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 95%|▉| 426/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 95%|▉| 426/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 427/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 427/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 428/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 428/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 429/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 429/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 430/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 430/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 431/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 96%|▉| 431/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 97%|▉| 432/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 97%|▉| 432/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 97%|▉| 433/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 97%|▉| 433/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 97%|▉| 434/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 97%|▉| 434/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 97%|▉| 435/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 97%|▉| 435/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 436/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 436/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 437/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 437/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 438/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 438/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 439/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 439/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 440/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 98%|▉| 440/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 99%|▉| 441/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 99%|▉| 441/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 99%|▉| 442/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 99%|▉| 442/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 99%|▉| 443/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 99%|▉| 443/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 99%|▉| 444/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 99%|▉| 444/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 100%|▉| 445/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 100%|▉| 445/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 100%|▉| 446/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 100%|▉| 446/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 100%|█| 447/447 [00:00<00:00, 1894.94it/s, Materializing param=

Loading weights: 100%|█| 447/447 [00:00<00:00, 1947.54it/s, Materializing param=
2026-07-31 02:12:41,760 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/discussions?p=0 "HTTP/1.1 200 OK"
2026-07-31 02:12:41,839 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:41,920 - INFO - Shot 88 sharpest frame pruned (sim=0.9939)
2026-07-31 02:12:41,987 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/commits/refs%2Fpr%2F3 "HTTP/1.1 200 OK"
2026-07-31 02:12:42,030 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/tokenizer_config.json "HTTP/1.1 200 OK"
2026-07-31 02:12:42,200 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/refs%2Fpr%2F3/model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-07-31 02:12:42,233 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/tokenizer_config.json "HTTP/1.1 200 OK"

tokenizer_config.json: 100%|███████████████████| 384/384 [00:00<00:00, 1.54MB/s]
2026-07-31 02:12:42,413 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/refs%2Fpr%2F3/model.safetensors "HTTP/1.1 302 Found"
2026-07-31 02:12:42,479 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-07-31 02:12:42,622 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/xet-read-token/79b58ed25fc00386262a2bea4b19fd21dc4310a0 "HTTP/1.1 200 OK"

model.safetensors: 0%| | 0.00/614M [00:00<?, ?B/s]2026-07-31 02:12:42,700 - INFO - HTTP Request: GET https://huggingface.co/api/models/laion/clap-htsat-unfused/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"

model.safetensors: 0%| | 0.00/614M [00:00<?, ?B/s]2026-07-31 02:12:42,900 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/vocab.json "HTTP/1.1 307 Temporary Redirect"

model.safetensors: 0%| | 0.00/614M [00:00<?, ?B/s]2026-07-31 02:12:43,097 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/vocab.json "HTTP/1.1 200 OK"

model.safetensors: 0%| | 480k/614M [00:00<04:16, 2.39MB/s]2026-07-31 02:12:43,294 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/vocab.json "HTTP/1.1 200 OK"

vocab.json: 798kB [00:00, 14.1MB/s]A
2026-07-31 02:12:43,556 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/merges.txt "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:43,563 - INFO - Shot 90 sharpest frame pruned (sim=0.9969)
2026-07-31 02:12:43,718 - INFO - Shot 5 sharpest frame pruned (sim=0.9960)
2026-07-31 02:12:43,753 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/merges.txt "HTTP/1.1 200 OK"
2026-07-31 02:12:43,949 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/merges.txt "HTTP/1.1 200 OK"

merges.txt: 456kB [00:00, 9.28MB/s]A
2026-07-31 02:12:44,208 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/tokenizer.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:44,403 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/tokenizer.json "HTTP/1.1 200 OK"
2026-07-31 02:12:44,596 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/tokenizer.json "HTTP/1.1 200 OK"

tokenizer.json: 2.11MB [00:00, 69.0MB/s]A
2026-07-31 02:12:44,835 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/added_tokens.json "HTTP/1.1 404 Not Found"
2026-07-31 02:12:45,041 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/special_tokens_map.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:45,240 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/special_tokens_map.json "HTTP/1.1 200 OK"
2026-07-31 02:12:45,437 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/special_tokens_map.json "HTTP/1.1 200 OK"

special_tokens_map.json: 100%|██████████████████| 280/280 [00:00<00:00, 960kB/s]
2026-07-31 02:12:45,646 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/chat_template.jinja "HTTP/1.1 404 Not Found"
2026-07-31 02:12:46,226 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-07-31 02:12:46,425 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:46,619 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/preprocessor_config.json "HTTP/1.1 200 OK"
2026-07-31 02:12:46,815 - INFO - HTTP Request: GET https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/preprocessor_config.json "HTTP/1.1 200 OK"

preprocessor_config.json: 100%|█████████████████| 541/541 [00:00<00:00, 434kB/s]
2026-07-31 02:12:47,040 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-07-31 02:12:47,258 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:47,274 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/preprocessor_config.json "HTTP/1.1 200 OK"
2026-07-31 02:12:47,489 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-07-31 02:12:47,695 - INFO - HTTP Request: HEAD https://huggingface.co/laion/clap-htsat-unfused/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-07-31 02:12:47,707 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/laion/clap-htsat-unfused/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/preprocessor_config.json "HTTP/1.1 200 OK"
2026-07-31 02:12:47,710 - INFO - CLAP zero-shot model loaded successfully.
/usr/local/lib/python3.12/dist-packages/torch/hub.py:247: UserWarning: You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to load(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or load(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
\_check_repo_is_trusted(
Downloading: "https://github.com/snakers4/silero-vad/zipball/master" to /root/.cache/torch/hub/master.zip
2026-07-31 02:12:50,276 - INFO - Silero VAD model loaded successfully.

model.safetensors: 44%|█████████▏ | 269M/614M [00:09<00:10, 31.5MB/s]

model.safetensors: 100%|█████████████████████| 614M/614M [00:10<00:00, 59.0MB/s]
2026-07-31 02:12:53,135 - INFO - Extracted 139 keyframe artifacts to 'processed_data/1_frames'.
malloc(): unaligned tcache chunk detected
