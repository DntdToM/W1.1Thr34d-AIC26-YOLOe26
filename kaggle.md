# AIC 2026 Kaggle GPU Executor Cells

Dưới đây là mã nguồn chuẩn xác đã được căn lề (indent) và sửa lỗi cú pháp Python (như lỗi `\*` khi tính dung lượng) để sếp paste thẳng vào các Cell của Kaggle.

### Cell 1: Ingest source code from GitHub

```python
import os

# Đảm bảo chúng ta luôn ở thư mục gốc của Kaggle
if os.path.exists("/kaggle/working"):
    os.chdir("/kaggle/working")

GITHUB_TOKEN = ""

# Lấy token từ Kaggle Secrets (nếu có)
try:
    from kaggle_secrets import UserSecretsClient
    secret_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if secret_token:
        GITHUB_TOKEN = secret_token
        print("Successfully loaded GITHUB_TOKEN from Kaggle Secrets.")
except Exception:
    pass

# Chèn token vào URL nếu đây là Private Repo, ngược lại dùng URL thường
if GITHUB_TOKEN:
    REPO_URL = f"https://{GITHUB_TOKEN}@github.com/DntdToM/W1.1Thr34d---copy.git"
else:
    REPO_URL = "https://github.com/DntdToM/W1.1Thr34d-AIC26-YOLOe26.git"

if not os.path.exists("W1.1Thr34d-AIC26-YOLOe26"):
    print("Cloning repository from GitHub...")
    !git clone {REPO_URL}
    %cd W1.1Thr34d-AIC26-YOLOe26
else:
    print("Pulling latest commits from GitHub...")
    %cd W1.1Thr34d-AIC26-YOLOe26
    !git remote set-url origin {REPO_URL}
    !git pull origin main
```

### Cell 2: Install system dependencies

```python
!apt-get update && apt-get install -y ffmpeg libsm6 libxext6
```

### Cell 3: Install Python dependencies

```python
!pip install -r requirements.txt
```

### Cell 4: Environment setup, API keys, and Data/Model downloads

```python
import os
import zipfile
import gdown

# Groq keys
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    # os.environ["GROQ_API_KEY"] = user_secrets.get_secret("GROQ_API_KEY")
    os.environ["GEMINI_API_KEY"] = user_secrets.get_secret("GEMINI_API_KEY")
    os.environ["GDRIVE_FILE_ID"] = user_secrets.get_secret("GDRIVE_FILE_ID")
    print("API keys loaded successfully from Kaggle Secrets.")
except Exception:
    pass

GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID")
TARGET_DIR = "data/official_videos"
os.makedirs(TARGET_DIR, exist_ok=True)

if GDRIVE_FILE_ID:
    print(f"Downloading video dataset from Google Drive (ID: {GDRIVE_FILE_ID})...")
    zip_path = "videos_dataset.zip"

    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        zip_path,
        quiet=False
    )

    print(f"Extracting {zip_path} into {TARGET_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)

    print("Dataset extraction complete.")
else:
    print("Missing GDRIVE_FILE_ID; dataset access denied.")

if not os.path.exists("models/siglip-base-patch16-224") or not os.path.exists("models/bge-m3"):
    print("Downloading model weights to local models/ directory...")
    !python scripts/download_models.py
else:
    print("Model weights verified in models/ directory.")

# Tải trọng số YOLOE-26L-PF (Bản Large, Prompt-Free) trực tiếp từ Ultralytics
yoloe_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg-pf.pt"
yoloe_path = "/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/yoloe-26l-seg-pf.pt"
if not os.path.exists(yoloe_path):
    print("Downloading YOLOE-26L-PF...")
    !wget -q {yoloe_url} -O {yoloe_path}

```

### Cell 5: Execute Offline Indexing Pipeline

```python
!python run_pipeline.py
```

### Cell 6: Archive processed output artifacts

```python
import os
import shutil

OUTPUT_ZIP_NAME = "aic-5-yoloe26"
SOURCE_DIR = "processed_data"

if os.path.exists(SOURCE_DIR):
    print(f"Archiving directory '{SOURCE_DIR}' into '{OUTPUT_ZIP_NAME}.zip'...")
    shutil.make_archive(OUTPUT_ZIP_NAME, 'zip', SOURCE_DIR)

    zip_file_path = f"{OUTPUT_ZIP_NAME}.zip"
    size_mb = os.path.getsize(zip_file_path) / (1024 * 1024)
    print(f"Archive created successfully: {zip_file_path} ({size_mb:.2f} MB)")
else:
    print(f"Directory '{SOURCE_DIR}' not found.")
```
