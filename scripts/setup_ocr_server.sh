#!/bin/bash
# setup_ocr_server.sh
# Script to setup an isolated virtual environment for PaddleOCR 3.7.0 and start the local OCR API server.

set -e

# Define environment path
ENV_DIR="/kaggle/working/ocr_env"
SERVER_SCRIPT="/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/src/preprocessing/ocr_server.py"
LOG_FILE="/kaggle/working/ocr_server.log"

# Use Conda environment instead of venv
CONDA_ENV_NAME="ocr_env"
CONDA_EXE="/opt/conda/bin/conda"

echo "=========================================="
echo "    PADDLEOCR 3.7.0 CONDA ISOLATION       "
echo "=========================================="

if $CONDA_EXE info --envs | grep -q "$CONDA_ENV_NAME"; then
    echo "[INFO] Conda environment '$CONDA_ENV_NAME' already exists."
else
    echo "[INFO] Creating new conda environment '$CONDA_ENV_NAME' (Python 3.9)..."
    $CONDA_EXE create -y -n $CONDA_ENV_NAME python=3.9
    
    echo "[INFO] Installing libgl1-mesa-glx for OpenCV..."
    apt-get update && apt-get install -y libgl1-mesa-glx
    
    echo "[INFO] Installing OCR dependencies into conda environment..."
    $CONDA_EXE run -n $CONDA_ENV_NAME pip install --upgrade pip
    $CONDA_EXE run -n $CONDA_ENV_NAME pip install flask requests
    $CONDA_EXE run -n $CONDA_ENV_NAME pip install paddlepaddle-gpu
    $CONDA_EXE run -n $CONDA_ENV_NAME pip install paddleocr==3.7.0 paddlex
    
    echo "[SUCCESS] Installation complete."
fi

echo "[INFO] Starting OCR Microservice Server in Conda..."

# Kill any existing server on port 5050
if lsof -Pi :5050 -sTCP:LISTEN -t >/dev/null ; then
    echo "[WARNING] Port 5050 is already in use. Killing old process..."
    lsof -Pi :5050 -sTCP:LISTEN -t | xargs kill -9
fi

# Run the server in the background using the isolated Conda Python
# CRITICAL: unset PYTHONPATH to prevent Kaggle's base environment leaking into our isolated Conda environment
unset PYTHONPATH
nohup $CONDA_EXE run -n $CONDA_ENV_NAME python "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &

echo "[SUCCESS] Server is starting in the background."
echo "[INFO] Logs are being written to $LOG_FILE"
echo "[INFO] Wait a few seconds for the OCR Engine to initialize..."
sleep 10
echo "[INFO] You can verify it by running: curl http://localhost:5050/health"
