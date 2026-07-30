#!/bin/bash
# setup_ocr_server.sh
# Script to setup an isolated virtual environment for PaddleOCR 3.7.0 and start the local OCR API server.

set -e

# Define environment path
ENV_DIR="/kaggle/working/ocr_env"
SERVER_SCRIPT="/kaggle/working/W1.1Thr34d-AIC26-YOLOe26/src/preprocessing/ocr_server.py"
LOG_FILE="/kaggle/working/ocr_server.log"

echo "=========================================="
echo "    PADDLEOCR 3.7.0 ISOLATION SETUP       "
echo "=========================================="

if [ -d "$ENV_DIR" ]; then
    echo "[INFO] Isolated environment already exists at $ENV_DIR"
else
    echo "[INFO] Creating new virtual environment at $ENV_DIR..."
    # Use --without-pip to avoid Debian ensurepip bugs on Kaggle
    python3 -m venv --without-pip "$ENV_DIR"
    
    echo "[INFO] Installing pip into virtual environment..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$ENV_DIR/bin/python"
    
    echo "[INFO] Activating environment and installing PaddleOCR 3.7.0..."
    source "$ENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Flask for the API Server
    pip install flask requests
    
    # Install paddlepaddle-gpu (Kaggle uses Linux with CUDA)
    echo "[INFO] Installing paddlepaddle-gpu..."
    pip install paddlepaddle-gpu
    
    # Install PaddleOCR and Paddlex
    echo "[INFO] Installing paddleocr==3.7.0 and paddlex..."
    pip install paddleocr==3.7.0 paddlex
    
    deactivate
    echo "[SUCCESS] Installation complete."
fi

echo "[INFO] Starting OCR Microservice Server..."

# Kill any existing server on port 5050
if lsof -Pi :5050 -sTCP:LISTEN -t >/dev/null ; then
    echo "[WARNING] Port 5050 is already in use. Killing old process..."
    lsof -Pi :5050 -sTCP:LISTEN -t | xargs kill -9
fi

# Run the server in the background using the isolated Python
nohup "$ENV_DIR/bin/python" "$SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &

echo "[SUCCESS] Server is starting in the background."
echo "[INFO] Logs are being written to $LOG_FILE"
echo "[INFO] Wait a few seconds for the OCR Engine to initialize..."
sleep 10
echo "[INFO] You can verify it by running: curl http://localhost:5050/health"
