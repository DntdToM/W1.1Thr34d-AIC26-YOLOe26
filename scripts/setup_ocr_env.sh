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
