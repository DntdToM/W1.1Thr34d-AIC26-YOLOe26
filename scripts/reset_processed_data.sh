#!/bin/bash
# Reset processed_data script for AIC 2026 Retrieval System
echo "=========================================================="
echo "=== RESETTING PIPELINE DATA (processed_data/ 1_, 2_, 3_) ==="
echo "=========================================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_DIR" || exit 1

for folder in "processed_data/1_frames" "processed_data/2_embeddings" "processed_data/3_metadata"; do
    if [ -d "$folder" ]; then
        echo "Đang dọn dẹp thư mục: $folder ..."
        find "$folder" -type f ! -name '.gitkeep' -delete
        find "$folder" -type d ! -path "$folder" -exec rm -rf {} + 2>/dev/null
    fi
done

echo "[SUCCESS] Đã dọn dẹp sạch sẽ toàn bộ dữ liệu cũ trong processed_data!"
echo "=========================================================="
