import os
import json
import argparse
from paddleocr import PaddleOCR
import logging

# Tắt bớt log rác của thư viện
logging.getLogger('ppocr').setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Thư mục chứa ảnh cần nhận diện")
    parser.add_argument("--output_file", type=str, required=True, help="Đường dẫn file JSON đầu ra")
    args = parser.parse_args()

    print("[OCR_ENV] Đang khởi tạo PaddleOCR...")
    # Khởi tạo model (chỉ load 1 lần khi script này được gọi)
    ocr = PaddleOCR(use_textline_orientation=True, lang='vi')
    
    results = {}
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    print(f"[OCR_ENV] Bắt đầu xử lý ảnh trong: {args.input_dir}")
    if os.path.exists(args.input_dir):
        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(args.input_dir, filename)
                
                # Chạy nhận diện
                result = ocr.ocr(img_path, cls=True)
                
                # Trích xuất text từ kết quả
                extracted_text = []
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        extracted_text.append(text)
                
                results[filename] = " ".join(extracted_text)

    # Lưu kết quả ra JSON
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[OCR_ENV] Đã lưu kết quả tại: {args.output_file}")

if __name__ == "__main__":
    main()
