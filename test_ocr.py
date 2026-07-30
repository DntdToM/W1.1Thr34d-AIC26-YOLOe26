import os
import sys
import json

try:
    from paddleocr import PaddleOCR
    print("[INFO] Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='vi', use_mkldnn=False, use_gpu=False)
    print("[SUCCESS] PaddleOCR initialized successfully!")
    
    img_path = r"c:\Users\Hacker.Nasa\Documents\AIC25\aic2026_yoloe26\ahjbfi.jpg"
    print(f"[INFO] Testing OCR on image: {img_path}")
    
    result = ocr.ocr(img_path, cls=False)
    
    extracted_data = []
    if result and len(result) > 0 and result[0]:
        for res_item in result[0]:
            if not res_item or len(res_item) < 2:
                continue
            box = res_item[0]
            text_info = res_item[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                text_str = str(text_info[0]).strip()
                extracted_data.append(text_str)
                
    print("\n[RESULT] Extracted Text:")
    print(json.dumps(extracted_data, ensure_ascii=False, indent=2))
    
except Exception as e:
    print(f"[ERROR] Failed: {e}")
    sys.exit(1)
