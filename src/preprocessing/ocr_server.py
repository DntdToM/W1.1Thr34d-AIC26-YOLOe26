import os
import logging
from flask import Flask, request, jsonify

# Flask App Initialization
app = Flask(__name__)

# Disable PaddleX connectivity check to reduce logs and speed up startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable for OCR Engine to ensure it loads once
ocr_engine = None

def init_ocr():
    global ocr_engine
    if ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            # Initialize OCR engine (PaddleOCR 3.7.0 style)
            ocr_engine = PaddleOCR(use_textline_orientation=True, lang='vi')
            logger.info("PaddleOCR engine initialized successfully on isolated server.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise e

@app.before_request
def before_first_request():
    # Attempt to load the model on the first request if not loaded
    init_ocr()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to ensure server is running and model is loaded."""
    if ocr_engine is not None:
        return jsonify({"status": "healthy", "message": "OCR Engine is loaded."}), 200
    else:
        return jsonify({"status": "unhealthy", "message": "OCR Engine is NOT loaded."}), 503

@app.route('/ocr', methods=['POST'])
def process_ocr():
    """Process an image path and return OCR results."""
    global ocr_engine
    if ocr_engine is None:
        return jsonify({"error": "OCR Engine is not initialized."}), 500

    data = request.get_json()
    if not data or 'image_path' not in data:
        return jsonify({"error": "Missing 'image_path' in JSON body."}), 400

    image_path = data['image_path']

    if not os.path.exists(image_path):
        return jsonify({"error": f"Image file not found: {image_path}"}), 404

    try:
        # Perform prediction (PaddleOCR 3.7.0 uses predict instead of ocr)
        result = ocr_engine.predict(image_path)
        
        extracted_data = []
        # Parse PaddleOCR v3.7.0 / v2.x result format
        if result and len(result) > 0 and result[0]:
            for res_item in result[0]:
                if not res_item or len(res_item) < 2:
                    continue
                
                box = res_item[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text_info = res_item[1]
                
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                    text_str = str(text_info[0]).strip()
                    if text_str:
                        extracted_data.append({"text": text_str, "box": box})
                elif isinstance(text_info, str) and text_info.strip():
                    extracted_data.append({"text": text_info.strip(), "box": box})
                    
        return jsonify({"status": "success", "results": extracted_data}), 200

    except Exception as e:
        logger.error(f"Error processing OCR for {image_path}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize the engine immediately on boot
    init_ocr()
    logger.info("Starting OCR Microservice on port 5050...")
    # Disable threaded mode if PaddlePaddle is not thread-safe in this setup, 
    # but for typical single-GPU processing, it's safer to use single thread or careful locking.
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=False)
