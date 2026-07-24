"""Vision analytics module for object detection, OCR extraction, and window-based LLM context summarization."""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import yaml
import requests
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VisionAnalytics")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


COCO_VI_MAP = {
    "person": "người", "bicycle": "xe đạp", "car": "ô tô", "motorcycle": "xe máy", "airplane": "máy bay",
    "bus": "xe buýt", "train": "tàu hỏa", "truck": "xe tải", "boat": "thuyền", "traffic light": "đèn giao thông",
    "fire hydrant": "vòi chữa cháy", "stop sign": "biển dừng", "parking meter": "đồng hồ đỗ xe", "bench": "ghế dài",
    "bird": "chim", "cat": "mèo", "dog": "chó", "horse": "ngựa", "sheep": "cừu", "cow": "bò", "elephant": "voi",
    "bear": "gấu", "zebra": "ngựa vằn", "giraffe": "hươu cao cổ", "backpack": "ba lô", "umbrella": "ô dù",
    "handbag": "túi xách", "tie": "cà vạt", "suitcase": "va li", "frisbee": "đĩa bay", "skis": "gậy trượt tuyết",
    "snowboard": "ván trượt tuyết", "sports ball": "bóng thể thao", "kite": "diều", "baseball bat": "gậy bóng chày",
    "baseball glove": "găng tay bóng chày", "skateboard": "ván trượt", "surfboard": "ván lướt sóng",
    "tennis racket": "vợt tennis", "bottle": "chai nước", "wine glass": "ly rượu", "cup": "cốc", "fork": "nĩa",
    "knife": "dao", "spoon": "thìa", "bowl": "bát", "banana": "chuối", "apple": "táo", "sandwich": "bánh mì sandwich",
    "orange": "cam", "broccoli": "súp lơ", "carrot": "củ cà rốt", "hot dog": "bánh mì xúc xích", "pizza": "bánh pizza",
    "donut": "bánh donut", "cake": "bánh ngọt", "chair": "ghế", "couch": "ghế sofa", "potted plant": "chậu cây",
    "bed": "giường", "dining table": "bàn ăn", "toilet": "bồn cầu", "tv": "tivi", "laptop": "máy tính xách tay",
    "mouse": "chuột máy tính", "remote": "điều khiển", "keyboard": "bàn phím", "cell phone": "điện thoại",
    "microwave": "lò vi sóng", "oven": "lò nướng", "toaster": "máy nướng bánh mì", "sink": "bồn rửa",
    "refrigerator": "tủ lạnh", "book": "sách", "clock": "đồng hồ", "vase": "bình hoa", "scissors": "kéo",
    "teddy bear": "gấu bông", "hair drier": "máy sấy tóc", "toothbrush": "bàn chải đánh răng"
}


def _clean_llm_response(text: str) -> str:
    """Sanitize LLM output by removing intro phrases and markdown list formatting."""
    if not text:
        return ""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        if any(lower_line.startswith(prefix) for prefix in [
            "sau khi", "dưới đây", "kết quả", "tôi đã", "đây là", "bảng tóm tắt",
            "1. sửa lỗi", "2. lọc", "dữ liệu đã"
        ]):
            continue
        cleaned_line = line.lstrip("-*•1234567890. ").strip()
        if cleaned_line and len(cleaned_line) > 2:
            cleaned_lines.append(cleaned_line)
    
    final_text = " ".join(cleaned_lines)
    return final_text if final_text else text.strip()


def call_groq_api(prompt: str, groq_key: str, model_name: str = "llama-3.1-8b-instant") -> Optional[str]:
    """Execute Groq API request with multi-key rotation support."""
    if not groq_key or groq_key == "YOUR_GROQ_API_KEY_HERE":
        return None

    keys_list = [k.strip() for k in groq_key.split(",") if k.strip() and k.strip() != "YOUR_GROQ_API_KEY_HERE"]
    if not keys_list:
        return None

    system_prompt = (
        "Bạn là hệ thống trích xuất metadata video chuyên sâu cho công cụ tìm kiếm đa phương tiện.\n"
        "Yêu cầu bắt buộc:\n"
        "1. Chỉ xuất đúng ĐÚNG MỘT ĐOẠN VĂN XUÔI tiếng Việt tự nhiên hoàn chỉnh, tối đa 60 từ.\n"
        "2. Mô tả theo diễn tiến thời gian các sự kiện và hành động (kể cả hành động ngắn thoáng qua như vẫy tay, ngoảnh đầu, cầm đồ vật).\n"
        "3. Nếu đầu vào có OCR, hãy sửa lỗi chính tả tiếng Việt trước khi sử dụng thông tin đó để tạo mô tả.\n"
        "4. Nếu OCR không rõ hoặc không liên quan, hãy bỏ qua.\n"
        "5. BẮT BUỘC giữ nguyên tên người, địa danh, thương hiệu, tên sản phẩm và các tên tiếng Anh gốc.\n"
        "6. Không chào hỏi, không giải thích, không markdown, không gạch đầu dòng, không đánh số, không dùng ký tự mũi tên; chỉ xuất duy nhất đoạn văn mô tả."
    )

    for active_key in keys_list:
        try:
            from groq import Groq
            client = Groq(api_key=active_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            if completion.choices:
                res_text = completion.choices[0].message.content.strip()
                if res_text:
                    cleaned_out = _clean_llm_response(res_text)
                    logger.info(f"Groq SDK call successful ({model_name}).")
                    return cleaned_out
        except Exception:
            pass

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {active_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 150
        }
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    res_text = choices[0]["message"].get("content", "").strip()
                    if res_text:
                        cleaned_out = _clean_llm_response(res_text)
                        logger.info(f"Groq REST API call successful ({model_name}).")
                        return cleaned_out
            elif res.status_code == 429:
                logger.warning(f"Groq API Key rate-limited (429). Rotating to next key...")
                continue
        except Exception as e:
            logger.warning(f"Groq REST API execution error ({model_name}): {e}")

    return None


def call_gemini_api(prompt: str, gemini_key: str) -> Optional[str]:
    """Execute Google Gemini Flash REST API request with automated rate-limit delay."""
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        return None
        
    for model_name in ["gemini-flash-latest", "gemini-2.0-flash", "gemini-pro-latest"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=12)
            if res.status_code == 200:
                res_data = res.json()
                candidates = res_data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        text_out = parts[0]["text"].strip()
                        logger.info(f"Gemini API call successful ({model_name}).")
                        time.sleep(4)
                        return text_out
            elif res.status_code == 429:
                logger.warning(f"Gemini API rate-limited (429). Pausing for 10s...")
                time.sleep(10)
                continue
        except Exception as e:
            logger.warning(f"Gemini API execution error ({model_name}): {e}")
            
    return None


def correct_ocr_text(raw_text: str, config_path: str = "config.yaml") -> str:
    """Correct Vietnamese OCR text using Groq or Gemini Flash API rotation."""
    if not raw_text or not raw_text.strip():
        return ""

    config = load_config(config_path)
    llm_cfg = config.get("models", {}).get("llm", {})
    
    prompt = (
        "Bạn là một AI chuyên sửa lỗi OCR tiếng Việt cho hệ thống tìm kiếm đa phương tiện.\n"
        "Hãy khôi phục đầy đủ dấu tiếng Việt, sửa lỗi chính tả và chuẩn hóa lại đoạn văn bản thô do OCR đọc từ ảnh dưới đây.\n"
        "Yêu cầu:\n"
        "1. Giữ nguyên ý nghĩa gốc.\n"
        "2. CHỈ TRẢ VỀ ĐOẠN VĂN BẢN ĐÃ SỬA, KHÔNG THÊM BẤT KỲ LỜI DẪN HAY GIẢI THÍCH NÀO.\n\n"
        f"Văn bản OCR thô:\n\"{raw_text}\""
    )

    groq_key = os.getenv("GROQ_API_KEY") or llm_cfg.get("groq_api_key", "")
    groq_model = llm_cfg.get("groq_model", "llama-3.1-8b-instant")
    gemini_key = os.getenv("GEMINI_API_KEY") or llm_cfg.get("gemini_api_key", "")

    for attempt in range(3):
        if groq_key and groq_key != "YOUR_GROQ_API_KEY_HERE":
            groq_res = call_groq_api(prompt, groq_key, model_name=groq_model)
            if groq_res:
                return groq_res

        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
            gemini_res = call_gemini_api(prompt, gemini_key)
            if gemini_res:
                return gemini_res

        time.sleep(3)

    return raw_text.strip()


def summarize_window_with_llm(window_data_str: str, config_path: str = "config.yaml") -> str:
    """Generate concise scene context summary for a temporal window batch via LLM providers."""
    if not window_data_str or not window_data_str.strip():
        return ""

    config = load_config(config_path)
    llm_cfg = config.get("models", {}).get("llm", {})

    prompt = (
        f"Dữ liệu thô từ các khung hình video trong khoảng thời gian này:\n"
        f"{window_data_str}\n\n"
        "Hãy tóm tắt ngắn gọn bối cảnh và diễn biến chính thành 1 đoạn văn tiếng Việt duy nhất (< 40 từ)."
    )

    groq_key = os.getenv("GROQ_API_KEY") or llm_cfg.get("groq_api_key", "")
    groq_model = llm_cfg.get("groq_model", "llama-3.1-8b-instant")
    gemini_key = os.getenv("GEMINI_API_KEY") or llm_cfg.get("gemini_api_key", "")

    for attempt in range(3):
        if groq_key and groq_key != "YOUR_GROQ_API_KEY_HERE":
            groq_res = call_groq_api(prompt, groq_key, model_name=groq_model)
            if groq_res:
                return groq_res

        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
            logger.info("Failing over to Gemini Flash API...")
            gemini_res = call_gemini_api(prompt, gemini_key)
            if gemini_res:
                return gemini_res

        logger.warning(f"Provider rotation attempt {attempt + 1}/3 rate-limited. Retrying in 4s...")
        time.sleep(4)

    return window_data_str.strip()


def window_based_summarize(frames_data: List[Dict[str, Any]], window_size: Optional[int] = None, config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    """Group keyframe analytics into temporal window batches and attach LLM context summaries."""
    if not frames_data:
        return []

    config = load_config(config_path)
    if window_size is None:
        window_size = config.get("preprocessing", {}).get("video", {}).get("window_size", 30)
    
    batch_windows = config.get("preprocessing", {}).get("video", {}).get("batch_windows", 3)
    window_ms = window_size * 1000
    
    windows: Dict[int, List[Dict[str, Any]]] = {}
    for frame in frames_data:
        ts_ms = frame.get("timestamp_ms", 0)
        win_idx = int(ts_ms // window_ms)
        if win_idx not in windows:
            windows[win_idx] = []
        windows[win_idx].append(frame)

    sorted_win_indices = sorted(windows.keys())
    total_batches = (len(sorted_win_indices) + batch_windows - 1) // batch_windows
    
    for batch_i in range(0, len(sorted_win_indices), batch_windows):
        batch_indices = sorted_win_indices[batch_i:batch_i + batch_windows]
        batch_num = batch_i // batch_windows + 1
        
        batch_start_sec = batch_indices[0] * window_size
        batch_end_sec = (batch_indices[-1] + 1) * window_size
        batch_frames = []
        all_objects = []
        all_ocr = []
        
        for win_idx in batch_indices:
            win_frames = windows[win_idx]
            batch_frames.extend(win_frames)
            
            for f in win_frames:
                objs = f.get("objects", [])
                if isinstance(objs, list):
                    all_objects.extend(objs)
                elif isinstance(objs, str) and objs:
                    all_objects.append(objs)
                
                ocr_text = f.get("ocr_fixed") or f.get("ocr_raw") or ""
                if ocr_text.strip():
                    all_ocr.append(ocr_text.strip())

        unique_objects = list(set(all_objects))
        objects_vi = [COCO_VI_MAP.get(obj, obj) for obj in unique_objects]
        unique_ocr = list(set(all_ocr))

        window_data_str = (
            f"Khoảng thời gian: {batch_start_sec}s - {batch_end_sec}s.\n"
            f"Vật thể xuất hiện: {', '.join(objects_vi) if objects_vi else 'Không có'}.\n"
            f"Văn bản OCR đọc được: {' | '.join(unique_ocr) if unique_ocr else 'Không có'}."
        )

        logger.info(f"Summarizing LLM context batch {batch_num}/{total_batches} [{batch_start_sec}s - {batch_end_sec}s] ({len(batch_frames)} frames)...")
        context_summary = summarize_window_with_llm(window_data_str, config_path=config_path)

        for f in batch_frames:
            f["context_summary"] = context_summary

    return frames_data


class VisionAnalytics:
    """Object detection via YOLOv9 and Vietnamese OCR text extraction via EasyOCR / PaddleOCR."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        obj_cfg = self.config.get("models", {}).get("object_detection", {})
        ocr_cfg = self.config.get("models", {}).get("ocr", {})

        self.yolo_model_path = obj_cfg.get("yolo_model", "yolov9c.pt")
        self.conf_threshold = obj_cfg.get("confidence_threshold", 0.35)
        self.ocr_lang = ocr_cfg.get("lang", "vi")
        self.use_angle_cls = ocr_cfg.get("use_angle_cls", True)

        self.yolo_model = self._init_yolo()
        self.easyocr_reader = None
        self.paddleocr_engine = None
        self._init_ocr_engines()

    def _init_yolo(self) -> Optional[Any]:
        """Initialize YOLOv9 model object from ultralytics library."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.yolo_model_path)
            logger.info(f"YOLOv9 model loaded successfully from '{self.yolo_model_path}'.")
            return model
        except Exception as e:
            logger.error(f"YOLOv9 initialization failed from '{self.yolo_model_path}': {e}")
            return None

    def _init_ocr_engines(self):
        """Initialize EasyOCR and PaddleOCR engines."""
        try:
            import easyocr
            gpu = torch.cuda.is_available()
            self.easyocr_reader = easyocr.Reader(['vi', 'en'], gpu=gpu)
            logger.info(f"EasyOCR reader initialized (languages=['vi', 'en'], gpu={gpu}).")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")

        try:
            from paddleocr import PaddleOCR
            self.paddleocr_engine = PaddleOCR(use_angle_cls=False, lang=self.ocr_lang, show_log=False)
            logger.info(f"PaddleOCR engine initialized (lang={self.ocr_lang}).")
        except Exception as e:
            logger.info(f"PaddleOCR engine unavailable: {e}")

    def detect_objects(self, frame_path: str) -> List[str]:
        """Perform object detection on a frame image using YOLOv9."""
        if self.yolo_model is None:
            return []

        try:
            results = self.yolo_model.predict(source=frame_path, conf=self.conf_threshold, verbose=False)
            detected_classes = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for cls_id in result.boxes.cls.cpu().numpy():
                        class_name = result.names[int(cls_id)]
                        detected_classes.append(class_name)

            unique_classes = list(set(detected_classes))
            return unique_classes
        except Exception as e:
            logger.error(f"YOLOv9 object detection error on '{frame_path}': {e}")
            return []

    def extract_ocr_raw(self, frame_path: str) -> str:
        """Extract raw OCR text from a frame image using EasyOCR or PaddleOCR."""
        lines = []

        if self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(frame_path, detail=0)
                if results:
                    lines = [str(t).strip() for t in results if str(t).strip()]
                    if lines:
                        return " ".join(lines)
            except Exception as e:
                logger.warning(f"EasyOCR extraction error on '{frame_path}': {e}")

        if self.paddleocr_engine is not None:
            try:
                result = self.paddleocr_engine.ocr(frame_path, cls=False)
                if result and len(result) > 0 and result[0]:
                    for res_item in result[0]:
                        if not res_item:
                            continue
                        if isinstance(res_item, (list, tuple)) and len(res_item) >= 2:
                            text_info = res_item[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                                text_str = str(text_info[0]).strip()
                                if text_str:
                                    lines.append(text_str)
                            elif isinstance(text_info, str) and text_info.strip():
                                lines.append(text_info.strip())
                    if lines:
                        return " ".join(lines)
            except Exception as e:
                logger.warning(f"PaddleOCR extraction error on '{frame_path}': {e}")

        return " ".join(lines)

    def analyze_frame(self, frame_path: str) -> Dict[str, Any]:
        """Extract object detection and OCR text features for a single keyframe."""
        if not os.path.exists(frame_path):
            logger.error(f"Frame image file not found: {frame_path}")
            return {
                "frame_path": frame_path,
                "objects": [],
                "ocr_raw": "",
                "ocr_fixed": ""
            }

        objects = self.detect_objects(frame_path)
        ocr_raw = self.extract_ocr_raw(frame_path)
        ocr_fixed = ocr_raw

        return {
            "frame_path": frame_path,
            "objects": objects,
            "ocr_raw": ocr_raw,
            "ocr_fixed": ocr_fixed
        }


def process_single_frame(frame_path: str) -> Dict[str, Any]:
    analytics = VisionAnalytics()
    return analytics.analyze_frame(frame_path)


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "processed_data/1_frames/test_transnet_shot_0001_sharpest.jpg"
    if os.path.exists(test_path):
        res = process_single_frame(test_path)
        print("Analysis Result:", res)
    else:
        print("Test image file does not exist.")
