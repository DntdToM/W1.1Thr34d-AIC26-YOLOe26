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
        
        all_audio = []
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
                    
                asr_text = f.get("asr_text", "")
                audio_event = f.get("audio_event", "")
                if asr_text:
                    all_audio.append(asr_text)
                if audio_event:
                    all_audio.append(audio_event)

        unique_objects = list(set(all_objects))
        unique_ocr = list(set(all_ocr))
        unique_audio = list(set(all_audio))

        window_data_str = (
            f"Khoảng thời gian: {batch_start_sec}s - {batch_end_sec}s.\n"
            f"Vật thể xuất hiện: {', '.join(unique_objects) if unique_objects else 'Không có'}.\n"
            f"Văn bản OCR đọc được: {' | '.join(unique_ocr) if unique_ocr else 'Không có'}.\n"
            f"Âm thanh/Lời nói: {' | '.join(unique_audio) if unique_audio else 'Không có'}."
        )

        logger.info(f"Summarizing LLM context batch {batch_num}/{total_batches} [{batch_start_sec}s - {batch_end_sec}s] ({len(batch_frames)} frames)...")
        context_summary = summarize_window_with_llm(window_data_str, config_path=config_path)

        for f in batch_frames:
            f["context_summary"] = context_summary

    return frames_data


def boxes_intersect(ocr_poly, yolo_box):
    """Check if OCR polygon intersects with YOLO bounding box."""
    ocr_xs = [pt[0] for pt in ocr_poly]
    ocr_ys = [pt[1] for pt in ocr_poly]
    ocr_xmin, ocr_xmax = min(ocr_xs), max(ocr_xs)
    ocr_ymin, ocr_ymax = min(ocr_ys), max(ocr_ys)
    
    yolo_xmin, yolo_ymin, yolo_xmax, yolo_ymax = yolo_box
    
    if ocr_xmax < yolo_xmin or ocr_xmin > yolo_xmax:
        return False
    if ocr_ymax < yolo_ymin or ocr_ymin > yolo_ymax:
        return False
    return True

class VisionAnalytics:
    """Object detection via YOLOE-26 and Vietnamese OCR text extraction via PaddleOCR."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        obj_cfg = self.config.get("models", {}).get("object_detection", {})
        ocr_cfg = self.config.get("models", {}).get("ocr", {})

        self.yolo_model_path = obj_cfg.get("yolo_model", "yoloe-26l-seg-pf.pt")
        self.conf_threshold = obj_cfg.get("confidence_threshold", 0.05)
        self.agnostic_nms = obj_cfg.get("agnostic_nms", True)
        self.ocr_lang = ocr_cfg.get("lang", "vi")
        self.use_angle_cls = ocr_cfg.get("use_angle_cls", True)

        self.yolo_model = self._init_yolo()
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
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR
            self.paddleocr_engine = PaddleOCR(use_angle_cls=False, lang=self.ocr_lang)
            logger.info(f"PaddleOCR engine initialized (lang={self.ocr_lang}).")
        except Exception as e:
            logger.info(f"PaddleOCR engine unavailable: {e}")

    def detect_objects_with_boxes(self, frame_path: str) -> List[Dict[str, Any]]:
        """Perform object detection returning bounding boxes."""
        if self.yolo_model is None:
            return []
        try:
            results = self.yolo_model.predict(source=frame_path, conf=self.conf_threshold, verbose=False, agnostic_nms=self.agnostic_nms)
            detected = []
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls.cpu().numpy()[0])
                        class_name = result.names[cls_id]
                        xyxy = box.xyxy.cpu().numpy()[0].tolist()
                        detected.append({"class": class_name, "box": xyxy})
            return detected
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

    def extract_ocr_with_boxes(self, frame_path: str) -> List[Dict[str, Any]]:
        """Extract OCR text with bounding boxes using PaddleOCR."""
        extracted = []
        if self.paddleocr_engine is not None:
            try:
                result = self.paddleocr_engine.ocr(frame_path, cls=False)
                if result and len(result) > 0 and result[0]:
                    for res_item in result[0]:
                        if not res_item:
                            continue
                        if isinstance(res_item, (list, tuple)) and len(res_item) >= 2:
                            box = res_item[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                            text_info = res_item[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                                text_str = str(text_info[0]).strip()
                                if text_str:
                                    extracted.append({"text": text_str, "box": box})
                            elif isinstance(text_info, str) and text_info.strip():
                                extracted.append({"text": text_info.strip(), "box": box})
            except Exception as e:
                logger.warning(f"PaddleOCR extraction error: {e}")
        return extracted

    def analyze_frame(self, frame_path: str) -> Dict[str, Any]:
        """Extract features with spatial intersection logic."""
        if not os.path.exists(frame_path):
            logger.error(f"Frame image file not found: {frame_path}")
            return {
                "frame_path": frame_path,
                "objects": [],
                "ocr_raw": "",
                "ocr_fixed": ""
            }

        yolo_results = self.detect_objects_with_boxes(frame_path)
        ocr_results = self.extract_ocr_with_boxes(frame_path)
        
        valid_ocr_texts = []
        for ocr_item in ocr_results:
            ocr_poly = ocr_item["box"]
            ocr_text = ocr_item["text"]
            
            intersects = False
            for y_item in yolo_results:
                if boxes_intersect(ocr_poly, y_item["box"]):
                    intersects = True
                    break
                    
            if intersects or len(yolo_results) == 0:
                valid_ocr_texts.append(ocr_text)

        unique_classes = list(set([item["class"] for item in yolo_results]))
        ocr_combined = " ".join(valid_ocr_texts)

        return {
            "frame_path": frame_path,
            "objects": unique_classes,
            "ocr_raw": ocr_combined,
            "ocr_fixed": ocr_combined
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
