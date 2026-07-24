"""
Vision Analytics Module (YOLOv9 & EasyOCR / PaddleOCR + Window-Based LLM Context Summarization + Gemini API)
- Trích xuất đặc trưng độc lập: YOLOv9 (Object Detection) & EasyOCR / PaddleOCR (Text Reading).
- Tổng hợp ngữ cảnh theo cửa sổ (Window-Based): Gọi Gemini Flash API ĐÚNG 1 LẦN cho mỗi batch cửa sổ
  để sửa lỗi OCR, lọc nhiễu vật thể và viết metadata tổng hợp (< 50 từ) mô tả bối cảnh chung.
- Rate-limit Gemini API: 4s delay giữa các lần gọi để tránh bị chặn bởi Free Tier (15 req/min).
- Tự động phát hiện & vô hiệu hóa Ollama fallback khi chạy trên Kaggle/Cloud (tránh spam Connection refused).
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import yaml
import requests
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VisionAnalytics")

# === Auto-detect Ollama availability (chỉ kiểm tra 1 lần duy nhất khi import module) ===
_OLLAMA_AVAILABLE: Optional[bool] = None

def _check_ollama_available(ollama_url: str = "http://localhost:11434") -> bool:
    """Kiểm tra Ollama có đang chạy không. Chỉ gọi 1 lần duy nhất, cache kết quả."""
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is not None:
        return _OLLAMA_AVAILABLE
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=2)
        _OLLAMA_AVAILABLE = res.status_code == 200
    except Exception:
        _OLLAMA_AVAILABLE = False
    if not _OLLAMA_AVAILABLE:
        logger.info("Ollama Local LLM không khả dụng (Kaggle/Cloud). Đã vô hiệu hóa fallback Ollama.")
    return _OLLAMA_AVAILABLE


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
    """Loại bỏ các câu chào hỏi/lời dẫn thừa và định dạng markdown rác từ phản hồi LLM."""
    if not text:
        return ""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        # Bỏ qua các dòng lời dẫn chatty không chứa ngữ cảnh video
        if any(lower_line.startswith(prefix) for prefix in [
            "sau khi", "dưới đây", "kết quả", "tôi đã", "đây là", "bảng tóm tắt",
            "1. sửa lỗi", "2. lọc", "dữ liệu đã", "văn bản đọc"
        ]):
            continue
        # Bỏ ký tự markdown bullet điểm
        cleaned_line = line.lstrip("-*•1234567890. ").strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    final_text = " ".join(cleaned_lines)
    return final_text if final_text else text.strip()


def call_groq_api(prompt: str, groq_key: str, model_name: str = "llama-3.1-8b-instant") -> Optional[str]:
    """
    Gọi Groq API hỗ trợ xoay vòng nhiều API Keys (phân cách bởi dấu phẩy).
    Sử dụng mô hình Llama-3.1-8B-Instant (14.4k RPD / 500k TPD) siêu tốc > 800 tokens/sec.
    """
    if not groq_key or groq_key == "YOUR_GROQ_API_KEY_HERE":
        return None

    # Tách danh sách keys nếu truyền nhiều keys (ví dụ: "gsk_1,gsk_2,gsk_3")
    keys_list = [k.strip() for k in groq_key.split(",") if k.strip() and k.strip() != "YOUR_GROQ_API_KEY_HERE"]
    if not keys_list:
        return None

    system_prompt = (
        "Bạn là bộ trích xuất Metadata Video chuyên nghiệp cho Hệ thống Tìm kiếm Đa phương tiện.\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. CHỈ XUẤT ĐÚNG 1 ĐOẠN VĂN TIẾNG VIỆT THUẦN (TỐI ĐA 40 TỪ) MÔ TẢ NỘI DUNG VÀ BỐI CẢNH NỔI BẬT.\n"
        "2. TUYỆT ĐỐI KHÔNG CHÀO HỎI, KHÔNG LỜI DẪN ('Dưới đây là...', 'Sau khi xử lý...', 'Kết quả...').\n"
        "3. TUYỆT ĐỐI KHÔNG SỬ DỤNG GẠCH ĐẦU DÒNG, DANH SÁCH, ĐÁNH SỐ THỨ TỰ HAY BẤT KỲ ĐỊNH DẠNG MARKDOWN NÀO.\n"
        "4. SỬA LỖI CHÍNH TẢ TIẾNG VIỆT CHO VĂN BẢN OCR (NẾU CÓ), GIỮ NGUYÊN TÊN RIÊNG VÀ KHÔNG DỊCH TÊN TIẾNG ANH GỐC."
    )

    for active_key in keys_list:
        # 1. Thử dùng thư viện groq chính thức nếu có
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
                    logger.info(f"Đã phản hồi thành công từ Groq SDK ({model_name}).")
                    return cleaned_out
        except Exception:
            pass

        # 2. Fallback sang REST API thuần (không cần pip install groq)
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
                        logger.info(f"Đã phản hồi thành công từ Groq REST API ({model_name}).")
                        return cleaned_out
            elif res.status_code == 429:
                logger.warning(f"Groq Key ({active_key[:8]}...) dính rate-limit (429). Chuyển sang Key kế tiếp...")
                continue
        except Exception as e:
            logger.warning(f"Lỗi khi gọi Groq REST API ({model_name}): {e}")

    return None


def call_gemini_api(prompt: str, gemini_key: str) -> Optional[str]:
    """
    Gọi REST API của Google Gemini Flash (`gemini-flash-latest`).
    Tự động chèn delay 4s sau mỗi lần gọi thành công để tuân thủ rate-limit Free Tier (15 req/min).
    """
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
                        logger.info(f"Đã phản hồi thành công từ Gemini API ({model_name}).")
                        # Rate-limit: chờ 4s giữa các request để không bị Gemini Free Tier chặn
                        time.sleep(4)
                        return text_out
            elif res.status_code == 429:
                logger.warning(f"Gemini API rate-limited (429). Chờ 10s rồi thử lại...")
                time.sleep(10)
                continue
        except Exception as e:
            logger.warning(f"Lỗi khi gọi Gemini API ({model_name}): {e}")
            
    return None


def correct_ocr_text(raw_text: str, config_path: str = "config.yaml") -> str:
    """
    Sửa lỗi chính tả & khôi phục dấu tiếng Việt cho chuỗi OCR thô bằng Groq Multi-Keys / Gemini Flash API xoay vòng.
    """
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

    # Vòng lặp xoay vòng liên tục tối đa 3 lượt (Groq Multi-Keys -> Gemini -> Retry)
    for attempt in range(3):
        # 1. Thử xoay vòng qua danh sách Groq API Keys (Llama 3.1 8B Instant)
        if groq_key and groq_key != "YOUR_GROQ_API_KEY_HERE":
            groq_res = call_groq_api(prompt, groq_key, model_name=groq_model)
            if groq_res:
                return groq_res

        # 2. Nếu tất cả Groq Keys bị 429 -> Chuyển qua Gemini Flash API
        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
            gemini_res = call_gemini_api(prompt, gemini_key)
            if gemini_res:
                return gemini_res

        time.sleep(3)

    return raw_text.strip()


def summarize_window_with_llm(window_data_str: str, config_path: str = "config.yaml") -> str:
    """
    Vòng lặp xoay vòng liên tục giữa Groq Multi-Keys (Llama 3.1 8B Instant) và Gemini Flash API.
    ĐÚNG 1 LẦN cho mỗi batch 90s để tối ưu 98% số lần gọi API.
    """
    if not window_data_str or not window_data_str.strip():
        return ""

    config = load_config(config_path)
    llm_cfg = config.get("models", {}).get("llm", {})

    prompt = (
        f"Dưới đây là dữ liệu thô trích xuất từ các frames trong 30-90 giây của một video:\n"
        f"{window_data_str}\n\n"
        "Hãy sửa lỗi chính tả OCR tiếng Việt, loại bỏ các vật thể nhiễu/lặp lại, và viết một đoạn metadata tổng hợp ngắn gọn (< 50 từ) mô tả bối cảnh chung."
    )

    groq_key = os.getenv("GROQ_API_KEY") or llm_cfg.get("groq_api_key", "")
    groq_model = llm_cfg.get("groq_model", "llama-3.1-8b-instant")
    gemini_key = os.getenv("GEMINI_API_KEY") or llm_cfg.get("gemini_api_key", "")

    # Vòng lặp xoay vòng liên tục tối đa 3 lượt (Groq Multi-Keys -> Gemini -> Retry)
    for attempt in range(3):
        # 1. Thử xoay vòng qua danh sách Groq API Keys (Llama 3.1 8B Instant)
        if groq_key and groq_key != "YOUR_GROQ_API_KEY_HERE":
            groq_res = call_groq_api(prompt, groq_key, model_name=groq_model)
            if groq_res:
                return groq_res

        # 2. Nếu tất cả Groq Keys bị Rate-Limit (429) -> Nhảy sang Gemini Flash API
        if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
            logger.info("Chuyển sang thử Gemini Flash API...")
            gemini_res = call_gemini_api(prompt, gemini_key)
            if gemini_res:
                return gemini_res

        # 3. Nếu cả 4 Groq Keys và Gemini đều dính 429, chờ 4s rồi lặp lại xoay vòng lượt tiếp theo
        logger.warning(f"[Lượt xoay vòng {attempt + 1}/3] Cả Groq Multi-Keys và Gemini đều bị Rate-Limit. Chờ 4s rồi xoay vòng lại...")
        time.sleep(4)

    return window_data_str.strip()


def window_based_summarize(frames_data: List[Dict[str, Any]], window_size: Optional[int] = None, config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    """
    HÀM HẬU XỬ LÝ LLM THEO CỬA SỔ (WINDOW-BASED) + BATCHING:
    1. Gom nhóm (batching) dữ liệu thô của tất cả frames trong cùng cửa sổ 30 giây.
    2. GỘP 3 cửa sổ liên tiếp thành 1 batch (~90s) → gọi Gemini ĐÚNG 1 LẦN cho batch đó.
       Giảm 3x số lần gọi API, tuân thủ Gemini Free Tier rate-limit (15 req/min).
    3. Gắn chuỗi metadata tổng hợp vào trường `context_summary` cho tất cả frames thuộc batch.
    """
    if not frames_data:
        return []

    config = load_config(config_path)
    if window_size is None:
        window_size = config.get("preprocessing", {}).get("video", {}).get("window_size", 30)
    
    # Số cửa sổ gộp thành 1 batch LLM call (mặc định 3 → 90s/call)
    batch_windows = config.get("preprocessing", {}).get("video", {}).get("batch_windows", 3)

    window_ms = window_size * 1000
    
    # 1. Nhóm các frames theo window index (30s / cửa sổ)
    windows: Dict[int, List[Dict[str, Any]]] = {}
    for frame in frames_data:
        ts_ms = frame.get("timestamp_ms", 0)
        win_idx = int(ts_ms // window_ms)
        if win_idx not in windows:
            windows[win_idx] = []
        windows[win_idx].append(frame)

    # 2. Gộp batch_windows cửa sổ liên tiếp → 1 LLM call
    sorted_win_indices = sorted(windows.keys())
    total_batches = (len(sorted_win_indices) + batch_windows - 1) // batch_windows
    
    for batch_i in range(0, len(sorted_win_indices), batch_windows):
        batch_indices = sorted_win_indices[batch_i:batch_i + batch_windows]
        batch_num = batch_i // batch_windows + 1
        
        # Gom dữ liệu từ tất cả cửa sổ trong batch
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

        logger.info(f"Đang tổng hợp LLM Context batch {batch_num}/{total_batches} [{batch_start_sec}s - {batch_end_sec}s] ({len(batch_frames)} frames)...")
        context_summary = summarize_window_with_llm(window_data_str, config_path=config_path)

        for f in batch_frames:
            f["context_summary"] = context_summary

    return frames_data


class VisionAnalytics:
    """
    Module xử lý thị giác độc lập: YOLOv9-small nhận diện vật thể và EasyOCR / PaddleOCR đọc chữ tiếng Việt.
    """

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
        """Khởi tạo mô hình YOLOv9-small từ ultralytics."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.yolo_model_path)
            logger.info(f"Đã nạp mô hình YOLOv9 từ {self.yolo_model_path} thành công.")
            return model
        except Exception as e:
            logger.error(f"Không thể khởi tạo YOLOv9 ({self.yolo_model_path}): {e}")
            return None

    def _init_ocr_engines(self):
        """Khởi tạo EasyOCR (PyTorch Native, cực kỳ ổn định) và PaddleOCR."""
        try:
            import easyocr
            gpu = torch.cuda.is_available()
            self.easyocr_reader = easyocr.Reader(['vi', 'en'], gpu=gpu)
            logger.info(f"Đã nạp EasyOCR (languages=['vi', 'en'], gpu={gpu}) thành công.")
        except Exception as e:
            logger.warning(f"Không thể nạp EasyOCR: {e}")

        try:
            from paddleocr import PaddleOCR
            self.paddleocr_engine = PaddleOCR(use_angle_cls=False, lang=self.ocr_lang, show_log=False)
            logger.info(f"Đã nạp PaddleOCR (lang={self.ocr_lang}) thành công.")
        except Exception as e:
            logger.info(f"PaddleOCR chưa có sẵn: {e}")

    def detect_objects(self, frame_path: str) -> List[str]:
        """
        Sử dụng YOLOv9-small nhận diện đối tượng độc lập cho 1 khung hình.
        (Đã bỏ tham số 'half' để không bắn warning).
        """
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
            logger.error(f"Lỗi khi chạy YOLOv9 nhận diện đối tượng trên {frame_path}: {e}")
            return []

    def extract_ocr_raw(self, frame_path: str) -> str:
        """
        Đọc văn bản thô độc lập cho 1 khung hình.
        Sử dụng EasyOCR detail=0 (PyTorch Native) 100% không bao giờ bị lỗi unpack tuple.
        """
        lines = []

        # 1. Thử EasyOCR với detail=0 (Trả về danh sách chuỗi chữ trực tiếp, an toàn tuyệt đối)
        if self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(frame_path, detail=0)
                if results:
                    lines = [str(t).strip() for t in results if str(t).strip()]
                    if lines:
                        return " ".join(lines)
            except Exception as e:
                logger.warning(f"Lỗi EasyOCR trên {frame_path}: {e}")

        # 2. Fallback PaddleOCR nếu EasyOCR chưa đọc được
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
                logger.warning(f"Lỗi PaddleOCR trên {frame_path}: {e}")

        return " ".join(lines)

    def analyze_frame(self, frame_path: str) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng độc lập cho 1 frame duy nhất (objects & ocr_raw).
        """
        if not os.path.exists(frame_path):
            logger.error(f"File khung hình không tồn tại: {frame_path}")
            return {
                "frame_path": frame_path,
                "objects": [],
                "ocr_raw": "",
                "ocr_fixed": ""
            }

        objects = self.detect_objects(frame_path)
        ocr_raw = self.extract_ocr_raw(frame_path)
        # Không gọi LLM riêng lẻ cho từng frame để tránh spam API.
        # Việc sửa chính tả & tóm tắt được gộp 90s/1 LLM call ở window_based_summarize()
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
    print(f"Chạy thử nghiệm VisionAnalytics với frame: {test_path}")
    if os.path.exists(test_path):
        res = process_single_frame(test_path)
        print("Kết quả:", res)
    else:
        print("Không tìm thấy file ảnh test.")
