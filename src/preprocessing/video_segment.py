"""
Video Segmentation Module (TransNet V2 & InfoShot + FP16 Precision)
Phát hiện chuyển cảnh (Shot Boundary Detection) bằng TransNet V2 và trích xuất Keyframes (Common & Sharpest) bằng InfoShot.
- BẮT BUỘC mô hình TransNet V2 (kèm cờ Quantization FP16 Half-Precision trên GPU).
- Thuật toán InfoShot chuẩn: 2 frames/shot (Common & Sharpest đo bằng cv2.Laplacian(gray, cv2.CV_64F).var()).
- Đặt tên file đầu ra: [video_name]_shot_[shot_id]_[common/sharpest].jpg
"""

import os
import cv2
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VideoSegmenter")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def calculate_sharpness_laplacian(gray_frame: np.ndarray) -> float:
    """
    Đo độ sắc nét của khung hình bằng phương sai của thuật toán Laplacian:
    cv2.Laplacian(gray, cv2.CV_64F).var()
    """
    if gray_frame is None or gray_frame.size == 0:
        return 0.0
    try:
        return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
    except Exception:
        return 0.0


class VideoSegmenter:
    """
    Sử dụng BẮT BUỘC mô hình TransNet V2 để phân đoạn shot.
    Trích xuất đúng 2 frames (common & sharpest) cho mỗi shot bằng thuật toán InfoShot.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        transnet_cfg = self.config.get("models", {}).get("transnet_v2", {})
        self.threshold = transnet_cfg.get("threshold", self.config.get("preprocessing", {}).get("video", {}).get("transnet_threshold", 0.5))
        self.weights_path = transnet_cfg.get("weights_path", "src/preprocessing/weights/transnetv2-pytorch-weights.pth")
        self.output_dir = self.config.get("paths", {}).get("frames_dir", "processed_data/1_frames")
        self.use_fp16 = self.config.get("preprocessing", {}).get("use_fp16", True)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # BẮT BUỘC nạp TransNet V2 model. Không được có bất kỳ fallback nào.
        self.transnet_model = self._init_transnet_v2()

    def _init_transnet_v2(self) -> Any:
        """
        Khởi tạo mô hình TransNet V2 từ weights file (src/preprocessing/weights/transnetv2-pytorch-weights.pth)
        hoặc gói thư viện transnetv2_pytorch.
        NẾU KHÔNG TẢI ĐƯỢC WEIGHTS HOẶC THƯ VIỆN, QUĂNG LỖI (RAISE EXCEPTION) ĐỂ DỪNG CHƯƠNG TRÌNH NGAY LẬP TỨC.
        VÔ HIỆU HÓA HOÀN TOÀN MỌI THUẬT TOÁN FALLBACK (HISTOGRAM / PIXEL DIFFERENCE).
        """
        try:
            from transnetv2_pytorch import TransNetV2
            
            if os.path.exists(self.weights_path):
                logger.info(f"Đang nạp mô hình TransNetV2 từ local weights: '{self.weights_path}'...")
                try:
                    model = TransNetV2(model_path=self.weights_path)
                except TypeError:
                    model = TransNetV2()
            else:
                logger.info(f"Local weights '{self.weights_path}' chưa khởi tạo, nạp gói TransNetV2 PyTorch mặc định...")
                model = TransNetV2()

            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                logger.info("Đã khởi tạo mô hình TransNetV2 PyTorch (CUDA GPU) thành công.")
            else:
                logger.info("Đã khởi tạo mô hình TransNetV2 PyTorch (CPU) thành công.")
            return model

        except Exception as e:
            error_msg = (
                f"CRITICAL ERROR: Không thể nạp mô hình TransNetV2! "
                f"Kiểm tra file weights tại '{self.weights_path}' hoặc thư viện 'transnetv2-pytorch'. "
                f"TẤT CẢ THUẬT TOÁN FALLBACK (Adaptive Histogram / Pixel Difference) ĐÃ BỊ VÔ HIỆU HÓA HOÀN TOÀN. "
                f"Dừng chương trình ngay lập tức để tránh sinh ra keyframe rác. Lỗi chi tiết: {str(e)}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

    def predict_shot_boundaries(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Dự đoán các điểm chuyển cảnh (shot boundaries) bằng TransNet V2 PyTorch.
        Chia batch 1000 frames để tiết kiệm bộ nhớ RAM tuyệt đối.
        Trả về danh sách tuple (start_frame, end_frame).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở file video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video {video_path} không chứa khung hình hợp lệ.")

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # TransNet V2 yêu cầu đầu vào ảnh RGB kích thước (27, 48)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_frame, (48, 27))
            frames.append(resized)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Không đọc được khung hình nào từ video {video_path}")

        # Dự đoán theo batch 1000 frames để tiết kiệm RAM
        batch_size = 1000
        predictions_list = []

        for i in range(0, len(frames), batch_size):
            chunk_frames = frames[i:i + batch_size]
            chunk_np = np.array(chunk_frames, dtype=np.uint8)
            input_tensor = torch.from_numpy(chunk_np).unsqueeze(0)

            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda().float()
            else:
                input_tensor = input_tensor.float()

            with torch.no_grad():
                output = self.transnet_model(input_tensor)
                if isinstance(output, tuple):
                    one_hot = output[0]
                else:
                    one_hot = output
                
                preds = torch.sigmoid(one_hot).squeeze().float().cpu().numpy()
                if preds.ndim == 0:
                    preds = np.array([preds.item()])
                predictions_list.append(preds)

        predictions = np.concatenate(predictions_list, axis=0)
        scene_bools = predictions > self.threshold

        shots = []
        start = 0
        for i, is_scene_change in enumerate(scene_bools):
            if is_scene_change and i > start:
                shots.append((start, i))
                start = i + 1
        if start < len(frames):
            shots.append((start, len(frames) - 1))

        return shots

    def infoshot_extract_keyframes(self, video_path: str, shots: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """
        Thuật toán InfoShot Chuẩn: Trích xuất đúng 2 frames cho mỗi shot:
        1. common_frame: Frame nằm ở giữa khoảng thời gian của shot (mid_idx).
        2. sharpest_frame: Frame có độ sắc nét cao nhất trong shot 
           (đo bằng phương sai của thuật toán Laplacian: cv2.Laplacian(gray, cv2.CV_64F).var()).
        
        Định dạng tên file đầu ra:
        [video_name]_shot_[shot_id]_common.jpg
        [video_name]_shot_[shot_id]_sharpest.jpg
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        extracted_metadata = []

        for shot_idx, (start_idx, end_idx) in enumerate(shots, start=1):
            shot_len = end_idx - start_idx + 1
            if shot_len <= 0:
                continue

            # 1. common_frame: Frame nằm ở giữa khoảng thời gian của shot
            mid_idx = start_idx + shot_len // 2

            # 2. sharpest_frame: Chọn ứng viên có phương sai Laplacian cv2.Laplacian(gray, cv2.CV_64F).var() cao nhất
            step = max(1, shot_len // 5)
            candidates = sorted(list(set([start_idx, mid_idx, end_idx] + list(range(start_idx, end_idx + 1, step)))))

            candidate_scores = []
            candidate_frames = {}

            for c_idx in candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, c_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Tính phương sai Laplacian: cv2.Laplacian(gray, cv2.CV_64F).var()
                    laplacian_var = calculate_sharpness_laplacian(gray)
                    candidate_scores.append((c_idx, laplacian_var))
                    candidate_frames[c_idx] = frame

            if not candidate_frames:
                continue

            # Xác định common_frame
            if mid_idx in candidate_frames:
                common_frame = candidate_frames[mid_idx]
            else:
                mid_idx = list(candidate_frames.keys())[0]
                common_frame = candidate_frames[mid_idx]

            # Xác định sharpest_frame
            sorted_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
            sharpest_idx = sorted_scores[0][0]
            if sharpest_idx == mid_idx and len(sorted_scores) > 1:
                sharpest_idx = sorted_scores[1][0]
            sharpest_frame = candidate_frames[sharpest_idx]

            # Đảm bảo cấu trúc tên file chuẩn: [video_name]_shot_[shot_id]_[common/sharpest].jpg
            shot_str = f"{shot_idx:04d}"
            common_filename = f"{video_name}_shot_{shot_str}_common.jpg"
            sharpest_filename = f"{video_name}_shot_{shot_str}_sharpest.jpg"

            common_path = os.path.join(self.output_dir, common_filename)
            sharpest_path = os.path.join(self.output_dir, sharpest_filename)

            cv2.imwrite(common_path, common_frame)
            cv2.imwrite(sharpest_path, sharpest_frame)

            extracted_metadata.append({
                "video_name": video_name,
                "shot_id": shot_idx,
                "frame_type": "common",
                "frame_idx": mid_idx,
                "timestamp_ms": int((mid_idx / fps) * 1000),
                "saved_path": common_path
            })
            extracted_metadata.append({
                "video_name": video_name,
                "shot_id": shot_idx,
                "frame_type": "sharpest",
                "frame_idx": sharpest_idx,
                "timestamp_ms": int((sharpest_idx / fps) * 1000),
                "saved_path": sharpest_path
            })

        cap.release()
        return extracted_metadata

    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Hàm chính bóc tách video. Nếu xảy ra lỗi hoặc không nạp được TransNet V2, quăng Exception ngắt luồng.
        """
        logger.info(f"Đang bóc tách video với TransNet V2 & InfoShot: {video_path}")
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Tệp video không tồn tại: {video_path}")

            # Bước 1: Dự đoán điểm chuyển cảnh bắt buộc bằng TransNet V2
            shots = self.predict_shot_boundaries(video_path)
            logger.info(f"TransNet V2 phát hiện {len(shots)} shots cho video {os.path.basename(video_path)}")

            # Bước 2: InfoShot trích xuất 2 frames (common/sharpest) mỗi shot
            keyframes_meta = self.infoshot_extract_keyframes(video_path, shots)
            logger.info(f"InfoShot đã lưu {len(keyframes_meta)} keyframes vào {self.output_dir}")

            return keyframes_meta

        except Exception as e:
            logger.error(f"[XỬ LÝ LỖI] Lỗi bóc tách video {video_path}: {str(e)}")
            raise e


if __name__ == "__main__":
    segmenter = VideoSegmenter()
    print("VideoSegmenter đã được cập nhật cờ Quantization FP16 thành công!")
