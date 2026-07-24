"""Video segmentation module utilizing TransNetV2 and InfoShot keyframe extraction."""

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
    """Calculate frame sharpness using the variance of Laplacian operator."""
    if gray_frame is None or gray_frame.size == 0:
        return 0.0
    try:
        return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
    except Exception:
        return 0.0


class VideoSegmenter:
    """Shot boundary detection via TransNetV2 and keyframe extraction via InfoShot algorithm."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        transnet_cfg = self.config.get("models", {}).get("transnet_v2", {})
        self.threshold = transnet_cfg.get("threshold", self.config.get("preprocessing", {}).get("video", {}).get("transnet_threshold", 0.5))
        self.weights_path = transnet_cfg.get("weights_path", "src/preprocessing/weights/transnetv2-pytorch-weights.pth")
        self.output_dir = self.config.get("paths", {}).get("frames_dir", "processed_data/1_frames")
        self.use_fp16 = self.config.get("preprocessing", {}).get("use_fp16", True)
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.transnet_model = self._init_transnet_v2()

    def _init_transnet_v2(self) -> Any:
        """Initialize TransNetV2 model architecture from local weights or transnetv2_pytorch package."""
        try:
            from transnetv2_pytorch import TransNetV2
            
            if os.path.exists(self.weights_path):
                logger.info(f"Loading TransNetV2 model from local weights: '{self.weights_path}'")
                try:
                    model = TransNetV2(model_path=self.weights_path)
                except TypeError:
                    model = TransNetV2()
            else:
                logger.info(f"Local weights file '{self.weights_path}' not found. Initializing default TransNetV2 PyTorch model...")
                model = TransNetV2()

            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                logger.info("TransNetV2 model initialized successfully on CUDA GPU.")
            else:
                logger.info("TransNetV2 model initialized successfully on CPU.")
            return model

        except Exception as e:
            error_msg = f"TransNetV2 model initialization failed from '{self.weights_path}': {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def predict_shot_boundaries(self, video_path: str) -> List[Tuple[int, int]]:
        """Predict shot boundary frame indices using TransNetV2 PyTorch in mini-batches."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video file contains invalid frame count: {video_path}")

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_frame, (48, 27))
            frames.append(resized)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Failed to read any valid frames from video: {video_path}")

        if len(frames) < 10:
            logger.info(f"Short video detected ({len(frames)} frames). Defaulting to single shot segment.")
            return [(0, len(frames) - 1)]

        batch_size = 1000
        predictions_list = []

        for i in range(0, len(frames), batch_size):
            chunk_frames = frames[i:i + batch_size]
            chunk_np = np.array(chunk_frames, dtype=np.uint8)
            input_tensor = torch.from_numpy(chunk_np).unsqueeze(0)

            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

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
        """Extract common (mid-frame) and sharpest (max Laplacian variance) keyframes per shot."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        extracted_metadata = []

        for shot_idx, (start_idx, end_idx) in enumerate(shots, start=1):
            shot_len = end_idx - start_idx + 1
            if shot_len <= 0:
                continue

            mid_idx = start_idx + shot_len // 2

            step = max(1, shot_len // 5)
            candidates = sorted(list(set([start_idx, mid_idx, end_idx] + list(range(start_idx, end_idx + 1, step)))))

            candidate_scores = []
            candidate_frames = {}

            for c_idx in candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, c_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = calculate_sharpness_laplacian(gray)
                    candidate_scores.append((c_idx, laplacian_var))
                    candidate_frames[c_idx] = frame

            if not candidate_frames:
                continue

            if mid_idx in candidate_frames:
                common_frame = candidate_frames[mid_idx]
            else:
                mid_idx = list(candidate_frames.keys())[0]
                common_frame = candidate_frames[mid_idx]

            sorted_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
            sharpest_idx = sorted_scores[0][0]
            if sharpest_idx == mid_idx and len(sorted_scores) > 1:
                sharpest_idx = sorted_scores[1][0]
            sharpest_frame = candidate_frames[sharpest_idx]

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
        """Segment video into shots and extract keyframes using TransNetV2 and InfoShot."""
        logger.info(f"Segmenting video: {video_path}")
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file path not found: {video_path}")

            shots = self.predict_shot_boundaries(video_path)
            logger.info(f"TransNetV2 detected {len(shots)} shots in video '{os.path.basename(video_path)}'.")

            keyframes_meta = self.infoshot_extract_keyframes(video_path, shots)
            logger.info(f"Extracted {len(keyframes_meta)} keyframe artifacts to '{self.output_dir}'.")

            return keyframes_meta

        except Exception as e:
            logger.error(f"Segmentation failed for video '{video_path}': {str(e)}")
            raise e


if __name__ == "__main__":
    segmenter = VideoSegmenter()
    print("VideoSegmenter initialized successfully.")
