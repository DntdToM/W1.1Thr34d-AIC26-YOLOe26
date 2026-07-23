"""
Audio Processing & Speech-to-Text Module (Silero VAD & PhoWhisper + FP16 Precision)
Tách kênh audio mono 16kHz bằng ffmpeg / imageio-ffmpeg, lọc khoảng lặng bằng Silero VAD / scipy 
và chuyển đổi voice-to-text bằng PhoWhisper Small (kèm FP16 Quantization) via transformers.
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional
import yaml
import torch
import numpy as np
import subprocess
import scipy.io.wavfile as wavfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AudioASRProcessor")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def format_timestamp(ms: int) -> str:
    """Format milliseconds sang định dạng hh:mm:ss.mmm"""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def read_wav_file_fallback(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Đọc file WAV bằng scipy.io.wavfile hoàn toàn thuần Python mà không cần torchaudio."""
    try:
        sr, data = wavfile.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
        else:
            audio = data.astype(np.float32)
        return audio
    except Exception as e:
        logger.error(f"Không thể đọc file WAV {wav_path}: {e}")
        return np.array([], dtype=np.float32)


class AudioASRProcessor:
    """
    Xử lý âm thanh từ video: 
    1. Tách audio mono 16kHz bằng ffmpeg / imageio-ffmpeg.
    2. Lọc đoạn chứa giọng nói bằng Silero VAD / Chunking.
    3. Nhận diện giọng nói bằng PhoWhisper Small (chế độ FP16 trên GPU).
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        audio_cfg = self.config.get("preprocessing", {}).get("audio", {})
        
        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.vad_threshold = audio_cfg.get("vad_threshold", 0.5)
        self.phowhisper_model_name = audio_cfg.get("phowhisper_model", "vinai/phowhisper-small")
        self.use_fp16 = self.config.get("preprocessing", {}).get("use_fp16", True)

        self.vad_model = None
        self.vad_utils = None
        self.asr_pipeline = None
        self.ffmpeg_exe = self._find_ffmpeg_executable()

    def _find_ffmpeg_executable(self) -> str:
        """Tìm file thực thi ffmpeg (Ưu tiên imageio_ffmpeg bundled binary, fallback 'ffmpeg')."""
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if os.path.exists(exe):
                logger.info(f"Sử dụng bundled ffmpeg binary: {exe}")
                return exe
        except Exception:
            pass
        return "ffmpeg"

    def _init_silero_vad(self) -> bool:
        """Khởi tạo mô hình Silero VAD từ torch.hub."""
        if self.vad_model is None:
            try:
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.vad_model = model
                self.vad_utils = utils
                logger.info("Đã nạp mô hình Silero VAD thành công.")
                return True
            except Exception as e:
                logger.warning(f"Không thể nạp mô hình Silero VAD ({e}). Sẽ tự động dùng Chunking VAD Fallback.")
                return False
        return True

    def _init_phowhisper(self):
        """Khởi tạo mô hình PhoWhisper Small với FP16 precision."""
        if self.asr_pipeline is None:
            try:
                try:
                    from transformers import pipeline
                except ImportError:
                    from transformers.pipelines import pipeline
                device = 0 if torch.cuda.is_available() else -1
                torch_dtype = torch.float16 if (torch.cuda.is_available() and self.use_fp16) else torch.float32

                self.asr_pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=self.phowhisper_model_name,
                    torch_dtype=torch_dtype,
                    device=device
                )
                logger.info(f"Đã nạp mô hình PhoWhisper ({self.phowhisper_model_name}, dtype={torch_dtype}) thành công.")
            except Exception as e:
                logger.error(f"Không thể nạp mô hình PhoWhisper ({self.phowhisper_model_name}): {e}")
                raise e

    def extract_audio_ffmpeg(self, video_path: str, output_wav_path: str) -> bool:
        """
        Tách kênh audio từ video gốc sang mono 16kHz WAV bằng ffmpeg / imageio-ffmpeg.
        """
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-f", "wav",
            output_wav_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 1000
        except Exception as e:
            logger.warning(f"Không thể tách audio từ video {video_path} (Video có thể không chứa kênh âm thanh): {e}")
            return False

    def get_speech_timestamps_vad(self, wav_path: str, audio_np: np.ndarray) -> List[Dict[str, int]]:
        """
        Sử dụng Silero VAD tìm các phân đoạn có giọng nói (start_ms, end_ms).
        Nút thắt tự động Fallback sang Chunking 5s nếu Silero VAD không khả dụng.
        """
        has_silero = self._init_silero_vad()
        
        if has_silero and self.vad_utils is not None:
            try:
                (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = self.vad_utils
                wav_tensor = torch.from_numpy(audio_np)
                speech_timestamps = get_speech_timestamps(
                    wav_tensor,
                    self.vad_model,
                    threshold=self.vad_threshold,
                    sampling_rate=self.sample_rate
                )

                segments = []
                for stamp in speech_timestamps:
                    start_ms = int((stamp['start'] / self.sample_rate) * 1000)
                    end_ms = int((stamp['end'] / self.sample_rate) * 1000)
                    start_sample = stamp['start']
                    end_sample = stamp['end']
                    
                    segments.append({
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "start_sample": start_sample,
                        "end_sample": end_sample
                    })

                logger.info(f"Silero VAD phát hiện {len(segments)} phân đoạn giọng nói.")
                return segments
            except Exception as e:
                logger.warning(f"Lỗi khi chạy Silero VAD ({e}). Chuyển sang Chunking Fallback.")

        # Chunking Fallback: Chia audio thành từng phân đoạn 5 giây
        chunk_sec = 5.0
        chunk_samples = int(chunk_sec * self.sample_rate)
        total_samples = len(audio_np)
        segments = []
        
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            if (end_sample - start_sample) < int(0.5 * self.sample_rate):
                continue
            start_ms = int((start_sample / self.sample_rate) * 1000)
            end_ms = int((end_sample / self.sample_rate) * 1000)
            segments.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_sample": start_sample,
                "end_sample": end_sample
            })

        logger.info(f"Chunking Fallback phân tách {len(segments)} phân đoạn audio (5s/chunk).")
        return segments

    def process_audio(self, video_path: str) -> Dict[str, str]:
        """
        Hàm chính: 
        - Tách audio mono 16kHz
        - Tìm phân đoạn giọng nói bằng Silero VAD / Chunking Fallback
        - Transcribe bằng PhoWhisper Small via transformers
        - Trả về dictionary map giữa timestamp của shot/đoạn giọng nói và nội dung text tương ứng.
        """
        if not os.path.exists(video_path):
            logger.error(f"Tệp video không tồn tại: {video_path}")
            return {}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        timestamp_text_map = {}

        try:
            # 1. Tách kênh audio mono 16kHz
            has_audio = self.extract_audio_ffmpeg(video_path, temp_wav_path)
            if not has_audio or not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) <= 1000:
                logger.info(f"Video {video_path} không có âm thanh hoặc file audio rỗng.")
                return {}

            # 2. Đọc file WAV bằng scipy.io.wavfile
            audio_np = read_wav_file_fallback(temp_wav_path, target_sr=self.sample_rate)
            if len(audio_np) == 0:
                return {}

            # 3. Tìm các phân đoạn có giọng nói qua Silero VAD hoặc Chunking Fallback
            speech_segments = self.get_speech_timestamps_vad(temp_wav_path, audio_np)
            if not speech_segments:
                logger.info("Không phát hiện phân đoạn lời nói nào trong video.")
                return {}

            # 4. Nạp PhoWhisper Small ASR Pipeline
            self._init_phowhisper()

            # 5. Transcribe từng đoạn giọng nói
            for seg in speech_segments:
                start_ms = seg["start_ms"]
                end_ms = seg["end_ms"]
                start_sample = seg["start_sample"]
                end_sample = seg["end_sample"]

                audio_chunk = audio_np[start_sample:end_sample]
                if len(audio_chunk) == 0:
                    continue

                try:
                    asr_result = self.asr_pipeline(
                        audio_chunk,
                        return_timestamps=False,
                        generate_kwargs={"language": "vi"}
                    )
                    text_content = asr_result.get("text", "").strip()
                except Exception as e:
                    logger.warning(f"Lỗi khi transcribe đoạn {start_ms}-{end_ms}ms: {e}")
                    text_content = ""

                if text_content:
                    ts_key = f"{format_timestamp(start_ms)} - {format_timestamp(end_ms)}"
                    timestamp_text_map[ts_key] = text_content

            logger.info(f"Đã hoàn thành ASR cho {len(timestamp_text_map)} phân đoạn.")
            return timestamp_text_map

        except Exception as e:
            logger.error(f"[XỬ LÝ LỖI] Lỗi bóc tách ASR âm thanh {video_path}: {e}")
            return {}

        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)


def process_video_audio(video_path: str) -> Dict[str, str]:
    processor = AudioASRProcessor()
    return processor.process_audio(video_path)


if __name__ == "__main__":
    import sys
    test_video = sys.argv[1] if len(sys.argv) > 1 else "data/dummy_videos/test_transnet.mp4"
    print(f"Chạy thử nghiệm AudioASRProcessor với video: {test_video}")
    if os.path.exists(test_video):
        res_map = process_video_audio(test_video)
        print("Kết quả ASR Map:", res_map)
    else:
        print("Tệp video test không tồn tại.")
