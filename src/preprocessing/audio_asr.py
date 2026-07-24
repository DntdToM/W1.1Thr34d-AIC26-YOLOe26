"""Audio processing and speech-to-text module using Silero VAD and PhoWhisper ASR."""

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
    """Format millisecond integer into hh:mm:ss.mmm string representation."""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def read_wav_file_fallback(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Read mono WAV audio file into float32 numpy array using scipy."""
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
        logger.error(f"Failed to read WAV audio file '{wav_path}': {e}")
        return np.array([], dtype=np.float32)


class AudioASRProcessor:
    """Audio extraction, Voice Activity Detection (VAD), and automatic speech recognition processor."""

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
        """Locate ffmpeg binary path prioritizing imageio_ffmpeg bundled executable."""
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if os.path.exists(exe):
                logger.info(f"Using bundled ffmpeg binary at '{exe}'")
                return exe
        except Exception:
            pass
        return "ffmpeg"

    def _init_silero_vad(self) -> bool:
        """Initialize Silero VAD model via torch.hub."""
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
                logger.info("Silero VAD model loaded successfully.")
                return True
            except Exception as e:
                logger.warning(f"Silero VAD loading failed ({e}). Falling back to uniform 5s chunking.")
                return False
        return True

    def _init_phowhisper(self):
        """Initialize PhoWhisper ASR pipeline."""
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
                logger.info(f"PhoWhisper model '{self.phowhisper_model_name}' loaded successfully (dtype={torch_dtype}).")
            except Exception as e:
                logger.error(f"PhoWhisper initialization failed for '{self.phowhisper_model_name}': {e}")
                raise e

    def extract_audio_ffmpeg(self, video_path: str, output_wav_path: str) -> bool:
        """Extract mono 16kHz WAV audio stream from video container via ffmpeg."""
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
            logger.warning(f"Audio extraction failed for video '{video_path}' (stream may be absent): {e}")
            return False

    def get_speech_timestamps_vad(self, wav_path: str, audio_np: np.ndarray) -> List[Dict[str, int]]:
        """Detect speech segments using Silero VAD with uniform chunking fallback."""
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

                logger.info(f"Silero VAD detected {len(segments)} speech segments.")
                return segments
            except Exception as e:
                logger.warning(f"Silero VAD execution failed ({e}). Reverting to fallback chunking.")

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

        logger.info(f"Fallback chunking partitioned audio into {len(segments)} segments (5s interval).")
        return segments

    def process_audio(self, video_path: str) -> Dict[str, str]:
        """Extract audio stream, perform VAD segmentation, run ASR transcription, and return timestamp-text mappings."""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        timestamp_text_map = {}

        try:
            has_audio = self.extract_audio_ffmpeg(video_path, temp_wav_path)
            if not has_audio or not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) <= 1000:
                logger.info(f"Video file '{video_path}' contains no audio track or empty audio stream.")
                return {}

            audio_np = read_wav_file_fallback(temp_wav_path, target_sr=self.sample_rate)
            if len(audio_np) == 0:
                return {}

            speech_segments = self.get_speech_timestamps_vad(temp_wav_path, audio_np)
            if not speech_segments:
                logger.info(f"No speech segments detected in video: {video_path}")
                return {}

            self._init_phowhisper()

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
                    logger.warning(f"Transcription failed for audio interval {start_ms}-{end_ms}ms: {e}")
                    text_content = ""

                if text_content:
                    ts_key = f"{format_timestamp(start_ms)} - {format_timestamp(end_ms)}"
                    timestamp_text_map[ts_key] = text_content

            logger.info(f"ASR transcription completed for {len(timestamp_text_map)} audio segments.")
            return timestamp_text_map

        except Exception as e:
            logger.error(f"Audio ASR pipeline failed for video '{video_path}': {e}")
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
    if os.path.exists(test_video):
        res_map = process_video_audio(test_video)
        print("ASR Map Result:", res_map)
    else:
        print("Test video file does not exist.")
