"""Extract speech and audio features in one transcription pass."""
from dataclasses import dataclass
import numpy as np
from core.ai.decision_v2 import count_name
from core.config import BEAM_SIZE, CPU_THREADS, WHISPER_MODEL


@dataclass(frozen=True)
class SpeechFeatures:
    transcript: str
    loudness: float
    speaking_rate: float
    repetition_count: int


def load_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)


def extract_features(audio, model, user_name):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono audio at 16000 Hz.")
    if not audio.size:
        return SpeechFeatures("", 0.0, 0.0, 0)
    loudness = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    segments, _ = model.transcribe(
        audio, language="en", beam_size=BEAM_SIZE, vad_filter=False,
        word_timestamps=True, condition_on_previous_text=False,
    )
    segments = list(segments)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    words = [word for segment in segments for word in (segment.words or [])]
    duration = words[-1].end - words[0].start if words else 0.0
    rate = len(words) * 60 / duration if duration > 0 else 0.0
    return SpeechFeatures(transcript, loudness, rate, count_name(transcript, user_name))
