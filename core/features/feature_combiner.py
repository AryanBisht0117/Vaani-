
"""Extract speech features using online Deepgram STT."""
from dataclasses import dataclass
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error

import numpy as np

from core.ai.decision_v2 import count_name

SAMPLE_RATE = 16000
API_URL = "https://api.deepgram.com/v1/listen"


@dataclass(frozen=True)
class SpeechFeatures:
    transcript: str
    loudness: float
    speaking_rate: float
    repetition_count: int


def extract_features(audio, model, user_name):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono audio at 16000 Hz.")

    if not audio.size:
        return SpeechFeatures("", 0.0, 0.0, 0)

    loudness = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    duration = len(audio) / SAMPLE_RATE

    # Convert float audio (-1.0 to 1.0) into signed 16-bit PCM.
    pcm_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")

    params = urllib.parse.urlencode(
        [
            ("model", "nova-3"),
            ("language", "multi"),
            ("language_hint", "en"),
            ("language_hint", "hi"),
            ("encoding", "linear16"),
            ("sample_rate", str(SAMPLE_RATE)),
            ("channels", "1"),
            ("smart_format", "true"),
        ]
    )

    request = urllib.request.Request(
        f"{API_URL}?{params}",
        data=pcm_audio.tobytes(),
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )

    try:
        request_start = time.perf_counter()

        with urllib.request.urlopen(request, timeout=12) as response:
            response_data = response.read()

        request_elapsed = time.perf_counter() - request_start
        print(f"Deepgram HTTP time: {request_elapsed:.2f}s")

        result = json.loads(response_data.decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Deepgram transcription failed: {exc}") from exc

    try:
        transcript = (
            result["results"]["channels"][0]["alternatives"][0]["transcript"]
        ).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Malformed Deepgram transcription response.") from exc

    word_count = len(transcript.split())
    speaking_rate = word_count * 60 / duration if duration > 0 else 0.0

    return SpeechFeatures(
        transcript=transcript,
        loudness=loudness,
        speaking_rate=speaking_rate,
        repetition_count=count_name(transcript, user_name),
    )
