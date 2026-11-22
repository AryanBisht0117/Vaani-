
import subprocess
import sys
import time
import wave
from pathlib import Path
from core.alerts.notifications import show_notification

import numpy as np
from faster_whisper import WhisperModel

from core.ai.decision_v2 import decide_alert


ROOT = Path(__file__).resolve().parent
AUDIO_FILE = ROOT / "audio_v2.wav"
RECORDER = ROOT / "core" / "audio" / "microphone_v2.py"

USER_NAME = "Aryan"  # Temporary setting; configurable later.

print("Loading Whisper Tiny...")
model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
)

print("Vaani V2 ready.")
print(f"Configured user: {USER_NAME}")


def extract_loudness(path):
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    audio /= 32768.0

    if len(audio) == 0:
        return 0.0

    return float(np.sqrt(np.mean(audio ** 2)))


def process_recording():
    print("\nStarting microphone...")

    if AUDIO_FILE.exists():
        AUDIO_FILE.unlink()
    
    subprocess.run(
        [sys.executable, str(RECORDER)],
        cwd=ROOT,
        check=True,
    )

    if not AUDIO_FILE.exists():
        print("No recording found.")
        return

    start = time.perf_counter()

    segments, info = model.transcribe(
        str(AUDIO_FILE),
        language="en",
        beam_size=5,
        vad_filter=False,
        word_timestamps=True,
    )

    segments = list(segments)

    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    words = [
        word
        for segment in segments
        for word in (segment.words or [])
    ]

    if words:
        duration = max(
            words[-1].end - words[0].start,
            0.01,
        )
        speaking_rate = len(words) / duration * 60
    else:
        speaking_rate = 0.0

    loudness = extract_loudness(AUDIO_FILE)

    name_count = sum(
        word.strip(".,!? ").casefold() == USER_NAME.casefold()
        for word in transcript.split()
    )

    decision = decide_alert(
        transcript=transcript,
        user_name=USER_NAME,
        loudness=loudness,
        speaking_rate=speaking_rate,
        repetition_count=name_count,
    )

    elapsed = time.perf_counter() - start

    print("\n--- VAANI V2 RESULT ---")
    print(f"Transcript: {transcript!r}")
    print(f"Loudness: {loudness:.5f}")
    print(f"Speaking rate: {speaking_rate:.1f} WPM")
    print(f"Name occurrences: {name_count}")
    print(f"Decision: {decision}")
    show_notification(decision, transcript)
    print(f"Processing time: {elapsed:.2f}s")


if __name__ == "__main__":
    print("\nVaani V2 continuous listening started.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            process_recording()
    except KeyboardInterrupt:
        print("\nVaani stopped.")
