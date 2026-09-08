import wave
import numpy as np

AUDIO_FILE = "audio.wav"

with wave.open(AUDIO_FILE, "rb") as file:
    frames = file.readframes(file.getnframes())

audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

# Normalize to -1.0 to 1.0
audio = audio / 32768.0

# RMS loudness
rms = np.sqrt(np.mean(audio ** 2))

print(f"RMS Loudness: {rms:.4f}")