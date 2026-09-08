import torch
import wave
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

AUDIO_FILE = "audio.wav"
SAMPLE_RATE = 16000

print("Loading VAD model...")
model = load_silero_vad()

print("Reading audio...")

with wave.open(AUDIO_FILE, "rb") as file:
    frames = file.readframes(file.getnframes())

audio = np.frombuffer(frames, dtype=np.int16)
audio = torch.from_numpy(audio).float() / 32768.0

print("Detecting speech...")

speech_timestamps = get_speech_timestamps(
    audio,
    model,
    sampling_rate=SAMPLE_RATE
)

print("\nSpeech detected:")

if not speech_timestamps:
    print("No speech detected.")
else:
    total_speech = 0

    for segment in speech_timestamps:
        start = segment["start"] / SAMPLE_RATE
        end = segment["end"] / SAMPLE_RATE
        duration = end - start
        total_speech += duration

        print(f"{start:.2f}s → {end:.2f}s ({duration:.2f}s)")

    print(f"\nTotal speech: {total_speech:.2f} seconds")