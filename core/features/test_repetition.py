from faster_whisper import WhisperModel
from collections import Counter
import re

AUDIO_FILE = "audio.wav"

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    AUDIO_FILE,
    language=None
)

text = " ".join(segment.text for segment in segments)

print("Transcript:")
print(text)

# Extract individual words
words = re.findall(r"\b\w+\b", text.lower())

counts = Counter(words)

repeated = {
    word: count
    for word, count in counts.items()
    if count > 1
}

print("\nRepeated words:")

if repeated:
    for word, count in repeated.items():
        print(f"{word}: {count} times")
else:
    print("No repetition detected.")