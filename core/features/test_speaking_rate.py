from faster_whisper import WhisperModel

AUDIO_FILE = "audio.wav"

print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Transcribing...")

segments, info = model.transcribe(
    AUDIO_FILE,
    language=None,
    word_timestamps=True
)

total_words = 0
start_time = None
end_time = None

for segment in segments:
    if start_time is None:
        start_time = segment.start

    end_time = segment.end

    if segment.words:
        total_words += len(segment.words)

duration = end_time - start_time

words_per_second = total_words / duration
words_per_minute = words_per_second * 60

print("\nSpeaking Rate:")
print(f"Words: {total_words}")
print(f"Duration: {duration:.2f} seconds")
print(f"Rate: {words_per_minute:.1f} words/minute")