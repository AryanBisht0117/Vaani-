def main():
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
        language=None
    )

    print("\nTranscription:")

    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

    print("\nDone.")


if __name__ == "__main__":
    main()
