def main():
    import sounddevice as sd
    import numpy as np
    import wave

    SAMPLE_RATE = 16000
    DURATION = 5

    print("Recording for 5 seconds...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )

    sd.wait()

    with wave.open("audio.wav", "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(audio.tobytes())

    print("Recording complete.")
    print("Saved: audio.wav")


if __name__ == "__main__":
    main()
