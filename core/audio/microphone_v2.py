def main():
    import collections
    import wave

    import numpy as np
    import sounddevice as sd
    from silero_vad import load_silero_vad, VADIterator
    from core.audio.microphone_selection import resolve_builtin_microphone

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512
    SILENCE_SECONDS = 0.8
    MAX_RECORDING_SECONDS = 10
    PRE_SPEECH_SECONDS = 1.5

    device = resolve_builtin_microphone(SAMPLE_RATE)
    model = load_silero_vad()
    vad = VADIterator(
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_silence_duration_ms=800,
    )

    pre_buffer = collections.deque(
        maxlen=int(PRE_SPEECH_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
    )

    recorded_chunks = []
    speech_started = False
    speech_start_time = None

    print("Listening... Speak when ready.")

    try:
        with sd.InputStream(
        device=device,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE
    ) as stream:

            while True:
                chunk, overflowed = stream.read(CHUNK_SIZE)

                if overflowed:
                    print("Warning: audio input overflow.")

                mono = chunk[:, 0].copy()
                event = vad(mono)

                if not speech_started:
                    pre_buffer.append(mono)

                    if event and "start" in event:
                        speech_started = True
                        speech_start_time = event["start"] / SAMPLE_RATE
                        recorded_chunks.extend(pre_buffer)
                        print("Speech detected. Recording...")

                else:
                    recorded_chunks.append(mono)

                    if event and "end" in event:
                        print("Speech ended.")
                        break

                    if (
                        len(recorded_chunks) * CHUNK_SIZE / SAMPLE_RATE
                        >= MAX_RECORDING_SECONDS
                    ):
                        print("Maximum recording duration reached.")
                        break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    if recorded_chunks:
        audio = np.concatenate(recorded_chunks)
        audio_int16 = np.clip(
            audio * 32767, -32768, 32767
        ).astype(np.int16)

        with wave.open("audio_v2.wav", "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(SAMPLE_RATE)
            file.writeframes(audio_int16.tobytes())

        print(f"Saved: audio_v2.wav ({len(audio) / SAMPLE_RATE:.2f}s)")
    else:
        print("No speech recorded.")


if __name__ == "__main__":
    main()
