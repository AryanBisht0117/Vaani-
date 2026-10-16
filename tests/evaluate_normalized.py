def main():
    from pathlib import Path
    import wave
    import time
    import re

    import numpy as np
    from faster_whisper import WhisperModel

    DATASET = Path(__file__).resolve().parent / "audio_samples" / "positive"
    TARGET_RMS_DBFS = -25.0

    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    correct = 0
    total = 0

    print("\n--- VAANI V2: NORMALIZED AUDIO TEST ---\n")

    for path in sorted(DATASET.glob("*.wav")):
        with wave.open(str(path), "rb") as f:
            sample_rate = f.getframerate()
            channels = f.getnchannels()
            audio = np.frombuffer(
                f.readframes(f.getnframes()),
                dtype=np.int16
            ).astype(np.float32) / 32768.0

        if sample_rate != 16000 or channels != 1 or f.getsampwidth() != 2:
            raise ValueError(f"Unexpected audio format: {path.name}")

        if not audio.size:
            print(f"Skipping empty recording: {path.name}")
            continue
        original_rms = np.sqrt(np.mean(audio ** 2))
        target_rms = 10 ** (TARGET_RMS_DBFS / 20)

        if original_rms > 0:
            gain = target_rms / original_rms
            normalized = audio * gain
        else:
            normalized = audio.copy()

        # Prevent clipping while preserving the waveform shape.
        peak = np.max(np.abs(normalized))
        if peak > 0.99:
            normalized *= 0.99 / peak

        start = time.perf_counter()

        segments, _ = model.transcribe(
            normalized,
            language="en",
            beam_size=5,
            vad_filter=False
        )

        transcript = " ".join(
            segment.text.strip() for segment in segments
        )

        elapsed = time.perf_counter() - start

        detected = bool(
            re.search(r"\baryan\b", transcript, re.IGNORECASE)
        )

        total += 1
        correct += int(detected)

        status = "PASS" if detected else "FAIL"

        print(
            f"{path.name}: {status:<4} | "
            f"STT: {transcript!r} | "
            f"Time: {elapsed:.3f}s"
        )

    print("\n--- RESULTS ---")
    print(f"Detected: {correct}/{total}")
    print(f"Recall: {correct / total:.2%}" if total else "No samples found")
    print("Original baseline recall: 50.00%")


if __name__ == "__main__":
    main()
