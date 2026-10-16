def main():
    from pathlib import Path
    import re
    import time
    import wave

    import numpy as np
    from faster_whisper import WhisperModel

    ROOT = Path(__file__).resolve().parent / "sentence_samples"
    TARGET_NAME = "Aryan"

    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def transcribe(audio):
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False
        )
        return " ".join(s.text for s in segments).strip()

    def contains_name(text):
        return bool(re.search(
            rf"\b{re.escape(TARGET_NAME)}\b",
            text,
            re.IGNORECASE
        ))

    tp = fp = tn = fn = 0
    times = []
    second_passes = 0

    for label in ("positive", "negative"):
        for path in sorted((ROOT / label).glob("*.wav")):
            with wave.open(str(path), "rb") as f:
                sr = f.getframerate()
            if sr != 16000 or f.getnchannels() != 1 or f.getsampwidth() != 2:
                raise ValueError(f"Expected mono 16-bit 16000 Hz audio: {path}")
                audio = np.frombuffer(
                    f.readframes(f.getnframes()),
                    dtype=np.int16
                ).astype(np.float32) / 32768.0

            start = time.perf_counter()

            full = transcribe(audio)
            detected = contains_name(full)
            cropped = "SKIPPED"

            if not detected:
                second_passes += 1
                cropped = transcribe(audio[:int(1.4 * sr)])
                detected = contains_name(cropped)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            expected = label == "positive"

            if expected and detected:
                tp += 1
            elif expected:
                fn += 1
            elif detected:
                fp += 1
            else:
                tn += 1

            print(
                f"{path.name}: full={full!r}, "
                f"crop={cropped!r}, detected={detected}, "
                f"time={elapsed:.3f}s"
            )

    print("\n--- CONDITIONAL TWO-PASS RESULTS ---")
    print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"Second passes: {second_passes}/{len(times)}")
    print(f"Average processing time: {sum(times)/len(times):.3f}s" if times else "No samples found")
    print(f"Maximum processing time: {max(times):.3f}s" if times else "No samples found")


if __name__ == "__main__":
    main()
