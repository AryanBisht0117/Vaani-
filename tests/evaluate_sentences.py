def main():
    from pathlib import Path
    import re
    import time

    from faster_whisper import WhisperModel

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATASET_DIR = PROJECT_ROOT / "tests" / "sentence_samples"

    print("Loading Whisper Tiny...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    tp = fp = tn = fn = 0
    times = []

    for label in ("positive", "negative"):
        for audio_path in sorted((DATASET_DIR / label).glob("*.wav")):
            start = time.perf_counter()

            segments, _ = model.transcribe(
                str(audio_path),
                language="en",
                beam_size=5,
                vad_filter=False
            )

            transcript = " ".join(segment.text for segment in segments).strip()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            detected = bool(re.search(r"\baryan\b", transcript, re.IGNORECASE))
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
                f"{audio_path.name:18} "
                f"Expected: {label:8} "
                f"Detected: {str(detected):5} "
                f"STT: {transcript!r} "
                f"Time: {elapsed:.3f}s"
            )

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print("\n--- VAANI V2: SENTENCE RECOGNITION ---")
    print(f"True Positives:  {tp}")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"Accuracy:        {accuracy:.2%}")
    print(f"Precision:       {precision:.2%}")
    print(f"Recall:          {recall:.2%}")
    print(f"F1 Score:        {f1:.2%}")
    print(f"Average STT time: {sum(times) / len(times):.3f}s" if times else "No samples found")


if __name__ == "__main__":
    main()
