def main():
    from pathlib import Path
    import re
    import time
    from faster_whisper import WhisperModel

    DATASET = Path(__file__).resolve().parent / "audio_samples"
    TARGET = "aryan"

    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    results = []

    for label in ("positive", "negative"):
        folder = DATASET / label

        for audio_file in sorted(folder.glob("*.wav")):
            start = time.perf_counter()

            segments, _ = model.transcribe(
                str(audio_file),
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

            results.append({
                "file": audio_file.name,
                "label": label,
                "transcript": transcript,
                "detected": detected,
                "time": elapsed
            })

            print(
                f"{audio_file.name:<18} "
                f"Expected: {label:<8} "
                f"Detected: {str(detected):<5} "
                f"STT: {transcript!r} "
                f"Time: {elapsed:.3f}s"
            )

    # negative_02 contains "Arian", which can sound identical to "Aryan".
    # Exclude this ambiguous sample from strict accuracy calculations.
    scored = [
        r for r in results
        if r["file"] != "negative_02.wav"
    ]

    tp = sum(r["label"] == "positive" and r["detected"] for r in scored)
    fn = sum(r["label"] == "positive" and not r["detected"] for r in scored)
    fp = sum(r["label"] == "negative" and r["detected"] for r in scored)
    tn = sum(r["label"] == "negative" and not r["detected"] for r in scored)

    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    avg_time = sum(r["time"] for r in results) / len(results) if results else 0.0

    print("\n--- VAANI V2: WHISPER TINY BASELINE ---")
    print(f"True Positives:  {tp}")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"Accuracy:        {accuracy:.2%}")
    print(f"Precision:       {precision:.2%}")
    print(f"Recall:          {recall:.2%}")
    print(f"F1 Score:        {f1:.2%}")
    print(f"Average STT time: {avg_time:.3f}s")
    print("Excluded from scoring: negative_02.wav (Arian)")


if __name__ == "__main__":
    main()
