"""Compare old and shared STT settings on the same local sentence recordings."""
import json
import time
from pathlib import Path
from faster_whisper import WhisperModel
from core.ai.decision_v2 import contains_name


def main():
    root = Path(__file__).resolve().parent
    paths = sorted((root / "sentence_samples").glob("*/*.wav"))
    if not paths:
        raise SystemExit("No sentence samples found.")
    model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    # Exclude first-use initialization from timings for both configurations.
    list(model.transcribe(str(paths[0]), language="en")[0])
    rows = []
    for previous_context in [True, False]:
        elapsed_total = correct = 0
        for path in paths:
            start = time.perf_counter()
            segments, _ = model.transcribe(str(path), language="en", beam_size=5, vad_filter=False, word_timestamps=True, condition_on_previous_text=previous_context)
            text = " ".join(s.text.strip() for s in segments)
            elapsed = time.perf_counter() - start
            detected = contains_name(text, "Aryan")
            expected = path.parent.name == "positive"
            correct += detected == expected
            elapsed_total += elapsed
            rows.append(dict(file=path.name, label=path.parent.name, previous_context=previous_context, transcript=text, seconds=elapsed, correct=detected == expected))
        print(f"previous_context={previous_context}: name accuracy={correct}/{len(paths)}, average={elapsed_total / len(paths):.3f}s")
    (root / "benchmark_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
