def main():
    from pathlib import Path
    import shutil
    import subprocess
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RECORDER = PROJECT_ROOT / "core" / "audio" / "microphone_v2.py"
    SOURCE_AUDIO = PROJECT_ROOT / "audio_v2.wav"
    DATASET_DIR = PROJECT_ROOT / "tests" / "audio_samples"

    SAMPLES_PER_CLASS = 10

    for label in ("positive", "negative"):
        folder = DATASET_DIR / label
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {label.upper()} SAMPLES ---")

        for number in range(1, SAMPLES_PER_CLASS + 1):
            destination = folder / f"{label}_{number:02d}.wav"

            if destination.exists():
                print(f"Already exists: {destination.name}")
                continue

            input(f"\nSample {number}/10 — Press Enter when ready...")

            # Remove the previous recording so it cannot be reused accidentally.
            SOURCE_AUDIO.unlink(missing_ok=True)

            result = subprocess.run(
                [sys.executable, str(RECORDER)],
                cwd=PROJECT_ROOT
            )

            if result.returncode != 0 or not SOURCE_AUDIO.exists():
                print("Recording failed. Please try again.")
                break

            shutil.move(str(SOURCE_AUDIO), str(destination))
            print(f"Saved: {destination.relative_to(PROJECT_ROOT)}")

    print("\nDataset recording session finished.")


if __name__ == "__main__":
    main()
