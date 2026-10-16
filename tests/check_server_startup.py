"""Check launcher startup without opening the microphone."""
import subprocess
from start_vaani import ROOT, launch_server, wait_for_server
from core.config import MODEL


def main():
    with (ROOT / "startup-diagnostic.log").open("w", encoding="utf-8") as log:
        process = launch_server(log)
        try:
            wait_for_server(process, MODEL, log_path=ROOT / "startup-diagnostic.log")
            print("AI server ready.")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
