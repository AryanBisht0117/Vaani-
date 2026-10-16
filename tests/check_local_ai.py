"""Exercise the configured local server on labeled attention examples."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from core.ai.decision_v2 import decide_alert
from core.ai import classifier
from core.config import ROOT, SERVER_CONFIG, MODEL
from start_vaani import wait_for_server


def main():
    examples = [
        ("Aryan went shopping yesterday.", "IGNORE"),
        ("The fire drill is tomorrow.", "IGNORE"),
        ("Don't call the police, this is only a rehearsal.", "IGNORE"),
        ("Can you help me, Priya?", "IGNORE"),
        ("Aryan, please come here.", "FULL_ALERT"),
        ("Someone has fallen and is not breathing. Call an ambulance!", "FULL_ALERT"),
        ("Excuse me, could someone help me carry this?", "SOFT_ALERT"),
        ("We watched a movie last night.", "IGNORE"),
        ("Aryan told me he would arrive tomorrow.", "IGNORE"),
        ("Could you open the door for me, Neha?", "IGNORE"),
        ("Would someone please hold this door for me?", "SOFT_ALERT"),
        ("The kitchen is on fire right now, get outside!", "FULL_ALERT"),
    ]
    log_path = ROOT / "vaani-server.log"
    with log_path.open("a", encoding="utf-8") as log:
        env = os.environ.copy()
        env["PORT"] = "8011"
        env["HOST"] = "127.0.0.1"
        server = subprocess.Popen(
            [sys.executable, "-m", "llama_cpp.server", "--config_file", str(SERVER_CONFIG)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_for_server(server, MODEL, server_url="http://127.0.0.1:8011/v1/models")
            classifier.API_URL = "http://127.0.0.1:8011/v1/chat/completions"
            rows = []
            for text, expected in examples:
                start = time.perf_counter()
                actual = decide_alert(text, "Aryan")
                row = dict(transcript=text, expected=expected, actual=actual, seconds=time.perf_counter() - start)
                rows.append(row)
                print(json.dumps(row), flush=True)
            Path(__file__).with_name("ai_check_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"Correct: {sum(r['actual'] == r['expected'] for r in rows)}/{len(rows)}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
