
import json
import os
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path
from core.config import API_URL, SERVER_CONFIG

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
CONFIG = SERVER_CONFIG
SERVER_URL = API_URL.removesuffix("/chat/completions") + "/models"
SERVER_LOG = ROOT / "vaani-server.log"


def launch_server(log_file):
    """Bind to the same endpoint used by the classifier and readiness check."""
    endpoint = urlsplit(API_URL)
    if endpoint.scheme != "http" or not endpoint.hostname:
        raise ValueError("VAANI_API_URL must be an HTTP server endpoint.")
    env = os.environ.copy()
    env["HOST"] = endpoint.hostname
    env["PORT"] = str(endpoint.port or 80)
    # The installed server gives CONFIG_FILE precedence over --config_file.
    env.pop("CONFIG_FILE", None)
    return subprocess.Popen(
        [PYTHON, "-m", "llama_cpp.server", "--config_file", str(CONFIG)],
        cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def server_has_model(model_id):
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=2) as response:
            data = json.load(response)
    except (OSError, ValueError):
        return False
    ids = {item.get("id") for item in data.get("data", [])}
    if model_id not in ids:
        raise RuntimeError(f"An AI server is already running with different models: {ids}. Expected {model_id}.")
    return True


def wait_for_server(
    process, model_id, timeout=90, server_url=SERVER_URL, log_path=None,
    stop_event=None,
):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("AI server startup was cancelled.")
        if process.poll() is not None:
            details = ""
            if log_path is not None:
                details = f"\nServer log: {log_path}"
                try:
                    tail = "\n".join(Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
                    details += f"\n{tail}"
                except OSError:
                    pass
            raise RuntimeError(f"AI server exited during startup (exit code {process.returncode}).{details}")

        try:
            with urllib.request.urlopen(server_url, timeout=2) as response:
                data = json.load(response)

            model_ids = {
                item["id"] for item in data.get("data", [])
            }

            if model_id in model_ids:
                return

        except (OSError, ValueError, KeyError):
            pass

        if stop_event is None:
            time.sleep(1)
        elif stop_event.wait(min(1, max(0, deadline - time.monotonic()))):
            raise InterruptedError("AI server startup was cancelled.")

    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("AI server startup was cancelled.")
    raise TimeoutError("AI model did not become ready in time.")


def main():
    print("\n========== VAANI V2 ==========\n")

    user_name = input("What is your name? ").strip()

    while not user_name:
        user_name = input("Please enter your name: ").strip()

    print(f"\nWelcome, {user_name}!")
    with CONFIG.open(encoding="utf-8") as config_file:
        model_config = json.load(config_file)["models"][0]
    model_path = Path(model_config["model"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"AI model not found: {model_path}")
    model_id = model_config.get("model_alias") or model_config["model"]
    print("Starting local AI model...")

    server = None
    log_file = None
    try:
        if not server_has_model(model_id):
            log_file = SERVER_LOG.open("w", encoding="utf-8")
            server = launch_server(log_file)
        if server is not None:
            wait_for_server(server, model_id, log_path=SERVER_LOG)
        print("AI model ready.")

        env = os.environ.copy()
        env["VAANI_USER_NAME"] = user_name
        env["VAANI_AI_MODEL"] = model_id
        env["VAANI_API_URL"] = API_URL

        print("Starting microphone...")
        print("Press Ctrl+C to stop Vaani.\n")

        try:
            subprocess.run(
                [PYTHON, str(ROOT / "app_v2_continuous.py")],
                cwd=ROOT,
                env=env,
                check=True,
            )
        except KeyboardInterrupt:
            print("\nShutdown requested.")

    finally:
        print("\nShutting down Vaani...")

        if server is not None and server.poll() is None:
            server.terminate()

            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

        if log_file is not None:
            log_file.close()

        print("Vaani stopped.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"\nVaani startup failed: {exc}", file=sys.stderr)
        sys.exit(1)
