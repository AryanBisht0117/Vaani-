"""One-shot background lifecycle for the Tkinter application."""
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_SIZE = 3


class VaaniBackend:
    """Start and stop the existing Vaani pipeline without blocking its UI."""

    def __init__(self, user_name, event_callback):
        self.user_name = user_name
        self.event_callback = event_callback
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._thread = None

    def start(self):
        with self._lock:
            if self._started:
                return False
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="vaani-gui-lifecycle",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self):
        self.stop_event.set()

    def _emit(self, event, payload=None):
        self.event_callback(event, payload)

    def _load_server_settings(self):
        config_path = Path(
            os.environ.get("VAANI_SERVER_CONFIG", ROOT / "llama-server-1.7b.json")
        )
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config_path = config_path.resolve()
        with config_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)["models"][0]

        model_path = Path(model_config["model"])
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"AI model not found: {model_path}")

        model_id = model_config.get("model_alias") or model_config["model"]
        return config_path, model_id

    def _run(self):
        server = None
        log_file = None
        listener = None
        processor = None
        model = None
        try:
            config_path, model_id = self._load_server_settings()
            if self.stop_event.is_set():
                return

            os.environ["VAANI_SERVER_CONFIG"] = str(config_path)
            os.environ["VAANI_USER_NAME"] = self.user_name
            os.environ["VAANI_AI_MODEL"] = model_id

            from start_vaani import (
                SERVER_LOG,
                launch_server,
                server_has_model,
                wait_for_server,
            )

            self._emit("status", ("Starting", "Checking the local AI server..."))
            if not server_has_model(model_id):
                log_file = SERVER_LOG.open("w", encoding="utf-8")
                server = launch_server(log_file)
                wait_for_server(
                    server,
                    model_id,
                    log_path=SERVER_LOG,
                    stop_event=self.stop_event,
                )
            if self.stop_event.is_set():
                return

            self._emit("status", ("Starting", "Loading Faster-Whisper..."))
            from app_v2_continuous import process_audio
            from core.ai.decision_v2 import AttentionDecisionService
            from core.audio.continuous_listener import listen_continuously
            from core.features.feature_combiner import load_whisper

            decision_service = AttentionDecisionService()
            decision_service.reset()
            model = load_whisper()
            if self.stop_event.is_set():
                return

            audio_queue = queue.Queue(maxsize=QUEUE_SIZE)
            self._emit("status", ("Starting", "Opening the microphone..."))

            def process_recordings():
                while not self.stop_event.is_set():
                    try:
                        audio = audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    self._emit("status", ("Processing", "Transcribing and classifying speech..."))
                    succeeded = False
                    try:
                        process_audio(
                            audio,
                            model,
                            self.user_name,
                            on_result=lambda result: self._emit("result", result),
                            decision_service=decision_service,
                        )
                        succeeded = True
                    except Exception as exc:
                        print(f"Processing error: {exc}")
                        self._emit("processing_error", str(exc))
                    finally:
                        audio_queue.task_done()
                    if succeeded and not self.stop_event.is_set():
                        self._emit("status", ("Listening", "Listening for nearby speech..."))

            def listen():
                try:
                    listen_continuously(
                        audio_queue,
                        self.stop_event,
                        on_ready=lambda: self._emit(
                            "status", ("Listening", "Listening for nearby speech...")
                        ),
                    )
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._emit("error", str(exc))
                    self.stop_event.set()

            processor = threading.Thread(
                target=process_recordings,
                name="vaani-gui-processor",
                daemon=True,
            )
            listener = threading.Thread(
                target=listen,
                name="vaani-gui-listener",
                daemon=True,
            )
            processor.start()
            listener.start()
            listener.join()
        except InterruptedError:
            if not self.stop_event.is_set():
                self._emit("error", "Startup was interrupted unexpectedly.")
        except Exception as exc:
            if not self.stop_event.is_set():
                self._emit("error", str(exc))
        finally:
            self.stop_event.set()
            if listener is not None and listener.is_alive():
                listener.join()
            if processor is not None and processor.is_alive():
                processor.join()
            if "decision_service" in locals():
                decision_service.reset()
            model = None

            try:
                if server is not None and server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._emit("error", f"Could not stop the local AI server cleanly: {exc}")
            finally:
                if log_file is not None:
                    try:
                        log_file.close()
                    except OSError as exc:
                        self._emit("error", f"Could not close the AI server log: {exc}")
                self._emit("stopped")
