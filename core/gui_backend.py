
"""Background lifecycle for Vaani with online STT and AI."""
import os
import queue
import threading

from core.audio.continuous_listener import listen_continuously

QUEUE_SIZE = 3


class VaaniBackend:
    """Start and stop the online Vaani processing pipeline."""

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

    def _run(self):
        listener = None
        processor = None

        try:
            if self.stop_event.is_set():
                return

            os.environ["VAANI_USER_NAME"] = self.user_name

            from app_v2_continuous import process_audio
            from core.ai.decision_v2 import AttentionDecisionService

            decision_service = AttentionDecisionService()
            decision_service.reset()

            audio_queue = queue.Queue(maxsize=QUEUE_SIZE)

            self._emit("status", ("Starting", "Opening microphone..."))

            def process_recordings():
                while not self.stop_event.is_set():
                    try:
                        audio = audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    self._emit(
                        "status",
                        ("Processing", "Transcribing and classifying speech..."),
                    )

                    succeeded = False

                    try:
                        process_audio(
                            audio,
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
                        self._emit(
                            "status",
                            ("Listening", "Listening for nearby speech..."),
                        )

            def listen():
                try:
                    listen_continuously(
                        audio_queue,
                        self.stop_event,
                        on_ready=lambda: self._emit(
                            "status",
                            ("Listening", "Listening for nearby speech..."),
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

        except Exception as exc:
            if not self.stop_event.is_set():
                self._emit("error", str(exc))

        finally:
            self.stop_event.set()

            if listener is not None and listener.is_alive():
                listener.join()

            if processor is not None and processor.is_alive():
                processor.join(timeout=25)

            if "decision_service" in locals():
                decision_service.reset()

            self._emit("stopped")
