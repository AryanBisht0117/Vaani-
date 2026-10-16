"""Continuous microphone processing without temporary WAV files."""
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional
from core.audio.continuous_listener import listen_continuously
from core.ai.decision_v2 import AttentionDecisionService, decide_alert
from core.alerts.notifications import show_notification
from core.features.feature_combiner import extract_features, load_whisper

QUEUE_SIZE = 3


@dataclass(frozen=True)
class ProcessingResult:
    transcript: str
    decision: str
    elapsed_seconds: float


def process_audio(
    audio, model, user_name,
    on_result: Optional[Callable[[ProcessingResult], None]] = None,
    decision_service: Optional[AttentionDecisionService] = None,
):
    start = time.perf_counter()
    features = extract_features(audio, model, user_name)
    stt_elapsed = time.perf_counter() - start
    if not features.transcript:
        return
    decide = decision_service.decide_alert if decision_service is not None else decide_alert
    decision = decide(
        transcript=features.transcript, user_name=user_name,
        loudness=features.loudness, speaking_rate=features.speaking_rate,
        repetition_count=features.repetition_count,
    )
    elapsed = time.perf_counter() - start
    print("\n--- VAANI V2 RESULT ---")
    print(f"Transcript: {features.transcript!r}")
    print(f"Loudness: {features.loudness:.5f}")
    print(f"Speaking rate: {features.speaking_rate:.1f} WPM")
    print(f"Name occurrences: {features.repetition_count}")
    print(f"Decision: {decision}")
    print(f"Processing time: {elapsed:.2f}s (Whisper: {stt_elapsed:.2f}s)")
    if on_result is not None:
        on_result(ProcessingResult(features.transcript, decision, elapsed))
    show_notification(decision, features.transcript)
    return decision


def processing_worker(audio_queue, stop_event, model, user_name):
    decision_service = AttentionDecisionService()
    try:
        while not stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                process_audio(
                    audio, model, user_name, decision_service=decision_service
                )
            except Exception as exc:
                print(f"Processing error: {exc}")
            finally:
                audio_queue.task_done()
    finally:
        decision_service.reset()


def main():
    user_name = os.environ.get("VAANI_USER_NAME", "").strip()
    if not user_name:
        raise ValueError("Set VAANI_USER_NAME or run start_vaani.py.")
    print("Loading Whisper...")
    model = load_whisper()
    audio_queue = queue.Queue(maxsize=QUEUE_SIZE)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=processing_worker, args=(audio_queue, stop_event, model, user_name), daemon=True,
    )
    worker.start()
    print(f"Configured user: {user_name}. Press Ctrl+C to stop.")
    try:
        listen_continuously(audio_queue, stop_event)
    except KeyboardInterrupt:
        print("\nStopping Vaani...")
    finally:
        stop_event.set()
        worker.join(timeout=20)
        print("Vaani stopped.")


if __name__ == "__main__":
    main()
