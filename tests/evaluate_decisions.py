"""Evaluate the existing Vaani decision pipeline against labeled examples."""
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from core.ai import classifier, decision_v2
from core.config import API_URL, MODEL, SERVER_CONFIG
from start_vaani import SERVER_URL, server_has_model

USER_NAME = "Aryan"


@dataclass(frozen=True)
class Example:
    sentence: str
    expected: str


EXAMPLES = (
    Example("Aryan, could you come here for a moment?", "FULL_ALERT"),
    Example("Aryan, please listen to me.", "FULL_ALERT"),
    Example("Hey Aryan, are you free for a second?", "FULL_ALERT"),
    Example("Sam, could you help me with these boxes?", "IGNORE"),
    Example("Priya, would you open the door?", "IGNORE"),
    Example("Could someone please help me carry this bag?", "SOFT_ALERT"),
    Example("Excuse me, could anyone tell me where platform four is?", "SOFT_ALERT"),
    Example("Can you hear me from over there?", "SOFT_ALERT"),
    Example("Help me!", "FULL_ALERT"),
    Example("Someone has fallen and isn't breathing; call an ambulance now!", "FULL_ALERT"),
    Example("The kitchen is on fire right now; everyone get outside!", "FULL_ALERT"),
    Example("Look out, that cyclist is about to hit you!", "FULL_ALERT"),
    Example("I went to the market after work and bought some apples.", "IGNORE"),
    Example("Aryan said he would join us next week.", "IGNORE"),
    Example("The meeting starts at three, so I'll be there early.", "IGNORE"),
    Example("Do you know whether the library is open today?", "IGNORE"),
    Example('In the film, the actor shouted, "Help me!" before the scene ended.', "IGNORE"),
    Example("If there were a fire, I would call the fire department.", "IGNORE"),
    Example("During the drill, pretend someone collapsed and call an ambulance.", "IGNORE"),
    Example("Would you give me a hand with these groceries?", "SOFT_ALERT"),
)


def print_server_instructions():
    endpoint = urlsplit(API_URL)
    host = endpoint.hostname or "127.0.0.1"
    port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    python = Path(sys.executable)

    print(f"Local AI server is not available at {SERVER_URL}.")
    print("Start the configured local model manually in a separate PowerShell window:")
    print(f"$env:HOST = '{host}'")
    print(f"$env:PORT = '{port}'")
    print(f'& "{python}" -m llama_cpp.server --config_file "{SERVER_CONFIG}"')
    print("\nAfter the server is ready, run this evaluation again:")
    print(f'& "{python}" "{Path(__file__).resolve()}"')


def run_example(index, example):
    classifier_called = False
    classifier_error = None

    def tracked_classifier(**kwargs):
        nonlocal classifier_called, classifier_error
        classifier_called = True
        try:
            return classifier.classify_speech(**kwargs)
        except Exception as exc:
            classifier_error = exc
            raise

    start = time.perf_counter()
    decision_error = None
    with patch.object(decision_v2, "classify_speech", side_effect=tracked_classifier):
        try:
            actual = decision_v2.decide_alert(example.sentence, USER_NAME)
        except Exception as exc:
            actual = "ERROR"
            decision_error = exc
    elapsed = time.perf_counter() - start

    source = "AI classifier" if classifier_called else "hardcoded rule"
    error = classifier_error or decision_error
    passed = error is None and actual == example.expected
    status = "PASS" if passed else "FAIL"

    if classifier_error is not None and actual != "ERROR":
        actual_display = f"{actual} (fallback after classifier error)"
    else:
        actual_display = actual

    print(f"\n[{index:02d}/{len(EXAMPLES)}] {example.sentence}")
    print(f"  Expected: {example.expected}")
    print(f"  Actual:   {actual_display}")
    print(f"  Source:   {source}" + (" — ERROR" if error is not None else ""))
    if error is not None:
        print(f"  Error:    {type(error).__name__}: {error}")
    print(f"  Result:   {status}")
    print(f"  Time:     {elapsed:.3f}s")

    return passed, error


def main():
    if len(EXAMPLES) != 20:
        raise RuntimeError(f"Expected exactly 20 examples, found {len(EXAMPLES)}.")

    try:
        available = server_has_model(MODEL)
    except RuntimeError as exc:
        print(f"Local AI server model mismatch: {exc}")
        print_server_instructions()
        return 2

    if not available:
        print_server_instructions()
        return 2

    print(f"Vaani decision evaluation — configured user: {USER_NAME}")
    print(f"Model: {MODEL}")
    print(f"Cases: {len(EXAMPLES)}")

    results = [
        run_example(index, example)
        for index, example in enumerate(EXAMPLES, start=1)
    ]
    correct = sum(passed for passed, _ in results)
    failures = [
        (index, example, error)
        for index, (example, (passed, error)) in enumerate(
            zip(EXAMPLES, results), start=1
        )
        if not passed
    ]
    accuracy = correct / len(EXAMPLES) * 100

    print("\n--- SUMMARY ---")
    print(f"Correct predictions: {correct}/{len(EXAMPLES)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Failures: {len(failures)}")
    for index, example, error in failures:
        detail = (
            f" — {type(error).__name__}: {error}"
            if error is not None
            else f" — expected {example.expected}"
        )
        print(f"  [{index:02d}] {example.sentence!r}: {detail}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
