"""Use fast rules only when the intent is unambiguous."""
import logging
import re
import threading
import time
from collections import deque
from functools import lru_cache
from core.ai.classifier import classify_speech

logger = logging.getLogger(__name__)
GENERAL_REQUEST = re.compile(
    r"^(?:excuse me[,!.?]?\s*)?(?:can|could|would|will)\s+"
    r"(?:someone|somebody|anyone|anybody)\s+(?:please\s+)?"
    r"(?:help|hold|give|open|come|listen)\b", re.IGNORECASE,
)
GENERIC_GREETING = re.compile(
    r"^(?:hi|hello|hey)(?:[.!?,]?\s+(?:hi|hello|hey))*[.!?,]*$",
    re.IGNORECASE,
)
STANDALONE_EXCUSE = re.compile(r"^excuse\s+me[.!?,]*$", re.IGNORECASE)
STANDALONE_EMERGENCY = re.compile(
    r"^[.!?,]*(?:help(?:\s+me)?|watch\s+out|call\s+an\s+ambulance|call\s+the\s+police)[.!?,]*$",
    re.IGNORECASE,
)
REPETITION_WINDOW_SECONDS = 15.0
REPETITION_COOLDOWN_SECONDS = 5.0


@lru_cache(maxsize=128)
def name_pattern(user_name):
    parts = user_name.strip().split()
    return re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, parts)) + r"(?!\w)", re.IGNORECASE)


def count_name(transcript, user_name):
    if not user_name.strip():
        return 0
    return len(name_pattern(user_name).findall(transcript))


def contains_name(transcript, user_name):
    return count_name(transcript, user_name) > 0


def _immediate_decision(transcript, user_name):
    """Return a hard-rule result that is independent of repetition state."""
    if not isinstance(user_name, str) or not user_name.strip():
        raise ValueError("A configured user name is required.")
    transcript = transcript.strip()
    if not transcript:
        return "IGNORE"
    if contains_name(transcript, user_name):
        return "FULL_ALERT"
    normalized = " ".join(transcript.casefold().split()).strip(".!?, ")
    if STANDALONE_EMERGENCY.fullmatch(transcript):
        return "FULL_ALERT"
    if normalized in {"can you hear me", "could you come here", "can you help me", "could you help me"}:
        return "SOFT_ALERT"
    if GENERAL_REQUEST.match(transcript):
        return "SOFT_ALERT"
    return None


def _classify_ambiguous(transcript, user_name, loudness, speaking_rate, repetition_count):
    # Negations, quotations, and recipient identity require context.
    try:
        return classify_speech(
            transcript=transcript, user_name=user_name, loudness=loudness,
            speaking_rate=speaking_rate, repetition_count=repetition_count,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("AI classification failed; using SOFT_ALERT: %s", exc)
        return "SOFT_ALERT"


def decide_alert(transcript, user_name, loudness=0.0, speaking_rate=0.0, repetition_count=0):
    """Stateless decision entry point retained for one-off callers and evaluations."""
    immediate = _immediate_decision(transcript, user_name)
    if immediate is not None:
        return immediate
    if STANDALONE_EXCUSE.fullmatch(transcript.strip()):
        return "SOFT_ALERT"
    return _classify_ambiguous(
        transcript, user_name, loudness, speaking_rate, repetition_count
    )


class AttentionDecisionService:
    """Thread-safe repetition state scoped to one microphone listening session."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._greetings = deque()
        self._excuse_me = deque()
        self._cooldowns = {}

    def reset(self):
        """Clear all repetition and cooldown state at a session boundary."""
        with self._lock:
            self._greetings.clear()
            self._excuse_me.clear()
            self._cooldowns.clear()

    def decide_alert(
        self, transcript, user_name, loudness=0.0, speaking_rate=0.0,
        repetition_count=0,
    ):
        immediate = _immediate_decision(transcript, user_name)
        if immediate is not None:
            return immediate

        normalized = transcript.strip()
        is_greeting = bool(GENERIC_GREETING.fullmatch(normalized))
        is_excuse = bool(STANDALONE_EXCUSE.fullmatch(normalized))
        if is_greeting or is_excuse:
            key = "greetings" if is_greeting else "excuse_me"
            with self._lock:
                now = self._clock()
                window = self._greetings if is_greeting else self._excuse_me
                cooldown_until = self._cooldowns.get(key, 0.0)
                if now < cooldown_until:
                    return "IGNORE"

                cutoff = now - REPETITION_WINDOW_SECONDS
                while window and window[0] < cutoff:
                    window.popleft()
                window.append(now)

                if is_greeting:
                    if len(window) < 3:
                        return "IGNORE"
                    window.clear()
                    self._cooldowns[key] = now + REPETITION_COOLDOWN_SECONDS
                    return "SOFT_ALERT"

                if len(window) < 2:
                    return "SOFT_ALERT"
                window.clear()
                self._cooldowns[key] = now + REPETITION_COOLDOWN_SECONDS
                return "FULL_ALERT"

        return _classify_ambiguous(
            transcript, user_name, loudness, speaking_rate, repetition_count
        )
