import json
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
from core.ai.classifier import classify_speech
from core.ai.decision_v2 import AttentionDecisionService, decide_alert, count_name
from core.audio.continuous_listener import enqueue_latest, listen_continuously, CHUNK_SIZE
from core.features.feature_combiner import extract_features
from start_vaani import server_has_model


class DecisionTests(unittest.TestCase):
    def test_clear_calls_skip_ai(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            for text, expected in [("Aryan, Aryan!", "FULL_ALERT"), ("Help me!", "FULL_ALERT"), ("Can you help me?", "SOFT_ALERT"), ("Aryan, please come here.", "FULL_ALERT"), ("Aryan, could you come here?", "FULL_ALERT"), ("Hey Aryan, are you free for a second?", "FULL_ALERT"), ("Would someone please hold this door for me?", "SOFT_ALERT"), ("", "IGNORE")]:
                self.assertEqual(decide_alert(text, "Aryan"), expected)
            ai.assert_not_called()

    def test_greeting_directly_addresses_configured_user(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            for text in ["Hello, Aryan.", "Hi Aryan!", "Hey, Aryan.", "Good morning, Aryan."]:
                self.assertEqual(decide_alert(text, "Aryan"), "FULL_ALERT")
            self.assertEqual(decide_alert("Hello, Alex.", "Alex"), "FULL_ALERT")
            ai.assert_not_called()

    def test_context_is_not_overridden_by_keywords(self):
        with patch("core.ai.decision_v2.classify_speech", return_value="IGNORE") as ai:
            for text in ["Aryan went shopping.", "Aryan said he would join us next week.", "Aryan, my friend, went shopping.", "I said hello to Aryan yesterday.", "Hello, Sam.", '"Hello, Aryan."', '"Hello, Aryan," she said.', "The fire drill is tomorrow.", "Don't call the police.", "Can you help me, Priya?", "Sam, could you come here?", "Could someone have helped yesterday?"]:
                expected = "FULL_ALERT" if "Aryan" in text else "IGNORE"
                self.assertEqual(decide_alert(text, "Aryan"), expected)
            self.assertEqual(ai.call_count, 6)

    def test_names_have_boundaries_and_support_multiple_words(self):
        self.assertEqual(count_name("Aryana met ARYAN, Aryan!", "Aryan"), 2)
        self.assertEqual(count_name("Hello, Aryan   Kumar!", "Aryan Kumar"), 1)
        self.assertEqual(count_name("anything", ""), 0)

    def test_failure_is_visible_and_uses_fallback(self):
        with patch("core.ai.decision_v2.classify_speech", side_effect=RuntimeError("offline")):
            with self.assertLogs("core.ai.decision_v2", level="WARNING"):
                self.assertEqual(decide_alert("Please listen.", "Aryan"), "SOFT_ALERT")

    def test_missing_name_is_configuration_error(self):
        with self.assertRaises(ValueError):
            decide_alert("hello", " ")


class AttentionDecisionServiceTests(unittest.TestCase):
    class Clock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    def setUp(self):
        self.clock = self.Clock()
        self.service = AttentionDecisionService(clock=self.clock)

    def decide(self, transcript, user_name="Michael"):
        return self.service.decide_alert(transcript, user_name)

    def test_generic_greeting_threshold_is_three_speech_events(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            self.assertEqual(self.decide("Hi"), "IGNORE")
            self.assertEqual(self.decide("Hello."), "IGNORE")
            self.assertEqual(self.decide("Hey!"), "SOFT_ALERT")
            ai.assert_not_called()

    def test_greeting_words_in_one_event_count_once(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            self.assertEqual(self.decide("Hi! Hello! Hey!"), "IGNORE")
            self.assertEqual(self.decide("Hello"), "IGNORE")
            self.assertEqual(self.decide("Hey"), "SOFT_ALERT")
            ai.assert_not_called()

    def test_greeting_events_outside_window_do_not_accumulate(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            self.assertEqual(self.decide("Hi"), "IGNORE")
            self.clock.advance(15.1)
            self.assertEqual(self.decide("Hello"), "IGNORE")
            self.assertEqual(self.decide("Hey"), "IGNORE")
            self.assertEqual(self.decide("Hi"), "SOFT_ALERT")
            ai.assert_not_called()

    def test_greeting_cooldown_and_reset(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            self.assertEqual(self.decide("Hi"), "IGNORE")
            self.assertEqual(self.decide("Hi"), "IGNORE")
            self.assertEqual(self.decide("Hi"), "SOFT_ALERT")
            self.assertEqual(self.decide("Hello"), "IGNORE")
            self.clock.advance(5.1)
            self.assertEqual(self.decide("Hello"), "IGNORE")
            self.assertEqual(self.decide("Hey"), "IGNORE")
            self.assertEqual(self.decide("Hi"), "SOFT_ALERT")
            self.service.reset()
            self.assertEqual(self.decide("Hey"), "IGNORE")
            ai.assert_not_called()

    def test_repetition_cooldown_does_not_suppress_emergency_or_name(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            self.assertEqual(self.decide("Hi"), "IGNORE")
            self.assertEqual(self.decide("Hello"), "IGNORE")
            self.assertEqual(self.decide("Hey"), "SOFT_ALERT")
            self.assertEqual(self.decide("Hello"), "IGNORE")
            self.assertEqual(self.decide("Help!"), "FULL_ALERT")
            self.assertEqual(self.decide("Michael!"), "FULL_ALERT")
            ai.assert_not_called()

    def test_excuse_me_escalates_on_second_event(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            self.assertEqual(self.decide("Excuse me."), "SOFT_ALERT")
            self.assertEqual(self.decide("EXCUSE ME!"), "FULL_ALERT")
            ai.assert_not_called()

    def test_excuse_me_outside_window_starts_over(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            self.assertEqual(self.decide("Excuse me"), "SOFT_ALERT")
            self.clock.advance(15.1)
            self.assertEqual(self.decide("Excuse me!"), "SOFT_ALERT")
            self.assertEqual(self.decide("Excuse me?"), "FULL_ALERT")
            ai.assert_not_called()

    def test_username_address_and_greeting_remain_immediate(self):
        with patch("core.ai.decision_v2.classify_speech") as ai:
            for transcript in (
                "Michael!",
                "Hello Michael.",
                "Hi, Michael!",
                "Michael, can you help me?",
            ):
                self.assertEqual(self.decide(transcript), "FULL_ALERT")
            self.assertEqual(self.decide("Hi"), "IGNORE")
            ai.assert_not_called()

    def test_emergencies_are_immediate_only_as_standalone_calls(self):
        with patch("core.ai.decision_v2.classify_speech", return_value="IGNORE") as ai:
            for transcript in (
                "Help!",
                "Help me!",
                "Watch out!",
                "Call an ambulance!",
                "Call the police!",
            ):
                self.assertEqual(self.decide(transcript), "FULL_ALERT")
            for transcript in (
                '"Help!" she said.',
                "In the play, call the police!",
                "Imagine shouting help me.",
            ):
                self.assertEqual(self.decide(transcript), "IGNORE")
            self.assertEqual(ai.call_count, 3)

    def test_any_configured_username_mention_is_immediate_but_other_names_use_qwen(self):
        with patch("core.ai.decision_v2.classify_speech", return_value="IGNORE") as ai:
            self.assertEqual(self.decide("I met Michael yesterday."), "FULL_ALERT")
            self.assertEqual(self.decide("Hello Sam."), "IGNORE")
            ai.assert_called_once()

    def test_ambiguous_speech_still_uses_qwen(self):
        with patch("core.ai.decision_v2.classify_speech", return_value="SOFT_ALERT") as ai:
            self.assertEqual(self.decide("I heard someone ask for help yesterday."), "SOFT_ALERT")
            ai.assert_called_once()


class ClassifierTests(unittest.TestCase):
    def test_existing_server_requires_matching_model(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"data": [{"id": "expected"}]}).encode()
        with patch("start_vaani.urllib.request.urlopen", return_value=response):
            self.assertTrue(server_has_model("expected"))
            with self.assertRaises(RuntimeError):
                server_has_model("different")

    def test_request_and_response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": "IGNORE"}}]}).encode()
        with patch("core.ai.classifier.urllib.request.urlopen", return_value=response) as request:
            self.assertEqual(classify_speech("hello", 0, 0, 0, "Aryan"), "IGNORE")
            payload = json.loads(request.call_args.args[0].data)
            self.assertIn("1.7B", payload["model"])
            self.assertEqual(payload["temperature"], 0)

    def test_malformed_response(self):
        for result in [{}, {"choices": []}, {"choices": [{"message": {"content": None}}]}]:
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(result).encode()
            with patch("core.ai.classifier.urllib.request.urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    classify_speech("hello", 0, 0, 0, "Aryan")


class AudioTests(unittest.TestCase):
    def test_queue_keeps_latest_and_tracks_tasks(self):
        clips = queue.Queue(maxsize=2)
        for item in [1, 2, 3]:
            enqueue_latest(clips, item)
        self.assertEqual(clips.get_nowait(), 2)
        clips.task_done()
        self.assertEqual(clips.get_nowait(), 3)
        clips.task_done()
        self.assertEqual(clips.unfinished_tasks, 0)

    def test_features_use_word_times_and_multiword_name(self):
        words = [SimpleNamespace(start=0.5, end=1), SimpleNamespace(start=1, end=1.5)]
        model = MagicMock()
        model.transcribe.return_value = (iter([SimpleNamespace(text=" Aryan Kumar! ", words=words)]), None)
        features = extract_features(np.full(16000, 0.25, dtype=np.float32), model, "Aryan Kumar")
        self.assertEqual(features.speaking_rate, 120)
        self.assertEqual(features.repetition_count, 1)
        self.assertEqual(features.loudness, 0.25)

    def test_empty_and_invalid_audio(self):
        model = MagicMock()
        self.assertEqual(extract_features([], model, "Aryan").transcript, "")
        model.transcribe.assert_not_called()
        for audio in [np.zeros((2, 2)), [float("nan")]]:
            with self.assertRaises(ValueError):
                extract_features(audio, model, "Aryan")

    def test_long_speech_is_split_without_resetting_vad(self):
        stop = MagicMock()
        stop.is_set.side_effect = [False] * 4 + [True]
        vad = MagicMock(side_effect=[{"start": 0}, None, None, {"end": 2048}])
        stream = MagicMock()
        stream.__enter__.return_value.read.return_value = (np.zeros((CHUNK_SIZE, 1), dtype=np.float32), False)
        clips = queue.Queue()
        with patch("core.audio.continuous_listener.load_silero_vad"), patch("core.audio.continuous_listener.VADIterator", return_value=vad), patch("core.audio.continuous_listener.sd.InputStream", return_value=stream), patch("core.audio.continuous_listener.MAX_RECORDING_SECONDS", 0.06):
            listen_continuously(clips, stop)
        self.assertGreaterEqual(clips.qsize(), 2)
        vad.reset_states.assert_called_once()


if __name__ == "__main__":
    unittest.main()
