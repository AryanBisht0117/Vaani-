import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from start_vaani import launch_server, wait_for_server, SERVER_URL


class StartupTests(unittest.TestCase):
    def test_readiness_uses_models_endpoint(self):
        self.assertTrue(SERVER_URL.endswith("/v1/models"))
        self.assertNotIn("/chat/models", SERVER_URL)

    def test_server_binds_to_classifier_endpoint_and_captures_errors(self):
        log = io.StringIO()
        with patch("start_vaani.API_URL", "http://127.0.0.1:8123/v1/chat/completions"), patch.dict(
            "os.environ", {"HOST": "localhost", "PORT": "8000", "CONFIG_FILE": "other.json"}
        ), patch("start_vaani.subprocess.Popen") as start:
            launch_server(log)
        settings = start.call_args.kwargs
        self.assertEqual(settings["env"]["HOST"], "127.0.0.1")
        self.assertEqual(settings["env"]["PORT"], "8123")
        self.assertNotIn("CONFIG_FILE", settings["env"])
        self.assertIs(settings["stdout"], log)
        self.assertIsNotNone(settings["stderr"])

    def test_startup_failure_includes_server_error(self):
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 1
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("ERROR: address already in use", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "address already in use"):
                wait_for_server(process, "model", log_path=log)


if __name__ == "__main__":
    unittest.main()
