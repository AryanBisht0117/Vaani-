# Vaani

Vaani listens to nearby English speech and decides whether to notify a headphone user.
Audio and classification run locally.

Run from the project directory using the existing virtual environment:

```powershell
.venv\Scripts\python.exe start_vaani.py
```

Enter your name at startup. The launcher checks the configured model, waits for the
AI server, and passes the correct model ID to the continuous speech app. Whisper
loads before the microphone starts. Vaani binds to `127.0.0.1:8011` by default.
Server diagnostics are saved to `vaani-server.log`; startup failures show the log tail.

To launch the Tkinter desktop interface instead:

```powershell
.\.venv\Scripts\python.exe .\start_vaani_gui.py
```

Enter your name in the window and select **Start Listening**. The GUI uses the same
local speech pipeline and Windows notifications as the terminal launcher. It is
intentionally one-shot: after stopping, close and relaunch the GUI to listen again.

Settings can be supplied as environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VAANI_SERVER_CONFIG` | `llama-server-1.7b.json` | Local AI server configuration |
| `VAANI_API_URL` | `http://127.0.0.1:8011/v1/chat/completions` | Server endpoint; the launcher binds to this host and port |
| `VAANI_WHISPER_MODEL` | `tiny` | Transcription model; larger models require separate evaluation |
| `VAANI_BEAM_SIZE` | `5` | Transcription search width |
| `VAANI_CPU_THREADS` | `4` | Whisper CPU thread count |
| `VAANI_BUILTIN_MICROPHONE` | Automatically detected | Exact built-in input device name; use this when detection is ambiguous |
| `VAANI_USER_NAME` | Required | Name when launching the app directly |

`app_v2.py` and `app_v2_continuous.py` use the same in-memory pipeline.
`app_v2_backup.py` preserves the previous file-based experiment.
When processing falls behind, the bounded queue discards the oldest waiting clip.
Clear standalone calls use fast rules; contextual speech uses the AI. If the AI
fails, the app logs the failure and uses `SOFT_ALERT`.

Checks, run from the project directory:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_pipeline.py -v
.venv\Scripts\python.exe -m tests.benchmark_pipeline
.venv\Scripts\python.exe -m tests.check_local_ai
```

The benchmark uses cached Whisper Tiny and local sentence recordings and writes
per-recording results to `tests/benchmark_results.json`. The AI check starts and
stops its own server on port 8011. Its results are written to
`tests/ai_check_results.json`. These small datasets do not establish general
recognition or alert accuracy.
