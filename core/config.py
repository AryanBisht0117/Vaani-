"""Shared runtime settings for the launcher and speech pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_CONFIG = Path(os.environ.get("VAANI_SERVER_CONFIG", ROOT / "llama-server-1.7b.json"))
API_URL = os.environ.get("VAANI_API_URL", "http://127.0.0.1:8011/v1/chat/completions")
MODEL = os.environ.get("VAANI_AI_MODEL", "models/Qwen3-1.7B-Q4_K_M.gguf")
WHISPER_MODEL = os.environ.get("VAANI_WHISPER_MODEL", "tiny")
BEAM_SIZE = int(os.environ.get("VAANI_BEAM_SIZE", "5"))
CPU_THREADS = int(os.environ.get("VAANI_CPU_THREADS", "4"))
device = os.environ.get("VAANI_MICROPHONE_DEVICE", "").strip()
MICROPHONE_DEVICE = int(device) if device.isdecimal() else device or None
