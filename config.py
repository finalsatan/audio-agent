"""Shared configuration: .env loading, LLM factory, tunable knobs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Imported at module level: LiveKit plugins must be registered on the main thread,
# before the worker forks the job process that runs the entrypoint.
from livekit.plugins import openai

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- DeepSeek (OpenAI-compatible) LLM -------------------------------------
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

# --- Local STT / TTS ------------------------------------------------------
# MLX Whisper model on the Hugging Face Hub (Apple Silicon friendly).
WHISPER_MODEL = os.getenv("TUTOR_WHISPER_MODEL", "mlx-community/whisper-small.en-mlx")
# Kokoro voice + speed (see README for the list of voices).
KOKORO_VOICE = os.getenv("TUTOR_VOICE", "af_heart")
KOKORO_SPEED = float(os.getenv("TUTOR_SPEED", "1.0"))
MODELS_DIR = Path(os.getenv("TUTOR_MODELS_DIR", str(ROOT / "models")))

# --- Turn taking ----------------------------------------------------------
# How long the learner may stay silent before we consider their turn finished.
SILENCE_SECONDS = float(os.getenv("TUTOR_SILENCE_SECONDS", "1.2"))
# Hard upper bound on how long we wait for the learner to finish a turn.
MAX_SILENCE_SECONDS = float(os.getenv("TUTOR_MAX_SILENCE_SECONDS", "4.0"))
# Silence inside an utterance before the VAD closes the speech segment.
VAD_SILENCE_SECONDS = float(os.getenv("TUTOR_VAD_SILENCE_SECONDS", "0.45"))


def build_llm():
    """The LLM used by both the demo and the smoke test."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing; put it in .env (see .env.example)")

    return openai.LLM(
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    )
