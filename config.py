"""Shared configuration for the Pipecat English-practice voice agent.

Everything tunable lives here: the tutor prompt, model/voice choices and the
end-of-speech silence thresholds. `bot.py` (the demo) and `smoke_test.py` both
build their services from the factories below, so the smoke test exercises the
same configuration the demo runs with.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

load_dotenv()

# --- Tutor prompt ---------------------------------------------------------

TUTOR_SYSTEM_PROMPT = """
You are Sam, a friendly English conversation partner. The learner is practising
their spoken English with you.

This is a spoken conversation: everything you write is read aloud by a
text-to-speech voice, so write only plain spoken English.
- Reply in 1 to 3 short sentences (roughly 40 words at most).
- Never use markdown, emoji, bullet points, lists, headings or code.
- Always end your turn with one short follow-up question, so the learner keeps
  talking.
- If the learner makes a clear grammar or word-choice mistake, model the correct
  phrasing once, briefly and naturally (for example: "Ah, you went to the beach
  yesterday."), and then carry on. Do not lecture, explain rules or list
  corrections.
- If the learner asks for a correction or an explanation, give it in one short
  sentence, then ask your follow-up question.
"""

# Spoken once at start-up so the learner knows the mic is live.
GREETING_INSTRUCTION = (
    "Greet the learner with one short sentence and ask how their day is going."
)

# --- Settings (override any of these from the environment / .env) ---------

# P99 latency from end of speech to final transcript. Local MLX Whisper takes
# roughly 0.4 s for a 5 s utterance (warm), so this is a generous value; the
# turn-stop strategy uses it to bound how long it waits for the transcript.
STT_TTFS_P99_SECS = float(os.environ.get("STT_TTFS_P99_SECS", "1.5"))

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

# Local STT: MLX Whisper on Apple Silicon. Alternatives: whisper-tiny (fastest),
# mlx-community/distil-whisper-large-v3 (English only, fast).
STT_MODEL = os.environ.get("STT_MODEL", MLXModel.LARGE_V3_TURBO.value)

# Local TTS: Kokoro (kokoro-onnx). "af_heart" is a warm US English voice.
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))

# Learners pause a lot, so both silence thresholds are deliberately generous:
# VAD_STOP_SECS is how long Silero waits before it calls "speech ended", and
# SMART_TURN_STOP_SECS is the hard cap the turn detector waits before it gives
# up on a complete-sounding sentence.
VAD_STOP_SECS = float(os.environ.get("VAD_STOP_SECS", "1.2"))
SMART_TURN_STOP_SECS = float(os.environ.get("SMART_TURN_STOP_SECS", "1.5"))
VAD_CONFIDENCE = float(os.environ.get("VAD_CONFIDENCE", "0.6"))

# Optional PyAudio device indexes (None = system default). Run
# `.venv/bin/python -c "import pyaudio; ..."` or just use the default devices.
AUDIO_INPUT_DEVICE = (
    int(os.environ["AUDIO_INPUT_DEVICE"]) if os.environ.get("AUDIO_INPUT_DEVICE") else None
)
AUDIO_OUTPUT_DEVICE = (
    int(os.environ["AUDIO_OUTPUT_DEVICE"]) if os.environ.get("AUDIO_OUTPUT_DEVICE") else None
)

# Save each live practice session as a Markdown transcript under TRANSCRIPTS_DIR.
SAVE_TRANSCRIPT = os.environ.get("SAVE_TRANSCRIPT", "true").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
TRANSCRIPTS_DIR = Path(
    os.environ.get("TRANSCRIPTS_DIR", str(Path(__file__).resolve().parent / "transcripts"))
)


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


# --- Service factories ----------------------------------------------------


def create_llm() -> DeepSeekLLMService:
    """DeepSeek via pipecat's own DeepSeek service (OpenAI-compatible).

    Explicitly locks DeepSeek thinking mode to ``disabled`` so ``deepseek-flash``
    never spends extra latency on hidden reasoning tokens before speaking.
    """
    return DeepSeekLLMService(
        api_key=_require("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
        settings=DeepSeekLLMService.Settings(
            model=DEEPSEEK_MODEL,
            system_instruction=TUTOR_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=220,
            thinking=DeepSeekLLMService.ThinkingConfig(type="disabled"),
        ),
    )


def create_stt() -> WhisperSTTServiceMLX:
    """Local speech-to-text with MLX Whisper (Apple Silicon GPU)."""
    return WhisperSTTServiceMLX(
        ttfs_p99_latency=STT_TTFS_P99_SECS,
        settings=WhisperSTTServiceMLX.Settings(
            model=STT_MODEL,
            language=Language.EN,
            no_speech_prob=0.6,
            temperature=0.0,
        )
    )


def create_tts() -> KokoroTTSService:
    """Local text-to-speech with Kokoro (kokoro-onnx, 24 kHz)."""
    return KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=TTS_VOICE,
            language=Language.EN_US,
            speed=TTS_SPEED,
        )
    )


def create_user_aggregator_params() -> LLMUserAggregatorParams:
    """Silero VAD + Smart Turn v3 end-of-turn detection (barge-in enabled)."""
    turn_analyzer = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(stop_secs=SMART_TURN_STOP_SECS)
    )
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(stop_secs=VAD_STOP_SECS, confidence=VAD_CONFIDENCE)
        ),
        user_turn_strategies=UserTurnStrategies(
            # Start strategies default to VAD + transcript start, which is what
            # interrupts (barge-in) the bot while it is speaking.
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)],
        ),
    )
