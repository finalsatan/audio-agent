"""Minimal English speaking-practice voice agent built with LiveKit Agents 1.x.

Runs fully on the local machine in console mode (Mac mic + speakers), with local
STT (MLX Whisper) and local TTS (Kokoro) and DeepSeek as the LLM. No LiveKit
server or LiveKit Cloud account is required.

    uv run agent.py console
"""

from __future__ import annotations

import asyncio

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, inference, stt

import config
from local_stt import WhisperSTT
from local_tts import KokoroTTS
from prompt import TUTOR_PROMPT

class TutorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=TUTOR_PROMPT)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # Silero VAD ships inside `livekit-local-inference` (no download, no server).
    vad = inference.VAD(
        model="silero",
        min_silence_duration=config.VAD_SILENCE_SECONDS,
    )

    # Model loading (Kokoro ONNX + MLX Whisper weights) is blocking, so keep it
    # off the event loop; both constructors are pure Python/MLX work.
    whisper = await asyncio.to_thread(WhisperSTT, model=config.WHISPER_MODEL)
    kokoro = await asyncio.to_thread(
        KokoroTTS,
        voice=config.KOKORO_VOICE,
        speed=config.KOKORO_SPEED,
        models_dir=config.MODELS_DIR,
    )

    session = AgentSession(
        # Non-streaming local Whisper, driven by VAD through the stream adapter.
        stt=stt.StreamAdapter(stt=whisper, vad=vad),
        vad=vad,
        llm=config.build_llm(),
        tts=kokoro,
        turn_handling={
            # Local end-of-turn model ("v1-mini" runs on-device, no credentials).
            "turn_detection": inference.TurnDetector(version="v1-mini"),
            "endpointing": {
                "mode": "fixed",
                # Learners pause a lot: wait generously before taking the turn.
                "min_delay": config.SILENCE_SECONDS,
                "max_delay": config.MAX_SILENCE_SECONDS,
            },
            # VAD-based interruptions: the ML variant needs LiveKit Cloud creds.
            "interruption": {"mode": "vad"},
        },
    )

    await session.start(agent=TutorAgent(), room=ctx.room)

    await session.generate_reply(
        instructions=(
            "Greet the learner briefly, introduce yourself as their English "
            "conversation partner, and ask one friendly question to start."
        )
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
