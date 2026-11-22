"""Minimal local voice agent for English speaking practice, built on Pipecat.

Pipeline:
    LocalAudioTransport (mic) -> Whisper MLX STT -> user context aggregator
    (Silero VAD + Smart Turn v3) -> DeepSeek LLM -> Kokoro TTS
    -> LocalAudioTransport (speakers) -> assistant context aggregator

Everything except the LLM runs locally on the Mac. Barge-in is on: start
speaking and the bot stops talking.

Run with:  uv run bot.py
"""

import asyncio
import os
import sys

from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from config import (
    AUDIO_INPUT_DEVICE,
    AUDIO_OUTPUT_DEVICE,
    DEEPSEEK_MODEL,
    GREETING_INSTRUCTION,
    SAVE_TRANSCRIPT,
    SMART_TURN_STOP_SECS,
    STT_MODEL,
    TRANSCRIPTS_DIR,
    TTS_VOICE,
    VAD_CONFIDENCE,
    VAD_STOP_SECS,
    create_llm,
    create_stt,
    create_tts,
    create_user_aggregator_params,
)
from session_recorder import SessionRecorder


def create_transport() -> LocalAudioTransport:
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=AUDIO_INPUT_DEVICE,
            output_device_index=AUDIO_OUTPUT_DEVICE,
        )
    )


def create_pipeline_parts(context: LLMContext) -> dict:
    """Build the non-transport processors and the context aggregator pair."""
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=create_user_aggregator_params(),
    )
    return {
        "stt": create_stt(),
        "llm": create_llm(),
        "tts": create_tts(),
        "user_aggregator": user_aggregator,
        "assistant_aggregator": assistant_aggregator,
    }


async def main():
    logger.remove(0)
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

    transport = create_transport()
    context = LLMContext()
    parts = create_pipeline_parts(context)

    pipeline = Pipeline(
        [
            transport.input(),  # microphone
            parts["stt"],  # local Whisper (MLX)
            parts["user_aggregator"],  # VAD + smart turn -> user turns in context
            parts["llm"],  # DeepSeek
            parts["tts"],  # local Kokoro
            transport.output(),  # speakers
            parts["assistant_aggregator"],  # assistant turns in context
        ]
    )

    recorder = SessionRecorder(
        transcripts_dir=TRANSCRIPTS_DIR if SAVE_TRANSCRIPT else None,
        metadata={
            "LLM": f"{DEEPSEEK_MODEL} (thinking=disabled)",
            "STT": STT_MODEL,
            "TTS": f"kokoro ({TTS_VOICE})",
            "VAD": f"stop_secs={VAD_STOP_SECS}, confidence={VAD_CONFIDENCE}, smart_turn={SMART_TURN_STOP_SECS}",
        },
    )
    recorder.attach(parts["user_aggregator"], parts["assistant_aggregator"])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[recorder],
        # A learner who thinks for a while should not have the session killed.
        idle_timeout_secs=None,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner()
    await runner.add_workers(worker)

    print(
        f"English Tutor ready | STT={STT_MODEL} | LLM={DEEPSEEK_MODEL} | TTS=kokoro/{TTS_VOICE}",
        flush=True,
    )
    if SAVE_TRANSCRIPT:
        print(f"Transcripts will be saved under: {TRANSCRIPTS_DIR}", flush=True)

    # Open the conversation so the learner hears a voice right away.
    context.add_message({"role": "developer", "content": GREETING_INSTRUCTION})
    await worker.queue_frames([LLMRunFrame()])

    try:
        await runner.run()
    finally:
        if recorder.file_path is not None:
            print(f"\nSession transcript saved to: {recorder.file_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
