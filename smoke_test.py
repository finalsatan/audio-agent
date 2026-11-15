"""Headless checks for the Pipecat English-practice demo.

No microphone or speakers are used. This script:

1. imports the demo modules and builds the whole set of pipeline processors
   (Silero VAD, Smart Turn v3, Kokoro TTS, Whisper MLX STT, DeepSeek LLM),
   so a broken API/config shows up immediately;
2. sends one learner sentence through the *same* DeepSeek configuration the
   demo uses and prints the tutor's reply;
3. synthesizes that reply with the configured local Kokoro voice into
   `smoke_test_tts.wav`;
4. transcribes that wav again with the configured local MLX Whisper STT, as a
   round-trip check of the speech-to-text side.

Run with:  uv run smoke_test.py
"""

import asyncio
import sys
import wave

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

WAV_PATH = "smoke_test_tts.wav"
LEARNER_LINE = "Hi! Yesterday I go to the cinema with my friends, the film was very good."


class AudioSink(FrameProcessor):
    """Collects the TTS audio frames so they can be written to a wav file."""

    def __init__(self):
        super().__init__()
        self.chunks: list[bytes] = []
        self.sample_rate = 0
        self.num_channels = 1

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.chunks.append(frame.audio)
            self.sample_rate = frame.sample_rate
            self.num_channels = frame.num_channels
        await self.push_frame(frame, direction)


def check_imports_and_processors() -> dict:
    """Step 1: import the demo and build every processor (no audio devices)."""
    import bot
    import config

    print("imported bot.py and config.py")
    print("  LLM  : DeepSeek %s @ %s" % (config.DEEPSEEK_MODEL, config.DEEPSEEK_BASE_URL))
    print("  STT  : MLX Whisper %s" % config.STT_MODEL)
    print("  TTS  : Kokoro voice %r" % config.TTS_VOICE)
    print(
        "  turns: Silero VAD stop_secs=%.2f, Smart Turn stop_secs=%.2f"
        % (config.VAD_STOP_SECS, config.SMART_TURN_STOP_SECS)
    )
    print("  prompt: %d chars" % len(config.TUTOR_SYSTEM_PROMPT))

    context = LLMContext()
    parts = bot.create_pipeline_parts(context)
    print("built processors:", ", ".join(sorted(parts)))
    print("transport builder check:", bot.create_transport.__name__, "(not instantiated)")
    return parts


async def check_llm(llm) -> str:
    """Step 2: one text turn through the demo's DeepSeek configuration."""
    print("\nLLM turn (text only)")
    print("  learner: %s" % LEARNER_LINE)
    context = LLMContext(messages=[{"role": "user", "content": LEARNER_LINE}])
    reply = await llm.run_inference(context)
    if not reply:
        raise RuntimeError("DeepSeek returned an empty reply")
    print("  tutor  : %s" % reply.strip())
    return reply.strip()


async def check_tts(tts, text: str) -> str:
    """Step 3: synthesize one sentence with the configured local TTS."""
    print("\nTTS synthesis (local Kokoro)")
    print("  speaking: %s" % text)

    sink = AudioSink()
    worker = PipelineWorker(Pipeline([tts, sink]))
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await worker.queue_frames([TTSSpeakFrame(text), EndFrame()])
    await runner.run()

    if not sink.chunks or not sink.sample_rate:
        raise RuntimeError("Kokoro produced no audio frames")

    audio = np.frombuffer(b"".join(sink.chunks), dtype=np.int16)
    with wave.open(WAV_PATH, "wb") as wav:
        wav.setnchannels(sink.num_channels)
        wav.setsampwidth(2)
        wav.setframerate(sink.sample_rate)
        wav.writeframes(audio.tobytes())

    duration = len(audio) / (sink.sample_rate * sink.num_channels)
    print("  wrote %s: %.2f s at %d Hz" % (WAV_PATH, duration, sink.sample_rate))
    return WAV_PATH


async def check_stt(stt) -> str:
    """Step 4: transcribe the wav we just wrote with the configured local STT."""
    print("\nSTT transcription (local MLX Whisper)")

    with wave.open(WAV_PATH, "rb") as wav:
        sample_rate, frames = wav.getframerate(), wav.getnframes()
        audio = np.frombuffer(wav.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0

    # Whisper wants 16 kHz input. In the real pipeline the transport resamples;
    # here a simple linear resample is enough.
    num_out = int(len(audio) * 16000 / sample_rate)
    resampled = np.interp(
        np.linspace(0, len(audio) - 1, num_out), np.arange(len(audio)), audio
    ).astype(np.float32)
    pcm = np.clip(resampled * 32767, -32768, 32767).astype(np.int16).tobytes()

    transcript = ""
    async for frame in stt.run_stt(pcm):
        if isinstance(frame, TranscriptionFrame):
            transcript = frame.text.strip()
        elif isinstance(frame, ErrorFrame):
            raise RuntimeError("STT error: %s" % frame.error)

    if not transcript:
        raise RuntimeError("MLX Whisper returned no transcript")
    print("  heard  : %s" % transcript)
    return transcript


async def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    parts = check_imports_and_processors()
    reply = await check_llm(parts["llm"])
    await check_tts(parts["tts"], reply)
    await check_stt(parts["stt"])

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
