"""Hardware-free smoke test: no microphone, no speakers, no LiveKit server.

The DeepSeek call (step 3) does need network access; STT/TTS run locally.

Checks, in order:
  1. the pipeline modules import, including agent.py (entrypoint API guard)
  2. the local VAD / turn-detector components load without LiveKit credentials
  3. one text turn goes through the same DeepSeek LLM config the demo uses
  4. one sentence is synthesized with the configured local TTS into a wav file
  5. (bonus) that wav is transcribed back with the local STT

    uv run smoke_test.py
"""

from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path

import numpy as np
from livekit import rtc

import agent  # noqa: F401  (import check: catches entrypoint/API regressions)
import config
import local_stt
import local_tts
import prompt  # noqa: F401  (import check)
from local_stt import WhisperSTT
from local_tts import KokoroTTS

from livekit.agents import inference, llm

OUT_DIR = Path(__file__).resolve().parent / "out"

LEARNER_LINE = "Yesterday I go to the cinema with my friends and we watch a very funny movie."
TTS_LINE = "That sounds like a fun evening. What was the movie about?"


def check_imports() -> None:
    print("\n[1] pipeline modules + entrypoint")
    entrypoint = getattr(agent, "entrypoint", None)
    if not asyncio.iscoroutinefunction(entrypoint):
        raise RuntimeError("agent.entrypoint is missing or is not an async entrypoint")
    print(
        f"    imported agent/agent.py, prompt.py, local_stt.py, local_tts.py; "
        f"entrypoint={entrypoint.__module__}.{entrypoint.__name__}()"
    )


def check_local_components() -> None:
    print("\n[2] local VAD / turn-detector components")
    vad = inference.VAD(model="silero")
    detector = inference.TurnDetector(version="v1-mini")
    print(f"    vad={vad.model}  turn_detector={detector.model}")


async def check_llm() -> None:
    print("\n[3] LLM turn (DeepSeek via the openai plugin, network)")
    model = config.build_llm()
    extra_body = getattr(getattr(model, "_opts", None), "extra_body", None)
    print(f"    model={config.DEEPSEEK_MODEL} base_url={config.DEEPSEEK_BASE_URL}")
    print(
        f"    thinking=disabled  max_completion_tokens={config.MAX_REPLY_TOKENS}"
        f"  (extra_body={extra_body})"
    )

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=prompt.TUTOR_PROMPT)
    chat_ctx.add_message(role="user", content=LEARNER_LINE)

    started = time.perf_counter()
    ttft = None
    reply = ""
    stream = model.chat(chat_ctx=chat_ctx)
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            if ttft is None:
                ttft = time.perf_counter() - started
            reply += chunk.delta.content
    await stream.aclose()
    elapsed = time.perf_counter() - started

    if not reply.strip():
        raise RuntimeError("LLM returned an empty reply")
    print(f"    learner: {LEARNER_LINE}")
    print(f"    tutor (first token {ttft:.2f}s, done {elapsed:.2f}s): {reply.strip()}")


async def check_tts() -> np.ndarray:
    print("\n[4] local TTS (Kokoro, ONNX)")
    tts = KokoroTTS(
        voice=config.KOKORO_VOICE, speed=config.KOKORO_SPEED, models_dir=config.MODELS_DIR
    )
    started = time.perf_counter()
    frame = await tts.synthesize(TTS_LINE).collect()
    elapsed = time.perf_counter() - started

    OUT_DIR.mkdir(exist_ok=True)
    wav_path = OUT_DIR / "smoke_tts.wav"
    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(frame.num_channels)
        f.setsampwidth(2)
        f.setframerate(frame.sample_rate)
        f.writeframes(bytes(frame.data))

    print(f"    voice={config.KOKORO_VOICE} text={TTS_LINE!r}")
    print(
        f"    wrote {wav_path} ({frame.duration:.2f}s of {frame.sample_rate} Hz audio "
        f"in {elapsed:.2f}s)"
    )
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


async def check_stt(samples: np.ndarray) -> None:
    print("\n[5] local STT (MLX Whisper) on the synthesized audio")
    stt_impl = WhisperSTT(model=config.WHISPER_MODEL)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    frame = rtc.AudioFrame(
        data=pcm16.tobytes(),
        sample_rate=local_tts.SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=len(pcm16),
    )

    started = time.perf_counter()
    event = await stt_impl.recognize([frame])
    elapsed = time.perf_counter() - started

    text = event.alternatives[0].text if event.alternatives else ""
    if not text.strip():
        raise RuntimeError("local STT returned an empty transcript")
    print(f"    transcript ({elapsed:.2f}s): {text!r}")


async def main() -> None:
    print("LiveKit Agents demo - hardware-free smoke test (no mic; DeepSeek needs network)")
    check_imports()
    check_local_components()
    await check_llm()
    samples = await check_tts()
    await check_stt(samples)
    print(
        "\nOK: imports + entrypoint, local VAD/turn detector, DeepSeek LLM "
        "(thinking disabled), Kokoro TTS and MLX Whisper STT all work."
    )


if __name__ == "__main__":
    asyncio.run(main())
