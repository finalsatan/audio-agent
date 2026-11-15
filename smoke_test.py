#!/usr/bin/env python3
"""Hardware-free smoke test for the English tutor configuration.

It does four things:
  1. imports the same speech-to-speech pipeline modules the demo uses;
  2. sends one text turn through the configured DeepSeek Chat Completions
     backend and prints the reply;
  3. synthesizes one sentence with the configured local Qwen3-TTS model and
     writes smoke_tts.wav (16 kHz mono PCM16);
  4. transcribes that WAV with the configured local Parakeet TDT handler and
     asserts the transcript is non-empty.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Event

ROOT = Path(__file__).resolve().parent
PIPELINE_NAME = "english-tutor"
WAV_PATH = ROOT / "smoke_tts.wav"
SAMPLE_RATE = 16000


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE lines without overriding the caller's environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, separator, value = line.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_env(ROOT / ".env")
    os.environ.setdefault("TUTOR_PROMPT", (ROOT / "tutor_prompt.txt").read_text(encoding="utf-8").strip())

    # NLTK resources must be visible before the pipeline modules import nltk.
    import bootstrap_nltk

    bootstrap_nltk.ensure()

    missing = [name for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL") if not os.environ.get(name)]
    if missing:
        print(f"FAIL: missing environment values: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("[1/4] Importing pipeline modules ...")
    import speech_to_speech.s2s_pipeline  # noqa: F401
    from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler  # noqa: F401
    from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
    from speech_to_speech.VAD.vad_handler import VADHandler  # noqa: F401
    from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler

    from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
    from speech_to_speech.config import load_config, resolve_config, validate_config
    from speech_to_speech.LLM.chat import Chat, make_user_message
    from speech_to_speech.pipeline.messages import (
        EndOfResponse,
        GenerateResponseRequest,
        LLMResponseChunk,
        TTSInput,
        Transcription,
        VADAudio,
    )
    import numpy as np
    import soundfile as sf

    print("      OK: VAD, Parakeet STT, Chat Completions LLM, Qwen3 TTS")

    document = load_config(ROOT / "config.yaml")
    validate_config(document)
    resolved = resolve_config(document, names=[PIPELINE_NAME], include_client=True)
    stages = resolved.pipelines[PIPELINE_NAME].stages
    llm_config = dict(stages["llm"].selection.config)
    tts_config = dict(stages["tts"].selection.config)

    print(f"[2/4] One text turn through {llm_config['model_name']} @ {llm_config['base_url']} ...")
    llm = ChatCompletionsApiModelHandler(Event(), Queue(), Queue(), setup_kwargs=llm_config)
    runtime = RuntimeConfig(chat=Chat(int(llm_config.get("chat_size", 10))))
    runtime.session.instructions = llm_config["init_chat_prompt"]
    runtime.chat.add_item(
        make_user_message("Hi! I want to practise English. Yesterday I go to the park with my friend.")
    )
    request = GenerateResponseRequest(
        runtime_config=runtime,
        language_code="en",
        turn_id="smoke-test",
        turn_revision=0,
    )

    reply_parts: list[str] = []
    finished = False
    for output in llm.process(request):
        if isinstance(output, LLMResponseChunk):
            text = "".join(part.text for part in output.parts if part.type == "text") or output.text
            if text:
                reply_parts.append(text)
        elif isinstance(output, EndOfResponse):
            finished = True
            if output.error:
                print(f"FAIL: LLM response ended with error: {output.error}", file=sys.stderr)
                return 1

    reply = " ".join(part.strip() for part in reply_parts if part.strip()).strip()
    if not finished or not reply:
        print("FAIL: the LLM returned no text", file=sys.stderr)
        return 1
    print(f"      Tutor reply: {reply}")
    print(f"      follow-up question: {'yes' if '?' in reply else 'no'}")

    print(f"[3/4] Synthesizing one sentence with {tts_config['model_name']} (speaker={tts_config['speaker']}) ...")
    tts_config["should_listen"] = Event()
    tts = Qwen3TTSHandler(Event(), Queue(), Queue(), setup_kwargs=tts_config)
    audio_parts: list[np.ndarray] = []
    started = time.perf_counter()
    for output in tts.process(TTSInput(text="Hello! How was your day today?", language_code="en")):
        if isinstance(output, np.ndarray):
            audio_parts.append(output.astype(np.int16, copy=False))
        elif isinstance(output, bytes):
            audio_parts.append(np.frombuffer(output, dtype=np.int16))

    if not audio_parts:
        print("FAIL: TTS produced no audio", file=sys.stderr)
        return 1
    audio = np.concatenate(audio_parts)
    if audio.size == 0:
        print("FAIL: TTS produced an empty waveform", file=sys.stderr)
        return 1
    sf.write(WAV_PATH, audio, SAMPLE_RATE, subtype="PCM_16")
    duration = audio.size / SAMPLE_RATE
    print(f"      Wrote {WAV_PATH.name}: {duration:.2f}s, {WAV_PATH.stat().st_size} bytes, {time.perf_counter() - started:.1f}s wall")

    stt_config = dict(stages["stt"].selection.config)
    stt_config["enable_live_transcription"] = False
    print(f"[4/4] Transcribing {WAV_PATH.name} with {stt_config['model_name']} on {stt_config['device']} ...")
    stt_audio, stt_rate = sf.read(WAV_PATH, dtype="float32")
    if stt_rate != SAMPLE_RATE:
        print(f"FAIL: expected {SAMPLE_RATE} Hz WAV, got {stt_rate} Hz", file=sys.stderr)
        return 1
    stt = ParakeetTDTSTTHandler(Event(), Queue(), Queue(), setup_kwargs=stt_config)
    transcript = ""
    detected_language = None
    for output in stt.process(
        VADAudio(audio=stt_audio, mode="final", turn_id="smoke-stt", turn_revision=0)
    ):
        if isinstance(output, Transcription):
            transcript = output.text.strip()
            detected_language = output.language_code
    if not transcript:
        print("FAIL: Parakeet returned an empty transcript", file=sys.stderr)
        return 1
    print(f"      Transcript: {transcript}")
    if detected_language:
        print(f"      Detected language: {detected_language}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
