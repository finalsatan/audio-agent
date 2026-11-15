"""Minimal local (offline) speech-to-text for LiveKit Agents.

No official or community LiveKit plugin exists for MLX / local Whisper on PyPI,
so this is a small custom ``stt.STT`` implementation:

* non-streaming: one ``recognize()`` call per VAD speech segment
* wrapped in ``stt.StreamAdapter`` (see agent.py) so the pipeline still gets a
  streaming STT interface driven by Silero VAD
* inference runs locally with ``mlx_whisper`` (Apple Silicon / MLX)
"""

from __future__ import annotations

import asyncio
import logging

import mlx_whisper
import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

logger = logging.getLogger("tutor.stt")

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono float32


def _audio_to_float32_16k(buffer: utils.AudioBuffer) -> np.ndarray:
    """Downmix / resample the recorded frames to 16 kHz mono float32 in [-1, 1]."""
    frame = rtc.combine_audio_frames(buffer)

    if frame.sample_rate != SAMPLE_RATE:
        resampler = rtc.AudioResampler(
            input_rate=frame.sample_rate,
            output_rate=SAMPLE_RATE,
            num_channels=frame.num_channels,
        )
        out = resampler.push(frame) + resampler.flush()
        if out:
            frame = rtc.combine_audio_frames(out)

    samples = np.frombuffer(frame.data, dtype=np.int16)
    if frame.num_channels > 1:
        samples = samples.reshape(-1, frame.num_channels).mean(axis=1)
    return (samples.astype(np.float32) / 32768.0).copy()


class WhisperSTT(stt.STT):
    """Local Whisper (MLX) recognizer, English by default."""

    def __init__(self, *, model: str) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model = model
        # Warm up: loads the MLX weights once so the first real utterance is fast.
        logger.info("loading MLX Whisper model %s", model)
        self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "mlx-whisper"

    def _transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model,
            language=language,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        return str(result.get("text", "")).strip()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        audio = _audio_to_float32_16k(buffer)
        lang = language if isinstance(language, str) and language else "en"
        duration = len(audio) / SAMPLE_RATE

        text = await asyncio.to_thread(self._transcribe, audio, lang)
        logger.debug("recognized %.2fs of audio as %r", duration, text)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language=lang, text=text, start_time=0.0, end_time=duration)
            ],
        )
