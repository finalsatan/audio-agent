"""Minimal local (offline) Kokoro text-to-speech for LiveKit Agents.

No official or community LiveKit Kokoro plugin exists on PyPI, so this is a small
custom ``tts.TTS`` implementation on top of ``kokoro-onnx`` (ONNX Runtime, CPU,
runs fine on Apple Silicon). Non-streaming: the framework wraps it in
``tts.StreamAdapter`` and calls ``synthesize()`` once per sentence.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request
from pathlib import Path

import numpy as np
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

logger = logging.getLogger("tutor.tts")

SAMPLE_RATE = 24_000  # what Kokoro publishes
NUM_CHANNELS = 1

_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.int8.onnx"
)
_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)
_MODEL_FILE = "kokoro-v1.0.int8.onnx"
_VOICES_FILE = "voices-v1.0.bin"


def ensure_model_files(models_dir: Path) -> tuple[Path, Path]:
    """Download the Kokoro ONNX model + voices once (~115 MB) if missing."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / _MODEL_FILE
    voices_path = models_dir / _VOICES_FILE
    for path, url in ((model_path, _MODEL_URL), (voices_path, _VOICES_URL)):
        if path.exists() and path.stat().st_size > 0:
            continue
        logger.info("downloading %s -> %s", url, path)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(path)
    return model_path, voices_path


class KokoroTTS(tts.TTS):
    def __init__(
        self, *, voice: str = "af_heart", speed: float = 1.0, models_dir: Path
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        model_path, voices_path = ensure_model_files(models_dir)

        from kokoro_onnx import Kokoro

        logger.info("loading Kokoro voice %r", voice)
        self._voice = voice
        self._speed = speed
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        # Warm up the ONNX session + phonemizer so the first reply is not slow.
        self._synth("Hello there, welcome to your English practice session. How are you today?")

    @property
    def model(self) -> str:
        return "kokoro-v1.0"

    @property
    def provider(self) -> str:
        return "kokoro-onnx"

    def _synth(self, text: str) -> np.ndarray:
        samples, _sample_rate = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang="en-us",
        )
        return np.asarray(samples, dtype=np.float32)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return KokoroChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class KokoroChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: KokoroTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: KokoroTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",  # raw 16-bit little-endian PCM
        )
        samples = await asyncio.to_thread(self._tts._synth, self.input_text)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        output_emitter.push(pcm.tobytes())
        output_emitter.flush()
