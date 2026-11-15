# LiveKit Agents — English speaking-practice voice agent

A minimal voice agent for practising spoken English on a Mac (Apple Silicon).
You talk into the microphone, the agent answers out loud, and it always ends its
turn with a question so the conversation keeps going.

Everything runs locally except the LLM (DeepSeek's OpenAI-compatible API):

| Part | Implementation | Where it runs |
|---|---|---|
| Framework | [LiveKit Agents](https://github.com/livekit/agents) 1.x | — |
| Transport | `agent.py console` (fake job, no LiveKit server, no account) | local |
| VAD | Silero, bundled in `livekit-local-inference` (`inference.VAD`) | local, no download |
| End-of-turn | `inference.TurnDetector(version="v1-mini")` (on-device audio EOT model) | local |
| STT | `local_stt.WhisperSTT` — custom `stt.STT` over MLX Whisper (`whisper-small.en`) | local |
| LLM | `openai.LLM` plugin pointed at `https://api.deepseek.com`, model `deepseek-flash` | cloud |
| TTS | `local_tts.KokoroTTS` — custom `tts.TTS` over `kokoro-onnx` (Kokoro-82M, voice `af_heart`) | local |

There is **no official LiveKit plugin** for MLX/local Whisper or Kokoro on PyPI
(`livekit-plugins-whisper` and `livekit-plugins-kokoro` do not exist), so both are
small custom implementations: non-streaming `recognize()` / `synthesize()` wrapped
by the framework's `stt.StreamAdapter` (VAD-driven) and `tts.StreamAdapter`
(sentence-paced).

Nothing here needs a LiveKit server or LiveKit Cloud credentials: `console` mode
runs a mock job in-process, and VAD/turn-detection/STT/TTS models are all local.

## Setup

```bash
brew install portaudio        # needed by sounddevice for mic/speaker access
uv sync                       # Python 3.12 env from uv.lock

cp .env.example .env          # then put your real key in .env
#   DEEPSEEK_API_KEY=sk-...
#   DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Models are downloaded on first run (internet needed once):

- Kokoro ONNX model + voices (~115 MB) → `models/` (gitignored)
- MLX Whisper `mlx-community/whisper-small.en-mlx` (~480 MB) → Hugging Face cache
- Silero VAD / turn-detector-v1-mini → already inside the Python wheels

## Run

```bash
uv run agent.py console
```

Speak after the greeting. Ctrl+T switches to text mode (type instead of talk),
Ctrl+C quits. macOS will ask for microphone permission on first run.

Offline smoke test (no mic, no LiveKit server — imports, one DeepSeek turn, one
Kokoro sentence written to `out/smoke_tts.wav`, and an MLX Whisper transcription
of that wav):

```bash
uv run smoke_test.py
```

## Files

```
agent.py        entrypoint: VAD + StreamAdapter(Whisper STT) + DeepSeek LLM + Kokoro TTS, console CLI
prompt.py       TUTOR_PROMPT — the tutor's system prompt
config.py       .env loading, LLM factory, tunables (voice, silence, models)
local_stt.py    custom non-streaming Whisper STT (MLX)
local_tts.py    custom non-streaming Kokoro TTS (ONNX) + model download
smoke_test.py   hardware-free verification
```

## Tuning

Edit `.env` (or the defaults in `config.py`):

| Knob | Default | Meaning |
|---|---|---|
| `TUTOR_SILENCE_SECONDS` | `1.2` | Silence after the learner stops speaking before the turn is taken (`endpointing.min_delay`) |
| `TUTOR_MAX_SILENCE_SECONDS` | `4.0` | Hard cap on waiting for the learner to finish (`endpointing.max_delay`) |
| `TUTOR_VAD_SILENCE_SECONDS` | `0.45` | Silence that closes a speech segment before Whisper runs |
| `TUTOR_VOICE` | `af_heart` | Kokoro voice (`af_heart`, `af_bella`, `am_michael`, `am_fenrir`, `bf_emma`, `bm_george`, …) |
| `TUTOR_SPEED` | `1.0` | Kokoro speaking rate |
| `TUTOR_WHISPER_MODEL` | `mlx-community/whisper-small.en-mlx` | Any MLX Whisper HF repo (`whisper-base.en-mlx`, `whisper-large-v3-turbo`, …) |
| `DEEPSEEK_MODEL` | `deepseek-flash` | DeepSeek model name |

The tutor's behaviour lives in **`prompt.py`** (`TUTOR_PROMPT`) — edit it to change
persona, correction style, or reply length.

## Known limitations

- **Console mode is deprecated** in LiveKit Agents 1.8 (printed warning); it still
  works. The Go-based `lk agent console` is the future replacement and needs the
  LiveKit CLI installed.
- **Both STT and TTS are non-streaming**, so replies start after the first sentence
  is fully synthesized. Measured on an M5: Whisper ≈0.2 s for a 6 s utterance,
  Kokoro ≈0.3 s per sentence (≈0.1× real time), DeepSeek ≈1–3 s for the first token
  (it is a reasoning model).
- **No barge-in feedback loop is modelled**: the Mac speakers are picked up by the
  Mac microphone unless you use headphones, so the agent can hear itself. Use
  headphones for a clean demo. (Console mode does not run echo cancellation.)
- Turn detection uses the small local `v1-mini` model; a low end-of-turn
  probability means the agent waits up to `TUTOR_MAX_SILENCE_SECONDS` before
  answering.
- Harmless startup warnings in console mode: `LIVEKIT_INFERENCE_URL … falling back
  to turn-detector-v1-mini` (an eagerly-created default detector that is not used)
  and, when stdin is not a TTY, a `termios` traceback from the key listener.
- Only English is configured (`whisper-small.en` + `lang="en-us"` Kokoro).
- Verified without a microphone via `smoke_test.py`; the mic/speaker path was
  exercised manually on macOS only.
