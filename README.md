# English practice voice agent — Pipecat demo

A minimal voice agent for **English speaking and listening practice**, running
on a Mac. You talk into the microphone, a friendly tutor answers out loud with a
short reply plus a follow-up question. Barge-in is enabled: start speaking while
the bot talks and it stops immediately.

Everything that touches audio runs **locally** on Apple Silicon. Only the LLM
call goes to the network.

```
mic ──► LocalAudioTransport ──► Whisper (MLX, local STT)
                                    │
                                    ▼
                        user context aggregator
                        (Silero VAD + Smart Turn v3 turns)
                                    │
                                    ▼
                        DeepSeek (deepseek-flash) ──► network
                                    │
                                    ▼
                    Kokoro (kokoro-onnx, local TTS)
                                    │
                                    ▼
                    LocalAudioTransport ──► speakers
                                    │
                                    ▼
                        assistant context aggregator
```

| Piece | What is used | Where it runs |
|---|---|---|
| Transport | `LocalAudioTransport` (PyAudio) | local |
| VAD | `SileroVADAnalyzer` | local |
| End of turn | `LocalSmartTurnAnalyzerV3` (ONNX smart-turn v3) | local |
| STT | `WhisperSTTServiceMLX` — `mlx-community/whisper-large-v3-turbo` | local (MLX/GPU) |
| LLM | `DeepSeekLLMService` (pipecat's own DeepSeek service), model `deepseek-flash` | DeepSeek API |
| TTS | `KokoroTTSService` (kokoro-onnx, 24 kHz) | local |

Tested with `pipecat-ai` **1.16.0** on macOS 26 (Apple M5, Python 3.12).

## Files

| File | Purpose |
|---|---|
| `bot.py` | The demo. Builds the pipeline and runs it. |
| `config.py` | Tutor prompt, model/voice choices, silence thresholds, service factories. |
| `smoke_test.py` | Headless checks (no mic/speakers): imports, one LLM turn, TTS to wav, STT round-trip. |
| `.env.example` | Template for the DeepSeek credentials. |

## Setup

Requirements: macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/), PortAudio.

```bash
brew install portaudio          # PyAudio needs it
cp .env.example .env            # then put your DeepSeek key in .env
uv sync                         # creates .venv with Python 3.12
```

`.env` (gitignored) needs:

```
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

No other keys are used: STT and TTS are local models, downloaded on first run
(Kokoro ~330 MB into `~/.cache/pipecat/kokoro-onnx`, Whisper turbo ~1.6 GB into
the Hugging Face cache).

## Run

```bash
uv run bot.py
```

The bot greets you, then you just talk. Press `Ctrl-C` to quit.
Use **headphones**: the demo has no echo cancellation, so on speakers the bot
hears itself and may interrupt itself.

Useful environment variables:

```bash
LOG_LEVEL=DEBUG uv run bot.py                          # verbose logs
AUDIO_INPUT_DEVICE=2 AUDIO_OUTPUT_DEVICE=3 uv run bot.py  # pick PyAudio devices
```

List device indexes with:

```bash
uv run python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
```

## Verify without a microphone

```bash
uv run smoke_test.py
```

It (1) imports `bot.py`/`config.py` and builds every processor, (2) runs one
learner sentence through the same DeepSeek configuration the demo uses and
prints the reply, (3) synthesizes that reply with the configured Kokoro voice
into `smoke_test_tts.wav`, and (4) transcribes that wav back with the configured
MLX Whisper model.

## Changing things

- **Tutor prompt** — `TUTOR_SYSTEM_PROMPT` in `config.py` (it is deliberately
  plain-spoken: 1–3 short sentences, no markdown, always ending with a question,
  and it models corrections once instead of lecturing).
- **Greeting** — `GREETING_INSTRUCTION` in `config.py`.
- **Voice** — `TTS_VOICE` in `config.py` (or the env). Kokoro English voices:
  `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `am_michael`, `am_adam`,
  `bf_emma`, `bm_george` (prefix `a` = US, `b` = UK; `f`/`m` = female/male).
  `TTS_SPEED` slows speech down (e.g. `0.9`) for learners.
- **End-of-speech silence** — learners pause a lot, so the defaults are generous:
  `VAD_STOP_SECS=1.2` (Silero silence before "user stopped speaking") and
  `SMART_TURN_STOP_SECS=1.5` (hard cap for the turn detector). Raise both if the
  bot still cuts in too early; lower them for snappier replies. Both are env
  variables too.
- **Models** — `STT_MODEL` (any MLX Whisper repo, e.g.
  `mlx-community/whisper-small-mlx` for lower latency) and `DEEPSEEK_MODEL`.
- **Devices** — `AUDIO_INPUT_DEVICE` / `AUDIO_OUTPUT_DEVICE` (see above).

## Known limitations

- **No echo cancellation.** On speakers the bot can hear its own voice and
  interrupt itself. Headphones fix it. (Pipecat's filters, e.g. the Krisp
  filter, would need an extra key, so none are used here.)
- **Latency.** Local Whisper is not streaming: it transcribes after the silence
  window, roughly 0.3–1.5 s for a short utterance on an M-series Mac. The LLM
  round trip adds ~0.2–1 s. Replies feel good but not instant.
- **First run is slow** because of the model downloads (Kokoro, Whisper).
- `deepseek-flash` is used with DeepSeek "thinking" disabled (pipecat's default
  for this service) so the first token is fast; the system prompt is what keeps
  replies short.
- The log line `VAD stop_secs (1.2s) differs from the recommended default (0.2s)`
  is expected: we intentionally use a longer silence window. `ttfs_p99_latency`
  on the STT service was raised to 1.5 s to match.
- English only (Whisper is configured with `Language.EN`).
- Verified headlessly (see `smoke_test.py`) plus one local run of `bot.py`; the
  mic/speaker conversation was not verified with a real human speaker.
