# English speaking practice — Hugging Face speech-to-speech

A minimal voice agent for practising spoken English on an Apple Silicon Mac.
You talk into the microphone; the agent answers through the speakers.

Everything except the language model runs locally:

| Stage | Backend | Model |
|---|---|---|
| VAD | Silero VAD + Smart Turn v3.2 | built into `speech-to-speech` |
| STT | Parakeet TDT via MLX Audio | `mlx-community/parakeet-tdt-0.6b-v3` |
| LLM | OpenAI-compatible Chat Completions | DeepSeek `deepseek-flash` |
| TTS | Qwen3-TTS via MLX Audio | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (6-bit MLX), speaker `Aiden` |

The demo is a YAML-configured `speech-to-speech local` session: one process runs
the Realtime server and the microphone/speaker client over loopback. No cloud
STT or TTS service is used.

## Requirements

- macOS on Apple Silicon (MLX backends).
- [uv](https://docs.astral.sh/uv/) with Python 3.12 (the project pins 3.12).
- PortAudio for the microphone and speakers: `brew install portaudio`.
- A DeepSeek API key. The key stays in `.env`, which is gitignored.

## Setup

```bash
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY
uv sync
```

`.env` must contain:

```dotenv
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

The first run downloads the Parakeet, Smart Turn and Qwen3-TTS models into the
Hugging Face cache. This can take several minutes on a slow connection.

## Run

```bash
./run.sh
```

`run.sh` loads `.env`, exports the tutor prompt from `tutor_prompt.txt`, bootstraps
the small NLTK data files that `speech-to-speech` needs, and launches:

```bash
uv run speech-to-speech local -f config.yaml
```

Speak, then pause. The end-of-speech threshold is 1.2 seconds. The pipeline also
uses Smart Turn, so an obviously unfinished sentence can wait a little longer.
Stop with `Ctrl+C`.

To check the setup without a microphone, run the smoke test:

```bash
uv run python smoke_test.py
```

It imports the same pipeline modules, sends one text turn through the configured
DeepSeek backend, prints the reply, and synthesizes one sentence to
`smoke_tts.wav` using the configured local Qwen3-TTS voice.

## Configuration

Most settings are in `config.yaml`; no long CLI flag list is needed.

### Tutor prompt

Edit `tutor_prompt.txt`. It is injected into the LLM block as
`init_chat_prompt` through the `TUTOR_PROMPT` environment variable. The prompt is
written for speech: short replies, no markdown or emoji, one follow-up question,
and one brief correction when the learner makes a clear mistake.

### Voice

The default English voice is Qwen3-TTS CustomVoice speaker `Aiden`. Change it in
`config.yaml`:

```yaml
qwen3_tts_speaker: Ryan
qwen3_tts_language: english
```

Other CustomVoice speakers can be used (for example `Dylan`, `Eric`, `Ryan`,
`Serena`, `Vivian`). To use 4-bit or 8-bit MLX weights, change
`qwen3_tts_mlx_quantization` (`bf16`, `4bit`, `6bit`, `8bit`).

### Silence threshold and turn-taking

The learner-friendly pause is set in the VAD block:

```yaml
min_silence_ms: 1200   # 1.2 s of silence ends the turn
```

Other useful VAD settings:

- `thresh` (default 0.5): lower is more sensitive to quiet speech.
- `min_speech_ms` (384): ignore very short sounds.
- `smart_turn_max_wait_ms` (2000): extra grace when Smart Turn thinks the sentence
  is incomplete.
- Set `smart_turn: false` to use only the fixed silence threshold.

### Feedback and barge-in

`runtime.client.block_mic_during_playback: true` stops the microphone while the
agent speaks. This prevents speaker output from being transcribed as a new turn
when using laptop speakers. If you use headphones and want to interrupt the
agent by voice, set it to `false`.

## Files

| File | Purpose |
|---|---|
| `run.sh` | Loads `.env` and starts the demo. |
| `config.yaml` | Realtime pipeline: VAD, STT, LLM and TTS blocks. |
| `tutor_prompt.txt` | System prompt for the conversation partner. |
| `smoke_test.py` | Hardware-free check of imports, LLM turn and TTS output. |
| `bootstrap_nltk.py` | Fetches the NLTK resources used at startup. |
| `.env.example` | Placeholder environment variables. |

## Known limitations

- Live speech was not available in the build environment. The local command
  started and reached `Ready — start speaking.`, the server answered a
  WebSocket text turn with spoken audio, and `smoke_test.py` synthesized audio
  to a WAV file. Real microphone capture and speaker playback still need a
  human check.
- DeepSeek is a network dependency. The demo keeps STT and TTS local, but the
  transcribed text and conversation history are sent to the DeepSeek API.
- The installed `speech-to-speech` version (1.1.0) does not expose acoustic
  echo cancellation, so speaker use either blocks the mic during playback or
  risks the agent hearing itself. Headphones are the best experience.
- `deepseek-flash` is a reasoning model. `config.yaml` sends
  `reasoning_effort: none` to disable thinking tokens; without that, first-token
  latency is noticeably worse.
- Parakeet TDT v3 is multilingual rather than strictly English-only. It detects
  the spoken language automatically; the tutor prompt and TTS voice are pinned
  to English.
- First-run model download and loading time is significant. After the models are
  cached, startup is much faster.
- The default reply latency is dominated by the 1.2 s end-of-speech wait plus
  Cloud LLM and local TTS time. This trade-off is intentional for learners who
  pause while thinking.
