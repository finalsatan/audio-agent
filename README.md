# 英文口语练习语音 Agent：三个 Demo 概览

平台：macOS Apple Silicon。三个 demo 分别在不同 git worktree / 分支中，互不修改。
**只有 LLM 走云端**：DeepSeek 官方 API，模型 `deepseek-flash`。STT 和 TTS 都在本地运行。`.env` 只需要 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL`，不要把密钥写进代码或提交到 git。

## 分支、路径与运行命令

| 分支 | 绝对路径 | 启动命令 | 无麦克风 smoke test |
|---|---|---|---|
| `demo/hf-s2s` | `/Users/murphy/code/ai/audio-agent-hf-s2s` | `./run.sh` | `uv run python smoke_test.py` |
| `demo/pipecat` | `/Users/murphy/code/ai/audio-agent-pipecat` | `uv run bot.py` | `uv run smoke_test.py` |
| `demo/livekit` | `/Users/murphy/code/ai/audio-agent-livekit` | `uv run agent.py console` | `uv run smoke_test.py` |

三个 demo 都需要先安装 PortAudio（`brew install portaudio`）、用 `uv sync` 创建 Python 3.12 环境，并把 DeepSeek key 放进各自目录下被 gitignore 的 `.env`。

## STT / TTS 差异

| 分支 | 本地 STT | 本地 TTS | 备注 |
|---|---|---|---|
| `demo/hf-s2s` | NVIDIA Parakeet TDT 0.6B v3，MLX（`mlx-community/parakeet-tdt-0.6b-v3`），多语言自动识别 | Qwen3-TTS 12Hz 1.7B CustomVoice，MLX 6bit，英文音色 `Aiden` | Hugging Face `speech-to-speech` 的 local 模式；Silero VAD + Smart Turn；默认 1.2 s 停顿；播放时默认关闭麦克风以避免自激 |
| `demo/pipecat` | MLX Whisper large-v3-turbo（`mlx-community/whisper-large-v3-turbo`） | Kokoro-82M（`kokoro-onnx`，24 kHz），默认 `af_heart` | Pipecat `LocalAudioTransport`；Silero VAD + Smart Turn；约 1.2 s 停顿；支持打断 |
| `demo/livekit` | MLX Whisper small.en（`mlx-community/whisper-small.en-mlx`） | Kokoro-82M（`kokoro-onnx`），默认 `af_heart` | LiveKit Agents `console` 模式（不需要 LiveKit server/账号）；Silero VAD + TurnDetector；约 1.2 s endpointing；`Ctrl+T` 可切文字模式 |

三个 demo 的 STT/TTS 模型、量化、框架和流式策略都不同，因此**这不是同一模型下的 apples-to-apples 延迟对比**；只能说明各方案能跑通，不能据此直接比较谁更快或音质更好。

## 使用建议

- **请使用耳机。** 三个实现都没有可靠的声学回声消除；外放时 agent 可能听到自己的声音并误触发打断。`demo/hf-s2s` 默认在播放时关闭麦克风来规避，但这样也会失去语音打断。
- 首次运行会下载本地模型，可能较慢；之后走本地缓存。
- 三个 demo 的 turn-taking 都刻意放宽了停顿阈值（约 1.0–1.5 s），适合学习者思考时停顿。

## 验证状态

- 三个 demo 都提供了无麦克风的 smoke test：导入管线、跑一轮 DeepSeek 文本对话、用本地 TTS 合成 WAV；Pipecat/LiveKit 以及最新的 `hf-s2s` smoke test 还会把 WAV 回灌给本地 STT 做识别检查。
- **真实麦克风输入和真人对话尚未验证。** 当前验证只覆盖模块导入、LLM 文本回合、TTS 合成，以及三个 demo 均已覆盖的 TTS WAV → STT 回环识别；没有覆盖真人说话、设备权限、回声、打断和实际听感。需要人工在 macOS 上戴耳机复核。
- 云端依赖只有 DeepSeek API；如果 DeepSeek 不可达，三个 demo 都无法生成回复。STT/TTS/音频采集/播放均为本地。

## 修改入口

- `demo/hf-s2s`：`tutor_prompt.txt`（系统提示）、`config.yaml`（音色、VAD 停顿、模型、客户端设置）。
- `demo/pipecat`：`config.py`（`TUTOR_SYSTEM_PROMPT`、`STT_MODEL`、`TTS_VOICE`、`VAD_STOP_SECS`、`SMART_TURN_STOP_SECS`）。
- `demo/livekit`：`prompt.py`（`TUTOR_PROMPT`）、`config.py`（Whisper/Kokoro 模型、音色、endpointing 参数）。

## 实现说明

本概览与三个 demo 的协作实现/验证流程使用 Herdr + pi + WorkBuddy DeepSeek Flash。
