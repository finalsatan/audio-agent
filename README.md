# 英文口语练习语音 Agent（macOS Apple Silicon）

面向 **macOS Apple Silicon** 的本地英文口语练习语音 Agent：
- **本地 STT**：MLX Whisper `large-v3-turbo`（Apple Silicon GPU 加速，带静音与重复幻觉过滤）
- **本地 TTS**：Kokoro-82M（`kokoro-onnx`，24 kHz，默认美式女声 `af_heart`）
- **本地轮次检测**：Silero VAD（`confidence=0.6`, `stop_secs=1.2s`）+ Smart Turn v3（`1.5s`），支持语音打断（barge-in）
- **云端 LLM**：DeepSeek 官方 API（`deepseek-flash`，显式锁定 `thinking=disabled` 降低首字延迟）
- **实时字幕与复盘记录**：终端实时打印每轮对话与 STT/LLM/TTS 耗时，并自动将练习记录保存至 `transcripts/YYYY-MM-DD_HH-MM-SS.md`

> **请佩戴耳机使用。** 当前为本地 CLI 音频流，不含声学回声消除（AEC）；外放时可能误触发打断。

---

## 快速开始

### 1. 安装依赖

```bash
brew install portaudio
uv sync
```

### 2. 配置 DeepSeek API Key

```bash
cp .env.example .env
# 在 .env 中填入你的 DEEPSEEK_API_KEY
```

### 3. 运行无麦克风 Smoke Test（可选）

验证模块导入、DeepSeek 显式关闭 thinking 配置、单轮文本对话、Kokoro 合成 WAV 与 MLX Whisper 回环识别：

```bash
uv run smoke_test.py
```

### 4. 启动语音口语练习

戴上耳机后运行：

```bash
uv run bot.py
```

启动后 Agent（Sam）会先主动打招呼，听到声音即可直接对话。对话过程中终端会实时打印：
- `[HH:MM:SS] YOU   : ...`（学习者转写 + STT 耗时）
- `[HH:MM:SS] TUTOR : ...`（助手回复，若被打断会标注 `[interrupted]`）
- `[HH:MM:SS] STATS : LLM / TTS / Tokens`（每轮延迟与 token 消耗）

按 `Ctrl+C` 结束会话，完整对话记录会实时落盘在 `transcripts/` 目录下（已加入 `.gitignore`）。

---

## 常用配置项

所有参数均可在 [config.py](config.py) 修改，或直接在 `.env` 中通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | （必填） | DeepSeek 官方 API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-flash` | 对话模型（代码已显式锁定 `thinking=disabled`） |
| `STT_MODEL` | `mlx-community/whisper-large-v3-turbo` | 本地 MLX Whisper 模型 |
| `TTS_VOICE` | `af_heart` | Kokoro 音色（如 `af_heart`, `af_bella`, `am_michael`, `bf_emma`, `bm_george`） |
| `TTS_SPEED` | `1.0` | 语速（学习者可设为 `0.9` 放慢语速） |
| `VAD_STOP_SECS` | `1.2` | 说话停顿判定阈值（秒），放宽以允许思考 |
| `SMART_TURN_STOP_SECS` | `1.5` | Smart Turn v3 最大等待上限（秒） |
| `VAD_CONFIDENCE` | `0.6` | Silero VAD 置信度门槛，过滤轻微底噪 |
| `SAVE_TRANSCRIPT` | `true` | 是否自动保存 Markdown 对话记录 |
| `TRANSCRIPTS_DIR` | `./transcripts` | 对话记录保存目录 |
| `AUDIO_INPUT_DEVICE` / `AUDIO_OUTPUT_DEVICE` | 系统默认 | 可选 PyAudio 设备索引 |

修改人设或纠错风格请编辑 [config.py](config.py) 中的 `TUTOR_SYSTEM_PROMPT` 与 `GREETING_INSTRUCTION`。

---

## 架构选型记录（三个 Demo 实测对比）

项目早期通过三个独立分支（`demo/pipecat`、`demo/livekit`、`demo/hf-s2s`）并行原型验证，并在 macOS 上佩戴耳机完成了真人语音实测与同模型 A/B 对比，最终选定 **`demo/pipecat`** 合入 `main`：

| 排名 | 历史分支 | 实测核心结论 |
|---|---|---|
| **#1（已合入 `main`）** | `demo/pipecat` | **综合体验最优**：`whisper-large-v3-turbo` 对非母语口语准确率最稳；VAD（`confidence=0.6` + `1.2s` + Smart Turn v3）抗底噪极佳，打断流畅；官方原生集成 STT/TTS/VAD/DeepSeek，零胶水代码、零代理环境坑。 |
| **#2** | `demo/livekit` | 框架的 `TurnDetector v1-mini` 与抢跑生成（Preemptive Generation）很强，但默认 `whisper-small.en` 存在大量 `"Thank you."` 幻觉与重复乱码；即便换上同样的 `whisper-large-v3-turbo` 与 `no_speech_prob` 过滤，本地 `console` 模式下默认 Silero VAD 过于敏感且 `vad` 打断激进，仍会频繁把底噪识成 `"Thank you."` 并误打断助手；且 `console` 模式已标记 deprecated。 |
| **#3** | `demo/hf-s2s` | Parakeet TDT 0.6B 推理极快（50–170 ms）且监控埋点完整，但对带口音英文的词错率偏高（易带偏多轮对话）、端到端首响固定 2.4–3.1 s、默认播音关麦不支持打断，且在开启 macOS 系统 SOCKS 代理时本地 WebSocket 易启动失败。 |

三个历史对比分支仍保留在仓库中供查阅。
