#!/usr/bin/env bash
# Run the English tutor demo: local mic/speakers, local STT/TTS, DeepSeek LLM.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env and add DEEPSEEK_API_KEY." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is not set in .env}"
: "${DEEPSEEK_BASE_URL:=https://api.deepseek.com}"
export DEEPSEEK_BASE_URL

export TUTOR_PROMPT="$(cat tutor_prompt.txt)"
export NLTK_DATA="$PWD/.nltk_data"

# speech-to-speech imports NLTK's punkt_tab and POS tagger at startup.
uv run python bootstrap_nltk.py >/dev/null

exec uv run speech-to-speech local -f config.yaml
