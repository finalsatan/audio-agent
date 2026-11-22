"""Real-time terminal transcript/latency logging and Markdown session recording.

Attached as a Pipecat ``BaseObserver`` plus aggregator turn handlers so it
observes metrics and finalized user/assistant turns without altering the audio
pipeline. Turns are flushed to ``transcripts/YYYY-MM-DD_HH-MM-SS.md`` as they
happen, so pressing Ctrl+C never loses the session log.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


def _short_service_name(processor: str) -> str:
    lower = processor.lower()
    if "whisper" in lower or "stt" in lower:
        return "STT"
    if "deepseek" in lower or "llm" in lower or "openai" in lower:
        return "LLM"
    if "kokoro" in lower or "tts" in lower:
        return "TTS"
    return processor.split("#", 1)[0]


class SessionRecorder(BaseObserver):
    """Logs live turns + latency metrics and saves a Markdown review log."""

    def __init__(
        self,
        *,
        transcripts_dir: Path | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__(observe_every_push=False)
        self._transcripts_dir = transcripts_dir
        self._metadata = metadata or {}
        self._started_at = datetime.now()
        self._file_path: Path | None = None
        self._turn_count = 0
        self._turn_metrics: dict[str, str] = {}

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    def attach(self, user_aggregator: Any, assistant_aggregator: Any) -> None:
        """Subscribe to finalized user and assistant turn events."""

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def _on_user_turn_stopped(_aggregator, _strategy, message) -> None:
            text = (getattr(message, "content", None) or "").strip()
            if text:
                self.record_user_turn(text)

        @assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def _on_assistant_turn_stopped(_aggregator, message) -> None:
            text = (getattr(message, "content", None) or "").strip()
            interrupted = bool(getattr(message, "interrupted", False))
            if text:
                self.record_assistant_turn(text, interrupted=interrupted)

    def _ensure_file(self) -> Path | None:
        if self._transcripts_dir is None:
            return None
        if self._file_path is None:
            self._transcripts_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._started_at.strftime("%Y-%m-%d_%H-%M-%S")
            self._file_path = self._transcripts_dir / f"{stamp}.md"
            lines = [
                f"# English Practice Session ({self._started_at.strftime('%Y-%m-%d %H:%M:%S')})",
                "",
            ]
            if self._metadata:
                for key, value in self._metadata.items():
                    lines.append(f"- **{key}**: `{value}`")
                lines.append("")
            lines.append("## Conversation")
            lines.append("")
            self._file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self._file_path

    def _append_markdown(self, block: str) -> None:
        path = self._ensure_file()
        if path is None:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(block)

    def _format_metrics_summary(self) -> str:
        if not self._turn_metrics:
            return ""
        ordered_keys = ("STT", "LLM", "TTS", "Tokens")
        parts = [f"{k}: {self._turn_metrics[k]}" for k in ordered_keys if k in self._turn_metrics]
        for k, v in self._turn_metrics.items():
            if k not in ordered_keys:
                parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def record_user_turn(self, text: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self._turn_count += 1
        stt_note = self._turn_metrics.pop("STT", None)
        suffix = f"  ({stt_note})" if stt_note else ""
        print(f"[{now}] YOU   : {text}{suffix}", flush=True)
        meta = f" _(STT: {stt_note})_" if stt_note else ""
        self._append_markdown(f"- **[{now}] You**{meta}: {text}\n")

    def record_assistant_turn(self, text: str, *, interrupted: bool = False) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        tag = " [interrupted]" if interrupted else ""
        print(f"[{now}] TUTOR : {text}{tag}", flush=True)

        summary = self._format_metrics_summary()
        self._turn_metrics.clear()
        if summary:
            print(f"[{now}] STATS : {summary}", flush=True)

        meta_parts = []
        if interrupted:
            meta_parts.append("interrupted")
        if summary:
            meta_parts.append(summary)
        meta = f" _({'; '.join(meta_parts)})_" if meta_parts else ""
        self._append_markdown(f"- **[{now}] Tutor**{meta}: {text}\n\n")

    async def on_push_frame(self, data: FramePushed) -> None:
        if not isinstance(data.frame, MetricsFrame):
            return

        for item in data.frame.data:
            service = _short_service_name(item.processor)

            if isinstance(item, TTFATMetricsData):
                think = f", think {item.thinking_time:.2f}s" if item.thinking_time >= 0.01 else ""
                self._turn_metrics[service] = f"ttfat {item.ttfat:.2f}s{think}"
            elif isinstance(item, TTFAMetricsData):
                self._turn_metrics[service] = f"ttfa {item.ttfa:.2f}s"
            elif isinstance(item, TTFBMetricsData):
                # Prefer richer TTFAT/TTFA if already recorded for this turn.
                if service not in self._turn_metrics or self._turn_metrics[service].startswith("ttfb"):
                    self._turn_metrics[service] = f"ttfb {item.value:.2f}s"
            elif isinstance(item, ProcessingMetricsData):
                if service == "STT":
                    self._turn_metrics["STT"] = f"{item.value:.2f}s"
            elif isinstance(item, LLMUsageMetricsData):
                usage = item.value
                reasoning = (
                    f", reasoning={usage.reasoning_tokens}"
                    if usage.reasoning_tokens
                    else ""
                )
                self._turn_metrics["Tokens"] = (
                    f"in={usage.prompt_tokens}, out={usage.completion_tokens}{reasoning}"
                )
