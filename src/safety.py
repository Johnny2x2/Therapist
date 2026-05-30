from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Pattern

from .config import AppConfig
from .llm_therapist import OllamaClient


@dataclass
class SafetyResult:
    flagged: bool
    risk_level: str
    reason: str
    source: str  # "classifier" | "keyword" | "fallback"


class SafetyMonitor:
    def __init__(self, config: AppConfig, client: OllamaClient):
        self.config = config
        self.client = client
        self._patterns: List[Pattern[str]] = []
        for raw in config.safety_keywords:
            try:
                self._patterns.append(re.compile(raw, re.IGNORECASE))
            except re.error:
                continue

    def assess(self, user_text: str) -> SafetyResult:
        keyword_hit = self._keyword_match(user_text)
        result = self._classify(user_text)
        if keyword_hit is not None and not result.flagged:
            result = SafetyResult(
                flagged=True,
                risk_level="high",
                reason=f"Keyword pre-screen matched: {keyword_hit!r} (classifier reason: {result.reason})",
                source="keyword",
            )
        elif keyword_hit is not None:
            result = SafetyResult(
                flagged=True,
                risk_level=result.risk_level or "high",
                reason=f"Keyword pre-screen + classifier agreed (keyword {keyword_hit!r}; {result.reason})",
                source="keyword+classifier",
            )
        self._log(user_text, result)
        return result

    def build_crisis_reply(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.config.prompts["therapist"]},
            {"role": "system", "content": self.config.prompts["crisis"]},
            {"role": "user", "content": user_text},
        ]
        return "".join(self.client.stream_chat(messages, model=self.config.models.therapist_model)).strip()

    # ------------------------------------------------------------------ internals
    def _keyword_match(self, text: str):
        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def _classify(self, user_text: str) -> SafetyResult:
        prompt = self.config.prompts["safety"]
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ]
        # Try JSON-mode first; fall back to free-form if the model rejects it.
        try:
            data = self.client.chat(
                messages,
                model=self.config.models.safety_model,
                fmt="json",
            )
            output = ((data.get("message") or {}).get("content") or "").strip()
        except Exception:
            output = "".join(
                self.client.stream_chat(messages, model=self.config.models.safety_model)
            ).strip()
        try:
            parsed: Dict[str, object] = json.loads(output)
        except ValueError:
            return SafetyResult(
                flagged=False,
                risk_level="unknown",
                reason="Classifier response was not valid JSON.",
                source="fallback",
            )
        return SafetyResult(
            flagged=bool(parsed.get("flagged", False)),
            risk_level=str(parsed.get("risk_level", "low")),
            reason=str(parsed.get("reason", "")),
            source="classifier",
        )

    def _log(self, user_text: str, result: SafetyResult) -> None:
        try:
            path: Path = self.config.safety_log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "flagged": result.flagged,
                "risk_level": result.risk_level,
                "reason": result.reason,
                "source": result.source,
                "preview": user_text[:160],
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            pass
