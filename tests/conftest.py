from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig  # noqa: E402


@pytest.fixture()
def tmp_config(tmp_path, monkeypatch) -> AppConfig:
    monkeypatch.setenv("THERAPIST_PROFILE", "test")
    monkeypatch.setenv("THERAPIST_LIBRARIAN_ENABLED", "0")
    cfg = AppConfig()
    cfg.data_dir = tmp_path
    cfg.__post_init__()
    for d in (cfg.session_dir, cfg.chroma_dir, cfg.notebook_dir, cfg.notebook_trash_dir):
        d.mkdir(parents=True, exist_ok=True)
    cfg.prompts = {
        "therapist": "be calm",
        "safety": "classify",
        "crisis": "be careful",
        "librarian": "be tidy",
    }
    cfg.safety_keywords = [r"\bkill\s+myself\b"]
    return cfg


class FakeClient:
    def __init__(self):
        self.embed_calls = 0

    def embed(self, text: str):
        self.embed_calls += 1
        # Deterministic, no Ollama required.
        vec = [0.0] * 32
        for i, ch in enumerate(text.encode("utf-8")[:32]):
            vec[i] = (ch % 17) / 17.0
        return vec
