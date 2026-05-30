from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import AppConfig
from src.llm_therapist import (
    ConversationState,
    Message,
    TherapistEngine,
)
from src.main import TherapistApp


SUMMARY_PREFIX = "Running summary of earlier conversation"


class _SummaryClient:
    """Fake Ollama client: records summarization calls, yields a canned summary."""

    def __init__(self):
        self.calls: List[List[Dict[str, Any]]] = []

    def stream_chat(self, messages, model: Optional[str] = None, fmt: Optional[str] = None):
        messages = list(messages)
        self.calls.append(messages)
        yield "condensed summary of earlier turns"


class _FakeNotebook:
    def __init__(self):
        self.writes = 0
        self.updates = 0
        self.last_body = ""

    def write_note(self, category, title, body, tags=None, pinned=False,
                   related=None, source_session=""):
        self.writes += 1
        self.last_body = body
        self.category = category
        self.tags = tags
        return {"id": "note-1"}

    def update_note(self, note_id, append=None, replace_body=None, **kw):
        self.updates += 1
        if replace_body is not None:
            self.last_body = replace_body
        return {"id": note_id}


def _fill_pairs(state: ConversationState, count: int, size: int = 200) -> None:
    for _ in range(count):
        state.append("user", "u" * size)
        state.append("assistant", "a" * size)


# --------------------------------------------------------------- pure windowing
def test_split_keeps_recent_pairs_and_never_touches_pinned():
    state = ConversationState(messages=[
        Message(role="system", content="prompt", pinned=True),
    ])
    _fill_pairs(state, 6)
    evicted, kept = state.split_tail_for_eviction(keep_tokens=50)
    # Pinned head excluded from both.
    assert all(not m.pinned for m in evicted + kept)
    # At least the most recent pair is retained.
    assert kept[-1].role == "assistant"
    assert kept[-2].role == "user"
    # Everything is accounted for, nothing duplicated.
    assert len(evicted) + len(kept) == 12
    # Oldest turns are the ones evicted.
    assert evicted[0].content == "u" * 200


# ------------------------------------------------------------------ engine flow
def _engine(num_ctx: int = 200) -> TherapistEngine:
    cfg = AppConfig()
    cfg.num_ctx = num_ctx
    cfg.prompts = {"therapist": "be calm", "context_summary": "merge summary"}
    eng = TherapistEngine(cfg, client=_SummaryClient())
    return eng


def test_compaction_below_trigger_is_noop():
    eng = _engine(num_ctx=100000)  # huge budget -> never triggers
    _fill_pairs(eng.state, 3)
    assert eng.compact_context(projected_extra_tokens=10) is None
    assert eng.client.calls == []  # no summarization attempted


def test_compaction_summarizes_evicts_and_pins_single_summary():
    eng = _engine(num_ctx=200)  # budget=100, trigger=90, keep=50
    _fill_pairs(eng.state, 8)
    result = eng.compact_context(projected_extra_tokens=10)

    assert result is not None
    assert result.evicted, "older pairs should be evicted"
    assert result.summary == "condensed summary of earlier turns"
    # Head system prompt still first and pinned.
    assert eng.state.messages[0].content == "be calm"
    assert eng.state.messages[0].pinned
    # Exactly one running-summary message, pinned.
    summaries = [m for m in eng.state.messages if m.content.startswith(SUMMARY_PREFIX)]
    assert len(summaries) == 1
    assert summaries[0].pinned
    # Summary sits in the head, before any user/assistant tail.
    first_tail = next(i for i, m in enumerate(eng.state.messages) if not m.pinned)
    assert eng.state.messages[first_tail - 1] is summaries[0]


def test_second_compaction_replaces_summary_not_duplicates():
    eng = _engine(num_ctx=200)
    _fill_pairs(eng.state, 8)
    eng.compact_context(projected_extra_tokens=10)
    _fill_pairs(eng.state, 8)
    eng.compact_context(projected_extra_tokens=10)

    summaries = [m for m in eng.state.messages if m.content.startswith(SUMMARY_PREFIX)]
    assert len(summaries) == 1
    # Second summarization saw the previous summary as prior context.
    second_call = eng.client.calls[-1]
    assert "PREVIOUS SUMMARY" in second_call[-1]["content"]


# -------------------------------------------------------------------- app wiring
def test_app_persists_one_note_then_updates(tmp_config):
    tmp_config.num_ctx = 200
    app = TherapistApp(tmp_config)
    app.engine = _engine(num_ctx=200)
    fake_nb = _FakeNotebook()
    app.notebook = fake_nb

    _fill_pairs(app.engine.state, 8)
    app._maybe_compact_context("hello there")
    assert fake_nb.writes == 1
    assert fake_nb.updates == 0
    assert app._compaction_note_id == "note-1"

    _fill_pairs(app.engine.state, 8)
    app._maybe_compact_context("hello again")
    assert fake_nb.writes == 1
    assert fake_nb.updates == 1


def test_app_compaction_disabled_skips(tmp_config):
    tmp_config.num_ctx = 200
    tmp_config.context.enabled = False
    app = TherapistApp(tmp_config)
    app.engine = _engine(num_ctx=200)
    app.engine.config.context.enabled = False
    fake_nb = _FakeNotebook()
    app.notebook = fake_nb

    _fill_pairs(app.engine.state, 8)
    app._maybe_compact_context("hello")
    assert fake_nb.writes == 0
