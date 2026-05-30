from __future__ import annotations

from unittest.mock import MagicMock

from src.librarian import Librarian
from src.llm_therapist import ToolCall, ToolChatResult
from src.notebook import NotebookStore


def test_consolidate_runs_write_note_with_stub(tmp_config):
    # Force librarian on for this test only.
    tmp_config.librarian.enabled = True

    from tests.conftest import FakeClient
    embed_client = FakeClient()
    store = NotebookStore(tmp_config, client=embed_client)

    client = MagicMock()
    # Reuse the real embed implementation.
    client.embed.side_effect = embed_client.embed

    calls = iter([
        ToolChatResult(
            content="",
            tool_calls=[ToolCall(name="search_notes", arguments={"query": "manager stress"})],
        ),
        ToolChatResult(
            content="",
            tool_calls=[ToolCall(
                name="write_note",
                arguments={
                    "category": "triggers",
                    "title": "Stress with manager",
                    "body": "## Observations\nFelt dismissed in 1:1.",
                    "tags": ["work", "manager"],
                },
            )],
        ),
        ToolChatResult(content="Created one note about manager stress.", tool_calls=[]),
    ])

    def fake_chat_with_tools(messages, tools, model=None, timeout=None):
        return next(calls)
    client.chat_with_tools.side_effect = fake_chat_with_tools

    # Stitch the same loop logic that OllamaClient uses.
    from src.llm_therapist import OllamaClient
    real_loop = OllamaClient.run_tool_loop
    client.run_tool_loop.side_effect = lambda **kw: real_loop(client, **kw)

    librarian = Librarian(tmp_config, client, store)
    transcript = [
        {"role": "user", "content": "My manager dismissed my idea again."},
        {"role": "assistant", "content": "That sounds frustrating."},
    ]
    report = librarian.consolidate("sess1", transcript)
    assert report.error is None
    tools_used = [op["tool"] for op in report.operations]
    assert "search_notes" in tools_used
    assert "write_note" in tools_used
    listing = store.list_notes()
    assert any("Stress with manager" == m["title"] for m in listing)
