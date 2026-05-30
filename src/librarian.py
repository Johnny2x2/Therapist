from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import AppConfig
from .llm_therapist import OllamaClient
from .notebook import NotebookStore


# ------------------------------------------------------------ tool schemas
def _tool_schemas(categories: List[str]) -> List[Dict[str, Any]]:
    enum_cats = list(categories)
    return [
        {
            "type": "function",
            "function": {
                "name": "list_notes",
                "description": "List notes, optionally filtered by category, tag, or pinned status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "Optional category filter."},
                        "tag": {"type": "string", "description": "Optional tag filter."},
                        "pinned_only": {"type": "boolean", "default": False},
                        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_note",
                "description": "Return the full body of a note by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"note_id": {"type": "string"}},
                    "required": ["note_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_notes",
                "description": "Hybrid (vector + keyword) search across all notes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                        "category": {"type": "string"},
                        "tag": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_note",
                "description": "Create a new note. Always search_notes first; prefer update_note over duplicating.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": enum_cats},
                        "title": {"type": "string"},
                        "body": {"type": "string", "description": "Markdown body. Use conventional sections."},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "pinned": {"type": "boolean", "default": False},
                        "related": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["category", "title", "body"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_note",
                "description": "Modify an existing note. Prefer append for incremental observations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note_id": {"type": "string"},
                        "append": {"type": "string", "description": "Markdown to append (will be added with a newline)."},
                        "replace_body": {"type": "string"},
                        "set_tags": {"type": "array", "items": {"type": "string"}},
                        "set_pinned": {"type": "boolean"},
                        "add_related": {"type": "array", "items": {"type": "string"}},
                        "set_title": {"type": "string"},
                    },
                    "required": ["note_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "link_notes",
                "description": "Link two notes bidirectionally.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a_id": {"type": "string"},
                        "b_id": {"type": "string"},
                    },
                    "required": ["a_id", "b_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_note",
                "description": "Soft-delete a note (moves it to trash). Use rarely.",
                "parameters": {
                    "type": "object",
                    "properties": {"note_id": {"type": "string"}},
                    "required": ["note_id"],
                },
            },
        },
    ]


@dataclass
class LibrarianRunReport:
    operations: List[Dict[str, Any]] = field(default_factory=list)
    surfaced_note_ids: List[str] = field(default_factory=list)
    final_message: str = ""
    skipped: bool = False
    error: Optional[str] = None


class Librarian:
    """Background notebook agent driven by an Ollama tool-calling model."""

    def __init__(self, config: AppConfig, client: OllamaClient, store: NotebookStore):
        self.config = config
        self.client = client
        self.store = store
        self._new_note_count = 0

    # ------------------------------------------------------------ entrypoints
    def warmup(self, session_id: str, seed_text: str) -> LibrarianRunReport:
        if not self.config.librarian.enabled:
            return LibrarianRunReport(skipped=True)
        index = self.store.list_notes(limit=30)
        index_blurb = "\n".join(
            f"- {m['id']} [{m['category']}] {m['title']} (tags: {', '.join(m.get('tags') or []) or '-'})"
            for m in index
        ) or "(notebook is empty)"
        messages = self._base_messages() + [{
            "role": "user",
            "content": (
                f"Session {session_id} is about to start.\n"
                f"Seed topic: {seed_text!r}\n\n"
                f"Existing notes (id [category] title):\n{index_blurb}\n\n"
                "Use search_notes to find up to 3 notes that are most relevant for this "
                "session's opening. Do NOT create or modify notes. When done, reply with "
                "a one-line summary that lists the ids you chose, e.g. "
                "'surfaced: <id1>, <id2>'."
            ),
        }]
        return self._run(messages, session_id, allow_writes=False)

    def per_turn(self, session_id: str, user_text: str, recent_turns: List[str]) -> LibrarianRunReport:
        if not self.config.librarian.enabled or not self.config.librarian.per_turn_enabled:
            return LibrarianRunReport(skipped=True)
        recent_blurb = "\n".join(recent_turns[-6:]) or "(no prior turns this session)"
        messages = self._base_messages() + [{
            "role": "user",
            "content": (
                f"Session {session_id}, mid-conversation.\n"
                f"Recent turns:\n{recent_blurb}\n\n"
                f"New user message:\n{user_text}\n\n"
                "If (and only if) there is an obviously relevant existing note, call "
                "search_notes once and reply with 'surfaced: <ids>'. Do NOT write or "
                "update anything. If nothing is clearly relevant, reply 'surfaced: none'."
            ),
        }]
        return self._run(
            messages,
            session_id,
            allow_writes=False,
            timeout=self.config.librarian.per_turn_timeout_s,
            max_iterations=3,
        )

    def consolidate(self, session_id: str, transcript: List[Dict[str, Any]]) -> LibrarianRunReport:
        if not self.config.librarian.enabled:
            return LibrarianRunReport(skipped=True)
        transcript_text = _format_transcript(transcript)
        index = self.store.list_notes(limit=50)
        index_blurb = "\n".join(
            f"- {m['id']} [{m['category']}] {m['title']} (tags: {', '.join(m.get('tags') or []) or '-'})"
            for m in index
        ) or "(notebook is empty)"
        messages = self._base_messages() + [{
            "role": "user",
            "content": (
                f"Session {session_id} just ended.\n\n"
                f"Existing notes (search first; prefer update over duplicate):\n{index_blurb}\n\n"
                f"Full transcript:\n{transcript_text}\n\n"
                "Decide what to file. You may:\n"
                "- write_note for genuinely new themes (max "
                f"{self.config.librarian.max_new_notes_per_session} new notes this session).\n"
                "- update_note to append observations to relevant existing notes.\n"
                "- link_notes to connect related entries.\n"
                "Always search_notes before write_note. When finished, reply with a brief "
                "one-sentence summary of what you changed."
            ),
        }]
        return self._run(messages, session_id, allow_writes=True)

    # ------------------------------------------------------------ runner
    def _run(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        allow_writes: bool,
        timeout: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> LibrarianRunReport:
        self._new_note_count = 0
        report = LibrarianRunReport()
        tools = _tool_schemas(self.config.notebook_categories)
        executors = self._build_executors(allow_writes, report, session_id)
        on_tool: Callable[[str, Dict[str, Any], Any], None] = lambda name, args, result: (
            self._record_op(report, session_id, name, args, result)
        )
        try:
            result = self.client.run_tool_loop(
                messages=messages,
                tools=tools,
                executors=executors,
                model=self.config.models.librarian_model,
                max_iterations=max_iterations or self.config.librarian.max_tool_calls,
                timeout=timeout,
                on_tool=on_tool,
            )
            report.final_message = result.content
        except Exception as exc:
            report.error = f"{type(exc).__name__}: {exc}"
        # If any search returned ids, expose them for the caller (warmup / per_turn).
        for op in report.operations:
            if op.get("tool") == "search_notes":
                for hit in op.get("result") or []:
                    nid = hit.get("id")
                    if nid and nid not in report.surfaced_note_ids:
                        report.surfaced_note_ids.append(nid)
        self._write_audit(session_id, report)
        return report

    def _base_messages(self) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": self.config.prompts["librarian"]}]

    def _build_executors(
        self,
        allow_writes: bool,
        report: LibrarianRunReport,
        session_id: str,
    ) -> Dict[str, Callable[..., Any]]:
        store = self.store
        dry = self.config.librarian.dry_run
        limit_new = self.config.librarian.max_new_notes_per_session

        def list_notes(**kw):
            return store.list_notes(**kw)

        def read_note(note_id: str):
            note = store.read_note(note_id)
            if note is None:
                return {"error": "not found"}
            return note.to_dict(include_body=True)

        def search_notes(**kw):
            return store.search_notes(**kw)

        def write_note(**kw):
            if not allow_writes:
                return {"error": "write_note not permitted in this run"}
            if self._new_note_count >= limit_new:
                return {"error": f"new-note limit reached ({limit_new}) for this session"}
            if dry:
                return {"dry_run": True, "intent": "write_note", "args": kw}
            kw.setdefault("source_session", session_id)
            out = store.write_note(**kw)
            self._new_note_count += 1
            return out

        def update_note(**kw):
            if not allow_writes:
                return {"error": "update_note not permitted in this run"}
            if dry:
                return {"dry_run": True, "intent": "update_note", "args": kw}
            out = store.update_note(**kw)
            if out is None:
                return {"error": "not found"}
            return out

        def link_notes(**kw):
            if not allow_writes:
                return {"error": "link_notes not permitted in this run"}
            if dry:
                return {"dry_run": True, "intent": "link_notes", "args": kw}
            return store.link_notes(**kw)

        def delete_note(**kw):
            if not allow_writes:
                return {"error": "delete_note not permitted in this run"}
            if dry:
                return {"dry_run": True, "intent": "delete_note", "args": kw}
            return {"deleted": store.delete_note(**kw)}

        return {
            "list_notes": list_notes,
            "read_note": read_note,
            "search_notes": search_notes,
            "write_note": write_note,
            "update_note": update_note,
            "link_notes": link_notes,
            "delete_note": delete_note,
        }

    def _record_op(
        self,
        report: LibrarianRunReport,
        session_id: str,
        name: str,
        args: Dict[str, Any],
        result: Any,
    ) -> None:
        report.operations.append({
            "tool": name,
            "args": args,
            "result": result,
        })

    def _write_audit(self, session_id: str, report: LibrarianRunReport) -> None:
        try:
            path: Path = self.config.librarian_log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "session_id": session_id,
                    "skipped": report.skipped,
                    "error": report.error,
                    "final_message": report.final_message,
                    "operations": report.operations,
                }, default=str) + "\n")
        except Exception:
            pass


def _format_transcript(transcript: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in transcript:
        role = entry.get("role", "?")
        content = entry.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
