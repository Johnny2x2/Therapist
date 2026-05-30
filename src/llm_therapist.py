from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError

from .config import AppConfig


class UnsupportedAudioError(RuntimeError):
    """Raised when the model/endpoint rejects an attached audio payload."""


def _audio_field_name() -> str:
    # Ollama 0.24+ unifies media into the `images` field and routes by sniffing
    # the decoded bytes (WAV -> audio encoder, PNG/JPEG -> vision encoder). The
    # field is isolated here so it can be retargeted without touching the rest
    # of the payload-building code.
    return os.getenv("THERAPIST_OLLAMA_AUDIO_FIELD", "images")


def _message_has_audio(message: Dict[str, Any]) -> bool:
    field_name = _audio_field_name()
    return bool(message.get(field_name))


def _estimate_msg_tokens(role: str, content: str) -> int:
    # Rough heuristic: ~4 characters per token, plus a small per-message
    # overhead for role framing.
    return (len(content) + len(role)) // 4 + 4


def estimate_text_tokens(text: str) -> int:
    """Estimate the token cost of a bare string using the same heuristic."""
    return len(text) // 4 + 4


@dataclass
class Message:
    role: str
    content: str
    pinned: bool = False


@dataclass
class CompactionResult:
    evicted: List["Message"]
    summary: str


@dataclass
class ConversationState:
    messages: List[Message] = field(default_factory=list)

    def append(self, role: str, content: str, pinned: bool = False) -> None:
        self.messages.append(Message(role=role, content=content, pinned=pinned))

    def as_payload(self) -> List[Dict[str, str]]:
        return [{"role": item.role, "content": item.content} for item in self.messages]

    def estimate_tokens(self) -> int:
        # Rough heuristic: ~4 characters per token, plus a small per-message
        # overhead for role framing. Good enough for a usage gauge.
        total = 0
        for item in self.messages:
            total += _estimate_msg_tokens(item.role, item.content)
        return total

    def estimate_tail_tokens(self) -> int:
        """Estimated tokens of the non-pinned (evictable) tail only."""
        total = 0
        for item in self.messages:
            if item.pinned:
                continue
            total += _estimate_msg_tokens(item.role, item.content)
        return total

    def split_tail_for_eviction(
        self, keep_tokens: int
    ) -> Tuple[List[Message], List[Message]]:
        """Group the non-pinned tail into consecutive (user, assistant) pairs and
        keep the most recent pairs that fit under ``keep_tokens``.

        Returns ``(evicted, kept)`` where ``evicted`` are the oldest messages to
        roll into the summary and ``kept`` are the most recent messages to retain.
        The most recent pair is always kept even if it exceeds the budget.
        """
        tail = [m for m in self.messages if not m.pinned]
        if not tail:
            return [], []

        # Build pairs from the tail in order. A pair is one or more leading
        # messages followed by messages up to (and including) the next assistant.
        pairs: List[List[Message]] = []
        current: List[Message] = []
        for msg in tail:
            current.append(msg)
            if msg.role == "assistant":
                pairs.append(current)
                current = []
        if current:
            pairs.append(current)

        # Walk newest -> oldest, keeping pairs while under budget.
        kept_rev: List[List[Message]] = []
        used = 0
        for pair in reversed(pairs):
            cost = sum(_estimate_msg_tokens(m.role, m.content) for m in pair)
            if kept_rev and used + cost > keep_tokens:
                break
            kept_rev.append(pair)
            used += cost
        kept_pairs = list(reversed(kept_rev))
        kept_count = len(kept_pairs)
        evicted_pairs = pairs[: len(pairs) - kept_count]

        evicted = [m for pair in evicted_pairs for m in pair]
        kept = [m for pair in kept_pairs for m in pair]
        return evicted, kept

    def apply_compaction(
        self, kept_tail: List[Message], summary_message: Message
    ) -> None:
        """Rebuild the message list as pinned-head + single summary + kept tail."""
        head = [m for m in self.messages if m.pinned and m is not summary_message]
        self.messages = head + [summary_message] + list(kept_tail)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolChatResult:
    content: str
    tool_calls: List[ToolCall]


class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config

    # ------------------------------------------------------------ audio
    @staticmethod
    def attach_audio(
        message: Dict[str, Any],
        audio_path: str,
        mime_type: str = "audio/wav",
    ) -> Dict[str, Any]:
        """Return a copy of ``message`` with a base64-encoded audio clip attached.

        The original text content is preserved so the model still receives the
        Whisper transcript alongside the raw audio.
        """
        del mime_type  # Ollama infers type from the decoded bytes.
        data = Path(audio_path).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        enriched = dict(message)
        enriched[_audio_field_name()] = [encoded]
        return enriched

    # ------------------------------------------------------------ chat (stream)
    def stream_chat(
        self,
        messages: Iterable[Dict[str, Any]],
        model: Optional[str] = None,
        fmt: Optional[str] = None,
    ) -> Iterator[str]:
        messages = list(messages)
        has_audio = any(_message_has_audio(item) for item in messages)
        body: Dict[str, Any] = {
            "model": model or self.config.models.therapist_model,
            "stream": True,
            "keep_alive": self.config.keep_alive,
            "messages": messages,
            "options": {"num_ctx": self.config.num_ctx},
        }
        if fmt:
            body["format"] = fmt
        req = self._post("/api/chat", body)
        try:
            response = request.urlopen(req)
        except (HTTPError, URLError) as exc:
            if has_audio:
                raise UnsupportedAudioError(str(exc)) from exc
            raise
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    if has_audio:
                        raise UnsupportedAudioError(str(chunk.get("error")))
                    raise RuntimeError(str(chunk.get("error")))
                message = chunk.get("message") or {}
                content = message.get("content")
                if content:
                    yield content

    # ------------------------------------------------------------ chat (block)
    def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        model: Optional[str] = None,
        fmt: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model or self.config.models.therapist_model,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "messages": list(messages),
            "options": {"num_ctx": self.config.num_ctx},
        }
        if fmt:
            body["format"] = fmt
        req = self._post("/api/chat", body)
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    # ------------------------------------------------------------ tools
    def chat_with_tools(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ToolChatResult:
        body: Dict[str, Any] = {
            "model": model or self.config.models.librarian_model,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "messages": list(messages),
            "tools": tools,
        }
        req = self._post("/api/chat", body)
        with request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        calls: List[ToolCall] = []
        for entry in raw_calls:
            fn = entry.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if name:
                calls.append(ToolCall(name=name, arguments=dict(args)))
        return ToolChatResult(content=str(message.get("content") or ""), tool_calls=calls)

    def run_tool_loop(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        executors: Dict[str, Callable[..., Any]],
        model: Optional[str] = None,
        max_iterations: int = 6,
        timeout: Optional[float] = None,
        on_tool: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
    ) -> ToolChatResult:
        history = list(messages)
        last = ToolChatResult(content="", tool_calls=[])
        for _ in range(max_iterations):
            last = self.chat_with_tools(history, tools=tools, model=model, timeout=timeout)
            if not last.tool_calls:
                return last
            # Echo the assistant tool-call turn so the model has its own history.
            history.append({
                "role": "assistant",
                "content": last.content or "",
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}}
                    for c in last.tool_calls
                ],
            })
            for call in last.tool_calls:
                executor = executors.get(call.name)
                if executor is None:
                    result: Any = {"error": f"unknown tool: {call.name}"}
                else:
                    try:
                        result = executor(**call.arguments)
                    except TypeError as exc:
                        result = {"error": f"bad arguments: {exc}"}
                    except Exception as exc:  # noqa: BLE001 - surface to model
                        result = {"error": str(exc)}
                if on_tool is not None:
                    try:
                        on_tool(call.name, call.arguments, result)
                    except Exception:
                        pass
                history.append({
                    "role": "tool",
                    "name": call.name,
                    "content": json.dumps(result, default=str),
                })
        return last

    # ------------------------------------------------------------ embeddings
    def embed(self, text: str) -> List[float]:
        body = {
            "model": self.config.models.embedding_model,
            "input": text,
            "keep_alive": self.config.keep_alive,
        }
        req = self._post("/api/embed", body)
        with request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
        embeddings = data.get("embeddings") or []
        if embeddings:
            return embeddings[0]
        raise RuntimeError("Ollama embedding response did not include an embedding vector.")

    # ------------------------------------------------------------ helpers
    def _post(self, path: str, body: Dict[str, Any]) -> request.Request:
        payload = json.dumps(body).encode("utf-8")
        return request.Request(
            url=self.config.ollama_host.rstrip("/") + path,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )


class TherapistEngine:
    def __init__(self, config: AppConfig, client: Optional[OllamaClient] = None):
        self.config = config
        self.client = client or OllamaClient(config)
        self.state = ConversationState(
            messages=[Message(role="system", content=config.prompts["therapist"], pinned=True)]
        )
        self._summary_message: Optional[Message] = None
        self._rolling_summary: str = ""

    def add_memory_context(self, snippets: Iterable[str]) -> None:
        content = "\n".join(item for item in snippets if item)
        if content:
            self.state.append("system", "Relevant past session context:\n" + content, pinned=True)

    def add_pinned_notes(self, snippets: Iterable[str]) -> None:
        content = "\n\n---\n\n".join(item for item in snippets if item)
        if content:
            self.state.append(
                "system",
                "The user's pinned reference notes (always relevant):\n" + content,
                pinned=True,
            )

    # ------------------------------------------------------------ compaction
    def _format_summary(self, text: str) -> str:
        return (
            "Running summary of earlier conversation (older turns were condensed "
            "to save context):\n" + text
        )

    def _ensure_summary_message(self, text: str) -> Message:
        self._rolling_summary = text
        content = self._format_summary(text)
        if self._summary_message is None:
            self._summary_message = Message(role="system", content=content, pinned=True)
        else:
            self._summary_message.content = content
            self._summary_message.pinned = True
        return self._summary_message

    def _summarize_rolling(self, prior: str, evicted: List[Message]) -> str:
        exchanges = "\n".join(f"{m.role}: {m.content}" for m in evicted)
        user_block = (
            "PREVIOUS SUMMARY:\n" + (prior or "(none)") + "\n\n"
            "NEW EXCHANGES:\n" + exchanges
        )
        messages = [
            {"role": "system", "content": self.config.prompts["context_summary"]},
            {"role": "user", "content": user_block},
        ]
        fragments: List[str] = []
        for token in self.client.stream_chat(
            messages, model=self.config.models.therapist_model
        ):
            fragments.append(token)
        return "".join(fragments).strip()

    def compact_context(self, projected_extra_tokens: int = 0) -> Optional[CompactionResult]:
        """Condense the oldest user/assistant pairs into a running summary when
        the live tail exceeds the configured budget. Returns the result, or None
        when no compaction was needed or possible."""
        ctx = self.config.context
        if not ctx.enabled:
            return None
        budget = int(self.config.num_ctx * ctx.budget_ratio)
        if budget <= 0:
            return None
        trigger = int(budget * ctx.trigger_ratio)
        keep = int(budget * ctx.keep_ratio)
        projected = self.state.estimate_tail_tokens() + max(0, projected_extra_tokens)
        if projected < trigger:
            return None
        evicted, kept = self.state.split_tail_for_eviction(keep)
        if not evicted:
            return None
        summary = self._summarize_rolling(self._rolling_summary, evicted)
        if not summary:
            return None
        summary_message = self._ensure_summary_message(summary)
        self.state.apply_compaction(kept, summary_message)
        return CompactionResult(evicted=evicted, summary=summary)

    def generate_reply(
        self,
        user_text: str,
        transient_context: Optional[Iterable[str]] = None,
        audio_path: Optional[str] = None,
        audio_mime_type: str = "audio/wav",
        audio_duration_seconds: float = 0.0,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        self.state.append("user", user_text)
        payload = self.state.as_payload()
        if transient_context:
            extras = [item for item in transient_context if item]
            if extras:
                # Insert just before the latest user turn so the model sees it as
                # immediately relevant context for this turn only.
                insert_at = max(0, len(payload) - 1)
                payload.insert(insert_at, {
                    "role": "system",
                    "content": "Possibly relevant notes for this turn:\n"
                               + "\n\n---\n\n".join(extras),
                })

        use_audio = bool(audio_path) and self.config.audio.send_audio_to_model
        if use_audio:
            max_s = self.config.audio.model_audio_max_seconds
            if audio_duration_seconds and max_s and audio_duration_seconds > max_s:
                # Clip exceeds the model's audio window; stay transcript-only.
                use_audio = False
                if on_status:
                    on_status(
                        f"Audio clip {audio_duration_seconds:.0f}s exceeds "
                        f"{max_s:.0f}s limit; sending transcript only."
                    )

        audio_payload = payload
        if use_audio:
            audio_payload = list(payload)
            # Attach the clip to the latest user turn only.
            audio_payload[-1] = OllamaClient.attach_audio(
                audio_payload[-1], audio_path, audio_mime_type,
            )

        fragments: List[str] = []
        try:
            for token in self.client.stream_chat(audio_payload):
                fragments.append(token)
                yield token
        except UnsupportedAudioError:
            if not (use_audio and self.config.audio.audio_fallback_text):
                raise
            if fragments:
                # Tokens already streamed; cannot cleanly restart this turn.
                raise
            if on_status:
                on_status("Model rejected audio; retrying with transcript only.")
            for token in self.client.stream_chat(payload):
                fragments.append(token)
                yield token

        reply = "".join(fragments).strip()
        if reply:
            self.state.append("assistant", reply)

