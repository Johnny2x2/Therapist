from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from urllib import request

from .config import AppConfig


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ConversationState:
    messages: List[Message] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def as_payload(self) -> List[Dict[str, str]]:
        return [{"role": item.role, "content": item.content} for item in self.messages]

    def estimate_tokens(self) -> int:
        # Rough heuristic: ~4 characters per token, plus a small per-message
        # overhead for role framing. Good enough for a usage gauge.
        total = 0
        for item in self.messages:
            total += (len(item.content) + len(item.role)) // 4 + 4
        return total



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

    # ------------------------------------------------------------ chat (stream)
    def stream_chat(
        self,
        messages: Iterable[Dict[str, str]],
        model: Optional[str] = None,
        fmt: Optional[str] = None,
    ) -> Iterator[str]:
        body: Dict[str, Any] = {
            "model": model or self.config.models.therapist_model,
            "stream": True,
            "keep_alive": self.config.keep_alive,
            "messages": list(messages),
            "options": {"num_ctx": self.config.num_ctx},
        }
        if fmt:
            body["format"] = fmt
        req = self._post("/api/chat", body)
        with request.urlopen(req) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
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
            messages=[Message(role="system", content=config.prompts["therapist"])]
        )

    def add_memory_context(self, snippets: Iterable[str]) -> None:
        content = "\n".join(item for item in snippets if item)
        if content:
            self.state.append("system", "Relevant past session context:\n" + content)

    def add_pinned_notes(self, snippets: Iterable[str]) -> None:
        content = "\n\n---\n\n".join(item for item in snippets if item)
        if content:
            self.state.append(
                "system",
                "The user's pinned reference notes (always relevant):\n" + content,
            )

    def generate_reply(
        self,
        user_text: str,
        transient_context: Optional[Iterable[str]] = None,
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
        fragments: List[str] = []
        for token in self.client.stream_chat(payload):
            fragments.append(token)
            yield token
        reply = "".join(fragments).strip()
        if reply:
            self.state.append("assistant", reply)
