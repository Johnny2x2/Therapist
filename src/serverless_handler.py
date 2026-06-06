"""RunPod serverless handler for the Therapist engine.

Exposes a single streaming handler that runs one therapist turn and yields
ndjson-style events: ``status``, ``text`` tokens, optional ``audio`` (base64
WAV, server-side TTS), and ``error``.

The heavy lifting (Ollama LLM, safety, memory, notebook, TTS) reuses the same
``TherapistApp`` used by the local CLI/GUI. Long-term memory persists on the
mounted network volume via ``THERAPIST_DATA_DIR``; live in-session context is
only retained while a worker stays warm.

The ``runpod`` SDK is imported lazily so the core ``iter_turn`` generator can be
unit-tested without it installed.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import queue
import threading
import wave
from typing import Any, Dict, Iterator

import numpy as np

from .config import AppConfig
from .main import TherapistApp, _drain_complete_paragraphs

logger = logging.getLogger(__name__)

# Reuse warm TherapistApp instances per session while the worker is alive.
_SESSIONS: Dict[str, TherapistApp] = {}
_SESSIONS_LOCK = threading.Lock()


def _get_app(session_id: str) -> TherapistApp:
    with _SESSIONS_LOCK:
        app = _SESSIONS.get(session_id)
        if app is None:
            logger.info("Creating TherapistApp for session %s", session_id)
            app = TherapistApp(AppConfig.load())
            # Resume any persisted transcript for this session (survives cold
            # workers); fall back to a fresh warmup when there's nothing to load.
            resumed = app.restore_session(session_id)
            logger.info("Session %s resumed=%s", session_id, resumed)
            _SESSIONS[session_id] = app
        return app


def _encode_wav(data: "np.ndarray", sample_rate: int) -> str:
    pcm = np.asarray(data)
    if pcm.ndim > 1:
        pcm = pcm.reshape(pcm.shape[0], -1)[:, 0]
    pcm = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _check_auth(payload: Dict[str, Any]) -> bool:
    expected = os.getenv("THERAPIST_API_KEY", "")
    if not expected:
        return True  # auth disabled when no key configured
    provided = payload.get("api_key") or payload.get("apiKey") or ""
    return provided == expected


def iter_turn(payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Run one therapist turn and yield streaming event dicts.

    ``payload`` keys: ``session_id`` (str), ``user_text`` (str),
    ``speak`` (bool, optional), ``api_key`` (str, optional).
    """
    if not _check_auth(payload):
        yield {"type": "error", "content": "unauthorized"}
        return

    session_id = str(payload.get("session_id") or "default")
    user_text = payload.get("user_text") or ""
    speak = bool(payload.get("speak", False))
    if not user_text.strip():
        yield {"type": "error", "content": "user_text is required"}
        return

    app = _get_app(session_id)
    sync_q: "queue.Queue[Any]" = queue.Queue()

    def on_token(token: str) -> None:
        sync_q.put({"type": "text", "content": token})

    def on_status(status: str) -> None:
        sync_q.put({"type": "status", "content": status})

    def synth_and_emit(text: str) -> None:
        if not speak or not text.strip():
            return
        try:
            result = app.speaker.synthesize(text)
            if result is None:
                return
            data, sr = result
            sync_q.put({"type": "audio", "data": _encode_wav(data, sr), "text": text})
        except Exception as exc:  # noqa: BLE001 - TTS is non-fatal
            logger.error("TTS synth error: %s", exc)

    class _Streamer:
        def __init__(self) -> None:
            self._buffer = ""

        def push_token(self, token: str) -> None:
            on_token(token)
            ready, self._buffer = _drain_complete_paragraphs(self._buffer + token)
            for chunk in ready:
                synth_and_emit(chunk)

        def finish(self) -> None:
            tail = self._buffer.strip()
            if tail:
                synth_and_emit(tail)

    streamer = _Streamer()

    def run_turn() -> None:
        try:
            app._handle_turn(
                user_text=user_text,
                speak=False,
                on_token=streamer.push_token,
                on_status=on_status,
                emit_console=False,
            )
            streamer.finish()
        except Exception as exc:  # noqa: BLE001 - surface to client
            logger.exception("Turn failed")
            sync_q.put({"type": "error", "content": str(exc)})
        finally:
            sync_q.put(None)

    worker = threading.Thread(target=run_turn, daemon=True)
    worker.start()

    while True:
        msg = sync_q.get()
        if msg is None:
            break
        yield msg


def handler(event: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """RunPod serverless entrypoint. ``event['input']`` holds the payload."""
    payload = (event or {}).get("input") or {}
    yield from iter_turn(payload)


if __name__ == "__main__":
    import runpod  # type: ignore

    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
