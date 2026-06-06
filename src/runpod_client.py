"""Client that drives a Therapist engine deployed on RunPod Serverless.

Implements the same high-level surface as ``TherapistApp`` (``run_once``,
``warm_memory``, ``record_mood``, ``end_session``, ``engine``) so the desktop UI
can use it interchangeably. Speech capture (STT) and audio playback stay local;
the LLM and optional server-side TTS run in the RunPod worker.

Configuration (env):
- ``THERAPIST_RUNPOD_ENDPOINT_ID``: the serverless endpoint id.
- ``THERAPIST_RUNPOD_API_KEY``: RunPod API key (Bearer auth for the job API).
- ``THERAPIST_API_KEY``: app-level key forwarded to the handler (optional).
- ``THERAPIST_RUNPOD_BASE``: override base URL (default https://api.runpod.ai/v2).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from typing import Callable, Optional

import requests

from .config import AppConfig

logger = logging.getLogger(__name__)


class RunPodTherapistApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.endpoint_id = os.getenv("THERAPIST_RUNPOD_ENDPOINT_ID", "").strip()
        self.runpod_key = os.getenv("THERAPIST_RUNPOD_API_KEY", "").strip()
        self.app_key = os.getenv("THERAPIST_API_KEY", "").strip()
        base = os.getenv("THERAPIST_RUNPOD_BASE", "https://api.runpod.ai/v2").rstrip("/")
        self.base_url = f"{base}/{self.endpoint_id}"
        if not self.endpoint_id or not self.runpod_key:
            raise ValueError(
                "RunPod client requires THERAPIST_RUNPOD_ENDPOINT_ID and "
                "THERAPIST_RUNPOD_API_KEY environment variables."
            )

        # Local speech I/O still runs on the client machine.
        from .stt_listener import SpeechListener
        from .tts_speaker import TextSpeaker
        self.listener = SpeechListener(config)
        self.speaker = TextSpeaker(config)
        self.session_id: str = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # ------------------------------------------------------------------ helpers
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.runpod_key}",
            "Content-Type": "application/json",
        }

    def warm_memory(self, seed_text: str) -> None:
        # Memory warmup happens server-side on first turn; nothing to do here.
        return None

    def record_mood(self, phase: str, value: int) -> None:
        return None

    def end_session(self) -> None:
        return None

    # ------------------------------------------------------------------ chat
    def run_once(
        self,
        user_text: str,
        speak: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        emit_console: bool = True,
        audio_path: Optional[str] = None,
        audio_mime_type: str = "audio/wav",
        audio_duration_seconds: float = 0.0,
    ) -> str:
        payload = {
            "input": {
                "session_id": self.session_id,
                "user_text": user_text,
                "speak": speak,
            }
        }
        if self.app_key:
            payload["input"]["api_key"] = self.app_key

        run = requests.post(
            f"{self.base_url}/run", json=payload, headers=self._headers(), timeout=30,
        )
        run.raise_for_status()
        job_id = run.json().get("id")
        if not job_id:
            raise RuntimeError(f"RunPod did not return a job id: {run.text[:200]}")

        full_reply: list[str] = []
        seen = 0
        stream_url = f"{self.base_url}/stream/{job_id}"
        while True:
            resp = requests.get(stream_url, headers=self._headers(), timeout=60)
            resp.raise_for_status()
            body = resp.json()
            for item in body.get("stream", []):
                output = item.get("output", item)
                self._dispatch(output, speak, on_token, on_status, full_reply)
            seen += len(body.get("stream", []))
            status = body.get("status")
            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                if status != "COMPLETED" and on_status:
                    on_status(f"Job {status.lower()}")
                break
            time.sleep(0.2)
        return "".join(full_reply)

    def _dispatch(self, output, speak, on_token, on_status, full_reply) -> None:
        if not isinstance(output, dict):
            return
        mtype = output.get("type")
        content = output.get("content", "")
        if mtype == "status" and on_status:
            on_status(content)
        elif mtype == "text":
            full_reply.append(content)
            if on_token:
                on_token(content)
        elif mtype == "audio" and speak:
            b64 = output.get("data", "")
            if b64:
                self._play_b64(b64)
        elif mtype == "error" and on_status:
            on_status(f"API Error: {content}")

    def _play_b64(self, b64: str) -> None:
        import threading

        def _play() -> None:
            try:
                import base64
                import io
                import wave
                import numpy as np
                audio_bytes = base64.b64decode(b64)
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    sr = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                self.speaker.play(data, sr)
            except Exception as exc:  # noqa: BLE001 - playback is non-fatal
                logger.error("Client audio error: %s", exc)

        threading.Thread(target=_play, daemon=True).start()

    class _MockEngine:
        class _MockState:
            def estimate_tokens(self):
                return 0

        state = _MockState()

    @property
    def engine(self):
        return self._MockEngine()
