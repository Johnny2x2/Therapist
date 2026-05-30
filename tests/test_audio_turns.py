from __future__ import annotations

import base64
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.llm_therapist import (
    OllamaClient,
    TherapistEngine,
    UnsupportedAudioError,
    _audio_field_name,
    _message_has_audio,
)


def _write_wav(path: Path) -> None:
    # Minimal RIFF/WAVE header + a couple of silent sample frames.
    path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 24)


# --------------------------------------------------------------------- serializer
def test_attach_audio_encodes_wav_and_keeps_text(tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    message = {"role": "user", "content": "I feel anxious"}
    enriched = OllamaClient.attach_audio(message, str(wav))

    field = _audio_field_name()
    assert enriched["content"] == "I feel anxious"
    assert field in enriched
    decoded = base64.b64decode(enriched[field][0])
    assert decoded == wav.read_bytes()
    # Original message is untouched.
    assert field not in message


def test_message_has_audio_detection():
    field = _audio_field_name()
    assert _message_has_audio({"role": "user", "content": "x", field: ["abc"]})
    assert not _message_has_audio({"role": "user", "content": "x"})


# --------------------------------------------------------------------- engine
class _FakeAudioClient:
    """Records the payload it was streamed and yields canned tokens."""

    def __init__(self, fail_on_audio: bool = False):
        self.fail_on_audio = fail_on_audio
        self.calls: List[List[Dict[str, Any]]] = []

    def stream_chat(self, messages, model=None, fmt=None):
        messages = list(messages)
        self.calls.append(messages)
        if self.fail_on_audio and any(_message_has_audio(m) for m in messages):
            raise UnsupportedAudioError("audio not supported")
        yield "ok "
        yield "reply"


def _make_engine(tmp_config, client):
    engine = TherapistEngine(tmp_config, client)
    return engine


def test_generate_reply_attaches_audio_to_latest_turn_only(tmp_config, tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    tmp_config.audio.send_audio_to_model = True
    client = _FakeAudioClient()
    engine = _make_engine(tmp_config, client)

    tokens = list(engine.generate_reply("hello there", audio_path=str(wav)))
    assert "".join(tokens) == "ok reply"

    sent = client.calls[0]
    field = _audio_field_name()
    # Only the final (latest user) message carries audio.
    assert field in sent[-1]
    assert all(field not in m for m in sent[:-1])

    # Conversation state stays text-only (no base64 blobs retained).
    for msg in engine.state.as_payload():
        assert field not in msg
        assert "audios" not in msg


def test_generate_reply_falls_back_to_text_when_audio_rejected(tmp_config, tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    tmp_config.audio.send_audio_to_model = True
    client = _FakeAudioClient(fail_on_audio=True)
    engine = _make_engine(tmp_config, client)

    tokens = list(engine.generate_reply("hello", audio_path=str(wav)))
    assert "".join(tokens) == "ok reply"

    # First attempt had audio, retry was transcript-only.
    field = _audio_field_name()
    assert _message_has_audio(client.calls[0][-1])
    assert not _message_has_audio(client.calls[1][-1])


def test_generate_reply_raises_when_fallback_disabled(tmp_config, tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    tmp_config.audio.send_audio_to_model = True
    tmp_config.audio.audio_fallback_text = False
    client = _FakeAudioClient(fail_on_audio=True)
    engine = _make_engine(tmp_config, client)

    with pytest.raises(UnsupportedAudioError):
        list(engine.generate_reply("hello", audio_path=str(wav)))


def test_long_clip_skips_audio(tmp_config, tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    tmp_config.audio.model_audio_max_seconds = 30
    client = _FakeAudioClient()
    engine = _make_engine(tmp_config, client)

    list(engine.generate_reply("hello", audio_path=str(wav), audio_duration_seconds=45.0))
    assert not _message_has_audio(client.calls[0][-1])


def test_audio_disabled_skips_attachment(tmp_config, tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    tmp_config.audio.send_audio_to_model = False
    client = _FakeAudioClient()
    engine = _make_engine(tmp_config, client)

    list(engine.generate_reply("hello", audio_path=str(wav)))
    assert not _message_has_audio(client.calls[0][-1])


# --------------------------------------------------------------------- app wiring
class _StubEngine:
    def __init__(self):
        self.received: Dict[str, Any] = {}

    def generate_reply(self, user_text, transient_context=None, audio_path=None,
                       audio_mime_type="audio/wav", audio_duration_seconds=0.0,
                       on_status=None):
        self.received = {
            "user_text": user_text,
            "audio_path": audio_path,
            "audio_duration_seconds": audio_duration_seconds,
        }
        yield "reply"


def test_run_once_passes_audio_into_engine(tmp_config, tmp_path, monkeypatch):
    monkeypatch.setenv("THERAPIST_SAFETY_ENABLED", "0")
    from src.main import TherapistApp

    app = TherapistApp.__new__(TherapistApp)
    # Minimal manual wiring to avoid constructing the full stack.
    import datetime as dt
    import threading
    app.config = tmp_config
    app.config.safety.enabled = False
    app.transcript = []
    app.session_id = "test"
    app._partial_path = tmp_path / "p.jsonl"
    app._lock = threading.Lock()
    app._closed = False
    app.engine = _StubEngine()

    class _NoNotes:
        def per_turn(self, *a, **k):
            raise RuntimeError("skip")
    app.librarian = _NoNotes()

    class _Speaker:
        def speak_text(self, *_a, **_k):
            pass
    app.speaker = _Speaker()

    wav = tmp_path / "clip.wav"
    _write_wav(wav)

    reply = app.run_once(
        "spoken text", speak=False, emit_console=False,
        audio_path=str(wav), audio_duration_seconds=5.0,
    )
    assert reply == "reply"
    assert app.engine.received["audio_path"] == str(wav)
    assert app.engine.received["audio_duration_seconds"] == 5.0
    # Temp audio is cleaned up after the turn.
    assert not wav.exists()


class _StreamingStubEngine:
    def __init__(self, speaker_called: threading.Event):
        self.speaker_called = speaker_called

    def generate_reply(self, user_text, transient_context=None, audio_path=None,
                       audio_mime_type="audio/wav", audio_duration_seconds=0.0,
                       on_status=None):
        del transient_context, audio_path, audio_mime_type, audio_duration_seconds, on_status
        assert user_text == "stream this"
        yield "First paragraph."
        yield "\n\n"
        assert self.speaker_called.wait(1.0)
        yield "Second paragraph."


def test_run_once_starts_tts_after_first_streamed_paragraph(tmp_config, tmp_path, monkeypatch):
    monkeypatch.setenv("THERAPIST_SAFETY_ENABLED", "0")
    from src.main import TherapistApp

    app = TherapistApp.__new__(TherapistApp)
    app.config = tmp_config
    app.config.safety.enabled = False
    app.transcript = []
    app.session_id = "test"
    app._partial_path = tmp_path / "p.jsonl"
    app._lock = threading.Lock()
    app._closed = False

    speaker_called = threading.Event()
    spoken = []
    app.engine = _StreamingStubEngine(speaker_called)

    class _NoNotes:
        def per_turn(self, *a, **k):
            raise RuntimeError("skip")
    app.librarian = _NoNotes()

    class _Speaker:
        def speak_text(self, text):
            spoken.append(text)
            speaker_called.set()
    app.speaker = _Speaker()

    reply = app.run_once("stream this", speak=True, emit_console=False)

    assert reply == "First paragraph.\n\nSecond paragraph."
    assert spoken == ["First paragraph.", "Second paragraph."]
