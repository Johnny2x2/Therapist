from __future__ import annotations

import logging
import re
import threading
from typing import List, Optional

import numpy as np

from .config import AppConfig

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> List[str]:
    chunks = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item.strip()]
    return chunks or ([text.strip()] if text.strip() else [])


class TextSpeaker:
    """Supertonic 3 ONNX-based TTS. Failures are non-fatal."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._tts = None
        self._style = None
        self._sounddevice = None
        self._lock = threading.Lock()
        self._failed = False
        self._stop_requested = False

    def _load_dependencies(self) -> bool:
        if self._failed:
            return False
        if self._tts is not None:
            return True
        with self._lock:
            if self._tts is not None:
                return True
            try:
                from supertonic import TTS
            except ImportError as exc:
                logger.warning("TTS dependencies missing (%s); voice output disabled.", exc)
                self._failed = True
                return False
            # sounddevice is only needed for local playback (play()); server-side
            # synthesis (synthesize()) works headless without it.
            try:
                import sounddevice as sounddevice_module
                self._sounddevice = sounddevice_module
            except Exception as exc:  # noqa: BLE001 - headless/no PortAudio is fine
                logger.info("sounddevice unavailable (%s); playback disabled, synth still works.", exc)
                self._sounddevice = None
            try:
                tts = TTS(auto_download=True)
                self._style = tts.get_voice_style(voice_name=self.config.audio.voice_name)
                self._tts = tts
                return True
            except Exception as exc:
                logger.warning("TTS initialization failed (%s); voice output disabled.", exc)
                self._failed = True
                return False

    def speak_text(self, text: str) -> None:
        if not text or not text.strip():
            return
        if not self._load_dependencies():
            return
        self._stop_requested = False
        for sentence in split_sentences(text):
            if self._stop_requested:
                break
            try:
                self._speak_sentence(sentence)
            except Exception as exc:
                logger.warning("TTS playback failed for one sentence (%s); skipping.", exc)

    def stop(self) -> None:
        """Signal any in-progress playback to stop. Best-effort and non-fatal."""
        self._stop_requested = True
        if self._sounddevice is not None:
            try:
                self._sounddevice.stop()
            except Exception:
                pass

    def _speak_sentence(self, sentence: str) -> None:
        result = self.synthesize(sentence)
        if result is None:
            return
        data, sample_rate = result
        self.play(data, sample_rate)

    def synthesize(self, text: str):
        """Synthesize speech for ``text`` without playing it.

        Returns ``(np.ndarray, sample_rate)`` ready for playback, or ``None``
        when TTS is unavailable. Used by the API server to stream audio.
        """
        if not text or not text.strip():
            return None
        if not self._load_dependencies():
            return None
        wav, _duration = self._tts.synthesize(
            text,
            voice_style=self._style,
            lang=self.config.audio.tts_language,
        )
        return self._to_playable(wav)

    def play(self, data, sample_rate) -> None:
        """Play already-synthesized PCM audio through the default output device."""
        data = np.asarray(data)
        if data.dtype != np.float32:
            if np.issubdtype(data.dtype, np.integer):
                max_val = float(np.iinfo(data.dtype).max) or 1.0
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
        if self._sounddevice is None:
            import sounddevice as sounddevice_module
            self._sounddevice = sounddevice_module
        channels = 1 if data.ndim == 1 else data.shape[1]
        with self._sounddevice.OutputStream(
            samplerate=int(sample_rate),
            channels=channels,
            dtype="float32",
        ) as stream:
            stream.write(data if data.ndim == 2 else data.reshape(-1, 1))

    def _to_playable(self, wav) -> tuple:
        sample_rate = getattr(self._tts, "sample_rate", None) or 44100
        if isinstance(wav, tuple) and len(wav) == 2:
            first, second = wav
            if isinstance(first, (int, np.integer)):
                sample_rate = int(first)
                arr = np.asarray(second)
            else:
                arr = np.asarray(first)
                sample_rate = int(second)
        else:
            arr = np.asarray(wav)
        if arr.dtype != np.float32:
            if np.issubdtype(arr.dtype, np.integer):
                max_val = float(np.iinfo(arr.dtype).max) or 1.0
                arr = arr.astype(np.float32) / max_val
            else:
                arr = arr.astype(np.float32)
        if arr.ndim == 2:
            # Supertonic returns (channels, frames); sounddevice wants (frames,) or (frames, channels)
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T
            if arr.shape[1] == 1:
                arr = arr[:, 0]
        return arr, int(sample_rate)
