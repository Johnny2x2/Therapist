from __future__ import annotations

import logging
import re
import threading
from typing import List, Optional

import numpy as np

from .config import AppConfig

logger = logging.getLogger(__name__)


# Supertonic's Python SDK is character-based and does NOT interpret expression
# tags (it would spell them out letter-by-letter). We translate the "pause-like"
# tags into punctuation pauses and drop the rest so they're never read aloud.
_PAUSE_TAGS = re.compile(r"\s*<\s*(breath|sigh|inhale|exhale|pause)\s*>\s*", re.IGNORECASE)
_ANY_TAG = re.compile(r"<\s*[a-zA-Z][^>]*>")


def clean_expression_tags(text: str) -> str:
    text = _PAUSE_TAGS.sub("... ", text)
    text = _ANY_TAG.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    text = clean_expression_tags(text)
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

    def _load_dependencies(self) -> bool:
        if self._failed:
            return False
        if self._tts is not None:
            return True
        with self._lock:
            if self._tts is not None:
                return True
            try:
                import sounddevice as sounddevice_module
                from supertonic import TTS
            except ImportError as exc:
                logger.warning("TTS dependencies missing (%s); voice output disabled.", exc)
                self._failed = True
                return False
            try:
                self._sounddevice = sounddevice_module
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
        for sentence in split_sentences(text):
            try:
                self._speak_sentence(sentence)
            except Exception as exc:
                logger.warning("TTS playback failed for one sentence (%s); skipping.", exc)

    def _speak_sentence(self, sentence: str) -> None:
        wav, _duration = self._tts.synthesize(
            sentence,
            voice_style=self._style,
            lang=self.config.audio.tts_language,
            total_steps=self.config.audio.tts_steps,
            speed=self.config.audio.tts_speed,
        )
        data, sample_rate = self._to_playable(wav)
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]
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
