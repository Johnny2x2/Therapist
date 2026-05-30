from __future__ import annotations

import logging
import queue
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


def split_paragraphs(text: str) -> List[str]:
    parts = [clean_expression_tags(item) for item in re.split(r"(?:\r?\n\s*){2,}", text) if item.strip()]
    return [item for item in parts if item]


def split_tts_chunks(text: str, backend: str) -> List[str]:
    cleaned = clean_expression_tags(text)
    if not cleaned:
        return []
    if (backend or "").strip().lower() == "chatterbox":
        return split_paragraphs(text) or [cleaned]
    return split_sentences(cleaned)



class TextSpeaker:
    """Text-to-speech with a selectable backend (Supertonic or Chatterbox).

    Failures are non-fatal: if dependencies or initialization fail, voice output
    is silently disabled and the app continues in text-only mode.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._backend = (config.audio.tts_backend or "supertonic").strip().lower()
        self._tts = None
        self._style = None
        self._sounddevice = None
        self._chatterbox_sr = None
        self._lock = threading.Lock()
        self._failed = False
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request that any in-progress playback halt as soon as possible."""
        self._stop_event.set()

    def _load_dependencies(self) -> bool:
        if self._failed:
            return False
        if self._tts is not None:
            return True
        with self._lock:
            if self._tts is not None:
                return True
            if self._backend == "chatterbox":
                return self._load_chatterbox()
            return self._load_supertonic()

    def _load_supertonic(self) -> bool:
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

    def _resolve_chatterbox_device(self) -> str:
        import os

        from .config import resolve_device

        device = resolve_device(self.config.audio.chatterbox_device or "auto")
        # Pin to a specific GPU when one isn't already specified (e.g. "cuda:1").
        if device == "cuda":
            return f"cuda:{int(os.getenv('THERAPIST_CHATTERBOX_GPU_INDEX', '1'))}"
        return device

    def _load_chatterbox(self) -> bool:
        try:
            import sounddevice as sounddevice_module
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            logger.warning(
                "Chatterbox TTS dependencies missing (%s); install with "
                "'pip install chatterbox-tts'. Voice output disabled.", exc,
            )
            self._failed = True
            return False
        try:
            self._sounddevice = sounddevice_module
            device = self._resolve_chatterbox_device()
            model = ChatterboxTTS.from_pretrained(device=device)
            self._chatterbox_sr = int(getattr(model, "sr", 24000))
            self._tts = model
            logger.info("Chatterbox TTS loaded on %s (sr=%d).", device, self._chatterbox_sr)
            return True
        except Exception as exc:
            logger.warning("Chatterbox TTS initialization failed (%s); voice output disabled.", exc)
            self._failed = True
            return False

    def speak_text(self, text: str) -> None:
        if not text or not text.strip():
            return
        if not self._load_dependencies():
            return
        self._stop_event.clear()
        chunks = split_tts_chunks(text, self._backend)
        if self._backend == "chatterbox" and len(chunks) > 1:
            self._speak_chatterbox_chunks(chunks)
            return
        for sentence in chunks:
            if self._stop_event.is_set():
                break
            try:
                self._speak_chunk(sentence)
            except Exception as exc:
                logger.warning("TTS playback failed for one sentence (%s); skipping.", exc)

    def _speak_chunk(self, sentence: str) -> None:
        if self._stop_event.is_set():
            return
        data, sample_rate = self._synthesize_chunk(sentence)
        if self._stop_event.is_set():
            return
        self._play_audio(data, sample_rate)

    def _synthesize_chunk(self, sentence: str) -> tuple:
        if self._backend == "chatterbox":
            return self._synthesize_chatterbox(sentence)
        return self._synthesize_supertonic(sentence)

    def _play_audio(self, data: np.ndarray, sample_rate: int) -> None:
        data = data if data.ndim == 2 else data.reshape(-1, 1)
        channels = data.shape[1]
        # Write in small blocks so a stop request interrupts mid-sentence
        # instead of waiting for the whole utterance to finish playing.
        block = max(1, int(sample_rate) // 10)  # ~100ms chunks
        with self._sounddevice.OutputStream(
            samplerate=int(sample_rate),
            channels=channels,
            dtype="float32",
        ) as stream:
            for start in range(0, data.shape[0], block):
                if self._stop_event.is_set():
                    stream.abort()
                    return
                stream.write(data[start:start + block])

    def _speak_chatterbox_chunks(self, chunks: List[str]) -> None:
        pending: "queue.Queue[object]" = queue.Queue(maxsize=1)
        done = object()

        def producer() -> None:
            try:
                for chunk in chunks:
                    if self._stop_event.is_set():
                        break
                    audio = self._synthesize_chunk(chunk)
                    if self._stop_event.is_set():
                        break
                    while not self._stop_event.is_set():
                        try:
                            pending.put((chunk, audio), timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                while True:
                    try:
                        pending.put(exc, timeout=0.1)
                        break
                    except queue.Full:
                        if self._stop_event.is_set():
                            return
            finally:
                while True:
                    try:
                        pending.put(done, timeout=0.1)
                        return
                    except queue.Full:
                        if self._stop_event.is_set():
                            return

        worker = threading.Thread(target=producer, name="tts-chatterbox-prefetch", daemon=True)
        worker.start()
        try:
            while not self._stop_event.is_set():
                try:
                    item = pending.get(timeout=0.1)
                except queue.Empty:
                    if not worker.is_alive():
                        break
                    continue
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise item
                _chunk, audio = item
                data, sample_rate = audio
                self._play_audio(data, sample_rate)
        finally:
            worker.join(timeout=0.2)

    def _synthesize_supertonic(self, sentence: str) -> tuple:
        wav, _duration = self._tts.synthesize(
            sentence,
            voice_style=self._style,
            lang=self.config.audio.tts_language,
            total_steps=self.config.audio.tts_steps,
            speed=self.config.audio.tts_speed,
        )
        return self._to_playable(wav)

    def _synthesize_chatterbox(self, sentence: str) -> tuple:
        kwargs = {
            "exaggeration": self.config.audio.chatterbox_exaggeration,
            "cfg_weight": self.config.audio.chatterbox_cfg_weight,
        }
        ref_wav = self.config.audio.chatterbox_ref_wav
        if ref_wav:
            kwargs["audio_prompt_path"] = ref_wav
        wav = self._tts.generate(sentence, **kwargs)
        arr = self._tensor_to_numpy(wav)
        return self._to_playable((arr, self._chatterbox_sr or 24000))

    @staticmethod
    def _tensor_to_numpy(wav) -> np.ndarray:
        # Chatterbox returns a torch tensor shaped (1, num_samples).
        detach = getattr(wav, "detach", None)
        if detach is not None:
            wav = detach()
        cpu = getattr(wav, "cpu", None)
        if cpu is not None:
            wav = cpu()
        numpy_fn = getattr(wav, "numpy", None)
        arr = numpy_fn() if numpy_fn is not None else np.asarray(wav)
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.float32, copy=False)

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
