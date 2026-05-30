from __future__ import annotations

import queue
import tempfile
import threading
import wave
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import AppConfig


@dataclass
class TranscriptChunk:
    text: str
    audio_path: str
    mime_type: str = "audio/wav"
    duration_seconds: float = 0.0


class SpeechListener:
    def __init__(self, config: AppConfig):
        self.config = config
        self._frames = queue.Queue()
        self._stream = None
        self._model = None
        self._vad_model = None

    def _load_dependencies(self) -> None:
        try:
            import sounddevice as sounddevice_module
            from faster_whisper import WhisperModel
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Speech dependencies are missing. Install requirements with Python 3.10+ before running microphone mode."
            ) from exc

        self.sounddevice = sounddevice_module
        if self._model is None:
            import os as _os
            device = _os.getenv("THERAPIST_WHISPER_DEVICE", "cpu")
            compute_type = _os.getenv("THERAPIST_WHISPER_COMPUTE", "float16" if device == "cuda" else "int8")
            kwargs = {"device": device, "compute_type": compute_type}
            if device == "cuda":
                kwargs["device_index"] = int(_os.getenv("THERAPIST_WHISPER_GPU", "1"))
            self._model = WhisperModel(self.config.models.whisper_model, **kwargs)
        if self._vad_model is None:
            self._vad_model = load_silero_vad()

    def capture_once(
        self,
        stop_event: Optional[threading.Event] = None,
        on_ready: Optional["Callable[[], None]"] = None,
    ) -> TranscriptChunk:
        self._load_dependencies()
        print("Speak now. Recording will stop after a short pause.")
        frames = []

        def callback(indata, frames_count, time_info, status):
            del frames_count, time_info
            if status:
                print(status)
            frames.append(indata.copy())

        import time

        with self.sounddevice.InputStream(
            samplerate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            device=self.config.audio.device_index,
            dtype="float32",
            callback=callback,
        ):
            if on_ready is not None:
                on_ready()  # Mic is live; safe to tell the user to start talking.
            start_time = time.time()
            is_speaking = False
            silence_start = None
            
            while time.time() - start_time < self.config.audio.max_record_seconds:
                if stop_event is not None and stop_event.is_set():
                    break  # User pressed the stop button
                self.sounddevice.sleep(100)
                if not frames:
                    continue
                
                # Check RMS energy of the latest chunk to detect silence
                latest_audio = np.concatenate(frames[-5:], axis=0) if len(frames) > 5 else frames[-1]
                rms = np.sqrt(np.mean(latest_audio**2))
                
                if rms > 0.005:  # Basic volume threshold for speech
                    is_speaking = True
                    silence_start = None
                elif is_speaking:
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) * 1000 > self.config.audio.silence_ms:
                        break  # Stop recording after silence

        if not frames:
            return TranscriptChunk(text="", audio_path="")

        audio = np.concatenate(frames, axis=0).flatten()
        duration_seconds = float(len(audio)) / float(self.config.audio.sample_rate or 1)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_file.name, "wb") as wav_file:
            wav_file.setnchannels(self.config.audio.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config.audio.sample_rate)
            pcm = np.clip(audio, -1.0, 1.0)
            wav_file.writeframes((pcm * 32767).astype(np.int16).tobytes())
        segments, _ = self._model.transcribe(
            temp_file.name,
            vad_filter=True,
            language="en",
            beam_size=1,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptChunk(
            text=text,
            audio_path=temp_file.name,
            mime_type="audio/wav",
            duration_seconds=duration_seconds,
        )
