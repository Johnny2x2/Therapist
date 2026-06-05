import datetime as dt
import json
import logging
from typing import Callable, Optional

import requests

from .config import AppConfig

logger = logging.getLogger(__name__)

class RemoteNotebookStore:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def list_notes(self, category: Optional[str] = None, limit: int = 100):
        # Implementation of remote call
        return []

    def search_notes(self, query: str, k: int = 5, category: Optional[str] = None):
        return []

    def read_note(self, note_id: str):
        return None

    def update_note(self, note_id: str, set_pinned: Optional[bool] = None, replace_body: Optional[str] = None):
        pass


class RemoteTherapistApp:
    """A proxy client that implements the same high-level interface as TherapistApp
    but routes all heavy lifting to the remote API.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.base_url = config.backend_url.rstrip("/")
        
        # We still need local speech listening and playback though
        from .stt_listener import SpeechListener
        self.listener = SpeechListener(config)
        
        # Actually in this mode we need a custom audio playback queue
        from .tts_speaker import TextSpeaker
        self.speaker = TextSpeaker(config)
        
        self.session_id: str = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.notebook = RemoteNotebookStore(self.base_url)

    def warm_memory(self, seed_text: str) -> None:
        # TODO call /session/start
        pass

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
        
        # We make a streaming request to /chat
        import httpx
        url = f"{self.base_url}/chat"
        payload = {
            "session_id": self.session_id,
            "user_text": user_text
        }
        
        # Stream the ndjson response
        full_reply = []
        with httpx.stream("POST", url, json=payload, timeout=None) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                
                mtype = msg.get("type")
                content = msg.get("content", "")
                
                if mtype == "status" and on_status:
                    on_status(content)
                elif mtype == "text":
                    full_reply.append(content)
                    if on_token:
                        on_token(content)
                elif mtype == "audio" and speak:
                    # In a real app we'd decode base64 and play
                    import base64
                    import io
                    import scipy.io.wavfile as wavfile
                    # For now just decode and mock play to speaker
                    b64 = msg.get("data", "")
                    if b64:
                        import threading
                        def _play():
                            try:
                                audio_bytes = base64.b64decode(b64)
                                bio = io.BytesIO(audio_bytes)
                                sr, data = wavfile.read(bio)
                                self.speaker._play_audio(data, sr)
                            except Exception as e:
                                logger.error(f"Client audio error: {e}")
                        threading.Thread(target=_play, daemon=True).start()
                elif mtype == "error":
                    if on_status:
                        on_status(f"API Error: {content}")
        return "".join(full_reply)

    def record_mood(self, phase: str, value: int):
        pass

    def end_session(self):
        # Call /session/end
        pass
    
    class _MockEngine:
        class _MockState:
            def estimate_tokens(self):
                return 0
        state = _MockState()

    @property
    def engine(self):
        return self._MockEngine()
