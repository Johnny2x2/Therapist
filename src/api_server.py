import asyncio
import base64
import json
import logging
from typing import AsyncGenerator, Dict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import AppConfig
from .main import TherapistApp

logger = logging.getLogger(__name__)

# Maintain stateful session apps
# In a real ephemeral serverless environment, you'd load everything from the 
# data drive on every request instead.
_SESSIONS: Dict[str, TherapistApp] = {}


class ChatRequest(BaseModel):
    session_id: str
    user_text: str


app = FastAPI(title="Therapist API")

def get_app(session_id: str) -> TherapistApp:
    if session_id not in _SESSIONS:
        logger.info(f"Creating new TherapistApp for session {session_id}")
        config = AppConfig.load()
        # Ensure it works in backend mode
        app_inst = TherapistApp(config)
        app_inst.session_id = session_id
        # Reload history if needed
        # app_inst.warm_memory(...)
        _SESSIONS[session_id] = app_inst
    return _SESSIONS[session_id]


async def chat_stream_generator(req: ChatRequest) -> AsyncGenerator[str, None]:
    therapist_app = get_app(req.session_id)
    
    # We will buffer text to send to TTS
    audio_queue = asyncio.Queue()
    text_queue = asyncio.Queue()
    done_event = asyncio.Event()

    def on_token(token: str):
        # We need to run this synchronously but push to asyncio
        # So we'll use a hack or just rely on the fact that this generator runs
        # loop.run_in_executor anyway.
        pass

    # A better approach: Run the handle_turn in a thread. 
    # Capture tokens and paragraph boundary audio synthesis.
    
    import queue
    import threading

    sync_q = queue.Queue()

    def on_token_sync(token: str):
        sync_q.put({"type": "text", "content": token})
    
    def on_status_sync(status: str):
        sync_q.put({"type": "status", "content": status})

    # To do TTS, we need the synthesized chunks
    class AudioCaptureSpeaker:
        def __init__(self, original_speaker):
            self._speaker = original_speaker

        def synthesize_and_emit(self, text: str):
            if not text.strip():
                return
            try:
                result = self._speaker.synthesize(text)
                if result is None:
                    return
                data, sr = result
                import io
                import wave
                import numpy as np

                pcm = np.asarray(data)
                if pcm.ndim > 1:
                    pcm = pcm.reshape(pcm.shape[0], -1)[:, 0]
                pcm = np.clip(pcm.astype(np.float32), -1.0, 1.0)
                out_f = io.BytesIO()
                with wave.open(out_f, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(int(sr))
                    wav_file.writeframes((pcm * 32767).astype(np.int16).tobytes())
                b64 = base64.b64encode(out_f.getvalue()).decode("utf-8")
                sync_q.put({"type": "audio", "data": b64, "text": text})
            except Exception as e:
                logger.error(f"TTS Capture Error: {e}")

    # We need to replicate the streaming reply behavior from main.py
    # but capturing audio instead of playing it.
    from .main import _drain_complete_paragraphs

    class TextAudioStreamer:
        def __init__(self, capture_speaker):
            self.capture_speaker = capture_speaker
            self._buffer = ""

        def push_token(self, token: str):
            on_token_sync(token)
            ready, self._buffer = _drain_complete_paragraphs(self._buffer + token)
            for chunk in ready:
                self.capture_speaker.synthesize_and_emit(chunk)

        def finish(self):
            tail = self._buffer.strip()
            if tail:
                self.capture_speaker.synthesize_and_emit(tail)

    capture_speaker = AudioCaptureSpeaker(therapist_app.speaker)
    streamer = TextAudioStreamer(capture_speaker)

    def run_turn():
        try:
            # We must override the internal behaviour of handle_turn slightly.
            # Handle turn actually directly streams to `on_token` but doesn't handle TTS itself.
            # Main.py uses _StreamingReplySpeaker around handle_turn.
            # Let's just call `handle_turn` with `speak=False` and use `on_token=streamer.push_token`.

            reply = therapist_app._handle_turn(
                user_text=req.user_text,
                speak=False,
                on_token=streamer.push_token,
                on_status=on_status_sync,
                emit_console=False
            )
            streamer.finish()
            sync_q.put(None) # EOF
        except Exception as e:
            logger.error(f"Turn error: {e}")
            sync_q.put({"type": "error", "content": str(e)})
            sync_q.put(None)

    worker = threading.Thread(target=run_turn, daemon=True)
    worker.start()

    while True:
        try:
            # wait in asyncio loop without blocking
            msg = await asyncio.get_running_loop().run_in_executor(None, sync_q.get)
            if msg is None:
                break
            yield json.dumps(msg) + "\n"
        except Exception as e:
            break


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    return StreamingResponse(chat_stream_generator(req), media_type="application/x-ndjson")

if __name__ == "__main__":
    import os
    import uvicorn
    host = os.getenv("THERAPIST_API_HOST", "0.0.0.0")
    port = int(os.getenv("THERAPIST_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
