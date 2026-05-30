# Usage Guide

This guide explains exactly how to run Therapist Engine, what each mode does, and what to expect from the current implementation.

## Running the App

### Desktop UI

Launch the desktop interface with:

```powershell
python -m src.main --mode gui
```

What you will see:

- A conversation transcript area
- A multi-line input box
- A `Send` button
- A `Listen Once` button
- A `Speak replies` checkbox
- A status line at the bottom

What each control does:

- `Send`: submits the typed message to the therapist pipeline
- `Listen Once`: records one microphone utterance and then sends the transcript to the therapist pipeline
- `Speak replies`: if enabled, the app will try to synthesize the therapist reply as audio

Keyboard shortcut:

- `Ctrl+Enter` sends the current text input

## CLI Modes

### Interactive text mode

```powershell
python -m src.main --mode text
```

Behavior:

- Prompts with `You:`
- Sends each line to the therapist pipeline
- Prints the therapist reply as it streams back from Ollama
- Exits on `quit` or `exit`
- Summarizes and stores the session at the end

### One-shot mode

```powershell
python -m src.main --once "I feel anxious and distracted."
```

Behavior:

- Runs one user turn
- Prints the therapist response
- Persists a session summary before exiting

Optional speech on one-shot mode:

```powershell
python -m src.main --once "I am having a rough day." --speak
```

### Voice mode

```powershell
python -m src.main --mode voice
```

Behavior in the current implementation:

- Records one captured audio block
- Runs whisper transcription on the saved WAV
- Sends the transcript into the same therapist pipeline
- Optionally speaks the response depending on the call path

Important note:

- The current `capture_once()` implementation is not yet true end-of-speech detection. It records up to the configured maximum duration and then transcribes the result. The user-facing text says it stops after a short pause, but that behavior is not fully implemented yet.

## Configuration

The application uses defaults from `src/config.py`.

Key values:

- Ollama host: `http://127.0.0.1:11434`
- Ollama keep-alive: `5m`
- Audio sample rate: `16000`
- Audio channels: `1`
- Max recording length: `30` seconds
- Default speaker reference WAV: `speaker_reference.wav`

The prompt files live under `prompts/`:

- `therapist_system.txt`
- `safety_classifier.txt`
- `crisis_response.txt`

## Data Written to Disk

The application creates `.data/` automatically.

It stores:

- `.data/sessions/*.json`: session summaries
- `.data/chroma/`: Chroma vector database files used for memory retrieval

## Typical User Flow

### Text path

1. Launch the GUI or CLI.
2. Enter a message.
3. The safety model screens the message.
4. The therapist model generates a reply.
5. The reply appears in the UI or terminal.
6. When the session ends, a summary is generated and stored.

### Voice path

1. Launch GUI mode and click `Listen Once`, or run `--mode voice`.
2. The app captures microphone audio.
3. Whisper transcribes the audio.
4. The transcript is passed into the same safety and therapist flow.
5. If enabled and available, the reply is spoken with XTTS.

## Known Limitations

- Full speech support requires Python 3.10+ and the installed audio/TTS stack.
- Continuous VAD, live barge-in, and sentence-level streamed TTS playback are not finished yet.
- Safety output depends on the safety model returning valid JSON.
- The UI stores the conversation in memory during the session and persists only the summary at shutdown.

## Troubleshooting

### The GUI opens and closes immediately

- Verify Tkinter is available in the active Python installation.
- Run `python -c "import tkinter"`.

### Text mode works but voice features fail

- The likely cause is the Python version or missing speech dependencies.
- Run the setup script with a Python 3.10+ interpreter.

### Ollama calls fail

- Make sure Ollama is running locally.
- Confirm the models are installed.
- Check whether `OLLAMA_HOST` points to a different endpoint.
