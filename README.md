# Therapist Engine

Therapist Engine is a fully local therapist-style assistant for Windows that uses open-source models for conversation, safety checks, long-term memory, speech-to-text, and text-to-speech.

The current implementation already supports:

- Local text conversations through Ollama
- A desktop UI built with Tkinter
- Safety screening on every user message using a separate model
- Session summaries and vector-memory storage for later recall
- Optional speech input/output modules, pending the full Python 3.10+ speech stack installation

It is designed to keep all data local. The app talks to Ollama on `localhost` and stores local session artifacts under `.data/`.

This is not a substitute for a licensed clinician, emergency services, or crisis intervention.

## What It Does

At a high level, the application does five things:

1. Accepts a user message from the CLI or desktop UI.
2. Runs a separate safety pass against that message.
3. Generates a therapist-style response with the main Ollama model.
4. Optionally converts the response to speech.
5. Summarizes the session and stores memory embeddings locally when the session ends.

The default model assignments are:

- Therapist model: `tripolskypetr/qwen3.5-uncensored-aggressive:9b`
- Safety model: `nemotron-mini:4b-instruct-q8_0`
- Embedding model: `nomic-embed-text:latest`
- Speech-to-text model: `distil-large-v3` through `faster-whisper`
- Text-to-speech model: `xtts_v2` through `coqui-tts`

## Current Status

What works now:

- Text-mode chat through Ollama
- Desktop GUI launch
- Streaming therapist responses in the UI
- Safety classification before response generation
- Local memory retrieval and session summarization

What is scaffolded but depends on the speech runtime:

- Microphone capture through `sounddevice`
- Whisper transcription through `faster-whisper`
- XTTS speech generation through `coqui-tts`

Important current limitation:

- The speech path requires Python 3.10+ and the heavier audio dependencies. If the environment is still using Python 3.7, the text path works but the STT/TTS modules will not be usable yet.

## Quick Start

### 1. Requirements

- Windows
- Ollama running locally
- Python 3.10 or newer for the full stack
- GPU recommended for STT/TTS workloads

### 2. Pull the required Ollama models

```powershell
ollama pull tripolskypetr/qwen3.5-uncensored-aggressive:9b
ollama pull nemotron-mini:4b-instruct-q8_0
ollama pull nomic-embed-text:latest
```

### 3. Set up the Python environment

Use the included setup script:

```powershell
.\scripts\setup.ps1 -PythonExe "C:\Path\To\Python310\python.exe"
```

The setup script will:

- Validate Python version
- Create a virtual environment
- Install PyTorch with CUDA 12.1 wheels
- Install the Python dependencies from `requirements.txt`
- Pull the required Ollama models

### 4. Launch the app

Desktop UI:

```powershell
python -m src.main --mode gui
```

CLI text mode:

```powershell
python -m src.main --mode text
```

One-shot prompt:

```powershell
python -m src.main --once "I feel overwhelmed today."
```

Voice mode entrypoint:

```powershell
python -m src.main --mode voice
```

The voice entrypoint exists, but it only works once the speech dependencies are installed with a supported Python version.

## Documentation

- Usage guide: [docs/USAGE.md](docs/USAGE.md)
- Architecture and internal flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
