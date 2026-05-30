from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
DATA_DIR = ROOT_DIR / ".data"


def _read_prompt(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback.strip()


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


@dataclass
class AudioSettings:
    sample_rate: int = 16000
    channels: int = 1
    silence_ms: int = field(default_factory=lambda: int(os.getenv("THERAPIST_SILENCE_MS", "3000")))
    max_record_seconds: int = field(default_factory=lambda: int(os.getenv("THERAPIST_MAX_RECORD_S", "30")))
    device_index: int = None
    voice_name: str = field(default_factory=lambda: os.getenv("THERAPIST_TTS_VOICE", "F5"))
    tts_language: str = field(default_factory=lambda: os.getenv("THERAPIST_TTS_LANG", "en"))
    tts_steps: int = field(default_factory=lambda: int(os.getenv("THERAPIST_TTS_STEPS", "16")))
    tts_speed: float = field(default_factory=lambda: float(os.getenv("THERAPIST_TTS_SPEED", "1.15")))
    # Multimodal voice turns: attach the captured WAV to the therapist model.
    send_audio_to_model: bool = field(default_factory=lambda: os.getenv("THERAPIST_SEND_AUDIO_TO_MODEL", "1") != "0")
    model_audio_max_seconds: float = field(default_factory=lambda: float(os.getenv("THERAPIST_MODEL_AUDIO_MAX_S", "30")))
    audio_fallback_text: bool = field(default_factory=lambda: os.getenv("THERAPIST_AUDIO_FALLBACK_TEXT", "1") != "0")


@dataclass
class ModelSettings:
    therapist_model: str = "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:latest"
    safety_model: str = "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:latest"
    embedding_model: str = "nomic-embed-text:latest"
    librarian_model: str = field(default_factory=lambda: os.getenv("THERAPIST_LIBRARIAN_MODEL", "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:latest"))
    whisper_model: str = "distil-large-v3"
    tts_model: str = "Supertone/supertonic-3"


@dataclass
class SafetySettings:
    enabled: bool = field(default_factory=lambda: os.getenv("THERAPIST_SAFETY_ENABLED", "1") != "0")


@dataclass
class LibrarianSettings:
    enabled: bool = field(default_factory=lambda: os.getenv("THERAPIST_LIBRARIAN_ENABLED", "1") != "0")
    dry_run: bool = field(default_factory=lambda: os.getenv("THERAPIST_LIBRARIAN_DRY_RUN", "0") == "1")
    max_tool_calls: int = 8
    max_new_notes_per_session: int = 3
    per_turn_timeout_s: float = 2.5
    per_turn_enabled: bool = True


@dataclass
class MemorySettings:
    retrieval_limit: int = 3
    recency_half_life_days: float = 30.0
    persist_transcript: bool = field(default_factory=lambda: os.getenv("THERAPIST_PERSIST_TRANSCRIPT", "1") != "0")


@dataclass
class AppConfig:
    profile: str = field(default_factory=lambda: os.getenv("THERAPIST_PROFILE", "default"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    session_dir: Path = field(init=False)
    chroma_dir: Path = field(init=False)
    notebook_dir: Path = field(init=False)
    notebook_trash_dir: Path = field(init=False)
    mood_log_path: Path = field(init=False)
    safety_log_path: Path = field(init=False)
    librarian_log_path: Path = field(init=False)
    keep_alive: str = field(default_factory=lambda: os.getenv("OLLAMA_KEEP_ALIVE", "5m"))
    num_ctx: int = field(default_factory=lambda: int(os.getenv("THERAPIST_NUM_CTX", "16384")))
    audio: AudioSettings = field(default_factory=AudioSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    librarian: LibrarianSettings = field(default_factory=LibrarianSettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    prompts: Dict[str, str] = field(default_factory=dict)
    safety_keywords: List[str] = field(default_factory=list)
    notebook_categories: List[str] = field(default_factory=lambda: [
        "people", "goals", "coping_strategies", "triggers", "events",
        "reflections", "safety_plan", "values", "homework",
    ])

    def __post_init__(self) -> None:
        base = self.data_dir / self.profile if self.profile and self.profile != "default" else self.data_dir
        self.session_dir = base / "sessions"
        self.chroma_dir = base / "chroma"
        self.notebook_dir = base / "notebook"
        self.notebook_trash_dir = self.notebook_dir / "_trash"
        self.mood_log_path = base / "mood.jsonl"
        self.safety_log_path = base / "safety_log.jsonl"
        self.librarian_log_path = base / "librarian_log.jsonl"

    @classmethod
    def load(cls) -> "AppConfig":
        config = cls()
        for directory in (
            config.session_dir,
            config.chroma_dir,
            config.notebook_dir,
            config.notebook_trash_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        config.prompts = {
            "therapist": _read_prompt(
                PROMPTS_DIR / "DrRebecca.txt",
                "You are a calm, empathic therapist assistant. Reflect feelings, ask open questions, avoid diagnosis, and do not claim to replace emergency or professional care.",
            ),
            "safety": _read_prompt(
                PROMPTS_DIR / "safety_classifier.txt",
                "Classify whether the user message indicates imminent self-harm, suicide risk, abuse, or a mental health crisis. Return JSON with keys risk_level, flagged, reason.",
            ),
            "crisis": _read_prompt(
                PROMPTS_DIR / "crisis_response.txt",
                "If there is immediate danger, encourage contacting local emergency services or 988 in the United States. Keep the tone calm and direct.",
            ),
            "librarian": _read_prompt(
                PROMPTS_DIR / "librarian_system.txt",
                "You are a background notebook librarian. Use the provided tools to organize notes.",
            ),
        }
        config.safety_keywords = _read_lines(PROMPTS_DIR / "safety_keywords.txt")
        return config
