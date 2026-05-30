from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import AppConfig
from .librarian import Librarian, LibrarianRunReport
from .llm_therapist import OllamaClient, TherapistEngine
from .memory import MemoryStore
from .mood import MoodLog
from .notebook import NotebookStore
from .safety import SafetyMonitor, SafetyResult
from .stt_listener import SpeechListener
from .tts_speaker import TextSpeaker


@dataclass
class TurnRecord:
    role: str
    content: str
    timestamp: str
    safety: Optional[Dict[str, Any]] = None
    surfaced_note_ids: List[str] = field(default_factory=list)


class TherapistApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OllamaClient(config)
        self.memory = MemoryStore(config, self.client)
        self.notebook = NotebookStore(config, self.client)
        self.librarian = Librarian(config, self.client, self.notebook)
        self.mood = MoodLog(config)
        self.engine = TherapistEngine(config, self.client)
        self.safety = SafetyMonitor(config, self.client)
        self.listener = SpeechListener(config)
        self.speaker = TextSpeaker(config)
        self.transcript: List[TurnRecord] = []
        self.session_id: str = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self._partial_path: Path = config.session_dir / f"{self.session_id}.partial.jsonl"
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------ memory
    def warm_memory(self, seed_text: str) -> None:
        # Pinned notes always come first.
        pinned = [n.to_markdown() for n in self.notebook.pinned_notes()]
        if pinned:
            self.engine.add_pinned_notes(pinned)

        items = self.memory.retrieve(seed_text)
        if items:
            self.engine.add_memory_context([item.text for item in items])

        # Ask the librarian which notebook entries to surface for this session.
        try:
            report = self.librarian.warmup(self.session_id, seed_text)
            snippets = self._render_surfaced_notes(report.surfaced_note_ids)
            if snippets:
                self.engine.add_memory_context(snippets)
        except Exception:
            pass

    # ------------------------------------------------------------------ loops
    def run_text_loop(self) -> None:
        print("Therapist engine ready. Type 'quit' to exit.")
        try:
            while True:
                user_text = input("You: ").strip()
                if not user_text:
                    continue
                if user_text.lower() in {"quit", "exit"}:
                    break
                self._handle_turn(user_text, speak=False)
        finally:
            self.end_session()

    def run_voice_loop(self) -> None:
        print("Voice mode ready. Press Ctrl+C to stop.")
        try:
            while True:
                chunk = self.listener.capture_once()
                if not chunk.text:
                    continue
                self._handle_turn(
                    chunk.text,
                    speak=True,
                    audio_path=chunk.audio_path or None,
                    audio_mime_type=chunk.mime_type,
                    audio_duration_seconds=chunk.duration_seconds,
                )
        finally:
            self.end_session()

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
        return self._handle_turn(
            user_text,
            speak=speak,
            on_token=on_token,
            on_status=on_status,
            emit_console=emit_console,
            audio_path=audio_path,
            audio_mime_type=audio_mime_type,
            audio_duration_seconds=audio_duration_seconds,
        )

    # ------------------------------------------------------------------ turn
    def _handle_turn(
        self,
        user_text: str,
        speak: bool,
        on_token: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        emit_console: bool = True,
        audio_path: Optional[str] = None,
        audio_mime_type: str = "audio/wav",
        audio_duration_seconds: float = 0.0,
    ) -> str:
        user_record = TurnRecord(
            role="user",
            content=user_text,
            timestamp=_now_iso(),
        )
        self._append_turn(user_record)

        if self.config.safety.enabled:
            if on_status:
                on_status("Running safety check...")
            safety_result = self.safety.assess(user_text)
            user_record.safety = _safety_to_dict(safety_result)
        else:
            safety_result = SafetyResult(
                flagged=False, risk_level="disabled", reason="safety layer disabled", source="disabled",
            )
            user_record.safety = {"flagged": False, "risk_level": "disabled",
                                  "reason": "", "source": "disabled"}
        self._rewrite_partial()

        if safety_result.flagged:
            reply = self.safety.build_crisis_reply(user_text)
            if emit_console:
                print("Therapist: " + reply)
            if on_token:
                on_token(reply)
            if on_status:
                on_status(f"Crisis response ready ({safety_result.risk_level}).")
            self._emit_reply(reply, speak)
            self._cleanup_turn_audio(audio_path)
            return reply

        # Mid-session per-turn retrieval (best-effort).
        transient: List[str] = []
        surfaced: List[str] = []
        try:
            recent = [f"{t.role}: {t.content}" for t in self.transcript[-7:-1]]
            per_turn = self.librarian.per_turn(self.session_id, user_text, recent)
            surfaced = per_turn.surfaced_note_ids
            transient = self._render_surfaced_notes(surfaced)
        except Exception:
            pass
        user_record.surfaced_note_ids = surfaced

        if emit_console:
            print("Therapist: ", end="")
        if on_status:
            on_status("Generating response...")
        tokens: List[str] = []
        try:
            for token in self.engine.generate_reply(
                user_text,
                transient_context=transient,
                audio_path=audio_path,
                audio_mime_type=audio_mime_type,
                audio_duration_seconds=audio_duration_seconds,
                on_status=on_status,
            ):
                if emit_console:
                    print(token, end="", flush=True)
                if on_token:
                    on_token(token)
                tokens.append(token)
        finally:
            self._cleanup_turn_audio(audio_path)
        if emit_console:
            print()
        reply = "".join(tokens)
        if on_status:
            on_status("Speaking response..." if speak else "Response ready.")
        self._emit_reply(reply, speak)
        if on_status:
            on_status("Ready.")
        return reply

    def _cleanup_turn_audio(self, audio_path: Optional[str]) -> None:
        if not audio_path:
            return
        try:
            p = Path(audio_path)
            if p.exists():
                p.unlink()
        except Exception:
            pass


    def _emit_reply(self, reply: str, speak: bool) -> None:
        self._append_turn(TurnRecord(role="assistant", content=reply, timestamp=_now_iso()))
        if speak:
            try:
                self.speaker.speak_text(reply)
            except Exception as exc:
                print(f"[tts disabled: {type(exc).__name__}: {exc}]")

    # ------------------------------------------------------------------ mood
    def record_mood(self, scope: str, value: int, note: str = "") -> None:
        try:
            self.mood.record(scope=scope, value=value, session_id=self.session_id, note=note)
        except Exception:
            pass

    # ------------------------------------------------------------------ end
    def end_session(self) -> Dict[str, Any]:
        with self._lock:
            if self._closed:
                return {"closed": True}
            self._closed = True
        result: Dict[str, Any] = {
            "session_id": self.session_id,
            "summary": None,
            "librarian": None,
        }
        if not self.transcript:
            self._cleanup_partial()
            return result
        # Summary + memory persistence.
        try:
            summary = self.memory.summarize([t.__dict__ for t in self.transcript])
            if summary:
                self.memory.remember_session(self.session_id, summary)
                result["summary"] = summary
        except Exception as exc:
            result["summary_error"] = f"{type(exc).__name__}: {exc}"
        # Persist structured transcript (if enabled) alongside summary.
        try:
            self._write_session_record(summary_text=result.get("summary"))
        except Exception:
            pass
        # Librarian consolidation.
        try:
            report = self.librarian.consolidate(
                self.session_id,
                [t.__dict__ for t in self.transcript],
            )
            result["librarian"] = _report_to_dict(report)
        except Exception as exc:
            result["librarian_error"] = f"{type(exc).__name__}: {exc}"
        self._cleanup_partial()
        return result

    # ------------------------------------------------------------------ misc
    def _append_turn(self, turn: TurnRecord) -> None:
        with self._lock:
            self.transcript.append(turn)
            self._append_partial(turn)

    def _append_partial(self, turn: TurnRecord) -> None:
        if not self.config.memory.persist_transcript:
            return
        try:
            self._partial_path.parent.mkdir(parents=True, exist_ok=True)
            with self._partial_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(turn.__dict__) + "\n")
        except Exception:
            pass

    def _rewrite_partial(self) -> None:
        if not self.config.memory.persist_transcript:
            return
        try:
            with self._partial_path.open("w", encoding="utf-8") as fh:
                for t in self.transcript:
                    fh.write(json.dumps(t.__dict__) + "\n")
        except Exception:
            pass

    def _write_session_record(self, summary_text: Optional[str]) -> None:
        path = self.config.session_dir / f"{self.session_id}.json"
        record = {
            "session_id": self.session_id,
            "ended_at": _now_iso(),
            "summary": summary_text,
        }
        if self.config.memory.persist_transcript:
            record["transcript"] = [t.__dict__ for t in self.transcript]
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _cleanup_partial(self) -> None:
        try:
            if self._partial_path.exists():
                self._partial_path.unlink()
        except Exception:
            pass

    def _render_surfaced_notes(self, ids: List[str]) -> List[str]:
        snippets: List[str] = []
        for nid in ids:
            note = self.notebook.read_note(nid)
            if note is None:
                continue
            snippets.append(f"[{note.meta.category}] {note.meta.title}\n{note.body.strip()}")
        return snippets

    # Back-compat alias used by older UI code paths.
    def _persist_session(self) -> None:
        self.end_session()


def _safety_to_dict(result: SafetyResult) -> Dict[str, Any]:
    return {
        "flagged": result.flagged,
        "risk_level": result.risk_level,
        "reason": result.reason,
        "source": result.source,
    }


def _report_to_dict(report: LibrarianRunReport) -> Dict[str, Any]:
    return {
        "skipped": report.skipped,
        "error": report.error,
        "final_message": report.final_message,
        "operations": report.operations,
        "surfaced_note_ids": report.surfaced_note_ids,
    }


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local therapist engine")
    parser.add_argument("--mode", choices=["text", "voice", "gui"], default="text")
    parser.add_argument("--seed-memory", default="feeling overwhelmed")
    parser.add_argument("--once", help="Run a single text turn and exit.")
    parser.add_argument("--speak", action="store_true", help="Speak the reply when using --once.")
    parser.add_argument("--profile", help="Profile name (namespaces .data/<profile>/).")
    parser.add_argument(
        "--librarian-dry-run",
        action="store_true",
        help="Do not actually write notebook changes; log intended operations instead.",
    )
    safety_group = parser.add_mutually_exclusive_group()
    safety_group.add_argument(
        "--safety", dest="safety", action="store_true", default=None,
        help="Force the safety classifier ON for this run (overrides env).",
    )
    safety_group.add_argument(
        "--no-safety", dest="safety", action="store_false", default=None,
        help="Force the safety classifier OFF for this run (overrides env).",
    )
    return parser


def _install_crash_logger() -> None:
    import faulthandler
    import os
    import sys
    import threading as _threading
    import traceback
    from datetime import datetime

    crash_log = Path(__file__).resolve().parent.parent / ".data" / "crash.log"
    crash_log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(crash_log, "a", buffering=1, encoding="utf-8")
    fh.write(f"\n=== process start {datetime.now().isoformat()} pid={os.getpid()} ===\n")
    faulthandler.enable(file=fh, all_threads=True)

    class _Tee:
        def __init__(self, primary, mirror):
            self.primary = primary
            self.mirror = mirror
        def write(self, s):
            try:
                self.primary.write(s)
            except Exception:
                pass
            try:
                self.mirror.write(s)
            except Exception:
                pass
        def flush(self):
            try:
                self.primary.flush()
            except Exception:
                pass
            try:
                self.mirror.flush()
            except Exception:
                pass

    sys.stderr = _Tee(sys.stderr, fh)

    def _log(prefix: str, exc_type, exc_value, exc_tb) -> None:
        fh.write(f"\n--- {prefix} {datetime.now().isoformat()} ---\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=fh)
        fh.flush()
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.__stderr__)

    def _sys_hook(exc_type, exc_value, exc_tb):
        _log("sys.excepthook", exc_type, exc_value, exc_tb)

    def _thread_hook(args):
        _log(f"thread {args.thread.name}", args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _sys_hook
    _threading.excepthook = _thread_hook


def main() -> None:
    _install_crash_logger()
    args = build_parser().parse_args()
    if args.profile:
        import os
        os.environ["THERAPIST_PROFILE"] = args.profile
    if args.librarian_dry_run:
        import os
        os.environ["THERAPIST_LIBRARIAN_DRY_RUN"] = "1"
    if args.safety is not None:
        import os
        os.environ["THERAPIST_SAFETY_ENABLED"] = "1" if args.safety else "0"
    if args.mode == "gui":
        from .ui import launch_desktop_ui
        launch_desktop_ui(seed_memory=args.seed_memory)
        return
    config = AppConfig.load()
    app = TherapistApp(config)
    app.warm_memory(args.seed_memory)
    if args.once:
        app.run_once(args.once, speak=args.speak)
        app.end_session()
        return
    if args.mode == "voice":
        app.run_voice_loop()
    else:
        app.run_text_loop()


if __name__ == "__main__":
    main()
