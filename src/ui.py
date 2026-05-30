from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List, Optional

from .config import AppConfig
from .main import TherapistApp


class DesktopUI:
    def __init__(self, root: tk.Tk, app: TherapistApp):
        self.root = root
        self.app = app
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.current_reply_active = False
        self.input_enabled = True
        self._stop_recording = threading.Event()

        self.root.title(f"Therapist Engine \u2014 {app.session_id}")
        self.root.geometry("1180x760")
        self.root.minsize(900, 600)
        self.root.configure(bg="#eef1ea")

        self.status_var = tk.StringVar(value="Ready.")
        self.speak_var = tk.BooleanVar(value=False)
        self.category_var = tk.StringVar(value="(all)")
        self.context_var = tk.StringVar(value="")
        self.record_var = tk.StringVar(value="")
        self._record_deadline: Optional[float] = None
        self._record_timer_id: Optional[str] = None

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(60, self._drain_events)
        self.root.after(250, self._prompt_start_mood)
        self.root.after(400, self._refresh_notebook_list)

    # -------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#eef1ea")
        style.configure("Side.TFrame", background="#e4e8df")
        style.configure("Header.TLabel", font=("Georgia", 22, "bold"), background="#eef1ea", foreground="#1f3024")
        style.configure("Body.TLabel", font=("Segoe UI", 10), background="#eef1ea", foreground="#32463a")
        style.configure("Side.TLabel", font=("Segoe UI", 10, "bold"), background="#e4e8df", foreground="#1f3024")
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"))

        outer = ttk.Frame(self.root, style="App.TFrame", padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        main_col = ttk.Frame(outer, style="App.TFrame", padding=(0, 0, 12, 0))
        main_col.grid(row=0, column=0, sticky="nsew")
        main_col.columnconfigure(0, weight=1)
        main_col.rowconfigure(2, weight=1)

        side_col = ttk.Frame(outer, style="Side.TFrame", padding=10)
        side_col.grid(row=0, column=1, sticky="nsew")
        side_col.columnconfigure(0, weight=1)
        side_col.rowconfigure(3, weight=1)
        side_col.rowconfigure(5, weight=1)

        self._build_main(main_col)
        self._build_sidebar(side_col)

    def _build_main(self, container: ttk.Frame) -> None:
        header = ttk.Frame(container, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Therapist Engine", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Local. Private. Not a substitute for professional or emergency care. "
                 "In immediate danger call 911 or 988.",
            style="Body.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        controls = ttk.Frame(container, style="App.TFrame", padding=(0, 16, 0, 12))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(2, weight=1)

        self.send_button = ttk.Button(controls, text="Send", style="Action.TButton", command=self._submit_text)
        self.send_button.grid(row=0, column=0, padx=(0, 8))

        self.listen_button = ttk.Button(controls, text="Listen Once", command=self._capture_voice)
        self.listen_button.grid(row=0, column=1, padx=(0, 8))

        self.stop_button = ttk.Button(controls, text="Done Speaking", command=self._stop_listening, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, padx=(0, 8))

        self.speak_check = ttk.Checkbutton(controls, text="Speak replies", variable=self.speak_var)
        self.speak_check.grid(row=0, column=3, sticky="e")

        self.end_button = ttk.Button(controls, text="End Session", command=self._end_session)
        self.end_button.grid(row=0, column=4, padx=(8, 0))

        transcript_wrap = ttk.Frame(container, style="App.TFrame")
        transcript_wrap.grid(row=2, column=0, sticky="nsew")
        transcript_wrap.columnconfigure(0, weight=1)
        transcript_wrap.rowconfigure(0, weight=1)

        self.transcript = ScrolledText(
            transcript_wrap,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            padx=16,
            pady=16,
            background="#fcfcf8",
            foreground="#202622",
            insertbackground="#202622",
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.transcript.grid(row=0, column=0, sticky="nsew")
        self.transcript.tag_configure("you", foreground="#355c4a", font=("Segoe UI Semibold", 11))
        self.transcript.tag_configure("therapist", foreground="#7a3320", font=("Georgia", 11, "bold"))
        self.transcript.tag_configure("meta", foreground="#5a655e", font=("Segoe UI", 10, "italic"))
        self.transcript.tag_configure("warn", foreground="#8a1d1d", font=("Segoe UI", 10, "bold"))
        self.transcript.insert(tk.END, "Therapist: Ready when you are.\n\n", "therapist")
        self.transcript.configure(state=tk.DISABLED)

        composer = ttk.Frame(container, style="App.TFrame", padding=(0, 12, 0, 0))
        composer.grid(row=3, column=0, sticky="ew")
        composer.columnconfigure(0, weight=1)

        self.input_box = tk.Text(
            composer, height=5, wrap=tk.WORD, font=("Segoe UI", 11),
            padx=14, pady=12, background="#ffffff", foreground="#1f241f",
            insertbackground="#1f241f", relief=tk.FLAT,
        )
        self.input_box.grid(row=0, column=0, sticky="ew")
        self.input_box.bind("<Control-Return>", self._submit_text_event)
        self.input_box.bind("<Control-KP_Enter>", self._submit_text_event)

        footer = ttk.Frame(container, style="App.TFrame", padding=(0, 10, 0, 0))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(footer, textvariable=self.status_var, style="Body.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")

        style.configure("Record.Horizontal.TProgressbar", troughcolor="#d6dccf", background="#3f7d5a")
        style.configure("RecordWarn.Horizontal.TProgressbar", troughcolor="#d6dccf", background="#c47f24")
        style.configure("RecordLow.Horizontal.TProgressbar", troughcolor="#d6dccf", background="#b23b3b")
        self.record_bar = ttk.Progressbar(
            footer, orient="horizontal", mode="determinate",
            length=160, maximum=float(self.app.config.audio.max_record_seconds or 1),
            style="Record.Horizontal.TProgressbar",
        )
        self.record_label = ttk.Label(footer, textvariable=self.record_var, style="Body.TLabel")
        # Recording indicator widgets stay hidden until a capture starts.

        context_row = ttk.Frame(container, style="App.TFrame", padding=(0, 6, 0, 0))
        context_row.grid(row=5, column=0, sticky="ew")
        context_row.columnconfigure(1, weight=1)
        ttk.Label(context_row, text="Context:", style="Body.TLabel").grid(row=0, column=0, sticky="w")
        self.context_bar = ttk.Progressbar(
            context_row, orient="horizontal", mode="determinate",
            maximum=self.app.config.num_ctx,
        )
        self.context_bar.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Label(context_row, textvariable=self.context_var, style="Body.TLabel").grid(row=0, column=2, sticky="e")
        self._update_context_meter()

    def _build_sidebar(self, container: ttk.Frame) -> None:
        ttk.Label(container, text="Notebook", style="Side.TLabel").grid(row=0, column=0, sticky="w")

        filter_row = ttk.Frame(container, style="Side.TFrame")
        filter_row.grid(row=1, column=0, sticky="ew", pady=(6, 4))
        filter_row.columnconfigure(1, weight=1)
        ttk.Label(filter_row, text="Category:", style="Side.TLabel").grid(row=0, column=0, sticky="w")
        cat_values = ["(all)"] + list(self.app.config.notebook_categories)
        self.category_combo = ttk.Combobox(
            filter_row, values=cat_values, textvariable=self.category_var, state="readonly", width=18,
        )
        self.category_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.category_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_notebook_list())

        search_row = ttk.Frame(container, style="Side.TFrame")
        search_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        search_row.columnconfigure(0, weight=1)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        search_entry.grid(row=0, column=0, sticky="ew")
        search_entry.bind("<Return>", lambda _e: self._search_notebook())
        ttk.Button(search_row, text="Search", command=self._search_notebook).grid(row=0, column=1, padx=(6, 0))

        self.note_list = tk.Listbox(container, height=14, activestyle="dotbox")
        self.note_list.grid(row=3, column=0, sticky="nsew", pady=(0, 6))
        self.note_list.bind("<<ListboxSelect>>", lambda _e: self._show_selected_note())

        button_row = ttk.Frame(container, style="Side.TFrame")
        button_row.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        button_row.columnconfigure(2, weight=1)
        ttk.Button(button_row, text="Refresh", command=self._refresh_notebook_list).grid(row=0, column=0)
        ttk.Button(button_row, text="Pin/Unpin", command=self._toggle_pin).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(button_row, text="Librarian Log", command=self._show_librarian_log).grid(row=0, column=2, sticky="e")

        self.note_preview = ScrolledText(
            container, height=14, wrap=tk.WORD, font=("Consolas", 10),
            background="#fbfaf3", foreground="#202622", relief=tk.FLAT,
        )
        self.note_preview.grid(row=5, column=0, sticky="nsew")
        self.note_preview.configure(state=tk.DISABLED)

        self._note_index: List[Dict[str, Any]] = []

    # -------------------------------------------------------------- transcript helpers
    def _append_line(self, speaker: str, text: str, tag: str) -> None:
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, speaker + ": ", tag)
        self.transcript.insert(tk.END, text + "\n\n")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)

    def _append_reply_chunk(self, token: str) -> None:
        self.transcript.configure(state=tk.NORMAL)
        if not self.current_reply_active:
            self.transcript.insert(tk.END, "Therapist: ", "therapist")
            self.current_reply_active = True
        self.transcript.insert(tk.END, token)
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)

    def _finish_reply(self) -> None:
        if not self.current_reply_active:
            return
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, "\n\n")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)
        self.current_reply_active = False

    def _update_context_meter(self) -> None:
        limit = max(1, self.app.config.num_ctx)
        used = self.app.engine.state.estimate_tokens()
        pct = min(100, round(used * 100 / limit))
        self.context_bar.configure(value=min(used, limit))
        self.context_var.set(f"~{used:,} / {limit:,} tokens ({pct}%)")

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.send_button.configure(state=state)
        self.listen_button.configure(state=state)
        self.end_button.configure(state=state)
        self.input_box.configure(state=state)
        if not busy:
            self.stop_button.configure(state=tk.DISABLED)
        self.input_enabled = not busy

    # -------------------------------------------------------------- send / capture
    def _submit_text_event(self, event: tk.Event) -> str:
        del event
        self._submit_text()
        return "break"

    def _submit_text(self) -> None:
        if not self.input_enabled:
            return
        user_text = self.input_box.get("1.0", tk.END).strip()
        if not user_text:
            return
        self.input_box.delete("1.0", tk.END)
        self._append_line("You", user_text, "you")
        self._set_busy(True)
        self.status_var.set("Submitting text...")
        self._run_worker(target=self._process_text_turn, args=(user_text, self.speak_var.get()))

    def _capture_voice(self) -> None:
        if not self.input_enabled:
            return
        self._set_busy(True)
        self._stop_recording.clear()
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Preparing microphone (loading speech model)...")
        self._append_line("System", "Loading speech model...", "meta")
        self._run_worker(target=self._process_voice_turn, args=(self.speak_var.get(),))

    def _stop_listening(self) -> None:
        self._stop_recording.set()
        self.stop_button.configure(state=tk.DISABLED)
        self._stop_record_countdown()
        self.status_var.set("Finishing recording...")

    # ------------------------------------------------------- record timer
    def _start_record_countdown(self) -> None:
        limit = float(self.app.config.audio.max_record_seconds or 0)
        if limit <= 0:
            return
        self._record_deadline = time.monotonic() + limit
        self.record_bar.configure(maximum=limit)
        self.record_bar.grid(row=0, column=1, sticky="e", padx=(8, 6))
        self.record_label.grid(row=0, column=2, sticky="e")
        self._tick_record_countdown()

    def _tick_record_countdown(self) -> None:
        if self._record_deadline is None:
            return
        remaining = self._record_deadline - time.monotonic()
        if remaining <= 0:
            self.record_var.set("Time's up")
            self.record_label.configure(foreground="#b23b3b")
            self.record_bar.configure(value=self.record_bar["maximum"])
            self._record_timer_id = None
            return
        limit = float(self.record_bar["maximum"]) or 1.0
        self.record_bar.configure(value=limit - remaining)
        self.record_var.set(f"{remaining:0.0f}s left")
        if remaining <= 5:
            color = "#b23b3b"  # running out
            self.record_bar.configure(style="RecordLow.Horizontal.TProgressbar")
        elif remaining <= 10:
            color = "#c47f24"  # getting close
            self.record_bar.configure(style="RecordWarn.Horizontal.TProgressbar")
        else:
            color = "#32463a"
            self.record_bar.configure(style="Record.Horizontal.TProgressbar")
        self.record_label.configure(foreground=color)
        self._record_timer_id = self.root.after(200, self._tick_record_countdown)

    def _stop_record_countdown(self) -> None:
        if self._record_timer_id is not None:
            try:
                self.root.after_cancel(self._record_timer_id)
            except Exception:
                pass
            self._record_timer_id = None
        self._record_deadline = None
        self.record_var.set("")
        self.record_bar.grid_remove()
        self.record_label.grid_remove()

    def _process_text_turn(self, user_text: str, speak: bool) -> None:
        try:
            self.app.run_once(
                user_text, speak=speak,
                on_token=self._queue_token,
                on_status=self._queue_status,
                emit_console=False,
            )
            self.events.put({"type": "done"})
        except Exception as exc:
            self.events.put({"type": "error", "message": self._format_error(exc)})

    def _process_voice_turn(self, speak: bool) -> None:
        try:
            chunk = self.app.listener.capture_once(
                stop_event=self._stop_recording,
                on_ready=lambda: self.events.put({"type": "recording_started"}),
            )
            self.events.put({"type": "recording_done"})
            if not chunk.text:
                self.events.put({"type": "meta", "text": "No speech detected."})
                self.events.put({"type": "done"})
                return
            self.events.put({"type": "heard", "text": chunk.text})
            self.app.run_once(
                chunk.text, speak=speak,
                on_token=self._queue_token,
                on_status=self._queue_status,
                emit_console=False,
                audio_path=chunk.audio_path or None,
                audio_mime_type=chunk.mime_type,
                audio_duration_seconds=chunk.duration_seconds,
            )
            self.events.put({"type": "done"})
        except Exception as exc:
            self.events.put({"type": "error", "message": self._format_error(exc)})

    def _run_worker(self, target, args) -> None:
        worker = threading.Thread(target=target, args=args, daemon=True)
        worker.start()

    def _queue_token(self, token: str) -> None:
        self.events.put({"type": "token", "text": token})

    def _queue_status(self, status: str) -> None:
        self.events.put({"type": "status", "text": status})

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event["type"]
            if kind == "token":
                self._append_reply_chunk(event["text"])
            elif kind == "status":
                self.status_var.set(event["text"])
            elif kind == "heard":
                self._append_line("You", event["text"], "you")
            elif kind == "meta":
                self._append_line("System", event["text"], "meta")
            elif kind == "warn":
                self._append_line("System", event["text"], "warn")
            elif kind == "recording_started":
                self.status_var.set("Listening \u2014 speak now! Click 'Done Speaking' when finished.")
                self._append_line("System", "Microphone ready \u2014 go ahead.", "meta")
                self._start_record_countdown()
            elif kind == "recording_done":
                self.stop_button.configure(state=tk.DISABLED)
                self._stop_record_countdown()
            elif kind == "done":
                self._finish_reply()
                self._set_busy(False)
                self._update_context_meter()
                if self.status_var.get() not in ("Ready.",):
                    self.status_var.set("Ready.")
            elif kind == "error":
                self._finish_reply()
                self._set_busy(False)
                self._stop_record_countdown()
                self.status_var.set("Error.")
                messagebox.showerror("Therapist Engine", event["message"])
            elif kind == "session_ended":
                self._show_session_summary(event["result"])
        self.root.after(60, self._drain_events)

    # -------------------------------------------------------------- mood
    def _prompt_start_mood(self) -> None:
        value = simpledialog.askinteger(
            "Mood check-in",
            "On a 1\u201310 scale, how would you rate your current mood? (Cancel to skip)",
            parent=self.root, minvalue=1, maxvalue=10,
        )
        if value is not None:
            self.app.record_mood("start", value)
            self._append_line("System", f"Mood at start: {value}/10", "meta")

    def _prompt_end_mood(self) -> Optional[int]:
        value = simpledialog.askinteger(
            "End-of-session mood",
            "On a 1\u201310 scale, how do you feel now? (Cancel to skip)",
            parent=self.root, minvalue=1, maxvalue=10,
        )
        if value is not None:
            self.app.record_mood("end", value)
        return value

    # -------------------------------------------------------------- end session
    def _end_session(self) -> None:
        if not self.input_enabled:
            return
        if not messagebox.askyesno(
            "End session",
            "End this session now? A summary will be generated and notes may be filed.",
            parent=self.root,
        ):
            return
        self._prompt_end_mood()
        self._set_busy(True)
        self.status_var.set("Ending session (summary + librarian)...")
        self._run_worker(target=self._do_end_session, args=())

    def _do_end_session(self) -> None:
        try:
            result = self.app.end_session()
            self.events.put({"type": "session_ended", "result": result})
        except Exception as exc:
            self.events.put({"type": "error", "message": self._format_error(exc)})

    def _show_session_summary(self, result: Dict[str, Any]) -> None:
        self._set_busy(False)
        self.status_var.set("Session ended.")
        summary = result.get("summary") or "(no summary generated)"
        librarian = result.get("librarian") or {}
        ops = librarian.get("operations") or []
        op_lines = []
        for op in ops:
            tool = op.get("tool")
            args = op.get("args") or {}
            title = args.get("title") or args.get("note_id") or args.get("query") or ""
            op_lines.append(f"  - {tool}: {title}")
        msg = "Summary:\n" + summary + "\n\nLibrarian operations:\n" + (
            "\n".join(op_lines) if op_lines else "  (none)"
        )
        messagebox.showinfo("Session ended", msg, parent=self.root)
        self._refresh_notebook_list()

    # -------------------------------------------------------------- notebook
    def _refresh_notebook_list(self) -> None:
        category = self.category_var.get()
        if category == "(all)":
            category = None
        metas = self.app.notebook.list_notes(category=category, limit=200)
        self._populate_notes(metas)

    def _search_notebook(self) -> None:
        query = self.search_var.get().strip()
        if not query:
            self._refresh_notebook_list()
            return
        category = self.category_var.get()
        if category == "(all)":
            category = None
        results = self.app.notebook.search_notes(query, k=20, category=category)
        self._populate_notes(results)

    def _populate_notes(self, metas: List[Dict[str, Any]]) -> None:
        self._note_index = metas
        self.note_list.delete(0, tk.END)
        for meta in metas:
            pin = "* " if meta.get("pinned") else "  "
            cat = meta.get("category") or "?"
            title = meta.get("title") or "(untitled)"
            self.note_list.insert(tk.END, f"{pin}[{cat}] {title}")

    def _show_selected_note(self) -> None:
        idx = self._selected_note_index()
        if idx is None:
            return
        meta = self._note_index[idx]
        note = self.app.notebook.read_note(meta["id"])
        self.note_preview.configure(state=tk.NORMAL)
        self.note_preview.delete("1.0", tk.END)
        if note is None:
            self.note_preview.insert(tk.END, "(could not read note)")
        else:
            header = (f"id: {note.meta.id}\n"
                      f"title: {note.meta.title}\n"
                      f"category: {note.meta.category}\n"
                      f"tags: {', '.join(note.meta.tags) or '-'}\n"
                      f"pinned: {note.meta.pinned}\n"
                      f"updated: {note.meta.updated}\n\n")
            self.note_preview.insert(tk.END, header + note.body)
        self.note_preview.configure(state=tk.DISABLED)

    def _toggle_pin(self) -> None:
        idx = self._selected_note_index()
        if idx is None:
            return
        meta = self._note_index[idx]
        note = self.app.notebook.read_note(meta["id"])
        if note is None:
            return
        self.app.notebook.update_note(note.meta.id, set_pinned=not note.meta.pinned)
        self._refresh_notebook_list()

    def _selected_note_index(self) -> Optional[int]:
        sel = self.note_list.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if idx >= len(self._note_index):
            return None
        return idx

    def _show_librarian_log(self) -> None:
        path = self.app.config.librarian_log_path
        if not path.exists():
            messagebox.showinfo("Librarian log", "No librarian activity recorded yet.", parent=self.root)
            return
        try:
            tail = path.read_text(encoding="utf-8").splitlines()[-10:]
        except OSError as exc:
            messagebox.showerror("Librarian log", str(exc), parent=self.root)
            return
        win = tk.Toplevel(self.root)
        win.title("Librarian audit log (last 10 runs)")
        win.geometry("780x500")
        text = ScrolledText(win, wrap=tk.WORD, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True)
        for raw in tail:
            try:
                entry = json.loads(raw)
            except ValueError:
                text.insert(tk.END, raw + "\n")
                continue
            ts = entry.get("ts", "")
            sid = entry.get("session_id", "")
            text.insert(tk.END, f"=== {ts} session {sid} ===\n")
            if entry.get("error"):
                text.insert(tk.END, f"  error: {entry['error']}\n")
            for op in entry.get("operations") or []:
                text.insert(tk.END, f"  {op.get('tool')}: {op.get('args')}\n")
            if entry.get("final_message"):
                text.insert(tk.END, f"  final: {entry['final_message']}\n")
            text.insert(tk.END, "\n")
        text.configure(state=tk.DISABLED)

    # -------------------------------------------------------------- close
    def _on_close(self) -> None:
        try:
            # End session synchronously so summary + librarian commit before exit.
            self.app.end_session()
        finally:
            self.root.destroy()

    @staticmethod
    def _format_error(exc: Exception) -> str:
        return "{}\n\n{}".format(exc, traceback.format_exc(limit=3))


def launch_desktop_ui(seed_memory: str = "feeling overwhelmed") -> None:
    config = AppConfig.load()
    app = TherapistApp(config)
    # Warm memory off the main thread so the window pops fast.
    threading.Thread(target=app.warm_memory, args=(seed_memory,), daemon=True).start()
    root = tk.Tk()
    DesktopUI(root, app)
    root.mainloop()


if __name__ == "__main__":
    launch_desktop_ui()
