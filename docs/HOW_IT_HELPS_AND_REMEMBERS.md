# How Therapist Engine Helps You and Remembers Each Session

This document explains, in detail, two things:

1. The concrete ways the current Therapist Engine implementation is designed to help you.
2. Exactly what gets stored, where it is stored, and how it is recalled between sessions.

It reflects the actual behavior of the code in this repository today, not aspirational features. Where something is partially implemented, it is called out.

---

## Part 1 — How the App Is Designed to Help

Therapist Engine is a **fully local, conversational support tool**. It is not a clinician, not a crisis service, and not a medical device. It is built to give you a private space to talk through what you are feeling, with a few specific guardrails layered on top of a language model.

The help it provides comes from four cooperating layers.

### Layer 1 — A guided therapist persona

Every conversation is anchored to the system prompt in [prompts/therapist_system.txt](prompts/therapist_system.txt). That prompt instructs the model to:

- Use **person-centered and motivational interviewing** style language.
- **Reflect emotions first** before offering perspective.
- Ask **one open, grounded question at a time** rather than interrogating or overwhelming you.
- **Not diagnose** and **not claim to replace a licensed clinician**.
- Prioritize emergency resources if you appear to be in immediate danger.

In practice this means replies are shaped to:

- Acknowledge what you said in your own words.
- Slow the pace down instead of jumping straight to fixes.
- Surface one prompt at a time, so a session feels like a conversation, not a checklist.

This persona is reapplied on every turn, because the system prompt is held at the head of the conversation state inside [src/llm_therapist.py](src/llm_therapist.py).

### Layer 2 — A dedicated safety pass on every message

Before the therapist model ever sees your message and generates a reply, a **second, separate model** screens it. That logic lives in [src/safety.py](src/safety.py) and uses the prompt in [prompts/safety_classifier.txt](prompts/safety_classifier.txt).

The safety model is asked to return JSON with:

- `flagged` — boolean
- `risk_level` — `low`, `medium`, `high`, or `imminent`
- `reason` — short string

Two outcomes are possible:

- **Not flagged** — your message flows into the normal therapist pipeline and you get a regular streamed reply.
- **Flagged** — the normal therapist path is **bypassed**. Instead, the app generates a **crisis-aware reply** built from [prompts/crisis_response.txt](prompts/crisis_response.txt), which is written to:
  - Stay calm and compassionate.
  - Encourage contacting local emergency services if you may be in immediate danger.
  - Mention the **988 Suicide & Crisis Lifeline** for US users.
  - Avoid overwhelming you with too many steps.
  - Encourage reaching a trusted person nearby when appropriate.

This is the most important "help" feature beyond conversation itself: it means a single bad output from the main model cannot quietly override safety messaging. The decision to switch into crisis mode is made by a different model with a different prompt.

If the safety model returns malformed JSON, the app falls back to `flagged=False` with `risk_level="unknown"` and continues with a normal reply. This is a known limitation and is documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#safety-layer).

### Layer 3 — Continuity between sessions through memory

The app is designed so each session is not isolated. At startup, it queries your local memory store with a seed phrase (default: `"feeling overwhelmed"`, configurable via `--seed-memory`) and injects any relevant past **session summaries** back into the therapist's context as additional system messages. See `TherapistApp.warm_memory()` in [src/main.py](src/main.py).

The therapist model then enters the conversation already aware of recurring themes from previous sessions, without you having to repeat your history every time. Part 2 of this document covers exactly what is stored and how.

### Layer 4 — Multiple ways to talk

You can engage in whichever mode is least friction at that moment:

- **Desktop GUI** (`python -m src.main --mode gui`) — typed conversation with streaming replies, plus a `Listen Once` button and a `Speak replies` checkbox.
- **CLI text mode** (`python -m src.main --mode text`) — terminal chat, `quit` to end.
- **One-shot mode** (`python -m src.main --once "..."`) — single turn, useful when you just want to dump a thought.
- **Voice mode** (`python -m src.main --mode voice`) — captures audio, transcribes with `faster-whisper`, runs the same therapist pipeline, and optionally speaks back with XTTS.

Voice support requires Python 3.10+ and the full audio stack. See [docs/USAGE.md](docs/USAGE.md) and [README.md](README.md) for the current status of the speech path.

### What the app does **not** do

To set expectations honestly:

- It does not contact emergency services for you.
- It does not detect crisis from voice tone, only from the transcribed text.
- It does not provide diagnosis, treatment plans, or medication advice.
- It does not retain a verbatim transcript of past sessions (see Part 2).
- It does not share anything with the cloud. All inference happens through your local Ollama instance.

---

## Part 2 — How Memory and Progress Are Stored

This is the part most people get wrong when they assume an LLM "remembers" them. The behavior here is specific and bounded.

### What "memory" means in this app

Therapist Engine uses **session-summary memory**, not full-transcript memory. That is, when a session ends, the app asks the therapist model to produce a short summary of the conversation, embeds that summary, and stores it. Future sessions search those summaries by similarity and feed the most relevant ones back into the model's context.

This is implemented entirely in [src/memory.py](src/memory.py).

### Where data lives on disk

All persistent data is written under a single local folder created automatically at first run:

- `.data/` (project root)
  - `.data/sessions/` — one JSON file per session, containing the generated summary.
  - `.data/chroma/` — the Chroma vector database files used for similarity search.

Paths are defined in [src/config.py](src/config.py) (`session_dir`, `chroma_dir`).

Nothing is uploaded. Nothing leaves your machine except the local HTTP call to Ollama at `http://127.0.0.1:11434`.

### The within-session transcript

While a session is running, the app keeps an in-memory list called `transcript` on `TherapistApp` (see [src/main.py](src/main.py)). For every turn it appends two lines:

- `"user: " + user_text`
- `"assistant: " + reply`

This list is the input to summarization at shutdown. It is **not** written to disk as a raw transcript. If the process crashes before shutdown, the in-memory transcript is lost and no summary is produced for that session.

### The conversation state passed to the model

Separately from `transcript`, the therapist model itself receives a structured message list managed by `ConversationState` in [src/llm_therapist.py](src/llm_therapist.py). It contains:

1. The therapist system prompt (always first).
2. Any retrieved memory snippets from past sessions, inserted as additional `system` messages by `warm_memory()`.
3. All user and assistant turns from the current session.

So during a single session, the model has full short-term recall of everything you have said since you launched the app. Across sessions, it only sees the **summaries** described below.

### What happens at session end

In `TherapistApp._persist_session()` ([src/main.py](src/main.py)):

1. The full in-memory `transcript` list is passed to `MemoryStore.summarize()`.
2. `summarize()` ([src/memory.py](src/memory.py)) sends the transcript to the therapist model with this instruction:

    > Summarize the session in a short paragraph focusing on persistent themes, coping concerns, and follow-up topics.

3. The returned summary is the **only durable record** of the session.
4. A `session_id` is generated from the current UTC timestamp in the format `YYYYMMDDHHMMSS`.
5. `MemoryStore.remember_session(session_id, summary)` is called.

Inside `remember_session()`:

1. The summary is embedded via Ollama's `/api/embed` using the `nomic-embed-text:latest` model (see `OllamaClient.embed()` in [src/llm_therapist.py](src/llm_therapist.py)).
2. The summary, its embedding, and the session id are **upserted** into the Chroma collection named `therapist_sessions` under `.data/chroma/`.
3. A JSON file is written to `.data/sessions/<session_id>.json` with the shape:

    ```json
    { "summary": "..." }
    ```

That JSON file is currently a convenience copy; the operational store used for retrieval is Chroma.

Session persistence is triggered by:

- The `finally:` block of `run_text_loop()` and `run_voice_loop()` (normal exit, including `quit`/`exit` in text mode or Ctrl+C in voice mode).
- The explicit call at the end of `--once` mode.

If the GUI is closed without going through the standard shutdown path, the session may not be persisted. This is a current limitation of the desktop UI integration.

### What happens at session start

In `TherapistApp.warm_memory(seed_text)` ([src/main.py](src/main.py)):

1. The seed text (default `"feeling overwhelmed"`, override with `--seed-memory "..."`) is embedded.
2. Chroma is queried for the top `n` most similar past session summaries. The default in `MemoryStore.retrieve()` is `limit=3`.
3. Each result is wrapped in a `RetrievedMemory` with the summary text and a similarity distance.
4. The summary texts are handed to `TherapistEngine.add_memory_context()`, which inserts them into the conversation as additional `system` messages **before** any user turn.

The therapist then opens the conversation with those summaries already in context. This is what lets it pick up on recurring themes from earlier sessions.

### What "progress" looks like in this design

There is no explicit progress tracker, scoring, mood graph, or goal database in the current code. "Progress" emerges from two mechanisms:

1. The accumulated set of session summaries in Chroma. Over time, repeated themes will tend to surface in retrieval, and the summarization instruction explicitly asks for "persistent themes, coping concerns, and follow-up topics."
2. The model's behavior in-session, where it builds on the injected past summaries to reference patterns rather than treating every session as new.

If you want richer progress tracking (mood ratings, goals, weekly trends), that is not implemented yet and would be a future feature on top of `MemoryStore`.

### Memory quality, caveats, and failure modes

These are real behaviors of the current code you should know about:

- **Summaries are only as good as the model that produced them.** A short or rambling session can produce a weak summary, which then becomes a weak retrieval result later.
- **Retrieval is similarity-based, not chronological.** The app does not preferentially load the most recent session; it loads whatever is most similar to the seed phrase. If your seed phrase is always the default, retrieval is biased toward sessions about being overwhelmed.
- **Embeddings depend on Ollama being up.** If Ollama is not running at startup, `warm_memory()` will fail to retrieve and the session will run without prior context.
- **Chroma is optional at import time.** If `chromadb` is not installed, `_ensure_collection()` returns `None`, and the app silently runs without memory. Summaries will not be stored either.
- **No deletion or redaction UI.** To erase memory, you currently delete the contents of `.data/sessions/` and `.data/chroma/` manually. There is no in-app forget-this-session command yet.
- **The raw transcript is never persisted.** If you want a verbatim record, you need to copy it from the UI before closing.

### Quick reference: where each piece lives

| Concept | Code | On-disk location |
| --- | --- | --- |
| Therapist persona | [prompts/therapist_system.txt](prompts/therapist_system.txt) | — (loaded into prompt) |
| Safety classifier prompt | [prompts/safety_classifier.txt](prompts/safety_classifier.txt) | — |
| Crisis reply prompt | [prompts/crisis_response.txt](prompts/crisis_response.txt) | — |
| In-session transcript | `TherapistApp.transcript` in [src/main.py](src/main.py) | RAM only |
| Conversation messages sent to LLM | `ConversationState` in [src/llm_therapist.py](src/llm_therapist.py) | RAM only |
| Session summary generator | `MemoryStore.summarize()` in [src/memory.py](src/memory.py) | — |
| Session summaries (JSON) | `MemoryStore.remember_session()` | `.data/sessions/<session_id>.json` |
| Vector memory | `MemoryStore._ensure_collection()` (Chroma `therapist_sessions`) | `.data/chroma/` |
| Memory retrieval at startup | `TherapistApp.warm_memory()` in [src/main.py](src/main.py) | reads `.data/chroma/` |

---

## Reminder

This tool is intended to be a supportive, private space for self-reflection. It is **not** a substitute for a licensed clinician, emergency services, or a crisis line. If you may be in immediate danger, contact local emergency services. In the United States, you can call or text **988** for the Suicide & Crisis Lifeline.
