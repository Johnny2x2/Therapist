# RunPod Serverless Deployment

How to deploy the Therapist engine to RunPod Serverless and connect your local
desktop UI to it. The LLM (via Ollama), safety, memory, and optional TTS run in
the RunPod worker; speech capture (STT) and audio playback stay on your local
machine.

> **Mount path note**
> - **Serverless** mounts the network volume at **`/runpod-volume`** (default in
>   the Dockerfile and entry script).
> - **Pods** mount it at **`/workspace`**.
>
> All paths are environment-driven, so you can point at `/workspace` instead by
> overriding `OLLAMA_MODELS` and `THERAPIST_DATA_DIR` (see below).

---

## Architecture

```
Local PC (client)                         RunPod Serverless worker
-----------------                         ------------------------
Desktop UI (Tkinter)                      serverless_handler.handler
  - mic capture (STT)        HTTPS         - TherapistApp (LLM, safety, memory)
  - audio playback (TTS) <------------>    - Ollama (model on network volume)
  RunPodTherapistApp                       - optional server-side TTS
                                           Network volume (/runpod-volume)
                                             - ollama/models  (your model)
                                             - therapist-data (sessions, chroma)
```

Session continuity: the per-turn transcript is written to the network volume and
**replayed into the engine** when a (possibly cold) worker picks the session back
up, so conversations survive worker recycling. Long-term memory (Chroma +
notebook) also lives on the volume.

---

## 1. Prepare the network volume

Create a RunPod **Network Volume** and lay it out like this (do this once, e.g.
from a temporary Pod that mounts the same volume):

```
/runpod-volume/
  ollama/
    models/        <- your pre-pulled Ollama model lives here
  therapist-data/  <- created automatically; sessions, chroma, notebook
```

To pre-pull the model onto the volume from a temporary Pod:

```bash
export OLLAMA_MODELS=/runpod-volume/ollama/models   # or /workspace/ollama/models on a Pod
ollama serve &
ollama pull tripolskypetr/qwen3.5-uncensored-aggressive:9b
```

Once the model is on the volume, the serverless worker reuses it on every cold
start without re-downloading.

---

## 2. Build and push the image

From the repo root:

```bash
docker build -t <your-registry>/therapist-serverless:latest .
docker push <your-registry>/therapist-serverless:latest
```

The image (`Dockerfile`) installs Ollama + the server-only Python deps
(`requirements-server.txt`) and runs `scripts/serverless_entry.sh`, which boots
`ollama serve` and then the handler.

---

## 3. Create the Serverless endpoint

In the RunPod console, create a **Serverless Endpoint**:

- **Container image:** `<your-registry>/therapist-serverless:latest`
- **Network volume:** attach the volume from step 1
- **GPU:** size for the 9B model (e.g. 24 GB class)
- **Environment variables:**

| Variable | Value | Notes |
|---|---|---|
| `OLLAMA_MODELS` | `/runpod-volume/ollama/models` | Use `/workspace/...` on a Pod |
| `THERAPIST_DATA_DIR` | `/runpod-volume/therapist-data` | Sessions/memory on the volume |
| `THERAPIST_API_KEY` | `<your-secret>` | App-level auth; client must match |
| `THERAPIST_PULL_MODEL` | `tripolskypetr/qwen3.5-uncensored-aggressive:9b` | Optional; only needed if the model isn't already on the volume |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keep the model warm between turns |

> **Cold starts:** the first request after the worker scales from zero reloads
> the model into VRAM and is slow. Set a minimum active worker (or enable
> FlashBoot) if you need snappy first responses.

After it deploys, note the **Endpoint ID**.

---

## 4. Connect the local client

On your PC (in the repo, with the venv active):

```powershell
$env:THERAPIST_RUNPOD_ENDPOINT_ID = "<endpoint-id>"
$env:THERAPIST_RUNPOD_API_KEY     = "<runpod-api-key>"   # from RunPod account settings
$env:THERAPIST_API_KEY            = "<your-secret>"       # MUST match the endpoint
python -m src.main --mode gui
```

When `THERAPIST_RUNPOD_ENDPOINT_ID` is set, the desktop UI automatically uses
`RunPodTherapistApp` and routes every turn to your endpoint. Microphone capture
and speech playback still happen locally.

To go back to fully local mode, just clear those variables:

```powershell
Remove-Item Env:THERAPIST_RUNPOD_ENDPOINT_ID
python -m src.main --mode gui
```

---

## 5. Smoke-test the endpoint (optional)

A minimal request via the RunPod job API:

```powershell
$headers = @{ Authorization = "Bearer $env:THERAPIST_RUNPOD_API_KEY"; "Content-Type" = "application/json" }
$body = @{ input = @{ session_id = "smoke1"; user_text = "I feel overwhelmed today."; api_key = $env:THERAPIST_API_KEY } } | ConvertTo-Json
$run = Invoke-RestMethod -Method Post -Uri "https://api.runpod.ai/v2/$env:THERAPIST_RUNPOD_ENDPOINT_ID/run" -Headers $headers -Body $body
Invoke-RestMethod -Method Get -Uri "https://api.runpod.ai/v2/$env:THERAPIST_RUNPOD_ENDPOINT_ID/stream/$($run.id)" -Headers $headers
```

You should see streamed `status` and `text` events, ending with `COMPLETED`.

---

## Environment variable reference

| Variable | Side | Default | Purpose |
|---|---|---|---|
| `OLLAMA_MODELS` | server | `/runpod-volume/ollama/models` | Where Ollama reads models |
| `OLLAMA_HOST` | server | `http://127.0.0.1:11434` | Ollama API the app calls |
| `OLLAMA_KEEP_ALIVE` | server | `5m` | Keep model in VRAM |
| `THERAPIST_DATA_DIR` | server | `/runpod-volume/therapist-data` | Sessions/chroma/notebook |
| `THERAPIST_PULL_MODEL` | server | (unset) | Pull this model on boot if set |
| `THERAPIST_API_KEY` | both | (unset = no auth) | App-level auth secret |
| `THERAPIST_RUNPOD_ENDPOINT_ID` | client | (unset) | Selects RunPod mode in the UI |
| `THERAPIST_RUNPOD_API_KEY` | client | (unset) | RunPod account API key |
| `THERAPIST_RUNPOD_BASE` | client | `https://api.runpod.ai/v2` | Override API base URL |

---

## Troubleshooting

- **`unauthorized` in responses:** `THERAPIST_API_KEY` on the client doesn't match
  the endpoint's value.
- **First request very slow / times out:** cold start loading the 9B model. Keep a
  warm worker or raise the client request timeout.
- **Model re-downloads every start:** `OLLAMA_MODELS` isn't pointing at the volume,
  or the model was never pulled onto it (step 1).
- **No audio on the client:** confirm `speak=True` path and that local audio output
  works; server-side TTS failures are non-fatal and fall back to text.
- **Lost conversation after a gap:** confirm `THERAPIST_DATA_DIR` points at the
  persistent volume so transcripts can be replayed on resume.
