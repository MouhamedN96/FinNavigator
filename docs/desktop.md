# FinNavigator Desktop

The desktop build is a single binary that runs everything locally:
the Flet UI, the FastAPI server, and Ollama serving the fine-tuned Qwen3-VL.

## Prerequisites

1. **Install Ollama** — https://ollama.com/download (Win / Mac / Linux)
   The launcher detects Ollama and starts the daemon automatically.
2. ~6 GB free disk for the fine-tuned GGUF (`hf.co/MOH749/finnav-qwen3-VL-4b-gguf:Q4_K_M`).
3. ≥ 8 GB RAM. 16 GB if you also enable the local vision module.

## Build

```bash
pip install -r requirements.txt -r requirements-ui.txt
flet build windows           # or: macos | linux
```

The output binary lives under `build/<platform>/`. It launches `desktop/launcher.py`
which orchestrates Ollama → uvicorn → Flet (see that file for the boot order).

## First launch

On first run the launcher:
1. Verifies Ollama is installed (otherwise surfaces the install link).
2. Spawns `ollama serve` if no daemon is up.
3. `ollama pull hf.co/MOH749/finnav-qwen3-VL-4b-gguf:Q4_K_M` if the model isn't local. Expect ~2.5 GB download once.
4. Spawns `uvicorn api.server:app` on `127.0.0.1:8000` with `LLM_BACKEND=ollama`.
5. Opens the Flet window pointed at the local API.

Total cold-start: ~15 s after the model is cached, ~5–10 min on first ever launch (mostly the `ollama pull`).

## Switching to NVIDIA NIM (cloud)

In the app settings, set backend = `nvidia`. The launcher restarts uvicorn with
`LLM_BACKEND=nvidia`; you'll need `NVIDIA_API_KEY` set in your environment.
Inference moves off your machine, so the local Ollama daemon is unused but still running.

## Troubleshooting

- **"Ollama is not installed"** — install from https://ollama.com/download and relaunch.
- **Slow first message** — cold model load. Subsequent messages are fast.
- **Port 8000 in use** — set `FINNAV_DESKTOP_API_PORT=8123` (or any free port) in env before launching.
- **Model pull fails** — check internet, then `ollama pull hf.co/MOH749/finnav-qwen3-VL-4b-gguf:Q4_K_M` manually from a terminal.
