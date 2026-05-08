# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository at a glance

FinNavigator is a multi-agent financial intelligence app. Today's stack is:
- **Backend**: FastAPI server (`api/server.py`) hosting a LangChain supervisor + 3 specialised agents over a hybrid RAG (Chroma + BM25 + Flashrank).
- **Frontend**: Flet 0.84 single-codebase UI (`ui/`) targeting web (Cloudflare Pages), desktop (`flet build windows/macos/linux`), and PWA.
- **Inference**: 5 swappable LLM backends behind one adapter (`inference/backend.py`).

The original Streamlit UI is kept at `legacy_streamlit_app.py` for one milestone of parity testing.

## Common commands

All commands assume the repo root and `.venv` activated (`.\.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` elsewhere).

```bash
# Run API + UI locally (two terminals)
LLM_BACKEND=openai python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --log-level warning
FINNAV_API_URL=http://127.0.0.1:8000 python -m ui.app

# Quick syntax sanity (no full test suite exists yet — there's no pytest)
python -c "import ast, glob; [ast.parse(open(f, encoding='utf-8').read()) for f in glob.glob('**/*.py', recursive=True) if '.venv' not in f]"

# Smoke test the live API (every endpoint)
B=http://127.0.0.1:8000
curl -s $B/api/health
curl -s $B/api/config
curl -s -X POST $B/api/agent/run -H "Content-Type: application/json" -d '{"prompt":"What is 2+2?","show_reasoning":false}'
curl -N -s -X POST $B/api/agent/stream -H "Content-Type: application/json" -d '{"prompt":"hi","show_reasoning":false}'

# Switch backends at runtime (no restart needed)
curl -s -X POST $B/api/config/backend -H "Content-Type: application/json" -d '{"name":"llama"}'

# Pull the fine-tuned GGUF locally (3.3 GB, gitignored under models/)
python -c "from huggingface_hub import hf_hub_download; \
  [hf_hub_download('MOH749/finnav-qwen3-VL-4b-gguf', f, local_dir='models/finnav-gguf') \
   for f in ['Qwen3-VL-4B-Instruct.Q4_K_M.gguf','Qwen3-VL-4B-Instruct.F16-mmproj.gguf']]"

# Deploy
flyctl deploy -a finnavigator-api --remote-only          # backend → Fly.io
git push origin main                                      # frontend → Cloudflare Pages auto-deploys via cloudflare-build.sh
```

## Architecture you need to know

### Inference adapter — `inference/backend.py`

`get_llm(backend, model=None, temperature=None)` returns a LangChain chat model for one of: `ollama`, `openai`, `anthropic`, `nvidia` (NIM Catalog), `llama` (llama-cpp-python loading the local GGUF), or `webgpu` (raises — browser-only stub).

**Per-backend env vars are intentionally separate** to avoid collision: `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `NIM_MODEL`, `OLLAMA_MODEL`, `LLAMA_GGUF_PATH`. There is no shared `MODEL_NAME` — each backend reads its own slot.

`BackendError` is raised for missing keys / packages / model files. The API layer (`api/server.py`) catches and returns 503.

### Agent team — `agents/` and `api/server.py:_build_tools_and_team`

`SupervisorAgent` classifies an incoming prompt into one of `QUERY/RESEARCH/ANALYSIS/COMPLEX/EXECUTION` and delegates to `FinancialAgent`, `ResearchAgent`, or `AnalystAgent`. Each sub-agent is built fresh on every backend switch — `_build_tools_and_team` rebuilds the whole team using `state.enabled_tools` to filter.

**Pre-existing supervisor quirk**: prompts containing keywords like "filing", "risk", "research" classify as RESEARCH and may decompose to 0 subtasks, returning `"No results"`. The streaming endpoint `/api/agent/stream` bypasses the supervisor entirely (calls `state.llm.astream()` directly) — that's the path to use when the supervisor's tool-binding is the wrong shape (Ollama/llama-cpp can't bind tools).

### Two chat paths, on purpose

| Endpoint | Goes through | When |
|---|---|---|
| `POST /api/agent/run` | Supervisor → sub-agents → tools → LLM | Tool-using queries, complex research |
| `POST /api/agent/stream` | LLM directly (SSE) | Conversational chat where streaming matters |

The Flet chat page tries `agent_stream` first and falls back to `agent_run` on error or empty content.

### Tool registry — `state.tools_by_name`, `state.enabled_tools`

10 tools registered per agent build: `calculator`, `datetime`, `knowledge_search`, `knowledge_index`, `index_visual_context`, `send_message` (Voiceflow), `send_alert`, `sec_search`, `sec_extract`, `web_search` (DuckDuckGo, no auth).

`POST /api/tools/enabled {enabled: [...]}` filters which tools bind to agents. The Chat page renders these as toggleable pills above the input. Empty `enabled_tools` set = all enabled (startup default).

`POST /api/tools/{name}/run {args: {...}}` runs a tool directly without the agent — used by the Monitor playground panel.

### Persistent state lives in `data/` (gitignored)

| Path | Owner |
|---|---|
| `data/chroma_db/` | Chroma RAG vectorstore |
| `data/chat_history.jsonl` | `services/chat_history.py` — append-only |
| `data/schedules.json` | `services/scheduler.py` — APScheduler job store |
| `data/uploads/` | `services/uploads.py` — file ingest target |

The API server is single-tenant — there's no per-user partitioning yet.

### Flet 0.84 quirks to watch for

The UI is on Flet 0.84 which made several breaking renames vs. older docs/examples:

- `ft.padding.X / ft.border.X / ft.margin.X` → use `ft.Padding.X / ft.Border.X / ft.Margin.X` (capitalised)
- `ft.alignment.center` → `ft.Alignment(0, 0)`
- `ColorScheme(background=…, on_background=…)` — both kwargs were dropped (Material 3)
- `page.session.get/set` — gone. Use plain attributes on `page`, e.g. `page.research_prefill_ticker = "AAPL"`
- `page.open(dialog) / page.close(dialog)` — gone. Append dialog to `page.overlay`, then `dialog.open = True; page.update()`
- `ft.TextField(font_family=…)` — gone. Use `text_style=ft.TextStyle(font_family=…)`
- `ft.TextButton(text="…")` — gone. Use `content=ft.Text("…")`
- `ft.PieChart / ft.BarChart` — moved out of `flet` core into the separate `flet-charts` package: `import flet_charts as fc; fc.PieChart(...)`. `BarChartGroup(rods=...)` not `bar_rods`. `BarChart(groups=...)` not `bar_groups`.
- `FilePicker` — needs platform plugins. Works in `flet build` / `flet run`, **not** in `python -m ui.app`. The chat page uses an `AlertDialog` + path-text-field workaround for that mode.

### Linear-style design system — `ui/theme.py`

Class `T` exposes spacing tokens (`S1`–`S6` on a 4 px grid), radii, status colors, and a Linear-purple `ACCENT`. The `linear_theme()` builder produces `ft.Theme` with these mapped into Material's `ColorScheme`. Reach into `T.X` directly from any UI file — that's the canonical access pattern, not via Flet's theme tokens.

### Inference backends — operational notes

- **Ollama 0.23.2 has a Conv3D nil-pointer crash on Qwen3-VL** (`qwen3vl.(*VisionModel).Forward`). The local fine-tune cannot be served via Ollama on this version. Workaround: use the `llama` backend (llama-cpp-python). When testing on a fresh machine, expect Ollama to crash on `ollama run finnav-qwen3-vl:Q4_K_M` even though `ollama create` succeeded.
- **`/api/agent/run` with Ollama or llama-cpp returns 400** ("model does not support tools"). LangChain tries to bind tools and the GGUF doesn't expose function-calling. Use `/api/agent/stream` for these backends, or switch to a cloud backend (`openai`, `nvidia`) for tool-using paths.
- **NIM default model is `qwen/qwen3-next-80b-a3b-instruct`**. Several models in the catalog (e.g., `meta/llama-3.1-70b-instruct`) are listed but return 404 when called. If you change the default, verify with a test call first.

### Fine-tune

The training pipeline lives in the (gitignored) `finnav_unsloth_qwen3_4b (1).ipynb`. See `docs/dataset.md` for the dataset breakdown — three HuggingFace sources (`PatronusAI/financebench`, `virattt/financial-qa-10K`, `zeroshot/twitter-financial-news-sentiment`) plus a hand-curated 60-entry SEC glossary, mixed under task-prefix tokens (`[SEC]` / `[QA]` / `[SENTIMENT]` / `[GLOSSARY]`).

The current GGUF on HF (`MOH749/finnav-qwen3-VL-4b-gguf`) was produced from run `wi19zc3g` which stopped at step 1100 (1.3 of 3 epochs) before Colab disconnected. Loss curve looked healthy (train 2.86 → 0.998, eval 1.123) but it's under-cooked vs. plan. To re-run with crash protection, change `output_dir="outputs"` → `output_dir="/content/drive/MyDrive/finnav/outputs"` in the training cell.

## Important repo paths

- `legacy_streamlit_app.py` — the old Streamlit app, scheduled for deletion next milestone. Don't add features to it; mirror them in `ui/pages/`.
- `models/` — local fine-tuned GGUF + mmproj. **Gitignored, never commit.** Pulled from HF on first run.
- `data/` — runtime state (chat history, schedules, Chroma, uploads). **Gitignored.**
- `cloudflare-build.sh` — installs Flutter SDK on Cloudflare Pages' build env, runs `flet build web`. Each step prints a banner so the failed step is obvious in Pages logs.
- `desktop/launcher.py` — desktop binary entry. Boots Ollama → uvicorn → Flet in order. Used by `flet build windows/macos/linux`.
- `Modelfile` (root) — Ollama definition for the fine-tune. **Currently broken on Ollama 0.23.2** (see Conv3D note above). Kept for the eventual fix.
- `docs/dataset.md`, `docs/desktop.md` — authoritative reference for fine-tune data and desktop install.

## Deploy targets

- **API**: Fly.io app `finnavigator-api` → `https://finnavigator-api.fly.dev`. Single shared-cpu-2x machine in `iad`, auto-stops when idle. Secrets via `flyctl secrets set` (never `.env` on Fly). Persistent volume `finnav_data` mounted at `/data`.
- **Frontend**: Cloudflare Pages, GitHub-integrated, auto-deploys on push to `main`. Build runs `cloudflare-build.sh`. Output `build/web/`.
- **Desktop**: `flet build windows` (or `macos` / `linux`). Requires Flutter SDK installed locally.

## Security gotchas

- `.env` is gitignored but contains live keys (OpenAI / NVIDIA / SEC / wandb / etc.). Never echo values; pass through subprocess args when pushing to Fly secrets.
- The fine-tuning notebook (`finnav_unsloth_qwen3_4b (1).ipynb`) is gitignored because earlier cells contained a hardcoded HF write token. Even after rotation, keep it gitignored — it's a local Colab artifact, not part of the deployable repo.
- `.dockerignore` is intentionally aggressive (`models/`, `data/`, `*.ipynb`, `.env*`, `*.png`) so the build context stays under 10 MB. If you add a new directory that contains source the API server needs, allow-list it explicitly.
