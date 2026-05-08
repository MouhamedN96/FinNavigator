"""FastAPI server for FinNavigator.

Replaces the Streamlit `app.py` request layer. The same agent/tool/memory
modules are reused unchanged — this file is just transport.

Run locally:
    uvicorn api.server:app --reload --port 8000

Cloud deploy:
    Same Dockerfile, CMD changed to `uvicorn api.server:app --host 0.0.0.0 --port $PORT`.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents import AgentConfig, AnalystAgent, ResearchAgent
from agents.supervisor_agent import AgentTeam
from inference import BackendError, get_llm
from memory import MemoryManager, create_memory_manager
from services import chat_history, uploads
from services.scheduler import service as schedule_service
from tools import (
    AlertTool,
    CalculatorTool,
    DateTimeTool,
    KnowledgeBaseIndexTool,
    KnowledgeBaseSearchTool,
    SECExtractTool,
    SECSearchTool,
    SendMessageTool,
    VisualContextIndexTool,
    WebSearchTool,
    get_sec_edgar_tools,
)

log = logging.getLogger("finnav.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


class AppState:
    """Singleton bag of long-lived objects. Initialised once at startup."""
    llm: Any = None
    team: Optional[AgentTeam] = None
    supervisor: Any = None
    memory: Optional[MemoryManager] = None
    vectorstore: Any = None
    embeddings: Any = None
    backend: str = ""
    tools_by_name: Dict[str, Any] = {}        # name → tool instance (full inventory)
    tools_by_agent: Dict[str, List[str]] = {} # agent name → list of tool names (full inventory)
    enabled_tools: set = set()                # subset that gets bound to agents; empty = all enabled
    alerts: List[Dict[str, Any]] = []         # in-memory alert list
    next_alert_id: int = 1


state = AppState()


def _init_vectorstore():
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        os.makedirs(os.getenv("PERSIST_DIRECTORY", "data/chroma_db"), exist_ok=True)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        store = Chroma(
            persist_directory=os.getenv("PERSIST_DIRECTORY", "data/chroma_db"),
            embedding_function=embeddings,
            collection_name="financial_docs",
        )
        return store, embeddings
    except Exception as e:
        log.warning("Vectorstore init failed: %s", e)
        return None, None


def _build_tools_and_team(llm) -> tuple[AgentTeam, Any]:
    """(Re)build the agent team for the given LLM. Reuses already-loaded vectorstore.

    Honours `state.enabled_tools` — only tools whose name is in the set are bound
    to the agents. Empty set means "all tools enabled" (the startup default).
    """
    calc = CalculatorTool()
    dt = DateTimeTool()
    kb_search = KnowledgeBaseSearchTool(vectorstore=state.vectorstore, embeddings=state.embeddings)
    kb_index = KnowledgeBaseIndexTool(vectorstore=state.vectorstore)
    visual_index = VisualContextIndexTool(vectorstore=state.vectorstore)
    msg = SendMessageTool()
    alert = AlertTool(msg)
    sec = get_sec_edgar_tools()
    web = WebSearchTool()

    full_financial = [calc, dt, kb_search, kb_index, visual_index, msg, alert, web] + sec
    full_analyst   = [calc, dt, kb_search, web]
    full_research  = sec + [kb_search, kb_index, visual_index, web]

    # Full inventory — what /api/tools and Monitor see (independent of toggle state)
    state.tools_by_name = {}
    state.tools_by_agent = {"financial": [], "research": [], "analyst": []}
    for t in full_financial:
        state.tools_by_name[t.name] = t
        state.tools_by_agent["financial"].append(t.name)
    for t in full_research:
        state.tools_by_name[t.name] = t
        if t.name not in state.tools_by_agent["research"]:
            state.tools_by_agent["research"].append(t.name)
    for t in full_analyst:
        state.tools_by_name[t.name] = t
        if t.name not in state.tools_by_agent["analyst"]:
            state.tools_by_agent["analyst"].append(t.name)

    # First-build default: all tools enabled. Subsequent builds use whatever the user set.
    if not state.enabled_tools:
        state.enabled_tools = set(state.tools_by_name.keys())

    def _filter(tools):
        return [t for t in tools if t.name in state.enabled_tools]

    financial_tools = _filter(full_financial)
    analyst_tools   = _filter(full_analyst)
    research_tools  = _filter(full_research)

    vision_path = os.path.join(os.getcwd(), "models", "finnav_qwen3-VL_4b_gguf")
    if not os.path.isdir(vision_path):
        vision_path = None

    team = AgentTeam(llm)
    supervisor = team.setup_team(
        financial_tools=financial_tools,
        research_tools=research_tools,
        analyst_tools=analyst_tools,
        vision_model_path=vision_path,
    )
    return team, supervisor


async def _schedule_runner(prompt: str) -> str:
    """Adapter the scheduler calls — runs a prompt through the supervisor and
    appends the result to chat history (so users see scheduled answers in chat)."""
    if not state.supervisor:
        raise RuntimeError("agent team not ready")
    resp = await state.supervisor.process(prompt)
    content = getattr(resp, "content", "") or ""
    chat_history.append("user", f"[scheduled] {prompt}", backend=state.backend)
    chat_history.append("assistant", content, backend=state.backend)
    return content


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the agent team once at startup so each request is cheap."""
    backend = os.getenv("LLM_BACKEND", "ollama")
    log.info("Initialising LLM backend=%s", backend)
    state.backend = backend
    state.llm = get_llm(backend)  # may raise BackendError
    state.memory = create_memory_manager()
    state.vectorstore, state.embeddings = _init_vectorstore()
    state.team, state.supervisor = _build_tools_and_team(state.llm)
    schedule_service.start(_schedule_runner)
    log.info("Agent team ready.")
    yield
    schedule_service.shutdown()
    log.info("Shutdown.")


app = FastAPI(title="FinNavigator API", version="2.0.0", lifespan=lifespan)

# CORS — restrict in prod via FINNAV_WEB_ORIGIN env (comma-separated)
_origins_env = os.getenv("FINNAV_WEB_ORIGIN", "*")
_origins = [o.strip() for o in _origins_env.split(",")] if _origins_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------- request/response schemas -------

class AgentRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image_b64: Optional[str] = None
    show_reasoning: bool = True


class AgentRunResponse(BaseModel):
    content: str
    reasoning_steps: List[Dict[str, Any]] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    ticker: str
    focus: str = "overview"
    depth: str = "standard"


class PortfolioPosition(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    current_price: float


class PortfolioRequest(BaseModel):
    positions: List[PortfolioPosition]
    analysis_type: str = "overview"  # overview | risk | sector | rebalance


# ------- routes -------

@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "backend": state.backend,
        "team_ready": state.team is not None,
    }


@app.post("/api/agent/run", response_model=AgentRunResponse)
async def agent_run(req: AgentRunRequest) -> AgentRunResponse:
    if not state.supervisor:
        raise HTTPException(503, "Agent team not ready.")
    try:
        if req.image_b64:
            resp = await state.supervisor.process_vision(req.prompt, req.image_b64)
        else:
            resp = await state.supervisor.process(req.prompt)
    except BackendError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("agent_run failed")
        raise HTTPException(500, f"agent error: {e}")

    state.memory.add_conversation("user", req.prompt)
    state.memory.add_conversation("assistant", resp.content)

    steps: List[Dict[str, Any]] = []
    if req.show_reasoning and getattr(resp, "reasoning_steps", None):
        for s in resp.reasoning_steps[-10:]:
            steps.append(s if isinstance(s, dict) else {"raw": str(s)})

    chat_history.append("user", req.prompt, backend=state.backend)
    chat_history.append("assistant", resp.content, backend=state.backend, reasoning=steps)

    return AgentRunResponse(content=resp.content, reasoning_steps=steps)


@app.post("/api/agent/stream")
async def agent_stream(req: AgentRunRequest) -> StreamingResponse:
    async def event_stream():
        chat_history.append("user", req.prompt, backend=state.backend)
        accumulated = ""
        try:
            messages = [
                SystemMessage(
                    content="You are a helpful financial assistant for FinNavigator. Answer concisely and accurately."
                ),
                HumanMessage(content=req.prompt),
            ]
            async for chunk in state.llm.astream(messages):
                token = chunk.content
                if token:
                    accumulated += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            chat_history.append("assistant", accumulated, backend=state.backend)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            if accumulated:
                chat_history.append("assistant", accumulated, backend=state.backend)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---- chat history ----

@app.get("/api/memory/history")
async def get_chat_history(limit: int = 100) -> Dict[str, Any]:
    """Return recent chat turns from the persistent JSONL store."""
    return {"turns": chat_history.history(limit=limit)}


@app.delete("/api/memory/clear")
async def clear_chat_history() -> Dict[str, Any]:
    removed = chat_history.clear()
    return {"ok": True, "removed": removed}


@app.post("/api/research")
async def research(req: ResearchRequest) -> Dict[str, Any]:
    cfg = AgentConfig(
        name="research",
        role="Research Analyst",
        description="SEC filings research",
        system_prompt="You are a research analyst specializing in SEC filings.",
    )
    agent = ResearchAgent(
        config=cfg,
        llm_client=state.llm,
        sec_api_key=os.getenv("SEC_API_KEY", ""),
    )
    try:
        return await agent.research_company(req.ticker, req.focus.lower(), req.depth.lower())
    except Exception as e:
        log.exception("research failed")
        raise HTTPException(500, f"research error: {e}")


@app.post("/api/portfolio/analyze")
async def portfolio_analyze(req: PortfolioRequest) -> Dict[str, Any]:
    cfg = AgentConfig(
        name="analyst",
        role="Portfolio Analyst",
        description="Portfolio and risk analysis",
        system_prompt="You are a portfolio analyst specializing in risk metrics and allocation.",
    )
    agent = AnalystAgent(config=cfg, llm_client=state.llm)
    agent.set_portfolio([p.model_dump() for p in req.positions])

    out: Dict[str, Any] = {"positions": [p.model_dump() for p in req.positions]}
    try:
        if req.analysis_type in ("risk", "overview"):
            out["risk"] = await agent.calculate_var()
        if req.analysis_type in ("sector", "overview"):
            out["sector"] = await agent.analyze_sector_exposure()
    except Exception as e:
        log.exception("portfolio_analyze failed")
        raise HTTPException(500, f"analysis error: {e}")
    return out


@app.get("/api/team/status")
async def team_status() -> Dict[str, Any]:
    if not state.team:
        return {"ready": False}
    status = state.team.get_team_status()
    if state.memory:
        status["memory"] = state.memory.get_memory_summary()
    return status


# ---- runtime backend switcher ----

class BackendSwitchRequest(BaseModel):
    name: str


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """Tells the UI which backend is active and which can be selected."""
    from inference import SUPPORTED_BACKENDS

    available: List[str] = []
    for b in SUPPORTED_BACKENDS:
        if b == "webgpu":
            continue  # browser-only; the server cannot host it
        try:
            get_llm(b)
            available.append(b)
        except BackendError:
            pass  # missing key / package / model file — not selectable today
    return {
        "current": state.backend,
        "supported": list(SUPPORTED_BACKENDS),
        "available": available,
    }


@app.post("/api/config/backend")
async def set_backend(req: BackendSwitchRequest) -> Dict[str, Any]:
    try:
        new_llm = get_llm(req.name)
    except BackendError as e:
        raise HTTPException(400, str(e))
    state.llm = new_llm
    state.backend = req.name
    state.team, state.supervisor = _build_tools_and_team(state.llm)
    log.info("Backend switched → %s", req.name)
    return {"ok": True, "backend": req.name}


# ---- tools ----

@app.get("/api/tools")
async def list_tools() -> Dict[str, Any]:
    """Catalog of registered tools and which agent owns each one."""
    tools = []
    for name, t in state.tools_by_name.items():
        owners = [a for a, names in state.tools_by_agent.items() if name in names]
        tools.append({
            "name": name,
            "description": (getattr(t, "description", "") or "").strip().split("\n")[0][:200],
            "owners": owners,
        })
    tools.sort(key=lambda x: x["name"])
    return {"tools": tools, "by_agent": state.tools_by_agent}


class ToolRunRequest(BaseModel):
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolEnableRequest(BaseModel):
    enabled: List[str]


@app.get("/api/tools/enabled")
async def get_enabled_tools() -> Dict[str, Any]:
    """Which tools are currently bound to the agents."""
    return {
        "enabled": sorted(state.enabled_tools),
        "all": sorted(state.tools_by_name.keys()),
    }


@app.post("/api/tools/enabled")
async def set_enabled_tools(req: ToolEnableRequest) -> Dict[str, Any]:
    """Replace the enabled-tool set, then rebuild the team so changes take effect."""
    valid = set(state.tools_by_name.keys())
    requested = set(req.enabled)
    invalid = requested - valid
    state.enabled_tools = requested & valid
    # Rebuild team with the new set
    state.team, state.supervisor = _build_tools_and_team(state.llm)
    return {
        "ok": True,
        "enabled": sorted(state.enabled_tools),
        "ignored": sorted(invalid),
    }


@app.post("/api/tools/{name}/run")
async def run_tool(name: str, req: ToolRunRequest) -> Dict[str, Any]:
    """Invoke a registered tool with raw arguments. Postman-for-the-toolbelt."""
    tool = state.tools_by_name.get(name)
    if not tool:
        raise HTTPException(404, f"tool {name!r} not found. registered: {list(state.tools_by_name)}")
    try:
        if hasattr(tool, "ainvoke"):
            result = await tool.ainvoke(req.args)
        else:
            result = tool.invoke(req.args)
        return {"ok": True, "result": str(result)}
    except Exception as e:
        log.exception("tool run failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---- channels (Voiceflow) ----

class ChannelTestRequest(BaseModel):
    user_id: str = "finnav_test_user"
    message: str = "FinNavigator test ping."


@app.get("/api/channels")
async def list_channels() -> Dict[str, Any]:
    """Status of each messaging channel."""
    return {
        "channels": [
            {
                "name": "voiceflow",
                "configured": bool(os.getenv("VOICEFLOW_API_KEY")),
                "purpose": "Send messages and alerts to WhatsApp / iMessage / etc.",
            },
            {"name": "slack",   "configured": False, "purpose": "Not yet wired."},
            {"name": "discord", "configured": False, "purpose": "Not yet wired."},
            {"name": "email",   "configured": False, "purpose": "Not yet wired."},
        ]
    }


@app.post("/api/channels/test")
async def test_channel(req: ChannelTestRequest) -> Dict[str, Any]:
    """Fire a test message at Voiceflow."""
    msg_tool = state.tools_by_name.get("send_message")
    if not msg_tool:
        raise HTTPException(503, "send_message tool not registered.")
    if not os.getenv("VOICEFLOW_API_KEY"):
        return {"ok": False, "error": "VOICEFLOW_API_KEY not set."}
    try:
        result = msg_tool.invoke({
            "user_id": req.user_id,
            "message": req.message,
            "channel": "voiceflow",
        })
        return {"ok": True, "result": str(result)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---- alerts ----

class AlertCreate(BaseModel):
    ticker: str
    condition: str = Field(..., description="above | below | change_pct")
    threshold: float
    user_id: str = "finnav_test_user"
    channel: str = "voiceflow"


@app.get("/api/alerts")
async def list_alerts() -> Dict[str, Any]:
    return {"alerts": state.alerts}


@app.post("/api/alerts")
async def create_alert(req: AlertCreate) -> Dict[str, Any]:
    if req.condition not in ("above", "below", "change_pct"):
        raise HTTPException(400, "condition must be one of: above, below, change_pct")
    alert = {
        "id": state.next_alert_id,
        "ticker": req.ticker.upper(),
        "condition": req.condition,
        "threshold": req.threshold,
        "user_id": req.user_id,
        "channel": req.channel,
        "active": True,
    }
    state.next_alert_id += 1
    state.alerts.append(alert)
    return {"ok": True, "alert": alert}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: int) -> Dict[str, Any]:
    before = len(state.alerts)
    state.alerts = [a for a in state.alerts if a["id"] != alert_id]
    return {"ok": True, "removed": before - len(state.alerts)}


# ---- glossary ----

@app.get("/api/glossary")
async def get_glossary() -> Dict[str, Any]:
    """SEC form-type definitions, mirrors the fine-tune training set."""
    from ui.glossary import SEC_FORMS
    return {"forms": SEC_FORMS}


# ---- file uploads ----

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Save a file to data/uploads/ and index its text into the RAG vectorstore."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "file > 50 MB")
    return uploads.save_and_index(file.filename or "upload.bin", data, state.vectorstore, state.embeddings)


@app.get("/api/uploads")
async def list_uploads() -> Dict[str, Any]:
    return {"files": uploads.list_uploads()}


@app.delete("/api/uploads/{filename}")
async def delete_upload(filename: str) -> Dict[str, Any]:
    return {"ok": uploads.remove(filename)}


# ---- schedules ----

class ScheduleCreate(BaseModel):
    name: str = Field(..., min_length=1)
    cron: str = Field(..., description="Standard 5-field cron expression (m h dom mon dow).")
    prompt: str = Field(..., min_length=1)
    active: bool = True


@app.get("/api/schedules")
async def list_schedules() -> Dict[str, Any]:
    return {"jobs": schedule_service.list()}


@app.post("/api/schedules")
async def create_schedule(req: ScheduleCreate) -> Dict[str, Any]:
    try:
        from apscheduler.triggers.cron import CronTrigger
        CronTrigger.from_crontab(req.cron)
    except Exception as e:
        raise HTTPException(400, f"invalid cron: {e}")
    job = schedule_service.add(req.name, req.cron, req.prompt, active=req.active)
    return {"ok": True, "job": job}


@app.delete("/api/schedules/{job_id}")
async def delete_schedule(job_id: str) -> Dict[str, Any]:
    return {"ok": schedule_service.remove(job_id)}


@app.post("/api/schedules/{job_id}/run")
async def run_schedule_now(job_id: str) -> Dict[str, Any]:
    try:
        return await schedule_service.run_now(job_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
