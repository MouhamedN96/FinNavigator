# FinNavigator AI -  Deep Agents to Navigate and Figure out your Finance

A  financial intelligence system powered by LangChain Deep Agents, featuring autonomous reasoning, multi-agent collaboration, and comprehensive SEC filing analysis.

## Features

###  Deep Agent System

- **Financial Agent**: ReAct-based reasoning agent with chain-of-thought prompting
- **Research Agent**: Specialized in SEC filings and company research
- **Analyst Agent**: Portfolio and risk analysis with real-time metrics
- **Supervisor Agent**: Orchestrates multiple agents for complex tasks

###  SEC Filing Research

- Search SEC EDGAR database for regulatory filings (10-K, 10-Q, 8-K)
- Extract specific sections (Risk Factors, MD&A, Financials)
- Historical filing comparison
- Risk factor analysis and categorization

###  Portfolio Analysis

- Real-time portfolio performance tracking
- Risk metrics (VaR, Sharpe Ratio, Beta, Diversification)
- Sector exposure analysis
- Rebalancing recommendations

### 👁️ Multimodal Support (NEW)

- **Local Vision SLM**: Integrated Qwen3-VL-4B for local analysis of financial charts and documents.
- **Visual Context Indexing**: Store and retrieve descriptions of visual data alongside textual filings.
- **Hybrid RAG**: Combines semantic similarity (Chroma) with keyword matching (BM25) and Flashrank reranking.

### 💬 Agent Chat Interface

- Natural language interaction with financial agents
- Visible reasoning trace showing agent thought process
- Context-aware responses using RAG
- Memory of previous conversations

### 🔗 Integrations

- NVIDIA NIM (Llama 3.70B) for LLM
- Qwen3-VL (Local SLM) for Multimodal Vision
- Colab, Huggingface, Ollama for LLM Training and Inference
- SEC API for regulatory filings
- ChromaDB for vector storage
- Voiceflow for messaging alerts
- Flashrank for high-precision reranking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Supervisor Agent                          │
│         (Task Classification & Orchestration)                │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┐
    ▼             ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Financial│  │Research  │  │Analyst   │  │Messaging │
│ Agent   │  │ Agent    │  │ Agent    │  │ Agent    │
└───┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
    │            │             │             │
    └────────────┴──────┬──────┴─────────────┘
                       ▼
              ┌─────────────────┐
              │  Knowledge Base │
              │   (ChromaDB)    │
              └─────────────────┘
```

## Installation

```bash
# Install server + UI deps
pip install -r requirements.txt -r requirements-ui.txt

# Set environment variables (or copy .env.example → .env and fill in)
export NVIDIA_API_KEY="your_nvidia_api_key"
export SEC_API_KEY="your_sec_api_key"
export VOICEFLOW_API_KEY="your_voiceflow_api_key"

# Run the FastAPI backend
uvicorn api.server:app --reload --port 8000

# In another terminal, run the Flet UI (desktop dev)
flet run ui/app.py

# Or build the static web bundle for Cloudflare Pages
flet build web
```

> The original Streamlit interface is still in `legacy_streamlit_app.py` and works via
> `streamlit run legacy_streamlit_app.py` — kept one milestone for parity testing,
> will be removed in the next.

## Usage

### Agent Chat

Ask complex financial questions in natural language:
- "What's the latest risk factors for NVDA?"
- "Compare AMD and Intel's financial performance"
- "Should I rebalance my portfolio?"

### Research Mode

1. Enter a stock ticker
2. Select research focus (Overview, Risks, Financials)
3. Choose analysis depth (Quick, Standard, Deep)
4. View comprehensive research results

### Portfolio Analysis

1. Enter portfolio value
2. Add/edit positions
3. Run analysis (Overview, Risk, Sector, Rebalancing)
4. View visualizations and recommendations

## Agent Tools

### Base Tools
- `CalculatorTool`: Mathematical operations
- `DateTimeTool`: Date calculations
- `WikipediaSearchTool`: General knowledge

### Financial Tools
- `SECSearchTool`: EDGAR filing search
- `SECExtractTool`: Section extraction
- `PortfolioCalculatorTool`: Allocation calculations
- `RiskCalculatorTool`: Risk metrics
- `NewsSearchTool`: Financial news
- `StockDataTool`: Market data

### Knowledge Tools
- `KnowledgeBaseSearchTool`: Semantic search
- `KnowledgeBaseIndexTool`: Document indexing
- `VectorQueryTool`: Similarity search
- `MemorySearchTool`: Memory retrieval

### Messaging Tools
- `SendMessageTool`: Voiceflow messaging
- `AlertTool`: Financial alerts
- `PortfolioAlertTool`: Price alerts

## Memory System

The system includes a comprehensive memory management:

- **ConversationMemory**: Tracks dialogue history
- **PersistentMemory**: Stores key decisions and facts
- **VectorMemory**: Enables semantic memory search
- **MemoryManager**: Coordinates all memory types

## ReAct Reasoning

The Financial Agent uses the ReAct (Reason + Act) pattern:

1. **Think**: Analyze the question
2. **Act**: Select and execute appropriate tool
3. **Observe**: Review tool result
4. **Repeat** until answer is complete

Example reasoning trace:
```
Thought: The user wants to know NVDA's risk factors. I should search SEC filings first.
Action: sec_search
Action Input: {"ticker": "NVDA", "form_type": "10-Q"}
Observation: Found 3 recent filings

Thought: Now I need to extract the risk factors section.
Action: sec_extract
Action Input: {"ticker": "NVDA", "section": "item1a"}
Observation: Extracted 4500 characters of risk disclosures

Thought: I have enough information to provide a comprehensive answer.
```

## Project Structure

```
finavigator/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py       # Base agent class
│   ├── financial_agent.py  # ReAct agent implementation
│   ├── research_agent.py   # SEC research specialist
│   ├── analyst_agent.py    # Portfolio analysis
│   └── supervisor_agent.py # Multi-agent orchestration
├── tools/
│   ├── __init__.py
│   ├── base_tools.py       # Calculator, datetime
│   ├── financial_tools.py  # SEC, portfolio, risk
│   ├── knowledge_tools.py # Vector DB operations
│   └── messaging_tools.py  # Voiceflow integration
├── memory/
│   └── __init__.py         # Memory management
├── config/
│   └── __init__.py
├── inference/
│   ├── __init__.py
│   └── backend.py          # LLM backend adapter (ollama / nvidia / openai / anthropic / webgpu)
├── api/
│   ├── __init__.py
│   └── server.py           # FastAPI server (Fly.io / Railway / HF Spaces)
├── ui/
│   ├── app.py              # Flet entry — web (Cloudflare Pages) / desktop / PWA
│   ├── api_client.py
│   └── pages/              # chat, research, portfolio, monitor
├── desktop/
│   ├── launcher.py         # Bundles Ollama + uvicorn + Flet for native binaries
│   └── ollama_bootstrap.py
├── docs/
│   └── desktop.md
├── legacy_streamlit_app.py # Old Streamlit UI — kept one milestone for parity
├── Dockerfile              # API server image (FastAPI + uvicorn)
├── fly.toml
├── requirements.txt
├── requirements-ui.txt
├── .env.example
└── README.md
```

## Configuration

### Environment Variables

```env
# LLM Configuration
NVIDIA_API_KEY=your_nvidia_api_key
MODEL_NAME=meta/llama3-70b-instruct

# SEC API
SEC_API_KEY=your_sec_api_key

# Messaging
VOICEFLOW_API_KEY=your_voiceflow_api_key
VOICEFLOW_VERSION=development

# Vector Store
PERSIST_DIRECTORY=./finance_db
```

### Agent Parameters

- `max_iterations`: Maximum reasoning steps (default: 10)
- `temperature`: LLM temperature (default: 0.5)
- `max_tokens`: Maximum response tokens (default: 2048)

## Monitoring

Access the Agent Monitor tab to view:

- Agent status and health
- Task history
- Memory usage
- Tool usage statistics

## Deployment

### 🚆 Deploying to Railway

1.  Push your code to GitHub.
2.  Connect your repository to [Railway](https://railway.app/).
3.  Add the following Environment Variables in the Railway dashboard:
    *   `NVIDIA_API_KEY` (Required for primary LLM)
    *   `SEC_API_KEY` (Required for research)
    *   `VOICEFLOW_API_KEY` (Optional for alerts)
4.  Railway will automatically detect the `Dockerfile` and deploy the app.

### 🤗 Deploying to Hugging Face Spaces

1.  Create a new Space on [Hugging Face](https://huggingface.co/spaces) using the **Streamlit** SDK.
2.  Sync your GitHub repository or upload the files.
3.  Add your secrets (`NVIDIA_API_KEY`, etc.) in the Space settings.
4.  *Note: To use the Local Vision SLM, you will need a Space with at least 16GB RAM or a GPU tier.*

## License

MIT License

## Author
MouhamedN96
