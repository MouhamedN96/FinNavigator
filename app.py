"""
FinNavigator AI - Main Application with LangChain Deep Agents
============================================================

Streamlit application integrating:
- Multi-agent system (Financial, Research, Analyst, Supervisor)
- ReAct reasoning agents
- SEC filing research
- Portfolio analysis
- RAG-based knowledge retrieval
- Voiceflow messaging integration

Author: MiniMax Agent
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio
import os
import base64
from io import BytesIO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import agent modules
from agents import (
    FinancialAgent,
    ResearchAgent,
    AnalystAgent,
    SupervisorAgent,
    AgentConfig,
    BaseAgent
)
from agents.supervisor_agent import AgentTeam, TaskType
from tools import (
    CalculatorTool,
    DateTimeTool,
    SECSearchTool,
    SECExtractTool,
    KnowledgeBaseSearchTool,
    KnowledgeBaseIndexTool,
    VisualContextIndexTool,
    SendMessageTool,
    AlertTool,
    get_sec_edgar_tools,
)
from memory import create_memory_manager, MemoryManager

# Page config
st.set_page_config(
    page_title="FinNavigator AI - Deep Agents",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Configuration and State Management
# ============================================================================

def get_llm_client():
    """Get configured LLM client.

    Default: local Ollama (Qwen3-4B) — no API key needed.
    Override by setting LLM_BACKEND=openai (and OPENAI_API_KEY) or LLM_BACKEND=anthropic.
    """
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    try:
        if backend == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
                base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                temperature=temperature,
            )
        if backend == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("LLM_BACKEND=openai but OPENAI_API_KEY not set.")
                return None
            return ChatOpenAI(
                model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                api_key=api_key,
                temperature=temperature,
            )
        if backend == "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("LLM_BACKEND=anthropic but ANTHROPIC_API_KEY not set.")
                return None
            return ChatAnthropic(
                model=os.getenv("MODEL_NAME", "claude-3-5-haiku-latest"),
                api_key=api_key,
                temperature=temperature,
            )
        st.error(f"Unknown LLM_BACKEND: {backend}")
        return None
    except ImportError as e:
        st.error(f"LLM package not installed for backend={backend}: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        return None


def get_vectorstore():
    """Initialize Chroma vectorstore with HuggingFace embeddings."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        
        # Ensure data directory exists
        os.makedirs("data/chroma_db", exist_ok=True)
        
        # Use a lightweight, standard embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize the Chroma vectorstore
        vectorstore = Chroma(
            persist_directory="data/chroma_db",
            embedding_function=embeddings,
            collection_name="financial_docs"
        )
        return vectorstore, embeddings
    except Exception as e:
        st.warning(f"Could not initialize vectorstore: {e}")
        return None, None


def initialize_agents():
    """Initialize all agents"""
    if "agents_initialized" in st.session_state:
        return

    with st.spinner("Initializing agents..."):
        # Get LLM client
        llm_client = get_llm_client()

        if not llm_client:
            st.error("LLM client not available")
            return

        # Initialize memory manager
        memory_manager = create_memory_manager()

        # Initialize vectorstore for RAG
        vectorstore, embeddings = get_vectorstore()

        # Create tools
        calculator = CalculatorTool()
        datetime_tool = DateTimeTool()
        knowledge_search = KnowledgeBaseSearchTool(vectorstore=vectorstore, embeddings=embeddings)
        knowledge_index = KnowledgeBaseIndexTool(vectorstore=vectorstore)
        visual_index = VisualContextIndexTool(vectorstore=vectorstore)
        message_tool = SendMessageTool()
        alert_tool = AlertTool(message_tool)

        # SEC tools: prefer sec-edgar-agentkit if installed, fall back to bundled tools.
        sec_tools = get_sec_edgar_tools()

        financial_tools = [calculator, datetime_tool, knowledge_search, knowledge_index, visual_index, message_tool] + sec_tools
        analyst_tools = [calculator, datetime_tool, knowledge_search]
        research_tools = sec_tools + [knowledge_search, knowledge_index, visual_index]

        # Detect local vision model
        vision_path = os.path.join(os.getcwd(), "finnav_qwen3-VL_4b_gguf")
        use_local_vision = st.session_state.get("use_local_vision", False)
        active_vision_path = vision_path if use_local_vision else None

        # Create agent team
        team = AgentTeam(llm_client)
        supervisor = team.setup_team(
            financial_tools=financial_tools,
            research_tools=research_tools,
            analyst_tools=analyst_tools,
            vision_model_path=active_vision_path
        )

        # Store in session state
        st.session_state.llm_client = llm_client
        st.session_state.memory_manager = memory_manager
        st.session_state.team = team
        st.session_state.supervisor = supervisor
        st.session_state.agents_initialized = True

        st.success("Agents initialized successfully!")


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render sidebar with navigation and controls"""
    with st.sidebar:
        st.title("🛡️ FinNavigator")

        # Status indicator
        if st.session_state.get("agents_initialized"):
            st.success("● Agents Online")
        else:
            st.warning("○ Initializing...")

        st.divider()

        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select Mode",
            ["💬 AI Agent Chat", "🔬 Research", "📊 Portfolio Analysis", "🤖 Agent Monitor"],
            label_visibility="collapsed"
        )

        st.divider()

        # Agent settings
        st.subheader("⚙️ Settings")

        with st.expander("Agent Configuration"):
            max_iterations = st.slider("Max Reasoning Steps", 1, 20, 10)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
            show_reasoning = st.checkbox("Show Reasoning Trace", value=True)
            
            # Local Vision Toggle
            vision_path = os.path.join(os.getcwd(), "finnav_qwen3-VL_4b_gguf")
            local_vision_exists = os.path.exists(vision_path)
            use_local_vision = st.checkbox(
                "Use Local Vision SLM (Qwen3-VL)", 
                value=False, 
                disabled=not local_vision_exists,
                help="Requires ~10GB RAM and proper dependencies." if local_vision_exists else "Local model folder not found."
            )

            if use_local_vision != st.session_state.get("use_local_vision", False):
                st.session_state.use_local_vision = use_local_vision
                # Clear agents to re-initialize with new vision config
                if "agents_initialized" in st.session_state:
                    del st.session_state.agents_initialized
                    st.rerun()

            st.session_state.max_iterations = max_iterations
            st.session_state.temperature = temperature
            st.session_state.show_reasoning = show_reasoning

        st.divider()

        # Quick actions
        st.subheader("🚀 Quick Actions")

        if st.button("🔍 Index SEC Filings", use_container_width=True):
            st.session_state.quick_action = "index_filings"

        if st.button("📈 Portfolio Analysis", use_container_width=True):
            st.session_state.quick_action = "portfolio_analysis"

        if st.button("💾 Clear Memory", use_container_width=True):
            if st.session_state.get("memory_manager"):
                st.session_state.memory_manager.clear_session()
                st.success("Memory cleared!")

        st.divider()

        # Memory summary
        if st.session_state.get("memory_manager"):
            st.subheader("📝 Memory")
            mem_summary = st.session_state.memory_manager.get_memory_summary()
            st.caption(f"Turns: {mem_summary['conversation_turns']}")
            st.caption(f"Memories: {mem_summary['total_memories']}")
            st.caption(f"Facts: {mem_summary['facts']}")

        return page


def render_agent_chat():
    """Render AI Agent chat interface"""
    st.header("💬 Deep Agent Chat")

    # Initialize chat history
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # Display chat history
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.image(base64.b64decode(msg["image"]), width=300)
            if "reasoning" in msg and st.session_state.get("show_reasoning"):
                with st.expander("🔍 Reasoning Trace"):
                    st.code(msg["reasoning"], language=None)

    # Chat input area with file upload
    with st.container():
        # Optional image upload
        uploaded_file = st.file_uploader("🖼️ Attach image (charts, filings, tables...)", type=["jpg", "jpeg", "png"])
        
        # Chat input
        if prompt := st.chat_input("Ask about finances, SEC filings, or portfolio..."):
            # Prepare message data
            image_b64 = None
            if uploaded_file:
                bytes_data = uploaded_file.getvalue()
                image_b64 = base64.b64encode(bytes_data).decode("utf-8")
                
            # Add user message to history
            user_msg = {"role": "user", "content": prompt}
            if image_b64:
                user_msg["image"] = image_b64
            
            st.session_state.agent_messages.append(user_msg)

            with st.chat_message("user"):
                st.markdown(prompt)
                if image_b64:
                    st.image(uploaded_file, caption="Uploaded context", width=300)

            # Process with agent
            with st.chat_message("assistant"):
                with st.spinner("Agent reasoning..."):
                    if st.session_state.get("supervisor"):
                        # Run async agent
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        if image_b64:
                            # Use process_vision for multimodal tasks
                            response = loop.run_until_complete(
                                st.session_state.supervisor.process_vision(prompt, image_b64)
                            )
                        else:
                            # Standard text processing
                            response = loop.run_until_complete(
                                st.session_state.supervisor.process(prompt)
                            )
                        loop.close()

                        # Display response
                        st.markdown(response.content)

                        # Show reasoning if enabled
                        if st.session_state.get("show_reasoning") and response.reasoning_steps:
                            with st.expander("🔍 Reasoning Trace"):
                                for step in response.reasoning_steps[-5:]:
                                    if isinstance(step, dict):
                                        thought = step.get("thought", "")
                                        action = step.get("action", "")
                                        observation = step.get("observation", "")
                                        if thought:
                                            st.markdown(f"**Thought:** {thought}")
                                        if action:
                                            st.markdown(f"**Action:** {action}")
                                        if observation:
                                            st.markdown(f"**Observation:** {observation[:200]}...")

                        # Store in history
                        reasoning_text = "\n".join([
                            str(step) for step in response.reasoning_steps[-5:]
                        ]) if response.reasoning_steps else ""

                        st.session_state.agent_messages.append({
                            "role": "assistant",
                            "content": response.content,
                            "reasoning": reasoning_text
                        })

                        # Update memory
                        if st.session_state.get("memory_manager"):
                            st.session_state.memory_manager.add_conversation("user", prompt)
                            st.session_state.memory_manager.add_conversation("assistant", response.content)
                    else:
                        st.error("Agent not initialized")


def render_research_tab():
    """Render SEC research interface"""
    st.header("🔬 SEC Research Agent")

    col1, col2 = st.columns([1, 2])

    with col1:
        ticker = st.text_input("Stock Ticker", placeholder="NVDA, AAPL, TSLA...").upper()

        focus_options = ["Overview", "Risks", "Financials", "Comparison"]
        focus = st.selectbox("Research Focus", focus_options)

        depth_options = ["Quick", "Standard", "Deep"]
        depth = st.selectbox("Analysis Depth", depth_options)

        if st.button("🔍 Research", use_container_width=True):
            if ticker:
                st.session_state.research_request = {
                    "ticker": ticker,
                    "focus": focus.lower(),
                    "depth": depth.lower()
                }

    with col2:
        if "research_request" in st.session_state:
            req = st.session_state.research_request

            with st.spinner(f"Researching {req['ticker']}..."):
                # Create research agent
                research_config = AgentConfig(
                    name="research",
                    role="Research Analyst",
                    description="SEC filings research",
                    system_prompt="You are a research analyst specializing in SEC filings."
                )

                research_agent = ResearchAgent(
                    config=research_config,
                    llm_client=st.session_state.llm_client,
                    sec_api_key=os.getenv("SEC_API_KEY", "")
                )

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                research = loop.run_until_complete(
                    research_agent.research_company(req["ticker"], req["focus"], req["depth"])
                )
                loop.close()

                # Display results
                if "summary" in research:
                    st.markdown("### Research Summary")
                    st.markdown(research["summary"])

                if "findings" in research:
                    findings = research["findings"]

                    if "filings" in findings:
                        st.markdown("### Recent Filings")
                        filings = findings["filings"].get("filings", [])
                        if filings:
                            df = pd.DataFrame([
                                {
                                    "Type": f.get("formType", ""),
                                    "Filed": f.get("filedAt", "")[:10],
                                    "Description": f.get("description", "")[:50]
                                }
                                for f in filings
                            ])
                            st.dataframe(df, use_container_width=True)
        else:
            st.info("Enter a ticker symbol and click Research to begin")


def render_portfolio_analysis():
    """Render portfolio analysis interface"""
    st.header("📊 Portfolio Analysis Agent")

    # Portfolio input
    with st.expander("📝 Portfolio Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            portfolio_value = st.number_input("Total Portfolio Value ($)", min_value=0, value=100000)

        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Overview", "Risk Metrics", "Sector Exposure", "Rebalancing"]
            )

        st.subheader("Positions")

        # Sample positions (editable)
        if "portfolio_positions" not in st.session_state:
            st.session_state.portfolio_positions = [
                {"ticker": "NVDA", "shares": 50, "avg_cost": 450.0, "current_price": 905.0},
                {"ticker": "AAPL", "shares": 100, "avg_cost": 150.0, "current_price": 178.0},
                {"ticker": "MSFT", "shares": 30, "avg_cost": 300.0, "current_price": 415.0},
                {"ticker": "TSLA", "shares": 25, "avg_cost": 200.0, "current_price": 175.0},
            ]

        # Display positions table
        positions_df = pd.DataFrame(st.session_state.portfolio_positions)
        positions_df["market_value"] = positions_df["shares"] * positions_df["current_price"]
        positions_df["gain_loss"] = positions_df["market_value"] - (positions_df["shares"] * positions_df["avg_cost"])
        positions_df["allocation"] = (positions_df["market_value"] / portfolio_value * 100).round(2)

        edited_df = st.data_editor(
            positions_df,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "shares": st.column_config.NumberColumn("Shares", min_value=0),
                "avg_cost": st.column_config.NumberColumn("Avg Cost", format="%.2f"),
                "current_price": st.column_config.NumberColumn("Current Price", format="%.2f"),
                "market_value": st.column_config.NumberColumn("Market Value", format="%.2f"),
                "gain_loss": st.column_config.NumberColumn("Gain/Loss", format="%.2f"),
                "allocation": st.column_config.NumberColumn("Allocation %", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

        if st.button("📊 Analyze Portfolio", use_container_width=True):
            st.session_state.analysis_request = True

    # Analysis results
    if st.session_state.get("analysis_request"):
        # Create analyst agent
        analyst_config = AgentConfig(
            name="analyst",
            role="Portfolio Analyst",
            description="Portfolio and risk analysis",
            system_prompt="You are a portfolio analyst specializing in risk metrics and allocation."
        )

        analyst_agent = AnalystAgent(
            config=analyst_config,
            llm_client=st.session_state.llm_client
        )

        # Set portfolio
        positions = edited_df.to_dict("records")
        analyst_agent.set_portfolio(positions)

        col1, col2, col3 = st.columns(3)

        with col1:
            total_value = edited_df["market_value"].sum()
            total_gain = edited_df["gain_loss"].sum()
            st.metric("Total Value", f"${total_value:,.2f}")
            st.metric("Total Gain/Loss", f"${total_gain:,.2f}",
                     delta=f"{(total_gain / (total_value - total_gain) * 100):.1f}%")

        with col2:
            # Sector exposure chart
            st.markdown("### Sector Exposure")
            # Simplified sector mapping
            sector_data = {
                "NVDA": "Technology",
                "AAPL": "Technology",
                "MSFT": "Technology",
                "TSLA": "Consumer"
            }
            edited_df["sector"] = edited_df["ticker"].map(sector_data).fillna("Other")
            sector_alloc = edited_df.groupby("sector")["market_value"].sum()
            fig = px.pie(sector_alloc, values=sector_alloc.values, names=sector_alloc.index)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("### Allocation")
            fig = px.bar(
                edited_df,
                x="ticker",
                y="allocation",
                color="gain_loss",
                color_continuous_scale=["red", "green"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed analysis
        with st.spinner("Running risk analysis..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if analysis_type == "Risk Metrics":
                var_result = loop.run_until_complete(analyst_agent.calculate_var())
                st.markdown("### Risk Metrics")
                st.metric("VaR (95%)", f"${var_result['var_absolute']:,.2f}")
                st.caption(f"Daily Value at Risk at 95% confidence")

            elif analysis_type == "Sector Exposure":
                sector_result = loop.run_until_complete(analyst_agent.analyze_sector_exposure())
                st.markdown("### Sector Analysis")
                st.metric("Diversification Score", f"{sector_result['diversification_score']}/100")
                for sector, data in sector_result["sector_exposure"].items():
                    st.markdown(f"- **{sector}**: {data['allocation']:.1f}%")

            loop.close()


def render_agent_monitor():
    """Render agent monitoring interface"""
    st.header("🤖 Agent System Monitor")

    if not st.session_state.get("team"):
        st.warning("Agents not initialized")
        return

    team = st.session_state.team
    status = team.get_team_status()

    # System overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Agents", status.get("registered_agents", 0))

    with col2:
        if "memory_manager" in st.session_state:
            mem_summary = st.session_state.memory_manager.get_memory_summary()
            st.metric("Memories", mem_summary.get("total_memories", 0))

    with col3:
        st.metric("Conversations", mem_summary.get("conversation_turns", 0) if "memory_manager" in st.session_state else 0)

    with col4:
        st.metric("Facts Stored", mem_summary.get("facts", 0) if "memory_manager" in st.session_state else 0)

    st.divider()

    # Agent status
    st.subheader("Agent Status")

    agents_status = status.get("agents", {})

    for agent_name, agent_data in agents_status.items():
        with st.expander(f"🤖 {agent_name.title()} Agent"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Role:** {agent_data.get('role', 'N/A')}")
                st.write(f"**State:** {agent_data.get('state', 'N/A')}")
            with col2:
                st.write(f"**Messages:** {agent_data.get('message_count', 0)}")
                st.write(f"**Reasoning Steps:** {agent_data.get('reasoning_steps', 0)}")

            if agent_data.get("tools_used"):
                st.write("**Recent Tools:**")
                for tool in agent_data["tools_used"][-5:]:
                    st.caption(f"- {tool}")

    st.divider()

    # Task history
    st.subheader("Recent Tasks")

    if st.session_state.get("supervisor"):
        supervisor = st.session_state.supervisor
        if supervisor.task_history:
            for task in supervisor.task_history[-5:]:
                st.markdown(f"**{task.main_task[:80]}...**")
                st.caption(f"Type: {task.task_type.value} | Complexity: {task.estimated_complexity}")
                st.divider()
        else:
            st.info("No task history yet")


def render_main_content(page: str):
    """Render main content based on selected page"""
    if page == "💬 AI Agent Chat":
        render_agent_chat()
    elif page == "🔬 Research":
        render_research_tab()
    elif page == "📊 Portfolio Analysis":
        render_portfolio_analysis()
    elif page == "🤖 Agent Monitor":
        render_agent_monitor()


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""

    # Initialize agents on first load
    initialize_agents()

    # Render UI
    page = render_sidebar()
    render_main_content(page)


if __name__ == "__main__":
    main()
