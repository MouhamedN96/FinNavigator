"""
Supervisor Agent - Multi-Agent Orchestration
==========================================

Supervisor agent that orchestrates multiple specialized agents for complex tasks.
Implements the supervisor/hierarchical agent pattern for autonomous workflow management.

Author: MiniMax Agent
"""

from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from .base_agent import BaseAgent, AgentConfig, AgentResponse, AgentCapability, AgentState, AgentMessage
from .financial_agent import FinancialAgent
from .research_agent import ResearchAgent
from .analyst_agent import AnalystAgent


class TaskType(Enum):
    """Task type classification"""
    QUERY = "query"  # Simple question
    RESEARCH = "research"  # SEC/research task
    ANALYSIS = "analysis"  # Portfolio/financial analysis
    COMPLEX = "complex"  # Multi-step task requiring multiple agents
    EXECUTION = "execution"  # Action-oriented task


@dataclass
class SubTask:
    """Represents a subtask for delegation"""
    task_id: str
    description: str
    assigned_agent: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "assigned_agent": self.assigned_agent,
            "status": self.status,
            "dependencies": self.dependencies
        }


@dataclass
class TaskPlan:
    """Execution plan for a complex task"""
    main_task: str
    task_type: TaskType
    subtasks: List[SubTask]
    execution_order: List[str]  # Task IDs in execution order
    estimated_complexity: str = "medium"

    def to_dict(self) -> Dict:
        return {
            "main_task": self.main_task,
            "task_type": self.task_type.value,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "execution_order": self.execution_order,
            "estimated_complexity": self.estimated_complexity
        }


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent for Multi-Agent Orchestration

    Acts as the central coordinator for complex tasks:
    - Task classification and decomposition
    - Agent selection and delegation
    - Result aggregation and synthesis
    - Error handling and retry

    Agent Pool:
    - FinancialAgent: General financial reasoning
    - ResearchAgent: SEC filings and research
    - AnalystAgent: Portfolio and risk analysis

    Example:
        supervisor = SupervisorAgent(config, llm_client)
        response = await supervisor.delegate_task(
            "Compare NVDA and AMD's risk profiles and recommend allocation"
        )
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: Optional[Any] = None
    ):
        super().__init__(config, llm_client, memory_manager)

        # Initialize agent pool
        self.agents: Dict[str, BaseAgent] = {}
        self.task_history: List[TaskPlan] = []

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a specialized agent"""
        self.agents[name] = agent
        self.logger.debug(f"Registered agent: {name}")

    def _classify_task(self, task: str) -> TaskType:
        """Classify the task type"""
        task_lower = task.lower()

        # Keywords for classification
        research_keywords = ["sec", "filing", "10-k", "10-q", "8-k", "annual report",
                           "quarterly", "search", "find", "lookup"]
        analysis_keywords = ["portfolio", "allocation", "risk", "var", "rebalance",
                            "performance", "sector", "diversification"]
        execution_keywords = ["alert", "send", "notify", "buy", "sell", "execute",
                             "index", "store", "save"]
        complex_keywords = ["compare", "versus", "vs", "and", "both", "multiple",
                           "comprehensive", "full analysis"]

        # Count keyword matches
        research_score = sum(1 for kw in research_keywords if kw in task_lower)
        analysis_score = sum(1 for kw in analysis_keywords if kw in task_lower)
        execution_score = sum(1 for kw in execution_keywords if kw in task_lower)
        complex_score = sum(1 for kw in complex_keywords if kw in task_lower)

        # Determine task type
        if complex_score >= 2:
            return TaskType.COMPLEX
        elif research_score > analysis_score and research_score > execution_score:
            return TaskType.RESEARCH
        elif analysis_score > research_score and analysis_score > execution_score:
            return TaskType.ANALYSIS
        elif execution_score > 0:
            return TaskType.EXECUTION
        else:
            return TaskType.QUERY

    async def _decompose_task(self, task: str, task_type: TaskType) -> TaskPlan:
        """Decompose complex task into subtasks"""
        self.add_reasoning_step(f"Decomposing task: {task[:100]}...")

        # Simple decomposition based on task type
        subtasks = []

        if task_type == TaskType.COMPLEX:
            # Complex task - break into research and analysis
            task_lower = task.lower()

            # Identify companies mentioned
            import re
            tickers = re.findall(r'\b[A-Z]{2,5}\b', task)

            if "risk" in task_lower:
                for ticker in tickers:
                    subtasks.append(SubTask(
                        task_id=f"research_{ticker}",
                        description=f"Research risk factors for {ticker}",
                        assigned_agent="research"
                    ))

            if "portfolio" in task_lower or "allocation" in task_lower:
                subtasks.append(SubTask(
                    task_id="portfolio_analysis",
                    description="Analyze portfolio and provide recommendations",
                    assigned_agent="analyst"
                ))

        elif task_type == TaskType.RESEARCH:
            import re
            tickers = re.findall(r'\b[A-Z]{2,5}\b', task)
            for ticker in tickers:
                subtasks.append(SubTask(
                    task_id=f"research_{ticker}",
                    description=f"Research {ticker} filings and background",
                    assigned_agent="research"
                ))

        elif task_type == TaskType.ANALYSIS:
            subtasks.append(SubTask(
                task_id="portfolio_analysis",
                description="Perform portfolio analysis",
                assigned_agent="analyst"
            ))

        else:
            # Simple query - just use financial agent
            subtasks.append(SubTask(
                task_id="general_query",
                description=task,
                assigned_agent="financial"
            ))

        # Determine execution order (handle dependencies)
        execution_order = [s.task_id for s in subtasks]

        complexity = "high" if len(subtasks) > 3 else "medium" if len(subtasks) > 1 else "low"

        return TaskPlan(
            main_task=task,
            task_type=task_type,
            subtasks=subtasks,
            execution_order=execution_order,
            estimated_complexity=complexity
        )

    async def _select_agent(self, task_type: TaskType, context: Optional[Dict] = None) -> str:
        """Select appropriate agent for task type"""
        agent_mapping = {
            TaskType.QUERY: "financial",
            TaskType.RESEARCH: "research",
            TaskType.ANALYSIS: "analyst",
            TaskType.COMPLEX: "financial",  # Will delegate to others
            TaskType.EXECUTION: "financial",
        }
        return agent_mapping.get(task_type, "financial")

    async def _delegate_to_agent(
        self,
        agent_name: str,
        task: str,
        context: Optional[Dict] = None
    ) -> AgentResponse:
        """Delegate task to a specialized agent"""
        if agent_name not in self.agents:
            return AgentResponse(
                success=False,
                content=f"Agent '{agent_name}' not found"
            )

        agent = self.agents[agent_name]
        self.add_reasoning_step(f"Delegating to {agent_name}: {task[:100]}...")

        try:
            result = await agent.process(task, context)
            self.add_reasoning_step(f"{agent_name} completed task")
            return result
        except Exception as e:
            self.logger.error(f"Agent {agent_name} failed: {e}")
            return AgentResponse(
                success=False,
                content=f"Agent error: {str(e)}",
                errors=[str(e)]
            )

    async def _aggregate_results(
        self,
        plan: TaskPlan,
        results: List[AgentResponse]
    ) -> str:
        """Aggregate results from multiple agents"""
        self.add_reasoning_step("Aggregating results from all agents")

        # Create aggregation prompt
        results_text = "\n\n".join([
            f"### {plan.subtasks[i].description}\n{results[i].content}"
            for i, r in enumerate(results) if r.success
        ])

        aggregation_prompt = f"""Based on the following research and analysis results, provide a comprehensive answer:

{results_text}

Original Task: {plan.main_task}

Synthesize these findings into a clear, actionable response.
"""
        messages = [SystemMessage(content=aggregation_prompt)]
        response = await self.llm.ainvoke(messages)
        return response.content

    def get_capabilities(self) -> List[AgentCapability]:
        """Return supervisor capabilities"""
        return [
            AgentCapability.REASONING,
            AgentCapability.EXECUTION,
            AgentCapability.COMMUNICATION,
        ]

    async def process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process task by classifying, decomposing, and delegating.

        Args:
            input_text: User task or question
            context: Optional context

        Returns:
            Aggregated response from all agents
        """
        self.update_state(AgentState.THINKING)
        start_time = datetime.now()

        # Step 1: Classify task
        task_type = self._classify_task(input_text)
        self.add_reasoning_step(f"Task classified as: {task_type.value}")

        # Step 2: Decompose if complex
        plan = await self._decompose_task(input_text, task_type)
        self.task_history.append(plan)

        self.add_reasoning_step(f"Task decomposed into {len(plan.subtasks)} subtasks")

        # Step 3: Execute subtasks
        results = []

        if len(plan.subtasks) == 1:
            # Single task - direct execution
            agent_name = await self._select_agent(task_type)
            result = await self._delegate_to_agent(agent_name, input_text, context)
            results.append(result)
        else:
            # Multiple tasks - execute in order
            for subtask in plan.subtasks:
                agent_name = subtask.assigned_agent or "financial"
                result = await self._delegate_to_agent(agent_name, subtask.description, context)
                subtask.result = result
                subtask.status = "completed" if result.success else "failed"
                results.append(result)

        # Step 4: Aggregate results
        if len(results) > 1:
            final_content = await self._aggregate_results(plan, results)
        else:
            final_content = results[0].content if results else "No results"

        execution_time = (datetime.now() - start_time).total_seconds()

        # Collect all tools used
        all_tools = []
        for r in results:
            all_tools.extend(r.tools_used)

        return AgentResponse(
            success=any(r.success for r in results),
            content=final_content,
            reasoning_steps=self.reasoning_history,
            tools_used=list(set(all_tools)),
            metadata={
                "task_type": task_type.value,
                "plan": plan.to_dict(),
                "subtask_results": [r.to_dict() for r in results]
            },
            execution_time=execution_time
        )

    async def delegate_task(
        self,
        task: str,
        agents: Optional[List[str]] = None
    ) -> AgentResponse:
        """
        Explicitly delegate a task to specific agents or auto-select.

        Args:
            task: Task description
            agents: Optional list of specific agents to use

        Returns:
            Response from agent(s)
        """
        if agents:
            # Execute with specific agents
            results = []
            for agent_name in agents:
                result = await self._delegate_to_agent(agent_name, task)
                results.append(result)

            if len(results) == 1:
                return results[0]

            return AgentResponse(
                success=any(r.success for r in results),
                content="\n\n".join([r.content for r in results]),
                tools_used=[t for r in results for t in r.tools_used]
            )
        else:
            return await self.process(task)

    async def stream_process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Stream the orchestration process"""
        # Classify
        yield "## Task Classification\n\n"
        task_type = self._classify_task(input_text)
        yield f"**Type:** {task_type.value}\n\n"

        # Decompose
        yield "## Task Planning\n\n"
        plan = await self._decompose_task(input_text, task_type)
        yield f"**Complexity:** {plan.estimated_complexity}\n\n"
        yield "**Subtasks:**\n"
        for i, subtask in enumerate(plan.subtasks, 1):
            yield f"{i}. {subtask.description} (→ {subtask.assigned_agent})\n"
        yield "\n"

        # Execute
        yield "## Execution\n\n"
        if len(plan.subtasks) > 1:
            for i, subtask in enumerate(plan.subtasks):
                yield f"### Step {i+1}: {subtask.assigned_agent}\n\n"
                agent_name = subtask.assigned_agent or "financial"
                result = await self._delegate_to_agent(agent_name, subtask.description, context)
                yield f"{result.content[:1000]}...\n\n"
        else:
            result = await self.process(input_text, context)
            yield f"{result.content}\n\n"

        yield "\n## Summary\n\n"
        yield "Task orchestration complete."

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        status = {
            "supervisor": self.get_status(),
            "registered_agents": len(self.agents),
            "agents": {}
        }

        for name, agent in self.agents.items():
            status["agents"][name] = agent.get_status()

        return status


class AgentTeam:
    """
    Agent Team - Complete multi-agent system setup

    Factory and manager for creating and coordinating agent teams.
    """

    def __init__(self, llm_client: Any, config: Optional[Dict] = None):
        self.llm = llm_client
        self.config = config or {}
        self.supervisor: Optional[SupervisorAgent] = None
        self.agents: Dict[str, BaseAgent] = {}

    def setup_team(
        self,
        financial_tools: List[Any],
        research_tools: List[Any],
        analyst_tools: List[Any]
    ) -> SupervisorAgent:
        """Setup complete agent team"""

        # Create agent configurations
        financial_config = AgentConfig(
            name="financial",
            role="Financial Advisor",
            description="General financial reasoning and advice",
            system_prompt="You are an expert financial advisor helping with investment decisions."
        )

        research_config = AgentConfig(
            name="research",
            role="Research Analyst",
            description="SEC filings and company research",
            system_prompt="You are a research analyst specializing in SEC filings and company documents."
        )

        analyst_config = AgentConfig(
            name="analyst",
            role="Portfolio Analyst",
            description="Portfolio and risk analysis",
            system_prompt="You are a portfolio analyst specializing in risk metrics and allocation."
        )

        # Create agents
        self.agents["financial"] = FinancialAgent(
            config=financial_config,
            llm_client=self.llm
        )
        self.agents["financial"].register_tools(financial_tools)

        self.agents["research"] = ResearchAgent(
            config=research_config,
            llm_client=self.llm
        )

        self.agents["analyst"] = AnalystAgent(
            config=analyst_config,
            llm_client=self.llm
        )

        # Create supervisor
        supervisor_config = AgentConfig(
            name="supervisor",
            role="Team Supervisor",
            description="Orchestrates multiple agents for complex tasks",
            system_prompt="You are the supervisor of a financial analysis team."
        )

        self.supervisor = SupervisorAgent(
            config=supervisor_config,
            llm_client=self.llm
        )

        # Register all agents with supervisor
        for name, agent in self.agents.items():
            self.supervisor.register_agent(name, agent)

        return self.supervisor

    def get_team_status(self) -> Dict[str, Any]:
        """Get status of entire team"""
        if not self.supervisor:
            return {"error": "Team not set up"}

        return self.supervisor.get_agent_status()
