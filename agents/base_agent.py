"""
Base Agent Class for FinNavigator Deep Agents
==============================================

Provides the foundation for all specialized agents with common functionality
including tool execution, memory management, and state tracking.

Author: MiniMax Agent
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Enumeration of agent capabilities"""
    REASONING = "reasoning"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    CALCULATION = "calculation"


class AgentState(Enum):
    """Enumeration of agent states"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    RESPONDING = "responding"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for agent initialization"""
    name: str
    role: str
    description: str
    system_prompt: str
    model_name: str = "meta/llama3-70b-instruct"
    temperature: float = 0.5
    max_tokens: int = 2048
    tools: List[BaseTool] = field(default_factory=list)
    verbose: bool = True
    max_iterations: int = 10


@dataclass
class AgentMessage:
    """Standardized message format for agent communication"""
    sender: str
    recipient: Optional[str]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    thread_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "metadata": self.metadata,
            "attachments": self.attachments,
            "thread_id": self.thread_id,
        }


@dataclass
class AgentResponse:
    """Response from agent execution"""
    success: bool
    content: str
    reasoning_steps: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        return {
            "success": self.success,
            "content": self.content,
            "reasoning_steps": self.reasoning_steps,
            "tools_used": self.tools_used,
            "intermediate_steps": self.intermediate_steps,
            "errors": self.errors,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all FinNavigator agents.

    Provides common functionality for:
    - Tool execution and management
    - Message handling and state management
    - Reasoning step tracking
    - Error handling and recovery
    - Logging and observability

    Attributes:
        config: Agent configuration
        state: Current agent state
        message_history: List of all messages
        reasoning_history: List of reasoning steps
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: Optional[Any] = None
    ):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration
            llm_client: LLM client for reasoning
            memory_manager: Optional memory manager for conversation history
        """
        self.config = config
        self.llm = llm_client
        self.memory = memory_manager
        self.state = AgentState.IDLE
        self.message_history: List[AgentMessage] = []
        self.reasoning_history: List[str] = []
        self.tool_results: Dict[str, Any] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the agent"""
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
        if self.config.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    @property
    def name(self) -> str:
        """Get agent name"""
        return self.config.name

    @property
    def role(self) -> str:
        """Get agent role"""
        return self.config.role

    def update_state(self, new_state: AgentState) -> None:
        """Update agent state"""
        old_state = self.state
        self.state = new_state
        self.logger.debug(f"State transition: {old_state.value} -> {new_state.value}")

    def add_reasoning_step(self, step: str) -> None:
        """Add a reasoning step to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_step = f"[{timestamp}] {step}"
        self.reasoning_history.append(formatted_step)
        self.logger.debug(f"Reasoning: {step}")

    def add_message(self, message: AgentMessage) -> None:
        """Add message to history"""
        self.message_history.append(message)
        self.logger.debug(f"Message from {message.sender} to {message.recipient}: {message.content[:100]}...")

    async def execute_tool(
        self,
        tool: BaseTool,
        tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool and track the result.

        Args:
            tool: The tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result
        """
        self.update_state(AgentState.EXECUTING_TOOL)
        self.add_reasoning_step(f"Executing tool: {tool.name} with input: {tool_input}")

        try:
            result = await tool.ainvoke(tool_input)
            self.tool_results[tool.name] = result
            self.add_reasoning_step(f"Tool {tool.name} completed successfully")
            return {"success": True, "result": result, "tool": tool.name}
        except Exception as e:
            error_msg = f"Tool {tool.name} failed: {str(e)}"
            self.add_reasoning_step(error_msg)
            self.logger.error(error_msg)
            return {"success": False, "error": str(e), "tool": tool.name}

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        capabilities = ", ".join([c.value for c in self.get_capabilities()])
        return f"""{self.config.system_prompt}

You are a {self.config.role} agent named {self.config.name}.
Your capabilities include: {capabilities}.

Current date: {datetime.now().strftime('%Y-%m-%d')}
Always provide accurate, up-to-date financial information.
"""

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities this agent has"""
        pass

    @abstractmethod
    async def process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process input and generate response.

        Args:
            input_text: User input
            context: Optional context for the request

        Returns:
            Agent response with reasoning and results
        """
        pass

    def get_reasoning_trace(self) -> str:
        """Get formatted reasoning trace"""
        return "\n".join(self.reasoning_history)

    def clear_history(self) -> None:
        """Clear agent history"""
        self.message_history.clear()
        self.reasoning_history.clear()
        self.tool_results.clear()
        self.logger.info("Agent history cleared")

    def get_memory_context(self, limit: int = 10) -> str:
        """Get recent memory context for prompts"""
        if not self.memory:
            return ""

        recent_messages = self.message_history[-limit:]
        context_parts = []

        for msg in recent_messages:
            context_parts.append(f"{msg.sender}: {msg.content[:200]}")

        return "\n".join(context_parts)

    def format_response(
        self,
        content: str,
        include_reasoning: bool = False
    ) -> str:
        """Format response for display"""
        if include_reasoning and self.reasoning_history:
            reasoning = "\n\n".join([
                f"{i+1}. {step}"
                for i, step in enumerate(self.reasoning_history[-5:])
            ])
            return f"{content}\n\n**Reasoning Trace:**\n{reasoning}"
        return content

    async def stream_response(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Stream response for real-time display.

        Args:
            input_text: User input
            context: Optional context

        Yields:
            Response chunks
        """
        response = await self.process(input_text, context)

        if response.success:
            for chunk in response.content:
                yield chunk
        else:
            yield f"Error: {response.errors[0] if response.errors else 'Unknown error'}"

    def validate_input(self, input_text: str) -> bool:
        """Validate input before processing"""
        if not input_text or not input_text.strip():
            return False
        if len(input_text) > 10000:
            self.logger.warning(f"Input too long: {len(input_text)} chars")
            return False
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get agent status for monitoring"""
        return {
            "name": self.config.name,
            "role": self.config.role,
            "state": self.state.value,
            "message_count": len(self.message_history),
            "reasoning_steps": len(self.reasoning_history),
            "tools_available": len(self.config.tools),
            "tools_used": list(self.tool_results.keys()),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.config.name}, state={self.state.value})>"
