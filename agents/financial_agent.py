"""
Financial Agent - Deep ReAct Agent
==================================

A sophisticated reasoning agent with financial domain expertise using
the ReAct (Reason + Act) pattern for multi-step problem solving.

Features:
- ReAct reasoning loop with tool execution
- Chain-of-thought prompting
- Dynamic tool selection
- Error recovery and retry logic

Author: MiniMax Agent
"""

from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import asyncio
import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function

from .base_agent import BaseAgent, AgentConfig, AgentResponse, AgentCapability, AgentState, AgentMessage


class ReasoningStep:
    """Represents a single reasoning step in the ReAct loop"""

    def __init__(self, thought: str, action: Optional[str] = None,
                 action_input: Optional[Dict] = None, observation: Optional[str] = None):
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "timestamp": self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        parts = [f"Thought: {self.thought}"]
        if self.action:
            parts.append(f"Action: {self.action}")
        if self.action_input:
            parts.append(f"Action Input: {self.action_input}")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts)


class ReActOutputParser:
    """Parser for ReAct agent output format"""

    THOUGHT_PATTERN = r"Thought:(.+?)(?:Action:|Observation:|$)"
    ACTION_PATTERN = r"Action:\s*(\w+)"
    ACTION_INPUT_PATTERN = r"Action Input:\s*(\{[^}]+\}|\[[^\]]+\]|.+?)(?=Observation:|$)"
    OBSERVATION_PATTERN = r"Observation:\s*(.+?)(?=Thought:|Action:|$)"

    def __init__(self):
        self.reasoning_steps: List[ReasoningStep] = []

    def parse(self, text: str) -> ReasoningStep:
        """Parse a ReAct output string into a reasoning step"""
        thought_match = re.search(self.THOUGHT_PATTERN, text, re.DOTALL)
        action_match = re.search(self.ACTION_PATTERN, text)
        action_input_match = re.search(self.ACTION_INPUT_PATTERN, text, re.DOTALL)
        observation_match = re.search(self.OBSERVATION_PATTERN, text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else None

        action_input = None
        if action_input_match:
            raw_input = action_input_match.group(1).strip()
            try:
                action_input = json.loads(raw_input)
            except json.JSONDecodeError:
                action_input = raw_input

        observation = observation_match.group(1).strip() if observation_match else None

        step = ReasoningStep(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation
        )
        self.reasoning_steps.append(step)
        return step

    def get_trace(self) -> str:
        """Get formatted reasoning trace"""
        return "\n\n".join([str(step) for step in self.reasoning_steps])


@dataclass
class ReActConfig:
    """Configuration for ReAct agent"""
    max_iterations: int = 10
    max_execution_time: float = 60.0
    early_stopping_threshold: float = 0.8
    retry_on_error: bool = True
    max_retries: int = 3
    verbose: bool = True


class FinancialAgent(BaseAgent):
    """
    Deep ReAct Agent for Financial Analysis

    Implements the ReAct (Reason + Act) pattern for autonomous problem solving:
    1. Think about what needs to be done
    2. Select appropriate tool
    3. Execute tool and observe result
    4. Evaluate if goal is achieved
    5. Repeat or respond

    Tools are registered at initialization and dynamically selected based on context.

    Example:
        agent = FinancialAgent(config, llm_client)
        agent.register_tools([sec_search, calculator, knowledge_search])
        response = await agent.process("What's NVDA's latest revenue?")
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        react_config: Optional[ReActConfig] = None,
        memory_manager: Optional[Any] = None
    ):
        super().__init__(config, llm_client, memory_manager)
        self.react_config = react_config or ReActConfig()
        self.tools_map: Dict[str, Any] = {}
        self.parser = ReActOutputParser()

        # Setup system prompt for ReAct
        self._setup_react_prompt()

    def _setup_react_prompt(self) -> None:
        """Setup the ReAct prompt template"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.config.tools
        ])

        self.react_system_prompt = f"""You are a Financial Expert AI with access to various tools.

You operate using the ReAct (Reason + Act) pattern:
1. THINK: Analyze the question and plan your approach
2. ACT: Use a tool to gather information or perform an action
3. OBSERVE: Review the result of your action
4. Repeat until you can provide a complete answer

Available Tools:
{tool_descriptions}

Guidelines:
- Always check your knowledge base first for existing information
- Use SEC tools for regulatory filings and official data
- Use calculator for numerical analysis
- Use messaging tools to alert users of important findings
- Be thorough but efficient - avoid redundant tool calls
- If a tool fails, try an alternative approach

Response Format:
Thought: [Your reasoning about what to do next]
Action: [Tool name or "final" if answering]
Action Input: [Parameters for the tool, if applicable]
Observation: [Result from the tool, if applicable]
"""

    def register_tool(self, tool: Any) -> None:
        """Register a tool for use by the agent"""
        self.tools_map[tool.name] = tool
        self.logger.debug(f"Registered tool: {tool.name}")

    def register_tools(self, tools: List[Any]) -> None:
        """Register multiple tools"""
        for tool in tools:
            self.register_tool(tool)

    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return [
            AgentCapability.REASONING,
            AgentCapability.RESEARCH,
            AgentCapability.ANALYSIS,
            AgentCapability.EXECUTION,
            AgentCapability.CALCULATION,
        ]

    def _format_tools_for_prompt(self) -> str:
        """Format tools list for prompt"""
        lines = []
        for name, tool in self.tools_map.items():
            lines.append(f"{name}: {tool.description}")
        return "\n".join(lines)

    async def _generate_react_response(
        self,
        question: str,
        context: Optional[str] = None,
        run_manager: Optional[Any] = None
    ) -> str:
        """Generate ReAct response from LLM"""
        prompt = f"""{self.react_system_prompt}

{'Context from previous conversation:\n' + context if context else ''}

Question: {question}

Remember to use the ReAct format and only call tools when necessary.
If you have enough information to answer, respond directly.
"""

        messages = [SystemMessage(content=prompt)]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            return f"Error generating response: {str(e)}"

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: Union[str, Dict]
    ) -> str:
        """Execute a tool and return observation"""
        if tool_name not in self.tools_map:
            return f"Error: Tool '{tool_name}' not found"

        tool = self.tools_map[tool_name]

        try:
            if isinstance(tool_input, str):
                tool_input = json.loads(tool_input)

            result = await tool.ainvoke(tool_input)
            return str(result)

        except json.JSONDecodeError:
            return f"Error: Invalid JSON in tool input"
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _should_stop(self, steps: List[ReasoningStep], answer_threshold: float = 0.8) -> bool:
        """Determine if the agent should stop reasoning"""
        if not steps:
            return False

        # Check if last step was a final response
        last_step = steps[-1]
        if last_step.action is None and last_step.observation is None:
            return True

        # Check if we've hit max iterations
        if len(steps) >= self.react_config.max_iterations:
            return True

        # Check for convergence - if last few steps have same thought pattern
        if len(steps) >= 3:
            recent_thoughts = [s.thought.lower() for s in steps[-3:]]
            if len(set(recent_thoughts)) == 1:
                return True

        return False

    def _construct_final_response(self, steps: List[ReasoningStep], question: str) -> str:
        """Construct final response from reasoning steps"""
        # Find the final answer from the last thought
        for step in reversed(steps):
            if step.action is None or step.action.lower() in ["final", "respond", "answer"]:
                return step.thought

        # Otherwise, synthesize from reasoning
        return steps[-1].thought if steps else "Unable to provide an answer."

    async def process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process input using LangGraph's tool-calling ReAct agent.

        Replaces the legacy regex-based ReAct loop with native tool-calling,
        which is significantly more reliable on local SLMs (Qwen3, Llama 3+).
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage, SystemMessage as _SystemMessage

        self.update_state(AgentState.THINKING)
        self.reasoning_history = []
        start_time = datetime.now()

        tools = list(self.tools_map.values()) or list(self.config.tools)

        try:
            agent = create_react_agent(
                self.llm,
                tools,
                prompt=self.get_system_prompt(),
            )

            messages: List[Any] = []
            if context and isinstance(context, dict) and context.get("context"):
                messages.append(_SystemMessage(content=str(context["context"])))
            messages.append(HumanMessage(content=input_text))

            self.add_reasoning_step(f"Invoking LangGraph agent with {len(tools)} tools")

            result = await agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": self.react_config.max_iterations * 2},
            )

            history = result.get("messages", [])
            final_message = history[-1] if history else None
            final_content = getattr(final_message, "content", "") or "(no response)"

            tools_used: List[str] = []
            reasoning_steps: List[Dict[str, Any]] = []
            for msg in history:
                tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                    if name:
                        tools_used.append(name)
                        reasoning_steps.append({
                            "thought": getattr(msg, "content", "") or "",
                            "action": name,
                            "action_input": args,
                            "observation": None,
                        })

            self.update_state(AgentState.IDLE)
            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResponse(
                success=True,
                content=final_content if isinstance(final_content, str) else str(final_content),
                reasoning_steps=reasoning_steps,
                tools_used=tools_used,
                execution_time=execution_time,
                metadata={"backend": "langgraph.create_react_agent", "message_count": len(history)},
            )

        except Exception as e:
            self.logger.error(f"LangGraph agent error: {e}")
            self.update_state(AgentState.ERROR)
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResponse(
                success=False,
                content=f"Agent error: {e}",
                errors=[str(e)],
                execution_time=execution_time,
            )

    async def process_vision(
        self,
        input_text: str,
        image_data: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process multimodal input using LangGraph's tool-calling ReAct agent."""
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage, SystemMessage as _SystemMessage

        self.update_state(AgentState.THINKING)
        self.reasoning_history = []
        start_time = datetime.now()

        tools = list(self.tools_map.values()) or list(self.config.tools)

        try:
            agent = create_react_agent(
                self.llm,
                tools,
                prompt=self.get_system_prompt(),
            )

            # Format multimodal message
            content = [
                {"type": "text", "text": input_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]

            messages: List[Any] = []
            if context and isinstance(context, dict) and context.get("context"):
                messages.append(_SystemMessage(content=str(context["context"])))
            messages.append(HumanMessage(content=content))

            self.add_reasoning_step(f"Invoking multimodal agent with image and {len(tools)} tools")

            result = await agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": self.react_config.max_iterations * 2},
            )

            history = result.get("messages", [])
            final_message = history[-1] if history else None
            final_content = getattr(final_message, "content", "") or "(no response)"

            # Extract tool usage for trace
            tools_used: List[str] = []
            reasoning_steps: List[Dict[str, Any]] = []
            for msg in history:
                tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name:
                        tools_used.append(name)
                        reasoning_steps.append({
                            "thought": getattr(msg, "content", "") or "",
                            "action": name,
                            "action_input": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None),
                            "observation": None,
                        })

            self.update_state(AgentState.IDLE)
            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResponse(
                success=True,
                content=final_content if isinstance(final_content, str) else str(final_content),
                reasoning_steps=reasoning_steps,
                tools_used=tools_used,
                execution_time=execution_time,
                metadata={"backend": "langgraph.multimodal", "message_count": len(history)},
            )

        except Exception as e:
            self.logger.error(f"Multimodal agent error: {e}")
            self.update_state(AgentState.ERROR)
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResponse(
                success=False,
                content=f"Multimodal error: {e}",
                errors=[str(e)],
                execution_time=execution_time,
            )

    async def stream_process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Stream the reasoning process for real-time display.

        Yields:
            Reasoning updates and final response
        """
        response = await self.process(input_text, context)

        yield "## ReAct Reasoning Trace\n\n"

        for i, step_dict in enumerate(response.reasoning_steps):
            step = ReasoningStep(
                thought=step_dict["thought"],
                action=step_dict.get("action"),
                action_input=step_dict.get("action_input"),
                observation=step_dict.get("observation")
            )

            yield f"### Step {i + 1}\n\n"
            yield f"**Thought:** {step.thought}\n\n"

            if step.action:
                yield f"**Action:** {step.action}\n\n"
            if step.action_input:
                yield f"**Action Input:** {json.dumps(step.action_input, indent=2)}\n\n"
            if step.observation:
                yield f"**Observation:** {step.observation[:500]}...\n\n"

            yield "\n---\n\n"

        yield f"\n\n## Final Answer\n\n{response.content}"

    def get_tool_status(self) -> Dict[str, Any]:
        """Get status of registered tools"""
        return {
            "total_tools": len(self.tools_map),
            "tools": list(self.tools_map.keys()),
            "tool_descriptions": {
                name: tool.description[:100]
                for name, tool in self.tools_map.items()
            }
        }


class PlanAndExecuteAgent(FinancialAgent):
    """
    Plan-and-Execute Agent Variant

    High-level planner that creates a plan first, then executes each step.
    Better for complex, multi-step tasks that require planning.

    Flow:
    1. Create plan (break down task into steps)
    2. Execute plan steps sequentially
    3. Synthesize results into final response
    """

    async def _create_plan(self, task: str) -> List[Dict[str, Any]]:
        """Create execution plan from task"""
        planning_prompt = f"""Break down this financial task into sequential steps:

Task: {task}

Create a numbered list of steps, where each step is a concrete action.
Each step should be self-contained and use one tool or action.

Steps:"""

        messages = [SystemMessage(content=planning_prompt)]
        response = await self.llm.ainvoke(messages)

        # Parse steps from response
        steps = []
        for line in response.content.split("\n"):
            if line.strip() and line[0].isdigit():
                step_text = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                steps.append({"task": step_text, "status": "pending"})

        return steps

    async def process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process with plan-and-execute pattern"""
        self.update_state(AgentState.THINKING)
        start_time = datetime.now()

        self.add_reasoning_step(f"Creating execution plan for: {input_text[:100]}...")

        # Create plan
        plan = await self._create_plan(input_text)
        self.add_reasoning_step(f"Created plan with {len(plan)} steps")

        results = []
        tools_used = []

        # Execute plan
        for i, step in enumerate(plan):
            self.add_reasoning_step(f"Executing step {i+1}: {step['task']}")

            # Select appropriate tool for this step
            step_response = await self._execute_step(step["task"])
            results.append(step_response)

            if step_response.get("tool_used"):
                tools_used.append(step_response["tool_used"])

        # Synthesize results
        synthesis = await self._synthesize_results(input_text, results)

        execution_time = (datetime.now() - start_time).total_seconds()

        return AgentResponse(
            success=True,
            content=synthesis,
            reasoning_steps=[{"plan": plan, "results": results}],
            tools_used=tools_used,
            execution_time=execution_time
        )

    async def _execute_step(self, step_task: str) -> Dict[str, Any]:
        """Execute a single plan step"""
        # Match step to appropriate tool
        step_lower = step_task.lower()

        if "sec" in step_lower or "filing" in step_lower or "10-k" in step_lower or "10-q" in step_lower:
            tool = self.tools_map.get("sec_search")
        elif "search" in step_lower or "find" in step_lower or "look up" in step_lower:
            tool = self.tools_map.get("knowledge_search")
        elif "calculate" in step_lower or "compute" in step_lower or "math" in step_lower:
            tool = self.tools_map.get("calculator")
        else:
            tool = None

        if tool:
            result = await tool.ainvoke({"query": step_task})
            return {"result": result, "tool_used": tool.name}

        return {"result": step_task, "tool_used": None}

    async def _synthesize_results(self, original_task: str, results: List[Dict]) -> str:
        """Synthesize step results into final response"""
        synthesis_prompt = f"""Based on the following results from executing steps for the task:

Original Task: {original_task}

Results:
{json.dumps(results, indent=2)}

Provide a comprehensive answer that addresses the original task using all the results.
"""

        messages = [SystemMessage(content=synthesis_prompt)]
        response = await self.llm.ainvoke(messages)
        return response.content
