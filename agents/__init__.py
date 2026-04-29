"""
FinNavigator AI - LangChain Deep Agents Module
==============================================

This module provides advanced autonomous agent capabilities for financial intelligence.
Implements ReAct agents, tool-calling agents, and multi-agent collaboration systems.

Architecture:
- Financial Agent: Core reasoning agent with financial domain expertise
- Research Agent: Specialized in SEC filings and market research
- Analyst Agent: Focused on data analysis and portfolio insights
- Supervisor Agent: Orchestrates multiple agents for complex tasks

Author: MiniMax Agent
Version: 1.0.0
"""

from .financial_agent import FinancialAgent
from .research_agent import ResearchAgent
from .analyst_agent import AnalystAgent
from .supervisor_agent import SupervisorAgent
from .base_agent import BaseAgent, AgentConfig

__all__ = [
    "FinancialAgent",
    "ResearchAgent",
    "AnalystAgent",
    "SupervisorAgent",
    "BaseAgent",
    "AgentConfig",
]

__version__ = "1.0.0"
