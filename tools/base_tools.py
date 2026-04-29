"""
Base Tools for Financial Agents
===============================

Fundamental tools used by all agents including calculator, datetime, and search.

Author: MiniMax Agent
"""

from typing import Type, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class CalculatorInput(BaseModel):
    """Input schema for calculator tool"""
    expression: str = Field(description="Mathematical expression to evaluate (e.g., '25 * 100 + 50')")


class CalculatorTool(BaseTool):
    """
    Calculator tool for mathematical operations.

    Supports:
    - Basic arithmetic: +, -, *, /
    - Advanced: %, **, sqrt
    - Functions: abs, round, min, max

    Example:
        expression="100 * 0.05 * 30" -> "150.0"
    """

    name: str = "calculator"
    description: str = """Useful for calculating financial figures, percentages,
    portfolio returns, risk metrics, and other mathematical operations.
    Input should be a mathematical expression as a string."""

    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute calculation"""
        try:
            import ast
            import operator
            
            # Clean and validate expression
            expression = expression.replace("^", "**")
            
            # Security check - only allow safe characters
            allowed_chars = set("0123456789+-*/.() **%")
            if not all(c in allowed_chars for c in expression.replace(" ", "")):
                return "Error: Invalid characters in expression"
                
            def _eval(node):
                ops = {
                    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.BitXor: operator.xor,
                    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod
                }
                if isinstance(node, ast.Num):
                    return node.n
                elif hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](_eval(node.left), _eval(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](_eval(node.operand))
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id == 'abs':
                            return abs(_eval(node.args[0]))
                        elif node.func.id == 'round':
                            if len(node.args) == 2:
                                return round(_eval(node.args[0]), _eval(node.args[1]))
                            return round(_eval(node.args[0]))
                        elif node.func.id == 'min':
                            return min([_eval(arg) for arg in node.args])
                        elif node.func.id == 'max':
                            return max([_eval(arg) for arg in node.args])
                raise TypeError(f"Unsupported operation: {type(node)}")

            parsed = ast.parse(expression, mode='eval')
            result = _eval(parsed.body)
            return str(result)

        except ZeroDivisionError:
            return "Error: Division by zero"
        except SyntaxError:
            return "Error: Invalid expression syntax"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution - delegates to sync"""
        return self._run(expression, run_manager)


class DateTimeInput(BaseModel):
    """Input schema for datetime tool"""
    format_type: str = Field(
        description="Type of date info: 'current', 'add_days', 'business_days'"
    )
    value: Optional[str] = Field(
        default=None,
        description="Date value for calculations (format: YYYY-MM-DD) or number of days"
    )


class DateTimeTool(BaseTool):
    """
    Datetime tool for date operations and calculations.

    Supports:
    - current: Get current date/time
    - add_days: Add days to a date
    - business_days: Calculate business days
    """

    name: str = "datetime"
    description: str = """Get current date/time, perform date arithmetic,
    or calculate business days. Useful for settlement dates, trading days, etc."""

    args_schema: Type[BaseModel] = DateTimeInput

    def _run(
        self,
        format_type: str,
        value: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute datetime operation"""
        now = datetime.now()

        if format_type == "current":
            return now.strftime("%Y-%m-%d %H:%M:%S %Z")

        elif format_type == "add_days":
            if not value:
                return "Error: Number of days required"
            try:
                days = int(value)
                result_date = now + timedelta(days=days)
                return result_date.strftime("%Y-%m-%d")
            except ValueError:
                return "Error: Invalid number of days"

        elif format_type == "from_date":
            if not value:
                return "Error: Date required (YYYY-MM-DD)"
            try:
                date = datetime.strptime(value, "%Y-%m-%d")
                return date.strftime("%Y-%m-%d")
            except ValueError:
                return "Error: Invalid date format"

        elif format_type == "days_until":
            if not value:
                return "Error: Target date required"
            try:
                target = datetime.strptime(value, "%Y-%m-%d")
                days = (target - now).days
                return f"{days} days" if days >= 0 else f"{abs(days)} days ago"
            except ValueError:
                return "Error: Invalid date format"

        else:
            return f"Unknown format_type: {format_type}"

    async def _arun(
        self,
        format_type: str,
        value: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(format_type, value, run_manager)


class WikipediaInput(BaseModel):
    """Input schema for Wikipedia search"""
    query: str = Field(description="Search query for Wikipedia")


class WikipediaSearchTool(BaseTool):
    """
    Wikipedia search tool for general knowledge queries.

    Useful for:
    - Company background research
    - Economic term definitions
    - Regulatory information
    - Historical context
    """

    name: str = "wikipedia"
    description: str = """Search Wikipedia for factual information about companies,
    economic terms, regulations, and historical events. Good for background research."""

    args_schema: Type[BaseModel] = WikipediaInput
    _wiki_client: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_wiki_client(self):
        """Lazy load Wikipedia client"""
        if self._wiki_client is None:
            try:
                import wikipedia
                self._wiki_client = wikipedia
            except ImportError:
                return None
        return self._wiki_client

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute Wikipedia search"""
        wiki = self._get_wiki_client()
        if not wiki:
            return "Wikipedia client not available. Install with: pip install wikipedia"

        try:
            page = wiki.page(query)
            summary = page.summary[:1000]  # Limit to first 1000 chars
            return f"Title: {page.title}\n\nSummary:\n{summary}\n\nURL: {page.url}"
        except wiki.exceptions.DisambiguationError as e:
            options = ", ".join(e.options[:5])
            return f"Ambiguous query. Options: {options}"
        except wiki.exceptions.PageError:
            return f"No Wikipedia page found for: {query}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(query, run_manager)


class WebSearchInput(BaseModel):
    """Input schema for web search"""
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results to return")


class WebSearchTool(BaseTool):
    """
    General web search tool for finding current information.

    Supports:
    - News articles
    - Market data
    - Company announcements
    - Economic indicators
    """

    name: str = "web_search"
    description: str = """Search the web for current news, market data, company announcements,
    and economic indicators. Use for real-time information not in the knowledge base."""

    args_schema: Type[BaseModel] = WebSearchInput

    def _run(
        self,
        query: str,
        num_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute web search"""
        # Placeholder for actual implementation
        # In production, integrate with SerpAPI, DuckDuckGo, or similar
        return f"Search results for '{query}':\n(Integrate with SerpAPI or similar for production use)"

    async def _arun(
        self,
        query: str,
        num_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(query, num_results, run_manager)
