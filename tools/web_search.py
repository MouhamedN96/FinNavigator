"""Web search tool — DuckDuckGo, no API key required.

Returns a short list of result titles + URLs + snippets the agent can reason over.
"""

from __future__ import annotations

from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(description="What to search for. Be specific.")
    max_results: int = Field(default=5, description="How many results to return (1-10).")


class WebSearchTool(BaseTool):
    """DuckDuckGo web search — text-only, no auth."""

    name: str = "web_search"
    description: str = (
        "Search the live web (DuckDuckGo). Use for current events, recent news, "
        "or anything the model wouldn't know from training data. Returns ranked "
        "list of {title, href, body}."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            return f"Error: duckduckgo-search not installed ({e})"

        max_results = max(1, min(int(max_results or 5), 10))
        try:
            with DDGS() as ddg:
                hits: List[dict] = list(ddg.text(query, max_results=max_results))
        except Exception as e:
            return f"Search failed: {type(e).__name__}: {e}"

        if not hits:
            return f"No results for: {query}"

        lines = [f"Search results for: {query}", ""]
        for i, h in enumerate(hits, 1):
            title = h.get("title", "")
            href = h.get("href") or h.get("url", "")
            body = h.get("body", "")[:240]
            lines.append(f"{i}. {title}")
            lines.append(f"   {href}")
            if body:
                lines.append(f"   {body}")
            lines.append("")
        return "\n".join(lines)

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(query, max_results, run_manager)
