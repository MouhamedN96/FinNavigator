"""
SEC EDGAR AgentKit wrapper
==========================

Thin shim around the third-party `sec-edgar-agentkit` package
(https://github.com/stefanoamorelli/sec-edgar-agentkit) so the rest of
FinNavigator can opt into community-maintained SEC tools instead of the
bundled `SECSearchTool` / `SECExtractTool`.

[Unverified] The exact import surface of `sec-edgar-agentkit` is not pinned
here — package APIs evolve. This wrapper tries a few common LangChain-toolkit
patterns and gracefully falls back to the existing tools if none resolve.
If you've installed the package and `get_sec_edgar_tools()` returns the legacy
fallback list, open the package's README and replace the `_load_*` helpers
below with the actual import path it documents.

Usage:
    from tools.sec_agentkit_tools import get_sec_edgar_tools
    sec_tools = get_sec_edgar_tools()
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def _load_via_get_tools() -> List[BaseTool] | None:
    """Pattern A: top-level `get_tools()` factory function."""
    try:
        from sec_edgar_agentkit import get_tools  # type: ignore[attr-defined]
        tools = get_tools()
        return list(tools) if tools else None
    except Exception:
        return None


def _load_via_toolkit() -> List[BaseTool] | None:
    """Pattern B: `SECEdgarToolkit().get_tools()` (LangChain toolkit convention)."""
    try:
        from sec_edgar_agentkit import SECEdgarToolkit  # type: ignore[attr-defined]
        return list(SECEdgarToolkit().get_tools())
    except Exception:
        return None


def _load_via_langchain_module() -> List[BaseTool] | None:
    """Pattern C: `sec_edgar_agentkit.langchain` submodule with `tools` list."""
    try:
        from sec_edgar_agentkit.langchain import tools as _tools  # type: ignore[attr-defined]
        return list(_tools)
    except Exception:
        return None


def _load_via_tools_module() -> List[BaseTool] | None:
    """Pattern D: explicit tool classes under `sec_edgar_agentkit.tools`."""
    try:
        import sec_edgar_agentkit.tools as _t  # type: ignore[import-not-found]
        candidates = [
            getattr(_t, name)
            for name in dir(_t)
            if not name.startswith("_")
        ]
        instantiated: List[BaseTool] = []
        for cls in candidates:
            try:
                if isinstance(cls, type) and issubclass(cls, BaseTool):
                    instantiated.append(cls())
            except Exception:
                continue
        return instantiated or None
    except Exception:
        return None


def _legacy_fallback() -> List[BaseTool]:
    """Fallback: the SEC tools bundled with FinNavigator."""
    try:
        from .financial_tools import SECSearchTool, SECExtractTool
        return [SECSearchTool(), SECExtractTool()]
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not instantiate legacy SEC tools: {e}")
        return []


def get_sec_edgar_tools() -> List[BaseTool]:
    """Return SEC EDGAR tools, preferring `sec-edgar-agentkit` if installed.

    Tries known import patterns, then falls back to the bundled SECSearchTool
    and SECExtractTool. Always returns a list (possibly empty) — never raises.
    """
    for loader in (
        _load_via_get_tools,
        _load_via_toolkit,
        _load_via_langchain_module,
        _load_via_tools_module,
    ):
        tools = loader()
        if tools:
            logger.info(
                f"Loaded {len(tools)} SEC tools via {loader.__name__} "
                f"from sec-edgar-agentkit"
            )
            return tools

    logger.info("sec-edgar-agentkit not detected; using bundled SEC tools")
    return _legacy_fallback()


__all__ = ["get_sec_edgar_tools"]
