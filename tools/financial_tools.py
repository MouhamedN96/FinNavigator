"""
Financial Tools for Agents
==========================

Specialized tools for SEC filings, portfolio analysis, risk calculation,
and market data retrieval.

Author: MiniMax Agent
"""

from typing import Type, Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import os
import requests


class SECSearchInput(BaseModel):
    """Input schema for SEC search tool"""
    ticker: str = Field(description="Stock ticker symbol (e.g., 'NVDA', 'AAPL')")
    form_type: str = Field(default="10-Q", description="Form type: 10-K, 10-Q, 8-K, etc.")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    max_filings: int = Field(default=5, description="Maximum number of filings to return")


class SECExtractInput(BaseModel):
    """Input schema for SEC extraction tool"""
    ticker: str = Field(description="Stock ticker symbol")
    section: str = Field(description="Section to extract: item1, item1a, item2, item7, etc.")
    filing_date: Optional[str] = Field(default=None, description="Specific filing date (YYYY-MM-DD)")


class SECSearchTool(BaseTool):
    """
    SEC EDGAR Search Tool for finding regulatory filings.

    Use this to:
    - Find 10-K (annual reports) and 10-Q (quarterly reports)
    - Locate 8-K (material events)
    - Search for proxy statements (DEF 14A)
    - Track filing history

    Returns filing metadata including dates, descriptions, and links.
    """

    name: str = "sec_search"
    description: str = """Search SEC EDGAR database for company filings.
    Use ticker symbol to find annual reports (10-K), quarterly reports (10-Q),
    and material events (8-K). Returns filing metadata and links."""

    args_schema: Type[BaseModel] = SECSearchInput
    api_key: str = Field(default_factory=lambda: os.getenv("SEC_API_KEY", ""), exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("SEC_API_KEY", "")

    def _run(
        self,
        ticker: str,
        form_type: str = "10-Q",
        start_date: Optional[str] = None,
        max_filings: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute SEC search"""
        if not self.api_key:
            return "Error: SEC_API_KEY not configured"

        try:
            from sec_api import QueryApi
            query_api = QueryApi(api_key=self.api_key)

            query = {
                "query": f'ticker:{ticker} AND formType:"{form_type}"',
                "from": "0",
                "size": str(max_filings),
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            if start_date:
                query["query"] += f' AND filedAt:[{start_date} TO *]'

            response = query_api.get_filings(query)
            filings = response.get("filings", [])

            if not filings:
                return f"No {form_type} filings found for {ticker}"

            results = []
            for filing in filings[:max_filings]:
                results.append({
                    "ticker": filing.get("ticker", ticker),
                    "form": filing.get("formType", form_type),
                    "filedAt": filing.get("filedAt", "N/A"),
                    "description": filing.get("description", "N/A"),
                    "link": filing.get("linkToFiling", "N/A"),
                })

            output = f"Found {len(results)} {form_type} filings for {ticker}:\n\n"
            for i, f in enumerate(results, 1):
                output += f"{i}. {f['form']} filed on {f['filedAt'][:10]}\n"
                output += f"   Description: {f['description']}\n"
                output += f"   Link: {f['link']}\n\n"

            return output

        except ImportError:
            return "Error: sec-api not installed. Run: pip install sec-api"
        except Exception as e:
            return f"Error searching SEC: {str(e)}"

    async def _arun(
        self,
        ticker: str,
        form_type: str = "10-Q",
        start_date: Optional[str] = None,
        max_filings: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(ticker, form_type, start_date, max_filings, run_manager)


class SECExtractTool(BaseTool):
    """
    SEC Section Extraction Tool for reading specific filing sections.

    Extracts text from specific sections:
    - Item 1: Business Overview
    - Item 1A: Risk Factors
    - Item 2: MD&A (Management Discussion)
    - Item 7: Critical Accounting Estimates
    - Item 7A: Quantitative Market Risks
    - Item 8: Financial Statements
    """

    name: str = "sec_extract"
    description: str = """Extract specific sections from SEC filings.
    Use for reading Management Discussion, Risk Factors, or other
    specific sections from 10-K or 10-Q filings."""

    args_schema: Type[BaseModel] = SECExtractInput
    api_key: str = Field(default_factory=lambda: os.getenv("SEC_API_KEY", ""), exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("SEC_API_KEY", "")

    def _run(
        self,
        ticker: str,
        section: str,
        filing_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute SEC extraction"""
        if not self.api_key:
            return "Error: SEC_API_KEY not configured"

        try:
            from sec_api import QueryApi, ExtractorApi
            query_api = QueryApi(api_key=self.api_key)
            extractor_api = ExtractorApi(api_key=self.api_key)

            # Find the filing
            query = {
                "query": f'ticker:{ticker} AND formType:"10-Q"',
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            if filing_date:
                query["query"] += f' AND filedAt:[* TO {filing_date}]'

            response = query_api.get_filings(query)
            filings = response.get("filings", [])

            if not filings:
                return f"No filings found for {ticker}"

            filing_url = filings[0]["linkToFilingDataSummaries"]

            # Extract section
            section_map = {
                "item1": "item1",
                "item1a": "item1a",
                "item2": "item2",
                "item7": "item7",
                "item7a": "item7a",
                "item8": "item8",
            }

            section_key = section_map.get(section.lower(), section)

            try:
                text = extractor_api.get_section(filing_url, section_key, "text")
                # Truncate if too long
                if len(text) > 3000:
                    text = text[:3000] + "\n\n[Truncated - full text available in original filing]"
                return text
            except Exception as e:
                return f"Error extracting section: {str(e)}"

        except ImportError:
            return "Error: sec-api not installed"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        ticker: str,
        section: str,
        filing_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(ticker, section, filing_date, run_manager)


class PortfolioCalculatorInput(BaseModel):
    """Input schema for portfolio calculator"""
    portfolio_value: float = Field(description="Total portfolio value")
    allocation: Dict[str, float] = Field(description="Asset allocation as percentage dict")


class PortfolioCalculatorTool(BaseTool):
    """
    Portfolio Allocation Calculator.

    Calculates:
    - Dollar amounts for each allocation
    - Rebalancing recommendations
    - Risk-adjusted positioning
    """

    name: str = "portfolio_calculator"
    description: str = """Calculate portfolio allocations and rebalancing recommendations.
    Input total portfolio value and allocation percentages to get dollar amounts."""

    args_schema: Type[BaseModel] = PortfolioCalculatorInput

    def _run(
        self,
        portfolio_value: float,
        allocation: Dict[str, float],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute portfolio calculation"""
        try:
            # Validate allocation sums to 100%
            total_pct = sum(allocation.values())
            if abs(total_pct - 100) > 0.01:
                return f"Warning: Allocation percentages sum to {total_pct}%, not 100%. Normalizing..."

            results = []
            for asset, pct in allocation.items():
                dollar_value = portfolio_value * (pct / 100)
                results.append(f"{asset}: {pct}% = ${dollar_value:,.2f}")

            return "Portfolio Breakdown:\n" + "\n".join(results)

        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        portfolio_value: float,
        allocation: Dict[str, float],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(portfolio_value, allocation, run_manager)


class RiskCalculatorInput(BaseModel):
    """Input schema for risk calculator"""
    metric: str = Field(description="Risk metric: var, sharpe, beta, sortino")
    params: Dict[str, float] = Field(description="Parameters for calculation")


class RiskCalculatorTool(BaseTool):
    """
    Risk Metrics Calculator.

    Calculates:
    - VaR (Value at Risk)
    - Sharpe Ratio
    - Beta
    - Sortino Ratio
    - Maximum Drawdown
    """

    name: str = "risk_calculator"
    description: str = """Calculate risk metrics for investments.
    Supports VaR, Sharpe Ratio, Beta, Sortino Ratio, and other risk measures."""

    args_schema: Type[BaseModel] = RiskCalculatorInput

    def _run(
        self,
        metric: str,
        params: Dict[str, float],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute risk calculation"""
        try:
            metric = metric.lower()

            if metric == "var":
                # Value at Risk
                portfolio_value = params.get("portfolio_value", 100000)
                confidence = params.get("confidence", 0.95)
                volatility = params.get("volatility", 0.15)
                # Parametric VaR
                import numpy as np
                from scipy import stats
                z = stats.norm.ppf(1 - confidence)
                var = portfolio_value * volatility * z
                return f"VaR ({confidence*100}%): ${var:,.2f}\nInterpretation: {100-confidence}% chance of losing more than ${var:,.2f} in one day"

            elif metric == "sharpe":
                # Sharpe Ratio
                return "Sharpe Ratio requires: return, risk_free_rate, volatility"

            elif metric == "beta":
                # Beta calculation
                stock_vol = params.get("stock_volatility", 0.20)
                market_vol = params.get("market_volatility", 0.15)
                correlation = params.get("correlation", 0.7)
                beta = correlation * (stock_vol / market_vol)
                return f"Beta: {beta:.2f}\n>1.0: More volatile than market\n<1.0: Less volatile than market"

            else:
                return f"Unknown metric: {metric}. Supported: var, sharpe, beta, sortino"

        except ImportError:
            return "Error: numpy/scipy required"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        metric: str,
        params: Dict[str, float],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(metric, params, run_manager)


class NewsSearchInput(BaseModel):
    """Input schema for news search"""
    query: str = Field(description="Search query for financial news")
    ticker: Optional[str] = Field(default=None, description="Optional ticker symbol")
    days_back: int = Field(default=7, description="Days to search back")


class NewsSearchTool(BaseTool):
    """
    Financial News Search Tool.

    Searches for:
    - Company news and announcements
    - Market news
    - Economic indicators
    - Industry trends
    """

    name: str = "news_search"
    description: str = """Search financial news for companies, markets, and economic events.
    Use ticker symbol for company-specific news or general queries for market news."""

    args_schema: Type[BaseModel] = NewsSearchInput

    def _run(
        self,
        query: str,
        ticker: Optional[str] = None,
        days_back: int = 7,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute news search"""
        # Placeholder - integrate with news API (Alpha Vantage, NewsAPI, etc.)
        return f"News search results for '{query}' (ticker: {ticker or 'N/A'}):\n[Integrate with NewsAPI or similar for production]"

    async def _arun(
        self,
        query: str,
        ticker: Optional[str] = None,
        days_back: int = 7,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(query, ticker, days_back, run_manager)


class StockDataInput(BaseModel):
    """Input schema for stock data"""
    ticker: str = Field(description="Stock ticker symbol")
    data_type: str = Field(default="price", description="Type: price, volume, metrics")


class StockDataTool(BaseTool):
    """
    Stock Market Data Tool.

    Retrieves:
    - Current price and change
    - Trading volume
    - Key metrics (P/E, market cap, etc.)
    - Historical data
    """

    name: str = "stock_data"
    description: str = """Get stock market data for a ticker symbol.
    Returns current price, volume, and key metrics."""

    args_schema: Type[BaseModel] = StockDataInput

    def _run(
        self,
        ticker: str,
        data_type: str = "price",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute stock data retrieval"""
        # Placeholder - integrate with stock API (yfinance, Alpha Vantage, etc.)
        return f"Stock data for {ticker} ({data_type}):\n[Integrate with yfinance or similar for production]"

    async def _arun(
        self,
        ticker: str,
        data_type: str = "price",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(ticker, data_type, run_manager)
