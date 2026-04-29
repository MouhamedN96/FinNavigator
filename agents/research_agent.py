"""
Research Agent - SEC Filing Specialist
======================================

Specialized agent for SEC filings research and company analysis.
Handles regulatory filings, document extraction, and company background research.

Author: MiniMax Agent
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import re
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentConfig, AgentResponse, AgentCapability, AgentState
from tools.vision_inference import LocalVisionInference


@dataclass
class SECSection:
    """Represents an SEC filing section"""
    name: str
    key: str
    description: str
    importance: str = "medium"  # low, medium, high, critical


# Common SEC sections with importance
SEC_SECTIONS = {
    "item1": SECSection("Business", "item1", "Company overview and business description", "high"),
    "item1a": SECSection("Risk Factors", "item1a", "Material risks facing the company", "critical"),
    "item2": SECSection("MD&A", "item2", "Management Discussion and Analysis", "critical"),
    "item7": SECSection("Critical Accounting", "item7", "Critical accounting estimates", "high"),
    "item7a": SECSection("Market Risks", "item7a", "Quantitative market risk disclosures", "high"),
    "item8": SECSection("Financials", "item8", "Financial statements and notes", "critical"),
}


class ResearchAgent(BaseAgent):
    """
    Research Agent for SEC Filings and Company Analysis

    Specialized capabilities:
    - SEC EDGAR filing search and extraction
    - Company background and industry research
    - Financial statement analysis
    - Regulatory compliance checks
    - Historical filing comparison

    Example:
        agent = ResearchAgent(config, llm_client)
        analysis = await agent.research_company("NVDA", focus="risk_analysis")
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        sec_api_key: Optional[str] = None,
        memory_manager: Optional[Any] = None,
        vision_model_path: Optional[str] = None
    ):
        super().__init__(config, llm_client, memory_manager)
        self.sec_api_key = sec_api_key
        # Initialize local vision if path provided
        self.local_vision = None
        if vision_model_path and os.path.exists(vision_model_path):
            self.local_vision = LocalVisionInference(vision_model_path)
            self.add_reasoning_step(f"Local vision model detected at {vision_model_path}")
        
        # Initialize RAG tools
        self.kb_search = None
        self.kb_index = None
        self.visual_index = None
        
        if config.tools:
            for tool in config.tools:
                if tool.name == "knowledge_search":
                    self.kb_search = tool
                elif tool.name == "knowledge_index":
                    self.kb_index = tool
                elif tool.name == "index_visual_context":
                    self.visual_index = tool

    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return [
            AgentCapability.RESEARCH,
            AgentCapability.REASONING,
            AgentCapability.EXECUTION,
        ]

    def _extract_ticker_from_query(self, query: str) -> Optional[str]:
        """Extract stock ticker from query"""
        # Common patterns
        patterns = [
            r'\b([A-Z]{1,5})\b',  # Capital letters
            r'ticker[:\s]+([A-Z]{1,5})',
            r'stock[:\s]+([A-Z]{1,5})',
        ]

        for pattern in patterns:
            match = re.search(pattern, query.upper())
            if match:
                return match.group(1)

        return None

    async def search_filings(
        self,
        ticker: str,
        form_types: List[str] = None,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Search for SEC filings"""
        self.add_reasoning_step(f"Searching SEC filings for {ticker}")

        if form_types is None:
            form_types = ["10-K", "10-Q"]

        results = {
            "ticker": ticker,
            "filings": [],
            "summary": {}
        }

        # Import SEC API tools
        try:
            from sec_api import QueryApi
            import os

            api_key = self.sec_api_key or os.getenv("SEC_API_KEY", "")
            if not api_key:
                return {"error": "SEC_API_KEY not configured"}

            query_api = QueryApi(api_key=api_key)

            for form_type in form_types:
                query = {
                    "query": f'ticker:{ticker} AND formType:"{form_type}"',
                    "from": "0",
                    "size": str(max_results),
                    "sort": [{"filedAt": {"order": "desc"}}]
                }

                response = query_api.get_filings(query)
                filings = response.get("filings", [])

                results["filings"].extend(filings)
                results["summary"][form_type] = len(filings)

            self.add_reasoning_step(f"Found {len(results['filings'])} filings")

        except ImportError:
            results["error"] = "sec-api not installed"
        except Exception as e:
            results["error"] = str(e)

        return results

    async def extract_filing_section(
        self,
        ticker: str,
        section_key: str,
        filing_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract specific section from SEC filing"""
        self.add_reasoning_step(f"Extracting {section_key} for {ticker}")

        section_info = SEC_SECTIONS.get(section_key, {})
        section_name = section_info.name if section_info else section_key

        result = {
            "ticker": ticker,
            "section": section_key,
            "section_name": section_name,
            "content": "",
            "metadata": {}
        }

        # Check Knowledge Base First
        if self.kb_search:
            self.add_reasoning_step(f"Checking knowledge base for existing {section_key} data")
            kb_query = f"{ticker} {section_key} {section_name}"
            kb_result = await self.kb_search.ainvoke({"query": kb_query, "top_k": 1})
            if "### Found" in kb_result and len(kb_result) > 200:
                self.add_reasoning_step("Found relevant data in knowledge base, using cached version")
                # Parsing logic for cached result could be added here
                # For now, we continue to live fetch if cache isn't 'perfect'
                pass

        try:
            from sec_api import QueryApi, ExtractorApi
            import os

            api_key = self.sec_api_key or os.getenv("SEC_API_KEY", "")
            if not api_key:
                return {"error": "SEC_API_KEY not configured"}

            query_api = QueryApi(api_key=api_key)
            extractor_api = ExtractorApi(api_key=api_key)

            # Find filing
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
                return {"error": f"No filings found for {ticker}"}

            filing_url = filings[0]["linkToFilingDataSummaries"]
            result["metadata"] = {
                "filed_at": filings[0].get("filedAt"),
                "form_type": filings[0].get("formType"),
                "description": filings[0].get("description")
            }

            # Extract section
            try:
                content = extractor_api.get_section(filing_url, section_key, "text")
                result["content"] = content[:10000]  # Increased limit for better RAG
                result["content_length"] = len(content)
                
                # Auto-index into Knowledge Base for RAG
                if self.kb_index and len(content) > 100:
                    self.add_reasoning_step(f"Auto-indexing {section_key} into knowledge base")
                    await self.kb_index.ainvoke({
                        "texts": [content[:20000]], # Index up to 20k chars
                        "metadata": [{
                            "ticker": ticker,
                            "source": f"SEC {filings[0].get('formType')} - {section_name}",
                            "date": filings[0].get("filedAt"),
                            "section": section_key
                        }]
                    })
            except Exception as e:
                result["error"] = f"Extraction failed: {str(e)}"

        except ImportError:
            result["error"] = "sec-api not installed"
        except Exception as e:
            result["error"] = str(e)

        return result

    async def analyze_risk_factors(self, ticker: str) -> Dict[str, Any]:
        """Analyze risk factors from 10-K filing"""
        self.add_reasoning_step(f"Analyzing risk factors for {ticker}")

        risk_analysis = {
            "ticker": ticker,
            "risks": [],
            "summary": "",
            "categories": {
                "operational": [],
                "financial": [],
                "market": [],
                "regulatory": [],
                "other": []
            }
        }

        # Extract risk factors
        risk_data = await self.extract_filing_section(ticker, "item1a")

        if "error" in risk_data:
            return risk_data

        content = risk_data.get("content", "")

        # Categorize risks (simplified pattern matching)
        risk_keywords = {
            "operational": ["supply chain", "manufacturing", "production", "operations"],
            "financial": ["liquidity", "credit", "currency", "interest rate"],
            "market": ["competition", "market share", "customer", "demand"],
            "regulatory": ["regulation", "compliance", "legal", "policy"]
        }

        # Split content into risk items (usually bullet points)
        risk_items = re.split(r'[•\-\*]|\d+\.', content)

        for item in risk_items[:20]:  # Analyze first 20 items
            item = item.strip()
            if len(item) < 50:
                continue

            categorized = False
            for category, keywords in risk_keywords.items():
                if any(kw.lower() in item.lower() for kw in keywords):
                    risk_analysis["categories"][category].append(item)
                    categorized = True
                    break

            if not categorized:
                risk_analysis["categories"]["other"].append(item)

        # Count totals
        risk_analysis["total_risks"] = sum(
            len(risks) for risks in risk_analysis["categories"].values()
        )

        return risk_analysis

    async def compare_filings(
        self,
        ticker: str,
        metric: str = "revenue",
        periods: int = 4
    ) -> Dict[str, Any]:
        """Compare metric across multiple filing periods"""
        self.add_reasoning_step(f"Comparing {metric} for {ticker} over {periods} periods")

        comparison = {
            "ticker": ticker,
            "metric": metric,
            "periods": [],
            "trend": "unknown"
        }

        # This would use actual extraction and comparison
        # Simplified for demonstration
        comparison["note"] = "Full implementation would extract specific financial metrics"

        return comparison

    async def research_company(
        self,
        ticker: str,
        focus: str = "overview",
        depth: str = "standard"
    ) -> Dict[str, Any]:
        """
        Comprehensive company research.

        Args:
            ticker: Stock ticker symbol
            focus: Research focus - overview, risks, financials, comparison
            depth: Research depth - quick, standard, deep

        Returns:
            Research findings and analysis
        """
        self.update_state(AgentState.THINKING)
        self.add_reasoning_step(f"Starting {depth} research on {ticker}, focus: {focus}")

        research = {
            "ticker": ticker,
            "focus": focus,
            "depth": depth,
            "timestamp": datetime.now().isoformat(),
            "findings": {}
        }

        if focus in ["overview", "risks", "financials"]:
            # Search for filings
            filings_result = await self.search_filings(ticker, max_results=3)
            research["findings"]["filings"] = filings_result

        if focus in ["overview"]:
            # Company overview from business section
            overview = await self.extract_filing_section(ticker, "item1")
            research["findings"]["overview"] = overview

        if focus in ["risks"]:
            # Risk factor analysis
            risks = await self.analyze_risk_factors(ticker)
            research["findings"]["risks"] = risks

        if focus in ["financials"]:
            # MD&A analysis
            mda = await self.extract_filing_section(ticker, "item2")
            research["findings"]["mdna"] = mda

        # Generate summary using LLM
        summary_prompt = f"""Summarize the research findings for {ticker}:

Focus: {focus}
Depth: {depth}

Findings: {json.dumps(research['findings'], indent=2)}

Provide a concise executive summary suitable for a financial analyst.
"""
        messages = [SystemMessage(content=summary_prompt)]
        summary_response = await self.llm.ainvoke(messages)
        research["summary"] = summary_response.content

        self.update_state(AgentState.IDLE)
        return research

    async def process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process research request"""
        self.update_state(AgentState.THINKING)
        start_time = datetime.now()

        # Extract ticker
        ticker = self._extract_ticker_from_query(input_text)

        if not ticker:
            self.add_reasoning_step("No ticker found, providing general research")
            return AgentResponse(
                success=False,
                content="Please specify a stock ticker symbol for research (e.g., 'NVDA', 'AAPL')"
            )

        # Determine focus
        focus = "overview"
        if "risk" in input_text.lower():
            focus = "risks"
        elif "financial" in input_text.lower() or "revenue" in input_text.lower():
            focus = "financials"

        # Conduct research
        research = await self.research_company(ticker, focus=focus)

        execution_time = (datetime.now() - start_time).total_seconds()

        return AgentResponse(
            success="error" not in research.get("findings", {}),
            content=research.get("summary", "Research completed"),
            reasoning_steps=self.reasoning_history,
            metadata=research
        )

    async def process_vision(
        self,
        input_text: str,
        image_data: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process multimodal research request.
        Analyzes images of filings, tables, or financial documents.
        """
        self.update_state(AgentState.THINKING)
        self.add_reasoning_step("Analyzing visual input (image) for research context")
        
        # Prepare multimodal message
        image_url = f"data:image/jpeg;base64,{image_data}"
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": input_text or "Analyze this financial document image."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        )
        
        try:
            # Use local vision model if available, otherwise fallback to cloud LLM
            if self.local_vision:
                self.add_reasoning_step("Using local Qwen3-VL for image analysis")
                content = self.local_vision.process_image(
                    prompt=input_text or "Analyze this financial document image.",
                    image_base64=image_data
                )
                response_content = content
            else:
                self.add_reasoning_step("Using cloud LLM for vision (local model not active)")
                response = await self.llm.ainvoke([message])
                response_content = response.content
            
            # Index visual context if available
            if self.visual_index and response_content:
                self.add_reasoning_step("Indexing visual context for future retrieval")
                await self.visual_index.ainvoke({
                    "image_description": response.content,
                    "image_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "metadata": {"ticker": self._extract_ticker_from_query(input_text) or "Unknown"}
                })

            self.update_state(AgentState.IDLE)
            return AgentResponse(
                success=True,
                content=response.content,
                reasoning_steps=self.reasoning_history,
                metadata={"has_image": True, "input_text": input_text}
            )
        except Exception as e:
            self.update_state(AgentState.IDLE)
            return AgentResponse(
                success=False,
                content=f"Research vision processing failed: {str(e)}",
                reasoning_steps=self.reasoning_history
            )

    async def stream_process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Stream research progress"""
        ticker = self._extract_ticker_from_query(input_text)

        if not ticker:
            yield "Please specify a stock ticker symbol for research."
            return

        yield f"## Researching {ticker}\n\n"

        # Stream filing search
        yield "1. Searching SEC filings...\n"
        filings = await self.search_filings(ticker)
        yield f"   Found {len(filings.get('filings', []))} filings\n\n"

        # Stream section extraction
        yield "2. Extracting key sections...\n"
        overview = await self.extract_filing_section(ticker, "item1")
        if "content" in overview:
            yield f"   - Business overview: {len(overview['content'])} chars\n"

        yield "\n## Summary\n\n"
        yield "Research completed. Use research_company() for full analysis."


class MultiCompanyResearcher(ResearchAgent):
    """
    Multi-company research agent for competitive analysis.

    Extends ResearchAgent to handle multiple companies simultaneously
    and generate comparative analyses.
    """

    def __init__(self, config: AgentConfig, llm_client: Any, **kwargs):
        super().__init__(config, llm_client, **kwargs)
        self.companies = []

    def set_companies(self, tickers: List[str]) -> None:
        """Set list of companies to research"""
        self.companies = tickers
        self.add_reasoning_step(f"Set companies for research: {tickers}")

    async def research_multiple(self, focus: str = "comparison") -> Dict[str, Any]:
        """Research multiple companies"""
        if not self.companies:
            return {"error": "No companies set"}

        results = {}
        for ticker in self.companies:
            results[ticker] = await self.research_company(ticker, focus=focus)

        # Generate comparison
        comparison = await self._generate_comparison(results)
        return comparison

    async def _generate_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis"""
        comparison_prompt = f"""Compare the following companies:

{json.dumps(results, indent=2)}

Generate a comparative analysis highlighting:
1. Key differences in business models
2. Risk profile comparison
3. Financial performance comparison

Format as a structured comparison table and narrative summary.
"""
        messages = [SystemMessage(content=comparison_prompt)]
        response = await self.llm.ainvoke(messages)

        return {
            "companies": self.companies,
            "comparison": response.content,
            "details": results
        }
