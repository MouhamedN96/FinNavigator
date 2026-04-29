"""
Analyst Agent - Portfolio and Risk Analysis
==========================================

Specialized agent for portfolio analysis, risk assessment, and investment insights.
Handles numerical analysis, charting recommendations, and risk metrics.

Author: MiniMax Agent
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import re
import math

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent, AgentConfig, AgentResponse, AgentCapability, AgentState


@dataclass
class PortfolioPosition:
    """Represents a portfolio position"""
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    allocation: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def gain_loss(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def gain_loss_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.gain_loss / self.cost_basis) * 100


@dataclass
class RiskMetrics:
    """Risk metrics for a portfolio"""
    portfolio_value: float
    volatility: float
    sharpe_ratio: float = 0.0
    beta: float = 1.0
    var_95: float = 0.0
    max_drawdown: float = 0.0
    sortino_ratio: float = 0.0


class AnalystAgent(BaseAgent):
    """
    Analyst Agent for Portfolio and Risk Analysis

    Specialized capabilities:
    - Portfolio performance analysis
    - Risk metrics calculation (VaR, Sharpe, Beta, etc.)
    - Asset allocation analysis
    - Rebalancing recommendations
    - Investment thesis development

    Example:
        agent = AnalystAgent(config, llm_client)
        analysis = await agent.analyze_portfolio(positions, benchmark="SPY")
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
        memory_manager: Optional[Any] = None
    ):
        super().__init__(config, llm_client, memory_manager)
        self.portfolio: Dict[str, PortfolioPosition] = {}
        self.benchmark = "SPY"

    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return [
            AgentCapability.ANALYSIS,
            AgentCapability.CALCULATION,
            AgentCapability.REASONING,
        ]

    def set_portfolio(self, positions: List[Dict[str, Any]]) -> None:
        """Set portfolio positions"""
        total_value = 0.0

        for pos in positions:
            ticker = pos.get("ticker", "")
            position = PortfolioPosition(
                ticker=ticker,
                shares=pos.get("shares", 0),
                avg_cost=pos.get("avg_cost", 0),
                current_price=pos.get("current_price", 0)
            )
            self.portfolio[ticker] = position
            total_value += position.market_value

        # Calculate allocations
        for ticker, position in self.portfolio.items():
            position.allocation = (position.market_value / total_value * 100) if total_value > 0 else 0

        self.add_reasoning_step(f"Set portfolio with {len(positions)} positions, total value: ${total_value:,.2f}")

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return sum(pos.market_value for pos in self.portfolio.values())

    async def calculate_portfolio_metrics(
        self,
        returns_data: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        self.add_reasoning_step("Calculating portfolio metrics")

        total_value = self._get_portfolio_value()
        positions = list(self.portfolio.values())

        # Basic metrics
        metrics = {
            "total_value": total_value,
            "position_count": len(positions),
            "positions": {},
            "allocation": {},
            "performance": {},
            "risk": {}
        }

        # Per-position metrics
        for pos in positions:
            metrics["positions"][pos.ticker] = {
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "cost_basis": pos.cost_basis,
                "gain_loss": pos.gain_loss,
                "gain_loss_pct": pos.gain_loss_pct,
                "allocation": pos.allocation
            }

            metrics["allocation"][pos.ticker] = pos.allocation

        # Calculate aggregate performance
        total_cost = sum(pos.cost_basis for pos in positions)
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0

        metrics["performance"] = {
            "total_cost_basis": total_cost,
            "total_market_value": total_value,
            "total_gain_loss": total_gain_loss,
            "total_gain_loss_pct": total_gain_loss_pct
        }

        # Risk metrics (simplified - would use real data in production)
        if returns_data:
            import numpy as np
            returns = np.array(returns_data)
            metrics["risk"]["volatility"] = float(np.std(returns) * math.sqrt(252))  # Annualized
            metrics["risk"]["mean_return"] = float(np.mean(returns) * 252)  # Annualized
            metrics["risk"]["sharpe_ratio"] = metrics["risk"]["mean_return"] / metrics["risk"]["volatility"] if metrics["risk"]["volatility"] > 0 else 0
            metrics["risk"]["var_95"] = float(np.percentile(returns, 5) * total_value)

        return metrics

    async def analyze_allocation(
        self,
        target_allocation: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Analyze current allocation vs target"""
        self.add_reasoning_step("Analyzing portfolio allocation")

        current_value = self._get_portfolio_value()
        analysis = {
            "current_value": current_value,
            "current_allocation": {},
            "target_allocation": target_allocation or {},
            "deviations": {},
            "rebalancing": {}
        }

        # Current allocation
        for ticker, pos in self.portfolio.items():
            analysis["current_allocation"][ticker] = {
                "percentage": pos.allocation,
                "value": pos.market_value,
                "target": target_allocation.get(ticker, 0) if target_allocation else 0
            }

            # Calculate deviation
            if target_allocation and ticker in target_allocation:
                target = target_allocation[ticker]
                deviation = pos.allocation - target
                analysis["deviations"][ticker] = deviation

                # Rebalancing needed
                target_value = current_value * (target / 100)
                rebalance_amount = target_value - pos.market_value
                analysis["rebalancing"][ticker] = {
                    "action": "buy" if rebalance_amount > 0 else "sell",
                    "amount": abs(rebalance_amount)
                }

        return analysis

    async def calculate_var(
        self,
        confidence: float = 0.95,
        time_horizon: int = 1
    ) -> Dict[str, Any]:
        """Calculate Value at Risk"""
        self.add_reasoning_step(f"Calculating VaR at {confidence*100}% confidence")

        portfolio_value = self._get_portfolio_value()

        # Historical VaR calculation (simplified)
        # In production, use actual return distributions
        daily_volatility = 0.02  # 2% daily volatility assumption
        z_score = 1.645 if confidence == 0.95 else 2.326  # 95% or 99%

        var_absolute = portfolio_value * daily_volatility * z_score * math.sqrt(time_horizon)
        var_percentage = daily_volatility * z_score * math.sqrt(time_horizon) * 100

        return {
            "portfolio_value": portfolio_value,
            "confidence": confidence,
            "time_horizon_days": time_horizon,
            "var_absolute": var_absolute,
            "var_percentage": var_percentage,
            "interpretation": f"With {confidence*100}% confidence, the portfolio will not lose more than ${var_absolute:,.2f} over {time_horizon} day(s)"
        }

    async def generate_rebalancing_report(
        self,
        target_allocation: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate rebalancing recommendations"""
        self.add_reasoning_step("Generating rebalancing report")

        # Update current prices
        for ticker, price in current_prices.items():
            if ticker in self.portfolio:
                self.portfolio[ticker].current_price = price

        # Recalculate metrics
        total_value = self._get_portfolio_value()
        analysis = await self.analyze_allocation(target_allocation)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_portfolio_value": total_value,
            "current_allocation": analysis["current_allocation"],
            "target_allocation": target_allocation,
            "trades_recommended": []
        }

        # Generate trade recommendations
        for ticker, rebalance_info in analysis["rebalancing"].items():
            if abs(rebalance_info["amount"]) > 100:  # Only suggest if > $100
                current_pos = self.portfolio[ticker]
                trade_value = rebalance_info["amount"]
                trade_shares = trade_value / current_pos.current_price

                report["trades_recommended"].append({
                    "ticker": ticker,
                    "action": rebalance_info["action"],
                    "shares": round(trade_shares, 2),
                    "estimated_value": round(trade_value, 2),
                    "current_price": current_pos.current_price
                })

        # Sort by trade size
        report["trades_recommended"].sort(key=lambda x: x["estimated_value"], reverse=True)

        return report

    async def analyze_sector_exposure(self) -> Dict[str, Any]:
        """Analyze sector exposure of portfolio"""
        self.add_reasoning_step("Analyzing sector exposure")

        # Sector mapping (simplified - would use actual sector data)
        sector_mapping = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "NVDA": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary",
            "JPM": "Financials",
            "BAC": "Financials",
            "XOM": "Energy",
            "CVX": "Energy",
        }

        sector_exposure = {}
        total_value = self._get_portfolio_value()

        for ticker, pos in self.portfolio.items():
            sector = sector_mapping.get(ticker, "Other")
            if sector not in sector_exposure:
                sector_exposure[sector] = {"value": 0, "allocation": 0}

            sector_exposure[sector]["value"] += pos.market_value
            sector_exposure[sector]["allocation"] = (sector_exposure[sector]["value"] / total_value * 100) if total_value > 0 else 0

        return {
            "total_value": total_value,
            "sector_exposure": sector_exposure,
            "diversification_score": self._calculate_diversification_score(sector_exposure)
        }

    def _calculate_diversification_score(self, sector_exposure: Dict) -> float:
        """Calculate portfolio diversification score (0-100)"""
        if not sector_exposure:
            return 0

        # Herfindahl index based score
        allocations = [s["allocation"] / 100 for s in sector_exposure.values()]
        hhi = sum(a ** 2 for a in allocations)

        # Convert to score (1 = fully diversified, 0 = concentrated)
        score = (1 - hhi) * 100
        return round(score, 2)

    async def process(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process portfolio analysis request"""
        self.update_state(AgentState.THINKING)
        start_time = datetime.now()

        input_lower = input_text.lower()
        result_content = ""

        if "metric" in input_lower or "analyze" in input_lower:
            metrics = await self.calculate_portfolio_metrics()
            result_content = f"## Portfolio Metrics\n\n"
            result_content += f"**Total Value:** ${metrics['total_value']:,.2f}\n\n"
            result_content += f"**Positions:** {metrics['position_count']}\n\n"

            if "performance" in metrics:
                perf = metrics["performance"]
                result_content += f"### Performance\n"
                result_content += f"- Total Gain/Loss: ${perf['total_gain_loss']:,.2f} ({perf['total_gain_loss_pct']:.2f}%)\n"

        elif "var" in input_lower or "risk" in input_lower:
            var_result = await self.calculate_var(confidence=0.95)
            result_content = f"## Value at Risk Analysis\n\n"
            result_content += f"**Portfolio Value:** ${var_result['portfolio_value']:,.2f}\n\n"
            result_content += f"**VaR ({var_result['confidence']*100}%):** ${var_result['var_absolute']:,.2f}\n\n"
            result_content += f"**{var_result['interpretation']}**\n"

        elif "sector" in input_lower or "exposure" in input_lower:
            sector_result = await self.analyze_sector_exposure()
            result_content = f"## Sector Exposure Analysis\n\n"
            result_content += f"**Diversification Score:** {sector_result['diversification_score']}/100\n\n"
            result_content += f"### By Sector\n"
            for sector, data in sector_result["sector_exposure"].items():
                result_content += f"- {sector}: {data['allocation']:.1f}% (${data['value']:,.2f})\n"

        elif "rebalance" in input_lower:
            # Get target allocation from context if available
            target = context.get("target_allocation", {}) if context else {}
            if target:
                report = await self.generate_rebalancing_report(target, {})
                result_content = f"## Rebalancing Report\n\n"
                result_content += f"**Total Value:** ${report['total_portfolio_value']:,.2f}\n\n"
                result_content += f"### Recommended Trades\n"
                for trade in report["trades_recommended"]:
                    result_content += f"- **{trade['action'].upper()}** {trade['ticker']}: {trade['shares']} shares (${trade['estimated_value']:,.2f})\n"
            else:
                result_content = "Please provide target allocation for rebalancing analysis."

        else:
            # General analysis
            metrics = await self.calculate_portfolio_metrics()
            sector = await self.analyze_sector_exposure()
            var_result = await self.calculate_var()

            result_content = f"## Portfolio Analysis Summary\n\n"
            result_content += f"**Total Value:** ${metrics['total_value']:,.2f}\n\n"
            result_content += f"**Performance:** ${metrics['performance']['total_gain_loss']:,.2f} ({metrics['performance']['total_gain_loss_pct']:.2f}%)\n\n"
            result_content += f"**Diversification:** {sector['diversification_score']}/100\n\n"
            result_content += f"**VaR (95%):** ${var_result['var_absolute']:,.2f}\n"

        execution_time = (datetime.now() - start_time).total_seconds()

        return AgentResponse(
            success=True,
            content=result_content,
            reasoning_steps=self.reasoning_history,
            execution_time=execution_time
        )


class TechnicalAnalysisAgent(AnalystAgent):
    """
    Technical Analysis Agent

    Extends AnalystAgent with technical analysis capabilities:
    - Moving averages
    - RSI, MACD
    - Support/resistance levels
    - Trend analysis
    """

    async def calculate_moving_averages(
        self,
        prices: List[float],
        periods: List[int] = None
    ) -> Dict[str, Any]:
        """Calculate moving averages"""
        if periods is None:
            periods = [20, 50, 200]

        import numpy as np

        result = {"prices": prices, "moving_averages": {}}

        for period in periods:
            if len(prices) >= period:
                ma = float(np.mean(prices[-period:]))
                result["moving_averages"][f"ma_{period}"] = ma

                # Signal
                if len(prices) >= period + 1:
                    prev_ma = float(np.mean(prices[-period-1:-1]))
                    result["moving_averages"][f"signal_{period}"] = "bullish" if ma > prev_ma else "bearish"

        return result

    async def calculate_rsi(
        self,
        prices: List[float],
        period: int = 14
    ) -> Dict[str, float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return {"error": "Not enough data"}

        import numpy as np

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        return {
            "rsi": float(rsi),
            "signal": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        }

    async def identify_support_resistance(
        self,
        prices: List[float],
        window: int = 5
    ) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        import numpy as np

        levels = {"support": [], "resistance": []}

        for i in range(window, len(prices) - window):
            is_support = True
            is_resistance = True

            # Check if local minimum/maximum
            for j in range(i - window, i + window + 1):
                if j != i:
                    if prices[j] < prices[i]:
                        is_support = False
                    if prices[j] > prices[i]:
                        is_resistance = False

            if is_support:
                levels["support"].append(float(prices[i]))
            if is_resistance:
                levels["resistance"].append(float(prices[i]))

        # Deduplicate and sort
        levels["support"] = sorted(set(levels["support"]))
        levels["resistance"] = sorted(set(levels["resistance"]), reverse=True)

        return levels
