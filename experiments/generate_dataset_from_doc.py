"""
Generate high-quality fine-tuning dataset from the Global Corporate Convergence paper.
Outputs Alpaca-format JSONL for Unsloth/SFTTrainer.

Run:  python experiments/generate_dataset_from_doc.py
"""
import json, random, re
from pathlib import Path

SEED = 42
EVAL_FRAC = 0.2
OUT_DIR = Path(__file__).parent
TRAIN = OUT_DIR / "finnav_train.jsonl"
EVAL  = OUT_DIR / "finnav_eval.jsonl"

# ── All examples hand-crafted from the document ──────────────────────────

examples: list[dict] = []

def add(instruction: str, inp: str, output: str):
    examples.append({"instruction": instruction, "input": inp, "output": output})

# ═══════════════════════════════════════════════════════════════
# CATEGORY 1: CORPORATE/FINANCIAL ANALYSIS (from paper sections)
# ═══════════════════════════════════════════════════════════════

add(
    "Summarize JPMorgan Chase's position in the 2026 SMB banking landscape.",
    "",
    "JPMorgan Chase bridges elite corporate career aspirations with foundational SMB financial architecture. The SMB banking sector represents a $130 billion growth opportunity. The average small business maintains daily cash balances of $13,900 — over 4x that of a typical consumer — and generates $242 billion in unsecured loans annually. JPMorgan is addressing SMB fragmentation (95% of small businesses use digital tools but across 6 different platforms) through integrations with platforms like Melio, enabling credit-card-to-ACH vendor payments. Their 'American Dream Initiative' targets 10 million small businesses, up from 7 million served historically."
)

add(
    "What is Answer Engine Optimization (AEO) and why are SMBs researching it in 2026?",
    "",
    "Answer Engine Optimization (AEO), also called Generative Engine Optimization (GEO), is a new discipline where businesses optimize to appear in AI chat responses (Gemini, Perplexity, ChatGPT) rather than traditional search results. SMBs are urgently researching AEO because Google's deployment of AI Overviews and agentic search capabilities is siphoning traditional organic and paid click-through rates. The focus is on citation analysis, measuring share of voice within LLM outputs, and sentiment analysis of how AI describes brands. Success is no longer about 'being found' on a page — it's about 'being the answer' in a conversational interface."
)

add(
    "Analyze NVIDIA's compensation structure for AI engineers in 2026.",
    "",
    "NVIDIA AI Engineer compensation has reached historic highs driven by a massive labor supply deficit in specialized ML talent. Average annual total compensation is $322,000, with the top 10% earning over $526,000. The highest reported salaries approach $873,000. When equity and performance bonuses are included, senior ML engineers specializing in LLMs and RAG command total packages exceeding $400,000. LLM development and generative architecture command a significantly higher premium than traditional computer vision or predictive analytics. This deficit is the second-largest hurdle to global AI adoption."
)

add(
    "Compare Google Workspace and Microsoft 365 pricing for SMBs in 2026.",
    "",
    "In mid-2026, premium tiers for both suites have reached exact price parity at $22.00/user/month on annual plans. Since cost is no longer a differentiator, SMB procurement has pivoted to ecosystem alignment. Google Workspace wins on user experience, real-time collaboration speed, and superior third-party integrations — particularly for organizations not relying on advanced local desktop clients. Microsoft 365's advantage centers on Copilot AI integration, with data showing a 4.2x ROI within the first 12 months of deployment. The choice ultimately depends on whether the SMB prioritizes native collaboration or deep AI-embedded productivity."
)

add(
    "Explain the 'HubSpot Trap' that SMBs are encountering in 2026.",
    "",
    "The 'HubSpot Trap' refers to the massive pricing jump SMBs encounter when transitioning from cheap starter plans to necessary professional tiers. Starter plans are affordable, but upgrading to Professional involves an $890/month fee plus a $3,000 mandatory onboarding charge. Many SMBs begin on the free CRM tier, build dependencies, then face sticker shock when they need features like advanced automation, A/B testing, or custom reporting. This creates a lock-in effect where migrating away is costly due to data and workflow dependencies, but staying requires a significant budget increase."
)

add(
    "What are the key findings about AI infrastructure investment in 2026?",
    "",
    "AI has transitioned from experimental pilots to aggressive commercial deployment. Approximately $2.9 trillion in global data center construction and AI infrastructure investment is projected by 2028. Key findings: 76% of small businesses that increased tech spending report measurable revenue growth; 86% of organizations report AI budgets will continue increasing in 2026; nearly 40% expect increases exceeding 10%. Primary goals are operational efficiency (cited by 34% of leaders) and improving employee productivity. The market is aggressively distinguishing between AI disruptors and companies vulnerable to disruption."
)

add(
    "Describe Amazon's proprietary chip strategy and its impact on SMBs.",
    "Amazon is projecting $200 billion in capital spending for 2026, primarily for AI infrastructure.",
    "Amazon is challenging NVIDIA's hardware monopoly through proprietary silicon. Their Graviton processors are used by 98% of AWS's large clients, while Trainium chips deliver a 30% cost-performance improvement over standard GPU-based training. For SMBs struggling with exorbitant AI deployment costs, Amazon's chips represent a critical pathway to affordable machine learning. This is significant because 58% of small organizations consider open-source solutions 'very to extremely important' for bypassing commercial monopolies, and Amazon's affordable compute enables them to fine-tune models with proprietary data at a fraction of the cost."
)

add(
    "Analyze the structural shift in the Salesforce job market in 2026.",
    "",
    "The Salesforce labor market is undergoing dramatic structural change. Basic Administrator certification, once a guaranteed path to high-paying employment, has been disrupted. Global generalist Salesforce talent supply grew 27% year-over-year, heavily outpacing demand. AI has automated routine coding, causing a 12% drop in demand for standard developers. However, Salesforce Architect roles have surged in demand by up to 27%. The market now rewards strategic governance skills — designing, securing, and governing automated systems — while penalizing those who only possess operational execution skills."
)

add(
    "What are ServiceNow's hidden implementation costs that concern mid-market companies?",
    "",
    "Mid-market financial leaders are discovering that ServiceNow consulting and implementation fees often eclipse the base software license costs by a factor of 3-5x. Certified senior architects in major cities command rates of $225-$300+/hour. The total cost of ownership significantly exceeds initial estimates because organizations underestimate the complexity of workflow customization, integration with existing systems, and ongoing maintenance. This creates 'implementation shock' where the procurement decision based on license costs alone severely underestimates the true multi-year financial commitment."
)

add(
    "Explain the economic value of remote work as a financial instrument in 2026.",
    "",
    "Remote work has permanently stabilized at 23.7-26% of all paid US workdays, confirmed by building access records and cellphone telemetry. RTO mandates have failed to reverse this trend. Employees mathematically value hybrid/remote work as equivalent to an 8% base salary increase. 64% of fully remote workers would immediately leave if required to return to office full-time, and 37% would accept a 10% pay cut to maintain remote status. This makes geographical flexibility a structural financial instrument in talent acquisition and retention, not merely a perk."
)

# ═══════════════════════════════════════════════════════════════
# CATEGORY 2: FINANCIAL Q&A (domain knowledge)
# ═══════════════════════════════════════════════════════════════

add(
    "What is Value at Risk (VaR) and how is it calculated for a portfolio?",
    "",
    "Value at Risk (VaR) quantifies the maximum expected loss over a specified time horizon at a given confidence level. For example, a 1-day 95% VaR of $1M means there's a 95% probability the portfolio won't lose more than $1M in a single day. Common calculation methods: (1) Historical simulation — uses actual past returns distribution; (2) Variance-covariance (parametric) — assumes normal distribution, uses portfolio standard deviation; (3) Monte Carlo simulation — generates thousands of random scenarios. Limitations include assuming normal market conditions and ignoring tail risk beyond the confidence level."
)

add(
    "Explain the Sharpe Ratio and what constitutes a good value.",
    "",
    "The Sharpe Ratio measures risk-adjusted return: (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation. It quantifies excess return per unit of total risk. Interpretation: below 1.0 is suboptimal; 1.0-1.99 is acceptable to good; 2.0-2.99 is very good; 3.0+ is excellent. A higher ratio indicates better risk-adjusted performance. Limitations: it assumes normally distributed returns, penalizes upside volatility equally with downside, and uses historical data which may not predict future performance. The Sortino Ratio addresses the downside-only volatility limitation."
)

add(
    "What is sector exposure analysis and why does it matter for portfolio management?",
    "",
    "Sector exposure analysis measures the percentage allocation of a portfolio across economic sectors (Technology, Healthcare, Financials, Energy, etc.). It matters because: (1) Over-concentration in one sector creates systematic risk — if that sector declines, the entire portfolio suffers; (2) Different sectors have different correlations with economic cycles; (3) Regulatory changes can impact entire sectors simultaneously. Best practice is to benchmark sector weights against an index (e.g., S&P 500) and identify overweight/underweight positions. A portfolio with 60%+ in Technology would be flagged as dangerously concentrated."
)

add(
    "How does portfolio rebalancing work and when should it be triggered?",
    "",
    "Portfolio rebalancing restores target asset allocations after market movements cause drift. Methods: (1) Calendar-based — rebalance at fixed intervals (quarterly, annually); (2) Threshold-based — trigger when any allocation drifts beyond a band (e.g., ±5%); (3) Hybrid — check at intervals but only act if thresholds are breached. Considerations: transaction costs, tax implications of selling appreciated assets, and market conditions. A common approach is the 5/25 rule: rebalance when any asset class drifts 5 percentage points from target or 25% of its target weight (e.g., a 20% target triggers at 15% or 25%)."
)

add(
    "What are the key risk factors typically disclosed in a 10-K filing?",
    "",
    "SEC 10-K filings require disclosure of material risk factors in Item 1A. Common categories include: (1) Market/competitive risks — disruption, pricing pressure, market share loss; (2) Operational risks — supply chain, key personnel, cybersecurity; (3) Financial risks — currency exposure, interest rates, credit/liquidity; (4) Regulatory/legal risks — compliance, litigation, regulatory changes; (5) Technology risks — obsolescence, IP protection, data breaches; (6) Macroeconomic risks — recession, inflation, geopolitical events. Companies must update these annually and disclose material changes in 10-Q quarterly filings."
)

add(
    "Explain the difference between 10-K and 10-Q SEC filings.",
    "",
    "The 10-K is an annual comprehensive report filed within 60 days of fiscal year-end (for large accelerated filers). It includes audited financial statements, full MD&A discussion, risk factors, business overview, and executive compensation. The 10-Q is filed quarterly (within 40 days) for Q1-Q3 and contains unaudited financial statements and condensed MD&A. Key differences: 10-K requires independent auditor review; 10-Q does not. 10-K includes complete Item 1A risk factors; 10-Q only updates material changes. 10-K provides full business description; 10-Q focuses on interim financial performance."
)

# ═══════════════════════════════════════════════════════════════
# CATEGORY 3: TASK CLASSIFICATION (for supervisor agent)
# ═══════════════════════════════════════════════════════════════

add(
    "Classify the following user query into a task type: RESEARCH, ANALYSIS, EXECUTION, COMPLEX, or QUERY.",
    "What are the risk factors in NVIDIA's latest 10-K filing?",
    "RESEARCH  This query requests SEC filing information (10-K risk factors) for a specific company (NVIDIA). It should be routed to the Research Agent which specializes in SEC EDGAR search and section extraction."
)

add(
    "Classify the following user query into a task type: RESEARCH, ANALYSIS, EXECUTION, COMPLEX, or QUERY.",
    "Calculate the VaR and Sharpe ratio for my portfolio of AAPL, MSFT, and GOOGL.",
    "ANALYSIS — This query requests quantitative portfolio metrics (VaR, Sharpe ratio) for specific holdings. It should be routed to the Analyst Agent for portfolio risk calculations."
)

add(
    "Classify the following user query into a task type: RESEARCH, ANALYSIS, EXECUTION, COMPLEX, or QUERY.",
    "Compare AMD and NVDA risk profiles from their 10-K filings and recommend portfolio allocation.",
    "COMPLEX  This query requires both SEC research (comparing risk profiles from filings) AND portfolio analysis (allocation recommendation). It should be decomposed into subtasks: (1) Research Agent extracts risk factors for AMD and NVDA, (2) Analyst Agent provides allocation recommendations based on the risk comparison."
)

add(
    "Classify the following user query into a task type: RESEARCH, ANALYSIS, EXECUTION, COMPLEX, or QUERY.",
    "Send me an alert if TSLA drops below $200.",
    "EXECUTION — This query requests an automated action (price alert). It should be routed to the execution pipeline for alert configuration and monitoring."
)

add(
    "Classify the following user query into a task type: RESEARCH, ANALYSIS, EXECUTION, COMPLEX, or QUERY.",
    "What is the current P/E ratio and what does it indicate?",
    "QUERY — This is a general financial knowledge question that doesn't require SEC research, portfolio analysis, or execution. The Financial Agent can answer this directly using its domain knowledge."
)

add(
    "Classify the following user query into a task type: RESEARCH, ANALYSIS, EXECUTION, COMPLEX, or QUERY.",
    "Research AMZN's MD&A section and then analyze how their $200B capex impacts sector exposure in my tech-heavy portfolio.",
    "COMPLEX — This requires multi-agent coordination: (1) Research Agent extracts MD&A from Amazon's filing, (2) Analyst Agent evaluates sector concentration risk given Amazon's massive infrastructure spending and its implications for a tech-heavy portfolio."
)

# ═══════════════════════════════════════════════════════════════
# CATEGORY 4: SEC FILING SUMMARIZATION
# ═══════════════════════════════════════════════════════════════

add(
    "Summarize the following excerpt from a company's 10-K risk factors section.",
    "The company faces intense competition in the GPU and AI accelerator market. Competitors including AMD, Intel, and custom silicon providers such as Amazon (Trainium), Google (TPU), and Microsoft are developing proprietary chips that could reduce demand for the company's products. The rapid pace of technological change requires significant R&D investment, and there is no assurance that the company's products will maintain their current market position. Additionally, export restrictions on advanced semiconductors to certain countries may materially impact international revenue.",
    "Key risk factors identified: (1) **Competitive pressure** — Multiple competitors (AMD, Intel) plus cloud providers developing custom silicon (Amazon Trainium, Google TPU) threaten GPU market dominance. (2) **Technology obsolescence** — Rapid innovation cycles demand heavy R&D spending with uncertain returns. (3) **Regulatory/export risk** — Semiconductor export restrictions to certain countries could materially reduce international revenue. Overall assessment: The company faces convergent risks from both traditional competitors and vertically-integrated cloud customers building alternatives to reduce GPU dependency."
)

add(
    "Extract and categorize the key risks from this MD&A excerpt.",
    "Management expects capital expenditures to reach $200 billion in 2026, primarily directed at AI infrastructure including data centers and proprietary chip fabrication. Third-party sellers now account for 62% of all paid units, with 82% utilizing Fulfillment by Amazon. The pending FTC antitrust trial scheduled for October 2026 represents a material uncertainty regarding potential forced divestiture of AWS or marketplace restructuring.",
    "**Capital Risk**: $200B planned capex for AI infrastructure — massive commitment with execution risk if AI demand softens. **Concentration Risk**: Heavy dependency on third-party marketplace (62% of units, 82% using FBA) creates platform liability exposure. **Regulatory Risk**: FTC v. Amazon antitrust trial (Oct 2026) could force AWS divestiture or marketplace restructuring, potentially fragmenting the company's integrated business model. **Severity Assessment**: The antitrust trial represents the highest-impact risk given potential structural changes to the company's core operations."
)

add(
    "Analyze this excerpt about AI market dynamics from a corporate filing perspective.",
    "Approximately $2.9 trillion in global data center construction and AI infrastructure investment is projected by 2028. 86% of organizations report AI budgets will increase in 2026. The market is distinguishing between AI disruptors and companies vulnerable to disruption, with the most severe equity pullbacks occurring in service sectors reliant on knowledge-based fields.",
    "This excerpt reveals a structural bifurcation in equity markets. The $2.9T infrastructure buildout creates massive capex commitments for hyperscalers while opening opportunities for semiconductor and infrastructure providers. The 86% budget increase signals sustained demand but also raises concerns about potential overinvestment. The key insight for investors: service-sector companies in knowledge-based fields face the highest disruption risk, while infrastructure providers (cloud, chips, data centers) are positioned as beneficiaries. Portfolio implications: overweight infrastructure enablers, underweight knowledge-services firms vulnerable to AI substitution."
)

# ═══════════════════════════════════════════════════════════════
# CATEGORY 5: PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════════

add(
    "Analyze the sector exposure of a portfolio containing NVDA (30%), AAPL (25%), MSFT (20%), AMZN (15%), JPM (10%).",
    "",
    "**Sector Breakdown**: Technology: 75% (NVDA 30% + AAPL 25% + MSFT 20%), Consumer Discretionary: 15% (AMZN), Financials: 10% (JPM).\n\n**Risk Assessment**: This portfolio is dangerously over-concentrated in Technology at 75%, far exceeding the S&P 500's ~30% tech weighting. Key risks: (1) Sector-specific regulatory action (antitrust, AI regulation) could simultaneously impact 75% of holdings; (2) Rising interest rates disproportionately affect high-growth tech valuations; (3) Correlation risk — tech stocks tend to move together during selloffs.\n\n**Recommendation**: Reduce Technology to 40-50% by trimming NVDA and AAPL. Add Healthcare (10-15%), Energy (5-10%), and increase Financials to 15-20% for better diversification."
)

add(
    "A client's portfolio has drifted from targets. Original: 60% equities, 30% bonds, 10% alternatives. Current: 72% equities, 22% bonds, 6% alternatives. Should we rebalance?",
    "",
    "**Drift Analysis**: Equities drifted +12 percentage points (20% relative drift from 60% target). Bonds drifted -8pp. Alternatives drifted -4pp.\n\n**Rebalancing Trigger**: Yes — using the 5/25 rule, equities have drifted 12pp (exceeds 5pp absolute threshold) and 20% relative to target (close to 25% threshold). This warrants immediate rebalancing.\n\n**Recommended Actions**: (1) Sell equities to bring from 72% to 60% (~$12 of every $100 in equities); (2) Buy bonds to restore 30% allocation; (3) Buy alternatives to restore 10%. Consider tax implications of equity sales — prioritize selling positions with losses for tax-loss harvesting, or use new cash inflows to rebalance gradually."
)

add(
    "Explain why a tech-heavy portfolio might be risky given the current 2026 AI market dynamics.",
    "",
    "While the AI buildout favors tech companies, concentration risk is elevated for several reasons: (1) **Valuation risk** — $2.9T in projected AI infrastructure spending has already been priced into many tech valuations; any demand slowdown could trigger sharp corrections. (2) **Regulatory risk** — FTC antitrust actions (e.g., FTC v. Amazon Oct 2026) and potential AI regulation create sector-wide uncertainty. (3) **Competition convergence** — Cloud providers are building custom chips (Amazon Trainium, Google TPU) to reduce dependency on NVIDIA, threatening the dominant hardware player's margins. (4) **Market bifurcation** — The market is already distinguishing disruptors from disrupted; not all tech companies will benefit equally. Recommended hedge: diversify into infrastructure beneficiaries (REITs for data centers, utilities for power) rather than pure software plays."
)

# ═══════════════════════════════════════════════════════════════
# CATEGORY 6: COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════

add(
    "Compare NVIDIA and Amazon's approaches to AI infrastructure and their investment implications.",
    "",
    "**NVIDIA**: Hardware-first approach — designs and sells GPUs/accelerators that power AI training and inference globally. Revenue driven by chip sales to cloud providers and enterprises. Risk: customers (Amazon, Google, Microsoft) are developing custom chips to reduce GPU dependency. Strength: dominant market position, extreme talent premium ($322K avg AI engineer comp).\n\n**Amazon**: Vertically-integrated approach — builds custom silicon (Graviton for compute, Trainium for AI training) within its AWS ecosystem. Projected $200B capex in 2026. Trainium offers 30% cost-performance improvement over standard GPUs. Risk: massive capex commitment, antitrust trial. Strength: 98% of large AWS clients use Graviton; controls the customer relationship.\n\n**Investment Implication**: These companies represent competing visions — horizontal platform (NVIDIA sells to everyone) vs. vertical integration (Amazon builds its own). A balanced portfolio should hold both as a hedge, with NVIDIA offering higher upside but greater competitive risk, and Amazon offering more defensible but capital-intensive growth."
)

add(
    "Compare the career growth trajectories at Microsoft vs. Alphabet for a young cloud engineer in 2026.",
    "",
    "**Alphabet**: Entry-level cloud engineers start at ~$101,337 base, scaling past $125K at mid-level, with Cloud Architects averaging $135K+. Ranked most innovative company of 2026 and #2 for career growth. Strength: exposure to cutting-edge TPU chips, Gemini AI, and the AEO/search transformation. Culture emphasizes engineering excellence.\n\n**Microsoft**: Ranks #1 for career growth globally. Distinguishes itself by training employees to BUILD custom AI tools, not just use them. Promotes skills-based advancement over traditional tenure. Copilot ecosystem shows 4.2x ROI for enterprises. Multi-cloud certifications are a focus area.\n\n**Recommendation**: For pure technical depth in AI/ML infrastructure, Alphabet offers stronger exposure to frontier research. For career breadth and faster advancement through skills-based promotion, Microsoft's structure provides more upward mobility. Both offer strong compensation; the choice depends on whether the candidate prioritizes deep specialization or versatile career progression."
)

# ═══════════════════════════════════════════════════════════════
# CATEGORY 7: ADDITIONAL FINANCIAL DOMAIN Q&A
# ═══════════════════════════════════════════════════════════════

add(
    "What is the Management Discussion and Analysis (MD&A) section of a 10-K?",
    "",
    "Item 7 of the 10-K, the MD&A section, is management's narrative explanation of the company's financial condition and results of operations. It covers: (1) Revenue drivers and trends; (2) Cost structure and margin analysis; (3) Liquidity and capital resources; (4) Known trends, demands, and uncertainties; (5) Critical accounting estimates. Unlike financial statements (which are backward-looking numbers), MD&A provides management's forward-looking perspective. Analysts value it because it reveals management's interpretation of results and expectations. Red flags include vague language about 'challenges' without quantification, or sudden changes in tone from prior filings."
)

add(
    "How should an investor interpret a company's capital expenditure trends?",
    "",
    "Capex trends reveal strategic priorities and growth trajectory. Rising capex in core operations signals investment in future growth (e.g., Amazon's $200B AI infrastructure spend). Key metrics: (1) Capex-to-revenue ratio — shows reinvestment intensity; (2) Maintenance vs. growth capex — maintenance preserves existing operations, growth capex funds expansion; (3) Capex vs. depreciation — if capex consistently exceeds depreciation, the company is expanding its asset base. Warning signs: declining capex may indicate mature/declining business, while excessive capex without corresponding revenue growth suggests poor capital allocation. Always compare capex trends against industry peers and revenue growth rates."
)

add(
    "What is a market capitalization and how does it affect portfolio weighting?",
    "",
    "Market capitalization = share price × total outstanding shares. It classifies companies: Mega-cap ($200B+), Large-cap ($10-200B), Mid-cap ($2-10B), Small-cap ($300M-2B), Micro-cap (under $300M). For portfolio weighting: (1) Market-cap weighting mirrors index composition (S&P 500 is cap-weighted, so larger companies get more weight); (2) Equal weighting gives each position identical allocation regardless of size; (3) Risk-based weighting adjusts for volatility. Cap-weighted portfolios naturally overweight mega-caps (AAPL, MSFT, NVDA), which can create concentration risk — the top 10 S&P 500 stocks often represent 30%+ of the index."
)

add(
    "Explain the concept of total cost of ownership (TCO) in enterprise software procurement.",
    "",
    "TCO captures all direct and indirect costs of owning software beyond the license fee. Components: (1) License/subscription fees — the visible sticker price; (2) Implementation/consulting — often 3-5x the license cost (as seen with ServiceNow where architects charge $225-300+/hour); (3) Customization and integration — connecting to existing systems; (4) Training and change management — onboarding staff; (5) Ongoing maintenance and upgrades; (6) Opportunity cost of staff time. The 'HubSpot Trap' illustrates TCO risk: starter plans are cheap but Professional tiers jump to $890/month plus $3,000 onboarding. SMBs should calculate 3-year TCO, not just Year 1 costs, before procurement decisions."
)

add(
    "What does it mean when 76% of small businesses that increased tech spending report revenue growth?",
    "According to 2026 market data, 76% of small businesses that increased technology spending report corresponding measurable revenue growth.",
    "This statistic suggests a strong positive correlation between technology investment and revenue outcomes for SMBs, but requires careful interpretation: (1) **Correlation ≠ Causation** — businesses that can afford to increase tech spending may already be growing, creating a selection bias; (2) **Survivorship bias** — struggling businesses that increased tech spending and still failed aren't captured; (3) **Directional signal** — despite caveats, the magnitude (76%) strongly supports that technology adoption is a significant growth enabler. For investors, this suggests companies providing SMB tech solutions (Salesforce, HubSpot, ServiceNow) have a large addressable market with demonstrated willingness to pay. For SMBs, it validates prioritizing tech investment over cost-cutting in the 2026 environment."
)

# ── Write output ──────────────────────────────────────────────

def main():
    random.seed(SEED)
    data = list(examples)
    random.shuffle(data)
    n_eval = max(1, int(len(data) * EVAL_FRAC))
    eval_set, train_set = data[:n_eval], data[n_eval:]

    for path, subset in [(TRAIN, train_set), (EVAL, eval_set)]:
        with path.open("w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset)} examples -> {path}")

    print(f"\nTotal: {len(data)} examples ({len(train_set)} train / {len(eval_set)} eval)")
    print("\nSample:")
    s = data[0]
    print(f"  Instruction: {s['instruction'][:80]}...")
    print(f"  Output: {s['output'][:120]}...")

if __name__ == "__main__":
    main()
