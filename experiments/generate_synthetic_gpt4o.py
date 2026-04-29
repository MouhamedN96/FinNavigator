"""
Generate synthetic fine-tuning data using GPT-4o as a teacher model.
Reads the Global Corporate Convergence paper + existing seed examples,
then asks GPT-4o to generate diverse new training pairs.

Usage:
    set OPENAI_API_KEY=sk-...
    python experiments/generate_synthetic_gpt4o.py

Merges output with existing finnav_train/eval.jsonl files.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

# Auto-load .env file if present
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────
SEED = 42
EVAL_FRAC = 0.2
OUT_DIR = Path(__file__).parent
TRAIN_PATH = OUT_DIR / "finnav_train.jsonl"
EVAL_PATH = OUT_DIR / "finnav_eval.jsonl"
DOC_PATH = Path(__file__).resolve().parent.parent / "Global Corporate Convergence_ An Analysis of Resea....docx"

# How many examples to generate per category per batch
EXAMPLES_PER_CATEGORY = 15
MODEL = "gpt-4o"

CATEGORIES = [
    {
        "name": "SEC Filing Summarization",
        "system": "You are an expert financial analyst who summarizes SEC filings.",
        "prompt": """Generate {n} diverse instruction/input/output training examples for fine-tuning a financial AI assistant.

Category: SEC Filing Summarization & Risk Factor Analysis

Each example should have:
- "instruction": A realistic user query about SEC filings (10-K, 10-Q risk factors, MD&A sections)
- "input": A realistic excerpt from a corporate filing (2-4 sentences of plausible filing language)
- "output": An expert-quality analytical summary (150-250 words) that categorizes risks, identifies key themes, and provides severity assessments

Cover these companies: NVDA, AAPL, MSFT, AMZN, GOOGL, AMD, INTC, TSLA, META, JPM, GS, BAC
Vary the filing types: 10-K annual, 10-Q quarterly, 8-K current reports
Vary the sections: Item 1A Risk Factors, Item 7 MD&A, Item 1 Business, Item 8 Financial Statements

Here is context from a research paper about these companies in 2026:
{context}

Return ONLY a JSON array of objects with keys: instruction, input, output""",
    },
    {
        "name": "Financial Q&A",
        "system": "You are a financial education expert teaching portfolio management concepts.",
        "prompt": """Generate {n} diverse instruction/input/output training examples for fine-tuning a financial AI assistant.

Category: Financial Domain Q&A

Each example should have:
- "instruction": A question about financial concepts, metrics, or analysis techniques
- "input": "" (empty string - these are knowledge questions)
- "output": A clear, expert-level explanation (100-200 words) with practical examples

Topics to cover:
- Portfolio metrics: Beta, Alpha, Treynor ratio, Information ratio, Maximum drawdown
- Risk concepts: Systematic vs unsystematic risk, correlation, diversification benefit
- Valuation: DCF, P/E, PEG ratio, EV/EBITDA, price-to-book
- Fixed income: Duration, yield curve, credit spreads, bond ratings
- Market structure: Market makers, dark pools, ETFs vs mutual funds, options Greeks
- Corporate actions: Stock splits, dividends, buybacks, M&A impact on shareholders

Return ONLY a JSON array of objects with keys: instruction, input, output""",
    },
    {
        "name": "Task Classification",
        "system": "You are an AI routing expert who classifies financial queries.",
        "prompt": """Generate {n} diverse instruction/input/output training examples for fine-tuning a financial AI assistant.

Category: Task Classification for Multi-Agent Routing

Each example should have:
- "instruction": "Classify the following user query into a task type: RESEARCH, ANALYSIS, EXECUTION, COMPLEX, or QUERY."
- "input": A realistic user query to a financial AI assistant
- "output": The classification (one of RESEARCH, ANALYSIS, EXECUTION, COMPLEX, QUERY) followed by a 1-2 sentence explanation of WHY and which agent should handle it

Task types:
- RESEARCH: SEC filing lookups, company background research, filing comparisons
- ANALYSIS: Portfolio calculations, risk metrics, sector exposure, rebalancing
- EXECUTION: Alerts, notifications, buy/sell orders, saving/indexing data
- COMPLEX: Multi-step tasks requiring both research AND analysis across multiple companies
- QUERY: General financial knowledge questions, definitions, explanations

Make queries realistic and varied. Include edge cases where classification is ambiguous.

Return ONLY a JSON array of objects with keys: instruction, input, output""",
    },
    {
        "name": "Portfolio Analysis",
        "system": "You are a portfolio manager providing actionable investment analysis.",
        "prompt": """Generate {n} diverse instruction/input/output training examples for fine-tuning a financial AI assistant.

Category: Portfolio Analysis & Risk Assessment

Each example should have:
- "instruction": A realistic portfolio analysis request
- "input": Portfolio details (positions, weights, sectors) or "" if embedded in instruction
- "output": Expert analysis (150-250 words) with specific metrics, risk assessment, and actionable recommendations

Cover these scenarios:
- Sector concentration analysis with rebalancing recommendations
- VaR and risk metric interpretation
- Dividend portfolio construction
- Growth vs value portfolio comparison
- International diversification assessment
- Tax-loss harvesting opportunities
- Retirement portfolio allocation by age
- Benchmark comparison (vs S&P 500, Russell 2000)

Use realistic tickers and weights. Include specific numbers and percentages.

Here is context about market dynamics in 2026:
{context}

Return ONLY a JSON array of objects with keys: instruction, input, output""",
    },
    {
        "name": "Comparative Analysis",
        "system": "You are a senior equity research analyst writing comparative reports.",
        "prompt": """Generate {n} diverse instruction/input/output training examples for fine-tuning a financial AI assistant.

Category: Comparative Corporate Analysis

Each example should have:
- "instruction": A request to compare two or more companies on specific dimensions
- "input": "" or relevant financial context
- "output": A structured comparison (200-300 words) with clear categories, data points, and an investment conclusion

Compare across these dimensions:
- Revenue growth trajectories, margin profiles
- AI strategy and infrastructure investment
- Competitive moats and market positioning
- Risk profiles from SEC filings
- Workforce and talent strategy
- Capital allocation (buybacks vs R&D vs acquisitions)

Use these company pairs: NVDA vs AMD, AAPL vs MSFT, AMZN vs GOOGL, META vs NFLX, JPM vs GS, CRM vs NOW

Here is context from a 2026 research paper:
{context}

Return ONLY a JSON array of objects with keys: instruction, input, output""",
    },
    {
        "name": "Market Analysis",
        "system": "You are a macroeconomic strategist analyzing market trends.",
        "prompt": """Generate {n} diverse instruction/input/output training examples for fine-tuning a financial AI assistant.

Category: Market & Macro Analysis

Each example should have:
- "instruction": A question about market trends, sector dynamics, or economic indicators
- "input": Relevant market data or context (or "" for general questions)
- "output": An analytical response (150-250 words) connecting macro trends to investment implications

Topics:
- AI infrastructure buildout ($2.9T by 2028) and sector winners/losers
- Interest rate impacts on growth vs value stocks
- Remote work as structural economic shift
- Enterprise software TCO and procurement trends
- Semiconductor supply chain dynamics
- SMB technology adoption patterns
- Labor market shifts: skills-based vs tenure-based advancement
- Open-source AI vs commercial model economics

Here is 2026 market context:
{context}

Return ONLY a JSON array of objects with keys: instruction, input, output""",
    },
]


def load_document() -> str:
    """Extract text from the .docx research paper."""
    try:
        import docx
        doc = docx.Document(str(DOC_PATH))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[warn] Could not load .docx: {e}")
        return ""


def load_existing_examples() -> list[dict]:
    """Load any existing JSONL examples to avoid duplicates."""
    existing = []
    for path in [TRAIN_PATH, EVAL_PATH]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.append(json.loads(line))
    return existing


def generate_batch(client: OpenAI, category: dict, context: str) -> list[dict]:
    """Call GPT-4o to generate a batch of examples for one category."""
    prompt = category["prompt"].format(
        n=EXAMPLES_PER_CATEGORY,
        context=context[:4000],  # trim to fit context window
    )

    print(f"  Generating {EXAMPLES_PER_CATEGORY} '{category['name']}' examples...")

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": category["system"]},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        if raw.startswith("json"):
            raw = raw[4:]

        items = json.loads(raw.strip())
        if isinstance(items, list):
            # Validate structure
            valid = []
            for item in items:
                if all(k in item for k in ("instruction", "output")):
                    item.setdefault("input", "")
                    item["input"] = str(item["input"]) if item["input"] else ""
                    item["instruction"] = str(item["instruction"])
                    item["output"] = str(item["output"])
                    valid.append(item)
            print(f"    -> Got {len(valid)} valid examples")
            return valid
        return []
    except Exception as e:
        print(f"    -> Error: {e}")
        return []


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        print("  PowerShell:  $env:OPENAI_API_KEY = 'sk-...'")
        print("  CMD:         set OPENAI_API_KEY=sk-...")
        return

    client = OpenAI(api_key=api_key)
    random.seed(SEED)

    # Load document context
    print("Loading research paper...")
    doc_text = load_document()
    if not doc_text:
        print("WARNING: Could not load document. Generating without paper context.")

    # Load existing seed examples
    existing = load_existing_examples()
    print(f"Found {len(existing)} existing seed examples")

    # Generate new examples via GPT-4o
    new_examples: list[dict] = []
    for cat in CATEGORIES:
        batch = generate_batch(client, cat, doc_text)
        new_examples.extend(batch)
        time.sleep(1)  # rate limit politeness

    print(f"\nGenerated {len(new_examples)} new examples from GPT-4o")

    # Merge with existing
    all_examples = existing + new_examples
    print(f"Total after merge: {len(all_examples)}")

    # Deduplicate by instruction text
    seen = set()
    deduped = []
    for ex in all_examples:
        key = str(ex["instruction"])[:100] + "|" + str(ex.get("input", ""))[:50]
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    print(f"After dedup: {len(deduped)}")

    # Split train/eval
    random.shuffle(deduped)
    n_eval = max(1, int(len(deduped) * EVAL_FRAC))
    eval_set = deduped[:n_eval]
    train_set = deduped[n_eval:]

    # Write
    for path, subset in [(TRAIN_PATH, train_set), (EVAL_PATH, eval_set)]:
        with path.open("w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train_set)} train -> {TRAIN_PATH}")
    print(f"Wrote {len(eval_set)} eval  -> {EVAL_PATH}")
    print(f"Total: {len(deduped)} examples")

    # Stats
    cats = {}
    for ex in deduped:
        inst = ex["instruction"]
        if "Classify" in inst:
            c = "Task Classification"
        elif "portfolio" in inst.lower() or "sector exposure" in inst.lower():
            c = "Portfolio Analysis"
        elif "Compare" in inst or "vs" in inst.lower():
            c = "Comparative Analysis"
        elif "10-K" in inst or "10-Q" in inst or "SEC" in inst or "filing" in inst.lower():
            c = "SEC Summarization"
        elif "market" in inst.lower() or "2026" in inst:
            c = "Market Analysis"
        else:
            c = "Financial Q&A"
        cats[c] = cats.get(c, 0) + 1
    print("\nCategory distribution:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
