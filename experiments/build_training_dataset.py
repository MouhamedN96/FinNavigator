"""
Build a small instruction-tuning dataset from SEC filings + Alpha Vantage.
==========================================================================

Output: experiments/finnav_train.jsonl   (and finnav_eval.jsonl, 20% holdout)

Format (Alpaca-style — Unsloth's default training notebook accepts this):
    {"instruction": "...", "input": "...", "output": "..."}

Quick experiment scope: ~10-30 companies x several Q&A per company = 100-500 pairs.
keep volume small for a fast LoRA run.

Run:
    pip install sec-api requests
    export SEC_API_KEY=...
    export ALPHA_VANTAGE_KEY=...
    python experiments/build_training_dataset.py
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Iterable

OUT_DIR = Path(__file__).parent
TRAIN_PATH = OUT_DIR / "finnav_train.jsonl"
EVAL_PATH = OUT_DIR / "finnav_eval.jsonl"

# Keep this small for a quick experiment. Add tickers later if eval looks promising.
TICKERS = ["NVDA", "AAPL", "MSFT", "AMD", "INTC", "GOOGL", "AMZN", "TSLA", "META", "NFLX"]
SECTIONS = [
    ("1A", "risk factors"),       # critical-importance section
    ("7",  "MD&A"),               # management discussion
]
CHUNK_CHARS = 1500   # ~400 tokens at 4 char/token, fits comfortably in context
EVAL_FRACTION = 0.2
SEED = 42


def fetch_sec_section(ticker: str, section: str) -> str | None:
    """Pull a single section of the latest 10-K via sec-api."""
    api_key = os.getenv("SEC_API_KEY")
    if not api_key:
        print("[skip SEC] SEC_API_KEY not set")
        return None
    try:
        from sec_api import QueryApi, ExtractorApi
    except ImportError:
        print("[skip SEC] pip install sec-api")
        return None

    q = QueryApi(api_key=api_key)
    e = ExtractorApi(api_key=api_key)
    resp = q.get_filings({
        "query": f'ticker:{ticker} AND formType:"10-K"',
        "from": "0", "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    filings = resp.get("filings", [])
    if not filings:
        return None
    url = filings[0].get("linkToFilingDetails") or filings[0].get("linkToFilingDataSummaries")
    if not url:
        return None
    try:
        return e.get_section(url, section, "text")
    except Exception as exc:
        print(f"[warn SEC] {ticker} section {section}: {exc}")
        return None


def fetch_alpha_vantage_overview(ticker: str) -> dict | None:
    """Pull the company overview JSON from Alpha Vantage (free tier)."""
    key = os.getenv("ALPHA_VANTAGE_KEY")
    if not key:
        print("[skip AV] ALPHA_VANTAGE_KEY not set")
        return None
    try:
        import requests
    except ImportError:
        return None
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "OVERVIEW", "symbol": ticker, "apikey": key},
            timeout=15,
        )
        data = r.json()
        # AV rate limit: 5 req/min on free tier — be polite.
        time.sleep(13)
        return data if data and "Symbol" in data else None
    except Exception as exc:
        print(f"[warn AV] {ticker}: {exc}")
        return None


def chunk_text(text: str, size: int = CHUNK_CHARS) -> Iterable[str]:
    text = " ".join(text.split())  # collapse whitespace
    for i in range(0, len(text), size):
        chunk = text[i:i + size]
        if len(chunk) >= 200:  # drop tiny tail chunks
            yield chunk


def make_sec_examples(ticker: str, section_id: str, label: str, body: str) -> list[dict]:
    """Turn a filing section into 2-4 instruction/response pairs."""
    chunks = list(chunk_text(body))
    if not chunks:
        return []
    out: list[dict] = []
    # 1) Summarization prompt on the first chunk
    out.append({
        "instruction": f"Summarize the {label} disclosed in {ticker}'s most recent 10-K.",
        "input": chunks[0],
        "output": chunks[0][:600],  # placeholder — see note below
    })
    # 2) Extract-the-key-points prompt on a middle chunk
    if len(chunks) > 1:
        mid = chunks[len(chunks) // 2]
        out.append({
            "instruction": f"List three concrete {label} from the following excerpt of {ticker}'s 10-K. Quote directly.",
            "input": mid,
            "output": mid[:500],
        })
    return out


def make_av_examples(ticker: str, ov: dict) -> list[dict]:
    """Turn an Alpha Vantage company overview into Q&A pairs."""
    examples: list[dict] = []
    fields = [
        ("Sector", "What sector does {t} operate in?"),
        ("Industry", "What industry is {t} classified under?"),
        ("MarketCapitalization", "What is {t}'s market capitalization (USD)?"),
        ("PERatio", "What is {t}'s P/E ratio?"),
        ("ProfitMargin", "What is {t}'s profit margin?"),
        ("Description", "Briefly describe {t}'s business."),
    ]
    for key, q_tmpl in fields:
        val = ov.get(key)
        if val and val != "None":
            examples.append({
                "instruction": q_tmpl.format(t=ticker),
                "input": "",
                "output": str(val)[:800],
            })
    return examples


def main() -> None:
    random.seed(SEED)
    examples: list[dict] = []

    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")

        for sec_id, label in SECTIONS:
            body = fetch_sec_section(ticker, sec_id)
            if body:
                examples.extend(make_sec_examples(ticker, sec_id, label, body))

        ov = fetch_alpha_vantage_overview(ticker)
        if ov:
            examples.extend(make_av_examples(ticker, ov))

    print(f"\nTotal raw examples: {len(examples)}")
    if not examples:
        print("No data collected — check API keys and try again.")
        return

    # IMPORTANT: the SEC examples above use the input chunk as a placeholder
    # for the output. Before training, hand-edit the JSONL or run a stronger
    # model (e.g. GPT-4o-mini) over the dataset to produce real summaries.
    # The skeleton lets you see the shape; the *quality* of the output field
    # is what you'll fine-tune on.

    random.shuffle(examples)
    n_eval = max(1, int(len(examples) * EVAL_FRACTION))
    eval_set = examples[:n_eval]
    train_set = examples[n_eval:]

    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with EVAL_PATH.open("w", encoding="utf-8") as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_set)} train -> {TRAIN_PATH}")
    print(f"Wrote {len(eval_set)} eval  -> {EVAL_PATH}")


if __name__ == "__main__":
    main()
