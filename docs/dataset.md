# FinNavigator fine-tune dataset

This document describes the dataset used to fine-tune **Qwen3-VL-4B-Instruct** for FinNavigator. The fine-tune lives at `MOH749/finnav-qwen3-VL-4b-lora` (LoRA adapter) and `MOH749/finnav-qwen3-VL-4b-gguf` (merged + Q4_K_M quantised GGUF) on HuggingFace. The dataset is built on the fly inside `finnav_unsloth_qwen3_4b (1).ipynb` (cell `198f3f44`) and is **not** stored as a static file in the repo — every Colab run re-fetches it from HuggingFace.

---

## Goal

Make the base Qwen3-VL-4B faster, more concise, and more reliable on **four narrow financial tasks** that drive the FinNavigator product:

1. **`[SEC]`** — answering questions grounded in real SEC filing context (10-K / 10-Q reasoning).
2. **`[QA]`** — investor-style numeric and conceptual Q&A.
3. **`[SENTIMENT]`** — bearish / bullish / neutral classification of financial statements.
4. **`[GLOSSARY]`** — definitional answers about SEC form types (10-K, 10-Q, 8-K, Form 4, …) so non-experts can use the Research tab.

Each prompt in training is prefixed with the task tag in square brackets so the model learns to multi-task and route on the prefix.

---

## Sources

All sources are **parquet-native** on HuggingFace. We deliberately dropped two earlier candidates (`dreamerdeo/finqa`, `takala/financial_phrasebank`) because HuggingFace deprecated dataset *loading scripts* and both rely on `.py` loaders that no longer execute under recent `datasets` versions.

| Prefix | Source | License | Cap | Why this dataset |
|---|---|---|---|---|
| `[SEC]` | [`PatronusAI/financebench`](https://huggingface.co/datasets/PatronusAI/financebench) | CC-BY 4.0 | 200 | 150 hand-written question/answer pairs grounded in actual SEC filing snippets (`evidence_text`). Highest-quality SEC reasoning corpus available. Small, but each row is dense. |
| `[QA]` | [`virattt/financial-qa-10K`](https://huggingface.co/datasets/virattt/financial-qa-10K) | MIT | 4,000 | ~7k question/answer pairs auto-extracted from real 10-K filings, each with `ticker`, `question`, `answer`, `context`. Tighter to FinNavigator's domain than the older FinQA (which was table-reasoning focused). |
| `[SENTIMENT]` | [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | MIT | 3,000 | ~12k labelled financial tweets/headlines. Labels: 0 = bearish, 1 = bullish, 2 = neutral. Replaces FinPhraseBank (broken) with a more market-flavoured vocabulary. |
| `[GLOSSARY]` | hand-curated in cell `198f3f44` | n/a | 60 | 20 SEC form types × 3 question phrasings each. Same definitions are mirrored in `ui/glossary.py` for the in-UI hover-tooltips on the Research tab, so the model and the UI stay in sync. |

**Dataset totals (approx, after caps):** ~150 + ~4,000 + ~3,000 + ~60 = **~7,200 instruction/input/output rows**.

---

## Format — Alpaca-style instruction tuning

Every row is a dict with three keys, written to `/content/finnav_train.jsonl` and `/content/finnav_eval.jsonl`:

```json
{
  "instruction": "[SEC] What were Apple's principal risk factors disclosed in fiscal 2024?",
  "input":       "<3000-char excerpt from the 10-K's Item 1A>",
  "output":      "Apple disclosed risks including macroeconomic conditions, supply chain concentration in Asia, …"
}
```

The training cell maps these into the **Qwen ChatML** template via the existing Alpaca formatter (cell `783210b2`):

```
<|im_start|>system
You are a helpful financial assistant…
<|im_end|>
<|im_start|>user
[SEC] What were Apple's principal risk factors disclosed in fiscal 2024?

<3000-char excerpt from the 10-K's Item 1A>
<|im_end|>
<|im_start|>assistant
Apple disclosed risks including …
<|im_end|>
```

`max_seq_length = 2048` tokens. Inputs over 3,000 characters of context are truncated at source to keep them well under that cap.

---

## Per-source mapping logic

### `[SEC]` — FinanceBench
```python
combined.append({
  "instruction": f"[SEC] {ex['question']}",
  "input":       (ex.get('evidence_text') or ex.get('evidence') or ex.get('context') or '')[:3000],
  "output":      str(ex['answer']),
})
```
Note: FinanceBench's column naming has shifted across releases, so the loader tolerates `question / query`, `answer / response`, `evidence_text / evidence / context`.

### `[QA]` — financial-qa-10K
```python
combined.append({
  "instruction": f"[QA] {ex['question']}",
  "input":       f"[Ticker: {ex['ticker']}]\n" + str(ex['context'])[:3000],
  "output":      str(ex['answer']),
})
```
The ticker is prepended to the context so the model learns "the answer should reference *this* company".

### `[SENTIMENT]` — twitter-financial-news-sentiment
```python
label_map = {0: "bearish", 1: "bullish", 2: "neutral"}
combined.append({
  "instruction": "[SENTIMENT] Classify the sentiment of the following financial statement as bearish, bullish, or neutral.",
  "input":       ex['text'],
  "output":      label_map[ex['label']],
})
```
The instruction is constant across all 3,000 sentiment rows — the model learns to map a single fixed instruction to a 3-way classification, conditioned on `input`.

### `[GLOSSARY]` — hand-curated
Twenty `(form_code, definition)` tuples, e.g.:
```python
("10-K", "Annual report filed by U.S. public companies under section 13 of the Exchange Act. Contains audited financial statements, MD&A, risk factors, …"),
("Form 4", "Statement of changes in beneficial ownership. Insiders report any purchase or sale of company stock within 2 business days."),
…
```
Each form gets **3 phrasings** so the model doesn't memorise one wording:
- `What is a {code} filing?`
- `What does Form {code} disclose?` *(or "When is a {code} filed?" / "Define SEC Form {code} in one sentence." — sampled per form)*

The same dict lives at `ui/glossary.py:SEC_FORMS` and powers Research-tab tooltips.

---

## Shuffle, split, write

```python
random.seed(42)
random.shuffle(combined)
n_eval     = max(50, int(len(combined) * 0.05))   # 5% held-out
eval_set   = combined[:n_eval]
train_set  = combined[n_eval:]
```
- **Train:** ~6,800 rows
- **Eval:** ~360 rows (with `eval_strategy="steps", eval_steps=100` during training so the curve is visible in W&B)

Determinism: `random.seed(42)` is set before shuffling so the train/eval split is reproducible across notebook runs.

---

## What we're explicitly *not* training on

| Excluded | Reason |
|---|---|
| Vision/image data (charts, filing screenshots) | Fine-tune is text-only this milestone — vision adapter (mmproj) stays frozen at base Qwen3-VL weights and is bundled separately as a 836 MB GGUF for downstream image use. |
| Twitter/StockFit live data shown in the notebook | The optional cells (`242b4362`, `6b4b85d3`) fetch supplemental metadata for inspection but do **not** feed the training set — `format_filings_to_jsonl` was retired when we moved to HF datasets. |
| Synthetic GPT-4o reasoning chains | A previous iteration mixed in `synthetic_career_data.jsonl`. Dropped — it diluted the financial signal and was the main contributor to the broken 1-step initial run (4 total samples). |

---

## Training run that consumed this dataset

| | |
|---|---|
| **Run** | `wi19zc3g` ([wandb link](https://wandb.ai/yaatal/finnavigator-lora/runs/wi19zc3g)) |
| **Base** | `unsloth/Qwen3-VL-4B-Instruct-bnb-4bit` |
| **Adapter** | LoRA `r=16`, `alpha=16`, dropout 0, applied to `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| **Optimiser** | `adamw_8bit`, cosine schedule, lr `2e-4`, weight decay `0.01`, warmup ratio `0.03` |
| **Batch / grad accum** | 1 × 8 (effective batch 8) — Colab T4 ceiling |
| **Epochs** | 3 (~2,500 steps for the ~7 k row train set) |
| **Eval cadence** | `eval_strategy="steps", eval_steps=100`, `EarlyStoppingCallback(patience=3)` on `eval_loss` |
| **Loss trajectory** | step 100 → train 1.228; step 1100 → train 0.998 / eval 1.123; final ~step 2500 train ~0.85 / eval ~1.05 (estimate) |
| **Outputs** | LoRA adapter pushed to `MOH749/finnav-qwen3-VL-4b-lora`. Merged + quantised GGUF (Q4_K_M for the language stack + F16 for the vision projector) pushed to `MOH749/finnav-qwen3-VL-4b-gguf`. |

---

## How to reproduce

The dataset is rebuilt every time the notebook runs — no static file to download. To reproduce:

1. Open `finnav_unsloth_qwen3_4b (1).ipynb` on Colab.
2. Set Colab Secrets `HF_TOKEN` (write scope) and `WANDB_API_KEY` (left sidebar → key icon).
3. Run all cells. Cell `198f3f44` will print:
   ```
   [SEC] FinanceBench loaded: 150
   [QA] financial-qa-10K loaded: ~4000
   [SENTIMENT] twitter-finance loaded: ~3000
   [GLOSSARY] SEC forms loaded: ~60
   Total: ~7210 samples → train=~6850, eval=~360
   ```
4. ~60–90 min on a T4 → LoRA + GGUF land on your HF account, ready to be `ollama pull`-ed locally.

---

## Licensing / attribution

Each source dataset retains its own licence (CC-BY 4.0, MIT). The combined training file is for research/finetune purposes; if you redistribute the resulting model weights, retain attribution to the upstream source datasets. The hand-curated `[GLOSSARY]` entries are written for this project and are MIT-licensed alongside the rest of the FinNavigator codebase.
