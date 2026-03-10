# ERRGen — Evidence-Grounded Equity Research Report Generator

An agentic pipeline for generating professional equity research reports with
**paragraph-level citation tracking** and a **pre-prediction
verification-and-revision loop**.

Every analytical claim is grounded in real retrieved evidence. Every number is
computed by a deterministic calculator. No sentence makes it into the final
report without surviving the checker → reviser → re-checker loop.

---

## Architecture

```
User Request
    │
    ▼
A. Request Parser          (UserRequest model)
    │
    ▼
B. Data Collection         (FMP + NewsAPI → EvidenceChunk list)
    │  ├─ FMP: company profile, income statement, balance sheet, cash flow, news
    │  └─ NewsAPI: keyword-based news search
    ▼
C. Information Extraction  (FinancialExtractor + NewsExtractor)
    │  ├─ FinancialExtractor: deterministic fact extraction + calc requests
    │  └─ NewsExtractor: LLM-based event/signal extraction
    ▼
D. Analysis Agents         (one per section, evidence-first generation)
    │  ├─ CompanyOverview      (direct from profile, no LLM)
    │  ├─ NewsAnalysisAgent    (Recent Developments)
    │  ├─ FinancialAnalysisAgent
    │  ├─ BusinessAnalysisAgent
    │  └─ RiskAnalysisAgent
    ▼
E. Verification Loop       (per paragraph, max N iterations)
    │  ┌─────────────────────────────────────────────────┐
    │  │  Generate paragraph (analysis agent)            │
    │  │      ↓                                          │
    │  │  CheckerAgent (7 issue types, structured JSON)  │
    │  │      ↓ FAIL                                     │
    │  │  ReviserAgent (targeted fix using feedback)     │
    │  │      ↓                                          │
    │  │  CheckerAgent (re-check)                        │
    │  │      ↓ PASS → keep  |  max iter → UNRESOLVED    │
    │  └─────────────────────────────────────────────────┘
    ▼
F. Prediction Agent        (GATED — only runs if all sections pass)
    │  └─ Investment Recommendation & Outlook
    ▼
G. Report Assembly         (FinalReport with evidence + calc appendix)
    │
    ▼
H. Run Record              (full JSON artifact per run in runs/{run_id}/)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Real APIs only (FMP + NewsAPI) | No fake data. Missing keys → clear error + setup instructions |
| Paragraph-level `chunk_ids` | Every claim traceable to specific evidence |
| Deterministic calculator | LLM never does arithmetic; all numbers are auditable |
| Checker returns structured issues | 7 issue types with severity, span, fix direction |
| Prediction gate | Recommendation cannot be generated if upstream has unresolved critical issues |
| JSON run artifacts | Full pipeline state saved for debugging and research |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or
pip install -e ".[dev]"
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

You need three API keys (all have free tiers):

| Key | Provider | Free tier |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | Pay-per-use |
| `FMP_API_KEY` | [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs/) | 250 req/day |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org/) | 100 req/day |

### 3. Run the pipeline

```bash
# NVIDIA as of January 2025
python scripts/run_report.py \
    --ticker NVDA \
    --as-of 2025-01 \
    --focus "AI chips" "datacenter" "financial analysis" "risks" \
    --print-report
```

```bash
# Apple with custom request text
python scripts/run_report.py \
    --ticker AAPL \
    --as-of 2025-01 \
    --request "Write an equity research report for Apple (AAPL) as of January 2025 \
               focusing on iPhone cycle, services growth, and capital returns."
```

Output:
- Terminal summary with verification status
- `runs/{run_id}/report.md` — rendered Markdown research report
- `runs/{run_id}/` — full JSON artifacts for every pipeline stage

### 4. Run tests (no API keys needed)

```bash
pytest
# or with coverage
pytest --cov=errgen --cov=evaluation
```

---

## Verification Loop

Each paragraph passes through this loop:

```
draft → CheckerAgent → PASS ✅ → done
                     → FAIL ❌ → ReviserAgent → re-check
                                              → PASS ✅ → done
                                              → FAIL ❌ → (repeat up to MAX_REVISION_ITERATIONS)
                                                        → UNRESOLVED ⚠️ → kept with warning
```

The checker evaluates 7 issue types:

| Issue type | Example |
|---|---|
| `unsupported_claim` | Claim not in any cited chunk |
| `citation_mismatch` | Cited chunk doesn't support the specific claim |
| `hallucination` | Content not traceable to any evidence |
| `numerical_error` | Wrong number or formula vs cited chunk/calc |
| `internal_inconsistency` | Contradicts earlier verified content |
| `scope_violation` | References events after the as-of date |
| `overclaiming` | "Will definitely grow 200%" from uncertain evidence |

**The prediction section is gated**: if any upstream section has unresolved
critical/major issues, the recommendation is skipped and a warning is emitted.

---

## Run Artifacts

Every pipeline run saves structured JSON to `runs/{run_id}/`:

```
runs/{run_id}/
  manifest.json          run metadata, status, config snapshot
  request.json           parsed UserRequest
  sources.json           SourceMetadata list (every API call made)
  evidence_chunks.json   all EvidenceChunk objects with full provenance
  extracted_facts.json   structured facts extracted from evidence
  calc_requests.json     every CalculationRequest submitted
  calc_results.json      every deterministic CalculationResult
  paragraphs/
    {paragraph_id}.json  each paragraph with full revision history
  verdicts/
    {verdict_id}.json    each CheckerVerdict with structured issues
  revisions/
    {revision_id}.json   each RevisionRecord (what changed and why)
  report.json            complete FinalReport object
  report.md              rendered Markdown research report
```

Trace any sentence in the final report:
1. Find the paragraph in `report.json` → get its `chunk_ids` and `calc_ids`
2. Look up `chunk_id` in `evidence_chunks.json` → original API text
3. Look up `calc_id` in `calc_results.json` → formula and inputs

---

## Configuration

All settings can be set via `.env` or environment variables:

```bash
# Model selection
OPENAI_MODEL=gpt-4o              # analysis, checking, revision
OPENAI_FAST_MODEL=gpt-4o-mini   # extraction (cheaper)

# Pipeline tuning
MAX_REVISION_ITERATIONS=3        # checker→reviser loop limit per paragraph
MAX_NEWS_ARTICLES=20             # news articles per provider
MAX_FINANCIAL_PERIODS=4          # years of financial statement history

# Storage
RUNS_DIR=runs
```

---

## Evaluation Framework

`evaluation/metrics.py` defines metric interfaces for future research:

| Metric | Status | Description |
|---|---|---|
| `FactualGroundingScore` | Stub (checker pass rate) | Fraction of paragraphs passing verification |
| `CitationPrecision` | Stub (None) | Fraction of citations that actually support claims |
| `CitationRecall` | Stub (None) | Fraction of key facts actually cited |
| `NumericalCorrectnessScore` | Stub (calc error rate) | Numbers correct vs evidence |
| `ReportCompletenessScore` | Stub (section coverage) | Required sections present |
| `ConsistencyScore` | Stub (None) | Cross-section consistency |

Plug in real implementations by subclassing `BaseMetric`:

```python
class MyLLMJudge(BaseMetric):
    name = "citation_precision"
    def evaluate(self, report: FinalReport, **kwargs) -> MetricResult:
        # your LLM-as-judge logic here
        ...
```

---

## Extending the System

### Add a new data provider

```python
# errgen/data/my_provider.py
from errgen.data.base import BaseDataClient
from errgen.models import EvidenceChunk, SourceMetadata

class MyProvider(BaseDataClient):
    def get_data(self, ticker: str) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        raw = self._get("https://...", params={"symbol": ticker})
        # convert to EvidenceChunk objects
        ...
```

### Add a new analysis section

```python
# errgen/analysis/valuation.py
from errgen.analysis.base import BaseAnalysisAgent

class ValuationAnalysisAgent(BaseAnalysisAgent):
    section_name = "Valuation Analysis"

    def _build_user_prompt(self, ticker, request_context, chunk_block, calc_block, as_of_date):
        return f"Write the valuation section for {ticker}...\n{chunk_block}"
```

Then register it in `pipeline.py`'s `_analysis_agents` list.

---

## Project Structure

```
errgen/
├── errgen/                   main package
│   ├── config.py             environment-based configuration
│   ├── models.py             all Pydantic data models
│   ├── llm.py                OpenAI client with retry
│   ├── pipeline.py           main orchestrator
│   ├── report.py             assembler + Markdown renderer
│   ├── run_record.py         run artifact persistence
│   ├── data/                 provider adapters
│   │   ├── fmp.py            Financial Modeling Prep (income, balance, CF, news)
│   │   └── newsapi.py        NewsAPI keyword search
│   ├── calculator/
│   │   └── finance_calc.py   deterministic financial calculator
│   ├── extraction/
│   │   ├── financial.py      rule-based financial fact extraction
│   │   └── news.py           LLM-based news event extraction
│   ├── analysis/
│   │   ├── base.py           shared evidence-first generation contract
│   │   ├── financial.py      financial analysis agent
│   │   ├── news.py           recent developments agent
│   │   ├── risk.py           risk analysis agent
│   │   └── business.py       business & competitive analysis agent
│   └── verification/
│       ├── checker.py        7-type structured paragraph checker
│       └── reviser.py        targeted revision agent
├── evaluation/
│   └── metrics.py            evaluation metric interfaces + stubs
├── tests/
│   ├── test_calculator.py    deterministic calculator (no API needed)
│   ├── test_models.py        data model serialisation
│   └── test_verification.py  checker + reviser with mocked LLM
├── scripts/
│   └── run_report.py         CLI entry point
├── runs/                     generated at runtime
├── .env.example
├── requirements.txt
└── pyproject.toml
```

---

## Research Notes

This system is designed for academic research into:
- Evidence-grounded LLM generation
- Citation precision and recall in financial text
- Self-correction loops for factual accuracy
- Structured verification of numerical claims

The checker/reviser loop architecture is compatible with future work on:
- RL-based self-correction (use checker verdicts as reward signal)
- Fine-tuned analysers/checkers on domain-specific data
- Automated evaluation benchmarks using run artifacts
- Comparative analysis across different LLM backends

---

## License

For academic research use. Ensure compliance with the terms of service of
OpenAI, Financial Modeling Prep, and NewsAPI for your use case.
