# A Structured LLM-in-the-Loop Decision Framework for Portfolio Construction and Risk Optimization
Companion codebase for the Master's thesis submitted to THWS (Faculty of Computer Science and Business Information Systems, Artificial Intelligence Program).

This project investigates how financial news can be integrated into portfolio construction and risk optimization. It implements and evaluates three complementary news-integration approaches: (1) sentiment-based portfolio adjustments using FinBERT, (2) prediction-based portfolio constraints derived from supervised machine learning models, and (3) an LLM-driven reasoning layer that proposes portfolio actions from news evidence.

To support transparency and experimentation, all approaches are integrated into a unified portfolio optimization framework and exposed through an interactive Streamlit dashboard. The system combines traditional mean-variance optimization, financial news analysis, machine learning, and local LLM reasoning within a LangGraph-based workflow, allowing users to compare different news integration strategies and inspect their impact on portfolio decisions.


> **Note on scope.** This README documents the code as submitted alongside the thesis. The `data/` directory (raw price history, processed covariance matrices, cached news, trained models) is **not included** in this archive due to size. Instructions for regenerating it from scratch are provided in [Data Pipeline](#3-data-pipeline).

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Repository Structure](#2-repository-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Installation](#4-installation)
5. [Environment Configuration](#5-environment-configuration)
6. [Running the Dashboard](#6-running-the-dashboard)
7. [Training the News Prediction Model (Mode B)](#7-training-the-news-prediction-model-mode-b)
8. [Module Reference](#8-module-reference)
9. [Reproducing Thesis Evaluation Results](#9-reproducing-thesis-evaluation-results)
10. [Debug Flags](#10-debug-flags)
11. [Known Limitations](#11-known-limitations)

---

## 1. System Architecture

The system is built around three layers, corresponding to Chapter 1 (Introduction) and Chapter 4 (Methodology) of the thesis:

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM REASONING LAYER                          │
│   (llm_client.py — candidate selection, insight generation,      │
│    news snapshot / actions generation, chat intent routing)      │
└───────────────────────────┬───────────────────────────────────────┘
                             │
┌───────────────────────────▼───────────────────────────────────────┐
│                  NEWS INTEGRATION LAYER                          │
│  Mode A (probabilistic_news_integration.py) — FinBERT sentiment  │
│    adjusts μ̂ and Σ̂ before optimization                         │
│  Mode B (news_return_predictor.py +                              │
│    news_constraint_integration.py) — trained LR/RF classifier    │
│    imposes hard bullish/bearish weight constraints                │
│  Mode C (llm_client.py action generation +                       │
│    agents_langgraph.apply_news_actions_to_params) — LLM proposes │
│    evidence-grounded parameter adjustments (exclude ticker,       │
│    tighten w_max, shift objective, reduce exposure, hedge)        │
└───────────────────────────┬───────────────────────────────────────┘
                             │
┌───────────────────────────▼───────────────────────────────────────┐
│                     QUANTITATIVE CORE                             │
│   portfolio_core.py (base SLSQP MVO) /                            │
│   portfolio_prediction_core.py (Mode-B constrained MVO)           │
│   Deterministic, mathematically reproducible min-variance and     │
│   max-Sharpe optimization with ℓ2 regularization and PSD repair.  │
└─────────────────────────────────────────────────────────────────┘
```

All three modes are coordinated by a **LangGraph** state machine (`portfolio_langgraph_withllm.py`) exposing three compiled graph variants (standard, Mode A, Mode B) that share clarification, perception, baseline computation, candidate selection, and insight-generation stages while diverging in optimization/news routing. The system is exposed through `dashboard_langgraph_app.py`, a Streamlit application with a three-column base layout, a conversational chatbot, and downstream evaluation/explanation panels (Before/After comparison, efficient frontier overlay, "Why This Portfolio" cards, Portfolio Story Timeline, Asset Detail Drilldown).

---

## 2. Repository Structure

```
.
├── dashboard_langgraph_app.py          # Streamlit entry point (UI + session state)
├── portfolio_langgraph_withllm.py      # LangGraph state machine and workflow orchestration
├── agents_langgraph.py                 # Deterministic agents: data/optimization/risk/news-fetch/insight-prep
├── llm_client.py                       # All LLM calls (Ollama/HF): selection, snapshot, actions, insight
├── evidence_utils.py                   # Deterministic evidence_id assignment (SHA-1 based)
├── portfolio_core.py                   # Base Markowitz MVO engine (min-var / max-Sharpe / frontier)
├── portfolio_prediction_core.py        # Mode-B constrained MVO engine + ablation-study variant
├── probabilistic_news_integration.py   # Mode A: FinBERT scoring, μ̂/Σ̂ adjustment, evaluation helpers
├── news_return_predictor.py            # Mode B: dataset construction, LR/RF training, model persistence
├── news_constraint_integration.py      # Mode B: probability-threshold constraint builder
├── get_yahoodata.py                    # Step 1 of data pipeline: downloads raw daily prices via yfinance
├── build_returns_yahoodata.py          # Step 2 of data pipeline: train/test split, μ̂/Σ̂ computation
├── news.py                             # Standalone Finnhub API debug/probe CLI (not part of the app pipeline)
├── requirements.txt
├── .env.example                        # Template for required environment variables (no real secrets)
└── data/                               # NOT INCLUDED — see Section 3 to regenerate
    ├── raw/daily_yahoo/                # Per-ticker OHLCV CSVs from get_yahoodata.py
    ├── processed_yahoo/                # μ̂, Σ̂, train/test splits from build_returns_yahoodata.py
    ├── news_prediction/                # Mode B datasets, trained models, latest signal snapshots
    └── news_cache/                     # Disk cache for Finnhub API responses (agents_langgraph.py)
```

---

## 3. Data Pipeline

The dashboard depends on precomputed price statistics and (for Mode B / the "trained news prediction model" dashboard panel) a trained classifier. These are **not shipped** in this archive and must be regenerated in the following order.

### Step 1 — Download raw price history

```bash
python get_yahoodata.py
```

Downloads daily OHLCV data via `yfinance` for the 101-ticker NASDAQ universe (hardcoded ticker list inside the script, matching the universe evaluated in the thesis) from `2020-01-01` to the current date, and writes one CSV per ticker to `data/raw/daily_yahoo/{TICKER}_daily.csv`.

### Step 2 — Build training/test splits and annualized inputs

```bash
python build_returns_yahoodata.py
```

Reads all `*_daily.csv` files under `data/raw/daily_yahoo/`, merges them on the trading-date index, and produces the canonical train/test split used throughout the thesis evaluation:

| Constant | Value |
|---|---|
| `TRAIN_END_DATE` | `2026-01-14` |
| `TEST_START_DATE` | `2026-01-15` |
| `TEST_END_DATE` | `2026-05-22` |

Outputs written to `data/processed_yahoo/`:

- `prices_daily.csv`, `prices_train.csv`, `prices_test.csv`
- `returns_daily.csv` (training daily returns), `returns_test.csv` (held-out test returns)
- `summary_per_asset_annual.csv` (μ̂ per ticker, annualized: `mu_daily × 252`)
- `cov_daily.csv`, `cov_annual.csv` (Σ̂, annualized: `cov_daily × 252`)
- `debug_daily_vs_annual_returns.csv` (sanity-check table, also surfaced in the dashboard's "Show expected return calculation debug" panel)

**No look-ahead bias:** μ̂ and Σ̂ are computed exclusively from the training window (`≤ 2026-01-14`); the test window (`2026-01-15`–`2026-05-22`) is reserved for realized-performance evaluation only.

### Step 3 — (Optional) Train the Mode B news prediction model

Required only if you want the "trained news prediction model" dashboard panel and Mode B (prediction-constrained optimization) to function. See [Section 7](#7-training-the-news-prediction-model-mode-b).

---

## 4. Installation

Requires Python 3.10+. A CUDA-capable GPU is optional but recommended for FinBERT scoring throughput; CPU inference works but is slower.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas
numpy
scipy
requests
python-dotenv
pyarrow
streamlit
plotly
langgraph
scikit-learn
joblib
transformers
torch
yfinance
```

> **Note:** `transformers` + `torch` are required for FinBERT (`ProsusAI/finbert`, used in `probabilistic_news_integration.py`). The model weights are downloaded automatically from the Hugging Face Hub on first use and cached locally by the `transformers` library — no manual download step is required, but the first run will need network access.

---

## 5. Environment Configuration

Copy `.env.example` to `.env` and fill in your own credentials. **Never commit a `.env` file with real secrets.**

```env
# --- Finnhub (required: news fetching for Modes A, B, C) ---
FINNHUB_API_KEY=

# --- LLM provider selection ---
# "ollama" runs a local model via Ollama; "hf" calls the Hugging Face Inference API.
LLM_PROVIDER=ollama

# --- Ollama settings (used when LLM_PROVIDER=ollama) ---
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TEMPERATURE=0.0
OLLAMA_TIMEOUT_S=240

# --- Hugging Face Inference API settings (used when LLM_PROVIDER=hf) ---
HF_TOKEN=
HF_MODEL=Qwen/Qwen2.5-7B-Instruct

# --- Debug flags (optional, verbose console logging; see Section 10) ---
LLM_DEBUG_METRICS=0
LLM_DEBUG_INTENT=0
LLM_DEBUG_VERIFIER=0
LLM_DEBUG_NEWS_ACTIONS_LINES=0
LLM_DEBUG_CHAT_INTENT=0
LLM_DEBUG_NEWS_EVIDENCE_IDS=0
LLM_DEBUG_NEWS_ALLOWED_EIDS=0
LLM_DEBUG_NEWS_FORMATTER2=0
LLM_DEBUG_ACTIONS_PARSE=0
```

If you run with `LLM_PROVIDER=ollama` (the configuration used for the thesis evaluation), you must have [Ollama](https://ollama.com) installed and running locally with the target model pulled:

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

The thesis's Scope and Limitations section (Ch. 1) notes that the LLM components were evaluated with **Qwen2.5-3B-Instruct via Ollama** and **Qwen2.5-7B-Instruct via the Hugging Face Inference API** as the two locally-hosted, open-weight configurations; adjust `OLLAMA_MODEL` / `HF_MODEL` accordingly to reproduce either setting.

---

## 6. Running the Dashboard

The dashboard is a Streamlit application. Because Streamlit apps are not run with a plain `python` invocation, start it with:

```bash
streamlit run dashboard_langgraph_app.py
```

This opens the app in your default browser (typically `http://localhost:8501`). On first load:

1. Select the asset universe (defaults to all available tickers found in `data/processed_yahoo/summary_per_asset_annual.csv`).
2. Click **"Run Base Portfolio"** to generate the initial unconstrained maximum-Sharpe portfolio.
3. Use the chatbot or the refinement panel to request news integration (you will be prompted to choose between Mode A, Mode B, or Mode C), express portfolio preferences, or ask for an explanation of the active portfolio.

---

## 7. Training the News Prediction Model (Mode B)

Mode B (prediction-constrained optimization) and the dashboard's "trained news prediction model" panel both depend on a persisted classifier bundle at:

```
data/news_prediction/best_news_prediction_model.joblib
data/news_prediction/best_news_prediction_metrics.json
data/news_prediction/best_news_prediction_predictions.csv
data/news_prediction/latest_news_prediction_signals.csv
```

These are produced by `news_return_predictor.py`, which can be run directly as a script:

```bash
python news_return_predictor.py
```

This performs, in order:

1. **Historical news collection** — fetches up to 365 days of company news per ticker via Finnhub (`agents_langgraph.fetch_company_news_for_ticker_window`), excluding the most recent 14 days to avoid look-ahead bias, in 30-day sub-windows to respect API rate limits.
2. **FinBERT enrichment** — scores every article with `probabilistic_news_integration.build_article_signals_with_finbert`.
3. **Price-matching** — for each article, locates the first trading day strictly after publication and the price 7 trading days later, computing the forward return and 5-/20-day momentum and volatility features.
4. **Ticker–date aggregation** — collapses article-level rows to one row per (ticker, date), with confidence-weighted sentiment aggregation and 5-/20-day rolling sentiment/confidence flow features (lagged by one period to prevent look-ahead bias).
5. **Training** — a chronological 70/30 split (by row index on a date-sorted frame), training an `L2`-regularized `LogisticRegression` (`C=0.3`, `class_weight=balanced`) or a `RandomForestClassifier` (`n_estimators=500`, `max_depth=6`, `class_weight=balanced_subsample`), with one-hot ticker dummy variables included as inputs.
6. **Persistence** — saves the model bundle, metrics JSON, predictions CSV, and the latest per-ticker signal snapshot used live by the dashboard.

> **API cost warning.** This script makes real Finnhub API calls and is rate-limited (`MAX_API_CALLS_PER_RUN`, batched by `ALL_TICKER_BATCH_SIZE`/`ALL_TICKER_BATCH_INDEX`) to stay within free-tier limits. Building the full 101-ticker dataset from scratch requires multiple runs across ticker batches; the `__main__` block at the bottom of the script controls which experiment configuration executes. Historical raw news is cached to disk (`CACHE_TTL_S = 7 days`) so re-runs do not re-fetch already-collected windows.

The production configuration used in the thesis evaluation is `min_abs_return_for_signal=0.02`, `model_type="logistic"`, over the full 101-ticker universe — this is the exact experiment entry in the `experiments` list inside `news_return_predictor.py`'s `__main__` block, and its output is what gets copied to `best_news_prediction_model.joblib`.

---

## 8. Module Reference

### `dashboard_langgraph_app.py`
Streamlit UI layer. Owns all `st.session_state` (base/refined graph states, chat history, pending clarifications). Renders: setup panel, portfolio composition donut chart, risk/performance metric cards, the refinement chatbot (with intent-routed message rendering for news overview / news actions / prediction-constrained results / final insight), Before/After comparison charts, efficient frontier and dual-frontier overlays, "Why This Portfolio" cards, the Portfolio Story Timeline (rendered as embedded HTML via `streamlit.components.v1`), the Asset Detail Drilldown, and diagnostic/export panels (raw JSON state download).

### `portfolio_langgraph_withllm.py`
Defines `PortfolioState` (a `TypedDict` with ~50 fields) and the full LangGraph pipeline: 18 nodes (`node_ask_clarifications`, `node_perception`, `node_compute_baselines`, `node_data`, `node_optimize` / `node_optimize_prob_news` / `node_optimize_prediction_constraint`, `node_extract_candidates`, `node_risk_candidates`, `node_news_fetch`, `node_news_snapshot_and_risk`, `node_news_actions_generate`, `node_news_evidence_snapshot`, `node_news_actions_verify`, `node_llm_select_candidate`, `node_finalize_selection`, `node_insight_generator`, `node_explain`) and three compiled graph variants (`build_portfolio_graph`, `build_portfolio_graph_prob_news`, `build_portfolio_graph_prediction_constraint`), invoked via `run_graph`, `run_graph_prob_news`, and `run_graph_prediction_constraint` respectively. Candidate selection is explicitly **locked** (bypassing the LLM) in Modes A and B to keep the news-integration experimental comparison uncorrupted by an LLM-driven objective switch.

### `agents_langgraph.py`
Deterministic, non-LLM building blocks:
- `data_agent_get_mu_cov` — loads and aligns μ̂/Σ̂ for a selected ticker subset.
- `optimization_agent_from_mu_cov` / `prediction_constrained_optimization_agent` — thin wrappers around `portfolio_core.py` / `portfolio_prediction_core.py`.
- `risk_agent` — computes return, volatility, Sharpe, risk contributions, and concentration metrics for an arbitrary weight vector.
- `recommendation_agent` — deterministic (non-LLM) natural-language summary of a chosen candidate.
- `insight_agent_prepare` / `build_insight_payload` / `build_insight_prompts` / `verify_insight_output` — prepares the structured JSON payload and prompt scaffolding later consumed by `llm_client.generate_portfolio_insights`.
- `news_agent_fetch_for_tickers` / `historical_news_agent_fetch_for_tickers` — Finnhub company-news + market-news-fallback fetch layer with a two-tier (in-memory + on-disk SHA-1-keyed) cache.
- `apply_news_actions_to_params` — deterministically applies selected Mode C LLM actions (`exclude_ticker`, `set_w_max`, `shift_objective`, `reduce_exposure`, `hedge`) to the optimizer's parameter set.

### `llm_client.py`
All model calls to the local/hosted LLM (`LLMClient`, provider-agnostic over Ollama/Hugging Face). Responsibilities: chat-command intent classification for the dashboard chatbot; candidate selection with a self-verification pass; news snapshot generation with a strict evidence-anchored bullet format and a deterministic risk-JSON cleaner; Mode C action generation (batched by ~35 items per LLM call, followed by round-robin merging and a FinBERT-based sentiment-consistency quality filter that tags/drops actions whose direction contradicts the sentiment of their own cited evidence); evidence-snapshot narrative generation; and final portfolio insight generation (narrative or structured-JSON mode).

### `evidence_utils.py`
`assign_evidence_ids_and_map` — deterministic, stable `evidence_id` construction (`{TICKER}_{sha1(url)[:8]}`) used to make every LLM-generated Mode C action traceable back to a specific news article, per the thesis's evidence-grounding requirement.

### `portfolio_core.py`
Base Markowitz mean-variance engine. Implements `near_psd` (eigenvalue clipping for covariance repair), `portfolio_stats`, `risk_contributions`, `sharpe_ratio`, and `run_portfolio_optimization`, which solves both the ℓ2-regularized minimum-variance problem and the maximum-Sharpe problem via `scipy.optimize.minimize` (SLSQP), with an automatic feasibility guard relaxing `w_max` to `1/n` when infeasible, and approximates the efficient frontier over a 60-point target-return grid.

### `portfolio_prediction_core.py`
Extends the base engine with Mode-B side constraints (`run_portfolio_optimization_prediction`): given a `news_constraints` dict (bullish min-floor / bearish max-cap per ticker), adds inequality constraints to the SLSQP problem without modifying μ̂ or Σ̂. Falls back to `trust-constr` if SLSQP fails to converge. Also includes `run_portfolio_optimization_prediction_ablation`, a variant supporting a `relaxed_w_max` per-ticker override, used for the Evaluation chapter's configuration-ablation study (configurations A1/A2/B1/B2/C1/C2).

### `probabilistic_news_integration.py`
Mode A implementation. `FinBERTScorer` loads `ProsusAI/finbert` once and batches article scoring. `build_article_signals_with_finbert` computes per-article sentiment (`positive − negative` probability), composite confidence (model certainty, source credibility, recency decay, content richness), and combined article weight. `aggregate_news_signal_by_ticker` produces ticker-level weighted sentiment/confidence/variance. `adjust_expected_returns` / `adjust_covariance_matrix` apply the μ̂'/Σ̂' adjustment described in Methodology §4.4.1. `evaluate_news_adjustment_effect` and `evaluate_news_prediction_against_future_returns` / `build_historical_news_prediction_evaluation` implement the portfolio-level and directional-accuracy evaluations shown in the Mode A dashboard panels and referenced in the Evaluation chapter.

### `news_return_predictor.py`
Mode B dataset construction and model training (see [Section 7](#7-training-the-news-prediction-model-mode-b) for the full pipeline description). Also exposes `load_prediction_model` and `predict_ticker_probabilities`, used by the LangGraph Mode-B node to load the persisted classifier and generate live per-ticker probabilities.

### `news_constraint_integration.py`
`build_news_probability_constraints` — converts a trained model's per-ticker predicted probabilities into the bullish-floor (`p ≥ θ+`) / bearish-cap (`p ≤ θ−`) side-constraint dictionary consumed by `portfolio_prediction_core.py`, applying the production thresholds `θ+ = 0.60`, `θ− = 0.40`, `δ = 0.02` used throughout the thesis evaluation.

### `get_yahoodata.py` / `build_returns_yahoodata.py`
Data pipeline, see [Section 3](#3-data-pipeline).

### `news.py`
Standalone CLI probe for exploring the raw Finnhub API response shape during development (company news per ticker + market news + a simple keyword-based ticker filter). **Not imported by any part of the live application** — kept for reference/debugging only.

---

## 9. Reproducing Thesis Evaluation Results

| Thesis result | Producing script / function |
|---|---|
| μ̂, Σ̂, 585 training returns, 88 test returns (Table in §5.1) | `build_returns_yahoodata.py` |
| Efficient frontier of the 101-ticker universe (Fig. 5.1) | `portfolio_core.run_portfolio_optimization` (frontier field) |
| LR vs. RF model selection (Table 5.2 / 5.3) | `news_return_predictor.train_news_flow_predictor`, run once with `model_type="logistic"` and once with `model_type="random_forest"` |
| Feature-set ablation (Tables 5.4 / 5.5) | `news_return_predictor.get_feature_cols` restricted to each feature subset, re-run through `train_news_flow_predictor` and the portfolio-level realized-metric pipeline |
| Mode A parameter sensitivity (α, β_cov, t½ grid) | `probabilistic_news_integration.build_probabilistic_news_adjusted_inputs`, swept externally over the parameter grid |
| Configuration ablation A1/A2/B1/B2/C1/C2 (Table 5.7) | `portfolio_prediction_core.run_portfolio_optimization_prediction_ablation` (supports the `relaxed_w_max` override needed for A2/B2) |
| Mode-B 64-configuration parameter sensitivity (θ+, θ−, δ grid) | `news_constraint_integration.build_news_probability_constraints` swept over the threshold/delta grid, feeding `portfolio_prediction_core.run_portfolio_optimization_prediction` |
| News integration realized-performance comparison (Table 5.9) | Mode A via `run_graph_prob_news`, Mode B via `run_graph_prediction_constraint`, both evaluated against `data/processed_yahoo/returns_test.csv` |
| Baseline comparison (Equal Weight, Plain MVO, Zhang 2022, Colasanto 2022, NC-MVO) | NC-MVO and Plain MVO share `portfolio_core.py`; the Zhang/Colasanto adapted baselines are **not included in this repository** — they were implemented as standalone evaluation scripts external to the dashboard application and are not part of the live pipeline. |

---

## 10. Debug Flags

Set to `"1"` in your `.env` (or shell environment) to enable verbose console logging for the corresponding subsystem. All default to off (`"0"`/unset).

| Flag | Logs |
|---|---|
| `LLM_DEBUG_METRICS` | Metric tables passed into candidate-explanation prompts |
| `LLM_DEBUG_INTENT` | Parsed user-feedback intent JSON (risk aversion / return seeking) |
| `LLM_DEBUG_VERIFIER` | Candidate-selection self-verification corrections |
| `LLM_DEBUG_NEWS_ACTIONS_LINES` | Full Mode C action generation trace: per-batch raw LLM output, fixer invocations, FinBERT quality-filter keep/drop decisions |
| `LLM_DEBUG_CHAT_INTENT` | Dashboard chatbot intent classification raw output |
| `LLM_DEBUG_NEWS_EVIDENCE_IDS` | Evidence-ID assignment for news items entering the snapshot prompt |
| `LLM_DEBUG_NEWS_ALLOWED_EIDS` | Which evidence IDs are considered "allowed" for grounding a given snapshot/action |
| `LLM_DEBUG_NEWS_FORMATTER2` | Second-pass snapshot formatter (canonicalization) input/output |
| `LLM_DEBUG_ACTIONS_PARSE` | Line-based Mode C action parser intermediate state |

---

## 11. Known Limitations

These mirror the Limitations discussed in Chapter 6 of the thesis, restated here from an implementation perspective:

- **Static evaluation window.** μ̂ and Σ̂ are estimated once from the training period and held fixed through the 88-day test window; the system does not re-estimate inputs at each rebalancing date (see Future Work: rolling out-of-sample evaluation).
- **No transaction-cost modeling.** Neither `portfolio_core.py` nor `portfolio_prediction_core.py` account for transaction costs, market impact, or bid-ask spread.
- **Finnhub API dependency.** Both live dashboard usage and Mode-B model training depend on Finnhub API availability and rate limits; the on-disk cache (`data/news_cache/`) mitigates but does not eliminate this dependency.
- **Local LLM variability.** Because `llm_client.py` is provider-agnostic but was evaluated primarily against locally-hosted, open-weight models (Qwen2.5 family), output quality/consistency may differ if a different model or provider is substituted.
- **`ALPHAVANTAGE_API_KEY`** appears in some historical `.env` configurations used during development but is **not referenced by any script in this repository** — it can be safely omitted unless you extend the codebase to use it.