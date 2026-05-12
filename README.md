# Multi-Agent LLM Framework for Portfolio Risk Optimization

This project implements a hybrid portfolio optimization framework that combines classical quantitative finance methods with a structured multi-agent LLM architecture and a news-aware prediction pipeline.

The system integrates deterministic portfolio optimization, evidence-grounded news reasoning, and machine learning–based news-flow prediction into a unified research framework for explainable portfolio decision support.

The final output is an interactive portfolio optimization dashboard where users can:

- select asset universes,
- adjust optimization constraints,
- visualize portfolio risk–return trade-offs,
- inspect evidence-grounded news signals,
- and receive structured LLM-generated portfolio reasoning.

---

# System Architecture

The framework consists of three tightly integrated layers:

1. Numerical Finance Engine  
2. Multi-Agent LLM Reasoning Layer  
3. News-Driven Prediction and Risk Overlay Layer  

---

# Numerical Finance Engine

The numerical portfolio optimization engine performs:

- Daily return computation from historical price data  
- Volatility and covariance estimation  
- Sharpe ratio calculation  
- Efficient frontier generation  
- Constrained mean–variance optimization  
- Portfolio comparison (e.g., Max Sharpe vs. Min Variance)  
- Allocation constraint enforcement  

All portfolio computations are deterministic and reproducible.

---

# Multi-Agent LLM Layer

Instead of relying on a single monolithic prompt, the system applies a structured multi-agent reasoning pipeline with specialized agents.

### Data Agent

- Validates financial inputs
- Describes the portfolio universe
- Summarizes asset characteristics

### Risk Agent

- Interprets volatility and diversification metrics
- Detects concentration risks
- Explains portfolio risk structure

### Allocation Agent

- Compares portfolio candidates
- Explains allocation trade-offs
- Generates structured allocation reasoning

### Supervisor / Verifier Agent

- Performs consistency checks
- Detects contradictory outputs
- Enforces schema validity
- Corrects structurally invalid responses

This architecture improves interpretability, robustness, and controllability compared to single-prompt systems.

---

# News-Aware Risk Overlay

The framework includes a structured evidence-grounded news integration layer.

The system:

- Fetches company-level news for portfolio tickers
- Assigns deterministic `evidence_id`s to news items
- Generates structured news snapshots
- Produces validated `risk_json` outputs
- Detects risk flags and volatility regime shifts
- Generates evidence-grounded portfolio adjustment proposals

All generated actions:

- must reference valid evidence identifiers,
- are schema-validated,
- are filtered against the allowed asset universe,
- and may undergo additional verifier passes for structural compliance.

This keeps LLM outputs traceable, auditable, and explainable.

---

# News-Flow Prediction Pipeline

The framework also includes a machine learning pipeline for predicting short-term market direction from aggregated financial news sentiment.

The prediction system:

- Collects historical company news over rolling time windows
- Computes FinBERT-based sentiment signals
- Aggregates ticker-date level sentiment features
- Integrates historical return and volatility features
- Trains time-series classification models
- Produces ticker-level directional prediction signals

The pipeline supports:

- Logistic Regression
- Random Forest classification
- Threshold-based market movement filtering
- Chronological train-test evaluation
- Feature importance analysis
- Prediction confidence estimation

---

# Data Leakage Audits and Validation

Extensive validation and sanity-check procedures were implemented to ensure that model performance is not caused by data leakage.

The audit framework includes:

- Chronological date validation checks
- Future-price dependency validation
- Feature leakage inspection
- Train-test overlap verification
- Random-label sanity tests
- Threshold robustness experiments

The validation pipeline verifies that:

- future information is not used before news publication dates,
- target-dependent variables are excluded from training features,
- train/test ticker-date overlaps do not exist,
- and shuffled-label experiments collapse to near-random performance.

These checks support that the observed predictive performance reflects weak but genuine statistical signal rather than accidental leakage.

---

# Interactive Dashboard

The dashboard enables users to:

- Select and modify asset universes
- Adjust portfolio constraints
- Compare optimized portfolios
- Visualize efficient frontier curves
- Inspect allocation differences
- Review LLM-generated portfolio explanations
- Examine evidence-grounded news adjustments
- Monitor ticker-level prediction signals

All portfolio actions are displayed together with supporting evidence and contextual reasoning.

---

# Key Design Principles

- Deterministic portfolio computation
- Evidence-grounded reasoning
- Strict schema validation
- Separation of optimization and explanation
- Multi-stage verification pipelines
- Chronological time-series evaluation
- Leakage-aware ML experimentation
- Hybrid symbolic–numeric architecture

---

# Research Context

This repository accompanies the Master’s thesis of **Laçin Boz**.

The thesis investigates how large language models can be integrated into classical portfolio optimization systems through structured multi-agent reasoning and evidence-grounded financial analysis.

The research focuses on:

- Explainable portfolio optimization
- News-driven portfolio adjustment systems
- Hybrid LLM–quantitative finance architectures
- Structured multi-agent reasoning
- Leakage-aware financial prediction pipelines
- Interpretable AI-assisted investment systems

The project demonstrates that LLMs can function as controlled reasoning and interpretation components within quantitative finance systems rather than opaque autonomous decision-makers.
