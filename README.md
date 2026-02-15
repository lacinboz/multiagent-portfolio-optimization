# Multi-Agent LLM Framework for Portfolio Risk Optimization

This project implements a hybrid portfolio optimization system that combines classical quantitative finance methods with a structured multi-agent LLM architecture.

The framework performs constrained mean–variance optimization, computes risk metrics and efficient frontier points, and integrates an LLM-based reasoning layer for interpretation, validation, and news-aware portfolio adjustments.

The final output is an interactive portfolio optimization dashboard where users can select asset universes, adjust constraints, visualize risk–return trade-offs, and receive structured LLM-generated insights and evidence-grounded action proposals.

---

## System Architecture

The system consists of two tightly integrated layers:

### Numerical Finance Engine

- Compute daily returns from historical price data  
- Estimate volatility, covariance matrix, and Sharpe ratio  
- Generate efficient frontier points  
- Solve constrained mean–variance optimization problems  
- Enforce allocation constraints (e.g., maximum weight per asset)  
- Compare portfolio candidates (e.g., Max Sharpe vs. Min Variance)  

All portfolio computations are deterministic and reproducible.

---

###  Multi-Agent LLM Layer

Instead of using a single monolithic prompt, the system applies a structured reasoning pipeline with specialized agents:

- **Data Agent**  
  Validates input data and describes portfolio universe characteristics.

- **Risk Agent**  
  Interprets volatility, concentration, and diversification metrics.

- **Allocation Agent**  
  Compares portfolio candidates and explains trade-offs using structured metrics.

- **Supervisor / Verifier Agent**  
  Performs consistency checks and corrects invalid or contradictory outputs.

This architecture increases robustness, interpretability, and controllability compared to single-prompt designs.

---

## News-Aware Risk Overlay (Evidence-Grounded)

The framework includes a structured news integration layer:

- Fetches company-level news for selected tickers  
- Assigns deterministic `evidence_id`s to each news item  
- Generates a structured **News Snapshot** (ticker-by-ticker)  
- Produces a validated `risk_json` including:
  - risk flags  
  - confidence scores  
  - volatility regime classification  
- Proposes portfolio adjustment actions grounded in specific evidence IDs  

All proposed actions:
- Must reference valid `evidence_id`s  
- Are deterministically parsed and schema-validated  
- Are filtered against the allowed asset universe  
- May undergo a strict LLM fixer pass for structural compliance  

This ensures that LLM outputs remain structured, verifiable, and traceable back to concrete news evidence.

---

## Interactive Dashboard

The dashboard enables users to:

- Select and modify the asset universe  
- Adjust risk constraints (e.g., maximum weight)  
- Compare candidate portfolios  
- Visualize the efficient frontier  
- Inspect allocation differences  
- Review LLM-generated explanations  
- Examine news-grounded action proposals with attached evidence  

All proposed actions are displayed together with supporting evidence and contextual risk information.

---

## Key Design Principles

- Deterministic portfolio computation  
- Strict schema validation for LLM outputs  
- Evidence-grounded reasoning  
- Separation of decision and explanation  
- Multi-stage verification instead of blind trust  
- Clear separation between numerical engine and language layer  

---

## Research Context

This repository accompanies the Master’s thesis of **Laçin Boz**.

The thesis explores how large language models can be integrated into classical portfolio optimization workflows through a structured multi-agent architecture, enabling:

- Explainable portfolio decisions  
- Evidence-grounded news-driven adjustments  
- Controlled LLM reasoning pipelines  
- Hybrid symbolic–numeric financial systems  

The project demonstrates that LLMs can serve as interpretable reasoning components within quantitative finance systems rather than opaque decision-makers.
