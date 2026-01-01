import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from agents import optimization_agent, recommendation_agent, risk_agent

DATA_DIR = Path("data/processed_yahoo")


@st.cache_data
def load_available_tickers():
    summary = pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0)
    return list(summary.index)


def _safe_normalize_current_inputs(df: pd.DataFrame, mode: str) -> dict | None:
    """
    df index: tickers
    df column: Value
    mode: "Percent (%)" | "Amount (EUR)" | "Weight (0-1)"
    returns: dict(ticker->weight) normalized sum=1, or None if unusable
    """
    if df is None or df.empty:
        return None

    x = df["Value"].copy()

    # coerce to numeric
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)

    # no negatives
    x = x.clip(lower=0.0)

    if mode == "Percent (%)":
        x = x / 100.0
    elif mode == "Amount (EUR)":
        # amounts -> weights by normalization below
        pass
    elif mode == "Weight (0-1)":
        # already weights, still normalize below
        pass

    s = float(x.sum())
    if s <= 0:
        return None

    w = (x / s).to_dict()
    return {str(k): float(v) for k, v in w.items()}


st.set_page_config(
    page_title="Financial Risk & Portfolio Optimizer",
    layout="wide",
)

# --------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    .main { background-color: #050816; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .card {
        background: #0b1020;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid #20263a;
        box-shadow: 0 0 20px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: #0b1020;
        border-radius: 18px;
        padding: 16px 18px;
        border: 1px solid #20263a;
        text-align: left;
    }
    .metric-label { font-size: 0.85rem; color: #8b9ac5; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f7f9ff; }
    .metric-sub { font-size: 0.8rem; color: #9aa6d4; }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e6ff;
        margin-bottom: 0.5rem;
    }
    .header-title { font-size: 1.6rem; font-weight: 700; color: #f7f9ff; margin-bottom: 0.2rem; }
    .header-sub { font-size: 0.95rem; color: #9aa6d4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="card" style="margin-bottom: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div class="header-title">📈 Financial Risk & Portfolio Optimizer</div>
          <div class="header-sub">Markowitz mean–variance backend with multi-agent LLM analysis</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- TOP ROW: CONTROLS + COMPOSITION + METRICS ----------------
col_left, col_mid, col_right = st.columns([1.2, 1.1, 1.1])

portfolio_weights = None
portfolio_metrics = None
optimization_result = None

current_weights_dict = None
current_metrics = None

baseline_weights_dict = None
baseline_metrics = None

optimized_metrics = None  # <-- for risk contributions (optimized)

# ---- LEFT: Portfolio Controls ----
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎛 Portfolio Controls</div>', unsafe_allow_html=True)

    all_tickers = load_available_tickers()

    selected_tickers = st.multiselect(
        "Universe (stocks to include)",
        options=all_tickers,
        default=all_tickers,
        help="These assets will be available to the optimization agent.",
    )

    # ---------------- Current Portfolio (optional) ----------------
    st.markdown(
        '<div class="section-title" style="margin-top:0.8rem;">🧾 Current Portfolio (optional)</div>',
        unsafe_allow_html=True
    )

    use_current = st.checkbox(
        "I have an existing portfolio (compare vs optimized)",
        value=False,
        help="Optional. If you fill this, the dashboard will compute Current risk/return and show it on the frontier.",
    )

    current_mode = None
    if use_current:
        current_mode = st.selectbox(
            "How do you want to enter your current portfolio?",
            ["Percent (%)", "Amount (EUR)", "Weight (0-1)"],
            index=0,
            help="Percent is easiest. Amount works if you know EUR invested per asset. Weight is the most technical option.",
        )

    if use_current and selected_tickers:
        # session init + alignment
        if "current_input_df" not in st.session_state:
            st.session_state["current_input_df"] = (
                pd.DataFrame({"Ticker": selected_tickers, "Value": [0.0] * len(selected_tickers)})
                .set_index("Ticker")
            )
        else:
            existing = st.session_state["current_input_df"]
            st.session_state["current_input_df"] = existing.reindex(selected_tickers).fillna(0.0)

        # column label/format per mode
        if current_mode == "Percent (%)":
            col_label = "Portfolio share (%)"
            step = 1.0
            fmt = "%.2f"
            help_txt = "Example: AAPL 4 means 4%."
        elif current_mode == "Amount (EUR)":
            col_label = "Invested amount (EUR)"
            step = 50.0
            fmt = "%.2f"
            help_txt = "Example: AAPL 500 means €500 invested. We convert to weights automatically."
        else:
            col_label = "Weight (0–1)"
            step = 0.01
            fmt = "%.4f"
            help_txt = "Technical. Example: 0.04 means 4%. We still normalize automatically."

        edited_df = st.data_editor(
            st.session_state["current_input_df"],
            num_rows="fixed",
            column_config={
                "Value": st.column_config.NumberColumn(
                    col_label,
                    help=help_txt,
                    min_value=0.0,
                    step=step,
                    format=fmt,
                )
            },
            width="stretch",
        )
        st.session_state["current_input_df"] = edited_df.copy()

        ca, cb = st.columns(2)
        with ca:
            if st.button("Normalize within selected assets, percentages rescaled to sum to 100%", width="stretch"):
                w_tmp = _safe_normalize_current_inputs(st.session_state["current_input_df"], current_mode)
                if w_tmp is None:
                    st.warning("Current portfolio is empty (sum=0). Fill at least one asset.")
                else:
                    w_series = pd.Series(w_tmp).reindex(selected_tickers).fillna(0.0)
                    if current_mode == "Percent (%)":
                        st.session_state["current_input_df"]["Value"] = (w_series * 100.0).values
                    elif current_mode == "Amount (EUR)":
                        st.info("For Amount mode, normalization is applied at run time (amounts kept as entered).")
                    else:
                        st.session_state["current_input_df"]["Value"] = w_series.values
                    st.success("Normalized representation updated ✓")
        with cb:
            if st.button("🔁 Reset equal allocation", width="stretch"):
                n = len(selected_tickers)
                if n > 0:
                    if current_mode == "Percent (%)":
                        st.session_state["current_input_df"]["Value"] = (100.0 / n)
                    elif current_mode == "Amount (EUR)":
                        st.session_state["current_input_df"]["Value"] = 100.0
                    else:
                        st.session_state["current_input_df"]["Value"] = (1.0 / n)
                    st.success("Reset ✓")

        raw_sum = float(pd.to_numeric(st.session_state["current_input_df"]["Value"], errors="coerce").fillna(0.0).sum())
        st.caption(f"Current input sum: **{raw_sum:.2f}** (we auto-normalize to weights when you run)")

        current_weights_dict = _safe_normalize_current_inputs(st.session_state["current_input_df"], current_mode)

    elif use_current and not selected_tickers:
        st.info("Select at least 1 ticker to enter current portfolio.")

    # ---------------- Optimizer parameters ----------------
    rf = st.number_input(
        "Risk-free rate (annual)",
        value=0.02,
        min_value=-0.05,
        max_value=0.20,
        step=0.005,
        format="%.3f",
        help="Used in Sharpe ratio: (Return - rf) / Volatility",
    )

    w_max = st.slider(
        "Max weight per asset",
        min_value=0.05,
        max_value=1.00,
        value=0.30,
        step=0.05,
    )

    objective_label = st.selectbox(
        "Optimization objective",
        ["Max Sharpe", "Min Variance"],
        index=0,
    )

    objective_key = "maxsharpe" if objective_label == "Max Sharpe" else "minvar"
    run_button = st.button("🚀 Run Optimization", width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RUN BACKEND IF CLICKED ----
if selected_tickers and run_button:
    # ✅ Baseline equal-weight (always available if universe exists)
    n = len(selected_tickers)
    if n > 0:
        baseline_weights_dict = {t: 1.0 / n for t in selected_tickers}
        try:
            baseline_metrics = risk_agent(baseline_weights_dict, selected_tickers)
        except Exception as e:
            st.warning(f"Could not compute baseline metrics: {e}")
            baseline_metrics = None

    # Current metrics (only if current weights are usable)
    if current_weights_dict is not None:
        try:
            current_metrics = risk_agent(current_weights_dict, selected_tickers)
        except Exception as e:
            st.warning(f"Could not compute current portfolio metrics: {e}")
            current_metrics = None
    else:
        current_metrics = None

    optimization_result = optimization_agent(
        selected_tickers,
        rf=rf,
        w_max=w_max,
        lambda_l2=1e-3,
    )

    port = optimization_result[objective_key]
    weights = pd.Series(port["weights"])
    weights = weights[weights.abs() > 1e-6].sort_values(ascending=False)

    portfolio_weights = weights
    portfolio_metrics = {
        "return": float(port["return"]),
        "vol": float(port["vol"]),
        "sharpe": port.get("sharpe", None),
        "used_assets": int(len(weights)),
        "universe_assets": int(len(selected_tickers)),
    }

    # ✅ Optimized risk contributions (rc_pct) via risk_agent
    try:
        optimized_metrics = risk_agent(weights.to_dict(), selected_tickers)
    except Exception as e:
        st.warning(f"Could not compute optimized risk contributions: {e}")
        optimized_metrics = None

# ---- MIDDLE: Donut Portfolio Composition ----
with col_mid:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧱 Portfolio Composition</div>', unsafe_allow_html=True)

    if portfolio_weights is None:
        st.info("Run the optimization to see the portfolio composition.")
    else:
        pie_df = portfolio_weights.reset_index()
        pie_df.columns = ["Ticker", "Weight"]

        fig = px.pie(
            pie_df,
            names="Ticker",
            values="Weight",
            hole=0.6,
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0b1020",
            plot_bgcolor="#0b1020",
            font=dict(color="#E2E6FF"),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RIGHT: Metrics ----
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Risk & Performance</div>', unsafe_allow_html=True)

    if portfolio_metrics is None:
        st.info("Metrics will appear here after running the optimization.")
    else:
        opt = portfolio_metrics

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Sharpe (Optimized)</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{opt["sharpe"]:.2f}</div>' if opt["sharpe"] is not None else
                '<div class="metric-value">–</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="metric-sub">Uses rf input</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Return (Optimized)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{opt["return"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Volatility (Optimized)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{opt["vol"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized std dev</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Assets in portfolio</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{opt["used_assets"]} / {opt["universe_assets"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="metric-sub">Active assets / selected universe</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if current_metrics is not None:
            st.markdown(
                '<div class="section-title" style="margin-top:0.9rem;">📎 Current Portfolio Metrics</div>',
                unsafe_allow_html=True
            )
            cc1, cc2, cc3 = st.columns(3)

            with cc1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Return (Current)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{current_metrics["return"]*100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-sub">Annualized</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with cc2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Volatility (Current)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{current_metrics["vol"]*100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-sub">Annualized std dev</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with cc3:
                r_c = float(current_metrics["return"])
                v_c = float(current_metrics["vol"])
                sharpe_c = (r_c - rf) / v_c if v_c > 0 else np.nan

                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Sharpe (Current)</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{sharpe_c:.2f}</div>' if np.isfinite(sharpe_c) else
                    '<div class="metric-value">–</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="metric-sub">(Return - rf) / Vol</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            # ✅ show baseline note (no extra metrics cards; just explain)
            st.markdown(
                '<div class="metric-sub" style="margin-top:0.9rem;">'
                'No current portfolio provided. We will show <b>Baseline (Equal Weight)</b> on the frontier and risk-contribution chart.'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Efficient Frontier ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Expected Return vs Risk (Efficient Frontier)</div>', unsafe_allow_html=True)

if (optimization_result is None) or (not optimization_result.get("frontier")):
    st.info("Run the optimization to visualize the efficient frontier.")
else:
    frontier_df = pd.DataFrame(optimization_result["frontier"])

    fig_frontier = px.line(
        frontier_df,
        x="vol",
        y="realized_return",
        markers=True,
    )

    opt = optimization_result[objective_key]

    # Optimized marker
    fig_frontier.add_trace(
        go.Scatter(
            x=[opt["vol"]],
            y=[opt["return"]],
            mode="markers+text",
            name="Optimized",
            text=["Optimized"],
            textposition="top left",
            marker=dict(size=10),
        )
    )

    # Current marker if available, else Baseline marker
    if current_metrics is not None:
        fig_frontier.add_trace(
            go.Scatter(
                x=[current_metrics["vol"]],
                y=[current_metrics["return"]],
                mode="markers+text",
                name="Current",
                text=["Current"],
                textposition="bottom right",
                marker=dict(size=10),
            )
        )
    elif baseline_metrics is not None:
        fig_frontier.add_trace(
            go.Scatter(
                x=[baseline_metrics["vol"]],
                y=[baseline_metrics["return"]],
                mode="markers+text",
                name="Baseline (Equal Weight)",
                text=["Baseline (Equal Weight)"],
                textposition="bottom right",
                marker=dict(size=10),
            )
        )

    fig_frontier.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Risk (Volatility, σ)",
        yaxis_title="Expected Return (µ)",
    )
    fig_frontier.update_xaxes(tickformat=".0%")
    fig_frontier.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig_frontier, width="stretch")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Risk Contribution by Asset ----------------
st.markdown("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Risk Contribution by Asset</div>', unsafe_allow_html=True)

if optimization_result is None or optimized_metrics is None:
    st.info("Run the optimization to see risk contributions.")
else:
    # base tickers (aligned from risk_agent)
    tickers_rc = optimized_metrics["tickers"]
    opt_rc = np.array(optimized_metrics["rc_pct"], dtype=float)

    df_rc = pd.DataFrame({"Ticker": tickers_rc, "Optimized": opt_rc})

    if current_metrics is not None:
        cur_rc = np.array(current_metrics["rc_pct"], dtype=float)
        df_rc["Current"] = cur_rc
        compare_label = "Current"
    else:
        # baseline fallback
        if baseline_metrics is not None:
            base_rc = np.array(baseline_metrics["rc_pct"], dtype=float)
            df_rc["Baseline (Equal Weight)"] = base_rc
            compare_label = "Baseline (Equal Weight)"
        else:
            compare_label = None

    # long format
    df_long = df_rc.melt(id_vars="Ticker", var_name="Portfolio", value_name="Risk Contribution")

    fig_rc = px.bar(
        df_long,
        x="Ticker",
        y="Risk Contribution",
        color="Portfolio",
        barmode="group",
    )
    fig_rc.update_layout(
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#E2E6FF"),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Risk contribution (share of total σ)",
        xaxis_title="",
        legend_title="",
    )
    fig_rc.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig_rc, width="stretch")

    if compare_label is not None:
        st.caption(f"Bars show each asset’s share of total portfolio risk (σ). Comparison portfolio: **{compare_label}** vs **Optimized**.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BOTTOM ROW: Weights Table + Explanation ----------------
st.markdown("")
bottom_left, bottom_right = st.columns([1.3, 1.0])

with bottom_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📉 Optimization Results (Weights)</div>', unsafe_allow_html=True)

    if portfolio_weights is None:
        st.info("No portfolio yet. Run the optimization first.")
    else:
        df_weights = portfolio_weights.to_frame("Weight")
        st.dataframe(
            df_weights.style.format("{:.3f}"),
            width="stretch",
            height=360,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Recommendation (Rule-based)</div>', unsafe_allow_html=True)

    if (portfolio_weights is None) or (optimization_result is None):
        st.info("Explanation will appear here after optimization.")
    else:
        # If you updated agents.py recommendation_agent to support current_metrics/rf, pass them here.
        # If not, you can keep the old call.
        try:
            explanation = recommendation_agent(
                optimization_result,
                objective="max_sharpe" if objective_key == "maxsharpe" else "min_var",
                current_metrics=current_metrics,
                rf=rf,
            )
        except TypeError:
            # fallback to old signature
            explanation = recommendation_agent(
                optimization_result,
                objective="max_sharpe" if objective_key == "maxsharpe" else "min_var",
            )

        st.write(explanation)

    st.markdown("</div>", unsafe_allow_html=True)
