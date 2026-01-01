import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


from agents import optimization_agent, recommendation_agent

DATA_DIR = Path("data/processed_yahoo")


@st.cache_data
def load_available_tickers():
    summary = pd.read_csv(DATA_DIR / "summary_per_asset_annual.csv", index_col=0)
    return list(summary.index)


st.set_page_config(
    page_title="Financial Risk & Portfolio Optimizer",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #050816;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
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
    .metric-label {
        font-size: 0.85rem;
        color: #8b9ac5;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f7f9ff;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #9aa6d4;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e6ff;
        margin-bottom: 0.5rem;
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f7f9ff;
        margin-bottom: 0.2rem;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #9aa6d4;
    }
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
optimization_result = None  # efficient frontier vs. için de lazım

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
    

    rf = st.number_input(
        "Risk-free rate (annual)",
        value=0.02,
        min_value=-0.05,
        max_value=0.20,
        step=0.005,
        format="%.3f",
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

    if objective_label == "Max Sharpe":
        objective_key = "maxsharpe"
    else:
        objective_key = "minvar"

    run_button = st.button("🚀 Run Optimization", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RUN BACKEND IF CLICKED ----
if selected_tickers and run_button:
    optimization_result = optimization_agent(
        selected_tickers,
        rf=rf,
        w_max=w_max,
        lambda_l2=1e-3,
    )

    port = optimization_result[objective_key]
    weights = pd.Series(port["weights"])
    # sadece sıfır olmayanları al, büyükten küçüğe sırala
    weights = weights[weights.abs() > 1e-6].sort_values(ascending=False)

    portfolio_weights = weights
    portfolio_metrics = {
        "return": port["return"],
        "vol": port["vol"],
        "sharpe": port.get("sharpe", None),
        # aktif / seçilen varlık sayıları
        "used_assets": len(weights),
        "universe_assets": len(selected_tickers),
    }

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
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RIGHT: Metrics ----
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Risk & Performance</div>', unsafe_allow_html=True)

    if portfolio_metrics is None:
        st.info("Metrics will appear here after running the optimization.")
    else:
        m = portfolio_metrics
        c1, c2 = st.columns(2)
        with c1:
            # Sharpe
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Sharpe Ratio</div>', unsafe_allow_html=True)
            if m["sharpe"] is not None:
                st.markdown(f'<div class="metric-value">{m["sharpe"]:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-value">–</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Risk-adjusted return</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Expected return
            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Expected Return</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{m["return"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            # Vol
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Volatility</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{m["vol"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Annualized standard deviation</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Assets count
            st.markdown('<div class="metric-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Assets in portfolio</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{m["used_assets"]} / {m["universe_assets"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-sub">Active assets (non-zero weights) / stocks selected above</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MIDDLE SECTION: Expected Return vs Risk (Frontier) ----------------
st.markdown("")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Expected Return vs Risk (Efficient Frontier)</div>', unsafe_allow_html=True)

if (optimization_result is None) or (not optimization_result["frontier"]):
    st.info("Run the optimization to visualize the efficient frontier.")
else:
    frontier_df = pd.DataFrame(optimization_result["frontier"])
    # frontier_df: vol (x), realized_return (y)

    fig_frontier = px.line(
        frontier_df,
        x="vol",
        y="realized_return",
        markers=True,
    )

    minvar = optimization_result["minvar"]
    opt = optimization_result[objective_key]

    fig_frontier.add_trace(
        go.Scatter(
            x=[minvar["vol"]],
            y=[minvar["return"]],
            mode="markers+text",
            name="Current (min variance)",
            text=["Current"],
            textposition="bottom right",
            marker=dict(size=10),
        )
    )

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

    st.plotly_chart(fig_frontier, use_container_width=True)

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
            use_container_width=True,
            height=360,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 LLM Recommendation (Prototype)</div>', unsafe_allow_html=True)

    if (portfolio_weights is None) or (optimization_result is None):
        st.info("Explanation will appear here after optimization.")
    else:
        explanation = recommendation_agent(
            optimization_result,
            objective="max_sharpe" if objective_key == "maxsharpe" else "min_var",
        )
        st.write(explanation)

    st.markdown("</div>", unsafe_allow_html=True)
