"""
Figure 6: Efficient Frontier (101-ticker universe)

Run this from your project root.
"""

import matplotlib.pyplot as plt

from agents_langgraph import data_agent_get_mu_cov
from portfolio_core import run_portfolio_optimization

# --- 1) Full 101-ticker universe ---
TICKERS = [
    "EXC","FAST","CDNS","MRVL","MSFT","MSTR","MU","CPRT","CSGP","CSX","CTAS","CTSH",
    "GILD","GOOGL","MNST","IDXX","INTC","LRCX","MAR","AMZN","ASML","AAPL","ADBE",
    "ADI","ADP","ADSK","AEP","AZN","BKR","BIIB","AMAT","AMD","AMGN","PANW","KHC",
    "PYPL","SHOP","CCEP","AVGO","TEAM","TTD","GOOG","CDW","DXCM","CSCO","PCAR",
    "BKNG","PEP","TSLA","NXPI","KDP","META","WDAY","FANG","CHTR","VRSK","FTNT",
    "VRTX","LULU","REGN","ROP","ROST","SBUX","AXON","MELI","TMUS","NVDA","ODFL",
    "ON","ORLY","PAYX","TRI","XEL","TTWO","TXN","SNPS","QCOM","MDLZ","KLAC","EA",
    "INTU","ISRG","MCHP","HON","NFLX","CMCSA","COST","ABNB","DASH","APP","WBD",
    "DDOG","ZS","GFS","CEG","CRWD","PLTR","PDD","LIN","ARM","GEHC",
]

# --- 2) Load mu/cov via production data agent ---
mu, cov = data_agent_get_mu_cov(TICKERS)
print(f"Universe size: {len(mu)} tickers")

# --- 3) Run optimization (production function) ---
result = run_portfolio_optimization(
    mu=mu, cov=cov, rf=0.02, w_max=0.30, lambda_l2=1e-3, save_csv=False
)

frontier = result["frontier"]    # list of {target_return, realized_return, vol}
maxsharpe = result["maxsharpe"]  # {"return":..., "vol":..., "sharpe":...}
minvar    = result["minvar"]     # {"return":..., "vol":...}

frontier_vol = [p["vol"] for p in frontier]
frontier_ret = [p["realized_return"] for p in frontier]

# --- 4) Plot ---
fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.plot(frontier_vol, frontier_ret, color="#2e75b6", linewidth=2.2,
        label="Efficient Frontier", zorder=2)

ax.scatter(maxsharpe["vol"], maxsharpe["return"],
           color="#2e8b57", s=140, marker="*", zorder=5,
           label=f"Max Sharpe (S={maxsharpe['sharpe']:.2f})")

ax.scatter(minvar["vol"], minvar["return"],
           color="#c00000", s=80, marker="o", zorder=5,
           label="Min Variance")

ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")
ax.set_title(f"Efficient Frontier ({len(mu)}-ticker NASDAQ universe)",
              fontsize=12, fontweight="bold")

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figure6_efficient_frontier.png", dpi=200, bbox_inches="tight")
print("saved -> figure6_efficient_frontier.png")