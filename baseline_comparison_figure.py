"""
Figure: Baseline Comparison (2026-01-15 to 2026-05-22)
✅ New realized metrics (no look-ahead bias)
Run this from your project root.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── New realized metrics from baseline_comparison_realized.py ──
methods = [
    "Equal\nWeight",
    "Plain\nMVO\n(no news)",
    "Zhang\n(2022)\nLong-Short",
    "BL+FinBERT\n(Colasanto\n2022)",
    "NC-MVO\n[Ours]",
]

sharpes  = [1.26,  1.05,  0.99,  0.80,  1.14]
returns  = [23.02, 20.32, 17.98, 11.27, 21.44]
vols     = [16.65, 17.50, 16.07, 11.58, 17.01]

# Colors: grey for baselines, green for ours
colors = ["#aaaaaa", "#5a9bd4", "#aaaaaa", "#e07b00", "#2e8b57"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5.5))
fig.suptitle("Baseline Comparison (2026-01-15 to 2026-05-22)",
             fontsize=12, fontweight="bold")

x = np.arange(len(methods))
width = 0.6

# ── Panel 1: Sharpe ─────────────────────────────────────────
bars1 = ax1.bar(x, sharpes, color=colors, width=width, zorder=3)
ax1.set_title("Sharpe Ratio", fontsize=10)
ax1.set_ylabel("Sharpe")
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=7.5)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.set_ylim(0, max(sharpes) * 1.25)
for bar, val in zip(bars1, sharpes):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{val:.2f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold")

# ── Panel 2: Return ─────────────────────────────────────────
bars2 = ax2.bar(x, returns, color=colors, width=width, zorder=3)
ax2.set_title("Annualized Return (%)", fontsize=10)
ax2.set_ylabel("Return (%)")
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=7.5)
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.set_ylim(0, max(returns) * 1.25)
for bar, val in zip(bars2, returns):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             f"{val:.2f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold")

# ── Panel 3: Volatility ──────────────────────────────────────
bars3 = ax3.bar(x, vols, color=colors, width=width, zorder=3)
ax3.set_title("Annualized Volatility (%)", fontsize=10)
ax3.set_ylabel("Volatility (%)")
ax3.set_xticks(x)
ax3.set_xticklabels(methods, fontsize=7.5)
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.set_ylim(0, max(vols) * 1.35)
for bar, val in zip(bars3, vols):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             f"{val:.2f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold")

# ── Legend ───────────────────────────────────────────────────
green_patch = mpatches.Patch(color="#2e8b57", label="NC-MVO [Ours]")
blue_patch  = mpatches.Patch(color="#5a9bd4", label="Plain MVO baseline")
orange_patch= mpatches.Patch(color="#e07b00", label="BL+FinBERT baseline")
grey_patch  = mpatches.Patch(color="#aaaaaa", label="Other baselines")
fig.legend(handles=[green_patch, blue_patch, orange_patch, grey_patch],
           loc="lower center", ncol=4,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])

out = "figure10_baseline_comparison.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")