"""
Figure: Component-Level Portfolio Impact by Feature Set
Uses new realized metrics (look-ahead bias fixed)
Run this from your project root.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── New realized metrics from component_level_impact_study.py ──
data = {
    "feature_sets": [
        "price_only\n(3)",
        "sentiment_only\n(7)",
        "news_only\n(11)",
        "sentiment_price\n(10)",
        "all_features\n(14)\n[production]",
    ],
    "sharpe_delta": [-0.053, +0.009, +0.053, +0.010, +0.087],
    "turnover":     [12.4,   10.0,    8.0,   12.3,   10.0],
}

colors = ["#2e75b6" if i < 4 else "#2e8b57"
          for i in range(len(data["feature_sets"]))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle(
    "Portfolio-Level Impact by Feature Set "
    "(101-ticker universe, production-equivalent signals)",
    fontsize=14,
    fontweight="bold"
)

# ── Left panel: Realized ΔSharpe ────────────────────────────
x = np.arange(len(data["feature_sets"]))
bars1 = ax1.bar(x, data["sharpe_delta"], color=colors,
                width=0.6, zorder=3)
ax1.axhline(0, color="black", linewidth=1.0,
            linestyle="--", label="Baseline (ΔS = 0)", zorder=4)
ax1.set_title("Realized ΔSharpe by Feature Set\n"
              "(higher = better)", fontsize=12)
ax1.set_ylabel("ΔSharpe (vs. unconstrained baseline)", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(data["feature_sets"], fontsize=10)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.legend(fontsize=11)

for bar, val in zip(bars1, data["sharpe_delta"]):
    ypos = bar.get_height() + 0.001 if val >= 0 else bar.get_height() - 0.005
    ax1.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{val:+.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Right panel: Turnover ────────────────────────────────────
bars2 = ax2.bar(x, data["turnover"], color=colors,
                width=0.6, zorder=3)
ax2.set_title("Portfolio Turnover by Feature Set\n"
              "(lower = more stable)", fontsize=12)
ax2.set_ylabel("Portfolio Turnover (%)", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(data["feature_sets"], fontsize=10)
ax2.grid(axis="y", alpha=0.3, zorder=0)

for bar, val in zip(bars2, data["turnover"]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Legend ───────────────────────────────────────────────────
blue_patch  = mpatches.Patch(color="#2e75b6", label="Ablation configs")
green_patch = mpatches.Patch(color="#2e8b57", label="Production (all_features)")
fig.legend(handles=[blue_patch, green_patch],
           loc="lower center", ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1])

out = "figure7_feature_group_ablation.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")