"""
Figure: Baseline Comparison (2026-01-15 to 2026-05-22)
New realized metrics without look-ahead bias.

Run this script from the project root.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── Realized metrics from baseline_comparison_realized.py ────
methods = [
    "Equal\nWeight",
    "Plain\nMVO\n(no news)",
    "Zhang\n(2022)\nLong-Short",
    "BL+FinBERT\n(Colasanto\n2022)",
    "NC-MVO\n[Ours]",
]

sharpes = [1.26, 1.05, 0.99, 0.80, 1.13]
returns = [23.02, 20.32, 17.98, 11.27, 21.30]
vols = [16.65, 17.50, 16.07, 11.58, 17.02]

# Grey for other baselines, blue for plain MVO,
# orange for BL+FinBERT, and green for the proposed method.
colors = ["#aaaaaa", "#5a9bd4", "#aaaaaa", "#e07b00", "#2e8b57"]

x = np.arange(len(methods))
bar_width = 0.62


# ── Figure setup ─────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(
    1,
    3,
    figsize=(18, 7.5),
)

fig.suptitle(
    "Baseline Comparison (2026-01-15 to 2026-05-22)",
    fontsize=17,
    fontweight="bold",
    y=0.98,
)


def configure_axis(ax, title, ylabel, upper_limit):
    """Apply consistent formatting to one subplot."""
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=10)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, linespacing=1.05)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", pad=6)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, upper_limit)


def add_bar_labels(ax, bars, values, offset):
    """Add bold numerical values above bars."""
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


# ── Panel 1: Sharpe ratio ────────────────────────────────────
bars1 = ax1.bar(
    x,
    sharpes,
    color=colors,
    width=bar_width,
    zorder=3,
)

configure_axis(
    ax=ax1,
    title="Sharpe Ratio",
    ylabel="Sharpe",
    upper_limit=max(sharpes) * 1.28,
)

add_bar_labels(
    ax=ax1,
    bars=bars1,
    values=sharpes,
    offset=0.015,
)


# ── Panel 2: Annualized return ───────────────────────────────
bars2 = ax2.bar(
    x,
    returns,
    color=colors,
    width=bar_width,
    zorder=3,
)

configure_axis(
    ax=ax2,
    title="Annualized Return (%)",
    ylabel="Return (%)",
    upper_limit=max(returns) * 1.28,
)

add_bar_labels(
    ax=ax2,
    bars=bars2,
    values=returns,
    offset=0.30,
)


# ── Panel 3: Annualized volatility ───────────────────────────
bars3 = ax3.bar(
    x,
    vols,
    color=colors,
    width=bar_width,
    zorder=3,
)

configure_axis(
    ax=ax3,
    title="Annualized Volatility (%)",
    ylabel="Volatility (%)",
    upper_limit=max(vols) * 1.32,
)

add_bar_labels(
    ax=ax3,
    bars=bars3,
    values=vols,
    offset=0.22,
)


# ── Legend ───────────────────────────────────────────────────
green_patch = mpatches.Patch(
    color="#2e8b57",
    label="NC-MVO [Ours]",
)

blue_patch = mpatches.Patch(
    color="#5a9bd4",
    label="Plain MVO baseline",
)

orange_patch = mpatches.Patch(
    color="#e07b00",
    label="BL+FinBERT baseline",
)

grey_patch = mpatches.Patch(
    color="#aaaaaa",
    label="Other baselines",
)

fig.legend(
    handles=[
        green_patch,
        blue_patch,
        orange_patch,
        grey_patch,
    ],
    loc="lower center",
    ncol=4,
    fontsize=15,
    bbox_to_anchor=(0.5, 0.03),
    frameon=True,
)


# Leave enough room for the main title and bottom legend.
plt.tight_layout(rect=[0.02, 0.11, 0.98, 0.94])

out = "figure10_baseline_comparison.png"

plt.savefig(
    out,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved → {out}")