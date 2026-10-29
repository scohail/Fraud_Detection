"""Plotting helpers shared by the notebooks (matplotlib / seaborn)."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Validated colour-blind-safe categorical palette, used in a fixed order.
LEGIT_COLOR = "#2a78d6"   # blue
FRAUD_COLOR = "#eb6834"   # orange
FAMILY_COLORS = {
    "none": "#8a8984",          # neutral grey for the baseline
    "cost-sensitive": "#6250d6",
    "over": "#2a78d6",
    "under": "#eb6834",
    "hybrid": "#1baf7a",
}
CLASS_PALETTE = {0: LEGIT_COLOR, 1: FRAUD_COLOR}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def set_style():
    """Light, recessive axes and grid so the data carries the contrast."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def plot_confusion_matrix(y_true, y_pred, ax=None, title="Confusion matrix"):
    """Confusion matrix with raw counts and the row percentage of each cell."""
    ax = ax or plt.gca()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{c:,}\n({p:.1f} %)" for c, p in zip(r_c, r_p)]
                       for r_c, r_p in zip(cm, row_pct)])
    sns.heatmap(row_pct, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=100, cbar=False,
                linewidths=2, linecolor="white", ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_2d_projection(X2d, y, ax=None, title="", max_legit=3000, random_state=0):
    """Scatter of a 2-D projection; frauds are drawn on top of a legit subsample."""
    ax = ax or plt.gca()
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    legit = np.flatnonzero(y == 0)
    if len(legit) > max_legit:
        legit = rng.choice(legit, max_legit, replace=False)
    fraud = np.flatnonzero(y == 1)
    ax.scatter(X2d[legit, 0], X2d[legit, 1], s=8, alpha=0.35, c=LEGIT_COLOR,
               label=f"Legit ({(y == 0).sum():,})", linewidths=0)
    ax.scatter(X2d[fraud, 0], X2d[fraud, 1], s=10, alpha=0.6, c=FRAUD_COLOR,
               label=f"Fraud ({(y == 1).sum():,})", edgecolors="white", linewidths=0.3)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=8, markerscale=1.5)
    return ax
