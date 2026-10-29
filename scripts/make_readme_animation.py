"""Generate assets/resampling.gif: SMOTE vs random under-sampling, step by step.

Usage (from the project root):
    python scripts/make_readme_animation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "assets" / "resampling.gif"

LEGIT, FRAUD, SYNTH, REMOVED = "#2a78d6", "#eb6834", "#1baf7a", "#dcdbd6"
SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

rng = np.random.default_rng(7)
N_LEGIT, N_FRAUD = 260, 18
legit = rng.normal([0, 0], [1.1, 0.9], size=(N_LEGIT, 2))
fraud = np.vstack([rng.normal([2.2, 1.9], [0.9, 0.75], size=(12, 2)),
                   rng.normal([-2.3, 2.3], [0.6, 0.5], size=(6, 2))])

# --- SMOTE: interpolate between a fraud and one of its k nearest fraud neighbours
K = 4
dists = np.linalg.norm(fraud[:, None] - fraud[None], axis=2)
neighbours = np.argsort(dists, axis=1)[:, 1:K + 1]
n_new = N_LEGIT - N_FRAUD
parents = rng.integers(0, N_FRAUD, n_new)
partners = neighbours[parents, rng.integers(0, K, n_new)]
lam = rng.random(n_new)[:, None]
synthetic = fraud[parents] + lam * (fraud[partners] - fraud[parents])

# --- Random under-sampling: drop legit points until both classes have N_FRAUD rows
removal_order = rng.permutation(N_LEGIT)[: N_LEGIT - N_FRAUD]

INTRO, STEPS, OUTRO = 10, 44, 22
FRAMES = INTRO + STEPS + OUTRO
per_step_new = int(np.ceil(n_new / STEPS))
per_step_removed = int(np.ceil(len(removal_order) / STEPS))

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=90, facecolor=SURFACE)
fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.1, wspace=0.06)


def setup(ax, title):
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-4, 4.6)
    ax.set_ylim(-3.6, 4.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#e4e3df")
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=8)


def counter(ax, n_legit, n_fraud):
    ax.text(0.5, 0.03, f"{n_legit} legitimate  ·  {n_fraud} fraud", transform=ax.transAxes,
            ha="center", va="bottom", color=TEXT, fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="#e4e3df"))


def draw(frame):
    step = min(max(frame - INTRO, 0), STEPS)

    # Left: SMOTE
    ax = axes[0]
    setup(ax, "Over-sampling · SMOTE")
    shown = min(step * per_step_new, n_new)
    ax.scatter(*legit.T, s=12, c=LEGIT, alpha=0.45, linewidths=0)
    if shown:
        ax.scatter(*synthetic[:shown].T, s=12, c=SYNTH, alpha=0.7, linewidths=0, zorder=4)
    if 0 < step < STEPS:  # draw the segments used for the newest synthetic points
        for i in range(max(0, shown - per_step_new), shown):
            a, b = fraud[parents[i]], fraud[partners[i]]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=SYNTH, lw=1.2, alpha=0.8, zorder=5)
    ax.scatter(*fraud.T, s=34, c=FRAUD, edgecolors="white", linewidths=0.8, zorder=3)
    counter(ax, N_LEGIT, N_FRAUD + shown)

    # Right: random under-sampling
    ax = axes[1]
    setup(ax, "Under-sampling · Random")
    removed = removal_order[: min(step * per_step_removed, len(removal_order))]
    keep = np.ones(N_LEGIT, bool)
    keep[removed] = False
    ax.scatter(*legit[~keep].T, s=12, c=REMOVED, alpha=0.6, linewidths=0)
    ax.scatter(*legit[keep].T, s=14 if step == STEPS else 12, c=LEGIT, alpha=0.75, linewidths=0)
    ax.scatter(*fraud.T, s=42, c=FRAUD, edgecolors="white", linewidths=0.8, zorder=3)
    counter(ax, int(keep.sum()), N_FRAUD)

    phase = ("An imbalanced training set" if frame < INTRO else
             "Rebalancing the training data..." if step < STEPS else
             "Both classes now have the same size")
    fig.texts.clear()
    fig.text(0.5, 0.94, phase, ha="center", color=TEXT, fontsize=13)
    for x, color, label in [(0.27, LEGIT, "legitimate"), (0.42, FRAUD, "fraud"),
                            (0.53, SYNTH, "synthetic fraud"), (0.71, REMOVED, "removed")]:
        fig.text(x, 0.035, "●", color=color, fontsize=13, ha="right", va="center")
        fig.text(x + 0.008, 0.035, label, color=MUTED, fontsize=10, va="center")


if __name__ == "__main__":
    OUTPUT.parent.mkdir(exist_ok=True)
    FuncAnimation(fig, draw, frames=FRAMES).save(OUTPUT, writer=PillowWriter(fps=11))
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
