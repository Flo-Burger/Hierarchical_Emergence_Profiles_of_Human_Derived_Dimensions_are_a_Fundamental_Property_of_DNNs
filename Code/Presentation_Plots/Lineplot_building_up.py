#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlexNet line comparison (callable) with three plots:
1) First layer only (untrained), SAME axes as others, just one dot
2) Only untrained line
3) Trained vs untrained lines
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ── Styling ────────────────────────────────────────────────────────────────
try:
    import scienceplots
    plt.style.use(['science', 'nature'])
except Exception:
    pass

sns.set_context('paper', font_scale=1.2)
sns.set_style('ticks')
plt.rcParams.update({
    'font.family': 'arial',
    'pdf.fonttype': 42,
})

# ── Roots ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT_DEFAULT = PROJECT_ROOT / "Results"

def _r2_path(root: Path, model: str, dataset: str, combo: str) -> Path:
    return root / f"{model}_{dataset}" / combo / f"r2_{model}_{dataset}_{combo}_layers.csv"

def plot_alexnet_lines(
    dataset: str,
    combo: str,
    results_root: str | Path | None = None,
    labels: tuple[str, str] = ("AlexNet", "AlexNetUntrained"),
    display_labels: dict[str, str] | None = None,
    dpi: int = 600,
) -> tuple[Path, Path, Path]:
    
    results_root = Path(results_root).resolve() if results_root else RESULTS_ROOT_DEFAULT.resolve()
    plots_dir = results_root / "final_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if display_labels is None:
        display_labels = {
            "AlexNet": "Trained",
            "AlexNetUntrained": "Untrained",
        }

    # Load CSVs
    dfs = {}
    for lbl in labels:
        p = _r2_path(results_root, lbl, dataset, combo)
        if not p.exists():
            raise FileNotFoundError(f"Cannot find file: {p}")
        dfs[lbl] = pd.read_csv(p, index_col=0)

    # Align dims
    dims0 = dfs[labels[0]].index.tolist()
    for lbl in labels[1:]:
        if dfs[lbl].index.tolist() != dims0:
            raise ValueError(f"Dimension index mismatch in {lbl}")

    # Align layer count + rename to 1..n
    n_layers = min(df.shape[1] for df in dfs.values())
    for lbl in labels:
        df = dfs[lbl].iloc[:, :n_layers].copy()
        df.columns = list(range(1, n_layers + 1))
        dfs[lbl] = df

    layers = dfs[labels[0]].columns.tolist()

    # Mean R2 and normalize across both models
    df_mean = pd.DataFrame({lbl: df.mean(axis=0) for lbl, df in dfs.items()}, index=layers)
    mn, mx = df_mean.values.min(), df_mean.values.max()
    df_mean = (df_mean - mn) / (mx - mn if mx > mn else 1.0)

    untrained = "AlexNetUntrained"
    trained = "AlexNet"

    # Colors
    trained_color = "tab:blue"
    untrained_color = "tab:green"

    # ───────────────────────────────────────────────
    # 1) First layer only, but same figure frame
    # ───────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(3.33, 2.5))

    # add invisible line to force axis identical
    ax1.plot(layers, [np.nan]*len(layers), alpha=0)

    # tiny dot only
    ax1.scatter([1], [df_mean[untrained].values[0]],
                color=untrained_color, s=30)

    ax1.set_title("Emergence Profile", fontsize=10, pad=8)
    ax1.set_xticks(layers)
    ax1.set_xlabel("Layer", fontsize=8)
    ax1.set_ylabel("Normalized Mean R$^2$", fontsize=8)
    ax1.set_ylim(0, 1)
    fig1.tight_layout()

    p1 = plots_dir / f"{dataset}_{combo}_untrained_first_layer.png"
    fig1.savefig(p1, dpi=dpi)
    plt.close(fig1)

    # ───────────────────────────────────────────────
    # 2) Untrained only line
    # ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(3.33, 2.5))
    ax2.plot(layers, df_mean[untrained].values, linewidth=2,
             label="Untrained", color=untrained_color)

    ax2.set_title("Emergence Profile", fontsize=10, pad=8)
    ax2.set_xticks(layers)
    ax2.set_xlabel("Layer", fontsize=8)
    ax2.set_ylabel("Normalized Mean R$^2$", fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, prop={'size': 6}, loc='upper left')


    fig2.tight_layout()
    p2 = plots_dir / f"{dataset}_{combo}_untrained_only.png"
    fig2.savefig(p2, dpi=dpi)
    plt.close(fig2)

    # ───────────────────────────────────────────────
    # 3) Trained vs untrained
    # ───────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(3.33, 2.5))
    ax3.plot(layers, df_mean[trained].values, linewidth=2,
             label="Trained", color=trained_color)
    ax3.plot(layers, df_mean[untrained].values, linewidth=2,
             label="Untrained", color=untrained_color)

    ax3.set_title("Emergence Profile", fontsize=10, pad=8)
    ax3.set_xticks(layers)
    ax3.set_xlabel("Layer", fontsize=8)
    ax3.set_ylabel("Normalized Mean R$^2$", fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.legend(frameon=False, prop={'size': 6})

    fig3.tight_layout()
    p3 = plots_dir / f"{dataset}_{combo}_trained_vs_untrained.png"
    fig3.savefig(p3, dpi=dpi)
    plt.close(fig3)

    print("Saved:")
    print(p1)
    print(p2)
    print(p3)

    return p1, p2, p3


if __name__ == "__main__":
    plot_alexnet_lines(dataset="THINGS", combo="PCA100_Linear")
