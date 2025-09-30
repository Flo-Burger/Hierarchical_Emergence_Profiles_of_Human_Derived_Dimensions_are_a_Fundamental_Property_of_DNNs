#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlexNet line comparison (callable).

Loads per-layer R² CSVs for:
    - AlexNet (pretrained)
    - AlexNet (untrained)
Aligns by numeric layer index (min number of layers across models),
computes mean R² per layer per model, normalizes those means to [0,1]
over the union (so one curve hits 1.0), prints Pearson r on aligned
full R²s, and saves the plot to:
    Results/final_plots/<DATASET>_<PCA>_<REG>/compare_<DATASET>_<COMBO>_alexnet.png
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
    import scienceplots  # optional
    plt.style.use(['science', 'nature'])
except Exception:
    pass

sns.set_context('paper', font_scale=1.2)
sns.set_style('ticks')
plt.rcParams.update({
    'font.family': 'arial',
    'pdf.fonttype': 42,
})

# ── Roots like your analysis ───────────────────────────────────────────────
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
) -> Path:
    """
    Render AlexNet vs Untrained line plot and return PNG path.
    """
    results_root = Path(results_root).resolve() if results_root else RESULTS_ROOT_DEFAULT.resolve()
    plots_dir = results_root / "final_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if display_labels is None:
        display_labels = {
            "AlexNet":          "AlexNet (pretrained)",
            "AlexNetUntrained": "AlexNet (untrained)",
        }

    # Load CSVs
    dfs: dict[str, pd.DataFrame] = {}
    for lbl in labels:
        p = _r2_path(results_root, lbl, dataset, combo)
        if not p.exists():
            raise FileNotFoundError(f"Cannot find file: {p}")
        dfs[lbl] = pd.read_csv(p, index_col=0)

    # Check dims match
    dims0 = dfs[labels[0]].index.tolist()
    for lbl in labels[1:]:
        if dfs[lbl].index.tolist() != dims0:
            raise ValueError(f"Dimension index mismatch in {lbl}")

    # Align by numeric layer index
    n_layers = min(df.shape[1] for df in dfs.values())
    for lbl in labels:
        df = dfs[lbl].iloc[:, :n_layers].copy()
        df.columns = list(range(1, n_layers + 1))  # rename to 1..n
        dfs[lbl] = df

    layers = dfs[labels[0]].columns.tolist()

    # Mean per layer & normalize over union
    df_mean = pd.DataFrame({lbl: df.mean(axis=0) for lbl, df in dfs.items()}, index=layers)
    mn, mx = df_mean.values.min(), df_mean.values.max()
    span = (mx - mn) if (mx > mn) else 1.0
    df_mean = (df_mean - mn) / span

    # Plot
    x = np.arange(1, len(layers) + 1)
    fig, ax = plt.subplots(figsize=(3.33, 2.5))
    for lbl in labels:
        disp = display_labels.get(lbl, lbl)
        ax.plot(x, df_mean[lbl].values, linewidth=2.0, label=disp)

    title = ("DNN Architecture Shapes Material Dimension Encoding"
             if dataset == "STUFF" else
             "DNN Architecture Shapes Object Dimension Encoding")
    ax.set_title(title, fontsize=10, pad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=6)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_ylabel("Normalized Mean R$^2$", fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(frameon=False, prop={'size': 5}, loc='upper left')

    fig.tight_layout()
    # Subfolder by dataset/combo pattern used elsewhere
    pca_tag = combo.split('_')[0].replace('PCA', '')
    reg_tag = combo.split('_', 1)[1] if '_' in combo else "Linear"
    subdir = results_root / "final_plots" / f"{dataset}_{pca_tag}_{reg_tag}"
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"compare_{dataset}_{combo}_alexnet.png"

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"✔ Saved AlexNet lines: {out_path}")
    return out_path

if __name__ == "__main__":
    plot_alexnet_lines(dataset="THINGS", combo="PCA100_Linear")
