#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORnet line comparison (callable, AlexNet-lines style).

Models:
  - FeedforwardCORnet
  - RecurrentCORnet
  - SkipCORnet

Alignment:
  - Try name intersection across all three (keep Feedforward order).
  - If empty, fall back to positional pairing using the minimum length.

Computes mean R² per (aligned) layer for each model, normalizes means to [0,1]
over the union, prints pairwise Pearson r on the aligned full R²s (all cells),
and saves to:
  Results/final_plots/<DATASET>_<PCA>_<REG>/compare_<DATASET>_<COMBO>_cornet.png
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ── Styling (match the rest) ───────────────────────────────────────────────
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

def _align_layers_3(df_ff: pd.DataFrame, df_rec: pd.DataFrame, df_skip: pd.DataFrame) -> Tuple[list, list, list]:
    """Prefer name intersection (keep Feedforward order). Fallback to positional."""
    base = list(df_ff.columns)
    common = [L for L in base if (L in df_rec.columns) and (L in df_skip.columns)]
    if common:
        return common, common, common
    n = min(df_ff.shape[1], df_rec.shape[1], df_skip.shape[1])
    if n == 0:
        raise ValueError("No layers to align across Feedforward/Recurrent/Skip CORnet.")
    return base[:n], list(df_rec.columns)[:n], list(df_skip.columns)[:n]

def plot_cornet_lines(
    dataset: str,
    combo: str,
    results_root: str | Path | None = None,
    models: Tuple[str, str, str] = ("FeedforwardCORnet", "RecurrentCORnet", "SkipCORnet"),
    display_labels: Dict[str, str] | None = None,
    dpi: int = 600,
) -> Path:
    """
    Render CORnet line plot and return PNG path.

    Parameters
    ----------
    dataset : "THINGS" | "STUFF"
    combo   : e.g. "PCA100_Linear", "PCA0.95_SVR"
    results_root : base Results directory (defaults to ../../Results from this file)
    models : tuple of the three model folder stems (must match generator output)
    display_labels : optional mapping from model id -> display name
    dpi : figure DPI
    """
    # Resolve Results/
    results_root = Path(results_root).resolve() if results_root else RESULTS_ROOT_DEFAULT.resolve()

    # Load CSVs
    m_ff, m_rec, m_skip = models
    p_ff   = _r2_path(results_root, m_ff,   dataset, combo)
    p_rec  = _r2_path(results_root, m_rec,  dataset, combo)
    p_skip = _r2_path(results_root, m_skip, dataset, combo)
    if not p_ff.exists():  raise FileNotFoundError(f"Cannot find file: {p_ff}")
    if not p_rec.exists(): raise FileNotFoundError(f"Cannot find file: {p_rec}")
    if not p_skip.exists():raise FileNotFoundError(f"Cannot find file: {p_skip}")

    df_ff   = pd.read_csv(p_ff,   index_col=0)  # dims × layers
    df_rec  = pd.read_csv(p_rec,  index_col=0)
    df_skip = pd.read_csv(p_skip, index_col=0)

    # Dimension checks (must match the generator)
    idx = df_ff.index
    if not (idx.equals(df_rec.index) and idx.equals(df_skip.index)):
        raise ValueError("Dimension index mismatch across CORnet models.")

    # Align layers (name-intersection or positional)
    Lff, Lrec, Lskip = _align_layers_3(df_ff, df_rec, df_skip)
    df_ff   = df_ff[Lff].copy()
    df_rec  = df_rec[Lrec].copy()
    df_skip = df_skip[Lskip].copy()

    # Numeric x-axis (1..n)
    n_layers = df_ff.shape[1]
    layers = list(range(1, n_layers + 1))
    for df in (df_ff, df_rec, df_skip):
        df.columns = layers

    # Mean per layer & normalize across the union to [0,1]
    means = {
        m_ff:   df_ff.mean(axis=0),
        m_rec:  df_rec.mean(axis=0),
        m_skip: df_skip.mean(axis=0),
    }
    df_mean = pd.DataFrame(means, index=layers)
    mn, mx = df_mean.values.min(), df_mean.values.max()
    span = (mx - mn) if (mx > mn) else 1.0
    df_mean = (df_mean - mn) / span

    # Display labels
    if display_labels is None:
        display_labels = {
            "FeedforwardCORnet": "Feedforward",
            "RecurrentCORnet":   "Recurrent",
            "SkipCORnet":        "Skip",
        }

    # Plot
    x = np.arange(1, n_layers + 1)
    fig, ax = plt.subplots(figsize=(3.33, 2.5))
    for m in (m_ff, m_rec, m_skip):
        ax.plot(x, df_mean[m].values, linewidth=2.0, label=display_labels.get(m, m))

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

    # Save under the same structured subfolder as your other figures
    pca_tag = combo.split('_')[0].replace('PCA', '')
    reg_tag = combo.split('_', 1)[1] if '_' in combo else "Linear"
    subdir = results_root / "final_plots" / f"{dataset}_{pca_tag}_{reg_tag}"
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"compare_{dataset}_{combo}_cornet.png"

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved CORnet lines: {out_path}")
    return out_path

# Optional CLI test
if __name__ == "__main__":
    plot_cornet_lines(dataset="THINGS", combo="PCA100_Linear")
