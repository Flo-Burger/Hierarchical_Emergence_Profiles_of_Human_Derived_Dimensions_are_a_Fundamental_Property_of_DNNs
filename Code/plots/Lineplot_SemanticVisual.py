#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic vs Visual dimension comparison line plots for the 66d THINGS analysis.

For each model family (AlexNet / CORnet / Transformers), produces a figure with
one subplot per model. Each subplot shows mean R² ± SE across layers separately
for semantic and visual dimensions, using human_dimension_types.csv to classify
the 66 SPOSE dimensions. The band between the two curves is shaded to make the
semantic–visual gap the visual subject of the figure.

Two complementary statistics are computed per model:

1. Layer-by-layer magnitude test: two-sided label-permutation test on the
   difference of group means (visual − semantic) at each layer, FDR-corrected
   across layers (Benjamini–Hochberg). Significant layers are marked with *.

2. Peak-layer comparison: for each dimension the peak (argmax) layer is found,
   then a two-sided permutation test on the difference of mean peak layers
   (visual − semantic) is run. Mean peak layers are shown as vertical dashed
   lines on each panel. This directly addresses whether visual dimensions peak
   at a different level of the hierarchy than semantic ones.

Dimensions typed as "mix visual-semantic" or "unclear" are excluded from both
analyses. Their count is printed to the console.

Figure text is kept minimal (panel = model name, one shared legend, no suptitle).

Saves to:
    Results/final_plots/THINGS66d_<PCA>_<REG>/semantic_visual_<family>.png
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

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

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT_DEFAULT = PROJECT_ROOT / "Results"
DIM_TYPES_FILE = PROJECT_ROOT / "THINGS" / "human_dimension_types.csv"

SEMANTIC_COLOR = "#2166AC"
VISUAL_COLOR   = "#D6604D"
GAP_COLOR      = "#666666"

N_PERM    = 10000
PERM_SEED = 0
ALPHA     = 0.05


FAMILY_MODELS = {
    "AlexNet":      ["RawPixels", "AlexNetUntrained", "AlexNet"],
    "CORnet":       ["FeedforwardCORnet", "RecurrentCORnet", "SkipCORnet"],
    "Transformers": ["ViT", "CLIP", "DINOv3"],
}

DISPLAY_NAMES = {
    "RawPixels":         "Raw Pixels",
    "AlexNetUntrained":  "AlexNet (untrained)",
    "AlexNet":           "AlexNet (trained)",
    "FeedforwardCORnet": "Feedforward CORnet",
    "RecurrentCORnet":   "Recurrent CORnet",
    "SkipCORnet":        "Skip CORnet",
    "ViT":               "ViT",
    "CLIP":              "CLIP",
    "DINOv3":            "DINOv3",
}


def _r2_path(root: Path, model: str, dataset: str, combo: str) -> Path:
    return root / f"{model}_{dataset}" / combo / f"r2_{model}_{dataset}_{combo}_layers.csv"


def _load_dim_types(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Label"])
    df = df[df["Label"].str.strip() != ""]
    return df.set_index("Label")["Type"]


def _mean_se(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    se   = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, se


def _label_perm_pvalues(
    values: np.ndarray,
    sem_mask: np.ndarray,
    vis_mask: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
) -> np.ndarray:
    """
    Per-layer label-permutation test, two-sided, on the difference of group
    means (visual − semantic). Returns raw (uncorrected) p-values, one per layer.
    The semantic/visual labels are shuffled across the pooled set of classified
    dimensions to build the null distribution.
    """
    vis_vals = values[vis_mask]
    sem_vals = values[sem_mask]
    pooled   = np.concatenate([vis_vals, sem_vals])
    n_vis    = vis_vals.shape[0]
    n_total  = pooled.shape[0]
    n_layers = pooled.shape[1]

    if n_vis == 0 or sem_vals.shape[0] == 0:
        return np.ones(n_layers)

    obs = vis_vals.mean(axis=0) - sem_vals.mean(axis=0)

    rng = np.random.default_rng(seed)
    idx = rng.permuted(
        np.broadcast_to(np.arange(n_total), (n_perm, n_total)).copy(), axis=1
    )
    # pooled: (n_total, n_layers) → idx slices → (n_perm, n_vis/n_sem, n_layers)
    null = pooled[idx[:, :n_vis]].mean(axis=1) - pooled[idx[:, n_vis:]].mean(axis=1)
    count = (np.abs(null) >= np.abs(obs)[None, :]).sum(axis=0)

    return (count + 1.0) / (n_perm + 1.0)


def _peak_layer_perm(
    values: np.ndarray,
    sem_mask: np.ndarray,
    vis_mask: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED + 1,
) -> tuple[float, float, float]:
    """
    Two-sided permutation test on the difference of mean peak layers
    (visual − semantic). Peak layer is the argmax of R² across layers for each
    dimension (1-indexed). Returns (mean_peak_visual, mean_peak_semantic, p-value).
    Returns (nan, nan, 1.0) for single-layer models where the test is undefined.
    """
    if values.shape[1] <= 1:
        return (np.nan, np.nan, 1.0)

    peak_layers = np.argmax(values, axis=1).astype(float) + 1  # (n_dims,) 1-indexed

    vis_peaks = peak_layers[vis_mask]
    sem_peaks = peak_layers[sem_mask]

    if len(vis_peaks) == 0 or len(sem_peaks) == 0:
        return (np.nan, np.nan, 1.0)

    pooled  = np.concatenate([vis_peaks, sem_peaks])
    n_vis   = len(vis_peaks)
    n_total = len(pooled)
    obs     = vis_peaks.mean() - sem_peaks.mean()

    rng = np.random.default_rng(seed)
    idx = rng.permuted(
        np.broadcast_to(np.arange(n_total), (n_perm, n_total)).copy(), axis=1
    )
    null  = pooled[idx[:, :n_vis]].mean(axis=1) - pooled[idx[:, n_vis:]].mean(axis=1)
    count = int((np.abs(null) >= abs(obs)).sum())

    return (float(vis_peaks.mean()), float(sem_peaks.mean()), (count + 1.0) / (n_perm + 1.0))


def _star_y_position(ax: plt.Axes) -> float:
    lo, hi = ax.get_ylim()
    return hi - 0.04 * (hi - lo)


def _plot_model_subplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    sem_mask: np.ndarray,
    vis_mask: np.ndarray,
    title: str,
    show_legend: bool,
    mean_peak_vis: float | None = None,
    mean_peak_sem: float | None = None,
) -> None:
    values   = df.values  # (n_dims, n_layers)
    n_layers = values.shape[1]

    pvals_raw = _label_perm_pvalues(values, sem_mask, vis_mask)
    _, pvals_fdr, _, _ = multipletests(pvals_raw, alpha=ALPHA, method='fdr_bh')
    sig = pvals_fdr < ALPHA

    sem_mean, sem_se = _mean_se(values[sem_mask])
    vis_mean, vis_se = _mean_se(values[vis_mask])

    if n_layers == 1:
        x = np.array([1.0])
        ax.plot([1, 1], [sem_mean[0], vis_mean[0]], color=GAP_COLOR,
                linewidth=4, alpha=0.25, solid_capstyle="round", zorder=1)
        ax.errorbar(x, sem_mean, yerr=sem_se, fmt='o', color=SEMANTIC_COLOR,
                    capsize=3, label="Semantic", zorder=3)
        ax.errorbar(x, vis_mean, yerr=vis_se, fmt='o', color=VISUAL_COLOR,
                    capsize=3, label="Visual", zorder=3)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", zorder=0)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        ax.set_xlim(0.5, 1.5)
    else:
        x = np.arange(1, n_layers + 1)

        # Peak-layer markers (behind everything else)
        if mean_peak_vis is not None and np.isfinite(mean_peak_vis):
            ax.axvline(mean_peak_vis, color=VISUAL_COLOR, linewidth=1.0,
                       linestyle='--', alpha=0.55, zorder=0)
        if mean_peak_sem is not None and np.isfinite(mean_peak_sem):
            ax.axvline(mean_peak_sem, color=SEMANTIC_COLOR, linewidth=1.0,
                       linestyle='--', alpha=0.55, zorder=0)

        ax.fill_between(x, sem_mean, vis_mean, color=GAP_COLOR, alpha=0.12,
                        linewidth=0, zorder=1)
        for mean, se, color, label in [
            (sem_mean, sem_se, SEMANTIC_COLOR, "Semantic"),
            (vis_mean, vis_se, VISUAL_COLOR,   "Visual"),
        ]:
            ax.plot(x, mean, color=color, linewidth=1.8, label=label, zorder=3)
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2,
                            linewidth=0, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=6)
        ax.tick_params(axis='x', which='minor', length=0)

    ax._sv_sig = (np.arange(1, n_layers + 1), sig)

    ax.set_title(title, fontsize=8, pad=4)
    ax.tick_params(axis='both', which='major', labelsize=7)
    if show_legend:
        ax.legend(frameon=False, prop={'size': 6}, loc='lower right')


def plot_semantic_visual(
    family: str,
    combo: str,
    dataset: str = "THINGS66d",
    results_root: str | Path | None = None,
    dim_types_file: str | Path | None = None,
    dpi: int = 600,
) -> Path:
    """
    Render semantic vs visual R² line plot for a model family.

    Parameters
    ----------
    family        : "AlexNet", "CORnet", or "Transformers"
    combo         : e.g. "PCA100_Linear"
    dataset       : dataset suffix used in result folders (default "THINGS66d")
    results_root  : base Results directory
    dim_types_file: path to human_dimension_types.csv
    dpi           : output DPI
    """
    results_root   = Path(results_root).resolve() if results_root else RESULTS_ROOT_DEFAULT.resolve()
    dim_types_file = Path(dim_types_file) if dim_types_file else DIM_TYPES_FILE

    models = FAMILY_MODELS[family]

    dim_types = _load_dim_types(dim_types_file)

    dfs: dict[str, pd.DataFrame] = {}
    for m in models:
        p = _r2_path(results_root, m, dataset, combo)
        if not p.exists():
            raise FileNotFoundError(f"Cannot find: {p}")
        dfs[m] = pd.read_csv(p, index_col=0)

    dim_labels     = dfs[models[0]].index.tolist()
    types_for_dims = [dim_types.get(label, "unknown") for label in dim_labels]

    sem_mask = np.array([t == "semantic" for t in types_for_dims])
    vis_mask = np.array([t == "visual"   for t in types_for_dims])

    n_sem = int(sem_mask.sum())
    n_vis = int(vis_mask.sum())
    n_excl = len(dim_labels) - n_sem - n_vis
    print(f"[{family}] {dataset}/{combo}: {n_sem} semantic, {n_vis} visual, "
          f"{n_excl} excluded (mix/unclear)")

    # Rename columns to numeric indices
    for m in models:
        df = dfs[m].copy()
        df.columns = list(range(1, df.shape[1] + 1))
        dfs[m] = df

    # Compute peak-layer stats per model and print
    peak_stats: dict[str, tuple[float, float, float]] = {}
    for m in models:
        mpv, mps, p_peak = _peak_layer_perm(dfs[m].values, sem_mask, vis_mask)
        peak_stats[m] = (mpv, mps, p_peak)
        if np.isfinite(mpv):
            print(f"  [{DISPLAY_NAMES.get(m, m)}] mean peak layer — "
                  f"visual: {mpv:.1f}, semantic: {mps:.1f}, p={p_peak:.4f}")

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.33 * n_models, 2.8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, (ax, m) in enumerate(zip(axes, models)):
        mpv, mps, _ = peak_stats[m]
        _plot_model_subplot(
            ax=ax,
            df=dfs[m],
            sem_mask=sem_mask,
            vis_mask=vis_mask,
            title=DISPLAY_NAMES.get(m, m),
            show_legend=(i == 0),
            mean_peak_vis=mpv,
            mean_peak_sem=mps,
        )

    axes[0].set_ylabel("Mean R$^2$", fontsize=7)

    # Place significance asterisks after y-limits are final
    for ax in axes:
        xs, sig = getattr(ax, "_sv_sig", (np.array([]), np.array([], dtype=bool)))
        y = _star_y_position(ax)
        for xi, si in zip(xs, sig):
            if si:
                ax.text(xi, y, "*", ha="center", va="top", fontsize=9,
                        color="black")

    fig.tight_layout()

    pca_tag = combo.split('_')[0].replace('PCA', '')
    reg_tag = combo.split('_', 1)[1] if '_' in combo else "Linear"
    subdir  = results_root / "final_plots" / f"{dataset}_{pca_tag}_{reg_tag}"
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"semantic_visual_{family.lower()}_{combo}.png"

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_all_semantic_visual(
    combo: str = "PCA100_Linear",
    dataset: str = "THINGS66d",
    results_root: str | Path | None = None,
    dim_types_file: str | Path | None = None,
    dpi: int = 600,
) -> List[Path]:
    """Render semantic vs visual plots for all three model families."""
    out_paths = []
    for family in FAMILY_MODELS:
        try:
            p = plot_semantic_visual(
                family=family,
                combo=combo,
                dataset=dataset,
                results_root=results_root,
                dim_types_file=dim_types_file,
                dpi=dpi,
            )
            out_paths.append(p)
        except FileNotFoundError as e:
            print(f"[{family}] Skipping — {e}")
    return out_paths


if __name__ == "__main__":
    plot_all_semantic_visual()
