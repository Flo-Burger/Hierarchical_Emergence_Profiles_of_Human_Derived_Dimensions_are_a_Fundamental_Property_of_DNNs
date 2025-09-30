#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlexNet triptych plotting (callable).

- Resolves paths like your analysis code:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[2]
    RESULTS_ROOT_DEFAULT = PROJECT_ROOT / "Results"

- Panels:
  1) Raw Pixels — R² (stars if layerwise FDR CSVs exist AND cell R²>0)
  2) Untrained − Raw (ΔR²; stars if paired-permutation files exist)
  3) Trained − Untrained (ΔR²; stars if paired-permutation files exist)

- Saves into:
    Results/final_plots/<DATASET>_<PCA>_<REG>/alexnet_raw_to_trained_<DATASET>_<COMBO>_pairedperm.png
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# ── Styling (matches your look) ─────────────────────────────────────────────
try:
    import scienceplots  # optional
    plt.style.use(['science', 'nature'])
except Exception:
    pass

sns.set_context('paper', font_scale=1.15)
plt.rcParams.update({
    'font.family': 'arial',
    'pdf.fonttype': 42,
    'text.usetex': False,
    'axes.unicode_minus': False,
})

# ── Fonts & defaults ───────────────────────────────────────────────────────
FONTSIZE_TITLE = 14
FONTSIZE_AXIS  = 12
FONTSIZE_TICK  = 8
FONTSIZE_STAR  = 8
FONTSIZE_ANN   = 11
ALPHA_DEFAULT  = 0.05

# ── Roots resolved like your analysis ──────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT_DEFAULT = PROJECT_ROOT / "Results"

# ── Path helpers ───────────────────────────────────────────────────────────
def _model_base(results_root: Path, model: str, dataset: str, combo: str) -> Path:
    return results_root / f"{model}_{dataset}" / combo

def _r2_csv_path(results_root: Path, model: str, dataset: str, combo: str) -> Path:
    return _model_base(results_root, model, dataset, combo) / f"r2_{model}_{dataset}_{combo}_layers.csv"

def _perm_path(results_root: Path, model: str, dataset: str, combo: str, layer: str) -> Path:
    return _model_base(results_root, model, dataset, combo) / "permutation_testing" / f"{layer}_perm_r2.npy"

def _pvals_fdr_path(results_root: Path, model: str, dataset: str, combo: str, layer: str) -> Path:
    return _model_base(results_root, model, dataset, combo) / "permutation_testing" / f"{layer}_p_values_fdr.csv"

# ── IO helpers ─────────────────────────────────────────────────────────────
def _load_r2_table(results_root: Path, model: str, dataset: str, combo: str) -> pd.DataFrame:
    p = _r2_csv_path(results_root, model, dataset, combo)
    if not p.exists():
        raise FileNotFoundError(f"Missing R² CSV for {model}:\n  {p}")
    return pd.read_csv(p, index_col="dimension")

def _load_precomputed_layerwise_fdr_optional(results_root: Path, model: str, dataset: str, combo: str, layers: list[str]) -> np.ndarray | None:
    """Return [n_layers × n_dims] if ALL layerwise FDR CSVs exist; else None."""
    mats = []
    for L in layers:
        fp = _pvals_fdr_path(results_root, model, dataset, combo, L)
        if not fp.exists():
            return None
        dfp = pd.read_csv(fp)
        if "p_value_fdr" not in dfp.columns:
            return None
        mats.append(dfp["p_value_fdr"].values)
    return np.vstack(mats) if mats else None

def _load_perm_matrix_optional(results_root: Path, model: str, dataset: str, combo: str, layer: str) -> np.ndarray | None:
    p = _perm_path(results_root, model, dataset, combo, layer)
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.ndim != 2:
        return None
    return arr

def _align_layers_intersection(base_df: pd.DataFrame, other_df: pd.DataFrame) -> list[str]:
    base_cols  = list(base_df.columns)
    other_cols = set(other_df.columns)
    return [L for L in base_cols if L in other_cols]

# ── Public callable ────────────────────────────────────────────────────────
def plot_alexnet_heatmap(
    dataset: str,
    combo: str,
    results_root: str | Path | None = None,
    trained: str = "AlexNet",
    untrained: str = "AlexNetUntrained",
    rawpix: str = "RawPixels",
    alpha: float = ALPHA_DEFAULT,
    custom_dim_labels: dict[int, str] | None = None,
    dpi: int = 600
) -> Path:
    """
    Generate AlexNet 3-panel figure and save it into Results/final_plots/<DATASET>_<PCA>_<REG>/…

    Stars appear only when permutation/FDR files exist; otherwise the same heatmaps are drawn without stars.
    Returns the path to the saved PNG.
    """
    # Resolve Results/
    results_root = Path(results_root).resolve() if results_root else RESULTS_ROOT_DEFAULT.resolve()

    # Load R² tables (must exist)
    df_tr = _load_r2_table(results_root, trained,   dataset, combo)
    df_un = _load_r2_table(results_root, untrained, dataset, combo)
    df_rw = _load_r2_table(results_root, rawpix,    dataset, combo)
    if not (df_tr.index.equals(df_un.index) and df_tr.index.equals(df_rw.index)):
        raise ValueError("Dimension index mismatch across models; ensure identical 'dimension' index order.")

    # Panel 1: RawPixels — FDR stars only if layerwise CSVs exist
    layers_raw = list(df_rw.columns)
    pvals_raw  = _load_precomputed_layerwise_fdr_optional(results_root, rawpix, dataset, combo, layers_raw)

    # Panels 2/3: paired Δ + optional perms
    def paired_perm_delta_layerwise_optional(base_df, other_df, base_name: str, other_name: str):
        base_layers  = list(base_df.columns)
        other_layers = list(other_df.columns)

        if len(other_layers) == 1 and other_name == rawpix:
            common_layers = base_layers[:]
            base_vals  = base_df[common_layers].T.values
            other_vals = np.tile(other_df.iloc[:, 0].values, (len(common_layers), 1))
            other_fixed = other_layers[0]
            broadcast = True
        else:
            common_layers = _align_layers_intersection(base_df, other_df)
            if not common_layers:
                raise ValueError(f"No common layers between {base_name} and {other_name}.")
            base_vals  = base_df[common_layers].T.values
            other_vals = other_df[common_layers].T.values
            other_fixed = None
            broadcast = False

        delta_obs = base_vals - other_vals  # [L × D]
        p_fdr_all = []
        for i, L in enumerate(common_layers):
            A = _load_perm_matrix_optional(results_root, base_name, dataset, combo, L)
            B = _load_perm_matrix_optional(results_root, other_name, dataset, combo, other_fixed if broadcast else L)
            if A is None or B is None or A.shape != B.shape:
                return common_layers, delta_obs, None
            D = A - B
            obs = delta_obs[i, :][None, :]
            ge  = (np.abs(D) >= np.abs(obs)).sum(axis=0)
            p_raw = (ge + 1.0) / (D.shape[0] + 1.0)
            _, p_corr, _, _ = multipletests(p_raw, alpha=alpha, method='fdr_bh')
            p_fdr_all.append(p_corr)

        return common_layers, delta_obs, (np.vstack(p_fdr_all) if p_fdr_all else None)

    layers_UR, delta_UR, p_UR = paired_perm_delta_layerwise_optional(df_un, df_rw, untrained, rawpix)
    layers_TU, delta_TU, p_TU = paired_perm_delta_layerwise_optional(df_tr, df_un, trained, untrained)

    # ── Plotting ────────────────────────────────────────────────────────────
    n_dims   = df_rw.shape[0]
    n_rows1  = len(layers_raw)
    n_rows2  = len(layers_UR)
    n_rows3  = len(layers_TU)

    FIG_W = 12 if dataset == "THINGS" else 10
    FIG_H = max(5.0, 0.23*(n_rows1+n_rows2+n_rows3) + 2.2)

    fig, axes = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H), sharex=True,
        gridspec_kw={'height_ratios': [n_rows1, n_rows2, n_rows3], 'hspace': 0.20}
    )

    # Panel 1: Raw Pixels — R²
    vmin_r2, vmax_r2 = -0.2, 1.0
    mat_raw = df_rw[layers_raw].T.values
    sns.heatmap(mat_raw, ax=axes[0], cmap="RdBu_r", center=0, square=True,
                linewidths=0.3, linecolor="white", vmin=vmin_r2, vmax=vmax_r2, cbar=False)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis='both', which='both', length=0)
    axes[0].minorticks_off()
    axes[0].set_yticks(np.arange(n_rows1)+0.5)
    axes[0].set_yticklabels(np.arange(1, n_rows1+1), rotation=0, fontsize=FONTSIZE_TICK)
    axes[0].set_ylabel("Layer", fontsize=FONTSIZE_AXIS, labelpad=5)
    axes[0].set_title("Raw Pixels — R$^2$", fontsize=FONTSIZE_TITLE, pad=12, loc="center")

    if pvals_raw is not None:
        sig_raw = (pvals_raw < alpha) & (mat_raw > 0)
        for i in range(sig_raw.shape[0]):
            for j in range(sig_raw.shape[1]):
                if sig_raw[i, j]:
                    axes[0].text(j+0.5, i+0.5, "*", ha="center", va="center",
                                 color="black", fontsize=FONTSIZE_STAR)

    # Panels 2 & 3 (shared symmetric range)
    vmax_diff = max(
        np.abs(delta_UR).max() if delta_UR.size else 0,
        np.abs(delta_TU).max() if delta_TU.size else 0
    )
    vmax_diff = float(vmax_diff) if np.isfinite(vmax_diff) and vmax_diff > 0 else 0.1
    vmin_diff = -vmax_diff

    def _draw_diff(ax, layers, delta, pvals, title_ascii):
        sns.heatmap(delta, ax=ax, cmap="RdBu_r", center=0, square=True,
                    linewidths=0.3, linecolor="white", vmin=vmin_diff, vmax=vmax_diff, cbar=False)
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='both', length=0)
        ax.minorticks_off()
        ax.set_yticks(np.arange(len(layers))+0.5)
        ax.set_yticklabels(np.arange(1, len(layers)+1), rotation=0, fontsize=FONTSIZE_TICK)
        ax.set_ylabel("Layer", fontsize=FONTSIZE_AXIS, labelpad=5)
        ax.set_title(f"{title_ascii} ($\\Delta R^2$)", fontsize=FONTSIZE_TITLE, pad=12, loc="center")

        if pvals is not None:
            sig = (pvals < alpha)
            for i in range(delta.shape[0]):
                for j in range(delta.shape[1]):
                    if sig[i, j]:
                        ax.text(j+0.5, i+0.5, "*", ha="center", va="center",
                                color="black", fontsize=FONTSIZE_STAR)

    _draw_diff(axes[1], layers_UR, delta_UR, p_UR, "Untrained - Raw Pixels")
    _draw_diff(axes[2], layers_TU, delta_TU, p_TU, "Trained - Untrained")

    # Bottom dim labels (defaults)
    if custom_dim_labels is None:
        custom_dim_labels = (
            {3: "Animal-related", 13: "Electronic / technology", 23: "Red", 35: "Disgusting / bugs", 46: "Medicine-related"}
            if dataset == "THINGS" else
            {1: "Mineral", 24: "Blue colour", 34: "Bumpy"}
        )
    positions = np.arange(n_dims) + 0.5
    dims = np.arange(1, n_dims+1)
    tick_labels = ["" if d in custom_dim_labels else str(d) for d in dims]
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(tick_labels, rotation=0, fontsize=FONTSIZE_TICK)
    axes[-1].set_xlabel("Dimension", fontsize=FONTSIZE_AXIS, labelpad=25)
    for d, label in custom_dim_labels.items():
        idx = d - 1
        if 0 <= idx < n_dims:
            x = idx + 0.5
            axes[-1].annotate(
                label, xy=(x, 0), xycoords=('data', 'axes fraction'),
                xytext=(x, -.2), textcoords=('data', 'axes fraction'),
                ha='center', va='top', fontsize=FONTSIZE_ANN, clip_on=False,
                annotation_clip=False,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8,
                                shrinkA=0, shrinkB=0, clip_on=False)
            )

    # Shared colorbars
    fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.18, hspace=0.20)
    cbar_ax1 = fig.add_axes([0.90, 0.68, 0.014, 0.24])
    cbar1 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin_r2, vmax=vmax_r2),
                                               cmap="RdBu_r"), cax=cbar_ax1)
    cbar1.set_label("R$^2$ (panel 1)", fontsize=FONTSIZE_AXIS)
    cbar1.ax.tick_params(labelsize=FONTSIZE_TICK, length=0)

    cbar_ax2 = fig.add_axes([0.90, 0.15, 0.014, 0.45])
    cbar2 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin_diff, vmax=vmax_diff),
                                               cmap="RdBu_r"), cax=cbar_ax2)
    cbar2.set_label("$\\Delta R^2$ (panels 2–3)", fontsize=FONTSIZE_AXIS)
    cbar2.ax.tick_params(labelsize=FONTSIZE_TICK, length=0)

    # Output path
    pca_tag = combo.split('_')[0].replace('PCA', '')
    reg_tag = combo.split('_', 1)[1] if '_' in combo else "Linear"
    plots_dir = results_root / "final_plots" / f"{dataset}_{pca_tag}_{reg_tag}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / f"alexnet_raw_to_trained_{dataset}_{combo}_pairedperm.png"

    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"✔ Saved: {out_png}")
    return out_png

# Optional CLI test
if __name__ == "__main__":
    plot_alexnet_heatmap(dataset="THINGS", combo="PCA100_Linear")
