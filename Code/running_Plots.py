#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run all triptych heatmaps and line plots after analysis is done.

Discovers (dataset, combo) pairs by scanning:
    Results/{Model}_{DATASET}/{COMBO}/r2_{Model}_{DATASET}_{COMBO}_layers.csv

Families rendered:
- AlexNet:      requires RawPixels, AlexNetUntrained, AlexNet
- CORnet :      requires FeedforwardCORnet, RecurrentCORnet, SkipCORnet
- Transformers: requires ViT, CLIP, DINOv3

Figures are saved under:
    Results/final_plots/<DATASET>_<PCA>_<REG>/
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

# Resolve Results/ like your main code
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
RESULTS_ROOT_DEFAULT = PROJECT_ROOT / "Results"

# ── Heatmap plotting callables ─────────────────────────────────────────────
from plots.Heatmaps_AlexNet       import plot_alexnet_heatmap
from plots.Heatmaps_CORnet        import plot_cornet_heatmap
from plots.Heatmaps_Transformers  import plot_transformer_heatmap

# ── Line plotting callables ────────────────────────────────────────────────
from plots.Lineplot_AlexNet   import plot_alexnet_lines
from plots.Lineplot_CORnet    import plot_cornet_lines
from plots.Lineplot_Transformers import plot_transformer_lines

ALEXNET_REQUIRED      = {"RawPixels", "AlexNetUntrained", "AlexNet"}
CORNET_REQUIRED       = {"FeedforwardCORnet", "RecurrentCORnet", "SkipCORnet"}
TRANSFORMERS_REQUIRED = {"ViT", "CLIP", "DINOv3"}


def _have_r2(results_root: Path, model: str, dataset: str, combo: str) -> bool:
    p = results_root / f"{model}_{dataset}" / combo / f"r2_{model}_{dataset}_{combo}_layers.csv"
    return p.exists()


def _combos_for_model_dataset(results_root: Path, model: str, dataset: str) -> Set[str]:
    base = results_root / f"{model}_{dataset}"
    if not base.exists():
        return set()
    return {d.name for d in base.iterdir() if d.is_dir()}


def _discover_pairs_for_family(results_root: Path, required_models: Set[str],
                               datasets: Optional[Iterable[str]] = None,
                               combos_filter: Optional[Iterable[str]] = None) -> Set[Tuple[str, str]]:
    """
    Return all (dataset, combo) where all required_models have an R² CSV.
    Seeds combos from the first required model (sorted).
    """
    pairs: Set[Tuple[str, str]] = set()
    # discover datasets by folder suffixes (Model_DATASET)
    if datasets is None:
        ds_set = set()
        for child in results_root.iterdir():
            if child.is_dir() and "_" in child.name:
                ds_set.add(child.name.split("_")[-1])
        datasets = sorted(ds_set)

    seed_model = sorted(required_models)[0] if required_models else None
    if seed_model is None:
        return pairs

    for ds in datasets:
        combos = _combos_for_model_dataset(results_root, seed_model, ds)
        if combos_filter is not None:
            combos &= set(combos_filter)
        for combo in sorted(combos):
            if all(_have_r2(results_root, m, ds, combo) for m in required_models):
                pairs.add((ds, combo))
    return pairs


def render_all_plots(
    results_root: str | Path | None = None,
    datasets: Optional[Iterable[str]] = None,
    combos: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
) -> None:
    """
    Render triptychs and line plots for all valid (dataset, combo) pairs.
    """
    results_root = Path(results_root).resolve() if results_root else RESULTS_ROOT_DEFAULT.resolve()

    # AlexNet
    alex_pairs = _discover_pairs_for_family(results_root, ALEXNET_REQUIRED, datasets, combos)
    print(f"AlexNet: {len(alex_pairs)} pair(s) to plot.")
    for ds, combo in sorted(alex_pairs):
        try:
            plot_alexnet_heatmap(
                dataset=ds,
                combo=combo,
                results_root=results_root,
                alpha=alpha,
                trained="AlexNet",
                untrained="AlexNetUntrained",
                rawpix="RawPixels"
            )
            plot_alexnet_lines(dataset=ds, combo=combo, results_root=results_root)
        except FileNotFoundError as e:
            print(f"[AlexNet] Skipping {ds}/{combo}: {e}")

    # CORnet
    cor_pairs = _discover_pairs_for_family(results_root, CORNET_REQUIRED, datasets, combos)
    print(f"CORnet : {len(cor_pairs)} pair(s) to plot.")
    for ds, combo in sorted(cor_pairs):
        try:
            plot_cornet_heatmap(
                dataset=ds,
                combo=combo,
                results_root=results_root,
                alpha=alpha,
                feedforward="FeedforwardCORnet",
                recurrent="RecurrentCORnet",
                skip="SkipCORnet"
            )
            plot_cornet_lines(dataset=ds, combo=combo, results_root=results_root)
        except FileNotFoundError as e:
            print(f"[CORnet] Skipping {ds}/{combo}: {e}")

    # Transformers
    trf_pairs = _discover_pairs_for_family(results_root, TRANSFORMERS_REQUIRED, datasets, combos)
    print(f"Transformers: {len(trf_pairs)} pair(s) to plot.")
    for ds, combo in sorted(trf_pairs):
        try:
            plot_transformer_heatmap(
                dataset=ds,
                combo=combo,
                results_root=results_root,
                alpha=alpha,
                vit="ViT",
                clip="CLIP",
                dino="DINOv3"
            )
            plot_transformer_lines(dataset=ds, combo=combo, results_root=results_root)
        except FileNotFoundError as e:
            print(f"[Transformers] Skipping {ds}/{combo}: {e}")


if __name__ == "__main__":
    render_all_plots()
