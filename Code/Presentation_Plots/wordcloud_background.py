#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
THINGS — Word cloud of the 49 human-derived dimensions.

What it does:
- Loads THINGS labels (49 dimension names) and Y embeddings (N x 49).
- Computes a weight per dimension (default: mean absolute value across images).
- Builds and saves a high-res word cloud.

Output:
  Results/final_plots/THINGS_examples/wordclouds/THINGS_dimensions_wordcloud.png
  Results/final_plots/THINGS_examples/wordclouds/THINGS_dimensions_wordcloud.svg

Weighting modes:
- "mean_abs"  : mean of abs(Y[:, d])        [default, robust]
- "mean_pos"  : mean of max(Y[:, d], 0)
- "uniform"   : all dimensions equal size
"""

from pathlib import Path
import numpy as np
import scipy.io as sio
from wordcloud import WordCloud

# ── Paths (aligned with your repo; keep original paths) ────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "Results"
THINGS_DIR   = PROJECT_ROOT / "THINGS"

OUT_DIR      = RESULTS_ROOT / "final_plots" / "THINGS_examples" / "wordclouds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep original unused paths as requested (no effect here)
ORIGINAL_STUFF_IMAGE_DIR = Path("/Users/22119216/Desktop/PhD_First_Year/Projects/Datasets/STUFF_dataset_600_images")
IMAGE_DIR = Path("/Users/22119216/Desktop/PhD_First_Year/Projects/Datasets/THINGS_Dataset")

# ── MATLAB helpers ─────────────────────────────────────────────────────────
def matlab_to_str(x) -> str:
    import numpy as np
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return x.decode("latin-1", errors="ignore")
    if isinstance(x, np.ndarray):
        if x.dtype == object and x.size == 1:
            return matlab_to_str(x.item())
        if x.dtype.kind in ("U", "S"):
            return "".join(x.ravel().tolist())
        x = np.squeeze(x)
        if x.dtype == object and x.size == 1:
            return matlab_to_str(x.item())
        return str(x)
    return str(x)

def matlab_cellstr_to_list(arr) -> list[str]:
    import numpy as np
    return [matlab_to_str(el) for el in np.ravel(arr)]

# ── Load labels and embeddings ─────────────────────────────────────────────
def load_things_labels_and_Y():
    emb_path    = THINGS_DIR / "spose_embedding_49d_sorted.txt"
    labels_path = THINGS_DIR / "labels.mat"
    if not emb_path.exists():    raise FileNotFoundError(f"Missing {emb_path}")
    if not labels_path.exists(): raise FileNotFoundError(f"Missing {labels_path}")

    Y = np.loadtxt(emb_path)  # shape (N, 49)
    labels_raw = sio.loadmat(labels_path)["labels"].flatten()
    dim_names  = matlab_cellstr_to_list(labels_raw)  # list of 49 strings
    assert Y.shape[1] == 49, f"Expected 49 dims in Y, got {Y.shape[1]}"
    assert len(dim_names) == 49, f"Expected 49 labels, got {len(dim_names)}"
    return dim_names, Y

# ── Compute weights for the word cloud ─────────────────────────────────────
def compute_dimension_weights(dim_names, Y, mode="mean_abs", power=1.6, floor=0.35):
    """
    mode:
      - mean_abs: mean of abs(Y[:, d])
      - mean_pos: mean of max(Y[:, d], 0)
      - uniform : all ones
    power: nonlinearity to increase contrast in sizes
    floor : minimum size factor to keep all visible
    """
    if mode == "mean_abs":
        w = np.mean(np.abs(Y), axis=0)
    elif mode == "mean_pos":
        w = np.mean(np.clip(Y, 0, None), axis=0)
    elif mode == "uniform":
        w = np.ones(Y.shape[1], dtype=float)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Normalize and shape
    w = np.asarray(w, dtype=float)
    w = w - w.min()
    if w.max() > 0:
        w = w / w.max()
    else:
        w = np.ones_like(w)

    # Gentle nonlinearity to make big words pop but keep all readable
    w = floor + (1 - floor) * (w ** power)

    # Build dict
    return {name: float(val) for name, val in zip(dim_names, w)}

# ── Build and save word cloud ──────────────────────────────────────────────
def make_wordcloud(frequencies: dict,
                   width=2800,
                   height=1800,
                   bg="white",
                   colormap="viridis",
                   seed=7,
                   prefer_horizontal=0.95,
                   scale=2):
    wc = WordCloud(
        width=width,
        height=height,
        background_color=bg,
        colormap=colormap,
        random_state=seed,
        prefer_horizontal=prefer_horizontal,
        collocations=False,
        normalize_plurals=False,
        margin=1,
        scale=scale,
        max_words=len(frequencies)
    ).generate_from_frequencies(frequencies)
    return wc

def save_wordcloud(wc, out_png: Path, out_svg: Path, dpi=400, pad=0.02):
    # Save PNG
    wc.to_file(str(out_png))
    # Save SVG via internal SVG export
    try:
        svg = wc.to_svg(embed_font=True)
        out_svg.write_text(svg, encoding="utf-8")
    except Exception as e:
        print(f"[warn] SVG export failed: {e}")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Settings you can tweak quickly
    WEIGHTING_MODE = "mean_abs"   # "mean_abs" | "mean_pos" | "uniform"
    POWER          = 1.6          # contrast nonlinearity
    FLOOR          = 0.35         # minimum size factor

    COLORMAP       = "viridis"    # "viridis", "plasma", "magma", "tab20", etc.
    BG_COLOR       = "white"
    WIDTH, HEIGHT  = 2800, 1800
    SEED           = 7

    OUT_PNG = OUT_DIR / "THINGS_dimensions_wordcloud.png"
    OUT_SVG = OUT_DIR / "THINGS_dimensions_wordcloud.svg"

    # Load
    dim_names, Y = load_things_labels_and_Y()

    # Weights
    freqs = compute_dimension_weights(dim_names, Y, mode=WEIGHTING_MODE,
                                      power=POWER, floor=FLOOR)

    # Build
    wc = make_wordcloud(
        freqs,
        width=WIDTH,
        height=HEIGHT,
        bg=BG_COLOR,
        colormap=COLORMAP,
        seed=SEED,
        prefer_horizontal=0.95,
        scale=2
    )

    # Save
    save_wordcloud(wc, OUT_PNG, OUT_SVG)
    print(f"[done] Saved word clouds to:\n  {OUT_PNG}\n  {OUT_SVG}")
