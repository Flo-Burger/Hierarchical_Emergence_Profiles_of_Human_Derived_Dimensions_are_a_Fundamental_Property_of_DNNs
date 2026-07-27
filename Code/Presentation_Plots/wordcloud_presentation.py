#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
THINGS Figure 4A — render ALL images from THINGS/im.mat (low-res arrays), paired with Y[i].

Panel:
  [im.mat image (upscaled for display) | top-5 human-dimension wordcloud (size ∝ weight)]

Output:
  Results/final_plots/THINGS_examples/figure4A_THINGS_example_<INDEX>.png

Notes:
- This script does NOT touch the filesystem THINGS images; it only uses im.mat arrays.
- Keep it simple: march through rows i = 0..N-1, render, save.
"""

from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud

# ── Paths (aligned with your repo; keep original paths) ────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "Results"
THINGS_DIR   = PROJECT_ROOT / "THINGS"
OUT_DIR      = RESULTS_ROOT / "final_plots" / "THINGS_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep original unused paths as requested (no effect here)
ORIGINAL_STUFF_IMAGE_DIR = Path("/Users/22119216/Desktop/PhD_First_Year/Projects/Datasets/STUFF_dataset_600_images")
IMAGE_DIR = Path("/Users/22119216/Desktop/PhD_First_Year/Projects/Datasets/THINGS_Dataset")

# ── MATLAB helpers ─────────────────────────────────────────────────────────
def matlab_to_str(x) -> str:
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
    return [matlab_to_str(el) for el in np.ravel(arr)]

def matlab_image_to_pil(x) -> Image.Image:
    """
    Convert one im.mat entry to PIL RGB.
    Handles HxWx3 uint8, HxW uint8, float in [0,1] or [0,255], and nested cells.
    """
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    arr = np.array(x)
    arr = np.squeeze(arr)

    # Normalize dtype/range
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Channel order fixes (channel-first → channel-last)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # Create PIL image
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr, mode="RGB")
    elif arr.ndim == 3 and arr.shape[2] == 4:
        img = Image.fromarray(arr[:, :, :3], mode="RGB")
    else:
        # Fallback: try squeeze again and force grayscale→RGB
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            img = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            raise ValueError(f"Unsupported im.mat image shape: {arr.shape}")
    return img

# ── Load Y, labels, and im arrays (exactly like your analysis) ────────────
def load_things_Y_labels_im():
    emb_path    = THINGS_DIR / "spose_embedding_49d_sorted.txt"
    labels_path = THINGS_DIR / "labels.mat"
    im_path     = THINGS_DIR / "im.mat"

    if not emb_path.exists():    raise FileNotFoundError(f"Missing {emb_path}")
    if not labels_path.exists(): raise FileNotFoundError(f"Missing {labels_path}")
    if not im_path.exists():     raise FileNotFoundError(f"Missing {im_path}")

    Y = np.loadtxt(emb_path)  # (N, 49)
    labels_raw = sio.loadmat(labels_path)["labels"].flatten()
    dim_names  = matlab_cellstr_to_list(labels_raw)
    im_raw     = sio.loadmat(im_path)["im"].flatten()  # array/cell per image

    assert Y.shape[0] == len(im_raw), f"Y rows ({Y.shape[0]}) != #im entries ({len(im_raw)})"
    return Y, dim_names, im_raw

# ── Wordcloud & rendering helpers ─────────────────────────────────────────
def top5_pairs(row: np.ndarray, names: list[str]):
    idx = np.argsort(row)[::-1][:5]
    return [(names[i], float(row[i])) for i in idx]

def build_wordcloud(pairs, width=900, height=600, seed=7):
    vals = np.array([w for _, w in pairs], dtype=float)
    if np.allclose(vals.max(), vals.min()):
        vals = np.ones_like(vals)

    # normalize
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

    # nonlinear emphasis: big gets BIG, small stays visible
    vals = vals**1.8            # experiment: 1.5–2.5
    vals = 0.45 + 1 * vals     # floor = 0.3, peak ~3.0

    freqs = {name: float(v) for (name, _), v in zip(pairs, vals)}
    wc = WordCloud(
        width=2500,             # smaller internal canvas
        height=500,
        background_color="white",
        colormap="viridis",
        prefer_horizontal=1.0,
        collocations=False,
        normalize_plurals=False,
        random_state=seed,
        max_words=len(freqs),
        margin=0,              # reduce whitespace between words
        scale=2                # render higher res then shrink
    )

    return wc.generate_from_frequencies(freqs)

def upscale_for_display(pil_img: Image.Image, target_long_side=1800) -> Image.Image:
    w, h = pil_img.size
    s = max(w, h)
    if s < target_long_side:
        scale = target_long_side / s
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return pil_img

def render_example(idx: int, pil_img: Image.Image, Y: np.ndarray, dim_names: list[str]):
    img = upscale_for_display(pil_img, target_long_side=1800)
    pairs = top5_pairs(Y[idx], dim_names)
    wc = build_wordcloud(pairs)

    fig, axes = plt.subplots(1, 2, figsize=(25, 6.5), dpi=180,
                             gridspec_kw={'wspace': 0.02})
    axes[0].imshow(img, interpolation="nearest"); axes[0].axis("off")
    axes[1].imshow(wc, interpolation="nearest");  axes[1].axis("off")

    fig.tight_layout()
    out = OUT_DIR / f"figure4A_THINGS_example_{idx}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Y, dim_names, im_raw = load_things_Y_labels_im()
    N = len(im_raw)

    saved = 0
    for idx in range(N):
        try:
            pil_img = matlab_image_to_pil(im_raw[idx])
            out = render_example(idx, pil_img, Y, dim_names)
            if saved % 25 == 0:
                print(f"[info] saved {saved+1}/{N}: {out.name}")
            saved += 1
        except Exception as e:
            print(f"[skip] row {idx}: {e}")

    print(f"[done] Rendered {saved}/{N} panels to {OUT_DIR}")
