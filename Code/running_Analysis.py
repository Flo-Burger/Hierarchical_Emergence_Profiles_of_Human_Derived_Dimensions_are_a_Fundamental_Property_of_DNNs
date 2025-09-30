#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import scipy.io as sio
import torch
from statsmodels.stats.multitest import multipletests
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from PIL import Image
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
PCA_COMPONENTS_LIST = [100, 200, 0.95, 0.99]
REGRESSORS = {
    "Linear": LinearRegression(),
    "SVR":    SVR(kernel='rbf', C=1.0, epsilon=0.1),
}

PCA_COMPONENTS_LIST = [100]
REGRESSORS = {
    "Linear": LinearRegression()
}

# We have not run the permutation testing for any other setting beyond 100 PCA
# and linear regression due to the large computation time needed.
run_permutation   = False
n_perm            = 5000
alpha_thresh      = 0.05   # FDR threshold
CV = 10

# which dims to label (0-based) per dataset (kept for compatibility if you use elsewhere)
SELECTED_DIMS_MAP = {
    "THINGS": [2, 12, 22, 34, 45],
    "STUFF":  [0, 4, 23, 33]
}

# Paths
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "Results")

# NEW: dataset roots (relative)
THINGS_DIR = os.path.join(PROJECT_ROOT, "THINGS")
STUFF_DIR  = os.path.join(PROJECT_ROOT, "STUFF")

sys.path.append(PROJECT_ROOT)

from layer_extractions.AlexNet_extraction import extract_alexnet
from layer_extractions.AlexNet_extraction_untrained import extract_alexnet_untrained
from layer_extractions.feedforward_cornet import extract_feedforward_cornet_activations
from layer_extractions.recurrent_cornet   import extract_recurrent_cornet_activations
from layer_extractions.skip_cornet        import extract_skip_cornet_activations
from layer_extractions.vit_extraction    import extract_ViT
from layer_extractions.CLIP_extraction   import extract_CLIP
from layer_extractions.Dinov3_extraction import extract_dinov3 as extract_DINOv3
from layer_extractions.raw_pixel_extraction import extract_raw_pixels

from running_Plots import render_all_plots
MODEL_EXTRACTORS = {
    "AlexNet":           extract_alexnet,
    "AlexNetUntrained":  extract_alexnet_untrained,
    "RawPixels":         extract_raw_pixels,
    "FeedforwardCORnet": extract_feedforward_cornet_activations,
    "RecurrentCORnet":   extract_recurrent_cornet_activations,
    "SkipCORnet":        extract_skip_cornet_activations,
    "ViT":               extract_ViT,
    "CLIP":              extract_CLIP,
    "DINOv3":            extract_DINOv3,
}

def safe_loadmat(filepath, key):
    try:
        data = sio.loadmat(filepath)
        return data[key]
    except Exception as e:
        raise ValueError(f"Error loading {filepath} with key '{key}': {e}")

def compute_r2_scores_with_model(X, Y, reg):
    out = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        out[j] = cross_val_score(
            reg, X, Y[:, j], cv=CV, scoring="r2", n_jobs=-1
        ).mean()
    return out

# Main Analysis
# for ds in ["THINGS", "STUFF"]:
for ds in ["THINGS"]:
    print(f"\n\n===== DATASET: {ds} =====")
    if ds == "THINGS":
        # For THINGS
        embedding_file = os.path.join(THINGS_DIR, "spose_embedding_49d_sorted.txt")
        labels_file    = os.path.join(THINGS_DIR, "labels.mat")
        images_file    = os.path.join(THINGS_DIR, "im.mat")

        Y         = np.loadtxt(embedding_file)
        dim_names = [l[0] for l in safe_loadmat(labels_file, 'labels').flatten()]
        images    = safe_loadmat(images_file, 'im').flatten()
        assert Y.shape[0] == len(images)

    else: 
        # For STUFF
        embedding_file = os.path.join(STUFF_DIR, "spose_embedding36.mat")
        labels_file    = os.path.join(STUFF_DIR, "labels.mat")
        images_file    = os.path.join(STUFF_DIR, "im.mat")

        Y         = safe_loadmat(embedding_file, 'spose_embedding36')
        dim_names = [l[0] for l in safe_loadmat(labels_file, 'labels').flatten()]
        images    = safe_loadmat(images_file, 'im').flatten()
        assert Y.shape[0] == len(images)

    device = torch.device("cpu")
    torch.manual_seed(42); np.random.seed(42)

    # kept in case used downstream
    selected_dims = SELECTED_DIMS_MAP[ds]

    for model_name, extract_fn in MODEL_EXTRACTORS.items():
        print(f"\n--- Model: {model_name} on {ds} ---")
        base_out = os.path.join(RESULTS_ROOT, f"{model_name}_{ds}")
        os.makedirs(base_out, exist_ok=True)

        # Extract activations (per your extractor)
        activations = extract_fn(images, device)
        layers       = list(activations.keys())

        for PCA_K in PCA_COMPONENTS_LIST:
            time.sleep(10)  # avoid file system issues
            for reg_name, reg in REGRESSORS.items():
                combo = f"PCA{PCA_K}_{reg_name}"
                outd  = os.path.join(base_out, combo)
                os.makedirs(outd, exist_ok=True)

                print(f"  Running {combo}…", end="", flush=True)
                feats = {}
                for L in layers:
                    arr = np.stack(activations[L], axis=0)
                    k = min(PCA_K, arr.shape[1]) if isinstance(PCA_K, (int,float)) else arr.shape[1]
                    pca = PCA(n_components=k,
                              svd_solver="full" if isinstance(PCA_K,float) else "auto")
                    feats[L] = pca.fit_transform(arr)

                # original R²
                r2_dict = {
                    L: compute_r2_scores_with_model(feats[L], Y, reg)
                    for L in layers
                }
                df_r2 = pd.DataFrame(r2_dict, columns=layers, index=dim_names)
                df_r2.to_csv(
                    os.path.join(outd, f"r2_{model_name}_{ds}_{combo}_layers.csv"),
                    index_label="dimension"
                )

                # permutation + FDR (conditional)
                pvals_fdr_dict = {L: np.ones(len(dim_names)) for L in layers}
                if run_permutation:
                    perm_folder = os.path.join(outd, "permutation_testing")
                    os.makedirs(perm_folder, exist_ok=True)
                    pvals_raw = {}

                    def single_perm(X, Y, reg, seed):
                        rs = np.random.RandomState(seed)
                        perm = rs.permutation(Y.shape[0])
                        Yp   = Y[perm]
                        return compute_r2_scores_with_model(X, Yp, reg)

                    for L in layers:
                        X = feats[L]
                        print(f"\n    → Layer {L}: starting permutation test")
                        layer_start = time.time()

                        if isinstance(reg, (LinearRegression, Ridge)):
                            # 1) pseudo-inverse
                            X_pinv = np.linalg.pinv(X)

                            # 2) total SS
                            Y_mean = Y.mean(axis=0, keepdims=True)
                            ss_tot = ((Y - Y_mean)**2).sum(axis=0)

                            # 3) generate all permuted Ys with a progress bar
                            n_samples, n_dims = Y.shape
                            rng = np.random.RandomState(42)
                            perm_idx = np.empty((n_perm, n_samples), dtype=int)
                            for i in tqdm(range(n_perm), desc=f"{L} perm-idx", leave=False):
                                perm_idx[i] = rng.permutation(n_samples)
                            Yp_all = Y[perm_idx]  # shape (n_perm, n_samples, n_dims)

                            # 4) solve all B’s & predict
                            B_all    = np.einsum('fn,pnd->pfd', X_pinv, Yp_all)
                            Yhat_all = np.einsum('nf,pfd->pnd',    X,    B_all)

                            # 5) residual SS
                            ss_res = ((Yp_all - Yhat_all)**2).sum(axis=1)

                            # 6) R²
                            perms   = 1.0 - ss_res / ss_tot[None,:]

                        else:
                            # fallback for SVR (or any non-linear regressor)
                            def single_perm(X, Y, reg, seed):
                                rs   = np.random.RandomState(seed)
                                perm = rs.permutation(Y.shape[0])
                                Yp   = Y[perm]
                                return compute_r2_scores_with_model(X, Yp, reg)

                            # run the original cross-val loop, but show progress
                            seeds = list(range(n_perm))
                            perms = Parallel(n_jobs=-1)(
                                delayed(single_perm)(X, Y, reg, seed)
                                for seed in tqdm(seeds, desc=f"{L} SVR-perms", leave=False)
                            )
                            perms = np.vstack(perms)

                        # save + p-values as before …
                        np.save(os.path.join(perm_folder, f"{L}_perm_r2.npy"), perms)
                        real_r2 = r2_dict[L]
                        p_raw   = ((perms >= real_r2[None,:]).sum(axis=0) + 1) / (n_perm+1)
                        pvals_raw[L] = p_raw
                        pd.DataFrame({
                            'dimension':   dim_names,
                            'p_value_raw': p_raw
                        }).to_csv(
                            os.path.join(perm_folder, f"{L}_p_values_raw.csv"),
                            index=False
                        )

                        elapsed = time.time() - layer_start
                        print(f"    ✓ Layer {L} done in {elapsed:.1f}s")

                    # FDR across layers
                    all_p = np.hstack([pvals_raw[L] for L in layers])
                    _, pvals_fdr_flat, _, _ = multipletests(
                        all_p, alpha=alpha_thresh, method='fdr_bh'
                    )
                    pvals_fdr = pvals_fdr_flat.reshape(len(layers), -1)
                    pvals_fdr_dict = {L: pvals_fdr[i] for i, L in enumerate(layers)}
                    for i, L in enumerate(layers):
                        pd.DataFrame({
                            'dimension':     dim_names,
                            'p_value_fdr':   pvals_fdr[i]
                        }).to_csv(
                            os.path.join(perm_folder, f"{L}_p_values_fdr.csv"),
                            index=False
                        )

                print(" done.")
        print(f"All {model_name} on {ds} done.")
    print(f"Finished dataset {ds}")

# Create plots for all combinations
render_all_plots()
