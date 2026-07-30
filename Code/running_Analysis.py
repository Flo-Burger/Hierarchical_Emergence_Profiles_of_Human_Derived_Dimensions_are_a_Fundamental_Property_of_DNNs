#!/usr/bin/env python3
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import sys
import time
import numpy as np
import scipy.io as sio
import torch
from statsmodels.stats.multitest import multipletests
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from PIL import Image
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
DEBUG_N_DIMS = None  # set to an int to restrict to N dimensions for testing

START_FROM_MODEL = None  # set to None to run all models

OVERWRITE = True  # set to True to recompute and overwrite existing results

PCA_COMPONENTS_LIST = [0.95, 50, 100, 200]

REGRESSORS = {
    "Linear": LinearRegression(),
    "Ridge":  Ridge(alpha=1.0),
    "NNLS":   LinearRegression(positive=True),
}

CV = 10 # Cross-validation splits for each prediction
# Permutation testing is computationally expensive
run_permutation = True
n_perm            = 1000


alpha_thresh      = 0.05   # FDR threshold


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

def compute_r2_scores_with_model(X, Y, reg, dim_names=None):
    out = np.zeros(Y.shape[1])
    labels = dim_names if dim_names is not None else [str(j) for j in range(Y.shape[1])]
    pbar = tqdm(enumerate(labels), total=len(labels), desc="  dims", leave=False)
    for j, name in pbar:
        pbar.set_postfix(dim=name)
        out[j] = cross_val_score(
            reg, X, Y[:, j], cv=CV, scoring="r2", n_jobs=-1
        ).mean()
    return out


def compute_r2_linear_kfold(X, Y, k=10, alpha=0.0):
    """Vectorized K-fold CV for multi-output OLS/Ridge (with intercept) — solves all dims per fold.
    alpha=0 -> OLS, alpha>0 -> Ridge (centered normal equations, same as sklearn Ridge)."""
    kf = KFold(n_splits=k, shuffle=False)
    Y_mean = Y.mean(axis=0)
    ss_res = np.zeros(Y.shape[1])
    ss_tot = np.zeros(Y.shape[1])
    p = X.shape[1]
    for train_idx, test_idx in kf.split(X):
        x_mean = X[train_idx].mean(axis=0)
        y_mean = Y[train_idx].mean(axis=0)
        Xc = X[train_idx] - x_mean
        Yc = Y[train_idx] - y_mean
        if alpha == 0.0:
            B, _, _, _ = np.linalg.lstsq(Xc, Yc, rcond=None)
        else:
            B = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ Yc)
        pred = (X[test_idx] - x_mean) @ B + y_mean
        ss_res += ((Y[test_idx] - pred) ** 2).sum(axis=0)
        ss_tot += ((Y[test_idx] - Y_mean) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-10)


def _nnls_pgd(XtX, XtY, step, max_iter=3000, tol=1e-8):
    """FISTA projected gradient descent solving NNLS for all dims simultaneously.

    Minimises ||XB - Y||^2 s.t. B >= 0 for all columns of Y at once.
    XtX: (p,p), XtY: (p,d), step: 1/lambda_max(XtX) → B: (p,d) with B >= 0.
    """
    B = np.zeros_like(XtY)
    B_prev = B.copy()
    t = 1.0
    for _ in range(max_iter):
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        M = B + ((t - 1.0) / t_new) * (B - B_prev)
        B_new = np.maximum(M - step * (XtX @ M - XtY), 0.0)
        if np.linalg.norm(B_new - B, "fro") < tol * max(1.0, np.linalg.norm(B_new, "fro")):
            return B_new
        B_prev, B, t = B, B_new, t_new
    return B


def _nnls_pgd_batch(XtX, XtY_all, step, max_iter=3000, tol=1e-8):
    """Same as _nnls_pgd but for a batch of right-hand sides sharing the same XtX.
    XtX: (p,p), XtY_all: (batch,p,d) -> B_all: (batch,p,d) with B_all >= 0."""
    B = np.zeros_like(XtY_all)
    B_prev = B.copy()
    t = 1.0
    for _ in range(max_iter):
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        M = B + ((t - 1.0) / t_new) * (B - B_prev)
        B_new = np.maximum(M - step * (np.einsum('pq,bqd->bpd', XtX, M) - XtY_all), 0.0)
        if np.linalg.norm(B_new - B) < tol * max(1.0, np.linalg.norm(B_new)):
            return B_new
        B_prev, B, t = B, B_new, t_new
    return B


def compute_r2_nnls_kfold(X, Y, k=10):
    """Vectorized K-fold CV for multi-output NNLS (with intercept) — solves all dims per fold."""
    kf = KFold(n_splits=k, shuffle=False)
    Y_mean = Y.mean(axis=0)
    ss_res = np.zeros(Y.shape[1])
    ss_tot = np.zeros(Y.shape[1])
    for train_idx, test_idx in kf.split(X):
        x_mean = X[train_idx].mean(axis=0)
        y_mean = Y[train_idx].mean(axis=0)
        Xc = X[train_idx] - x_mean
        Yc = Y[train_idx] - y_mean
        XtX = Xc.T @ Xc
        step = 1.0 / max(float(np.linalg.eigvalsh(XtX)[-1]), 1e-12)
        B = _nnls_pgd(XtX, Xc.T @ Yc, step)
        pred = (X[test_idx] - x_mean) @ B + y_mean
        ss_res += ((Y[test_idx] - pred) ** 2).sum(axis=0)
        ss_tot += ((Y[test_idx] - Y_mean) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-10)


# Main Analysis
for ds in ["THINGS66d", "THINGS", "STUFF"]:
    print(f"\n\n===== DATASET: {ds} =====")
    if ds == "THINGS":
        # For THINGS (49d SPOSE embedding)
        embedding_file = os.path.join(THINGS_DIR, "spose_embedding_49d_sorted.txt")
        labels_file    = os.path.join(THINGS_DIR, "labels.mat")
        images_file    = os.path.join(THINGS_DIR, "im.mat")

        Y         = np.loadtxt(embedding_file)
        dim_names = [l[0] for l in safe_loadmat(labels_file, 'labels').flatten()]
        images    = safe_loadmat(images_file, 'im').flatten()
        assert Y.shape[0] == len(images)

    elif ds == "THINGS66d":
        # For THINGS (66d SPOSE embedding, semantic/visual analysis)
        embedding_file = os.path.join(THINGS_DIR, "spose_embedding_66d_sorted.txt")
        labels_file    = os.path.join(THINGS_DIR, "labels_spose_66d_short.txt")
        images_file    = os.path.join(THINGS_DIR, "im.mat")

        Y         = np.loadtxt(embedding_file)
        with open(labels_file) as f:
            dim_names = [l.strip() for l in f.readlines()]
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

    if DEBUG_N_DIMS is not None:
        Y = Y[:, :DEBUG_N_DIMS]
        dim_names = dim_names[:DEBUG_N_DIMS]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42); np.random.seed(42)

    reached_start = (START_FROM_MODEL is None)
    for model_name, extract_fn in MODEL_EXTRACTORS.items():
        if not reached_start:
            if model_name == START_FROM_MODEL:
                reached_start = True
            else:
                print(f"\n--- Skipping {model_name} on {ds} (before START_FROM_MODEL) ---")
                continue

        print(f"\n--- Model: {model_name} on {ds} ---")
        base_out = os.path.join(RESULTS_ROOT, f"{model_name}_{ds}")
        os.makedirs(base_out, exist_ok=True)

        # Extract activations (per your extractor)
        activations = extract_fn(images, device)
        layers       = list(activations.keys())

        for PCA_K in PCA_COMPONENTS_LIST:
            # Compute PCA features once per PCA_K, shared across all regressors
            print(f"  Computing PCA (k={PCA_K})…", end="", flush=True)
            pca_feats = {}
            for L in layers:
                arr = np.stack(activations[L], axis=0)
                k   = min(PCA_K, arr.shape[1]) if isinstance(PCA_K, (int, float)) else arr.shape[1]
                pca = PCA(n_components=k,
                          svd_solver="full" if isinstance(PCA_K, float) else "auto")
                pca_feats[L] = pca.fit_transform(arr)
            print(" done.")

            for reg_name, reg in REGRESSORS.items():
                combo = f"PCA{PCA_K}_{reg_name}"
                outd  = os.path.join(base_out, combo)
                os.makedirs(outd, exist_ok=True)

                csv_path = os.path.join(outd, f"r2_{model_name}_{ds}_{combo}_layers.csv")
                if os.path.exists(csv_path) and not OVERWRITE:
                    print(f"  Skipping {combo} (results already exist)")
                    continue

                print(f"  Running {combo}…", end="", flush=True)

                # original R²
                if isinstance(reg, (LinearRegression, Ridge)) and not getattr(reg, 'positive', False):
                    # alpha=0 for Linear, reg.alpha for Ridge
                    alpha = getattr(reg, 'alpha', 0.0)
                    r2_dict = {L: compute_r2_linear_kfold(pca_feats[L], Y, k=CV, alpha=alpha) for L in layers}
                elif getattr(reg, 'positive', False):  # NNLS
                    r2_dict = {L: compute_r2_nnls_kfold(pca_feats[L], Y, k=CV) for L in layers}
                else:
                    r2_dict = {
                        L: compute_r2_scores_with_model(pca_feats[L], Y, reg, dim_names=dim_names)
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

                    for L in layers:
                        X = pca_feats[L]
                        print(f"\n    → Layer {L}: starting permutation test")
                        layer_start = time.time()

                        if isinstance(reg, (LinearRegression, Ridge)) and not getattr(reg, 'positive', False):
                            alpha = getattr(reg, 'alpha', 0.0)
                            n_samples, n_dims = Y.shape

                            # global mean, same for every permutation
                            Y_mean_global = Y.mean(axis=0, keepdims=True)
                            ss_tot = ((Y - Y_mean_global) ** 2).sum(axis=0)

                            # generate all permuted Ys with a progress bar
                            rng = np.random.RandomState(42)
                            perm_idx = np.empty((n_perm, n_samples), dtype=int)
                            for i in tqdm(range(n_perm), desc=f"{L} perm-idx", leave=False):
                                perm_idx[i] = rng.permutation(n_samples)
                            Yp_all = Y[perm_idx]  # shape (n_perm, n_samples, n_dims)

                            # null via k-fold CV (not in-sample), same folds as the true R²
                            kf = KFold(n_splits=CV, shuffle=False)
                            ss_res = np.zeros((n_perm, n_dims))
                            p = X.shape[1]
                            perm_chunk = 100
                            for train_idx, test_idx in tqdm(list(kf.split(X)), desc=f"{L} {reg_name}-perm-folds", leave=False):
                                x_mean   = X[train_idx].mean(axis=0)
                                Xc_train = X[train_idx] - x_mean
                                Xc_test  = X[test_idx]  - x_mean
                                if alpha == 0:
                                    pinv = np.linalg.pinv(Xc_train)
                                else:
                                    pinv = np.linalg.solve(Xc_train.T @ Xc_train + alpha * np.eye(p), Xc_train.T)

                                for start in range(0, n_perm, perm_chunk):
                                    stop = min(start + perm_chunk, n_perm)
                                    Yp_train = Yp_all[start:stop, train_idx, :]
                                    y_mean   = Yp_train.mean(axis=1, keepdims=True)
                                    Yc_train = Yp_train - y_mean

                                    B_all = np.einsum('fn,bnd->bfd', pinv, Yc_train)
                                    pred  = np.einsum('nf,bfd->bnd', Xc_test, B_all) + y_mean
                                    ss_res[start:stop] += ((Yp_all[start:stop, test_idx, :] - pred) ** 2).sum(axis=1)

                            perms = 1.0 - ss_res / ss_tot[None, :]

                        elif getattr(reg, 'positive', False):
                            n_samples, n_dims = Y.shape
                            Y_mean_global = Y.mean(axis=0, keepdims=True)
                            ss_tot = np.maximum(((Y - Y_mean_global) ** 2).sum(axis=0), 1e-10)

                            rng = np.random.RandomState(42)
                            perm_idx = np.empty((n_perm, n_samples), dtype=int)
                            for i in tqdm(range(n_perm), desc=f"{L} perm-idx", leave=False):
                                perm_idx[i] = rng.permutation(n_samples)
                            Yp_all = Y[perm_idx]  # shape (n_perm, n_samples, n_dims)

                            # same k-fold CV null as above, solved via batched PGD
                            kf = KFold(n_splits=CV, shuffle=False)
                            ss_res = np.zeros((n_perm, n_dims))
                            perm_chunk = 100
                            for train_idx, test_idx in tqdm(list(kf.split(X)), desc=f"{L} NNLS-perm-folds", leave=False):
                                x_mean   = X[train_idx].mean(axis=0)
                                Xc_train = X[train_idx] - x_mean
                                Xc_test  = X[test_idx]  - x_mean
                                XtX  = Xc_train.T @ Xc_train
                                step = 1.0 / max(float(np.linalg.eigvalsh(XtX)[-1]), 1e-12)

                                for start in range(0, n_perm, perm_chunk):
                                    stop = min(start + perm_chunk, n_perm)
                                    Yp_train = Yp_all[start:stop, train_idx, :]
                                    y_mean   = Yp_train.mean(axis=1, keepdims=True)
                                    Yc_train = Yp_train - y_mean
                                    XtY_all  = np.einsum('nf,bnd->bfd', Xc_train, Yc_train)

                                    B_all = _nnls_pgd_batch(XtX, XtY_all, step, max_iter=1000, tol=1e-6)
                                    pred  = np.einsum('nf,bfd->bnd', Xc_test, B_all) + y_mean
                                    ss_res[start:stop] += ((Yp_all[start:stop, test_idx, :] - pred) ** 2).sum(axis=1)

                            perms = 1.0 - ss_res / ss_tot[None, :]

                        else:
                            # fallback for any non-linear regressor
                            def single_perm(X, Y, reg, seed):
                                rs   = np.random.RandomState(seed)
                                perm = rs.permutation(Y.shape[0])
                                Yp   = Y[perm]
                                return compute_r2_scores_with_model(X, Yp, reg)

                            # run the original cross-val loop, but show progress
                            seeds = list(range(n_perm))
                            perms = Parallel(n_jobs=-1)(
                                delayed(single_perm)(X, Y, reg, seed)
                                for seed in tqdm(seeds, desc=f"{L} Ridge-perms", leave=False)
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
                time.sleep(5)
        print(f"All {model_name} on {ds} done.")
    print(f"Finished dataset {ds}")

# Create plots for all combinations
render_all_plots()
