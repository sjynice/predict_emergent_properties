import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import re
import glob
import pandas as pd
import torch

# %%
def flat_to_matrix_with_zero_diag(X_flat, N):
    X_mat = np.zeros((X_flat.shape[0], N, N))
    mask = ~np.eye(N, dtype=bool)
    X_mat[:, mask] = X_flat
    return X_mat

# Used when the entire pipeline needs to be tensors
def flat_to_matrix_with_zero_diag_torch(X_flat, N):
    batch_size = X_flat.shape[0]
    X_mat = torch.zeros( (batch_size, N, N), dtype=X_flat.dtype, device=X_flat.device)
    mask = ~torch.eye(N, dtype=torch.bool, device=X_flat.device)
    X_mat[:, mask] = X_flat
    return X_mat

# %%
# Input narray contains both Xnum and ynum, but the last two character columns need to be removed
# Used when the output is steady-state count, steady-state entropy, or diversity

# Single dataset train/test split
def prepare_train_test_split(
    narray,
    NumSpecies,
    featgam,
    datasize,
    test_size=0.2,
    random_state=42,
    sample_random=True
):
    # ---------- random sample ----------
    if sample_random and datasize < narray.shape[0]:
        idx = np.random.choice(narray.shape[0], size=datasize, replace=False)
        narray = narray[idx]

    # ---------- X / y ----------
    X = narray[:, 0:NumSpecies + featgam + NumSpecies]
    y = narray[:, NumSpecies + featgam + NumSpecies + 1]

    # ---------- split input ----------
    X_g = X[:, :NumSpecies]
    X_A_flat = X[:, NumSpecies:NumSpecies + featgam]
    X_d = X[:, NumSpecies + featgam:NumSpecies + featgam + NumSpecies]

    X_A = flat_to_matrix_with_zero_diag(X_A_flat, NumSpecies)

    # ---------- train / test ----------
    idx_all = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=random_state
    )

    Xg_tr, Xg_te = X_g[idx_tr], X_g[idx_te]
    XA_tr, XA_te = X_A[idx_tr], X_A[idx_te]
    Xd_tr, Xd_te = X_d[idx_tr], X_d[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # ---------- scale ----------
    scaler_g = StandardScaler()
    scaler_d = StandardScaler()

    Xg_tr = scaler_g.fit_transform(Xg_tr)
    Xg_te = scaler_g.transform(Xg_te)

    Xd_tr = scaler_d.fit_transform(Xd_tr)
    Xd_te = scaler_d.transform(Xd_te)

    return {
        "Xg_tr": Xg_tr,
        "Xg_te": Xg_te,
        "XA_tr": XA_tr,
        "XA_te": XA_te,
        "Xd_tr": Xd_tr,
        "Xd_te": Xd_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "idx_tr": idx_tr,
        "idx_te": idx_te,
        "scaler_g": scaler_g,
        "scaler_d": scaler_d
    }

# Multi-dataset train/test split
def process_narray(
    narray,
    NumSpecies,
    featgam,
    task_num
):
    """
    Split narray -> Xg, XA, Xd, y (no scaling)
    """

    # ---------- X ----------
    X_all = narray[:, :NumSpecies + featgam + NumSpecies]

    # ---------- y ----------
    if task_num == 3 + 2 * NumSpecies: # Oscillation
        y = (narray[:, NumSpecies + featgam + NumSpecies + task_num] != 0).astype(int)
    elif task_num == 0: # Steady-state count
        y = narray[:, NumSpecies + featgam + NumSpecies + task_num].astype(int) - 1
    elif task_num == 3 + NumSpecies: # switching species
        y = narray[:, NumSpecies + featgam + NumSpecies + task_num: NumSpecies + featgam + NumSpecies + task_num + NumSpecies]
    else:
        y = narray[:, NumSpecies + featgam + NumSpecies + task_num]

    # ---------- split input ----------
    Xg = X_all[:, :NumSpecies]
    XA_flat = X_all[:, NumSpecies:NumSpecies + featgam]
    Xd = X_all[:, NumSpecies + featgam:NumSpecies + featgam + NumSpecies]

    XA = flat_to_matrix_with_zero_diag(XA_flat, NumSpecies)

    return Xg, XA, Xd, y

def prepare_train(
    narray_tr,
    NumSpecies,
    featgam,
    task_num,
    sample_size=None,
    random_state=None
):
    if random_state is not None:
        np.random.seed(random_state)

    # ---------- sampling ----------
    if sample_size is not None:
        idx = np.random.choice(narray_tr.shape[0], size=sample_size, replace=False)
    else:
        idx = np.random.choice(narray_tr.shape[0], size=narray_tr.shape[0], replace=False)

    narray_tr = narray_tr[idx]

    # ---------- split ----------
    Xg, XA, Xd, y = process_narray(
        narray_tr, NumSpecies, featgam, task_num
    )

    # ---------- scale ----------
    scaler_g = StandardScaler()
    scaler_d = StandardScaler()

    Xg = scaler_g.fit_transform(Xg)
    Xd = scaler_d.fit_transform(Xd)

    return Xg, XA, Xd, y, scaler_g, scaler_d, idx

def prepare_test(
    narray_te,
    NumSpecies,
    featgam,
    scaler_g,
    scaler_d,
    task_num,
    sample_size=None,
    random_state=None
):
    if random_state is not None:
        np.random.seed(random_state)

    # ---------- sampling ----------
    if sample_size is not None:
        idx = np.random.choice(narray_te.shape[0], size=sample_size, replace=False)
    else:
        idx = np.random.choice(narray_te.shape[0], narray_te.shape[0], replace=False)

    narray_te = narray_te[idx]

    # ---------- split ----------
    Xg, XA, Xd, y = process_narray(
        narray_te, NumSpecies, featgam, task_num
    )

    # ---------- scale ----------
    Xg = scaler_g.transform(Xg)
    Xd = scaler_d.transform(Xd)

    return Xg, XA, Xd, y, idx


# Used when output is abundance vector or switching species vector
def process_narray2(
    narray,
    NumSpecies,
    featgam,
    task_num
):
    """
    Split narray -> Xg, XA, Xd, y (no scaling)
    """

    # ---------- X ----------
    X_all = narray[:, :NumSpecies + featgam + NumSpecies]

    # ---------- y ----------
    y = narray[:, NumSpecies + featgam + NumSpecies + task_num: NumSpecies + featgam + NumSpecies + task_num + NumSpecies]

    # ---------- split input ----------
    Xg = X_all[:, :NumSpecies]
    XA_flat = X_all[:, NumSpecies:NumSpecies + featgam]
    Xd = X_all[:, NumSpecies + featgam:NumSpecies + featgam + NumSpecies]

    XA = flat_to_matrix_with_zero_diag(XA_flat, NumSpecies)

    return Xg, XA, Xd, y

def prepare_train2(
    narray_tr,
    NumSpecies,
    featgam,
    task_num,
    sample_size=None,
    random_state=None
):
    if random_state is not None:
        np.random.seed(random_state)

    # ---------- sampling ----------
    if sample_size is not None:
        idx = np.random.choice(narray_tr.shape[0], size=sample_size, replace=False)
    else:
        idx = np.random.choice(narray_tr.shape[0], size=narray_tr.shape[0], replace=False)

    narray_tr = narray_tr[idx]

    # ---------- split ----------
    Xg, XA, Xd, y = process_narray2(
        narray_tr, NumSpecies, featgam, task_num
    )

    # ---------- scale ----------
    scaler_g = StandardScaler()
    scaler_d = StandardScaler()

    Xg = scaler_g.fit_transform(Xg)
    Xd = scaler_d.fit_transform(Xd)

    return Xg, XA, Xd, y, scaler_g, scaler_d, idx

def prepare_test2(
    narray_te,
    NumSpecies,
    featgam,
    scaler_g,
    scaler_d,
    task_num,
    sample_size=None,
    random_state=None
):
    if random_state is not None:
        np.random.seed(random_state)

    # ---------- sampling ----------
    if sample_size is not None:
        idx = np.random.choice(narray_te.shape[0], size=sample_size, replace=False)
    else:
        idx = np.random.choice(narray_te.shape[0], narray_te.shape[0], replace=False)

    narray_te = narray_te[idx]

    # ---------- split ----------
    Xg, XA, Xd, y = process_narray2(
        narray_te, NumSpecies, featgam, task_num
    )

    # ---------- scale ----------
    Xg = scaler_g.transform(Xg)
    Xd = scaler_d.transform(Xd)

    return Xg, XA, Xd, y, idx


# Version with real species mask, standardizing only real species, and changed train-val split + scaling order
def process_narray_mask(
    narray,
    richness_label,
    NumSpecies,
    featgam,
    task_num
):
    """
    Split narray -> Xg, XA, Xd, y, node_mask

    Parameters
    ----------
    narray : ndarray, shape (n_samples, n_features)
        Community data already padded to NumSpecies.

    richness_label : array-like, shape (n_samples,)
        True richness of each sample, returned by load_split_dataset2.

    NumSpecies : int
        Maximum number of species, e.g. 20, not the actual richness of a single sample.

    featgam : int
        Number of flattened interaction features, should equal
        NumSpecies * (NumSpecies - 1).

    task_num : int
        Label index for the prediction task:
        0 = steady-state count
        1 = steady-state entropy
        2 = diversity
        3 = first species abundance
        3 + NumSpecies = switching species
        3 + 2 * NumSpecies = oscillation

    Returns
    -------
    Xg : ndarray, shape (n_samples, NumSpecies)
    XA : ndarray, shape (n_samples, NumSpecies, NumSpecies)
    Xd : ndarray, shape (n_samples, NumSpecies)
    y : ndarray
    node_mask : ndarray, shape (n_samples, NumSpecies)
        True means real species, False means padding.
    """

    n_samples = narray.shape[0]

    # ---------- 1. Check input ----------
    if featgam != NumSpecies * (NumSpecies - 1):
        raise ValueError(
            f"featgam should be NumSpecies*(NumSpecies-1)="
            f"{NumSpecies * (NumSpecies - 1)}, "
            f"got {featgam}"
        )

    richness_label = np.asarray(richness_label).reshape(-1)

    if len(richness_label) != n_samples:
        raise ValueError(
            f"Sample count mismatch: narray has {n_samples} rows, "
            f"richness_label has {len(richness_label)} values."
        )

    if not np.isfinite(richness_label).all():
        raise ValueError("NaN or Inf found in richness_label.")

    if not np.all(richness_label == np.floor(richness_label)):
        raise ValueError("richness_label must be integers.")

    richness_label = richness_label.astype(int)

    # ---------- 2. Construct node_mask ----------
    # According to padding_Xnum_to_maxspecies construction rule:
    # Real species are always placed in the first 'richness' positions.
    node_mask = (
        np.arange(NumSpecies)[None, :]
        < richness_label[:, None]
    )

    # ---------- 3. Split input features ----------
    n_features = NumSpecies + featgam + NumSpecies

    if narray.shape[1] < n_features + task_num + 1:
        raise ValueError("narray has insufficient columns to extract labels for the specified task.")

    X_all = narray[:, :n_features]

    Xg = X_all[:, :NumSpecies]
    XA_flat = X_all[:, NumSpecies:NumSpecies + featgam]
    Xd = X_all[:, NumSpecies + featgam: NumSpecies + featgam + NumSpecies]

    # ---------- 4. Reconstruct full NumSpecies × NumSpecies matrix ----------
    XA = flat_to_matrix_with_zero_diag(XA_flat, NumSpecies)

    # ---------- 5. Extract labels ----------
    if task_num == 3 + 2 * NumSpecies: # Oscillation
        y = (narray[:, NumSpecies + featgam + NumSpecies + task_num] != 0).astype(int)
    elif task_num == 0: # Steady-state count
        y = narray[:, NumSpecies + featgam + NumSpecies + task_num].astype(int) - 1
    elif task_num == 3 + NumSpecies: # switching species
        y = narray[:, NumSpecies + featgam + NumSpecies + task_num: NumSpecies + featgam + NumSpecies + task_num + NumSpecies]
    else:
        y = narray[:, NumSpecies + featgam + NumSpecies + task_num]

    return Xg, XA, Xd, y, node_mask

def fit_preprocessors(Xg, Xd, node_mask):
    """
    Fit scaler for mu and D using only real species in the training set.

    Xg, Xd:    [n_samples, NumSpecies]
    node_mask: [n_samples, NumSpecies]
        True means real species, False means padding.

    """
    node_mask = np.asarray(node_mask, dtype=bool)

    if not node_mask.any():
        raise ValueError("No real species in node_mask.")

    if not np.isfinite(Xg[node_mask]).all():
        raise ValueError("NaN or Inf found in Xg of real species.")

    if not np.isfinite(Xd[node_mask]).all():
        raise ValueError("NaN or Inf found in Xd of real species.")

    scaler_g = StandardScaler()
    scaler_d = StandardScaler()

    # Fit one global set of scaling parameters for each variable
    # Does not distinguish which position a species occupies
    scaler_g.fit(Xg[node_mask].reshape(-1, 1))
    scaler_d.fit(Xd[node_mask].reshape(-1, 1))

    return scaler_g, scaler_d

def transform_features(
    Xg, XA, Xd, node_mask,
    scaler_g, scaler_d
):
    """
    Standardize mu and D of real species.
    Keep gamma values as-is, but zero out species pairs involving padding.
    """
    node_mask = np.asarray(node_mask, dtype=bool)

    # ---------- 1. mu ----------
    Xg_scaled = np.zeros(Xg.shape, dtype=np.float32)
    Xg_scaled[node_mask] = scaler_g.transform(Xg[node_mask].reshape(-1, 1)).ravel()

    # ---------- 2. D ----------
    Xd_scaled = np.zeros(Xd.shape, dtype=np.float32)
    Xd_scaled[node_mask] = scaler_d.transform(Xd[node_mask].reshape(-1, 1)).ravel()

    # ---------- 3. gamma: no scaling, only pair mask ----------
    N = XA.shape[1]

    off_diagonal = ~np.eye(N, dtype=bool)

    pair_mask = (node_mask[:, :, None] & node_mask[:, None, :] & off_diagonal[None, :, :])

    # Check interaction values between real species
    if not np.isfinite(XA[pair_mask]).all():
        raise ValueError("NaN or Inf found in XA of real species pairs.")

    # Keep only off-diagonal elements where both endpoints are real species
    # No scaling, no mean subtraction
    XA_masked = np.zeros(XA.shape, dtype=np.float32)
    XA_masked[pair_mask] = XA[pair_mask]  # Double safeguard against nonzero values in padded parts, though our padded gamma should all be 0

    return Xg_scaled, XA_masked, Xd_scaled

def prepare_train_val_mask(
    narray_tr,
    richness_label_tr,
    NumSpecies,
    featgam,
    task_num,
    val_size=0.2,
    sample_size=None,
    random_state=42
):
    """
    Split original data first, then fit scaler using only the training subset.

    Returns
    -------
    train_data:
        Xg, XA, Xd, y, node_mask

    val_data:
        Xg, XA, Xd, y, node_mask

    preprocessors:
        scaler_g, scaler_d

    split_idx:
        selected, train, val row indices corresponding to original narray_tr
    """
    rng = np.random.default_rng(random_state)

    richness_label_tr = np.asarray(richness_label_tr).reshape(-1)

    n_total = len(narray_tr)

    if len(richness_label_tr) != n_total:
        raise ValueError("narray_tr and richness_label_tr have mismatched sample counts.")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between (0, 1).")

    # ---------- 1. Optional random sampling ----------
    if sample_size is not None:
        if not 2 <= sample_size <= n_total:
            raise ValueError(f"sample_size must be in [2, {n_total}].")

        idx_selected = rng.choice(n_total, size=sample_size, replace=False)
    else:
        idx_selected = np.arange(n_total)

    if len(idx_selected) < 2:
        raise ValueError("Insufficient samples to split into training and validation sets.")

    # ---------- 2. Ordinary random split, no stratification ----------
    idx_shuffled = rng.permutation(idx_selected)

    n_val = int(np.ceil(len(idx_shuffled) * val_size))
    n_val = max(1, min(n_val, len(idx_shuffled) - 1))

    idx_val = idx_shuffled[:n_val]
    idx_fit = idx_shuffled[n_val:]

    # ---------- 3. Parse training and validation sets separately ----------
    Xg_fit, XA_fit, Xd_fit, y_fit, mask_fit = process_narray_mask(narray_tr[idx_fit], richness_label_tr[idx_fit], NumSpecies, featgam, task_num)
    Xg_val, XA_val, Xd_val, y_val, mask_val = process_narray_mask(narray_tr[idx_val], richness_label_tr[idx_val], NumSpecies, featgam, task_num)
    
    # ---------- 4. Estimate scaler using only the training subset ----------
    scaler_g, scaler_d = fit_preprocessors(Xg_fit, Xd_fit, mask_fit)

    # ---------- 5. Transform both parts using the same parameters ----------
    Xg_fit, XA_fit, Xd_fit = transform_features(Xg_fit, XA_fit, Xd_fit, mask_fit, scaler_g, scaler_d)
    Xg_val, XA_val, Xd_val = transform_features(Xg_val, XA_val, Xd_val, mask_val, scaler_g, scaler_d)

    train_data = (Xg_fit, XA_fit, Xd_fit, y_fit, mask_fit)
    val_data = (Xg_val, XA_val, Xd_val, y_val, mask_val)

    preprocessors = {"scaler_g": scaler_g, "scaler_d": scaler_d}
    split_idx = {"selected": idx_selected, "train": idx_fit, "val": idx_val}

    print(f"Train shape = {len(idx_fit)}")
    print(f"Val shape   = {len(idx_val)}")

    if task_num == 3 + 2 * NumSpecies:
        print(f"Train positive rate = {np.mean(y_fit):.6f}")
        print(f"Val positive rate   = {np.mean(y_val):.6f}")

    return train_data, val_data, preprocessors, split_idx

def prepare_test_mask(
    narray_te,
    richness_label_te,
    NumSpecies,
    featgam,
    task_num,
    scaler_g,
    scaler_d,
    sample_size=None,
    random_state=42
):
    """
    Test set does not re-fit scaler; only uses the training set's scaler.
    Keep gamma values as-is; only mask invalid species pairs.
    """
    rng = np.random.default_rng(random_state)
    richness_label_te = np.asarray(richness_label_te).reshape(-1)
    n_total = len(narray_te)
    if len(richness_label_te) != n_total:
        raise ValueError("narray_te and richness_label_te have mismatched sample counts.")

    # ---------- 1. Optional random sampling ----------
    if sample_size is not None:
        if not 1 <= sample_size <= n_total:
            raise ValueError(f"sample_size must be in [1, {n_total}].")

        idx = rng.choice(n_total, size=sample_size, replace=False)
    else:
        # Use all, preserve original order
        idx = np.arange(n_total)

    # ---------- 2. Parse data ----------
    Xg, XA, Xd, y, node_mask = process_narray_mask(narray_te[idx], richness_label_te[idx], NumSpecies, featgam, task_num)

    # ---------- 3. Apply training-set-fitted scaler ----------
    Xg, XA, Xd = transform_features(Xg, XA, Xd, node_mask, scaler_g, scaler_d)

    return Xg, XA, Xd, y, node_mask, idx


# Used when narray only contains features and Xg, XA, Xd need to be returned
# Input Xnum_pad is array-type data
def process_narray_X(Xnum_pad, MaxNum, featgam2, scaler_g, scaler_d):
    
    Xnum_pad = np.atleast_2d(Xnum_pad)
    X_all = Xnum_pad[:, :MaxNum + featgam2 + MaxNum]

    Xg = X_all[:, :MaxNum]
    XA_flat = X_all[:, MaxNum:MaxNum + featgam2]
    Xd = X_all[:, MaxNum + featgam2:MaxNum + featgam2 + MaxNum]

    XA = flat_to_matrix_with_zero_diag(XA_flat, MaxNum)

    Xg = scaler_g.transform(Xg)
    Xd = scaler_d.transform(Xd)

    return Xg, XA, Xd

# Input Xnum_pad is tensor-type data
def process_tensor_X(Xnum_pad, NumSpecies, featgam, scaler_g, scaler_d):

    if Xnum_pad.ndim == 1:
        Xnum_pad = Xnum_pad.unsqueeze(0)
    X_all = Xnum_pad[:, :NumSpecies + featgam + NumSpecies]
    Xg = X_all[:, :NumSpecies]
    XA_flat = X_all[:, NumSpecies:NumSpecies+featgam]
    Xd = X_all[:, NumSpecies+featgam:]

    XA = flat_to_matrix_with_zero_diag_torch(XA_flat, NumSpecies)

    # scaler parameters
    g_mean = torch.as_tensor(scaler_g.mean_, dtype=torch.float32, device=Xnum_pad.device)
    g_std = torch.as_tensor(scaler_g.scale_, dtype=torch.float32, device=Xnum_pad.device)
    d_mean = torch.as_tensor(scaler_d.mean_, dtype=torch.float32, device=Xnum_pad.device)
    d_std = torch.as_tensor(scaler_d.scale_, dtype=torch.float32, device=Xnum_pad.device)
    Xg = (Xg - g_mean) / g_std
    Xd = (Xd - d_mean) / d_std

    return Xg, XA, Xd

# %%
# Padding related
# Used when padding Xnum to MaxNum
def pad_Xnum_to_maxspecies(Xnum, NumSpecies, MaxNum):

    Xnum = np.atleast_2d(Xnum)
    datasize = Xnum.shape[0]

    featgam_n = NumSpecies * (NumSpecies - 1)
    featgam_max = MaxNum * (MaxNum - 1)

    # Split
    mu = Xnum[:, :NumSpecies]
    gamma_flat = Xnum[:, NumSpecies:NumSpecies + featgam_n]
    D = Xnum[:, NumSpecies + featgam_n:]

    # ===== 1. Reconstruct gamma matrix (n x n, diagonal is 0) =====
    gamma_matrix = np.zeros((datasize, NumSpecies, NumSpecies))

    idx = 0
    for i in range(NumSpecies):
        for j in range(NumSpecies):
            if i != j:
                gamma_matrix[:, i, j] = gamma_flat[:, idx]
                idx += 1

    # ===== 2. Pad to MaxNum =====
    gamma_pad = np.zeros((datasize, MaxNum, MaxNum))
    gamma_pad[:, :NumSpecies, :NumSpecies] = gamma_matrix

    # ===== 3. Flatten again (remove diagonal) =====
    gamma_flat_pad = np.zeros((datasize, featgam_max))

    idx = 0
    for i in range(MaxNum):
        for j in range(MaxNum):
            if i != j:
                gamma_flat_pad[:, idx] = gamma_pad[:, i, j]
                idx += 1

    # ===== 4. Pad mu and D =====
    mu_pad = np.zeros((datasize, MaxNum))
    D_pad = np.zeros((datasize, MaxNum))

    mu_pad[:, :NumSpecies] = mu
    D_pad[:, :NumSpecies] = D

    # ===== 5. Concatenate =====
    Xnum_pad = np.concatenate([mu_pad, gamma_flat_pad, D_pad], axis=1)

    return Xnum_pad

# Used to recover real species data from padded Xnum
def recovery_Xnum(Xnum_pad, NumSpecies, MaxNum):
    Xnum_pad = np.atleast_2d(Xnum_pad)
    datasize = Xnum_pad.shape[0]
    featgam_max = MaxNum * (MaxNum - 1)

    # Split
    gamma_flat_pad = Xnum_pad[:, MaxNum:MaxNum + featgam_max]
    D_pad = Xnum_pad[:, MaxNum + featgam_max:]

    # Reconstruct gamma matrix (MaxNum x MaxNum, diagonal is 0)
    gamma_matrix_pad = np.zeros((datasize, MaxNum, MaxNum))

    idx = 0
    for i in range(MaxNum):
        for j in range(MaxNum):
            if i != j:
                gamma_matrix_pad[:, i, j] = gamma_flat_pad[:, idx]
                idx += 1

    # Take only the first NumSpecies species part
    mu_real = Xnum_pad[:, :NumSpecies]
    D_real = D_pad[:, :NumSpecies]

    gamma_matrix_real = gamma_matrix_pad[:, :NumSpecies, :NumSpecies]

    # Flatten again (remove diagonal)
    featgam_real = NumSpecies * (NumSpecies - 1)
    gamma_flat_real = np.zeros((datasize, featgam_real))

    idx = 0
    for i in range(NumSpecies):
        for j in range(NumSpecies):
            if i != j:
                gamma_flat_real[:, idx] = gamma_matrix_real[:, i, j]
                idx += 1

    Xnum_reduced = np.concatenate([mu_real, gamma_flat_real, D_real], axis=1)

    return Xnum_reduced

# Extract the interaction matrix (flattened version) of real species from padded Xnum
def extract_gamma_real_flatten(Xnum_pad, NumSpecies, MaxNum):
    datasize = Xnum_pad.shape[0]
    featgam_max = MaxNum * (MaxNum - 1)
    gamma_flat = Xnum_pad[:, MaxNum:MaxNum + featgam_max]
    gamma_mat = np.zeros((datasize, MaxNum, MaxNum)) # Reconstruct into matrix

    idx = 0
    for i in range(MaxNum):
        for j in range(MaxNum):
            if i != j:
                gamma_mat[:, i, j] = gamma_flat[:, idx]
                idx += 1

    sub_mat = gamma_mat[:, :NumSpecies, :NumSpecies] # Take only real species part
    mask = ~np.eye(NumSpecies, dtype=bool)
    gamma_real = sub_mat[:, mask].reshape(datasize, -1) # Remove diagonal

    return gamma_real # Finally get the de-zeroed, reduced interaction matrix (flattened version)

# %%
# For joint training scenarios
# For an already partitioned training set, use this function to get all data
def load_training_dataset_from_folder(folder_path, sample_size=None, random_state=None, sort_files=True):
    if random_state is not None:
        np.random.seed(random_state)
    # ---------- Find all csv files ----------
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))

    if len(file_list) == 0:
        raise ValueError(f"No csv files found in folder: {folder_path}")

    # ---------- Sort by integers in x-y_z ----------
    if sort_files:
        def extract_numbers(file_path):
            filename = os.path.basename(file_path)
            match = re.match(r"(\d+)-(\d+)_(\d+)\.csv", filename)
            if match:
                return tuple(map(int, match.groups()))
            return (float("inf"), float("inf"), float("inf"))

        file_list = sorted(file_list, key=extract_numbers)

    # ---------- Read one by one ----------
    df_list = []
    for file in file_list:
        df = pd.read_csv(file).iloc[:, :-2] # Remove the last two character columns

        if sample_size is not None:
            n = min(sample_size, len(df)) # Prevent sample_size from exceeding data count and causing errors
            idx = np.random.choice(len(df), size=n, replace=False)
            df = df.iloc[idx]
        else:
            idx = np.random.choice(len(df), size=len(df), replace=False)
            df = df.iloc[idx]
        
        df = df.reset_index(drop=True) # Reset row indices
        df_list.append(df)
        
    # ---------- Concatenate by rows ----------
    df_all = pd.concat(df_list, axis=0, ignore_index=True)

    # ---------- Convert to ndarray ----------
    narray_tr = df_all.values
    print(f"Loaded {len(file_list)} files, total {narray_tr.shape[0]} records.")

    return narray_tr, file_list


# Split data at each richness level into 8:2
def split_by_richness(folder_path, test_ratio=0.2, random_state=42):
    np.random.seed(random_state)

    file_list = glob.glob(os.path.join(folder_path, "*.csv"))

    richness_data = {}
    for file in file_list:

        filename = os.path.basename(file)
        match = re.match(r"(\d+)-(\d+)_(\d+)\.csv", filename)

        if match is None:
            continue

        richness = int(match.group(1))

        df = pd.read_csv(file)

        narray = df.iloc[:, :-2].values

        if richness not in richness_data:
            richness_data[richness] = []

        richness_data[richness].append(narray)

    split_data = {}

    for richness in sorted(richness_data):

        narray = np.vstack(richness_data[richness])

        idx = np.random.permutation(len(narray))

        n_test = int(len(narray) * test_ratio)

        idx_te = idx[:n_test]
        idx_tr = idx[n_test:]

        split_data[richness] = {
            "tr": narray[idx_tr],
            "te": narray[idx_te]
        }

        print(
            f"Richness={richness}, "
            f"Files={len(richness_data[richness])}, "
            f"Train={len(idx_tr)}, "
            f"Test={len(idx_te)}"
        )

    return split_data

# Generate training or test set, can choose which richness levels to include
def extract_from_split(split_data, richness_list, subset="tr"):
    data_list = []
    for richness in richness_list:
        if richness not in split_data:
            print(
                f"Warning: Richness "
                f"{richness} not found"
            )
            continue
        data_list.append(split_data[richness][subset])
    narray = np.vstack(data_list)
    print(
        f"{subset} shape = "
        f"{narray.shape}"
    )
    return narray


# Input is the split train & test index lookup, and the folder to retrieve from; output is train & test data
def load_split_dataset(split_df, folder_path, richness_list, subset="tr", sample_size=None, random_state=None):
    use_df = split_df[(split_df["split"] == subset) & (split_df["richness"].isin(richness_list))]
    # If sample_size is specified, randomly sample
    if sample_size is not None:
        if sample_size > len(use_df):
            raise ValueError(
                f"sample_size={sample_size} exceeds "
                f"available samples={len(use_df)}")
        use_df = use_df.sample(n=sample_size, random_state=random_state)

    data_list = []
    for filename, group in use_df.groupby("file"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        narray = df.iloc[:, :-2].values
        rows = group["row"].values
        data_list.append(narray[rows])
    result = np.vstack(data_list)
    print(
        f"{subset} shape = "
        f"{result.shape}"
    )
    return result

def load_split_dataset2(split_df, folder_path, richness_list, subset="tr", 
                        sample_size=None, random_state=None, sur_frac=False, MaxNum=20):
    use_df = split_df[(split_df["split"] == subset) & (split_df["richness"].isin(richness_list))]
    # If sample_size is specified, randomly sample
    if sample_size is not None:
        if sample_size > len(use_df):
            raise ValueError(
                f"sample_size={sample_size} exceeds "
                f"available samples={len(use_df)}")
        use_df = use_df.sample(n=sample_size, random_state=random_state)

    data_list = []
    richness_label = []
    for filename, group in use_df.groupby("file"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        if sur_frac == True:
            len_Xynum = MaxNum**2 + MaxNum*3 + 6
        else:
            len_Xynum = MaxNum**2 + MaxNum*3 + 5

        narray = df.iloc[:, :len_Xynum].values
        rows = group["row"].values
        data_list.append(narray[rows])
        richness_label.append(group["richness"].values)
    result = np.vstack(data_list)
    richness_label = np.concatenate(richness_label)
    print(
        f"{subset} shape = "
        f"{result.shape}"
    )
    return result, richness_label

def load_split_dataset3(split_df, folder_path, richness_list, subset="tr", 
                        sample_size=None, random_state=None, sur_frac=False, MaxNum=20):
    use_df = split_df[(split_df["split"] == subset) & (split_df["richness"].isin(richness_list))]
    # If sample_size is specified, randomly sample
    if sample_size is not None:
        if sample_size > len(use_df):
            raise ValueError(
                f"sample_size={sample_size} exceeds "
                f"available samples={len(use_df)}")
        use_df = use_df.sample(n=sample_size, random_state=random_state)

    data_list = []
    richness_label = []
    for filename, group in use_df.groupby("file"):
        file_path = os.path.join(folder_path, filename + ".csv")
        df = pd.read_csv(file_path)
        if sur_frac == True:
            len_Xynum = MaxNum**2 + MaxNum*3 + 6
        else:
            len_Xynum = MaxNum**2 + MaxNum*3 + 5

        narray = df.iloc[:, :len_Xynum].values
        rows = group["row"].values
        data_list.append(narray[rows])
        richness_label.append(group["richness"].values)
    result = np.vstack(data_list)
    richness_label = np.concatenate(richness_label)
    print(
        f"{subset} shape = "
        f"{result.shape}"
    )
    return result, richness_label