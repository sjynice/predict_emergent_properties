import torch
import numpy as np
import pandas as pd

def sensitivity_analysis(Xnum0, predictor, n_real, mode, emer_property, device,):
    '''
    mode can be "abs", "raise_y", "lower_y"
    '''
    
    MaxNum = predictor.ckpt["NumSpecies"]
    featgam = predictor.ckpt["featgam"]

    X = torch.tensor(Xnum0, dtype=torch.float32, device=device, requires_grad=True)
    if emer_property == "H" or emer_property == "D":
        y = predictor.predict_tensor_MD(X)
    elif emer_property == "N":
        y = predictor.predict_tensor_NS(X)
    else:
        raise ValueError("emer_property is out of reasonable range")
    
    y.backward()
    grad = X.grad.detach().cpu().numpy()

    ranges = np.concatenate([
        np.ones(MaxNum),          # mu: 0~1
        np.full(featgam, 15.0),       # gamma: -7.5~7.5
        np.full(MaxNum, 1e-7)     # D: 1e-13~1e-7
    ])

    if mode == "abs":
        importance = np.abs(grad) * ranges
    elif mode == "raise_y":
        importance = grad * ranges
    elif mode == "lower_y":
        importance = -grad * ranges
    else:
        raise ValueError("mode should be 'abs', 'raise_y' or 'lower_y'")

    mu_grad = grad[:MaxNum]
    mu_imp  = importance[:MaxNum]

    gamma_grad = grad[MaxNum:MaxNum + featgam]
    gamma_imp = importance[MaxNum:MaxNum + featgam]

    D_grad = grad[MaxNum + featgam:]
    D_imp = importance[MaxNum + featgam:]

    # Remove sensitivity considerations for padded parts, i.e., do not consider adding new species to optimize y
    mu_grad_real = mu_grad[:n_real]
    mu_imp_real  = mu_imp[:n_real]
    mu_real = Xnum0[:n_real]

    D_grad_real = D_grad[:n_real]
    D_imp_real  = D_imp[:n_real]
    D_real = Xnum0[MaxNum + featgam:MaxNum + featgam + n_real]

    gamma_pairs = []
    gamma_grad_real = []
    gamma_imp_real = []
    gamma_param_idx = []
    gamma_values_real = []

    idx = 0

    for i in range(MaxNum):
        for j in range(MaxNum):

            if i == j:
                continue

            if i < n_real and j < n_real:

                gamma_pairs.append((i+1, j+1))
                gamma_grad_real.append(gamma_grad[idx])
                gamma_imp_real.append(gamma_imp[idx])
                gamma_param_idx.append(MaxNum + idx)
                gamma_values_real.append(Xnum0[MaxNum + idx])

            idx += 1

    gamma_grad_real = np.array(gamma_grad_real)
    gamma_imp_real = np.array(gamma_imp_real)

    df_mu = pd.DataFrame({
        "type":"mu",
        "target":[
            f"mu_{i+1}"
            for i in range(n_real)
        ],
        "param_idx":np.arange(n_real),
        "value":mu_real,
        "grad":mu_grad_real,
        "importance":mu_imp_real
    })

    df_D = pd.DataFrame({
        "type":"D",
        "target":[
            f"D_{i+1}"
            for i in range(n_real)
        ],
        "param_idx":(MaxNum + featgam + np.arange(n_real)),
        "value":D_real,
        "grad":D_grad_real,
        "importance":D_imp_real
    })

    df_gamma = pd.DataFrame({
        "type":"gamma",
        "target":[
            f"gamma_{i}_{j}"
            for i,j in gamma_pairs
        ],
        "param_idx":gamma_param_idx,
        "value":gamma_values_real,
        "grad":gamma_grad_real,
        "importance":gamma_imp_real
    })

    df_all = pd.concat([df_mu, df_gamma, df_D], ignore_index=True)
    df_all["param_idx"] = df_all["param_idx"].astype(int)
    df_all = df_all.sort_values("importance", ascending=False).reset_index(drop=True)

    return df_all

# Set mask so that during optimization only the mu and gamma of important species are updated, and adding new species is avoided
def build_mask(top_species, n_real, MaxNum, device):

    featgam = MaxNum**2 - MaxNum
    mask = np.zeros(MaxNum + featgam + MaxNum, dtype=np.float32)

    # Optimize mu
    for s in top_species:
        mask[s] = 1

    # Optimize gamma
    idx = MaxNum
    for i in range(MaxNum):
        for j in range(MaxNum):
            if i == j:
                continue
            if (i < n_real and j < n_real): # Restrict from assigning values to non-existent species
                sender = j
                if sender in top_species:
                    mask[idx] = 1
            idx += 1

    return torch.tensor(mask, dtype=torch.float32, device=device)

# Update mask for a single parameter
def build_mask_single2(paranum, paraidx, df_sens, MaxNum, device):
    '''
    paranum is the number of parameters this step wants to optimize
    paraidx is the set of indices of parameters optimized in the previous step within Xnum, type is array
    df_sens is the importance of each parameter in Xnum, sorted from high to low importance
    '''
    featgam = MaxNum**2 - MaxNum
    mask = np.zeros(MaxNum + featgam + MaxNum, dtype=np.float32)
    indexs = df_sens["param_idx"].iloc[:paranum].to_numpy(dtype=int)
    # Find the first element in indexs that is not in paraidx
    new_idx = indexs[~np.isin(indexs, paraidx)][0]

    # Add it in
    paraidx = np.append(paraidx, new_idx)

    mask[paraidx] = 1
    return (torch.tensor(mask, dtype=torch.float32, device=device), paraidx)

# Name each parameter based on the upper bound of species count after padding
def build_param_names(MaxNum):

    names = []

    for i in range(MaxNum):
        names.append(f"mu_{i+1}")

    for i in range(MaxNum):
        for j in range(MaxNum):
            if i == j:
                continue
            names.append(f"gamma_{i+1}_{j+1}")

    for i in range(MaxNum):
        names.append(f"D_{i+1}")

    return names


#=========================================================
# Constrain Xnum values to remain within a reasonable range during optimization
def project_parameters(X, MaxNum):
    featgam = MaxNum**2 - MaxNum
    with torch.no_grad():
        X[:MaxNum].clamp_(0.1, 1.0)
        X[MaxNum:MaxNum + featgam].clamp_(-7, 7)
        X[MaxNum + featgam:].clamp_(1e-13, 1e-7)

def project_gamma_sign(X, X_ref, MaxNum):
    """
    Keep the sign structure of gamma interactions unchanged:

        gamma > 0 -> keep gamma > 0
        gamma < 0 -> keep gamma < 0
        gamma = 0 -> keep gamma = 0

    X     : current optimized parameters
    X_ref : gamma sign structure of reference state
    """
    featgam = MaxNum**2 - MaxNum

    gamma = X[MaxNum:MaxNum + featgam]
    gamma_ref = X_ref[MaxNum:MaxNum + featgam]

    eps = 1e-2

    with torch.no_grad():

        # Initially positive
        gamma = torch.where(
            gamma_ref > 0,
            torch.clamp(gamma, min=eps),
            gamma
        )

        # Initially negative
        gamma = torch.where(
            gamma_ref < 0,
            torch.clamp(gamma, max=-eps),
            gamma
        )

        # Initially 0
        gamma = torch.where(
            gamma_ref == 0,
            torch.zeros_like(gamma),
            gamma
        )

        X[MaxNum:MaxNum + featgam] = gamma

    return X

# X->Xs: map parameters of different magnitudes in X uniformly to an O(1) range
def X_to_scaled(X, MaxNum):
    featgam = MaxNum**2 - MaxNum
    Xs = X.clone()
    # mu
    Xs[:MaxNum] = X[:MaxNum]
    # gamma
    Xs[MaxNum:MaxNum+featgam] = (X[MaxNum:MaxNum+featgam] / 7.5)
    # D
    D = X[MaxNum+featgam:]
    D_scaled = torch.zeros_like(D)
    mask = D > 0
    D_scaled[mask] = (torch.log10(D[mask]) + 13)
    Xs[MaxNum+featgam:] = D_scaled
    return Xs

# Xs->X
def scaled_to_X(Xs, MaxNum):
    featgam = MaxNum**2 - MaxNum
    X = Xs.clone()
    # mu
    X[:MaxNum] = Xs[:MaxNum]
    # gamma
    X[MaxNum:MaxNum+featgam] = (Xs[MaxNum:MaxNum+featgam] * 7.5)
    # D
    D_scaled = Xs[MaxNum+featgam:]
    D = torch.zeros_like(D_scaled)
    mask = D_scaled > 0
    D[mask] = 10 ** (D_scaled[mask] - 13)
    X[MaxNum+featgam:] = D
    return X