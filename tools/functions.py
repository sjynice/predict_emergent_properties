from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
import torch

import architec as acht
import prepare as prep

import importlib
importlib.reload(prep)

def get_species_info(file_path):
    """
    Parse from filename:
    n_real      : actual richness
    MaxNum  : number of species after padding

    e.g.:
    3-20_96000.csv -> (3, 20)
    """
    stem = Path(file_path).stem   # 3-20_96000
    richness_part = stem.split("_")[0]   # 3-20
    n_real, MaxNum = map(int, richness_part.split("-"))

    return n_real, MaxNum

def pick_minority(NumSpecies, task_num, df):
    """Filter rows where emergent features are greater than 0.5, and save to a new CSV"""
    featgam = NumSpecies**2 - NumSpecies
    df2 = df[df.iloc[:, NumSpecies + featgam + NumSpecies + task_num] > 0.5].reset_index(drop=True)
    df3 = df2.iloc[:, :NumSpecies + featgam + NumSpecies]
    return df3

def split_csv(n_real, NumSpecies, df, output_dir, n_chunks):
    chunks = np.array_split(df, n_chunks)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        chunk.to_csv(outdir / f"{n_real}-{NumSpecies}_{i+1}.csv", index=False)

def intensive_sampling_Xnum(
    Xnum_reduced,
    n_real,
    num_samples=10,
    sigma_mu=0.07,
    sigma_gamma=0.7,
    sigma_D=7e-12,
    random_state=None
):
    """
    Local intensive sampling of each sample in Xnum_reduced
    Parameters
    ----------
    Xnum_reduced : ndarray, shape=(N, F)
    n_real : int, actual number of species
    num_samples : int, augmentation multiplier per sample (default 10)
    Returns
    -------
    Xnum_intensive : ndarray, shape=(N*num_samples, F)
    """

    rng = np.random.default_rng(random_state)
    N = Xnum_reduced.shape[0]
    featgam_real = n_real * (n_real - 1)

    mu = Xnum_reduced[:, :n_real]
    gamma = Xnum_reduced[:, n_real:n_real + featgam_real]
    D = Xnum_reduced[:, n_real + featgam_real:]

    # Expand center points
    mu_center = np.repeat(mu[:, None, :], num_samples, axis=1)
    gamma_center = np.repeat(gamma[:, None, :], num_samples, axis=1)
    D_center = np.repeat(D[:, None, :], num_samples, axis=1)


    # Normal perturbation
    mu_new = rng.normal(loc=mu_center, scale=sigma_mu)
    gamma_new = rng.normal(loc=gamma_center, scale=sigma_gamma)
    D_new = rng.normal(loc=D_center, scale=sigma_D)

    # Reasonable range constraint
    mu_new = np.clip(mu_new, 0.0, 1.0)
    D_new = np.maximum(D_new, 0.0)

    # Concatenate
    Xnum_intensive = np.concatenate([mu_new, gamma_new, D_new], axis=2)

    # (N,10,F) -> (N*10,F)
    Xnum_intensive = Xnum_intensive.reshape(N * num_samples, -1)

    return Xnum_intensive

'''
Data preprocessing
'''
def clean_csv_missing(df, drop=True, verbose=True):
    '''
    Find and clean missing values in a DataFrame.

    df comes from original output of NumStable13.py or IntensiveTraining13.py or NumStable14.py

    Parameters:
    drop (bool): Whether to drop rows with missing values, default is True
    verbose (bool): Whether to print the locations of missing values, default is True

    '''
    # Find missing values
    missing_mask = df.isna()
    rows, cols = missing_mask.to_numpy().nonzero()
    # Print the locations of missing values if verbose is True
    if verbose:
        for r, c in zip(rows, cols):
            print(f"Row {r}, Column '{df.columns[c]}' has a missing value")
    # Delete rows with missing values if drop is True
    if drop:
        df = df.dropna(axis=0).reset_index(drop=True)
    return df

def DealWithOutlier_ver194(NumSpecies, a, df):
    '''
    df comes from output of clean_csv_missing() function.
    '''
    df2 = df[df.iloc[:,NumSpecies**2+NumSpecies].isin(list(range(1, NumSpecies + 1)))] # Pick out samples whose number of stable states in range (1, NumSpecies) only.
    #df2 = df[df.iloc[:,NumSpecies**2+NumSpecies].isin(list(range(0, NumSpecies + 1)))] # Pick out samples whose number of stable states in range (0, NumSpecies) only.
    df2 = df2[df2.iloc[:, NumSpecies**2+3*NumSpecies+4] < a] #Pick out samples whose unresolved trajectory fraction is less than a.
    df2 = df2.reset_index(drop=True)
    print(df2.shape)
    return df2

def pad_Xnum_and_ynum(df, NumSpecies, MaxNum, sur_frac=False):
    '''
    df comes from output of DealWithOutlier_ver194() function.
    Pad both the Xnum and ynum from NumSpecies to MaxNum.
    No matter how much legend after Xynum is all ok for this function.
    '''
    if sur_frac==True:
        len_ynum = NumSpecies*2+6
        len_ynum2 = MaxNum*2+6
    else:
        len_ynum = NumSpecies*2+5
        len_ynum2 = MaxNum*2+5

    len_Xnum = NumSpecies**2+NumSpecies
    len_Xnum2 = MaxNum**2+MaxNum
    narray = df.iloc[:, :len_Xnum+len_ynum].values
    Xnum = narray[:, 0:len_Xnum]
    ynum = narray[:, len_Xnum:]
    datasize = len(narray[:, 0])

    # Add some 0 to Xnum, save it as Xnum2
    Xnum2 = prep.pad_Xnum_to_maxspecies(Xnum, NumSpecies, MaxNum)
    ynum2 = np.zeros((datasize, len_ynum2))
    ynum2[:, 0:3+NumSpecies] = ynum[:, :3+NumSpecies]
    ynum2[:, 3+MaxNum:3+MaxNum+NumSpecies] = ynum[:, 3+NumSpecies:3+NumSpecies+NumSpecies]
    ynum2[:, 3+2*MaxNum:] = ynum[:, 3+2*NumSpecies:]

    Xynum2 = np.hstack((Xnum2, ynum2))
    df_new = pd.concat([pd.DataFrame(Xynum2), df.iloc[:, len_Xnum+len_ynum:]], axis=1)
    df_new.columns = (
        list(range(len_Xnum2))
        + ["states_number", "relative_H", "relative_D"]
        + [f"abun_sp{i+1}" for i in range(MaxNum)] 
        + [f"switching_sp{i+1}" for i in range(MaxNum)] 
        + ["fluctuation", "outlier", "survival_fraction"] 
        + list(df_new.columns[Xynum2.shape[1]:])
    )
    return df_new

def pad_Xnum_and_ynum_nolegendversion(Xynum, NumSpecies, MaxNum):
    '''
    Last two columns of df is not legend of document, but have one column to discribe survival fraction.
    Pad both the Xnum and ynum from NumSpecies to MaxNum.
    '''
    narray = Xynum
    featgam = NumSpecies**2 - NumSpecies # Number of gamma features
    Xnum = narray[:, 0:NumSpecies + featgam + NumSpecies]
    ynum = narray[:, NumSpecies + featgam + NumSpecies:]
    datasize = len(narray[:, 0])

    # Get the index of the column corresponding valid gamma:
    ValidIndex = np.zeros(NumSpecies**2-NumSpecies) # !Cause the type of ValidIndex should be int, so we need to transform it to int later.
    temp = 0
    for i in range(MaxNum):
        for j in range(MaxNum):
            if i<NumSpecies and j<NumSpecies and i!=j:
                ValidIndex[temp] = MaxNum + i*(MaxNum-1) + j
                if j>i:
                    ValidIndex[temp] -= 1
                temp += 1

    # Add some 0 to Xnum, save it as Xnum2
    featgam2 = MaxNum**2 - MaxNum
    Xnum2 = np.zeros((datasize, MaxNum + featgam2 + MaxNum))
    ynum2 = np.zeros((datasize, 3+MaxNum*2+3))

    # Give valid mu value:
    Xnum2[:, 0:NumSpecies] = Xnum[:, 0:NumSpecies]

    # Give valid gamma value:
    for i in range(len(ValidIndex)):
        Xnum2[:, int(ValidIndex[i])] = Xnum[:, NumSpecies+i]

    # Give D value:
    Xnum2[:, (MaxNum+featgam2):(MaxNum+featgam2+NumSpecies)] = Xnum[:, (NumSpecies+featgam):(NumSpecies+featgam+NumSpecies)]

    ynum2[:, 0:3+NumSpecies] = ynum[:, 0:3+NumSpecies]
    ynum2[:, 3+MaxNum:3+MaxNum+NumSpecies] = ynum[:, 3+NumSpecies:3+NumSpecies+NumSpecies]
    ynum2[:, -3:] = ynum[:, -3:]
    
    Xynum2 = np.hstack((Xnum2, ynum2))

    return Xynum2


'''
Model loading
'''
# Architecture MD is used for Entropy, Diversity, Fluctuation
def load_saved_model_MD(load_dir, device):
    # scaler
    scaler_g = joblib.load(load_dir / "scaler_g.pkl")
    scaler_d = joblib.load(load_dir / "scaler_d.pkl")

    # config
    with open(load_dir / "config.json") as f:
        config = json.load(f)

    model = acht.MDTrans2(config["NumSpecies"]).to(device)

    model.load_state_dict(
        torch.load(
            load_dir / "model.pt",
            map_location=device
        )
    )
    return {
        "scaler_g": scaler_g,
        "scaler_d": scaler_d,
        "model": model,
        **config  # Also adds NumSpecies, featgam, etc. to the dictionary
    }

def load_saved_model_NS(load_dir, device):
    # scaler
    scaler_g = joblib.load(load_dir / "scaler_g.pkl")
    scaler_d = joblib.load(load_dir / "scaler_d.pkl")

    # config
    with open(load_dir / "config.json") as f:
        config = json.load(f)

    model = acht.NSTrans2(config["NumSpecies"], config["n_class"]).to(device)

    model.load_state_dict(
        torch.load(
            load_dir / "model.pt",
            map_location=device
        )
    )
    return {
        "scaler_g": scaler_g,
        "scaler_d": scaler_d,
        "model": model,
        **config  # Also adds NumSpecies, featgam, etc. to the dictionary
    }

def load_saved_model_SS(load_dir, device):
    # scaler
    scaler_g = joblib.load(load_dir / "scaler_g.pkl")
    scaler_d = joblib.load(load_dir / "scaler_d.pkl")

    # config
    with open(load_dir / "config.json") as f:
        config = json.load(f)

    model = acht.SSTrans2(config["NumSpecies"]).to(device)

    model.load_state_dict(
        torch.load(
            load_dir / "model.pt",
            map_location=device
        )
    )
    return {
        "scaler_g": scaler_g,
        "scaler_d": scaler_d,
        "model": model,
        **config  # Also adds NumSpecies, featgam, etc. to the dictionary
    }

'''
Predicting(Test or just predict)
'''
class TrteEvaluator:
    def __init__(self, ckpt):
        self.ckpt = ckpt

    def predict_array_MD(self, narray_te, device, sample_size=None):
        Xg_te, XA_te, Xd_te, y_te, idx_te = prep.prepare_test(
            narray_te,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"],
            self.ckpt["task_num"],
            sample_size,
            random_state=42
        )

        Xg_te = torch.as_tensor(Xg_te, dtype=torch.float32, device=device)
        XA_te = torch.as_tensor(XA_te, dtype=torch.float32, device=device)
        Xd_te = torch.as_tensor(Xd_te, dtype=torch.float32, device=device)
        
        self.ckpt["model"].eval()
        with torch.no_grad():
            pred = self.ckpt["model"](Xg_te, XA_te, Xd_te)
            y_pred = pred.cpu().numpy().ravel()
        return y_te, y_pred
    
    def predict_array_NS(self, narray_te, device, sample_size=None):
        Xg_te, XA_te, Xd_te, y_te, idx_te = prep.prepare_test( # Index of output y_te is from 0
            narray_te,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"],
            self.ckpt["task_num"],
            sample_size,
            random_state=42
        )

        Xg_te = torch.as_tensor(Xg_te, dtype=torch.float32, device=device)
        XA_te = torch.as_tensor(XA_te, dtype=torch.float32, device=device)
        Xd_te = torch.as_tensor(Xd_te, dtype=torch.float32, device=device)
        
        self.ckpt["model"].eval()
        with torch.no_grad():
            logits = self.ckpt["model"](Xg_te, XA_te, Xd_te)
            y_prob = torch.softmax(logits, dim=1).cpu().numpy()
        return y_te, y_prob
     
    def predict_array_Fl(self, narray_te, device, sample_size=None):
        Xg_te, XA_te, Xd_te, y_te, idx_te = prep.prepare_test(
            narray_te,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"],
            self.ckpt["task_num"],
            sample_size,
            random_state=42
        )

        Xg_te = torch.as_tensor(Xg_te, dtype=torch.float32, device=device)
        XA_te = torch.as_tensor(XA_te, dtype=torch.float32, device=device)
        Xd_te = torch.as_tensor(Xd_te, dtype=torch.float32, device=device)
        
        self.ckpt["model"].eval()
        with torch.no_grad():
            logits = self.ckpt["model"](Xg_te, XA_te, Xd_te)
            prob = torch.sigmoid(logits)
            y_prob = prob.cpu().numpy().ravel()
        return y_te, y_prob
        
    def predict_array_SS(self, narray_te, device, sample_size=None):
        Xg_te, XA_te, Xd_te, y_te, idx_te = prep.prepare_test(
            narray_te,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"],
            self.ckpt["task_num"],
            sample_size,
            random_state=42
        )
        Xg_te = torch.as_tensor(Xg_te, dtype=torch.float32, device=device)
        XA_te = torch.as_tensor(XA_te, dtype=torch.float32, device=device)
        Xd_te = torch.as_tensor(Xd_te, dtype=torch.float32, device=device)
        
        self.ckpt["model"].eval()
        with torch.no_grad():
            logits = self.ckpt["model"](Xg_te, XA_te, Xd_te)
            prob = torch.sigmoid(logits)
            y_prob = prob.cpu().numpy()

        return y_te, y_prob

class Predictor:
    def __init__(self, ckpt):

        self.ckpt = ckpt # eg. predictor = Predictor(model)

    def predict_tensor_MD(self, X):

        Xg, XA, Xd = prep.process_tensor_X(
            X, # X is after padding
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"]
        )
        self.ckpt["model"].eval()
        return self.ckpt["model"](Xg, XA, Xd) # Returns y_pred
    
    def predict_tensor_NS(self, X):

        if X.dim() == 1:
            X = X.unsqueeze(0)

        Xg, XA, Xd = prep.process_tensor_X(
            X,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"]
        )

        self.ckpt["model"].eval()
        logits = self.ckpt["model"](Xg, XA, Xd)

        y_prob = torch.softmax(logits, dim=1)
        N_values = torch.arange(1, y_prob.shape[1] + 1, dtype=y_prob.dtype, device=y_prob.device)

        # E[N] = sum_N N * P(N)
        expected_N = (y_prob * N_values).sum(dim=1)
        #print("expected_N:", expected_N.shape)

        return expected_N
    
    def predict_array_MD(self, Xnum_pad, device):

        Xg, XA, Xd = prep.process_narray_X(
            Xnum_pad,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"]
        )
        Xg = torch.as_tensor(Xg, dtype=torch.float32, device=device)
        XA = torch.as_tensor(XA, dtype=torch.float32, device=device)
        Xd = torch.as_tensor(Xd, dtype=torch.float32, device=device)
        
        self.ckpt["model"].eval()
        with torch.no_grad():
            pred = self.ckpt["model"](Xg, XA, Xd)
            y_pred = pred.cpu().numpy().ravel()
        return y_pred

    def predict_array_NS(self, Xnum_pad, device):

        Xg, XA, Xd = prep.process_narray_X(
            Xnum_pad,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"]
        )
        Xg = torch.as_tensor(Xg, dtype=torch.float32, device=device)
        XA = torch.as_tensor(XA, dtype=torch.float32, device=device)
        Xd = torch.as_tensor(Xd, dtype=torch.float32, device=device)
        
        self.ckpt["model"].eval()
        with torch.no_grad():
            logits = self.ckpt["model"](Xg, XA, Xd)
            y_prob = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(y_prob, axis=1) + 1 # Transform to 1-based index.
        return y_pred

    def predict_array_Fl(self, Xnum_pad, device, threshold=0.45):

        Xg, XA, Xd = prep.process_narray_X(
            Xnum_pad,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"]
        )
        Xg = torch.as_tensor(Xg, dtype=torch.float32, device=device)
        XA = torch.as_tensor(XA, dtype=torch.float32, device=device)
        Xd = torch.as_tensor(Xd, dtype=torch.float32, device=device)
        
        self.ckpt["model"].eval()
        with torch.no_grad():
            logits = self.ckpt["model"](Xg, XA, Xd)
            prob = torch.sigmoid(logits)
            y_prob = prob.cpu().numpy().ravel()
            y_pred = (y_prob >= threshold).astype(int)
        return y_pred
        
    def predict_array_SS(self, Xnum_pad, device, threshold=0.3):

        Xg, XA, Xd = prep.process_narray_X(
            Xnum_pad,
            self.ckpt["NumSpecies"],
            self.ckpt["featgam"],
            self.ckpt["scaler_g"],
            self.ckpt["scaler_d"]
        )
        Xg = torch.as_tensor(Xg, dtype=torch.float32, device=device)
        XA = torch.as_tensor(XA, dtype=torch.float32, device=device)
        Xd = torch.as_tensor(Xd, dtype=torch.float32, device=device)
        
        self.ckpt["model"].eval()
        with torch.no_grad():
            logits = self.ckpt["model"](Xg, XA, Xd)
            prob = torch.sigmoid(logits)
            y_prob = prob.cpu().numpy()
            y_pred = (y_prob >= threshold).astype(int)

        return y_pred