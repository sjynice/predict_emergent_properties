# %%
import numpy as np
import pandas as pd
import argparse
import os

import sys
sys.path.append("/work/home/acychkqsp6/zrm/MasterThesis/tools")
#sys.path.append("../../tools")
import generation as gr

# %% [markdown]
# NumSpecies: Number of species;
# 
# mu: Growth rate;
# 
# gamma: Quantification of positive/negative inter-species interactions;
# 
# D: Mortality rate;
# 
# delta: Another parameter for positive inter-species interactions;

# %%
parser = argparse.ArgumentParser(description="Save different csv files by task ID")
parser.add_argument("task_id", type=int, help="Task number (e.g., 1-10)")
parser.add_argument("job_id", type=int, help="SLURM job ID")
parser.add_argument("--richness", type=int)
parser.add_argument("--datasize", type=int, default=400, help="Size of the dataset to be generated")
parser.add_argument("--repeat", type=int, default=200, help="Number of initialization times")
parser.add_argument("--timespan", type=float, default=800, help="Time span for ODE integration")
args = parser.parse_args()

# parameters settings:
NumSpecies = args.richness
#MaxNum = 100
repeat = args.repeat # Number of initialization times
datasize = args.datasize # Size of the dataset to be generated
timespan = np.arange(0, args.timespan+0.1, 0.1)

featgam = NumSpecies**2 - NumSpecies # Number of gamma features
total_dim = NumSpecies + featgam + NumSpecies
density = np.random.rand(datasize)
pos_inall = np.random.rand(datasize)
gamma_neg_cen = 0.1 + 6.9*np.random.rand(datasize)
gamma_pos_cen = 0.1 + 6.9*np.random.rand(datasize)
gamma_mean = np.zeros(datasize)
n_edge = NumSpecies * (NumSpecies - 1) # Upper boundary of the number of interaction.
Xnum = np.zeros((datasize, total_dim)) # Feature order: mu0, mu1, [gamma(0), ..., gamma(featgam-1)], D

den_arr = np.zeros(datasize)
pos_arr = np.zeros(datasize)

# parameters settings:
Xnum[:, 0:NumSpecies] = 0.1+0.9*np.random.rand(datasize, NumSpecies) # mu ranges from 0.0 to 1.0

for i in range(datasize):

    n_nonzero = int(round(density[i] * n_edge)) # None zero interaction number.
    n_pos = int(round(pos_inall[i] * n_nonzero)) # Positive interaction number.
    n_neg = n_nonzero - n_pos

    idx_nonzero = np.random.choice(n_edge, n_nonzero, replace=False) # None zero interaction index(from 0).
    idx_pos = np.random.choice(idx_nonzero, n_pos, replace=False)

    mask_pos = np.zeros(n_edge, dtype=bool)
    mask_pos[idx_pos] = True

    mask_neg = np.zeros(n_edge, dtype=bool)
    mask_neg[idx_nonzero] = True
    mask_neg[idx_pos] = False

    m_neg = 3
    m_pos = 3
    gamma_sigma = 0.2

    # Randomly generate neg-interaction coefficient according to negative number.
    neg_value = -m_neg * ((1 - gamma_sigma) + 2 * gamma_sigma * np.random.rand(n_neg))
    pos_value = m_pos * ((1 - gamma_sigma) + 2 * gamma_sigma * np.random.rand(n_pos))

    # Scaled individually to the target gamma mean.
    if n_neg > 0:
        neg_value *= gamma_neg_cen[i] / np.mean(np.abs(neg_value))
    if n_pos > 0:
        pos_value *= gamma_pos_cen[i] / np.mean(pos_value)
    if n_nonzero > 0:
        gamma_mean[i] = (np.sum(np.abs(neg_value))+np.sum(pos_value)) / n_nonzero # Only calculate non zero interaction.

    gammas = np.zeros(n_edge)
    gammas[mask_neg] = neg_value
    gammas[mask_pos] = pos_value

    Xnum[i, NumSpecies:NumSpecies + featgam] = gammas

    if n_nonzero > 0:
        den_arr[i] = n_nonzero / n_edge
        pos_arr[i] = n_pos / n_nonzero
    else:
        den_arr[i] = 0
        pos_arr[i] = 0

#Xnum[:, NumSpecies:(NumSpecies+featgam)] = 0.5 + 1.5*np.random.rand(datasize, featgam) # |gamma| ranges from 0.5 to 2.0 (competition indext)
Xnum[:, (NumSpecies+featgam):(NumSpecies+featgam+NumSpecies)] = 1e-06/NumSpecies*np.random.rand(datasize, NumSpecies) # D ranges from 0.1 to 0.3
delta = 0.2*np.ones(NumSpecies)

# Conduct experiments under various parameters, with each experiment initialized repeat times.

Xynum = gr.predict_numerical(Xnum, NumSpecies, repeat, timespan)

# Statistic interaction network properties
richness_col = np.full(datasize, NumSpecies)
gamma_neg_arr = gamma_neg_cen
gamma_pos_arr = gamma_pos_cen
gamma_mean_arr = gamma_mean

n_feature_cols = NumSpecies**2+NumSpecies
extra = np.column_stack([Xynum, richness_col, den_arr, pos_arr, gamma_neg_arr, gamma_pos_arr, gamma_mean_arr])


def main():
    df = pd.DataFrame(extra)
    df.columns = list(df.columns[:n_feature_cols]) + ["states_number", "relative_H", "relative_D"] + [f"abun_sp{i+1}" 
    for i in range(NumSpecies)] + [f"switching_sp{i+1}" for i in range(NumSpecies)] + ["fluctuation", "outlier", 
    "survival_fraction"] + ["richness", "density", "pos_inall", "gamma_neg_mean", "gamma_pos_mean", "gamma_mean"]

    os.makedirs(f"outputcsv/sp{args.richness}", exist_ok=True)
    output_file = f"outputcsv/sp{args.richness}/output_{args.task_id}_{args.job_id}_NumMeta.csv"
    df.to_csv(output_file, index=False)
    print(f"Task {args.task_id}: saved {output_file}")

if __name__ == "__main__":
    main()

