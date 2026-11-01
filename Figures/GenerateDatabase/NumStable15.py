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
repeat = args.repeat # Number of initialization times
datasize = args.datasize # Size of the dataset to be generated
timespan = np.arange(0, args.timespan+0.1, 0.1)

featgam = NumSpecies**2 - NumSpecies # Number of gamma features
total_dim = NumSpecies + featgam + NumSpecies

Xnum = np.zeros((datasize, total_dim)) # Feature order: mu0, mu1, [gamma(0), ..., gamma(featgam-1)], D

# parameters settings:
Xnum[:, 0:NumSpecies] = 0.1+0.9*np.random.rand(datasize, NumSpecies) # mu ranges from 0.0 to 1.0
for i in range(datasize):
    # Generate gamma matrix with specified fractions of negative, positive, and zero interactions
    neg_fraq=0.5+0.5*np.random.rand()
    pos_fraq=np.random.rand()*(1-neg_fraq)
    m_neg=5*np.random.rand()
    m_pos=5*np.random.rand()
    gammas = []
    for m in range(NumSpecies):
        for n in range(NumSpecies):
            if m != n:
                r = np.random.rand()
                if r < neg_fraq:
                    gammas.append(-m_neg*(0.5+np.random.rand()))  # Negative interaction
                elif r > 1- pos_fraq:
                    gammas.append(m_pos*(0.5+np.random.rand()))   # Positive interaction
                else:
                    gammas.append(0.0)                            # No interaction
    Xnum[i, NumSpecies:(NumSpecies+featgam)] = gammas

#Xnum[:, NumSpecies:(NumSpecies+featgam)] = 0.5 + 1.5*np.random.rand(datasize, featgam) # |gamma| ranges from 0.5 to 2.0 (competition indext)
Xnum[:, (NumSpecies+featgam):(NumSpecies+featgam+NumSpecies)] = 1e-06/NumSpecies*np.random.rand(datasize, NumSpecies)
delta = 0.2*np.ones(NumSpecies)

# Conduct experiments under various parameters, with each experiment initialized repeat times.
Xynum = gr.predict_numerical(Xnum, NumSpecies, repeat, timespan)


def main():
    df = pd.DataFrame(Xynum)
    df.columns = list(df.columns[:total_dim]) + ["states_number", "relative_H", "relative_D"] + [f"abun_sp{i+1}" 
    for i in range(NumSpecies)] + [f"switching_sp{i+1}" for i in range(NumSpecies)] + ["fluctuation", "outlier", "survival_fraction"]

    os.makedirs(f"outputcsv/sp{args.richness}", exist_ok=True)
    output_file = f"outputcsv/sp{args.richness}/output_{args.task_id}_{args.job_id}_NumMeta.csv"
    df.to_csv(output_file, index=False)
    print(f"Task {args.task_id}: saved {output_file}")

if __name__ == "__main__":
    main()
