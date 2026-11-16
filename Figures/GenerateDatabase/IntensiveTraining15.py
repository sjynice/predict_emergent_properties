import numpy as np
import pandas as pd
import argparse
from scipy.integrate import solve_ivp
from pathlib import Path

import sys
sys.path.append("/work/home/acychkqsp6/zrm/MasterThesis/tools")
#sys.path.append("../../tools")
import generation as gr
import functions as func

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
global NumSpecies,mu,gamma,D,delta

parser = argparse.ArgumentParser(description="Save different CSVs by task ID")
parser.add_argument("--input", type=str, required=True, help="Input CSV file")
parser.add_argument("--output-root", type=str, required=True)
parser.add_argument("--job-id", type=int)
parser.add_argument("--array-id", type=int)
parser.add_argument("--proc-id", type=int)
parser.add_argument("--repeat", type=int, default=200, help="Number of initialization times")
parser.add_argument("--timespan", type=float, default=800, help="Time span for ODE integration")
args = parser.parse_args()

# parameters settings:
df = pd.read_csv(args.input)
NumSpecies, _ = func.get_species_info(args.input)
Xnum = df.values
repeat = args.repeat # Number of initialization times
timespan = np.arange(0, args.timespan+0.1, 0.1) # Time span for ODE integration

Xynum = gr.predict_numerical(Xnum, NumSpecies, repeat, timespan)

def main():
    df = pd.DataFrame(Xynum)
    input_stem = Path(args.input).stem
    save_dir = (Path(args.output_root) / str(NumSpecies))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{input_stem}.csv"
    df.to_csv(save_path, index=False)
    print(f"saved {save_path}")

if __name__ == "__main__":
    main()
