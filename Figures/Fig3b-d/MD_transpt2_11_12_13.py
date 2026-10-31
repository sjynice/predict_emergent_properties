# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import joblib
import json
from pathlib import Path
from datetime import datetime
import argparse

#BASE_DIR = Path(__file__).resolve().parent
import sys
sys.path.append("/work/home/acychkqsp6/zrm/MasterThesis/tools")
#sys.path.append("../../tools")
import architec as acht
import evaluation as eva
import prepare as prep


# %%
import importlib
#importlib.reload(eva)
#importlib.reload(prep)

parser = argparse.ArgumentParser(description="Repeat experiment")
parser.add_argument("--repeat-id", type=int)
args = parser.parse_args()

# %%
# Use this section when train and test sets come from different files
NumSpecies = 20
featgam = NumSpecies**2 - NumSpecies
task_num = 1 # 0: num stable states, 1: steady-state entropy, 2: diversity, 3: abundance, 3+NumSpecies: switching species, 3+2*NumSpecies: oscillation

split_df = pd.read_csv(f"/work/home/acychkqsp6/zrm/MasterThesis/Principle_design_SynCom/Fig3/split_MD_{args.repeat_id}.csv")
folder_path = "/work/home/acychkqsp6/zrm/MasterThesis/Databases/Ver1.9/ver1.9.6_upto20"
#split_df = pd.read_csv("split_MD_1.csv")
#folder_path = "../../Databases/Ver1.9/ver1.9.6_upto20"

#--------------Set up training set-------------------------
# folder_tr = "../../Databases/Ver1.9/ver1.9.6_upto20_tr"
# narray_tr, used_files = load_training_dataset_from_folder(folder_tr)

narray_tr = prep.load_split_dataset(
    split_df,
    folder_path,
    richness_list=[11,12,13],
    #richness_list=list(range(3,21)),
    subset="tr"
)

#--------------Set up test set-------------------------
# overlap_te = 'Ver1.9/ver1.9.6_upto20_te/20-20_102400'
# df_te = pd.read_csv(f'../../Databases/{overlap_te}.csv')
# narray_te = df_te.iloc[:, :-2].values

narray_te = prep.load_split_dataset(
    split_df,
    folder_path,
    richness_list=[10],
    #richness_list=list(range(3,21)),
    subset="te"
)

#-------------------------------------------------------
Xg_tr, XA_tr, Xd_tr, y_tr, scaler_g, scaler_d, idx_tr = prep.prepare_train(
    narray_tr,
    NumSpecies,
    featgam,
    task_num,
    sample_size=210000,
    random_state=42
)

Xg_te, XA_te, Xd_te, y_te, idx_te = prep.prepare_test(
    narray_te,
    NumSpecies,
    featgam,
    scaler_g,
    scaler_d,
    task_num,
    #sample_size=20000,
    random_state=42
)


# %%
class CommunityDataset(Dataset):
    """
    e.g. example=CommunityDataset(Xg1, XA1, Xd1, y1) is equivalent to running __init__(example, Xg1, XA1, Xd1, y1)
    The result is that example will have the following properties:
    example.Xg retrieves example's 'Xg', i.e., torch.FloatTensor(Xg1)
    len(example) is equivalent to __len__(example), returns len(example.y)
    example[17] is equivalent to __getitem__(example, 17), returns four variables: example.Xg[17],... i.e., (torch.FloatTensor(Xg1))[17]
    """
    def __init__(self, Xg, XA, Xd, y):
        self.Xg = torch.FloatTensor(Xg)
        self.XA = torch.FloatTensor(XA)
        self.Xd = torch.FloatTensor(Xd)
        self.y = torch.FloatTensor(y).reshape(-1,1)

    def __len__(self):
        return len(self.y) # return the number of samples

    def __getitem__(self, idx):
        return (
            self.Xg[idx],
            self.XA[idx],
            self.Xd[idx],
            self.y[idx]
        )

train_dataset = CommunityDataset(Xg_tr, XA_tr, Xd_tr, y_tr)
test_dataset = CommunityDataset(Xg_te, XA_te, Xd_te, y_te)

n_val = int(len(train_dataset)*0.2)
n_train = len(train_dataset)-n_val
train_ds, val_ds = random_split(train_dataset, [n_train,n_val])

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


# %%
class TransformerEncoderBlock(nn.Module):

    def __init__(self, embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class MDTrans2(nn.Module):

    def __init__(self, NumSpecies):
        super().__init__()
        self.embed = nn.Linear(NumSpecies + 2, 64)
        self.encoder1 = TransformerEncoderBlock()
        self.encoder2 = TransformerEncoderBlock()
        self.fc = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )
        self.out = nn.Linear(64,1)

    def forward(self, Xg, XA, Xd):

        g = Xg.unsqueeze(-1)
        d = Xd.unsqueeze(-1)
        tokens = torch.cat([g,d,XA], dim=-1)
        x = self.embed(tokens)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return self.out(x)


# %%
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1

        return self.counter >= self.patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MDTrans2(NumSpecies).to(device)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

early_stop = EarlyStopping(
    patience=8,
    min_delta=1e-5
)

history = {
    "loss":[],
    "val_loss":[]
}


# %%
best_state = None

for epoch in range(140):
    model.train()
    train_loss = 0
    
    for Xg,XA,Xd,y in train_loader:

        Xg = Xg.to(device)
        XA = XA.to(device)
        Xd = Xd.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(Xg,XA,Xd)

        loss = criterion(logits,y)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    model.eval()

    val_loss = 0

    with torch.no_grad():

        for Xg,XA,Xd,y in val_loader:

            Xg = Xg.to(device)
            XA = XA.to(device)
            Xd = Xd.to(device)
            y = y.to(device)

            logits = model(Xg,XA,Xd)
            loss = criterion(logits,y)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    history["loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch {epoch+1} "
        f"train={train_loss:.5f} "
        f"val={val_loss:.5f}"
    )

    if val_loss < early_stop.best_loss:
        best_state = model.state_dict()

    if early_stop.step(val_loss):
        print("Early stopping")
        break

model.load_state_dict(best_state)


# %%
# Save model
save_dir = Path(f"/work/home/acychkqsp6/zrm/MasterThesis/saved_models/sta_entropy/sp11_12_13_MD_transpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
#save_dir = Path(f"/work/home/acychkqsp6/zrm/MasterThesis/saved_models/diversity/sp3to20_MD_transpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
#save_dir = Path(f"../../saved_models/sta_entropy/sp3to20_MD_transpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
save_dir.mkdir(parents=True, exist_ok=True)  # auto-create directory

torch.save(model.state_dict(), save_dir / "model.pt")

joblib.dump(scaler_g, save_dir / "scaler_g.pkl")
joblib.dump(scaler_d, save_dir / "scaler_d.pkl")

config = {
    "NumSpecies": NumSpecies,
    "featgam": featgam,
    "task_num": task_num,
    "embed_dim": 64,
    "num_heads": 4,
    "ff_dim": 128
}

with open(save_dir / "config.json", "w") as f:
    json.dump(config, f)


# %%
model.eval()
preds = []

with torch.no_grad():

    for Xg, XA, Xd, _ in test_loader:

        Xg = Xg.to(device)
        XA = XA.to(device)
        Xd = Xd.to(device)

        pred = model(Xg, XA, Xd)
        preds.append(pred.cpu().numpy())

y_pred = np.concatenate(preds).ravel()

metrics = eva.eval_regression(y_te, y_pred)
df_metrics = pd.DataFrame([metrics])
print(df_metrics.to_string())