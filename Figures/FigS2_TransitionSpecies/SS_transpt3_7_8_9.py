import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import joblib
import json
from pathlib import Path

#BASE_DIR = Path(__file__).resolve().parent
import sys
sys.path.append("/work/home/acychkqsp6/zrm/MasterThesis/tools")
#sys.path.append("../../tools")
import architec as acht
#import evaluation as eva
import prepare as prep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
import importlib
#importlib.reload(eva)


# %%
NumSpecies = 20
featgam = NumSpecies**2 - NumSpecies
task_num = 3+NumSpecies # 0 = steady-state count, 1 = steady-state entropy, 2 = diversity, 3 = abundance, 3+NumSpecies = switching species, 3+2*NumSpecies = oscillation

# The folder storing datasets splited latter.
folder_path = "/work/home/acychkqsp6/zrm/MasterThesis/Databases/Ver1.9/ver1.9.6_upto20"
#folder_path = "../../Databases/Ver1.9/ver1.9.6_upto20"

for rep in range(1,4):
    split_df = pd.read_csv(f"/work/home/acychkqsp6/zrm/MasterThesis/Principle_design_SynCom/Fig2/split_MD_v1/split_MD_{rep}.csv")
    #split_df = pd.read_csv(f"split_MD_v1/split_MD_{rep}.csv")

    #Trainset
    narray_tr = prep.load_split_dataset(
        split_df,
        folder_path,
        richness_list=[7,8,9],
        #richness_list=list(range(3,21)),
        subset="tr"
    )

    #Testset
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

    train_dataset = acht.CommunityDataset_MultiLabel(Xg_tr, XA_tr, Xd_tr, y_tr)
    test_dataset = acht.CommunityDataset_MultiLabel(Xg_te, XA_te, Xd_te, y_te)

    n_val = int(len(train_dataset)*0.2)
    n_train = len(train_dataset)-n_val
    train_ds, val_ds = random_split(train_dataset, [n_train,n_val])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = acht.SSTrans2(NumSpecies).to(device) # Select deep learning architecture.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stop = acht.EarlyStopping(patience=8, min_delta=1e-4)
    history = {"loss":[], "val_loss":[]}

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

    # Model saving
    save_dir = Path(f"/work/home/acychkqsp6/zrm/MasterThesis/saved_models/swi_species/sp7_8_9_SS_transpt3_r{rep}")
    #save_dir = Path(f"../../saved_models/swi_species_15fold/sp3to20_SS_transpt3_r{rep}")
    #save_dir = Path(f"../../saved_models/swi_species_15fold/sp10_SS_transpt3_r{rep}")

    save_dir.mkdir(parents=True, exist_ok=True)  # Auto-create directory if missing
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



