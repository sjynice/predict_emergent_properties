import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Pytorch version
# Tensorize data
class CommunityDataset(Dataset):
    """
    e.g. example=CommunityDataset(Xg1, XA1, Xd1, y1) is equivalent to running __init__(example, Xg1, XA1, Xd1, y1)
    After running, example will have the following properties:
    example.Xg will retrieve example's "Xg", i.e. torch.FloatTensor(Xg1)
    len(example) is equivalent to __len__(example), and will return len(example.y)
    example[17] is equivalent to __getitem__(example, 17), returning four variables: example.Xg[17],... i.e. (torch.FloatTensor(Xg1))[17]
    """
    def __init__(self, Xg, XA, Xd, y):
        self.Xg = torch.FloatTensor(Xg)
        self.XA = torch.FloatTensor(XA)
        self.Xd = torch.FloatTensor(Xd)
        self.y = torch.FloatTensor(y).reshape(-1,1)

    def __len__(self):
        return len(self.y) # Return the number of samples

    def __getitem__(self, idx):
        return (
            self.Xg[idx],
            self.XA[idx],
            self.Xd[idx],
            self.y[idx]
        )

class CommunityDataset_MultiLabel(Dataset):

    def __init__(self, Xg, XA, Xd, y):
        self.Xg = torch.FloatTensor(Xg)
        self.XA = torch.FloatTensor(XA)
        self.Xd = torch.FloatTensor(Xd)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.Xg[idx],
            self.XA[idx],
            self.Xd[idx],
            self.y[idx]
        )

class CommunityDataset_NS(Dataset):

    def __init__(self, Xg, XA, Xd, y):
        self.Xg = torch.FloatTensor(Xg)
        self.XA = torch.FloatTensor(XA)
        self.Xd = torch.FloatTensor(Xd)
        self.y = torch.LongTensor(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.Xg[idx], self.XA[idx], self.Xd[idx], self.y[idx]

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


# Architecture MD is used for Entropy, Diversity, Fluctuation
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

class NSTrans2(nn.Module):

    def __init__(self, NumSpecies, n_class):
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
        self.cls_head = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU()
        )
        self.out = nn.Linear(32,n_class)

    def forward(self, Xg, XA, Xd):
        g = Xg.unsqueeze(-1)
        d = Xd.unsqueeze(-1)
        tokens = torch.cat([g,d,XA], dim=-1)
        x = self.embed(tokens)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.cls_head(x)
        return self.out(x)

class SSTrans2(nn.Module):

    def __init__(self, NumSpecies):
        super().__init__()
        self.embed = nn.Linear(NumSpecies+2,64)
        self.encoder1 = TransformerEncoderBlock()
        self.encoder2 = TransformerEncoderBlock()
        self.fc = nn.Sequential(
            nn.Linear(64,64),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.out = nn.Linear(64,1)

    def forward(self,Xg,XA,Xd):
        g = Xg.unsqueeze(-1)
        d = Xd.unsqueeze(-1)
        tokens = torch.cat([g,d,XA],dim=-1)
        x = self.embed(tokens)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.fc(x)
        x = self.out(x)
        x = x.squeeze(-1)
        return x

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


# mask version
class CommunityDataset_mask(Dataset):

    def __init__(self, Xg, XA, Xd, node_mask, y, augment_permutation=False):

        self.Xg = torch.as_tensor(Xg, dtype=torch.float32)
        self.XA = torch.as_tensor(XA, dtype=torch.float32)
        self.Xd = torch.as_tensor(Xd, dtype=torch.float32)
        self.node_mask = torch.as_tensor(node_mask, dtype=torch.bool)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.augment_permutation = augment_permutation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        Xg = self.Xg[idx].clone()
        XA = self.XA[idx].clone()
        Xd = self.Xd[idx].clone()
        node_mask = self.node_mask[idx]
        y = self.y[idx]

        # Optional: randomly shuffle real species during training
        if self.augment_permutation:
            n_species = int(node_mask.sum().item())

            if n_species > 1:
                perm = torch.randperm(n_species)

                Xg[:n_species] = Xg[:n_species][perm]
                Xd[:n_species] = Xd[:n_species][perm]

                # Rows and columns of gamma must be shuffled synchronously with the same permutation
                A_valid = XA[:n_species, :n_species].clone()
                XA[:n_species, :n_species] = (A_valid[perm][:, perm])

        return Xg, XA, Xd, node_mask, y

class TransformerEncoderBlock_mask(nn.Module):

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

    def forward(self, x, pad_mask=None):
        """
        x:        [B, N, embed_dim]
        pad_mask: [B, N], True means padding.
        """

        attn_out, _ = self.attn(
            x, x, x,
            key_padding_mask=pad_mask,
            need_weights=False
        )

        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        # Clear representations produced by padding queries
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return x

class MDTrans2_mask(nn.Module):

    def __init__(self, NumSpecies):
        super().__init__()
        self.embed = nn.Linear(NumSpecies + 2, 64)
        self.encoder1 = TransformerEncoderBlock_mask()
        self.encoder2 = TransformerEncoderBlock_mask()
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.out = nn.Linear(64, 1)

    def forward(self, Xg, XA, Xd, node_mask):
        """
        Xg:       [B, N]
        XA:       [B, N, N]
        Xd:       [B, N]
        node_mask:[B, N], True means real species.
        """

        node_mask = node_mask.bool()
        pad_mask = ~node_mask

        g = Xg.unsqueeze(-1)
        d = Xd.unsqueeze(-1)

        # Each token corresponds to a receiving species i:
        # [mu_i, D_i, gamma_i1, ..., gamma_iN]
        tokens = torch.cat([g, d, XA], dim=-1)

        x = self.embed(tokens)

        # The embedding layer has bias, so padding tokens need to be zeroed again
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        x = self.encoder1(x, pad_mask)
        x = self.encoder2(x, pad_mask)

        # Mean pooling only over real species
        valid = node_mask.unsqueeze(-1).to(x.dtype)

        x = (x * valid).sum(dim=1)
        x = x / valid.sum(dim=1).clamp_min(1.0)

        x = self.fc(x)

        # Return logits, do not apply sigmoid here during training
        return self.out(x)