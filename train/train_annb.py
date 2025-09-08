# models/train/train_annb.py
# Author: Oscar Arana

"""
Train AN-B to choose the best network per snapshot
Inputs:
    - candidates.csv
    - snapshots.csv
    - rag_features.csv

Outputs:
    - data/annb.pt
    - data/annb_meta.json
"""
import os, json, argparse, sys
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

sys.path.append (r"C:\\Users\\cmr786\\stuff\\SmartSwitcher")
from models.ann_b import ANNB
from models.focal_loss import FocalLoss

device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
print ("Using device: ", device)

CAND_ORDER = ["LTE", "WiFi"]

# Per-candidate features
NUMERIC_COLS = [
    # Performance/Risk/Cost
    "tput_dl_mbps", "tput_ul_mbps", "rtt_ms", "jitter_ms", "loss_rate",
    "p_denial", "energy_mwh_mb", "cost_per_mb", "ambient_load",

    # RF
    "rsrp_dbm", "rsrq_db", "sinr_db",   # LTE
    "rssi_dbm", "snr_db"                # WiFi

    # Time/Context
    "hour", "weekday", "speed_mps"

    # RAG
    "knn_tput_p50", "knn_tput_p90", "knn_rtt_p90", "knn_loss_mean", "knn_p_denial_mean", "knn_avail_rate",
    "switch_penalty_ms"
]
APP_CLASSES = ["bg", "map", "voip"]

def _one_hot_app (app: str) -> List[float]:
    return [1.0 if app == a else 0.0 for a in APP_CLASSES]

def build_frame (cand_csv: str, snap_csv: str, rag_csv: str | None):
    cdf = pd.read_csv (cand_csv)
    sdf = pd.read_csv (snap_csv)[["snapshot_id", "app_class", "hour", "weekday", "speed_mps"]]

    # Recompute label_best_network via net_utility if missing
    if "label_best_network" not in cdf.columns or cdf["label_best_network"].isna ().any():
        best_idx = (
            cdf[cdf["available"] == 1]
            .groupby ("snapshot_id", sort=False)["net_utility"]
            .idxmax ()
        )
        best = cdf.loc[best_idx, ["snapshot_id", "candidate_network"]].rename (
            columns={"candidate_network" : "label_best_network"}
        )
        cdf = cdf.merge (best, on="snapshot_id", how="left")
    
    cdf = cdf.merge (sdf, on="snapshot_id", how="left")

    if rag_csv and os.path.exists (rag_csv):
        rdf = pd.read_csv (rag_csv)
        cdf = cdf.merge (rdf, on=["snapshot_id", "candidate_network"], how="left")

    for col in ["rsrp_dbm", "rsrq_db", "sinr_db", "rssi_dbm", "snr_db"]:
        if col in cdf.columns:
            cdf[col] = pd.to_numeric (cdf[col], errors="coerce").fillna (0.0)

    for col in NUMERIC_COLS:
        if col not in cdf.columns:
            cdf[col] = 0.0
        cdf[col] = pd.to_numeric (cdf[col], errors="coerce").fillna (0.0)

    for a in APP_CLASSES:
        cdf[f"app_{a}"] = (cdf["app_class"] == a).astype (np.float32)

    feature_cols = NUMERIC_COLS + [f"app_{a}" for a in APP_CLASSES]

    # Keep only snapshots where the label_best_network is one of CAND_ORDER
    cdf = cdf[cdf["candidate_network"].isin (CAND_ORDER)]
    has_both = cdf.groupby ("snapshot_id")["candidate_network"].nunique () == len (CAND_ORDER)
    keep_ids = has_both[has_both].index
    cdf = cdf[cdf["snapshot_id"].isin (keep_ids)]

    label_counts = cdf.drop_duplicates (subset="snapshot_id")["label_best_network"].value_counts ()
    print ("\n--- Actual ANN-B Class Distribution ---")
    print (label_counts)
    print ("---------------------------------------\n")

    X_list, Y_list, ids = [], [], []
    grp = cdf.groupby ("snapshot_id", sort=False)
    for sid, g in tqdm (grp, desc="Building Tensors..."):
        g = g.sort_values ("candidate_network", key=lambda s: s.map ({c:i for i, c in enumerate (CAND_ORDER)}))
        Xi = g[feature_cols].to_numpy (dtype=np.float32)
        Yi = CAND_ORDER.index (str (g["label_best_network"].iloc[0]))
        X_list.append (Xi)
        Y_list.append (Yi)
        ids.append (int (sid))

    X = np.stack (X_list, axis=0)
    Y = np.array (Y_list, dtype=np.int64)
    sid_arr = np.array (ids, dtype=np.int64)
    return X, Y, sid_arr, feature_cols

class PairDataset (Dataset):
    def __init__ (self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
        # Standardize over candidate rows
        B, N, D = X.shape
        Xf = X.reshape (B * N, D)
        self.mean = Xf.mean (axis=0)
        self.std = Xf.std (axis=0) + 1e-6
        Xn = (Xf - self.mean) / self.std
        self.Xn = Xn.reshape (B, N, D).astype (np.float32)

    def __len__ (self): return self.X.shape[0]
    def __getitem__ (self, i): return self.Xn[i], self.Y[i]
    def norm_stats (self): return self.mean.tolist (), self.std.tolist ()

def train (args):
    X, Y, sids, feature_cols = build_frame (args.candidates, args.snapshots, args.rag)
    ds = PairDataset (X, Y)

    # Train/val split
    n = len (ds)
    n_val = max (1, int (n * 0.15))
    n_train = n - n_val
    g = torch.Generator ().manual_seed (17)
    train_ds, val_ds = random_split (ds, [n_train, n_val], generator=g)
    train_labels = Y[train_ds.indices]
    class_weights_np = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    print(f"Computed Class Weights: {CAND_ORDER[0]}={class_weights[0]:.2f}, {CAND_ORDER[1]}={class_weights[1]:.2f}")

    train_loader = DataLoader (train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader (val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    hidden_layers = tuple ([256, 128, 64])

    model = ANNB (in_dim=ds.X.shape[2], hidden=hidden_layers, pdrop=0.1).to (device)
    opt = torch.optim.AdamW (model.parameters (), lr=args.lr, weight_decay=1e-4)
    criterion = FocalLoss (alpha=class_weights, gamma=2.0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (opt, mode='min', factor=0.5, patience=5)

    best_val_f1 = 0.0
    for epoch in range (1, args.epochs + 1):
        model.train ()
        tr_loss, tr_acc, m = 0.0, 0.0, 0
        for xb, yb in tqdm (train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            xb = xb.to (device)     # [B, 2, D]
            yb = yb.to (device)     # [B]
            opt.zero_grad ()
            probs, logits = model (xb)
            loss = criterion (logits, yb)
            loss.backward ()
            opt.step ()
            tr_loss += float (loss.item ()) * len (xb)
            pred = logits.argmax (dim=1)
            tr_acc += float ((pred == yb).sum ().item ())
            m += len (xb)

        tr_loss /= m

        model.eval ()
        all_preds, all_labels = [], []
        with torch.no_grad ():
            vl_loss, m = 0.0, 0
            for xb, yb in tqdm (val_loader, desc=f"Epoch {epoch} [val]", leave=False):
                xb = xb.to (device)
                yb = yb.to (device)
                probs, logits = model (xb)
                loss = criterion (logits, yb)
                vl_loss += float (loss.item ()) * len (xb)

                preds = logits.argmax (dim=1)
                all_preds.extend (preds.cpu ().numpy ())
                all_labels.extend (yb.cpu ().numpy ())
                m += len (xb)

            vl_loss /= m

        precision, recall, f1, _ = precision_recall_fscore_support (all_labels, all_preds, average='binary', pos_label=0, zero_division=0)
        val_acc = np.mean (np.array(all_preds) == np.array(all_labels))
        
        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, val_acc={val_acc:.3f} | LTE F1: {f1:.3f}, P: {precision:.3f}, R: {recall:.3f}")


        if f1 > best_val_f1:
            best_val_f1 = f1
            os.makedirs (os.path.dirname(args.out_model), exist_ok=True)
            torch.save (model.state_dict(), args.out_model)
            mean, std = ds.norm_stats ()
            meta = {
                "feature_cols": feature_cols, "mean": mean, "std": std,
                "candidate_order": CAND_ORDER, "hidden_layers": list (hidden_layers)
            }
            with open (args.out_meta, "w") as f: 
                json.dump (meta, f, indent=2)

            print (f"Saved best model (F1={f1:.3f}) to {args.out_model}")

        scheduler.step(vl_loss)

    print ("Done.")

def parse_args ():
    ap = argparse.ArgumentParser ()
    ap.add_argument ("--candidates", default="data\\candidates.csv")
    ap.add_argument ("--snapshots", default="data\\snapshots.csv")
    ap.add_argument ("--rag", default="data\\rag_features.csv")
    ap.add_argument ("--out-model", default="data\\ANN-B\\ann_b.pt")
    ap.add_argument ("--out-meta", default="data\\ANN-B\\ann_b_meta.json")
    ap.add_argument ("--batch-size", type=int, default=1024)
    ap.add_argument ("--epochs", type=int, default=15)
    ap.add_argument ("--lr", type=float, default=1e-4)
    return ap.parse_args ()

if __name__ == "__main__":
    train (parse_args ())