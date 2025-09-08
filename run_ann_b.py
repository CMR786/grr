# models/run_ann_b.py
# Author: Oscar Arana

"""
Run ANN-B on a given snapshot_id.
    - Outputs P(Network_Best) and the chosen network.
"""
import os, json, argparse, sys
import numpy as np
import pandas as pd
import torch

sys.path.append (r"C:\\Users\\cmr786\\stuff\\SmartSwitcher")
from models.ann_a import ANNA
from models.ann_b import ANNB

device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
print ("Using device: ", device)

def load_model(model_class, model_path, meta_path, device):
    """
    Generic function to load a model and its metadata.
    """
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    hidden_layers = tuple(meta.get("hidden_layers", (128, 64)))
    in_dim = len(meta["feature_cols"])
    
    model = model_class(in_dim=in_dim, hidden=hidden_layers)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded {model_class.__name__} with architecture: hidden={hidden_layers}")
    return model, meta

def get_snapshot_data(snapshot_id, cdf):
    """
    Extracts data for a specific snapshot_id and ensures both candidates are present.
    """
    g = cdf[cdf.snapshot_id == snapshot_id]
    if g.empty or g["candidate_network"].nunique() < 2:
        raise SystemExit(f"Snapshot {snapshot_id} does not have both LTE and WiFi rows")
    return g

def build_input_a(snap_df, current_network, feature_cols_a):
    """
    Builds the delta-based feature vector for ANN-A.
    """
    target_network = "WiFi" if current_network == "LTE" else "LTE"
    
    current_row = snap_df[snap_df.candidate_network == current_network].iloc[0]
    target_row = snap_df[snap_df.candidate_network == target_network].iloc[0]
    
    # Feature columns for ANN-A are deltas
    delta_cols = [c for c in feature_cols_a if c.startswith('delta_')]
    base_cols = [c.replace('delta_', '') for c in delta_cols]
    
    # --- FIX: Ensure we only use columns that actually exist in the dataframe ---
    # This handles cases where RAG features are in the meta file but not the input CSV
    existing_base_cols = [col for col in base_cols if col in snap_df.columns]
    
    deltas = target_row[existing_base_cols] - current_row[existing_base_cols]
    
    # The last feature is the penalty for switching TO the target
    switch_penalty = target_row['switch_penalty_ms']
    
    # Combine deltas and the penalty
    feature_vector = np.append(deltas.to_numpy(dtype=np.float32), switch_penalty)
    
    return feature_vector

def build_input_b(snap_df, feature_cols_b, candidate_order):
    """
    Builds the paired feature vector for ANN-B.
    """
    order_map = {c: i for i, c in enumerate(candidate_order)}
    g = snap_df.sort_values("candidate_network", key=lambda s: s.map(order_map))
    
    Xi = g[feature_cols_b].to_numpy(dtype=np.float32)
    return Xi, list(g["candidate_network"].values)

def main():
    ap = argparse.ArgumentParser(description="Run cascaded inference with ANN-A and ANN-B.")
    ap.add_argument("--candidates", default="data/candidates.csv")
    ap.add_argument("--snapshots", default="data/snapshots.csv")
    ap.add_argument("--rag", default="data/rag_features.csv", required=False)
    ap.add_argument("--model-a", default="data/ANN-A/ann_a.pt")
    ap.add_argument("--meta-a", default="data/ANN-A/ann_a_meta.json")
    ap.add_argument("--model-b", default="data/ANN-B/ann_b.pt")
    ap.add_argument("--meta-b", default="data/ANN-B/ann_b_meta.json")
    ap.add_argument("--snapshot-id", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for ANN-A to trigger a switch.")
    args = ap.parse_args()

    # --- Load Data Once ---
    cdf = pd.read_csv(args.candidates)
    sdf = pd.read_csv(args.snapshots)
    cdf = cdf.merge(sdf[['snapshot_id', 'current_network']], on='snapshot_id', how='left')

    # --- 1. Run ANN-A (The Gatekeeper) ---
    print("\n--- Step 1: Consulting ANN-A (The Switch Decider) ---")
    model_a, meta_a = load_model(ANNA, args.model_a, args.meta_a, device)
    
    snap_df = get_snapshot_data(args.snapshot_id, cdf)
    current_network = snap_df['current_network'].iloc[0]
    print(f"Current network for snapshot {args.snapshot_id} is: {current_network}")

    Xi_a = build_input_a(snap_df, current_network, meta_a['feature_cols'])
    mean_a = np.array(meta_a['mean'], dtype=np.float32)
    std_a = np.array(meta_a['std'], dtype=np.float32)
    Xn_a = (Xi_a - mean_a) / (std_a + 1e-6)
    xt_a = torch.from_numpy(Xn_a[None, :]).to(device)

    with torch.no_grad():
        logit_a = model_a(xt_a)
        prob_a = torch.sigmoid(logit_a).item()

    decision = "SWITCH" if prob_a >= args.threshold else "NO SWITCH"
    print(f"P(Switch) = {prob_a:.4f} -> Decision: {decision}")

    # --- 2. Conditionally Run ANN-B (The Chooser) ---
    if decision == "NO SWITCH":
        print("\nRecommendation: Stay on the current network.")
        return

    print("\n--- Step 2: Consulting ANN-B (The Network Chooser) ---")
    model_b, meta_b = load_model(ANNB, args.model_b, args.meta_b, device)
    
    Xi_b, names_b = build_input_b (snap_df, meta_b['feature_cols'], meta_b['candidate_order'])
    mean_b = np.array(meta_b['mean'], dtype=np.float32)
    std_b = np.array(meta_b['std'], dtype=np.float32)
    Xn_b = (Xi_b - mean_b) / (std_b + 1e-6)
    xt_b = torch.from_numpy(Xn_b[None, :, :]).to(device)
    
    with torch.no_grad():
        probs_b, _ = model_b(xt_b)
        p_b = probs_b.cpu().numpy()[0]

    best_idx = int(p_b.argmax())
    
    print("\n=== Final Recommendation ===")
    for i, n in enumerate(names_b):
        print(f"{n:5s}: prob = {p_b[i]:.4f}")
    print(f"\nChoice: Switch to {names_b[best_idx]}")


if __name__ == "__main__":
    main()


