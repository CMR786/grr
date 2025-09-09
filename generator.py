"""
generator.py
Author: Oscar Arana
"""


import math, json, random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd

tqdm.pandas (desc="Postprocessing")

@dataclass
class GenConfig:
    """
    Synthetic data generator configuration
    Defaults are tuned for 2 candidates (LTE, Wifi) around Miami, FL
    """
    n_snapshots: int = 4000
    step_seconds: int = 5
    seed: int = 42
    center_lat: float = 25.7617     # Miami latitude
    center_lon: float = -80.1918    # Miami longitude
    radius_km: float = 8.0
    candidates: tuple = ("LTE", "WiFi", "LMR")

    switch_margin: float = 0.015
    switch_penalty_scale: float = 0.6
    current_choice_tau: float = 0.35
    p_suboptimal_current: float = 0.4

APP_CLASSES = ["voip", "map", "bg"]
APP_PROBS = [0.15, 0.35, 0.5]

def _rng (cfg: GenConfig):
    return np.random.default_rng (cfg.seed)

def _sample_geo (cfg: GenConfig, n: int):
    """
    Uniform sampling within a circle centered at lat0, lon0 on Earth's surface.
    """
    lat0, lon0 = cfg.center_lat, cfg.center_lon
    R = cfg.radius_km / 111.0
    rng = _rng (cfg)
    angles = rng.uniform (0, 2 * np.pi, n)
    radii = np.sqrt (rng.uniform (0, 1, n)) * R
    dlat = radii * np.cos (angles)
    dlon = radii * np.sin (angles) / np.cos (np.deg2rad (lat0))

    return lat0 + dlat, lon0 + dlon

def _bucket (val, step):
    return math.floor (val / step) * step

def _rtt (tech, load, speed_mps, rng):
    """
    Baseline RTT by tech, inflated by load and mobility.
    """
    base = 30 if tech == "LTE" else 20
    mot = 0.05 * speed_mps
    jitter = rng.normal (0, 3)
    return max (5.0, base * (1 + 0.8 * load) + mot + jitter)

def _jitter (rtt, rng) -> float:
    return float (max (1.0, 0.15 * rtt + abs (rng.normal (0, 2))))

def _loss_from_bler (bler, rng) -> float:
    return float (max (0.0, min (0.3, 0.6 * bler + max (0, rng.normal (0, 0.01)))))

def _capacity_from_sinr (sinr_db, bw_mhz) -> float:
    """
    Very rough cellular/WiFi capacity estimation from Shannon mapping.
    """
    sinr_lin = 10 ** (sinr_db / 10)
    eff_bphz = np.log2 (1 + sinr_lin)
    return float (eff_bphz * (bw_mhz * 1e6) / 1e6)

def _wifi_rate (snr_db, bw_mhz) -> float:
    """
    Rough WiFi PHY rate curve that saturates with SNR and bandwidth
    """
    base = (np.log1p (max (snr_db, -5) + 6) / np.log1p (50))
    max_rate = {20: 150, 40: 300, 80: 650, 160: 1200}.get (bw_mhz, 150)
    return float (base * max_rate)

def _denial_prob (access_denials_5m, last_denial_s, load):
    p = 0.01 + 0.02 * access_denials_5m + 0.0005 * max (0, 600 - last_denial_s) + 0.3 * max (0, load - 0.8)
    return float (min (0.35, p))

def _energy_per_mb (tech, rssi_or_rsrp):
    """
    Coarse energy model: weaker signal costs more energy.
    WiFi generally cheaper than LTE.
    """
    base = {"LTE": 0.35, "WiFi": 0.18}.get (tech, 0.35)
    th = -95 if tech == "LTE" else -70
    weak_pen = 0.0
    if rssi_or_rsrp < th:
        weak_pen = min (0.25, (th - rssi_or_rsrp) * 0.004)

    return base + weak_pen

def _switch_outage (current, target, rng):
    """
    Expected outage during switch (ms). LTE<->LTE fast; LTE<->WiFi slower.
    """
    if current == target:
        base = 450.0
    elif {"LTE", "WiFi"} == {current, target}:
        base = 1500.0
    else:
        base = 1000.0

    return float (max (150.0, rng.normal (base, base * 0.25)))

def _utility (tput_mbps, rtt_ms, jitter_ms, loss, energy_mwh_mb, cost_per_mb, switch_pen_ms=0.0):
    """
    A simple utility function to balance QoE vs costs & switching penalty.
    """
    w_tput, w_rtt, w_jit, w_loss, w_energy, w_cost, w_sw = 1.0, 0.8, 0.3, 1.2, 0.3, 0.2, 0.0015
    nt = np.tanh (tput_mbps / 100.0)
    nr = 1 - np.tanh (max (5.0, rtt_ms) / 150.0)
    nj = 1 - np.tanh (jitter_ms / 100.0)
    nl = 1 - min (0.3, loss) / 0.3
    ne = 1 - np.tanh (energy_mwh_mb / 1.2)
    nc = 1 - np.tanh (cost_per_mb / 0.05)
    ns = 1 - np.tanh (switch_pen_ms / 1500.0)

    return (w_tput * nt + w_rtt * nr + w_jit * nj + w_loss * nl + w_energy * ne + w_cost * nc + w_sw * ns)

def generate (cfg: GenConfig, start_ts: datetime):
    """
    Generate two CSV-ready DataFrames:
    - snapshots: one row per decision tick
    - candidates: one row per snapshot x candidate network
    Also compute labels for ANN-A (switch/no-switch) and ANN-B (best network).
    """
    rng = _rng (cfg)
    N = cfg.n_snapshots

    timestamps = [
        start_ts + timedelta (seconds=int (i * cfg.step_seconds + rng.integers (0, 3)))
        for i in range (N)
    ]
    lat, lon = _sample_geo (cfg, N)
    speed = np.clip (rng.normal (6, 4, N), 0, 40)
    heading = rng.uniform (0, 360, N)
    app = rng.choice (APP_CLASSES, size=N, p=APP_PROBS)
    battery = np.clip (rng.normal (65, 22, N), 5, 100).astype (int)

    # SNAPSHOTS TABLE
    snapshots = pd.DataFrame ({
        "snapshot_id": np.arange (N, dtype=int),
        "timestamp": timestamps,
        "lat": lat.astype (float),
        "lon": lon.astype (float),
        "geo_bucket": [f"{math.floor (lat[i] / 0.01) * 0.01 : .2f}_{math.floor (lon[i] / 0.01) * 0.01 : .2f}" for i in range (N)],
        "speed_mps": speed.astype (float),
        "heading_deg": heading.astype (float),
        "weekday": [t.weekday () for t in timestamps],
        "hour": [t.hour for t in timestamps],
        "app_class": app.astype (str),
        "battery_pct": battery.astype (int)
    })

    # CANDIDATES TABLE
    rows = []
    for i in tqdm (range (N), desc=f"Generating candidates", leave=False):
        # Ambient load (shared across candidates)
        # More load if map than bg
        ambient_load = float (np.clip (rng.beta (2, 5) + 0.1 * (app[i] == "map"), 0, 1))
        wifi_available = rng.random () < (0.60 if ambient_load < 0.6 else 0.40)

        for tech in cfg.candidates:
            available = 1
            if tech == "WiFi" and not wifi_available:
                available = 0

            if tech == "LTE":
                rsrp = rng.normal (-92, 8)              # dBm
                rsrq = rng.normal (-10.5, 2.5)          # dB
                sinr = rng.normal (9, 6)                # dB
                ber = max (0.0, min (0.2, 0.02 + 0.002 * (10 - sinr) + abs (rng.normal (0, 0.01))))
                bler = max (0.0, min (0.3, 0.04 + 0.01 * (5 - sinr / 3) + abs (rng.normal (0, 0.02))))
                band = int (rng.choice ([3, 7, 28, 41, 66]))
                bw = int (rng.choice ([10, 15, 20]))
                rssi = np.nan
                snr = np.nan
            elif tech == "WiFi":
                rssi = rng.normal (-58, 8)              # dBm 
                snr = rng.normal (28, 7)                # dB
                rsrp = np.nan
                rsrq = np.nan
                sinr = snr - 3
                ber = max (0.0, min (0.15, 0.01 + 0.003 * (18 - snr) + abs (rng.normal (0, 0.01))))
                bler = max (0.0, min (0.20, 0.02 + 0.006 * (15 - snr) + abs (rng.normal (0, 0.02))))
                band = int (rng.choice ([2, 5, 6]))     # GHz band flags
                bw = int (rng.choice ([20, 40, 80]))    # MHz
            else:
                available = 1 if rng.random () < 0.6 else 0

                rssi = float (rng.normal (-96, 6))
                snr = float (np.clip (rng.normal (9, 4), 0, 20))
                rsrp = rsrq = sinr = ""

                
            access_denials_5m = int (rng.poisson (0.08 + 0.35 * ambient_load))
            last_denial_s = int (rng.integers (30, 1200))
            p_deny = _denial_prob (access_denials_5m, last_denial_s, ambient_load)

            # Throughput estimation
            if tech == "WiFi":
                tput_dl = max (
                    0.5,
                    _wifi_rate (float (snr) if np.isfinite (snr) else 15.0, bw) * (1 - 0.6 * ambient_load) + rng.normal (0, 10)
                )
                tput_ul = max (0.2, 0.6 * tput_dl + rng.normal (0, 6))
            else:
                cap = max (5.0, _capacity_from_sinr (float (sinr) if np.isfinite (sinr) else 5.0, bw))
                tput_dl = max (1.0, cap * 0.55 * (1 - 0.5 - ambient_load) + rng.normal (0, 8))
                tput_ul = max (0.5, 0.5 * tput_dl + rng.normal (0, 4))

            rtt = _rtt (tech, ambient_load, float (snapshots.loc[i, "speed_mps"]), rng)
            jitter = _jitter (rtt, rng)
            loss = _loss_from_bler (float (bler), rng)

            rssi_or_rsrp = float (rssi) if tech == "WiFi" else float (rsrp)
            if not np.isfinite (rssi_or_rsrp):
                rssi_or_rsrp = -95.0

            energy = _energy_per_mb (tech, rssi_or_rsrp)
            cost_per_mb = {"LTE": 0.004, "WiFi": 0.0005}.get (tech, 0.004)

            util = _utility (tput_dl, rtt, jitter, loss, energy, cost_per_mb, 0.0)

            rows.append ({
                "snapshot_id": i,
                "candidate_network": tech,
                "available": int (available),
                "ambient_load": float (ambient_load),

                # RF (cellular vs WiFi: leave missing as empty string for CSV friendliness)
                "rsrp_dbm": float (rsrp) if np.isfinite (rsrp) else "",
                "rsrq_db": float (rsrq) if np.isfinite (rsrq) else "",
                "sinr_db": float (sinr) if np.isfinite (sinr) else "",
                "rssi_dbm": float (rssi) if np.isfinite (rssi) else "",
                "snr_db": float (snr) if np.isfinite (snr) else "",
                "ber": float (ber),
                "bler": float (bler),
                "band": int (band),
                "chan_mb_mhz": int (bw),

                # Access issues
                "access_denials_5m": int (access_denials_5m),
                "last_denial_s": int (last_denial_s),
                "p_denial": float (p_deny),

                # Performance
                "tput_dl_mbps": float (tput_dl),
                "tput_ul_mbps": float (tput_ul),
                "rtt_ms": float (rtt),
                "jitter_ms": float (jitter),
                "loss_rate": float (loss),

                # Costs/Energy
                "energy_mwh_mb": float (energy),
                "cost_per_mb": float (cost_per_mb),

                # Utility
                "standalone_utility": float (util),
            })

    candidates = pd.DataFrame (rows)

    def _softmax (x, tau):
        x = np.array (x, dtype=float)
        x = (x - x.max ()) / max (1e-9, tau)
        ex = np.exp (x)
        return ex / (ex.sum () + 1e-9)
    
    def _choose_current_for_snapshot (sub_df, cfg, rng):
        sub = sub_df[sub_df.available == 1]
        if len (sub) == 0:
            return "WiFi"
        
        util = sub["standalone_utility"].to_numpy ()
        probs = _softmax (util, tau=cfg.current_choice_tau)

        if rng.random () < cfg.p_suboptimal_current:
            return str (sub.iloc[rng.choice (len (sub), p=probs)]["candidate_network"])
        else:
            return str (sub.iloc[int (np.argmax (util))]["candidate_network"])

    # Choose current network per snapshot (noisy best available)
    current = []
    for i in tqdm (range (N), desc=f"Choosing current network per snapshot", leave=False):
        sub = candidates[candidates.snapshot_id == i]
        current.append (_choose_current_for_snapshot (sub, cfg, rng))

    snapshots["current_network"] = current

    # add switch penalty and net utility per candidate
    def _swpen (cur, cand, cfg, rng):
        if cand == cur:
            return 0.0
        raw = _switch_outage (cur, cand, rng)
        return float (max (0.0, raw * cfg.switch_penalty_scale))
    
    candidates["switch_penalty_ms"] = candidates.progress_apply(
        lambda row: _swpen(
            snapshots.loc[snapshots.snapshot_id == row["snapshot_id"], "current_network"].values[0],
            row["candidate_network"],
            cfg,
            rng
        ),
        axis=1
    )

    def _netu (r):
        return _utility (
            r["tput_dl_mbps"], r["rtt_ms"], r["jitter_ms"], r["loss_rate"],
            r["energy_mwh_mb"], r["cost_per_mb"], r["switch_penalty_ms"]
        )
    
    candidates["net_utility"] = candidates.progress_apply (_netu, axis=1)

    best = (
        candidates[candidates["available"] == 1]
        .loc[:, ["snapshot_id", "candidate_network", "net_utility"]]
        .sort_values (["snapshot_id", "net_utility"], ascending=[True, False])
        .groupby ("snapshot_id")
        .first ()
        .reset_index ()
        .rename (columns= {
            "candidate_network": "label_best_network",
            "net_utility": "label_best_net_utility"
        })
    )
    candidates = candidates.merge (best, on="snapshot_id", how="left")

    # Label for ANN-A: switch or not
    def _switch_label (row):
        cur = snapshots.loc[snapshots.snapshot_id == row["snapshot_id"], "current_network"].values[0]
        cur_util = candidates[
            (candidates.snapshot_id == row["snapshot_id"]) &
            (candidates.candidate_network == cur)
        ].iloc[0]["net_utility"]

        return int (row["candidate_network"] != cur and (row["net_utility"] - cur_util) > cfg.switch_margin)
    
    candidates["label_switch_now"] = candidates.progress_apply (_switch_label, axis=1)



    return snapshots, candidates

def build_embeddings (candidates_df: pd.DataFrame, snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple, deterministic embedding per (snapshot, candidate) row
    CSV friendly and ready for Vector DB ingestion.

    Column used:
    - RF: rsrp_dbm, rsrq_db, sinr_db, rssi_dbm, snr_db
    - Context: hour, weekday, speed_mps (joined from snapshots)
    """
    cols = ["rsrp_dbm", "rsrq_db", "sinr_db", "rssi_dbm", "snr_db", "hour", "weekday", "speed_mps"]
    df = candidates_df.merge (
        snapshots_df[["snapshot_id", "hour", "weekday", "speed_mps"]],
        on="snapshot_id",
        how="left"
    )

    # Convert to numeric
    X_cols = []
    for c in cols:
        v = pd.to_numeric (df[c], errors='coerce').fillna (0.0).astype (float)
        X_cols.append (v)

    X = np.vstack (X_cols).T

    # Normalize each column then add tanh nonlinearity for a richer embedding
    X_norm = (X - X.mean (axis=0)) / (X.std (axis=0) + 1e-6)
    Z = np.hstack ([X_norm, np.tanh (X_norm)])
    emb_cols = [f"e({i})" for i in range (Z.shape[1])]

    emb = pd.DataFrame (Z, columns=emb_cols)
    emb.insert (0, "candidate_network", df["candidate_network"].values)
    emb.insert (0, "snapshot_id", df["snapshot_id"].values)

    return emb