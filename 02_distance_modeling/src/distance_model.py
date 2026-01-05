"""
Distance-model pipeline supporting both linear and nonlinear models.

Expected project layout (project root):
- data/tower/towers.json
- data/tower/pci.json (optional, only if fallback enabled)
- data/<context>/*.csv (e.g. data/waha/parsed_waha_1_city.csv)
- data/<context>/context.json (optional overrides)
- result/<context>/... (CSV outputs)
- report/<context>/... (quick TXT report)

Run via main.py:
python main.py waha
python main.py lln --nonlinear
"""

from __future__ import annotations

import os
import json
import glob
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from src.export_formula import generate_formula_json 

R_EARTH = 6371000.0  # meters

def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_context_config(context_dir: str, tower_dir: str) -> dict:
    """Load data/<context>/context.json if present; otherwise return defaults."""
    default = {
        "csv_glob": "parsed_*_city.csv",
        "towers_path": os.path.join(tower_dir, "towers.json"),
        "use_pci_fallback": False,
        "pci_path": os.path.join(tower_dir, "pci.json"),
        "max_neighbors": 18,
        "keep_only_lte_neighbors": True,
        "max_tower_distance_km": 10.0,
    }
    
    cfg_path = os.path.join(context_dir, "context.json")
    if not os.path.exists(cfg_path):
        return default
    
    user_cfg = _read_json(cfg_path) or {}
    out = default.copy()
    out.update(user_cfg)
    
    # Allow relative paths in context.json
    for k in ["towers_path", "pci_path"]:
        if (
            k in out
            and isinstance(out[k], str)
            and out[k]
            and not os.path.isabs(out[k])
        ):
            out[k] = os.path.normpath(os.path.join(context_dir, out[k]))
    
    return out

def load_towers(towers_path: str) -> pd.DataFrame:
    raw = _read_json(towers_path)
    rows = []
    for key, v in raw.items():
        if not isinstance(v, dict):
            continue
        if v.get("pci") is None or v.get("earfcn") is None:
            continue
        if v.get("latitude") is None or v.get("longitude") is None:
            continue
        rows.append({
            "tower_key": key,
            "pci": int(v["pci"]),
            "earfcn": int(v["earfcn"]),
            "lat_tower": float(v["latitude"]),
            "lon_tower": float(v["longitude"]),
            "band": v.get("band"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid towers found in {towers_path}")
    return df.drop_duplicates(subset=["pci", "earfcn"]).copy()

def load_pci_map(pci_path: str) -> pd.DataFrame:
    raw = _read_json(pci_path)
    rows = []
    for _key, v in raw.items():
        if not isinstance(v, dict):
            continue
        if v.get("pci") is None:
            continue
        lat = v.get("latitude") if v.get("latitude") is not None else v.get("lat")
        lon = v.get("longitude") if v.get("longitude") is not None else v.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({
            "pci": int(v["pci"]),
            "lat_tower": float(lat),
            "lon_tower": float(lon),
            "band": v.get("band"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid PCI rows (with lat/lon) found in {pci_path}")
    return df.sort_values(["pci"]).drop_duplicates(subset=["pci"]).copy()

def load_measurements(context_dir: str, csv_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(context_dir, csv_glob)))
    if not paths:
        raise FileNotFoundError(f"No CSV files match {csv_glob} in {context_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df["source_file"] = os.path.basename(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def build_pairs(
    meas: pd.DataFrame,
    max_neighbors: int = 18,
    keep_only_lte_neighbors: bool = True,
) -> pd.DataFrame:
    """Convert wide parsed drive-test rows into long-format rows."""
    meas = meas.rename(columns={
        "Latitude": "lat_phone",
        "Longitude": "lon_phone",
        "Timestamp": "ts",
        "PSC": "pci_serv",
        "ARFCN": "earfcn_serv",
        "Level": "rsrp_serv",
        "Qual": "rsrq_serv",
        "SNR": "snr_serv",
        "LTERSSI": "rssi_serv",
    })
    
    needed = ["lat_phone", "lon_phone", "pci_serv", "earfcn_serv", "rsrp_serv"]
    for c in needed:
        if c not in meas.columns:
            raise ValueError(f"Missing required column: {c}")
    
    base = [c for c in ["ts", "source_file", "lat_phone", "lon_phone"] if c in meas.columns]
    
    # Serving row
    serv_cols = base + [
        "pci_serv", "earfcn_serv", "rsrp_serv",
        "rsrq_serv", "snr_serv", "rssi_serv",
    ]
    serv_cols = [c for c in serv_cols if c in meas.columns]
    serv = meas[serv_cols].copy().rename(columns={
        "pci_serv": "pci", "earfcn_serv": "earfcn", "rsrp_serv": "rsrp",
        "rsrq_serv": "rsrq", "snr_serv": "snr", "rssi_serv": "rssi",
    })
    serv["is_neighbor"] = 0
    serv["neighbor_rank"] = 0
    
    # Neighbor rows
    neigh_frames = []
    for i in range(1, max_neighbors + 1):
        cols = {
            f"NCell{i}": "pci",
            f"NARFCN{i}": "earfcn",
            f"NRxLev{i}": "rsrp",
            f"NQual{i}": "rsrq",
            f"NTech{i}": "ntech",
        }
        present = [c for c in cols if c in meas.columns]
        if not present:
            continue
        tmp = meas[base + present].rename(columns=cols).copy()
        tmp["is_neighbor"] = 1
        tmp["neighbor_rank"] = i
        neigh_frames.append(tmp)
    
    neigh = pd.concat(neigh_frames, ignore_index=True) if neigh_frames else pd.DataFrame()
    if keep_only_lte_neighbors and (not neigh.empty) and ("ntech" in neigh.columns):
        neigh = neigh[neigh["ntech"].astype(str).str.contains("4G", na=False)].copy()
    
    pairs = pd.concat([serv, neigh], ignore_index=True)
    
    for c in ["pci", "earfcn", "rsrp", "rsrq", "snr", "rssi"]:
        if c in pairs.columns:
            pairs[c] = pd.to_numeric(pairs[c], errors="coerce")
    
    pairs = pairs.dropna(
        subset=["lat_phone", "lon_phone", "pci", "earfcn", "rsrp"]
    ).copy()
    pairs["pci"] = pairs["pci"].astype(int)
    pairs["earfcn"] = pairs["earfcn"].astype(int)
    
    return pairs

def attach_tower_coords(
    pairs: pd.DataFrame,
    towers_exact: pd.DataFrame,
    use_pci_fallback: bool = False,
    pci_df: Optional[pd.DataFrame] = None,
    max_distance_km: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attach lat/lon for tower; returns (matched_pairs, unmatched_summary).
    """
    max_dist_m = float(max_distance_km) * 1000.0
    pairs = pairs.copy()
    
    # Initialize columns
    for col in ["lat_tower", "lon_tower", "band", "tower_join", "tower_distance_m"]:
        if col not in pairs.columns:
            pairs[col] = np.nan
    
    towers_exact = towers_exact.copy()
    towers_exact["pci"] = towers_exact["pci"].astype(int)
    towers_exact["earfcn"] = towers_exact["earfcn"].astype(int)
    
    if pci_df is not None:
        pci_df = pci_df.copy()
        pci_df["pci"] = pci_df["pci"].astype(int)
    
    # Step 1: exact (pci, earfcn) with proximity
    for idx in pairs.index:
        pci = int(pairs.at[idx, "pci"])
        earfcn = int(pairs.at[idx, "earfcn"])
        lat_phone = float(pairs.at[idx, "lat_phone"])
        lon_phone = float(pairs.at[idx, "lon_phone"])
        
        candidates = towers_exact[
            (towers_exact["pci"] == pci) & (towers_exact["earfcn"] == earfcn)
        ]
        
        if candidates.empty:
            continue
        
        dists = haversine_m(
            lat_phone, lon_phone,
            candidates["lat_tower"].values,
            candidates["lon_tower"].values,
        )
        candidates = candidates.assign(dist_m=dists)
        candidates = candidates[candidates["dist_m"] <= max_dist_m]
        
        if candidates.empty:
            continue
        
        closest = candidates.loc[candidates["dist_m"].idxmin()]
        pairs.at[idx, "lat_tower"] = float(closest["lat_tower"])
        pairs.at[idx, "lon_tower"] = float(closest["lon_tower"])
        pairs.at[idx, "band"] = closest.get("band")
        pairs.at[idx, "tower_join"] = "pci+earfcn"
        pairs.at[idx, "tower_distance_m"] = float(closest["dist_m"])
    
    # Step 2: PCI-only fallback
    if use_pci_fallback:
        if pci_df is None:
            raise ValueError("PCI fallback requested but pci_df is None")
        missing_idx = pairs[pairs["lat_tower"].isna()].index
        for idx in missing_idx:
            pci = int(pairs.at[idx, "pci"])
            lat_phone = float(pairs.at[idx, "lat_phone"])
            lon_phone = float(pairs.at[idx, "lon_phone"])
            
            candidates = pci_df[pci_df["pci"] == pci]
            if candidates.empty:
                continue
            
            dists = haversine_m(
                lat_phone, lon_phone,
                candidates["lat_tower"].values,
                candidates["lon_tower"].values,
            )
            candidates = candidates.assign(dist_m=dists)
            candidates = candidates[candidates["dist_m"] <= max_dist_m]
            
            if candidates.empty:
                continue
            
            closest = candidates.loc[candidates["dist_m"].idxmin()]
            pairs.at[idx, "lat_tower"] = float(closest["lat_tower"])
            pairs.at[idx, "lon_tower"] = float(closest["lon_tower"])
            pairs.at[idx, "band"] = closest.get("band")
            pairs.at[idx, "tower_join"] = "pci_json"
            pairs.at[idx, "tower_distance_m"] = float(closest["dist_m"])
    
    # Step 3: unmatched summary
    unmatched = pairs[pairs["lat_tower"].isna()].copy()
    unmatched_summary = (
        unmatched.groupby(["pci", "earfcn", "is_neighbor"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    
    matched = pairs.dropna(subset=["lat_tower", "lon_tower"]).copy()
    return matched, unmatched_summary

def add_labels(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["d_true_m"] = haversine_m(
        pairs["lat_phone"].astype(float).values,
        pairs["lon_phone"].astype(float).values,
        pairs["lat_tower"].astype(float).values,
        pairs["lon_tower"].astype(float).values,
    )
    pairs["logd"] = np.log10(np.clip(pairs["d_true_m"].values, 1.0, None))
    pairs["earfcn_k"] = pairs["earfcn"] / 1000.0
    return pairs

def fit_log10_distance(train_df: pd.DataFrame, feats: List[str]) -> np.ndarray:
    df = train_df.dropna(subset=feats + ["logd"]).copy()
    y = df["logd"].astype(float).values
    X = df[feats].astype(float).values
    X = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def predict_distance(df: pd.DataFrame, feats: List[str], beta: np.ndarray) -> np.ndarray:
    X = df[feats].astype(float).values
    X = np.c_[np.ones(len(X)), X]
    return 10 ** (X @ beta)

def eval_distance(df: pd.DataFrame, feats: List[str], beta: np.ndarray) -> dict:
    df = df.dropna(subset=feats + ["d_true_m"]).copy()
    d_hat = predict_distance(df, feats, beta)
    err = d_hat - df["d_true_m"].values
    return {
        "n": int(len(df)),
        "mae_m": float(np.mean(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "median_ae_m": float(np.median(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / df["d_true_m"].values)),
    }

def lofo_cv(
    pairs: pd.DataFrame,
    candidates: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-file-out CV on source_file.
    Returns: per_fold, summary, coef_summary.
    """
    files = sorted(pairs["source_file"].dropna().unique().tolist())
    per_fold_rows = []
    
    for test_file in files:
        train = pairs[pairs["source_file"] != test_file].copy()
        test = pairs[pairs["source_file"] == test_file].copy()
        
        for model_name, feats in candidates.items():
            feats = [f for f in feats if f in pairs.columns]
            tr = train.dropna(subset=feats + ["logd"]) if feats else pd.DataFrame()
            te = test.dropna(subset=feats + ["d_true_m"]) if feats else pd.DataFrame()
            
            if len(tr) < max(50, 10 * max(1, len(feats))) or len(te) < 20:
                continue
            
            beta = fit_log10_distance(tr, feats)
            metrics = eval_distance(te, feats, beta)
            
            row = {
                "fold_test_file": test_file,
                "model": model_name,
                "features": ",".join(feats),
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "intercept": float(beta[0]),
            }
            for j, f in enumerate(feats, start=1):
                row[f"coef_{f}"] = float(beta[j])
            row.update(metrics)
            per_fold_rows.append(row)
    
    per_fold = pd.DataFrame(per_fold_rows)
    if per_fold.empty:
        return per_fold, per_fold, per_fold
    
    # Summary
    summary = (
        per_fold.groupby(["model", "features"], as_index=False)
        .agg(
            folds=("fold_test_file", "nunique"),
            mae_mean_m=("mae_m", "mean"),
            mae_std_m=("mae_m", "std"),
            rmse_mean_m=("rmse_m", "mean"),
            rmse_std_m=("rmse_m", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            test_n_total=("test_n", "sum"),
        )
        .sort_values(["mae_mean_m", "rmse_mean_m"], ascending=True)
    )
    
    # Coef summary
    coef_cols = [c for c in per_fold.columns if c.startswith("coef_") or c == "intercept"]
    agg_dict = {}
    for c in coef_cols:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"] = (c, "std")
    coef_summary = (
        per_fold.groupby(["model", "features"], as_index=False)
        .agg(**agg_dict)
        .sort_values(["model"])
    )
    
    return per_fold, summary, coef_summary

# Nonlinear feature generation (from original file)
def build_nonlinear_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add polynomial and interaction features exactly as before."""
    df = pairs.copy()
    
    # Base
    df["earfcn_k"] = df["earfcn"].astype(float) / 1000.0
    
    # Polynomials
    df["rsrp_sq"] = df["rsrp"].astype(float) ** 2
    df["earfcn_k_sq"] = df["earfcn_k"].astype(float) ** 2
    if "rsrq" in df.columns:
        df["rsrq_sq"] = df["rsrq"].astype(float) ** 2
    
    # Interactions
    df["rsrp_x_earfcnk"] = df["rsrp"].astype(float) * df["earfcn_k"].astype(float)
    if "rsrq" in df.columns:
        df["rsrp_x_rsrq"] = df["rsrp"].astype(float) * df["rsrq"].astype(float)
        df["rsrq_x_earfcnk"] = df["rsrq"].astype(float) * df["earfcn_k"].astype(float)
    if "is_neighbor" in df.columns:
        df["rsrp_x_is_neighbor"] = df["rsrp"].astype(float) * df["is_neighbor"].astype(float)
    
    return df

def get_nonlinear_candidates() -> Dict[str, List[str]]:
    """Return the exact same 14 candidates as the original file."""
    return {
        # Baseline
        "B1 logd ~ rsrp": ["rsrp"],
        "B2 logd ~ rsrp + rsrq": ["rsrp", "rsrq"],
        "B3 logd ~ rsrp + rsrq + earfcn_k": ["rsrp", "rsrq", "earfcn_k"],
        "B4 logd ~ rsrp + rsrq + earfcn_k + is_neighbor": [
            "rsrp", "rsrq", "earfcn_k", "is_neighbor"
        ],
        # Polynomial
        "P1 logd ~ rsrp + rsrp_sq": ["rsrp", "rsrp_sq"],
        "P2 logd ~ rsrp + rsrq + rsrp_sq": ["rsrp", "rsrq", "rsrp_sq"],
        "P3 logd ~ rsrp + rsrq + rsrp_sq + rsrq_sq": ["rsrp", "rsrq", "rsrp_sq", "rsrq_sq"],
        "P4 logd ~ rsrp + rsrq + earfcn_k + earfcn_k_sq": [
            "rsrp", "rsrq", "earfcn_k", "earfcn_k_sq"
        ],
        # Interactions
        "I1 logd ~ rsrp + rsrq + earfcn_k + rsrp_x_earfcnk": [
            "rsrp", "rsrq", "earfcn_k", "rsrp_x_earfcnk"
        ],
        "I2 logd ~ rsrp + rsrq + earfcn_k + rsrp_x_rsrq": [
            "rsrp", "rsrq", "earfcn_k", "rsrp_x_rsrq"
        ],
        "I3 logd ~ rsrp + rsrq + earfcn_k + rsrq_x_earfcnk": [
            "rsrp", "rsrq", "earfcn_k", "rsrq_x_earfcnk"
        ],
        "I4 logd ~ rsrp + rsrq + earfcn_k + is_neighbor + rsrp_x_is_neighbor": [
            "rsrp", "rsrq", "earfcn_k", "is_neighbor", "rsrp_x_is_neighbor"
        ],
        # Combined
        "C1 logd ~ rsrp + rsrq + earfcn_k + rsrp_sq + rsrp_x_earfcnk": [
            "rsrp", "rsrq", "earfcn_k", "rsrp_sq", "rsrp_x_earfcnk"
        ],
        "C2 logd ~ rsrp + rsrq + earfcn_k + rsrp_sq + rsrq_sq + rsrp_x_rsrq": [
            "rsrp", "rsrq", "earfcn_k", "rsrp_sq", "rsrq_sq", "rsrp_x_rsrq"
        ],
    }

def _format_report(
    context_name: str,
    cfg: dict,
    pairs: pd.DataFrame,
    unmatched_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    result_csv_path: str,
    mode: str,
) -> str:
    """Create human-readable summary."""
    n_files = int(pairs["source_file"].nunique()) if "source_file" in pairs.columns else 0
    join_counts = (
        pairs["tower_join"].value_counts(dropna=False).to_dict()
        if "tower_join" in pairs.columns else {}
    )
    
    lines: List[str] = []
    lines.append(f"Context: {context_name} ({mode})")
    lines.append(f"CSV pattern: {cfg.get('csv_glob')}")
    lines.append(f"Towers: {cfg.get('towers_path')}")
    lines.append(f"PCI fallback: {bool(cfg.get('use_pci_fallback'))}")
    if bool(cfg.get("use_pci_fallback")):
        lines.append(f"PCI map: {cfg.get('pci_path')}")
    lines.append(f"Max tower distance (km): {cfg.get('max_tower_distance_km')}")
    lines.append("")
    
    lines.append(f"Rows used (pairs): {len(pairs)}")
    lines.append(f"Source files: {n_files}")
    if join_counts:
        lines.append("Tower join counts:")
        for k, v in join_counts.items():
            lines.append(f" - {k}: {v}")
    lines.append("")
    
    if model_summary is None or model_summary.empty:
        lines.append("Model summary: EMPTY (not enough data)")
    else:
        best = model_summary.iloc[0].to_dict()
        lines.append("Best model (lowest MAE mean):")
        lines.append(f" - model: {best.get('model')}")
        lines.append(f" - features: {best.get('features')}")
        lines.append(f" - folds: {best.get('folds')}")
        lines.append(f" - mae_mean_m: {best.get('mae_mean_m')}")
        lines.append(f" - rmse_mean_m: {best.get('rmse_mean_m')}")
        lines.append(f" - mape_mean: {best.get('mape_mean')}")
    lines.append("")
    
    if unmatched_summary is not None and (not unmatched_summary.empty):
        lines.append("Top unmatched (pci, earfcn) pairs:")
        top = unmatched_summary.head(10)
        for _, r in top.iterrows():
            lines.append(
                f" - pci={int(r['pci'])}, earfcn={int(r['earfcn'])}, "
                f"is_neighbor={int(r['is_neighbor'])}, count={int(r['size'])}"
            )
    lines.append("")
    lines.append(f"Result CSV: {result_csv_path}")
    
    return "\n".join(lines) + "\n"

def run_context(
    context_name: str,
    nonlinear: bool = False,
    data_root: str = "data",
    tower_dir: str = os.path.join("data", "tower"),
    results_root: str = "result",
    report_root: str = "report",
) -> dict:
    """
    Unified pipeline: linear (default) or nonlinear (--nonlinear).
    
    Outputs CSVs to result/<context>/ and TXT to report/<context>/.
    """
    context_dir = os.path.join(data_root, context_name)
    if not os.path.isdir(context_dir):
        raise FileNotFoundError(f"Context folder not found: {context_dir}")
    
    cfg = load_context_config(context_dir, tower_dir=tower_dir)
    towers_exact = load_towers(cfg["towers_path"])
    
    pci_df = None
    if bool(cfg.get("use_pci_fallback")):
        pci_path = cfg.get("pci_path")
        if not pci_path or not os.path.exists(pci_path):
            raise FileNotFoundError(f"PCI fallback enabled but pci.json not found: {pci_path}")
        pci_df = load_pci_map(pci_path)
    
    meas = load_measurements(context_dir, cfg["csv_glob"])
    pairs = build_pairs(
        meas,
        max_neighbors=int(cfg.get("max_neighbors", 18)),
        keep_only_lte_neighbors=bool(cfg.get("keep_only_lte_neighbors", True)),
    )
    
    pairs, unmatched = attach_tower_coords(
        pairs,
        towers_exact=towers_exact,
        use_pci_fallback=bool(cfg.get("use_pci_fallback")),
        pci_df=pci_df,
        max_distance_km=float(cfg.get("max_tower_distance_km", 10.0)),
    )
    
    pairs = add_labels(pairs)
    
    mode_name = "nonlinear" if nonlinear else "linear"
    suffix = f"_nonlinear" if nonlinear else "_linear" if "pci_fallback" in str(cfg) else ""
    
    if nonlinear:
        pairs = build_nonlinear_features(pairs)
        candidates = get_nonlinear_candidates()
    else:
        # Default linear baseline
        candidates = {
            "F1 logd ~ rsrp": ["rsrp"],
            "F2 logd ~ rsrp + rsrq": ["rsrp", "rsrq"],
            "F3 logd ~ rsrp + rsrq + earfcn_k": ["rsrp", "rsrq", "earfcn_k"],
            "F4 logd ~ rsrp + rsrq + earfcn_k + is_neighbor": [
                "rsrp", "rsrq", "earfcn_k", "is_neighbor",
            ],
        }
    
    per_fold, summary, coef_summary = lofo_cv(pairs, candidates)
    
    # Create output dirs
    os.makedirs(results_root, exist_ok=True)
    context_results_dir = os.path.join(results_root, context_name)
    os.makedirs(context_results_dir, exist_ok=True)
    os.makedirs(report_root, exist_ok=True)
    context_report_dir = os.path.join(report_root, context_name)
    os.makedirs(context_report_dir, exist_ok=True)
    
    # CSV paths
    summary_path = os.path.join(context_results_dir, f"result_{context_name}{suffix}.csv")
    per_fold_path = os.path.join(context_results_dir, f"{context_name}_distance_models_per_fold{suffix}.csv")
    pairs_path = os.path.join(context_results_dir, f"{context_name}_training_pairs{suffix}.csv")
    unmatched_path = os.path.join(context_results_dir, f"{context_name}_unmatched_pairs{suffix}.csv")
    coef_path = os.path.join(context_results_dir, f"{context_name}_model_params{suffix}.csv")
    
    summary.to_csv(summary_path, index=False)
    per_fold.to_csv(per_fold_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)
    coef_summary.to_csv(coef_path, index=False)
    
    # Formula JSON
    formula_path = os.path.join(context_results_dir, f"{context_name}_formula{suffix}.json")
    try:
        generate_formula_json(
            context_name=context_name,
            summary=summary,
            coef_summary=coef_summary,
            n_train_samples=int(len(pairs)),
            output_path=formula_path,
        )
    except Exception as e:
        print(f"Warning: Could not generate formula JSON: {e}")
        formula_path = None
    
    # TXT report
    report_txt_path = os.path.join(context_report_dir, f"summary_{context_name}{suffix}.txt")
    report_txt = _format_report(
        context_name=context_name,
        cfg=cfg,
        pairs=pairs,
        unmatched_summary=unmatched,
        model_summary=summary,
        result_csv_path=summary_path,
        mode=mode_name,
    )
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
    
    return {
        "summary": summary_path,
        "report": report_txt_path,
        "pairs": pairs_path,
        "per_fold": per_fold_path,
        "unmatched": unmatched_path,
        "params": coef_path,
        "formula": formula_path,
        "n_rows": int(len(pairs)),
        "tower_join_counts": pairs["tower_join"].value_counts(dropna=False).to_dict()
        if "tower_join" in pairs.columns else {},
    }
