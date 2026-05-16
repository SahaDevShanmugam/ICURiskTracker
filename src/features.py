"""Rolling, trend, and alarm-context features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.io_schema import LABEL_HELPER_COLUMNS


BASE_KEEP = [
    "stay_id",
    "subject_id",
    "charttime",
    "age",
    "sex",
    "y_deterioration",
]


def _rolling_stats(s: pd.Series, w: int, name: str) -> pd.DataFrame:
    # Use min_periods=1 for mean/min so early timesteps reflect partial windows (current vitals),
    # not NaN→0 which reads as "MAP/SBP collapsed to zero" to the deterioration model.
    mean_r = s.rolling(window=w, min_periods=1).mean()
    min_r = s.rolling(window=w, min_periods=1).min()
    std_min_periods = 2 if w >= 2 else 1
    std_r = s.rolling(window=w, min_periods=std_min_periods).std()
    return pd.DataFrame(
        {
            f"{name}_mean_{w}": mean_r,
            f"{name}_std_{w}": std_r,
            f"{name}_min_{w}": min_r,
        }
    )


def _slope(y: np.ndarray) -> float:
    if len(y) < 2 or not np.isfinite(y).all():
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def compute_timeseries_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    windows = cfg["features"]["rolling_windows_minutes"]
    slope_pts = int(cfg["features"]["slope_points"])
    feats = df[["stay_id", "subject_id", "charttime", "age", "sex"]].copy()
    if "y_deterioration" in df.columns:
        feats["y_deterioration"] = df["y_deterioration"]

    for stay_id, g in df.groupby("stay_id", sort=False):
        idx = g.index
        for col in ["map", "sbp", "hr", "rr"]:
            s = g[col].astype(np.float64)
            for wmin in windows:
                w = max(1, int(round(wmin / grid)))
                roll = _rolling_stats(s, w, col)
                for c in roll.columns:
                    feats.loc[idx, c] = roll[c].to_numpy()
            slopes = []
            for i in range(len(s)):
                lo = max(0, i - slope_pts + 1)
                slopes.append(_slope(s.to_numpy()[lo : i + 1]))
            feats.loc[idx, f"{col}_slope"] = slopes

    if "lactate" in df.columns:
        feats["lactate_filled"] = df.groupby("stay_id", sort=False)["lactate"].ffill()

    feat_cols = [c for c in feats.columns if c not in BASE_KEEP and c != "y_deterioration"]
    for c in feat_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)
    return feats


def _baseline_map_prior(g: pd.DataFrame, alarm_idx: int, prior_steps: int) -> float:
    lo = max(0, alarm_idx - prior_steps)
    seg = g["map"].to_numpy()[lo:alarm_idx]
    if seg.size == 0:
        return float(g["map"].iloc[alarm_idx])
    return float(np.nanmean(seg))


def build_alarm_feature_table(
    ts_full: pd.DataFrame,
    ts_feats: pd.DataFrame,
    episodes: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Join per-alarm rows with features at alarm onset + alarm-specific scalars."""
    grid = int(cfg["data"]["grid_minutes"])
    prior_steps = max(3, int(round(30 / grid)))
    map_thr = cfg["alarms"]["map_threshold_mmhg"]
    rows = []
    for _, ep in episodes.iterrows():
        sid = ep["stay_id"]
        ai = int(ep["alarm_start_idx"])
        g = ts_full[ts_full["stay_id"] == sid].reset_index(drop=True)
        gf = ts_feats[ts_feats["stay_id"] == sid].reset_index(drop=True)
        if ai >= len(g):
            continue
        map_v = float(g.loc[ai, "map"])
        sbp_v = float(g.loc[ai, "sbp"])
        base = _baseline_map_prior(g, ai, prior_steps)
        row = {
            "stay_id": sid,
            "subject_id": ep["subject_id"],
            "alarm_start_idx": ai,
            "alarm_start_time": ep["alarm_start_time"],
            "y_alarm_valid": ep["y_alarm_valid"],
            "map_at_onset": map_v,
            "sbp_at_onset": sbp_v,
            "map_drop_from_baseline": base - map_v,
            "episode_length_steps": ep["episode_length_steps"],
        }
        for c in gf.columns:
            if c in ("stay_id", "subject_id", "charttime", "y_deterioration"):
                continue
            if c in LABEL_HELPER_COLUMNS:
                continue
            row[c] = gf.loc[ai, c]
        rows.append(row)
    alarm_df = pd.DataFrame(rows)
    alarm_df["below_map_threshold"] = (alarm_df["map_at_onset"] < map_thr).astype(np.float64)
    return alarm_df


def feature_columns_deterioration(ts_feats: pd.DataFrame) -> list[str]:
    exclude = set(BASE_KEEP + LABEL_HELPER_COLUMNS + ["charttime"])
    if "y_deterioration" in ts_feats.columns:
        exclude.add("y_deterioration")
    return [c for c in ts_feats.columns if c not in exclude]


def feature_columns_alarm(alarm_df: pd.DataFrame) -> list[str]:
    exclude = {
        "stay_id",
        "subject_id",
        "alarm_start_idx",
        "alarm_start_time",
        "y_alarm_valid",
    }
    return [c for c in alarm_df.columns if c not in exclude]
