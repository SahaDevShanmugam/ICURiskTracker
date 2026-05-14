"""Alarm episodes, validity, and deterioration labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _minutes_to_steps(minutes: int, grid_minutes: int) -> int:
    return max(1, int(np.ceil(minutes / float(grid_minutes))))


def hypotension_mask(
    df: pd.DataFrame, map_thr: float, sbp_thr: float
) -> np.ndarray:
    return ((df["map"] < map_thr) | (df["sbp"] < sbp_thr)).to_numpy()


def longest_run_length(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    m = mask.astype(np.int32)
    if m.max() == 0:
        return 0
    padded = np.concatenate([[0], m, [0]])
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


def max_sustained_run_in_slice(mask: np.ndarray) -> int:
    return longest_run_length(mask)


def add_deterioration_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Binary label: within the next deterioration_horizon_hours (after t), vasopressor
    start (vaso_marker) and/or sustained MAP/SBP hypotension for persistent_hypotension_minutes.
    """
    grid = int(cfg["data"]["grid_minutes"])
    h = cfg["labels"]["deterioration_horizon_hours"]
    horizon_steps = _minutes_to_steps(int(h * 60), grid)
    persist_steps = _minutes_to_steps(cfg["labels"]["persistent_hypotension_minutes"], grid)
    map_thr = cfg["alarms"]["map_threshold_mmhg"]
    sbp_thr = cfg["alarms"]["sbp_threshold_mmhg"]
    has_vaso = "vaso_marker" in df.columns

    out = df.copy()
    y = []
    for _, g in out.groupby("stay_id", sort=False):
        n = len(g)
        hypo = hypotension_mask(g, map_thr, sbp_thr)
        vaso = g["vaso_marker"].to_numpy() if has_vaso else np.zeros(n)
        row_labels = []
        for t in range(n):
            end = min(n, t + 1 + horizon_steps)
            future_hypo = hypo[t + 1 : end]
            interv = vaso[t + 1 : end].sum() > 0 if has_vaso else False
            persist = max_sustained_run_in_slice(future_hypo) >= persist_steps
            row_labels.append(1 if (interv or persist) else 0)
        y.extend(row_labels)
    out["y_deterioration"] = y
    return out


def _merged_hypotension_segments(
    hypo: np.ndarray, merge_gap: int
) -> list[tuple[int, int]]:
    n = len(hypo)
    segments: list[list[int]] = []
    i = 0
    while i < n:
        if not hypo[i]:
            i += 1
            continue
        s = i
        while i < n and hypo[i]:
            i += 1
        e = i - 1
        if not segments:
            segments.append([s, e])
        else:
            ps, pe = segments[-1]
            gap = s - pe - 1
            if gap <= merge_gap:
                segments[-1][1] = e
            else:
                segments.append([s, e])
    return [(a[0], a[1]) for a in segments]


def extract_alarm_episodes(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per alarm episode with onset index and validity label."""
    grid = int(cfg["data"]["grid_minutes"])
    merge_gap = _minutes_to_steps(cfg["alarms"]["merge_gap_minutes"], grid)
    sustained_steps = _minutes_to_steps(cfg["labels"]["sustained_hypotension_minutes"], grid)
    look_steps = _minutes_to_steps(cfg["labels"]["intervention_lookahead_minutes"], grid)
    map_thr = cfg["alarms"]["map_threshold_mmhg"]
    sbp_thr = cfg["alarms"]["sbp_threshold_mmhg"]
    has_vaso = "vaso_marker" in df.columns

    rows = []
    for stay_id, g in df.groupby("stay_id", sort=False):
        hypo = hypotension_mask(g, map_thr, sbp_thr)
        vaso = g["vaso_marker"].to_numpy() if has_vaso else np.zeros(len(g))
        charttime = g["charttime"].to_numpy()
        for start, end in _merged_hypotension_segments(hypo, merge_gap):
            episode_len = end - start + 1
            sustained = episode_len >= sustained_steps
            interv = False
            if has_vaso:
                lb = min(len(vaso), start + 1 + look_steps)
                interv = vaso[start + 1 : lb].sum() > 0
            valid = bool(sustained or interv)
            rows.append(
                {
                    "stay_id": stay_id,
                    "subject_id": int(g["subject_id"].iloc[0]),
                    "alarm_start_idx": start,
                    "alarm_end_idx": end,
                    "alarm_start_time": charttime[start],
                    "y_alarm_valid": int(valid),
                    "episode_length_steps": episode_len,
                }
            )
    return pd.DataFrame(rows)


def attach_labels_to_timeseries(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = add_deterioration_labels(df, cfg)
    return df
