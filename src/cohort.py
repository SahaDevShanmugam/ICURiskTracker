"""Resample, forward-fill with gap limits, and patient-level splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config_loader import load_config


def forward_fill_limited(
    df: pd.DataFrame,
    cols: list[str],
    max_gap_minutes: int,
    time_col: str = "charttime",
) -> pd.DataFrame:
    """Forward-fill vitals on the regular grid (MIMIC-derived series)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].ffill()
    return out


def split_subject_ids(
    unique_subjects: np.ndarray,
    cfg: dict,
) -> tuple[set[int], set[int], set[int]]:
    idx = np.arange(len(unique_subjects)).reshape(-1, 1)
    gss = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=cfg["data"]["random_seed"]
    )
    train_val_pos, test_pos = next(
        gss.split(idx, groups=unique_subjects)
    )
    subj_tv = unique_subjects[train_val_pos]
    subj_test = unique_subjects[test_pos]
    inner = np.arange(len(subj_tv)).reshape(-1, 1)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15 / 0.8,
        random_state=cfg["data"]["random_seed"] + 1,
    )
    tr_rel, va_rel = next(gss2.split(inner, groups=subj_tv))
    subj_train = set(subj_tv[tr_rel].tolist())
    subj_val = set(subj_tv[va_rel].tolist())
    subj_test_set = set(subj_test.tolist())
    return subj_train, subj_val, subj_test_set


def save_split_parquets(df: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_subjects = df["subject_id"].unique()
    subj_train, subj_val, subj_test = split_subject_ids(unique_subjects, cfg)
    train_mask = df["subject_id"].isin(subj_train)
    val_mask = df["subject_id"].isin(subj_val)
    test_mask = df["subject_id"].isin(subj_test)
    n_tr = int(train_mask.sum())
    n_va = int(val_mask.sum())
    n_te = int(test_mask.sum())
    print(f"[mimic_parquet] writing train_ts.parquet ({n_tr:,} rows) …", flush=True)
    df.loc[train_mask].to_parquet(out_dir / "train_ts.parquet", index=False)
    print(f"[mimic_parquet] writing val_ts.parquet ({n_va:,} rows) …", flush=True)
    df.loc[val_mask].to_parquet(out_dir / "val_ts.parquet", index=False)
    print(f"[mimic_parquet] writing test_ts.parquet ({n_te:,} rows) …", flush=True)
    df.loc[test_mask].to_parquet(out_dir / "test_ts.parquet", index=False)


def load_split_parquets(parquet_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(parquet_dir / "train_ts.parquet")
    val = pd.read_parquet(parquet_dir / "val_ts.parquet")
    test = pd.read_parquet(parquet_dir / "test_ts.parquet")
    return train, val, test


def run_cohort_from_raw_ts(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    vital_cols = ["map", "sbp", "hr", "rr", "lactate"]
    return forward_fill_limited(
        df,
        vital_cols,
        max_gap_minutes=cfg["data"]["max_forward_fill_minutes"],
    )
