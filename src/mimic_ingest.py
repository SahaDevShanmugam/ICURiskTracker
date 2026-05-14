"""
Build time-series tables from locally downloaded MIMIC-IV CSV.GZ files.

Obtain MIMIC-IV CSV.GZ files locally (full database: PhysioNet credentialing + DUA;
Clinical Database Demo: open PhysioNet subset). This module does not download
data. Expected layout::

    {mimic.data_root}/
      icu/icustays.csv.gz
      icu/chartevents.csv.gz
      icu/inputevents.csv.gz
      hosp/patients.csv.gz
      hosp/labevents.csv.gz   (optional; lactate)

See https://mimic.mit.edu/docs/ for table definitions and itemid validation.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_schema import LABEL_HELPER_COLUMNS, TS_COLUMNS


def _log(msg: str) -> None:
    """Line-buffered progress for long-running MIMIC reads (use `python -u` if output lags)."""
    print(f"[mimic_parquet] {msg}", flush=True)


def _mimic_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "icustays": root / "icu" / "icustays.csv.gz",
        "chartevents": root / "icu" / "chartevents.csv.gz",
        "inputevents": root / "icu" / "inputevents.csv.gz",
        "patients": root / "hosp" / "patients.csv.gz",
        "labevents": root / "hosp" / "labevents.csv.gz",
    }


def _load_icustays(paths: dict[str, Path], cfg: dict) -> pd.DataFrame:
    p = paths["icustays"]
    if not p.is_file():
        raise FileNotFoundError(f"Missing MIMIC icustays at {p}")
    usecols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
    ]
    stays = pd.read_csv(p, usecols=usecols, parse_dates=["intime", "outtime"])
    stays["los_hours"] = (
        stays["outtime"] - stays["intime"]
    ).dt.total_seconds() / 3600.0
    min_h = float(cfg["cohort"]["min_icu_length_hours"])
    stays = stays[stays["los_hours"] >= min_h].copy()
    if cfg["cohort"].get("first_icu_stay_only", True):
        stays = stays.sort_values(["subject_id", "intime"])
        stays = stays.groupby("subject_id", as_index=False).head(1)
    max_stays = cfg["mimic"].get("max_stays")
    if max_stays is not None:
        n = int(max_stays)
        if len(stays) > n:
            rs = int(cfg.get("data", {}).get("random_seed", 0))
            stays = (
                stays.sample(n=n, random_state=rs)
                .sort_values("stay_id")
                .reset_index(drop=True)
            )
            _log(f"icustays: sampled {len(stays):,} stays (max_stays={n}, random_state={rs})")
    _log(f"icustays: {len(stays):,} ICU stays after cohort filters")
    return stays.reset_index(drop=True)


def _load_patients(paths: dict[str, Path]) -> pd.DataFrame:
    p = paths["patients"]
    if not p.is_file():
        raise FileNotFoundError(f"Missing MIMIC patients at {p}")
    pt = pd.read_csv(
        p,
        usecols=["subject_id", "gender", "anchor_age"],
    )
    pt["sex"] = (pt["gender"] == "M").astype(np.int8)
    pt = pt.rename(columns={"anchor_age": "age"})
    return pt[["subject_id", "age", "sex"]]


def _itemid_to_vital(cfg: dict) -> dict[int, str]:
    m = cfg["mimic"]
    out: dict[int, str] = {}
    for i in m["map_itemids"]:
        out[int(i)] = "map"
    for i in m["sbp_itemids"]:
        out[int(i)] = "sbp"
    for i in m["hr_itemids"]:
        out[int(i)] = "hr"
    for i in m["rr_itemids"]:
        out[int(i)] = "rr"
    return out


def _iter_chartevents_filtered(
    path: Path,
    stay_ids: set[int],
    vital_ids: set[int],
    chunksize: int,
    *,
    log_every_raw_chunks: int = 40,
) -> Iterator[pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MIMIC chartevents at {path}")
    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    raw_i = 0
    kept_total = 0
    t0 = time.perf_counter()
    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )
    for raw_chunk in reader:
        raw_i += 1
        chunk = raw_chunk[raw_chunk["stay_id"].isin(stay_ids)]
        chunk = chunk[chunk["itemid"].isin(vital_ids)]
        chunk = chunk.dropna(subset=["valuenum"])
        if raw_i == 1 or (raw_i % log_every_raw_chunks == 0):
            elapsed = time.perf_counter() - t0
            _log(
                f"chartevents: raw chunk {raw_i} (~{chunksize:,} rows/chunk read from disk), "
                f"kept {kept_total:,} vital rows so far, {elapsed:.1f}s elapsed"
            )
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        kept_total += len(chunk)
        yield chunk


def _collect_chartevents_long(paths: dict[str, Path], cfg: dict, stays: pd.DataFrame) -> pd.DataFrame:
    id_map = _itemid_to_vital(cfg)
    vital_ids = set(id_map.keys())
    stay_ids = set(stays["stay_id"].astype(int).tolist())
    chunksize = int(cfg["mimic"].get("chartevents_chunksize", 500_000))
    _log(
        f"chartevents: scanning {paths['chartevents']} (this is usually the slowest step); "
        f"chunksize={chunksize:,}, {len(stay_ids):,} stay_ids, {len(vital_ids)} vital itemids"
    )
    t0 = time.perf_counter()
    parts: list[pd.DataFrame] = []
    for ch in _iter_chartevents_filtered(
        paths["chartevents"], stay_ids, vital_ids, chunksize
    ):
        ch = ch[ch["itemid"].isin(vital_ids)]
        ch["vital"] = ch["itemid"].map(id_map)
        parts.append(ch[["stay_id", "charttime", "vital", "valuenum"]])
    elapsed = time.perf_counter() - t0
    if not parts:
        _log(f"chartevents: done in {elapsed:.1f}s (no matching rows)")
        return pd.DataFrame(columns=["stay_id", "charttime", "vital", "valuenum"])
    out = pd.concat(parts, ignore_index=True)
    _log(f"chartevents: done in {elapsed:.1f}s; concatenated {len(out):,} vital rows")
    return out


def _floor_to_grid(ts: pd.Series, grid_min: int) -> pd.Series:
    # Anchor to grid buckets (floor minute, then snap to grid)
    t = pd.to_datetime(ts)
    return t.dt.floor(f"{int(grid_min)}min")


def _pivot_vitals_to_stay_grid(
    stay_row: pd.Series,
    ce_long: pd.DataFrame,
    grid_min: int,
    min_rows: int,
) -> pd.DataFrame | None:
    sid = int(stay_row["stay_id"])
    intime = pd.to_datetime(stay_row["intime"])
    outtime = pd.to_datetime(stay_row["outtime"])
    sub = ce_long[ce_long["stay_id"] == sid].copy()
    if len(sub) < min_rows:
        return None
    sub["charttime"] = _floor_to_grid(sub["charttime"], grid_min)
    agg = sub.groupby(["charttime", "vital"], as_index=False)["valuenum"].mean()
    wide = agg.pivot(index="charttime", columns="vital", values="valuenum")
    wide = wide.sort_index()
    idx = pd.date_range(
        intime.floor(f"{grid_min}min"),
        outtime.ceil(f"{grid_min}min"),
        freq=f"{grid_min}min",
    )
    wide = wide.reindex(idx)
    wide.index.name = "charttime"
    wide = wide.reset_index()
    wide["stay_id"] = sid
    for col in ["map", "sbp", "hr", "rr"]:
        if col not in wide.columns:
            wide[col] = np.nan
    return wide


def _load_inputevents_vaso(paths: dict[str, Path], cfg: dict, stay_ids: set[int]) -> pd.DataFrame:
    p = paths["inputevents"]
    if not p.is_file():
        raise FileNotFoundError(f"Missing MIMIC inputevents at {p}")
    vaso_ids = set(int(x) for x in cfg["mimic"]["vasopressor_itemids"])
    usecols = ["stay_id", "starttime", "itemid"]
    parts: list[pd.DataFrame] = []
    chunksize = int(cfg["mimic"].get("chartevents_chunksize", 500_000))
    _log(f"inputevents: scanning {p} (vasopressor rows only)")
    t0 = time.perf_counter()
    raw_i = 0
    kept = 0
    for chunk in pd.read_csv(p, usecols=usecols, chunksize=chunksize, low_memory=False):
        raw_i += 1
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        chunk = chunk[chunk["itemid"].isin(vaso_ids)]
        if raw_i == 1 or raw_i % 40 == 0:
            _log(
                f"inputevents: raw chunk {raw_i}, {kept:,} vaso rows kept, "
                f"{time.perf_counter() - t0:.1f}s elapsed"
            )
        if chunk.empty:
            continue
        chunk["starttime"] = pd.to_datetime(chunk["starttime"])
        kept += len(chunk)
        parts.append(chunk)
    elapsed = time.perf_counter() - t0
    if not parts:
        _log(f"inputevents: done in {elapsed:.1f}s (no vaso rows)")
        return pd.DataFrame(columns=["stay_id", "starttime", "itemid"])
    out = pd.concat(parts, ignore_index=True)
    _log(f"inputevents: done in {elapsed:.1f}s; {len(out):,} vaso rows")
    return out


def _load_lactate_labs(paths: dict[str, Path], cfg: dict, stays: pd.DataFrame) -> pd.DataFrame:
    p = paths["labevents"]
    if not p.is_file():
        _log("labevents: file not present; skipping lactate")
        return pd.DataFrame(columns=["hadm_id", "charttime", "valuenum"])
    lac_ids = set(int(x) for x in cfg["mimic"]["lactate_itemids"])
    hadm_ids = set(stays["hadm_id"].astype(int).tolist())
    usecols = ["hadm_id", "charttime", "itemid", "valuenum"]
    parts: list[pd.DataFrame] = []
    chunksize = int(cfg["mimic"].get("labevents_chunksize", 500_000))
    _log(f"labevents: scanning {p} (lactate itemids only)")
    t0 = time.perf_counter()
    raw_i = 0
    kept = 0
    for chunk in pd.read_csv(p, usecols=usecols, chunksize=chunksize, low_memory=False):
        raw_i += 1
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        chunk = chunk[chunk["itemid"].isin(lac_ids)]
        chunk = chunk.dropna(subset=["valuenum"])
        if raw_i == 1 or raw_i % 40 == 0:
            _log(
                f"labevents: raw chunk {raw_i}, {kept:,} lactate rows kept, "
                f"{time.perf_counter() - t0:.1f}s elapsed"
            )
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        kept += len(chunk)
        parts.append(chunk[["hadm_id", "charttime", "valuenum"]])
    elapsed = time.perf_counter() - t0
    if not parts:
        _log(f"labevents: done in {elapsed:.1f}s (no lactate rows)")
        return pd.DataFrame(columns=["hadm_id", "charttime", "valuenum"])
    labs = pd.concat(parts, ignore_index=True)
    labs = labs.groupby(["hadm_id", "charttime"], as_index=False)["valuenum"].mean()
    _log(f"labevents: done in {elapsed:.1f}s; {len(labs):,} aggregated lactate rows")
    return labs


def _attach_lactate_to_stay(
    grid: pd.DataFrame,
    stay_row: pd.Series,
    labs: pd.DataFrame,
    grid_min: int,
) -> pd.DataFrame:
    grid = grid.copy()
    if labs.empty:
        grid["lactate"] = np.nan
        return grid
    hadm = int(stay_row["hadm_id"])
    intime = pd.to_datetime(stay_row["intime"])
    outtime = pd.to_datetime(stay_row["outtime"])
    sub = labs[labs["hadm_id"] == hadm]
    sub = sub[(sub["charttime"] >= intime) & (sub["charttime"] <= outtime)]
    if sub.empty:
        grid["lactate"] = np.nan
        return grid
    sub = sub.copy()
    sub["charttime"] = _floor_to_grid(sub["charttime"], grid_min)
    sub = sub.groupby("charttime", as_index=False)["valuenum"].mean().rename(
        columns={"valuenum": "lactate"}
    )
    return grid.merge(sub, on="charttime", how="left")


def _add_vaso_markers(
    grid: pd.DataFrame,
    stay_row: pd.Series,
    vaso: pd.DataFrame,
    grid_min: int,
) -> pd.DataFrame:
    sid = int(stay_row["stay_id"])
    intime = pd.to_datetime(stay_row["intime"])
    vm = np.zeros(len(grid), dtype=np.int8)
    sub = vaso[vaso["stay_id"] == sid]
    step_sec = float(grid_min) * 60.0
    for _, r in sub.iterrows():
        t0 = pd.to_datetime(r["starttime"])
        # mark the grid bucket containing infusion start
        rel = int(np.floor((t0 - intime).total_seconds() / step_sec))
        if 0 <= rel < len(vm):
            vm[rel] = 1
    grid = grid.copy()
    grid["vaso_marker"] = vm
    return grid


def build_mimic_timeseries(cfg: dict) -> pd.DataFrame:
    root = cfg["mimic"].get("data_root") or ""
    if not str(root).strip():
        raise ValueError(
            "mimic.data_root must be set in config to your local MIMIC-IV directory "
            "(folder that contains icu/ and hosp/)."
        )
    paths = _mimic_paths(Path(root))
    grid_min = int(cfg["data"]["grid_minutes"])
    min_rows = int(cfg["mimic"].get("min_vital_rows_per_stay", 5))

    _log(f"starting build_mimic_timeseries from {root}")
    stays = _load_icustays(paths, cfg)
    _log("loading patients.csv.gz …")
    patients = _load_patients(paths)
    stays = stays.merge(patients, on="subject_id", how="left")
    stays["age"] = stays["age"].fillna(stays["age"].median())
    stays["sex"] = stays["sex"].fillna(0).astype(np.int8)

    ce_long = _collect_chartevents_long(paths, cfg, stays)
    vaso = _load_inputevents_vaso(paths, cfg, set(stays["stay_id"].astype(int)))
    labs = _load_lactate_labs(paths, cfg, stays)

    n_stays = len(stays)
    log_every = max(1, min(500, n_stays // 20))
    _log(
        f"building per-stay grids: {n_stays:,} stays, grid={grid_min}min, "
        f"progress every {log_every} stays"
    )
    t_stays = time.perf_counter()
    frames: list[pd.DataFrame] = []
    for si, (_, stay_row) in enumerate(stays.iterrows()):
        if si > 0 and si % log_every == 0:
            _log(
                f"per-stay grids: {si:,}/{n_stays:,} ({100 * si / n_stays:.0f}%), "
                f"{len(frames):,} stays with data, {time.perf_counter() - t_stays:.1f}s"
            )
        g = _pivot_vitals_to_stay_grid(stay_row, ce_long, grid_min, min_rows)
        if g is None:
            continue
        g = _attach_lactate_to_stay(g, stay_row, labs, grid_min)
        g = _add_vaso_markers(g, stay_row, vaso, grid_min)
        g["subject_id"] = int(stay_row["subject_id"])
        frames.append(g)
    _log(
        f"per-stay grids: finished {n_stays:,} stays in {time.perf_counter() - t_stays:.1f}s "
        f"({len(frames):,} stays with sufficient vitals)"
    )

    if not frames:
        raise RuntimeError(
            "No ICU stays with sufficient chartevents after filters. "
            "Loosen cohort limits or verify itemids / paths."
        )
    _log("concatenating stay frames …")
    df = pd.concat(frames, ignore_index=True)
    stay_profile = stays.set_index("stay_id")[["age", "sex"]]
    df = df.merge(stay_profile, left_on="stay_id", right_index=True, how="left")
    for c in ["map", "sbp", "hr", "rr", "lactate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cols = TS_COLUMNS + LABEL_HELPER_COLUMNS
    for c in cols:
        if c not in df.columns:
            if c == "lactate":
                df[c] = np.nan
            else:
                raise KeyError(f"Missing column {c}")
    _log(f"build_mimic_timeseries complete: {len(df):,} grid rows")
    return df[cols].sort_values(["stay_id", "charttime"]).reset_index(drop=True)
