"""
Build train/val/test parquets from locally downloaded MIMIC-IV files.

Requires a local MIMIC-IV folder (CSV.GZ under `icu/` and `hosp/`).
Set `mimic.data_root` in config to that folder.

If `mimic.max_stays` is set, after cohort filters a **random** subset of that
many stays is kept (reproducible via `data.random_seed` in config). Chartevents
is still scanned in full, but downstream work is smaller.

Example::

    export MIMIC_ROOT=/path/to/mimiciv/3.1
    python -u scripts/build_mimic_parquet.py --mimic-root "$MIMIC_ROOT"

``python -u`` runs unbuffered so ``[mimic_parquet]`` progress lines appear immediately.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cohort import run_cohort_from_raw_ts, save_split_parquets
from src.config_loader import load_config
from src.labels import attach_labels_to_timeseries
from src.mimic_ingest import build_mimic_timeseries


def main() -> None:
    parser = argparse.ArgumentParser(description="MIMIC-IV → project parquets")
    parser.add_argument(
        "--mimic-root",
        type=str,
        default=None,
        help="Path to MIMIC-IV root (contains icu/ and hosp/). Overrides config mimic.data_root.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: config/config.yaml).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config) if args.config else root / "config" / "config.yaml"
    cfg = load_config(cfg_path)
    if args.mimic_root:
        cfg = copy.deepcopy(cfg)
        cfg["mimic"]["data_root"] = args.mimic_root

    mr = Path(cfg["mimic"]["data_root"])
    if not mr.is_absolute():
        mr = (root / mr).resolve()
    cfg["mimic"]["data_root"] = str(mr)
    if not (mr / "icu" / "icustays.csv.gz").is_file():
        raise SystemExit(
            f"MIMIC root must contain icu/icustays.csv.gz; not found under {mr}"
        )

    out_dir = root / cfg["data"]["parquet_dir"]
    t0 = time.perf_counter()

    print("[mimic_parquet] step 1/4: build_mimic_timeseries (MIMIC CSV → grid)", flush=True)
    df = build_mimic_timeseries(cfg)
    print(f"[mimic_parquet] step 1/4 done in {time.perf_counter() - t0:.1f}s — {len(df):,} rows", flush=True)

    t1 = time.perf_counter()
    print("[mimic_parquet] step 2/4: run_cohort_from_raw_ts (forward-fill limits)", flush=True)
    df = run_cohort_from_raw_ts(df, cfg)
    print(f"[mimic_parquet] step 2/4 done in {time.perf_counter() - t1:.1f}s", flush=True)

    t2 = time.perf_counter()
    print("[mimic_parquet] step 3/4: attach_labels_to_timeseries", flush=True)
    df = attach_labels_to_timeseries(df, cfg)
    print(f"[mimic_parquet] step 3/4 done in {time.perf_counter() - t2:.1f}s", flush=True)

    t3 = time.perf_counter()
    print(f"[mimic_parquet] step 4/4: save_split_parquets → {out_dir}", flush=True)
    save_split_parquets(df, out_dir, cfg)
    print(f"[mimic_parquet] step 4/4 done in {time.perf_counter() - t3:.1f}s", flush=True)

    print(
        f"[mimic_parquet] finished in {time.perf_counter() - t0:.1f}s — "
        f"wrote train/val/test parquets to {out_dir} ({len(df):,} rows).",
        flush=True,
    )


if __name__ == "__main__":
    main()
