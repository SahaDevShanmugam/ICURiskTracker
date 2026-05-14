"""Evaluate deterioration risk models on the test split."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.cohort import load_split_parquets
from src.config_loader import load_config
from src.eval import (
    classification_report_binary,
    plot_calibration_curve,
    plot_roc,
)
from src.features import compute_timeseries_features
from src.models import load_model


def _log(msg: str) -> None:
    print(f"[evaluate] {msg}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    t_all = time.perf_counter()
    cfg = load_config(root / "config" / "config.yaml")
    parquet_dir = root / cfg["data"]["parquet_dir"]
    if not (parquet_dir / "test_ts.parquet").is_file():
        raise SystemExit(
            f"Missing {parquet_dir / 'test_ts.parquet'}. Run scripts/build_mimic_parquet.py then train."
        )
    fig_dir = root / "artifacts" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art = root / "artifacts" / "models"

    _log(f"reading {art / 'metadata.json'} …")
    with open(art / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    _log(f"loading test split from {parquet_dir} …")
    t0 = time.perf_counter()
    _, _, test_ts = load_split_parquets(parquet_dir)
    _log(f"loaded test_ts.parquet: {len(test_ts):,} rows in {time.perf_counter() - t0:.1f}s")

    _log("computing test timeseries features …")
    t1 = time.perf_counter()
    te_fe = compute_timeseries_features(test_ts, cfg).reset_index(drop=True)
    _log(f"features done in {time.perf_counter() - t1:.1f}s — {len(te_fe):,} rows")

    cols_d = meta["features_deterioration"]
    X_te = te_fe[cols_d].to_numpy(dtype=np.float64)
    y_det = te_fe["y_deterioration"].to_numpy()
    _log(f"prediction matrix {X_te.shape}; positives={int(y_det.sum()):,} / {len(y_det):,}")

    primary_id = meta.get("deterioration_primary", "logistic")
    model_entries = meta.get("deterioration_models")
    if not model_entries:
        raise SystemExit("metadata.json missing deterioration_models")

    summary: dict = {"deterioration": {}}
    for j, entry in enumerate(model_entries):
        mid = entry["id"]
        artifact = entry["artifact"]
        _log(f"model {j + 1}/{len(model_entries)}: {mid} — load {artifact} …")
        t_m = time.perf_counter()
        m = load_model(art / artifact)
        _log(f"predict_proba on test ({len(X_te):,} rows) …")
        p = np.asarray(m.predict_proba(X_te)[:, 1], dtype=np.float64)
        _log(f"inference done in {time.perf_counter() - t_m:.1f}s")

        rep = classification_report_binary(y_det, p, f"deterioration_{mid}")
        summary["deterioration"][mid] = rep
        _log(f"metrics {mid}: ROC-AUC={rep['roc_auc']:.4f} PR-AUC={rep['pr_auc']:.4f} Brier={rep['brier']:.4f}")

        t_f = time.perf_counter()
        plot_roc(
            y_det,
            p,
            f"Deterioration ROC ({mid})",
            fig_dir / f"roc_deterioration_{mid}.png",
        )
        _log(f"wrote ROC figure in {time.perf_counter() - t_f:.1f}s")
        if mid == primary_id:
            t_c = time.perf_counter()
            plot_calibration_curve(
                y_det,
                p,
                f"Calibration ({mid}, primary)",
                fig_dir / f"calibration_deterioration_{mid}.png",
            )
            _log(f"wrote calibration figure in {time.perf_counter() - t_c:.1f}s")

    _log(f"writing {fig_dir / 'metrics.json'} …")
    with open(fig_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _log(f"finished in {time.perf_counter() - t_all:.1f}s — outputs in {fig_dir}")


if __name__ == "__main__":
    main()
