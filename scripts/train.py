"""Train deterioration risk models: primary (product) + optional benchmarks, all optionally calibrated."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.inspection import permutation_importance

from src.cohort import load_split_parquets
from src.config_loader import load_config
from src.features import compute_timeseries_features, feature_columns_deterioration
from src.models import (
    fit_calibrated,
    imbalance_scale_pos_weight,
    make_pipeline,
    save_model,
)


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def _logistic_coef_first_fold(calibrated: object) -> list[float] | None:
    """Coefficients from first CV base estimator (standardized feature space)."""
    cals = getattr(calibrated, "calibrated_classifiers_", None)
    if not cals:
        return None
    est = getattr(cals[0], "estimator", None)
    if est is None or not hasattr(est, "named_steps"):
        return None
    clf = est.named_steps.get("clf")
    if clf is None or not hasattr(clf, "coef_"):
        return None
    return np.asarray(clf.coef_, dtype=np.float64).ravel().tolist()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    t_all = time.perf_counter()
    cfg = load_config(root / "config" / "config.yaml")
    parquet_dir = root / cfg["data"]["parquet_dir"]
    train_pq = parquet_dir / "train_ts.parquet"
    if not train_pq.is_file():
        raise SystemExit(
            f"Missing {train_pq}. Build MIMIC-IV parquets first:\n"
            "  python scripts/build_mimic_parquet.py"
        )
    art = root / "artifacts" / "models"
    art.mkdir(parents=True, exist_ok=True)

    _log(f"loading {train_pq} …")
    t0 = time.perf_counter()
    train_ts, _, _ = load_split_parquets(parquet_dir)
    _log(f"loaded train parquet: {len(train_ts):,} grid rows in {time.perf_counter() - t0:.1f}s")

    rs = int(cfg["data"]["random_seed"])
    mcfg = cfg["models"]
    cal_method = mcfg.get("calibration", "isotonic")
    spec = mcfg.get("deterioration_models")
    if not spec:
        raise SystemExit("config models.deterioration_models is required")
    primary_id = str(mcfg.get("deterioration_primary", spec[0]["id"]))

    _log(
        "computing timeseries features (rolling windows + slopes; can take several minutes) …"
    )
    t1 = time.perf_counter()
    tr_fe = compute_timeseries_features(train_ts, cfg)
    _log(f"features done in {time.perf_counter() - t1:.1f}s — {len(tr_fe):,} rows")

    cols_d = feature_columns_deterioration(tr_fe)
    X_tr = tr_fe[cols_d].to_numpy(dtype=np.float64)
    y_tr = tr_fe["y_deterioration"].to_numpy()
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    _log(f"matrix shape {X_tr.shape}; labels positive={pos:,} negative={neg:,} ({100 * pos / len(y_tr):.2f}% positive)")

    xgb_spw = imbalance_scale_pos_weight(y_tr)
    _log(f"XGBoost scale_pos_weight (neg/pos) ≈ {xgb_spw:.4f}")

    trained: dict[str, object] = {}
    model_rows: list[dict] = []

    for i, entry in enumerate(spec):
        mid = str(entry["id"])
        mtype = str(entry["type"])
        rs_m = rs + i * 17
        spw = xgb_spw if mtype.lower() == "xgboost" else None
        tag = " (primary / product default)" if mid == primary_id else " (benchmark / comparison)"
        _log(
            f"model {i + 1}/{len(spec)}: fitting + calibrating '{mid}'{tag} ({mtype}, {cal_method}) …"
        )
        t_m = time.perf_counter()
        pipe = make_pipeline(mtype, rs_m, scale_pos_weight=spw)
        model = fit_calibrated(pipe, X_tr, y_tr, cal_method, rs_m)
        fname = f"deterioration_{mid}.joblib"
        save_model(model, art / fname)
        _log(f"saved {fname} in {time.perf_counter() - t_m:.1f}s")
        trained[mid] = model
        model_rows.append(
            {
                "id": mid,
                "type": mtype,
                "artifact": fname,
                "calibration": cal_method,
            }
        )

    if primary_id not in trained:
        raise SystemExit(
            f"deterioration_primary={primary_id!r} not found in deterioration_models"
        )
    det_primary = trained[primary_id]
    primary_type = next(r["type"] for r in model_rows if r["id"] == primary_id)

    sub_n = min(2500, len(X_tr))
    _log(
        f"permutation importance on primary '{primary_id}' (n={sub_n:,} subsampled rows, 4 repeats) …"
    )
    t_p = time.perf_counter()
    ridx = np.random.default_rng(rs).choice(len(X_tr), size=sub_n, replace=False)
    perm = permutation_importance(
        det_primary,
        X_tr[ridx],
        y_tr[ridx],
        n_repeats=4,
        random_state=rs,
        n_jobs=1,
    )
    _log(f"permutation importance done in {time.perf_counter() - t_p:.1f}s")
    imp = np.maximum(perm.importances_mean, 0.0) + 1e-8
    imp = imp / imp.sum()

    logistic_coef = None
    if str(primary_type).lower() == "logistic":
        logistic_coef = _logistic_coef_first_fold(det_primary)

    meta = {
        "features_deterioration": cols_d,
        "deterioration_primary": primary_id,
        "deterioration_primary_type": primary_type,
        "deterioration_models": model_rows,
        "n_train_ts": len(train_ts),
        "n_train_grid_rows": int(len(X_tr)),
        "deterioration_feature_mean": X_tr.mean(axis=0).tolist(),
        "deterioration_feature_std": np.maximum(X_tr.std(axis=0), 1e-6).tolist(),
        "deterioration_feature_importance": imp.tolist(),
        "deterioration_logistic_coefficients": logistic_coef,
    }
    _log(f"writing {art / 'metadata.json'} …")
    with open(art / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _log(
        f"finished in {time.perf_counter() - t_all:.1f}s — "
        f"{len(model_rows)} model(s) in {art}"
    )


if __name__ == "__main__":
    main()
