# Hemodynamic deterioration risk — Technical report

## Problem

ICU monitors generate frequent hypotension-related signals. This prototype focuses on **short-horizon hemodynamic deterioration risk**: a calibrated estimate of whether meaningful circulatory worsening (vasopressor need and/or sustained hypotension) will occur within a configurable lookahead window, presented as a **0–100 score** for monitoring-style interpretation.

## System design

### Data

- **Training path:** MIMIC-IV ICU stays from a local **v3.1** (or compatible) PhysioNet CSV extract at `mimic-iv-3.1/` (`src/mimic_ingest.py` → `scripts/build_mimic_parquet.py`). Vitals are aligned to a fixed grid (`config/config.yaml`), forward-filled within policy, and split by **subject** to limit leakage.

### Deterioration label

- At each grid time \(t\), look forward over `labels.deterioration_horizon_hours` (from \(t+1\) onward). **Positive** if **any** of:
  - **Vasopressor initiation** in that window: `vaso_marker` flags infusion **start** times mapped to the grid (`icu/inputevents`, configured itemids).
  - **Sustained hypotension** in that window: longest contiguous span of (MAP &lt; threshold or SBP &lt; threshold) at least `labels.persistent_hypotension_minutes` (thresholds from `alarms.*`).

`vaso_marker` is **label-only** and must not appear in feature matrices.

### Models

- **Primary (default):** `LogisticRegression` in a pipeline with `StandardScaler`, wrapped in `CalibratedClassifierCV` (isotonic or Platt per `models.calibration`).
- **Secondary:** `XGBClassifier` with `scale_pos_weight` from class imbalance, same calibration wrapper.
- **Risk score:** `round(100 × predict_proba_positive)` at each grid step; **tiers** from `risk_display.tiers` in config (e.g. low / moderate / high / critical by score bands).

Alarm validity and a fusion decision engine are **out of scope** for the current training and UI paths (`src/decision.py` retained only for possible future fusion).

### Evaluation

Per-model **ROC-AUC**, **PR-AUC**, **Brier score**, and confusion-based precision/recall/F1 at threshold 0.5 on the held-out split; ROC and (for the primary model) a **reliability** figure are written under `artifacts/figures/`. Metrics are retrospective.

## Limitations

- Simulated trajectories in the UI are for engineering demonstration only.
- Labels are **proxies** (medications, hypotension persistence), not adjudicated deterioration.
- Single-center bias if using MIMIC alone; no prospective deployment or human-factors evaluation.

## Reproducibility

From the repository root: `scripts/build_mimic_parquet.py`, `scripts/train.py`, `scripts/evaluate.py` (see README). Figures and `metrics.json` are written to `artifacts/figures/`.

## Regulatory posture

Research prototype only — **not** a medical device and not for use in patient care.
