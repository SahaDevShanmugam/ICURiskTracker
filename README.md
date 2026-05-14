# Hemodynamic deterioration risk (MIMIC-IV)

Research prototype: **calibrated probability** of short-horizon hemodynamic deterioration from vitals, mapped to a **0–100 risk score** (`round(100 × P)`). Two deterioration estimators are trained for comparison (**logistic regression** as primary, **XGBoost** as secondary); both are isotonically calibrated unless `models.calibration` is changed.

**Alarm validity** and fusion decision UI are **deferred**; the FastAPI app focuses on dynamic risk and contributors only.

## Training data

- **Source:** credentialed **MIMIC-IV v3.1** (or compatible) CSV.GZ under `mimic-iv-3.1/` at the repo root (`icu/`, `hosp/`). `mimic.data_root` in [`config/config.yaml`](config/config.yaml) points there by default.
- **Override:** `python scripts/build_mimic_parquet.py --mimic-root /path/to/mimic-iv-3.1` or edit `mimic.data_root`. For large extracts, tune `chartevents_chunksize` / `labevents_chunksize` and consider `cohort.min_icu_length_hours` (e.g. 12) for stricter cohorts.
- **Subset:** `mimic.max_stays` (e.g. `10000`) keeps a **random** sample of eligible ICU stays after cohort filters, using `data.random_seed` for reproducibility. Set to `null` to use all eligible stays. Note: `chartevents` is still read in full; subsetting mainly shrinks later per-stay work and output size.

### Run pipeline

```bash
cd /Users/saha/Documents/mlnetworkproject
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_mimic_parquet.py
python scripts/train.py
python scripts/evaluate.py
uvicorn app.fastapi_app:app --reload
```

Outputs: `data/processed/*.parquet`, `artifacts/models/deterioration_*.joblib`, `artifacts/models/metadata.json`, `artifacts/figures/` (per-model ROC, primary calibration curve, `metrics.json`).

### Deterioration label (training target)

At each grid time \(t\), **positive** if within the next `labels.deterioration_horizon_hours` (excluding \(t\)) **either** a **vasopressor infusion start** (`vaso_marker` from `icu/inputevents`) **or** a run of MAP/SBP hypotension lasting at least `labels.persistent_hypotension_minutes`. See [`src/labels.py`](src/labels.py).

### FastAPI UI

Simulated vitals only (`src/sim_patients.py`); models are fit on real MIMIC parquets. The UI shows **risk score / 100**, **tier** (from `risk_display.tiers` in config), **risk trend** (SVG), vitals-based explanations, and **risk contributors** (logistic linear terms when the primary model is logistic, plus permutation-based drivers). Use the **Risk model** dropdown to compare logistic vs XGBoost on the same scenario.

Offline metrics are **retrospective**; they do not establish clinical safety or performance.

### Other ICU databases

Aligned time-series parquets per [`src/io_schema.py`](src/io_schema.py); MIMIC CSV ingest is [`src/mimic_ingest.py`](src/mimic_ingest.py).

## Project layout

- `config/config.yaml` — cohort, itemids, `deterioration_models`, `risk_display`, labels
- `src/mimic_ingest.py` — MIMIC CSV → grid + `vaso_marker`
- `scripts/build_mimic_parquet.py` — build `data/processed/*.parquet`
- `scripts/train.py` — train deterioration models only
- `scripts/evaluate.py` — per-model test metrics and figures
- `app/fastapi_app.py` — risk UI
- `docs/REPORT.md` — design notes

## License

Repository code: education/research use; **not** for clinical deployment. MIMIC-IV use is governed by PhysioNet.
