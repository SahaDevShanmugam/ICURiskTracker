"""FastAPI UI for hemodynamic deterioration risk (0–100) demo.

Default scoring uses the training metadata primary model (logistic for interpretability);
other trained models (e.g. XGBoost) are available only for validation and comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.explanations import (
    clinical_vital_explanations,
    model_feature_explanations,
    risk_tier_label,
)
from src.features import compute_timeseries_features
from src.models import load_model
from src.sim_patients import ScenarioId, build_scenario, scenario_display_name

# Cycled by bed index so the ward visibly mixes low / moderate / high / critical risk profiles.
BED_SCENARIO_PATTERN: tuple[ScenarioId, ...] = (
    "demo_stable",
    "demo_stable",
    "demo_transient_hypotension",
    "demo_transient_hypotension",
    "demo_progressive_decline",
    "demo_shock_pattern",
)


def scenario_for_bed(bed_idx: int) -> ScenarioId:
    return BED_SCENARIO_PATTERN[bed_idx % len(BED_SCENARIO_PATTERN)]


# Cached patient timelines for API + ward JSON (key: risk_model_id, seed, bed_idx).
_BED_TIMELINE_CACHE: dict[tuple[str, int, int], dict] = {}
_BED_CACHE_MAX = 48

app = FastAPI(title="ICU Monitor")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def load_artifacts(root: Path):
    cfg = load_config(root / "config" / "config.yaml")
    meta_path = root / "artifacts" / "models" / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}; run scripts/train.py first.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    models_dir = root / "artifacts" / "models"
    entries = meta.get("deterioration_models") or []
    if not entries:
        raise FileNotFoundError("metadata.json missing deterioration_models")
    det_models: dict[str, object] = {}
    for entry in entries:
        det_models[str(entry["id"])] = load_model(models_dir / entry["artifact"])
    primary_id = str(meta.get("deterioration_primary", entries[0]["id"]))
    if primary_id not in det_models:
        primary_id = str(entries[0]["id"])
    return cfg, meta, det_models, primary_id


CFG = None
META = None
DET_MODELS: dict[str, object] | None = None
PRIMARY_MODEL_ID: str | None = None
ARTIFACT_LOAD_ERROR = None

try:
    CFG, META, DET_MODELS, PRIMARY_MODEL_ID = load_artifacts(ROOT)
except FileNotFoundError as exc:
    ARTIFACT_LOAD_ERROR = str(exc)


def _resolve_model_id(requested: str | None) -> str:
    assert DET_MODELS is not None and PRIMARY_MODEL_ID is not None
    if requested and requested in DET_MODELS:
        return requested
    return PRIMARY_MODEL_ID


def model_ids_primary_first(meta: dict, primary_id: str) -> list[str]:
    """List deterioration model ids with config/metadata primary first (product default)."""
    raw = [str(e["id"]) for e in (meta.get("deterioration_models") or [])]
    if primary_id in raw:
        return [primary_id] + [x for x in raw if x != primary_id]
    return raw


def _get_or_build_timeline(
    scenario: ScenarioId,
    patient_seed: int,
    risk_model: str | None,
) -> dict:
    """Build g, fe, probabilities, risk scores, and chart stream for one simulated stay."""
    assert CFG is not None and META is not None and DET_MODELS is not None
    mid = _resolve_model_id(risk_model)
    model = DET_MODELS[mid]

    if scenario == "random":
        rng = np.random.default_rng(int(CFG["data"]["random_seed"]) + int(patient_seed))
    else:
        rng = np.random.default_rng(0)

    g = build_scenario(CFG, scenario, rng).reset_index(drop=True)
    fe = compute_timeseries_features(g, CFG).reset_index(drop=True)

    cols_d = META["features_deterioration"]
    p_det = np.asarray(
        model.predict_proba(fe[cols_d].to_numpy(dtype=np.float64))[:, 1],
        dtype=np.float64,
    )
    risk_scores = np.clip(np.round(100.0 * p_det), 0, 100).astype(int)
    chart_records: list[dict] = []
    for i, row in enumerate(g[["charttime", "map", "sbp", "hr", "rr"]].itertuples(index=False)):
        rs = int(risk_scores[i]) if i < len(risk_scores) else int(risk_scores[-1])
        chart_records.append(
            {
                "charttime": str(row[0]),
                "map": float(row[1]),
                "sbp": float(row[2]),
                "hr": float(row[3]),
                "rr": float(row[4]),
                "risk_score": rs,
                "tier": risk_tier_label(float(rs), CFG),
            }
        )
    return {
        "scenario": scenario,
        "scenario_name": scenario_display_name(scenario),
        "stay_id": int(g["stay_id"].iloc[0]),
        "age": int(g["age"].iloc[0]),
        "sex": "M" if g["sex"].iloc[0] == 1 else "F",
        "risk_model_id": mid,
        "p_det": p_det,
        "risk_scores": risk_scores,
        "g": g,
        "fe": fe,
        "chart_records": chart_records,
    }


def get_cached_bed_timeline(seed: int, bed_idx: int, risk_model: str | None) -> dict:
    """Timeline for ward bed `bed_idx` with cohort `seed` (patient_seed = seed + bed_idx)."""
    assert CFG is not None and META is not None and DET_MODELS is not None
    mid = _resolve_model_id(risk_model)
    key = (mid, int(seed), int(bed_idx))
    if key not in _BED_TIMELINE_CACHE:
        if len(_BED_TIMELINE_CACHE) >= _BED_CACHE_MAX:
            _BED_TIMELINE_CACHE.clear()
        scenario = scenario_for_bed(bed_idx)
        patient_seed = int(seed) + int(bed_idx)
        _BED_TIMELINE_CACHE[key] = _get_or_build_timeline(scenario, patient_seed, risk_model)
    return _BED_TIMELINE_CACHE[key]


def _bundle_to_dashboard(bundle: dict) -> dict:
    """Full dashboard dict from an in-memory timeline bundle (uses last timestep)."""
    assert CFG is not None and META is not None
    g = bundle["g"]
    fe = bundle["fe"]
    p_det = bundle["p_det"]
    risk_scores = bundle["risk_scores"]
    mid = bundle["risk_model_id"]
    cols_d = META["features_deterioration"]

    risk_now_score = int(risk_scores[-1])
    p_now = float(p_det[-1])
    tier = risk_tier_label(float(risk_now_score), CFG)
    last = g.iloc[-1]

    clin_lines = clinical_vital_explanations(g, CFG)
    model_lines = model_feature_explanations(fe.iloc[-1], cols_d, META)

    horizon_h = float(CFG["labels"]["deterioration_horizon_hours"])
    risk_svg_points = ""
    n = len(risk_scores)
    if n > 0:
        w, svg_h = 280.0, 88.0
        for i, s in enumerate(risk_scores):
            x = (i / max(n - 1, 1)) * w
            y = svg_h - (float(s) / 100.0) * svg_h
            risk_svg_points += f"{x:.1f},{y:.1f} "
    return {
        "scenario": bundle["scenario"],
        "scenario_name": bundle["scenario_name"],
        "stay_id": bundle["stay_id"],
        "age": bundle["age"],
        "sex": bundle["sex"],
        "risk_model_id": mid,
        "risk_now_prob": p_now,
        "risk_now_score": risk_now_score,
        "risk_scores": risk_scores.tolist(),
        "tier": tier,
        "horizon_hours": horizon_h,
        "last_vitals": {
            "map": float(last["map"]),
            "hr": float(last["hr"]),
            "rr": float(last["rr"]),
        },
        "clinical_explanations": clin_lines,
        "model_explanations": model_lines,
        "chart_records": bundle["chart_records"],
        "risk_svg_points": risk_svg_points.strip(),
    }


def compute_dashboard_data(
    scenario: ScenarioId,
    seed: int,
    risk_model: str | None = None,
) -> dict:
    bundle = _get_or_build_timeline(scenario, seed, risk_model)
    return _bundle_to_dashboard(bundle)


def ward_bed_json(seed: int, bed_idx: int, risk_model: str | None) -> dict:
    """Serializable ward payload for one bed (no DataFrames)."""
    b = get_cached_bed_timeline(seed, bed_idx, risk_model)
    return {
        "bed_idx": bed_idx,
        "bed_label": f"Bed {bed_idx + 1:02d}",
        "patient_label": f"P-{1000 + bed_idx}",
        "scenario": b["scenario"],
        "scenario_name": b["scenario_name"],
        "stay_id": b["stay_id"],
        "age": b["age"],
        "sex": b["sex"],
        "risk_model_id": b["risk_model_id"],
        "stream": b["chart_records"],
    }


def build_ward_overview(seed: int, bed_count: int, risk_model: str | None) -> list[dict]:
    beds: list[dict] = []
    for bed_idx in range(max(1, bed_count)):
        bundle = get_cached_bed_timeline(seed, bed_idx, risk_model)
        details = _bundle_to_dashboard(bundle)
        beds.append(
            {
                "bed_idx": bed_idx,
                "bed_label": f"Bed {bed_idx + 1:02d}",
                "patient_label": f"P-{1000 + bed_idx}",
                "scenario_name": details["scenario_name"],
                "risk_now_score": details["risk_now_score"],
                "tier": details["tier"],
                "stay_id": details["stay_id"],
                "details": details,
            }
        )
    return beds


def build_ward_streams(seed: int, bed_count: int, risk_model: str | None) -> list[dict]:
    n = max(1, bed_count)
    return [ward_bed_json(seed, i, risk_model) for i in range(n)]


@app.get("/api/bed-frame")
def api_bed_frame(
    seed: int = Query(default=0, ge=0),
    bed: int = Query(default=0, ge=0, le=35),
    step: int = Query(default=0, ge=0),
    risk_model: str | None = Query(default=None),
):
    if ARTIFACT_LOAD_ERROR:
        return JSONResponse({"error": ARTIFACT_LOAD_ERROR}, status_code=500)
    assert CFG is not None and META is not None and PRIMARY_MODEL_ID is not None
    model_options = model_ids_primary_first(META, PRIMARY_MODEL_ID)
    rm = risk_model if risk_model in model_options else PRIMARY_MODEL_ID
    bundle = get_cached_bed_timeline(seed, bed, rm)
    g = bundle["g"]
    fe = bundle["fe"]
    cols_d = META["features_deterioration"]
    n = len(g)
    if n == 0:
        return JSONResponse({"error": "empty timeline"}, status_code=400)
    st = max(0, min(int(step), n - 1))
    clin = clinical_vital_explanations(g.iloc[: st + 1], CFG)
    model_ex = model_feature_explanations(fe.iloc[st], cols_d, META)
    row = bundle["chart_records"][st]
    return JSONResponse(
        {
            "step": st,
            "n": n,
            "vitals": row,
            "risk_prob": float(bundle["p_det"][st]),
            "clinical": clin,
            "model": model_ex,
        }
    )


@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    seed: int = Query(default=0, ge=0),
    bed_count: int = Query(default=12, ge=1, le=36),
    selected_bed: int = Query(default=0, ge=0),
    risk_model: str | None = Query(default=None),
):
    if ARTIFACT_LOAD_ERROR:
        return HTMLResponse(
            content=(
                "<h2>Missing model artifacts</h2>"
                f"<p>{ARTIFACT_LOAD_ERROR}</p>"
                "<p>Run <code>python scripts/train.py</code> first.</p>"
            ),
            status_code=500,
        )

    assert META is not None and PRIMARY_MODEL_ID is not None
    model_options = model_ids_primary_first(META, PRIMARY_MODEL_ID)
    rm = risk_model if risk_model in model_options else PRIMARY_MODEL_ID

    bed_summaries = build_ward_overview(
        seed=seed, bed_count=bed_count, risk_model=rm
    )
    chosen_bed_idx = min(selected_bed, len(bed_summaries) - 1)
    selected_bed_row = bed_summaries[chosen_bed_idx]
    data = selected_bed_row["details"]

    ward_streams = build_ward_streams(seed=seed, bed_count=bed_count, risk_model=rm)

    return templates.TemplateResponse(
        request=request,
        name="fastapi_index.html",
        context={
            "seed": seed,
            "bed_count": bed_count,
            "bed_summaries": bed_summaries,
            "ward_streams": ward_streams,
            "selected_bed_idx": chosen_bed_idx,
            "selected_bed_row": selected_bed_row,
            "risk_model": rm,
            "model_options": model_options,
            "primary_model_id": PRIMARY_MODEL_ID,
            "risk_model_is_primary": rm == PRIMARY_MODEL_ID,
            **data,
        },
    )
