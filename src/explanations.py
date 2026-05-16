"""Human-readable risk explanations from vitals + model feature context."""

from __future__ import annotations

import numpy as np
import pandas as pd


def humanize_feature_name(fname: str) -> str:
    if fname == "lactate_filled":
        return "Lactate (forward-filled)"
    if fname.endswith("_slope"):
        base = fname.replace("_slope", "").upper()
        return f"{base} trend (recent slope)"
    parts = fname.split("_")
    if len(parts) >= 3 and parts[1] in ("mean", "std", "min") and parts[-1].isdigit():
        vital = parts[0].upper()
        stat = parts[1]
        mins = parts[-1]
        return f"{vital} {stat} (~{mins}m window)"
    if fname == "below_map_threshold":
        return "MAP below alarm threshold flag"
    return fname.replace("_", " ")


def clinical_vital_explanations(g: pd.DataFrame, cfg: dict) -> list[str]:
    """Interpretable statements from raw vitals (last values + short trend)."""
    if g.empty:
        return []
    out: list[str] = []
    last = g.iloc[-1]
    map_thr = float(cfg["alarms"]["map_threshold_mmhg"])
    sbp_thr = float(cfg["alarms"]["sbp_threshold_mmhg"])

    if last["map"] < map_thr:
        out.append(
            f"<strong>Hypotension:</strong> MAP is <strong>{last['map']:.0f} mmHg</strong>, "
            f"below the <strong>{map_thr:.0f}</strong> mmHg alarm line."
        )
    elif last["map"] < map_thr + 8:
        out.append(
            f"<strong>Borderline perfusion:</strong> MAP <strong>{last['map']:.0f} mmHg</strong> "
            f"is close to the <strong>{map_thr:.0f}</strong> threshold."
        )
    else:
        out.append(
            f"<strong>MAP</strong> is <strong>{last['map']:.0f} mmHg</strong>, "
            "above the usual hypotension alarm threshold."
        )

    if last["sbp"] < sbp_thr:
        out.append(
            f"<strong>SBP</strong> is <strong>{last['sbp']:.0f} mmHg</strong>, "
            f"below <strong>{sbp_thr:.0f}</strong> (hypotension rule)."
        )

    tail = min(18, len(g))
    if tail >= 4:
        m0 = float(g["map"].iloc[-tail])
        m1 = float(g["map"].iloc[-1])
        delta = (m1 - m0) / max(tail - 1, 1)
        if delta < -0.35:
            out.append(
                "<strong>Downward MAP trajectory</strong> over recent readings — "
                "consistent with worsening perfusion."
            )
        elif delta > 0.35:
            out.append("<strong>MAP is improving</strong> over recent readings.")

    if last["hr"] >= 105:
        out.append(
            f"<strong>Tachycardia signal:</strong> HR <strong>{last['hr']:.0f} bpm</strong> "
            "is elevated (stress / compensation)."
        )
    elif last["hr"] <= 55:
        out.append(
            f"<strong>Bradycardia:</strong> HR <strong>{last['hr']:.0f} bpm</strong> "
            "is relatively low for ICU monitoring context."
        )

    if last["rr"] >= 28:
        out.append(
            f"<strong>Respiratory stress:</strong> RR <strong>{last['rr']:.0f}</strong>/min is high."
        )

    lac = last.get("lactate")
    if lac is not None and np.isfinite(lac) and lac >= 2.0:
        out.append(
            f"<strong>Lactate {lac:.1f} mmol/L</strong> — supports concern for tissue hypoperfusion "
            "when interpreted clinically."
        )

    return out


def _logistic_contribution_lines(
    fe_row: pd.Series,
    feature_names: list[str],
    meta: dict,
    max_lines: int,
) -> tuple[list[str], set[int]]:
    coef = meta.get("deterioration_logistic_coefficients")
    mean = meta.get("deterioration_feature_mean")
    std = meta.get("deterioration_feature_std")
    if (
        not coef
        or not mean
        or not std
        or len(coef) != len(feature_names)
        or len(mean) != len(feature_names)
    ):
        return [], set()
    x = np.array([float(fe_row.get(c, 0.0)) for c in feature_names], dtype=np.float64)
    m = np.array(mean, dtype=np.float64)
    s = np.maximum(np.array(std, dtype=np.float64), 1e-6)
    z = (x - m) / s
    b = np.array(coef, dtype=np.float64)
    contrib = b * z
    order = np.argsort(-np.abs(contrib))
    lines: list[str] = []
    used: set[int] = set()
    for idx in order[:max_lines]:
        if abs(contrib[idx]) < 1e-8:
            break
        used.add(int(idx))
        name = humanize_feature_name(feature_names[idx])
        cj = float(contrib[idx])
        direction = "increases" if cj > 0 else "decreases"
        lines.append(
            f"<strong>{name}</strong> — in the primary logistic model this pattern "
            f"<strong>{direction}</strong> estimated log-risk (contribution ≈ {cj:+.2f} in standardized units)."
        )
    return lines, used


def model_feature_explanations(
    fe_row: pd.Series,
    feature_names: list[str],
    meta: dict,
    max_bullets: int = 5,
) -> list[str]:
    """
    Surface drivers: optional logistic linear contributions, else |z| × permutation importance.
    """
    lines: list[str] = []
    ptype = str(meta.get("deterioration_primary_type", "")).lower()
    used_idx: set[int] = set()
    if ptype == "logistic" and meta.get("deterioration_logistic_coefficients"):
        sub, used_idx = _logistic_contribution_lines(
            fe_row, feature_names, meta, min(3, max_bullets)
        )
        lines.extend(sub)
    remaining = max(0, max_bullets - len(lines))
    if remaining == 0:
        return lines

    mean = meta.get("deterioration_feature_mean")
    std = meta.get("deterioration_feature_std")
    imp = meta.get("deterioration_feature_importance")
    if not mean or not std or len(mean) != len(feature_names) or len(std) != len(feature_names):
        return lines
    x = np.array([float(fe_row.get(c, 0.0)) for c in feature_names], dtype=np.float64)
    m = np.array(mean, dtype=np.float64)
    s = np.maximum(np.array(std, dtype=np.float64), 1e-6)
    z = (x - m) / s
    w = np.array(imp, dtype=np.float64) if imp and len(imp) == len(feature_names) else np.ones(len(z))
    contrib = np.abs(z) * w
    order = np.argsort(-contrib)
    added = 0
    for idx in order:
        if int(idx) in used_idx:
            continue
        if contrib[idx] < 1e-9:
            continue
        name = humanize_feature_name(feature_names[idx])
        zi = float(z[idx])
        if zi > 1.0:
            qual = "notably <strong>above</strong> the training cohort average for this signal"
        elif zi < -1.0:
            qual = "notably <strong>below</strong> the training cohort average for this signal"
        elif abs(zi) < 0.35:
            qual = "close to the training cohort average"
        else:
            qual = "somewhat shifted from the training cohort average"
        lines.append(
            f"<strong>{name}</strong> — {qual} (standardized offset ≈ {zi:+.1f}). "
            "The risk model uses this pattern <strong>together with</strong> other features."
        )
        added += 1
        if added >= remaining:
            break
    return lines


def risk_tier_label(score_0_100: float, cfg: dict) -> str:
    """Map integer risk score 0–100 to tier id from risk_display.tiers (exclusive upper bounds)."""
    tiers = cfg.get("risk_display", {}).get("tiers")
    if not tiers:
        return "moderate"
    s = int(round(float(score_0_100)))
    s = max(0, min(100, s))
    for t in tiers:
        if s < int(t["below"]):
            return str(t["id"])
    return str(tiers[-1]["id"])
