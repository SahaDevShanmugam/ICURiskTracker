"""
Simulated ICU vital streams for UI / inference demos (not for model training).

Produces DataFrames compatible with compute_timeseries_features (no labels required).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ScenarioId = Literal[
    "random",
    "demo_stable",
    "demo_transient_hypotension",
    "demo_progressive_decline",
    "demo_shock_pattern",
]


def _time_index(cfg: dict, n_steps: int) -> pd.DatetimeIndex:
    grid = int(cfg["data"]["grid_minutes"])
    return pd.date_range("2024-01-01", periods=n_steps, freq=f"{grid}min")


def _spo2_temp_series(
    map_s: np.ndarray,
    hr_s: np.ndarray,
    rr_s: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Plausible bedside monitor saturation (%) and temperature (°C)."""
    n = len(map_s)
    hypo = np.clip(72 - map_s, 0.0, 28.0)
    tachy_rr = np.clip(rr_s - 18.0, 0.0, 16.0)
    tachy_hr = np.clip(hr_s - 95.0, 0.0, 45.0)
    spo2_raw = (
        98.0
        - 0.38 * hypo
        - 0.12 * tachy_rr
        - 0.06 * tachy_hr
        + rng.normal(0.0, 0.5, size=n)
    )
    spo2 = np.clip(spo2_raw, 82.0, 100.0)
    temp_raw = (
        36.55 + 0.018 * (hr_s - 80.0) + 0.028 * np.clip(rr_s - 17.0, 0.0, 14.0)
    ) + rng.normal(0.0, 0.07, size=n)
    temp_c = np.clip(temp_raw, 35.4, 40.3)
    return spo2.astype(np.float64), temp_c.astype(np.float64)


def _vitals_frame(
    cfg: dict,
    stay_id: int,
    subject_id: int,
    age: float,
    sex: int,
    map_s: np.ndarray,
    sbp_s: np.ndarray,
    hr_s: np.ndarray,
    rr_s: np.ndarray,
    rng: np.random.Generator,
    lactate: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(map_s)
    if lactate is None:
        lactate = np.full(n, np.nan)
    spo2, temp_c = _spo2_temp_series(map_s, hr_s, rr_s, rng)
    return pd.DataFrame(
        {
            "stay_id": stay_id,
            "subject_id": subject_id,
            "charttime": _time_index(cfg, n),
            "map": map_s,
            "sbp": sbp_s,
            "hr": hr_s,
            "rr": rr_s,
            "spo2": spo2,
            "temp_c": temp_c,
            "lactate": lactate,
            "age": float(age),
            "sex": int(sex),
        }
    )


def generate_random_scenario(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Random but plausible vital trajectory; may include hypo episodes or recovery."""
    grid = int(cfg["data"]["grid_minutes"])
    hours = rng.uniform(8.0, 28.0)
    n = max(48, int(hours * 60 / grid))
    stay_id = int(rng.integers(900_000, 999_999))
    subject_id = stay_id
    age = float(rng.integers(45, 88))
    sex = int(rng.integers(0, 2))

    map0 = rng.normal(78, 5)
    sbp0 = map0 + rng.normal(38, 6)
    hr0 = rng.normal(86, 12)
    rr0 = rng.normal(18, 2.5)

    t = np.arange(n, dtype=np.float64)
    map_s = map0 + rng.normal(0, 1.8, n) + 0.08 * np.sin(t / 25)
    sbp_s = sbp0 + rng.normal(0, 3.5, n) + 0.1 * np.sin(t / 22)
    hr_s = hr0 + rng.normal(0, 3.0, n)
    rr_s = np.clip(rr0 + rng.normal(0, 1.2, n), 10, 34)

    roll = rng.random()
    if roll < 0.45:
        # transient dip
        s = rng.integers(int(n * 0.2), int(n * 0.75))
        ln = rng.integers(2, max(3, int(25 // max(grid / 5, 1))))
        dip = rng.uniform(8, 18)
        e = min(s + ln, n)
        map_s[s:e] -= dip
        sbp_s[s:e] -= dip * 1.1
    if roll > 0.55:
        # progressive decline in second half
        s = int(n * 0.45)
        drift = np.linspace(0, rng.uniform(12, 26), n - s)
        map_s[s:] -= drift
        sbp_s[s:] -= drift * 1.05
        hr_s[s:] += np.linspace(0, rng.uniform(6, 22), n - s)
    if rng.random() < 0.35:
        s2 = rng.integers(int(n * 0.15), int(n * 0.6))
        map_s[s2 : s2 + 4] -= rng.uniform(15, 28)
        sbp_s[s2 : s2 + 4] -= rng.uniform(18, 32)
        hr_s[s2 : s2 + 6] += rng.uniform(8, 20)

    lac = np.full(n, np.nan)
    if rng.random() < 0.25:
        li = rng.integers(int(n * 0.4), n - 2)
        lac[li : li + 2] = rng.uniform(1.3, 3.8)

    return _vitals_frame(cfg, stay_id, subject_id, age, sex, map_s, sbp_s, hr_s, rr_s, rng, lac)


def scenario_demo_stable(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    n = max(40, int(10 * 60 / grid))
    t = np.arange(n, dtype=np.float64)
    map_s = 82 + 0.02 * t + np.sin(t / 12) * 2 + rng.normal(0, 1.0, n)
    sbp_s = 118 + 0.03 * t + np.sin(t / 10) * 3 + rng.normal(0, 1.2, n)
    hr_s = 78 + np.sin(t / 15) * 4 + rng.normal(0, 1.5, n)
    rr_s = np.clip(16 + np.sin(t / 14) * 1.5 + rng.normal(0, 0.45, n), 10, 28)
    stay_id = int(rng.integers(200_000, 899_999))
    subject_id = stay_id
    age = float(rng.integers(55, 79))
    sex = int(rng.integers(0, 2))
    return _vitals_frame(cfg, stay_id, subject_id, age, sex, map_s, sbp_s, hr_s, rr_s, rng)


def scenario_demo_transient_hypotension(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    n = max(50, int(12 * 60 / grid))
    m0 = float(np.clip(80.0 + rng.normal(0, 2.5), 72, 88))
    sb0 = float(np.clip(m0 + 36 + rng.normal(0, 4), 105, 135))
    hr0 = float(np.clip(82.0 + rng.normal(0, 5), 68, 98))
    rr0 = float(np.clip(18.0 + rng.normal(0, 1.5), 12, 26))
    map_s = np.full(n, m0) + rng.normal(0, 0.35, n)
    sbp_s = np.full(n, sb0) + rng.normal(0, 0.5, n)
    hr_s = np.full(n, hr0) + rng.normal(0, 0.6, n)
    rr_s = np.clip(np.full(n, rr0) + rng.normal(0, 0.25, n), 10, 30)
    s = int(n * (0.32 + rng.uniform(-0.06, 0.06)))
    ep = max(3, int(15 // max(grid / 5, 1))) + int(rng.integers(-2, 3))
    e = min(s + ep, n)
    dip_m = float(np.clip(18 + rng.uniform(-4, 5), 10, 28))
    dip_sb = float(np.clip(26 + rng.uniform(-5, 6), 16, 36))
    map_s[s:e] -= dip_m
    sbp_s[s:e] -= dip_sb
    hr_s[s:e] += float(np.clip(6 + rng.normal(0, 2.5), 2, 14))
    stay_id = int(rng.integers(200_000, 899_999))
    subject_id = stay_id
    age = float(rng.integers(58, 82))
    sex = int(rng.integers(0, 2))
    return _vitals_frame(cfg, stay_id, subject_id, age, sex, map_s, sbp_s, hr_s, rr_s, rng)


def scenario_demo_progressive_decline(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    n = max(60, int(16 * 60 / grid))
    t = np.linspace(0, 1, n)
    map_s = 88 - 28 * t**1.8 + rng.normal(0, 1.0, n)
    sbp_s = 128 - 38 * t**1.8 + rng.normal(0, 1.2, n)
    hr_s = 72 + 28 * t + np.linspace(0, 12, n) + rng.normal(0, 1.0, n)
    rr_s = np.clip(17 + 8 * t + rng.normal(0, 0.5, n), 10, 34)
    stay_id = int(rng.integers(200_000, 899_999))
    subject_id = stay_id
    age = float(rng.integers(54, 80))
    sex = int(rng.integers(0, 2))
    return _vitals_frame(cfg, stay_id, subject_id, age, sex, map_s, sbp_s, hr_s, rr_s, rng)


def scenario_demo_shock_pattern(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    n = max(55, int(14 * 60 / grid))
    map_hi = float(np.clip(84 + rng.normal(0, 3), 76, 92))
    map_lo = float(np.clip(56 + rng.normal(0, 3), 48, 64))
    map_s = np.concatenate(
        [
            np.linspace(map_hi, map_lo, n // 2),
            np.full(n - n // 2, map_lo) + rng.normal(0, 1.5, n - n // 2),
        ]
    )
    sbp_s = map_s + float(np.clip(32 + rng.normal(0, 2), 26, 40))
    hr_hi = float(np.clip(76 + rng.normal(0, 4), 66, 88))
    hr_lo = float(np.clip(112 + rng.normal(0, 6), 98, 128))
    hr_s = np.concatenate(
        [
            np.linspace(hr_hi, hr_lo, n // 2),
            np.full(n - n // 2, hr_lo) + np.linspace(0, float(rng.uniform(4, 10)), n - n // 2),
        ]
    )
    rr_s = np.clip(18 + np.linspace(0, 12, n) + rng.normal(0, 0.4, n), 12, 34)
    lac = np.full(n, np.nan)
    lac[n // 2 :] = np.linspace(
        float(rng.uniform(1.0, 1.6)), float(rng.uniform(2.6, 3.8)), n - n // 2
    )
    stay_id = int(rng.integers(200_000, 899_999))
    subject_id = stay_id
    age = float(rng.integers(56, 84))
    sex = int(rng.integers(0, 2))
    return _vitals_frame(cfg, stay_id, subject_id, age, sex, map_s, sbp_s, hr_s, rr_s, rng, lac)


def build_scenario(cfg: dict, scenario: ScenarioId, rng: np.random.Generator) -> pd.DataFrame:
    if scenario == "random":
        return generate_random_scenario(cfg, rng)
    if scenario == "demo_stable":
        return scenario_demo_stable(cfg, rng)
    if scenario == "demo_transient_hypotension":
        return scenario_demo_transient_hypotension(cfg, rng)
    if scenario == "demo_progressive_decline":
        return scenario_demo_progressive_decline(cfg, rng)
    if scenario == "demo_shock_pattern":
        return scenario_demo_shock_pattern(cfg, rng)
    raise ValueError(f"Unknown scenario: {scenario}")


def scenario_display_name(sid: ScenarioId) -> str:
    return {
        "random": "Random scenario (stream replay)",
        "demo_stable": "Demo: stable vitals (seeded per bay)",
        "demo_transient_hypotension": "Demo: transient hypotension (recovers)",
        "demo_progressive_decline": "Demo: progressive BP decline",
        "demo_shock_pattern": "Demo: shock-like pattern (low MAP + ↑ HR + lactate)",
    }[sid]
