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
    lactate: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(map_s)
    if lactate is None:
        lactate = np.full(n, np.nan)
    return pd.DataFrame(
        {
            "stay_id": stay_id,
            "subject_id": subject_id,
            "charttime": _time_index(cfg, n),
            "map": map_s,
            "sbp": sbp_s,
            "hr": hr_s,
            "rr": rr_s,
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

    return _vitals_frame(cfg, stay_id, subject_id, age, sex, map_s, sbp_s, hr_s, rr_s, lac)


def scenario_demo_stable(cfg: dict) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    n = max(40, int(10 * 60 / grid))
    t = np.arange(n)
    map_s = 82 + 0.02 * t + np.sin(t / 12) * 2
    sbp_s = 118 + 0.03 * t + np.sin(t / 10) * 3
    hr_s = 78 + np.sin(t / 15) * 4
    rr_s = 16 + np.sin(t / 14) * 1.5
    return _vitals_frame(cfg, 1001, 1001, 62.0, 0, map_s, sbp_s, hr_s, rr_s)


def scenario_demo_transient_hypotension(cfg: dict) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    n = max(50, int(12 * 60 / grid))
    map_s = np.full(n, 80.0)
    sbp_s = np.full(n, 118.0)
    hr_s = np.full(n, 82.0)
    rr_s = np.full(n, 18.0)
    s, e = int(n * 0.35), int(n * 0.35) + max(3, int(15 // max(grid / 5, 1)))
    map_s[s:e] -= 20
    sbp_s[s:e] -= 28
    hr_s[s:e] += 6
    return _vitals_frame(cfg, 1002, 1002, 71.0, 1, map_s, sbp_s, hr_s, rr_s)


def scenario_demo_progressive_decline(cfg: dict) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    n = max(60, int(16 * 60 / grid))
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, n)
    map_s = 88 - 28 * t**1.8 + rng.normal(0, 1.0, n)
    sbp_s = 128 - 38 * t**1.8
    hr_s = 72 + 28 * t + np.linspace(0, 12, n)
    rr_s = 17 + 8 * t
    return _vitals_frame(cfg, 1003, 1003, 68.0, 0, map_s, sbp_s, hr_s, rr_s)


def scenario_demo_shock_pattern(cfg: dict) -> pd.DataFrame:
    grid = int(cfg["data"]["grid_minutes"])
    n = max(55, int(14 * 60 / grid))
    map_s = np.concatenate(
        [
            np.linspace(84, 58, n // 2),
            np.full(n - n // 2, 56) + np.random.default_rng(2).normal(0, 1.5, n - n // 2),
        ]
    )
    sbp_s = map_s + 32
    hr_s = np.concatenate(
        [
            np.linspace(76, 108, n // 2),
            np.full(n - n // 2, 112) + np.linspace(0, 6, n - n // 2),
        ]
    )
    rr_s = np.clip(18 + np.linspace(0, 12, n), 12, 32)
    lac = np.full(n, np.nan)
    lac[n // 2 :] = np.linspace(1.2, 3.2, n - n // 2)
    return _vitals_frame(cfg, 1004, 1004, 74.0, 1, map_s, sbp_s, hr_s, rr_s, lac)


def build_scenario(cfg: dict, scenario: ScenarioId, rng: np.random.Generator) -> pd.DataFrame:
    if scenario == "random":
        return generate_random_scenario(cfg, rng)
    if scenario == "demo_stable":
        return scenario_demo_stable(cfg)
    if scenario == "demo_transient_hypotension":
        return scenario_demo_transient_hypotension(cfg)
    if scenario == "demo_progressive_decline":
        return scenario_demo_progressive_decline(cfg)
    if scenario == "demo_shock_pattern":
        return scenario_demo_shock_pattern(cfg)
    raise ValueError(f"Unknown scenario: {scenario}")


def scenario_display_name(sid: ScenarioId) -> str:
    return {
        "random": "Random scenario (stream replay)",
        "demo_stable": "Fixed: stable vitals",
        "demo_transient_hypotension": "Fixed: transient hypotension (recovers)",
        "demo_progressive_decline": "Fixed: progressive BP decline",
        "demo_shock_pattern": "Fixed: shock-like pattern (low MAP + ↑ HR + lactate)",
    }[sid]
