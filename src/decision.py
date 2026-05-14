"""Decision engine combining alarm validity and deterioration risk (reserved for future fusion)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Action = Literal["suppress", "keep", "escalate"]


@dataclass
class DecisionConfig:
    tau_d_low: float
    tau_d_high: float
    tau_a_false: float
    tau_a_true: float
    w_alarm: float = 0.5
    w_risk: float = 0.5


def fusion_score(p_alarm_true: np.ndarray, p_deterioration: np.ndarray, cfg: DecisionConfig) -> np.ndarray:
    return cfg.w_alarm * p_alarm_true + cfg.w_risk * p_deterioration


def decide_scalar(p_alarm_true: float, p_deterioration: float, cfg: DecisionConfig) -> tuple[Action, str]:
    """Return action and a short template explanation."""
    low_risk = p_deterioration < cfg.tau_d_low
    high_risk = p_deterioration >= cfg.tau_d_high
    likely_false = p_alarm_true < cfg.tau_a_false
    likely_true = p_alarm_true >= cfg.tau_a_true

    if low_risk and likely_false:
        return "suppress", (
            "Alarm suppressed: predicted false alarm and low deterioration risk "
            f"(P_valid={p_alarm_true:.2f}, P_risk={p_deterioration:.2f})."
        )
    if high_risk and likely_false:
        return "keep", (
            "Kept despite uncertain validity: high deterioration risk "
            f"(P_valid={p_alarm_true:.2f}, P_risk={p_deterioration:.2f})."
        )
    if low_risk and likely_true:
        return "keep", (
            "Kept: likely true hypotension even with lower short-term risk "
            f"(P_valid={p_alarm_true:.2f}, P_risk={p_deterioration:.2f})."
        )
    if high_risk and likely_true:
        return "escalate", (
            "Escalated: likely true alarm with elevated deterioration risk "
            f"(P_valid={p_alarm_true:.2f}, P_risk={p_deterioration:.2f})."
        )
    return "keep", (
        "Borderline: default keep for safety "
        f"(P_valid={p_alarm_true:.2f}, P_risk={p_deterioration:.2f})."
    )


def decide_batch(
    p_alarm_true: np.ndarray,
    p_deterioration: np.ndarray,
    cfg: DecisionConfig,
) -> tuple[list[Action], list[str]]:
    actions: list[Action] = []
    texts: list[str] = []
    for pa, pd in zip(p_alarm_true, p_deterioration):
        act, txt = decide_scalar(float(pa), float(pd), cfg)
        actions.append(act)
        texts.append(txt)
    return actions, texts


def decision_config_from_yaml(cfg: dict) -> DecisionConfig:
    d = cfg.get("decision") or {}
    w = d.get("fusion_weights", [0.5, 0.5])
    return DecisionConfig(
        tau_d_low=float(d.get("tau_d_low", 0.35)),
        tau_d_high=float(d.get("tau_d_high", 0.55)),
        tau_a_false=float(d.get("tau_a_false", 0.45)),
        tau_a_true=float(d.get("tau_a_true", 0.55)),
        w_alarm=float(w[0]),
        w_risk=float(w[1]),
    )
