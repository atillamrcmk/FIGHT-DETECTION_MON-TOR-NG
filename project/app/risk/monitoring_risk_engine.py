from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from app.utils.math_utils import clamp


@dataclass
class MonitoringRiskResult:
    risk_score: float
    level: str  # NORMAL/WARNING/ALARM
    main_reason: str
    components: Dict[str, float]
    direct_alarm: bool


class MonitoringRiskEngine:
    def __init__(self, weights: Dict[str, float], warning_thr: float = 40.0, alarm_thr: float = 70.0) -> None:
        self.w = weights
        self.warning_thr = float(warning_thr)
        self.alarm_thr = float(alarm_thr)

        # MVP direct alarm cutoffs
        self._collapse_direct = 75.0
        self._self_harm_direct = 75.0

    def compute(
        self,
        inactivity_score: float,
        agitation_score: float,
        posture_score: float,
        self_harm_risk_score: float,
        collapse_score: float,
        still_seconds_max: float,
    ) -> MonitoringRiskResult:
        components = {
            "inactivity_score": float(inactivity_score),
            "agitation_score": float(agitation_score),
            "posture_score": float(posture_score),
            "self_harm_risk_score": float(self_harm_risk_score),
            "collapse_score": float(collapse_score),
        }

        direct_alarm = (collapse_score >= self._collapse_direct) or (self_harm_risk_score >= self._self_harm_direct)

        score = 0.0
        score += self.w.get("inactivity", 1.0) * inactivity_score
        score += self.w.get("agitation", 1.0) * agitation_score
        score += self.w.get("posture", 1.0) * posture_score
        score += self.w.get("self_harm", 1.0) * self_harm_risk_score
        score += self.w.get("collapse", 1.0) * collapse_score
        score = clamp(score / 5.0, 0.0, 100.0)

        if direct_alarm:
            level = "ALARM"
        else:
            if score >= self.alarm_thr:
                level = "ALARM"
            elif score >= self.warning_thr:
                level = "UYARI"
            else:
                level = "NORMAL"

        reason = max(components.items(), key=lambda kv: kv[1])[0] if components else "n/a"

        # Special case: very long inactivity -> warning
        if level == "NORMAL" and still_seconds_max > 0:
            if still_seconds_max >= 60.0:
                level = "UYARI"
                reason = "long_inactivity"

        return MonitoringRiskResult(
            risk_score=float(score),
            level=level,
            main_reason=reason,
            components=components,
            direct_alarm=bool(direct_alarm),
        )

