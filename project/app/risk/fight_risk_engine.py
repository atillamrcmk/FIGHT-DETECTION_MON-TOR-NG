from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from app.utils.math_utils import clamp


@dataclass
class FightRiskResult:
    risk_score: float
    level: str  # NORMAL/WARNING/ALARM
    main_reason: str
    components: Dict[str, float]


class FightRiskEngine:
    def __init__(self, weights: Dict[str, float], warning_thr: float = 40.0, alarm_thr: float = 70.0) -> None:
        self.w = weights
        self.warning_thr = float(warning_thr)
        self.alarm_thr = float(alarm_thr)

    def compute(
        self,
        movement_score: float,
        agitation_score: float,
        interaction_score: float,
        clustering_score: float,
        scene_motion_score: float,
        fight_model_score: float,
        bonus_flag: bool,
        active_interactions: int,
    ) -> FightRiskResult:
        components = {
            "movement_score": float(movement_score),
            "pose_agitation_score": float(agitation_score),
            "interaction_score": float(interaction_score),
            "clustering_score": float(clustering_score),
            "scene_motion_score": float(scene_motion_score),
            "fight_model_score": float(fight_model_score),
        }

        score = 0.0
        score += self.w.get("movement", 1.0) * movement_score
        score += self.w.get("agitation", 1.0) * agitation_score
        score += self.w.get("interaction", 1.0) * interaction_score
        score += self.w.get("clustering", 1.0) * clustering_score
        score += self.w.get("scene_motion", 1.0) * scene_motion_score
        score += self.w.get("fight_model", 1.0) * fight_model_score

        if bonus_flag:
            score += float(self.w.get("bonus_fast_close_and_agitated", 12.0))
            components["bonus_fast_close_and_agitated"] = float(self.w.get("bonus_fast_close_and_agitated", 12.0))
        else:
            components["bonus_fast_close_and_agitated"] = 0.0

        score = clamp(score / 6.0, 0.0, 100.0)  # keep in 0-100 band (6 components)

        if score >= self.alarm_thr:
            level = "ALARM"
        elif score >= self.warning_thr:
            level = "UYARI"
        else:
            level = "NORMAL"

        # main reason: highest component
        reason = max(components.items(), key=lambda kv: kv[1])[0] if components else "n/a"
        if active_interactions > 0 and level != "NORMAL":
            reason = f"{reason} (+{active_interactions} interactions)"

        return FightRiskResult(risk_score=float(score), level=level, main_reason=reason, components=components)

