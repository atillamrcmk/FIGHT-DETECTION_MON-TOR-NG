from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.utils.math_utils import l2, normalize_0_100, safe_mean


@dataclass
class MovementFeatures:
    movement_score: float
    velocity: float


class MovementAnalyzer:
    """
    movement_score:
      - displacement over last N centers (pixels) normalized to 0-100
    velocity:
      - mean per-step displacement (pixels/frame) normalized-ish
    """

    def __init__(self, movement_window: int = 15, movement_max_px: float = 220.0, velocity_max_px_per_frame: float = 35.0):
        self.movement_window = int(movement_window)
        self.movement_max_px = float(movement_max_px)
        self.velocity_max = float(velocity_max_px_per_frame)

    def compute(self, history_centers: List[Tuple[float, float]]) -> MovementFeatures:
        if len(history_centers) < 2:
            return MovementFeatures(movement_score=0.0, velocity=0.0)

        window = history_centers[-self.movement_window :]
        disp = l2(window[0], window[-1])

        steps = [l2(window[i], window[i + 1]) for i in range(len(window) - 1)]
        v = safe_mean(steps, default=0.0)

        movement_score = normalize_0_100(disp, self.movement_max_px)
        velocity_score = normalize_0_100(v, self.velocity_max)
        return MovementFeatures(movement_score=movement_score, velocity=velocity_score)

