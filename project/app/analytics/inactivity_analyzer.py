from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from app.utils.math_utils import clamp, l2, normalize_0_100, safe_mean


@dataclass
class InactivityFeatures:
    inactivity_score: float  # 0-100 (higher => more inactive)
    still_seconds: float


class InactivityAnalyzer:
    """
    Measures how long a track has been almost still (based on recent centroid jitter).
    """

    def __init__(self, inactivity_seconds: float = 15.0, move_eps_px: float = 2.5) -> None:
        self.inactivity_seconds = float(inactivity_seconds)
        self.move_eps_px = float(move_eps_px)

    def compute(self, history_centers: List[Tuple[float, float]], fps: float) -> InactivityFeatures:
        if len(history_centers) < 2 or fps <= 0:
            return InactivityFeatures(inactivity_score=0.0, still_seconds=0.0)

        # Walk backwards until movement exceeds eps
        still_frames = 0
        for i in range(len(history_centers) - 2, -1, -1):
            d = l2(history_centers[i], history_centers[i + 1])
            if d <= self.move_eps_px:
                still_frames += 1
            else:
                break

        still_seconds = still_frames / fps
        inactivity_score = normalize_0_100(still_seconds, self.inactivity_seconds)
        return InactivityFeatures(inactivity_score=inactivity_score, still_seconds=still_seconds)

