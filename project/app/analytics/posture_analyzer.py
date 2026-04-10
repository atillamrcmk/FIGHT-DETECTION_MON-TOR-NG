from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.utils.math_utils import bbox_aspect_ratio, clamp, normalize_0_100


@dataclass
class PostureFeatures:
    posture_state: str  # "standing" | "sitting" | "lying" | "unknown"
    posture_score: float  # 0-100 (higher => more risky posture for monitoring)
    collapse_score: float  # 0-100 (higher => collapse/lying transition-like)


class PostureAnalyzer:
    """
    Very lightweight posture estimator (MVP):
    - Uses bbox aspect ratio + hip/shoulder vertical relation if available.
    - collapse_score is boosted when bbox becomes wide (lying-ish) and keypoints suggest low vertical spread.

    TODO: Replace with camera-calibrated posture classifier for production.
    """

    def __init__(self, collapse_angle_threshold: float = 0.55):
        self.collapse_aspect_thr = float(collapse_angle_threshold)

    def compute(
        self,
        xyxy: Tuple[float, float, float, float],
        keypoints_xy: Optional[np.ndarray],
        keypoints_conf: Optional[np.ndarray],
    ) -> PostureFeatures:
        ar = bbox_aspect_ratio(xyxy)  # w/h

        posture_state = "unknown"
        if ar > 1.25:
            posture_state = "lying"
        elif ar < 0.75:
            posture_state = "standing"
        else:
            posture_state = "sitting"

        # Compute a crude "vertical spread" if keypoints exist
        spread_score = 0.0
        if keypoints_xy is not None and keypoints_conf is not None and len(keypoints_xy) >= 17:
            mask = keypoints_conf > 0.25
            if mask.any():
                ys = keypoints_xy[mask, 1]
                spread = float(ys.max() - ys.min())
                # small spread may indicate curled/lying (camera dependent)
                spread_score = normalize_0_100(max(0.0, 250.0 - spread), 250.0)

        # posture_score: monitoring risk emphasis on lying
        if posture_state == "lying":
            posture_score = 80.0
        elif posture_state == "sitting":
            posture_score = 30.0
        elif posture_state == "standing":
            posture_score = 10.0
        else:
            posture_score = 20.0

        # collapse_score: wide bbox + low vertical spread => higher
        ar_excess = max(0.0, ar - 1.0)
        ar_score = normalize_0_100(ar_excess, 1.2)  # ar 2.2 => ~100
        collapse_score = clamp(0.7 * ar_score + 0.3 * spread_score, 0.0, 100.0)

        return PostureFeatures(
            posture_state=posture_state,
            posture_score=float(posture_score),
            collapse_score=float(collapse_score),
        )

