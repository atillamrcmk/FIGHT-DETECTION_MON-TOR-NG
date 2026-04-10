from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.utils.math_utils import clamp, l2, normalize_0_100


@dataclass
class AggressionFeatures:
    pose_agitation_score: float  # 0-100
    self_harm_risk_score: float  # 0-100


class AggressionAnalyzer:
    """
    pose_agitation_score:
      - wrist/elbow/shoulder keypoints' frame-to-frame displacement magnitude

    self_harm_risk_score (very heuristic MVP):
      - if wrists repeatedly get close to head/neck area (nose/eyes/ears), score up

    TODO: Replace with behavior model for production.
    """

    def __init__(self, self_harm_head_radius_px: float = 60.0) -> None:
        self.head_radius = float(self_harm_head_radius_px)

        # keypoint indices COCO17
        self.idx_nose = 0
        self.idx_leye = 1
        self.idx_reye = 2
        self.idx_lear = 3
        self.idx_rear = 4
        self.idx_lsho = 5
        self.idx_rsho = 6
        self.idx_lelb = 7
        self.idx_relb = 8
        self.idx_lwri = 9
        self.idx_rwri = 10

        self._agit_max_px = 85.0

    def compute(
        self,
        kxy_now: Optional[np.ndarray],
        kcf_now: Optional[np.ndarray],
        kxy_prev: Optional[np.ndarray],
        kcf_prev: Optional[np.ndarray],
    ) -> AggressionFeatures:
        if kxy_now is None or kcf_now is None or kxy_prev is None or kcf_prev is None:
            return AggressionFeatures(pose_agitation_score=0.0, self_harm_risk_score=0.0)
        if len(kxy_now) < 17 or len(kxy_prev) < 17:
            return AggressionFeatures(pose_agitation_score=0.0, self_harm_risk_score=0.0)

        # agitation: sum displacements for upper body joints
        joints = [
            self.idx_lsho,
            self.idx_rsho,
            self.idx_lelb,
            self.idx_relb,
            self.idx_lwri,
            self.idx_rwri,
        ]
        raw = 0.0
        used = 0
        for j in joints:
            if float(kcf_now[j]) < 0.25 or float(kcf_prev[j]) < 0.25:
                continue
            raw += l2(tuple(kxy_now[j]), tuple(kxy_prev[j]))
            used += 1
        if used > 0:
            raw = raw / used
        agitation = normalize_0_100(raw, self._agit_max_px)

        # self-harm heuristic: wrists close to head keypoints
        head_idxs = [self.idx_nose, self.idx_leye, self.idx_reye, self.idx_lear, self.idx_rear]
        head_pts = []
        for i in head_idxs:
            if float(kcf_now[i]) >= 0.25:
                head_pts.append(tuple(kxy_now[i]))
        if not head_pts:
            return AggressionFeatures(pose_agitation_score=float(agitation), self_harm_risk_score=0.0)

        def min_dist_to_head(p: Tuple[float, float]) -> float:
            return min(l2(p, hp) for hp in head_pts)

        sh_raw = 0.0
        for w in [self.idx_lwri, self.idx_rwri]:
            if float(kcf_now[w]) < 0.25:
                continue
            d = min_dist_to_head(tuple(kxy_now[w]))
            if d < self.head_radius:
                sh_raw += (self.head_radius - d)

        self_harm = normalize_0_100(sh_raw, self.head_radius * 1.5)
        return AggressionFeatures(pose_agitation_score=float(agitation), self_harm_risk_score=float(self_harm))

