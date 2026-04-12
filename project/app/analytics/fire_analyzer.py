from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config import FireConfig
from app.utils.math_utils import clamp


@dataclass
class FireResult:
    score: float  # 0..100
    mask_ratio: float  # 0..1
    flicker_ratio: float  # 0..1


class FireAnalyzer:
    """
    Heuristic fire/flame signal:
    - HSV color mask (red/orange/yellow)
    - temporal "flicker" via mask-change ratio
    """

    def __init__(self, cfg: FireConfig) -> None:
        self.cfg = cfg
        self._prev_mask: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_mask = None

    def compute(self, frame_bgr: np.ndarray) -> FireResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return FireResult(score=0.0, mask_ratio=0.0, flicker_ratio=0.0)

        img = frame_bgr
        if self.cfg.downscale_width and int(self.cfg.downscale_width) > 0:
            h, w = img.shape[:2]
            target_w = int(self.cfg.downscale_width)
            if w > target_w and w > 0:
                target_h = max(1, int(round(h * (target_w / float(w)))))
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        hue_max = int(max(0, min(179, int(self.cfg.hue_max))))
        s_min = int(max(0, min(255, int(self.cfg.s_min))))
        v_min = int(max(0, min(255, int(self.cfg.v_min))))

        mask = (h <= hue_max) & (s >= s_min) & (v >= v_min)
        mask_u8 = mask.astype(np.uint8)

        # Clean up tiny speckles
        mask_u8 = cv2.medianBlur(mask_u8 * 255, 5)
        mask_u8 = (mask_u8 > 0).astype(np.uint8)

        mask_ratio = float(mask_u8.mean())

        flicker_ratio = 0.0
        if self._prev_mask is not None and self._prev_mask.shape == mask_u8.shape:
            flicker_ratio = float(np.mean(mask_u8 != self._prev_mask))
        self._prev_mask = mask_u8

        # Score: area drives base, flicker boosts confidence but doesn't zero it out.
        min_area = float(max(1e-6, self.cfg.min_area_ratio))
        base = clamp((mask_ratio / min_area) * 100.0, 0.0, 100.0)
        flicker_ref = float(max(1e-6, self.cfg.flicker_ref))
        flicker_factor = clamp(flicker_ratio / flicker_ref, 0.0, 1.0)
        score = clamp(base * (0.65 + 0.35 * flicker_factor), 0.0, 100.0)

        return FireResult(score=float(score), mask_ratio=float(mask_ratio), flicker_ratio=float(flicker_ratio))

