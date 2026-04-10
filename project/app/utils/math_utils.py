from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def l2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def safe_mean(xs: Iterable[float], default: float = 0.0) -> float:
    xs = list(xs)
    if not xs:
        return default
    return float(sum(xs) / len(xs))


def normalize_0_100(x: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return clamp(100.0 * (x / max_value), 0.0, 100.0)


def bbox_center_xyxy(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_aspect_ratio(xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return float(w / h)


def pairwise_dist(points: List[Tuple[float, float]]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    diffs = arr[:, None, :] - arr[None, :, :]
    d = np.sqrt((diffs**2).sum(axis=-1))
    return d

