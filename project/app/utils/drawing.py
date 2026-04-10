from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    # stable pseudo-random
    r = (track_id * 37) % 255
    g = (track_id * 17) % 255
    b = (track_id * 97) % 255
    return int(b), int(g), int(r)


def draw_bbox(
    img: np.ndarray,
    xyxy: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_skeleton(
    img: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    color: Tuple[int, int, int],
    conf_thr: float = 0.25,
) -> None:
    # COCO17 connections
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    for i in range(keypoints_xy.shape[0]):
        if float(keypoints_conf[i]) < conf_thr:
            continue
        x, y = keypoints_xy[i]
        cv2.circle(img, (int(x), int(y)), 2, color, -1)

    for a, b in edges:
        if float(keypoints_conf[a]) < conf_thr or float(keypoints_conf[b]) < conf_thr:
            continue
        xa, ya = keypoints_xy[a]
        xb, yb = keypoints_xy[b]
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)


def draw_side_panel(
    frame: np.ndarray,
    panel_width: int,
    lines: List[Tuple[str, Tuple[int, int, int]]],
    bg_color: Tuple[int, int, int] = (25, 25, 25),
) -> np.ndarray:
    h, w = frame.shape[:2]
    panel = np.full((h, panel_width, 3), bg_color, dtype=np.uint8)
    y = 28
    for text, color in lines:
        cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        y += 28
        if y > h - 10:
            break
    return np.hstack([frame, panel])


def level_color(level: str) -> Tuple[int, int, int]:
    level = (level or "").upper()
    if level == "ALARM":
        return (0, 0, 255)
    if level in ("WARNING", "UYARI"):
        return (0, 165, 255)
    return (0, 200, 0)

