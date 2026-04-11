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


def _alpha_blend(dst: np.ndarray, src: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(max(0.0, min(1.0, alpha)))
    return cv2.addWeighted(src, alpha, dst, 1.0 - alpha, 0.0)


def draw_hud_panel(
    frame: np.ndarray,
    *,
    panel_width: int,
    title: str,
    lines: List[Tuple[str, Tuple[int, int, int]]],
    risk_score: float,
    level: str,
) -> np.ndarray:
    """
    Modern-ish HUD: translucent right panel + risk bar.
    Returns new image (frame + panel).
    """
    h, w = frame.shape[:2]
    pw = int(max(240, panel_width))

    panel = np.full((h, pw, 3), (10, 16, 30), dtype=np.uint8)
    panel2 = panel.copy()

    # Top header
    cv2.rectangle(panel2, (0, 0), (pw, 76), (15, 26, 48), -1)
    cv2.putText(panel2, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 245, 255), 2, cv2.LINE_AA)
    cv2.putText(panel2, "Live risk view", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 160, 185), 1, cv2.LINE_AA)

    # Risk bar
    r = float(max(0.0, min(100.0, risk_score)))
    bar_x1, bar_x2 = 16, pw - 16
    bar_y1, bar_y2 = 94, 112
    cv2.rectangle(panel2, (bar_x1, bar_y1), (bar_x2, bar_y2), (34, 48, 78), -1)
    fill = int(round((bar_x2 - bar_x1) * (r / 100.0)))
    color = level_color(level)
    cv2.rectangle(panel2, (bar_x1, bar_y1), (bar_x1 + fill, bar_y2), color, -1)
    cv2.putText(panel2, f"{r:0.1f}", (bar_x2 - 64, bar_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 240), 1, cv2.LINE_AA)

    # Divider
    cv2.line(panel2, (16, 132), (pw - 16, 132), (30, 44, 70), 1, cv2.LINE_AA)

    # Body lines
    y = 162
    for text, color in lines:
        cv2.putText(panel2, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        y += 26
        if y > h - 16:
            break

    panel = _alpha_blend(panel, panel2, 0.92)
    return np.hstack([frame, panel])


def level_color(level: str) -> Tuple[int, int, int]:
    level = (level or "").upper()
    if level == "ALARM":
        return (0, 0, 255)
    if level in ("WARNING", "UYARI"):
        return (0, 165, 255)
    return (0, 200, 0)

