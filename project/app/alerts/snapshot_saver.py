from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional

import cv2


def _slug_reason(s: str, max_len: int = 48) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return (s[:max_len] if s else "event")


@dataclass
class SnapshotSaverConfig:
    snapshots_dir: str = "data/snapshots"
    cooldown_seconds: float = 3.0
    jpeg_quality: int = 92
    show_preview_window: bool = True
    preview_max_width: int = 1280
    preview_window_name: str = "Uyari - orijinal kare"


class SnapshotSaver:
    """
    Saves raw BGR frames (no UI overlay) on UYARI/ALARM with time cooldown.
    Optional second OpenCV window shows the last saved frame (downscaled if wide).
    """

    def __init__(self, cfg: SnapshotSaverConfig) -> None:
        self.cfg = cfg
        os.makedirs(self.cfg.snapshots_dir, exist_ok=True)
        self._cooldown_until_t: float = 0.0

    def maybe_save(
        self,
        frame_bgr: np.ndarray,
        *,
        level: str,
        risk_score: float,
        main_reason: str,
        frame_idx: int,
    ) -> Optional[str]:
        now = time.time()
        if now < self._cooldown_until_t:
            return None

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        slug = _slug_reason(main_reason)
        fname = f"{ts}_f{frame_idx}_{level}_r{risk_score:.0f}_{slug}.jpg"
        path = os.path.join(self.cfg.snapshots_dir, fname)

        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)]
        ok = cv2.imwrite(path, frame_bgr, params)
        if not ok:
            return None

        self._cooldown_until_t = now + float(self.cfg.cooldown_seconds)

        if self.cfg.show_preview_window:
            self._show_preview(frame_bgr)

        return path

    def _show_preview(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        mw = max(320, int(self.cfg.preview_max_width))
        if w > mw:
            scale = mw / float(w)
            preview = cv2.resize(
                frame_bgr,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            preview = frame_bgr
        cv2.namedWindow(self.cfg.preview_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.cfg.preview_window_name, preview)
