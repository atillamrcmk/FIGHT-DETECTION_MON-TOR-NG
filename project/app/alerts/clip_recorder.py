from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ClipRecorderConfig:
    clips_dir: str
    pre_seconds: float = 3.0
    post_seconds: float = 5.0
    cooldown_seconds: float = 10.0
    fps_fallback: float = 25.0


class ClipRecorder:
    """
    Maintains a pre-event frame buffer and records post-event frames after trigger.
    Saves a clip when recording ends.
    """

    def __init__(self, cfg: ClipRecorderConfig) -> None:
        self.cfg = cfg
        os.makedirs(self.cfg.clips_dir, exist_ok=True)

        self._pre: Deque[np.ndarray] = deque(maxlen=1)
        self._recording = False
        self._record_until_t: float = 0.0
        self._cooldown_until_t: float = 0.0
        self._frames_to_write: list[np.ndarray] = []
        self._last_clip_path: Optional[str] = None

        self._last_shape: Optional[Tuple[int, int]] = None  # (w,h)
        self._last_fps: float = self.cfg.fps_fallback

    def _ensure_pre_buffer(self, fps: float) -> None:
        fps = fps if fps and fps > 0 else self.cfg.fps_fallback
        pre_len = int(round(self.cfg.pre_seconds * fps))
        pre_len = max(1, pre_len)
        if self._pre.maxlen != pre_len:
            old = list(self._pre)
            self._pre = deque(old[-pre_len:], maxlen=pre_len)

    def push_frame(self, frame_bgr: np.ndarray, fps: float) -> None:
        self._ensure_pre_buffer(fps)
        self._pre.append(frame_bgr.copy())
        self._last_shape = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        self._last_fps = fps if fps and fps > 0 else self.cfg.fps_fallback

        if self._recording:
            self._frames_to_write.append(frame_bgr.copy())
            if time.time() >= self._record_until_t:
                self._finalize()

    def can_trigger(self) -> bool:
        return time.time() >= self._cooldown_until_t and not self._recording

    def trigger(self, reason: str = "alarm") -> Optional[str]:
        if not self.can_trigger():
            return None

        # start recording: include pre-buffer first
        self._recording = True
        self._record_until_t = time.time() + float(self.cfg.post_seconds)
        self._cooldown_until_t = time.time() + float(self.cfg.cooldown_seconds)

        self._frames_to_write = [f.copy() for f in list(self._pre)]
        self._last_clip_path = None
        return None

    def _finalize(self) -> None:
        if not self._frames_to_write or self._last_shape is None:
            self._recording = False
            self._frames_to_write = []
            return

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        path = os.path.join(self.cfg.clips_dir, f"event_{ts}.mp4")
        w, h = self._last_shape
        fps = float(self._last_fps or self.cfg.fps_fallback)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        try:
            for fr in self._frames_to_write:
                if fr.shape[1] != w or fr.shape[0] != h:
                    fr = cv2.resize(fr, (w, h))
                writer.write(fr)
        finally:
            writer.release()

        self._last_clip_path = path
        self._recording = False
        self._frames_to_write = []

    def pop_last_clip_path(self) -> Optional[str]:
        p = self._last_clip_path
        self._last_clip_path = None
        return p

