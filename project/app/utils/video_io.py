from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Tuple

import cv2


@dataclass
class FramePacket:
    frame_bgr: Any
    frame_idx: int
    timestamp_s: float
    fps: float


class VideoSource:
    def __init__(self, source_cfg: Dict[str, Any]) -> None:
        self.source_cfg = source_cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps_fallback = float(source_cfg.get("fps_fallback", 25.0))

    def open(self) -> None:
        t = (self.source_cfg.get("type") or "").lower().strip()
        if t == "file":
            path = str(self.source_cfg["path"])
            # Normalize/absolutize to reduce Windows path issues
            path = os.path.expandvars(os.path.expanduser(path))
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            path = path.replace("/", os.sep)

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                # FFMPEG backend fallback (often helps with AVI codecs)
                cap.release()
                cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            self.cap = cap
        elif t == "webcam":
            index = int(self.source_cfg.get("index", 0))
            self.cap = cv2.VideoCapture(index)
        elif t == "rtsp":
            url = self.source_cfg["url"]
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            self.cap = cap
        else:
            raise ValueError("VIDEO_SOURCE.type must be 'file', 'webcam', or 'rtsp'")

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Could not open video source.")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def read_frames(self) -> Generator[FramePacket, None, None]:
        if self.cap is None:
            raise RuntimeError("VideoSource not opened.")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3 or fps != fps:
            fps = self._fps_fallback

        frame_idx = 0
        t0 = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            timestamp_s = (frame_idx / fps) if fps > 0 else (time.time() - t0)
            yield FramePacket(frame_bgr=frame, frame_idx=frame_idx, timestamp_s=timestamp_s, fps=fps)
            frame_idx += 1

