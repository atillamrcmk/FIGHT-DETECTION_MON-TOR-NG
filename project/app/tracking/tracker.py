from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.utils.math_utils import bbox_center_xyxy, l2


@dataclass
class Track:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    conf: float
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    history_centers: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        return bbox_center_xyxy(self.xyxy)


class CentroidTracker:
    """
    Minimal CPU-friendly tracker:
    - assigns detections to existing tracks by nearest centroid (greedy)
    - creates new tracks for unmatched detections
    - drops tracks not updated for max_age frames
    """

    def __init__(self, max_distance_px: float = 80.0, max_age_frames: int = 30, min_hits: int = 2) -> None:
        self.max_distance_px = float(max_distance_px)
        self.max_age_frames = int(max_age_frames)
        self.min_hits = int(min_hits)

        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Dict]) -> List[Track]:
        # increment age / time_since_update
        for t in self.tracks.values():
            t.age += 1
            t.time_since_update += 1

        det_centers = [bbox_center_xyxy(d["xyxy"]) for d in detections]
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid].center for tid in track_ids]

        matches: List[Tuple[int, int, float]] = []  # (track_idx, det_idx, dist)
        if track_centers and det_centers:
            for ti, tc in enumerate(track_centers):
                for di, dc in enumerate(det_centers):
                    matches.append((ti, di, l2(tc, dc)))
            matches.sort(key=lambda x: x[2])

        used_tracks = set()
        used_dets = set()

        # greedy assignment
        for ti, di, dist in matches:
            if dist > self.max_distance_px:
                break
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            tid = track_ids[ti]
            d = detections[di]
            tr = self.tracks[tid]
            tr.xyxy = d["xyxy"]
            tr.conf = float(d.get("conf", 0.0))
            tr.keypoints_xy = d.get("keypoints_xy")
            tr.keypoints_conf = d.get("keypoints_conf")
            tr.hits += 1
            tr.time_since_update = 0
            tr.history_centers.append(tr.center)
            if len(tr.history_centers) > 200:
                tr.history_centers = tr.history_centers[-200:]

        # new tracks for unmatched dets
        for di, d in enumerate(detections):
            if di in used_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            tr = Track(
                track_id=tid,
                xyxy=d["xyxy"],
                conf=float(d.get("conf", 0.0)),
                keypoints_xy=d.get("keypoints_xy"),
                keypoints_conf=d.get("keypoints_conf"),
                hits=1,
                time_since_update=0,
            )
            tr.history_centers.append(tr.center)
            self.tracks[tid] = tr

        # remove stale tracks
        to_del = [tid for tid, tr in self.tracks.items() if tr.time_since_update > self.max_age_frames]
        for tid in to_del:
            del self.tracks[tid]

        # return active tracks
        active = list(self.tracks.values())
        return active

