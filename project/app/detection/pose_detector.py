from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PoseDetection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    keypoints_xy: np.ndarray  # (K,2)
    keypoints_conf: np.ndarray  # (K,)


class PoseDetector:
    """
    Ultralytics YOLO Pose wrapper.

    Output is normalized for downstream modules:
    - bbox xyxy in pixels
    - keypoints (K=17) in pixels + confidence
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(model_name)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device  # None => auto

    def detect(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []
        if r0.keypoints is None:
            return []

        boxes_xyxy = r0.boxes.xyxy
        boxes_conf = r0.boxes.conf

        # Ultralytics: keypoints.xy (N,K,2), keypoints.conf (N,K)
        kxy = getattr(r0.keypoints, "xy", None)
        kcf = getattr(r0.keypoints, "conf", None)
        if kxy is None or kcf is None:
            # Older versions may expose .data (N,K,3) => x,y,conf
            data = getattr(r0.keypoints, "data", None)
            if data is None:
                return []
            d = data.detach().cpu().numpy()
            kxy_np = d[..., :2]
            kcf_np = d[..., 2]
        else:
            kxy_np = kxy.detach().cpu().numpy()
            kcf_np = kcf.detach().cpu().numpy()

        bxyxy_np = boxes_xyxy.detach().cpu().numpy()
        bconf_np = boxes_conf.detach().cpu().numpy()

        dets: List[PoseDetection] = []
        n = min(len(bxyxy_np), len(kxy_np))
        for i in range(n):
            xyxy = tuple(float(v) for v in bxyxy_np[i].tolist())
            conf = float(bconf_np[i])
            keypoints_xy = np.asarray(kxy_np[i], dtype=np.float32)
            keypoints_conf = np.asarray(kcf_np[i], dtype=np.float32)
            dets.append(
                PoseDetection(
                    xyxy=xyxy,
                    conf=conf,
                    keypoints_xy=keypoints_xy,
                    keypoints_conf=keypoints_conf,
                )
            )
        return dets

