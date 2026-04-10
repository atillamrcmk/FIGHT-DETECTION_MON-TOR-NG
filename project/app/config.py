from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Thresholds:
    warning: float = 40.0
    alarm: float = 70.0


@dataclass(frozen=True)
class ClipConfig:
    pre_seconds: float = 3.0
    post_seconds: float = 5.0
    cooldown_seconds: float = 10.0
    fps_fallback: float = 25.0


@dataclass(frozen=True)
class SnapshotConfig:
    """UYARI/ALARM aninda videonun resize/overlay oncesi ham karesini JPEG olarak kaydeder."""

    enabled: bool = True
    snapshots_dir: str = "data/snapshots"
    cooldown_seconds: float = 3.0
    jpeg_quality: int = 92
    show_preview_window: bool = True
    preview_max_width: int = 1280
    preview_window_name: str = "Uyari - orijinal kare"


@dataclass(frozen=True)
class VisualizationConfig:
    show: bool = True
    draw_boxes: bool = True
    draw_pose: bool = True
    draw_tracks: bool = True
    draw_panel: bool = True
    panel_width: int = 360
    max_skeleton_people: int = 20


@dataclass(frozen=True)
class DetectorConfig:
    model_name: str = "yolov8n-pose.pt"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: Optional[str] = None  # None -> auto (cuda if available)


@dataclass(frozen=True)
class TrackingConfig:
    max_distance_px: float = 80.0
    max_age_frames: int = 30
    min_hits: int = 2


@dataclass(frozen=True)
class FeatureConfig:
    history_size: int = 50  # per-track history
    movement_window: int = 15
    interaction_distance_px: float = 120.0
    clustering_distance_px: float = 110.0
    collapse_angle_threshold: float = 0.55  # bbox aspect ratio threshold for "lying-ish"
    inactivity_seconds: float = 15.0
    self_harm_head_radius_px: float = 60.0


@dataclass(frozen=True)
class FightWeights:
    movement: float = 1.0
    agitation: float = 1.2
    interaction: float = 1.0
    clustering: float = 0.8
    scene_motion: float = 0.6
    bonus_fast_close_and_agitated: float = 12.0


@dataclass(frozen=True)
class MonitoringWeights:
    inactivity: float = 1.2
    agitation: float = 0.8
    posture: float = 1.0
    self_harm: float = 1.5
    collapse: float = 2.0


@dataclass(frozen=True)
class AppConfig:
    # MODE: "FIGHT" or "MONITORING"
    mode: str = "FIGHT"

    # Video source:
    # - file: {"type":"file","path":"data/input/sample.mp4"}
    # - rtsp: {"type":"rtsp","url":"rtsp://user:pass@ip/stream"}
    VIDEO_SOURCE = {"type": "file", "path": "data/input/0_DzLlklZa0_6.avi"}

    thresholds: Thresholds = field(default_factory=Thresholds)
    clip: ClipConfig = field(default_factory=ClipConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fight_weights: FightWeights = field(default_factory=FightWeights)
    monitoring_weights: MonitoringWeights = field(default_factory=MonitoringWeights)

    debug: bool = True

    # Output paths (relative to project root)
    logs_dir: str = "data/logs"
    clips_dir: str = "data/clips"

    # UI
    window_name: str = "Prison Video Analytics MVP"
    resize_to: Optional[Tuple[int, int]] = None  # (w,h) or None

    # File source only: True = video bitince basa sar (pencere kapanmaz). False = son karede bekle (r/q).
    loop_file_on_end: bool = True

    def validate(self) -> None:
        mode = self.mode.upper().strip()
        if mode not in ("FIGHT", "MONITORING"):
            raise ValueError(f"Invalid mode: {self.mode}. Use 'FIGHT' or 'MONITORING'.")
