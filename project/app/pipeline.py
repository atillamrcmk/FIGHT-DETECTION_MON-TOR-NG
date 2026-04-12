from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import AppConfig
from app.detection.pose_detector import PoseDetector
from app.tracking.tracker import CentroidTracker, Track
from app.analytics.movement_analyzer import MovementAnalyzer
from app.analytics.interaction_analyzer import InteractionAnalyzer
from app.analytics.inactivity_analyzer import InactivityAnalyzer
from app.analytics.posture_analyzer import PostureAnalyzer
from app.analytics.aggression_analyzer import AggressionAnalyzer
from app.risk.fight_risk_engine import FightRiskEngine, FightRiskResult
from app.risk.monitoring_risk_engine import MonitoringRiskEngine, MonitoringRiskResult
from app.alerts.clip_recorder import ClipRecorder, ClipRecorderConfig
from app.alerts.snapshot_saver import SnapshotSaver, SnapshotSaverConfig
from app.utils.video_io import VideoSource
from app.utils.logger import JsonEventLogger, now_iso
from app.utils.drawing import color_for_id, draw_bbox, draw_hud_panel, draw_skeleton, level_color
from app.models.fight_classifier import FightClipClassifier, FightModelConfig


@dataclass
class FrameOutcome:
    frame_bgr: np.ndarray
    mode: str
    risk_score: float
    level: str
    person_count: int
    active_interactions: int
    main_reason: str
    components: Dict[str, float]
    clip_path: Optional[str]


class Pipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.cfg.validate()
        self.mode = self.cfg.mode.upper().strip()

        self.pose = PoseDetector(
            model_name=self.cfg.detector.model_name,
            conf=self.cfg.detector.conf,
            iou=self.cfg.detector.iou,
            imgsz=self.cfg.detector.imgsz,
            device=self.cfg.detector.device,
        )

        self.tracker = CentroidTracker(
            max_distance_px=self.cfg.tracking.max_distance_px,
            max_age_frames=self.cfg.tracking.max_age_frames,
            min_hits=self.cfg.tracking.min_hits,
        )

        self.movement = MovementAnalyzer(movement_window=self.cfg.features.movement_window)
        self.interaction = InteractionAnalyzer(
            interaction_distance_px=self.cfg.features.interaction_distance_px,
            clustering_distance_px=self.cfg.features.clustering_distance_px,
        )
        self.inactivity = InactivityAnalyzer(inactivity_seconds=self.cfg.features.inactivity_seconds)
        self.posture = PostureAnalyzer(collapse_angle_threshold=self.cfg.features.collapse_angle_threshold)
        self.aggression = AggressionAnalyzer(self_harm_head_radius_px=self.cfg.features.self_harm_head_radius_px)

        self.fight_engine = FightRiskEngine(
            weights={
                "movement": self.cfg.fight_weights.movement,
                "agitation": self.cfg.fight_weights.agitation,
                "interaction": self.cfg.fight_weights.interaction,
                "clustering": self.cfg.fight_weights.clustering,
                "scene_motion": self.cfg.fight_weights.scene_motion,
                "bonus_fast_close_and_agitated": self.cfg.fight_weights.bonus_fast_close_and_agitated,
                "fight_model": self.cfg.fight_weights.fight_model,
            },
            warning_thr=self.cfg.thresholds.warning,
            alarm_thr=self.cfg.thresholds.alarm,
        )
        self.monitor_engine = MonitoringRiskEngine(
            weights={
                "inactivity": self.cfg.monitoring_weights.inactivity,
                "agitation": self.cfg.monitoring_weights.agitation,
                "posture": self.cfg.monitoring_weights.posture,
                "self_harm": self.cfg.monitoring_weights.self_harm,
                "collapse": self.cfg.monitoring_weights.collapse,
            },
            warning_thr=self.cfg.thresholds.warning,
            alarm_thr=self.cfg.thresholds.alarm,
        )

        self.logger = JsonEventLogger(self.cfg.logs_dir)
        self.clip = ClipRecorder(
            ClipRecorderConfig(
                clips_dir=self.cfg.clips_dir,
                pre_seconds=self.cfg.clip.pre_seconds,
                post_seconds=self.cfg.clip.post_seconds,
                cooldown_seconds=self.cfg.clip.cooldown_seconds,
                fps_fallback=self.cfg.clip.fps_fallback,
            )
        )

        self.snapshot: Optional[SnapshotSaver] = None
        if self.cfg.snapshot.enabled:
            self.snapshot = SnapshotSaver(
                SnapshotSaverConfig(
                    snapshots_dir=self.cfg.snapshot.snapshots_dir,
                    cooldown_seconds=self.cfg.snapshot.cooldown_seconds,
                    jpeg_quality=self.cfg.snapshot.jpeg_quality,
                    show_preview_window=self.cfg.snapshot.show_preview_window,
                    preview_max_width=self.cfg.snapshot.preview_max_width,
                    preview_window_name=self.cfg.snapshot.preview_window_name,
                )
            )

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_centers_by_id: Dict[int, Tuple[float, float]] = {}
        self._prev_kpts_by_id: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # fight model buffer (used in FIGHT mode only)
        self._clip_buf = deque(maxlen=16)
        self._fight_model = FightClipClassifier(FightModelConfig())
        self._fight_model_score = 0.0

        # interactive UI state
        self._paused: bool = False
        self._zoom: float = 1.0
        self._pan_x: int = 0
        self._pan_y: int = 0

        # Mouse/seek state (file sources only)
        self._pending_seek_frame: Optional[int] = None
        self._trackbar_internal_set: bool = False
        self._dragging_pan: bool = False
        self._last_mouse_xy: Optional[Tuple[int, int]] = None
        self._force_process_once: bool = False

        # Auto-resize processing frames for usability
        self._auto_resize_set: bool = False
        self._resize_to: Optional[Tuple[int, int]] = self.cfg.resize_to

        # Last computed frame-level state (so pause/seek UI doesn't zero-out)
        self._last_risk_score: float = 0.0
        self._last_level: str = "NORMAL"
        self._last_main_reason: str = "n/a"
        self._last_components: Dict[str, float] = {}
        self._fight_model_score_valid: bool = False

    def _reset_temporal_state_after_seek(self) -> None:
        self._prev_gray = None
        self._prev_centers_by_id = {}
        self._prev_kpts_by_id = {}
        self._clip_buf.clear()
        self._fight_model_score = 0.0
        self._fight_model_score_valid = False

    def _file_eof_overlay_loop(self, last_frame: np.ndarray, cap: cv2.VideoCapture) -> bool:
        """
        After file EOF when not looping: show last frame until r (restart) or q/ESC (quit).
        Returns True if should quit main loop.
        """
        h, w = last_frame.shape[:2]
        msg = "END | r=restart | q=quit"
        while True:
            vis = last_frame.copy()
            cv2.rectangle(vis, (0, 0), (w, 36), (20, 20, 20), -1)
            cv2.putText(vis, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow(self.cfg.window_name, vis)
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord("q")):
                return True
            if key == ord("r"):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception:
                    pass
                self._reset_temporal_state_after_seek()
                return False

    def _scene_motion_score(self, frame_bgr: np.ndarray) -> float:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = g
            return 0.0
        diff = cv2.absdiff(g, self._prev_gray)
        self._prev_gray = g
        # normalize mean diff to 0-100 band
        mean = float(diff.mean())
        return float(min(100.0, (mean / 15.0) * 100.0))

    def _detections_from_pose(self, dets) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in dets:
            out.append(
                {
                    "xyxy": d.xyxy,
                    "conf": d.conf,
                    "keypoints_xy": d.keypoints_xy,
                    "keypoints_conf": d.keypoints_conf,
                }
            )
        return out

    def _apply_view(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        View transform for UI only: zoom + pan.
        Keeps processing on original frame.
        """
        z = float(self._zoom)
        if z <= 1.001:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        crop_w = max(32, int(round(w / z)))
        crop_h = max(32, int(round(h / z)))

        cx = w // 2 + int(self._pan_x)
        cy = h // 2 + int(self._pan_y)

        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return frame_bgr
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def _handle_keys(self, key: int, cap: cv2.VideoCapture, fps: float) -> bool:
        """
        Returns True if should quit.
        Controls:
        - q / ESC: quit
        - SPACE: pause/resume
        - r: restart (file only)
        - + / = : zoom in
        - - / _ : zoom out
        - 0: reset zoom/pan
        - w/a/s/d: pan (when zoomed)
        """
        if key in (27, ord("q")):
            return True
        if key == ord(" "):
            self._paused = not self._paused
            return False
        if key == ord("r"):
            # restart for file sources (RTSP seek may fail)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass
            self._reset_temporal_state_after_seek()
            return False
        if key in (ord("+"), ord("=")):
            self._zoom = min(4.0, self._zoom * 1.15)
            return False
        if key in (ord("-"), ord("_")):
            self._zoom = max(1.0, self._zoom / 1.15)
            if self._zoom <= 1.001:
                self._zoom = 1.0
                self._pan_x = 0
                self._pan_y = 0
            return False
        if key == ord("0"):
            self._zoom = 1.0
            self._pan_x = 0
            self._pan_y = 0
            return False

        if self._zoom > 1.001:
            step = 30
            if key == ord("a"):
                self._pan_x -= step
            elif key == ord("d"):
                self._pan_x += step
            elif key == ord("w"):
                self._pan_y -= step
            elif key == ord("s"):
                self._pan_y += step

        return False

    def _setup_mouse_and_trackbar(self, *, cap: cv2.VideoCapture, source_type: str) -> None:
        if not self.cfg.viz.show:
            return

        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)

        def on_mouse(event: int, x: int, y: int, flags: int, _userdata: Any) -> None:
            # Left click: pause/resume
            if event == cv2.EVENT_LBUTTONDOWN:
                self._paused = not self._paused
                return

            # Mouse wheel: zoom in/out
            if event == cv2.EVENT_MOUSEWHEEL:
                # flags sign indicates direction
                delta = 1 if flags > 0 else -1
                if delta > 0:
                    self._zoom = min(4.0, self._zoom * 1.15)
                else:
                    self._zoom = max(1.0, self._zoom / 1.15)
                    if self._zoom <= 1.001:
                        self._zoom = 1.0
                        self._pan_x = 0
                        self._pan_y = 0
                return

            # Right-button drag: pan (only when zoomed)
            if event == cv2.EVENT_RBUTTONDOWN:
                self._dragging_pan = True
                self._last_mouse_xy = (x, y)
                return
            if event == cv2.EVENT_RBUTTONUP:
                self._dragging_pan = False
                self._last_mouse_xy = None
                return
            if event == cv2.EVENT_MOUSEMOVE and self._dragging_pan and self._zoom > 1.001 and self._last_mouse_xy is not None:
                lx, ly = self._last_mouse_xy
                dx = x - lx
                dy = y - ly
                self._pan_x += int(dx)
                self._pan_y += int(dy)
                self._last_mouse_xy = (x, y)

        try:
            cv2.setMouseCallback(self.cfg.window_name, on_mouse)
        except Exception:
            pass

        if source_type != "file":
            return

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            frame_count = 0
        if frame_count <= 0:
            return

        def on_seek(pos: int) -> None:
            if self._trackbar_internal_set:
                return
            self._pending_seek_frame = int(pos)

        try:
            cv2.createTrackbar("Seek", self.cfg.window_name, 0, max(1, frame_count - 1), on_seek)
        except Exception:
            pass

    def _maybe_set_auto_resize(self, frame_bgr: np.ndarray) -> None:
        if self._auto_resize_set:
            return
        self._auto_resize_set = True

        if self._resize_to is not None:
            return
        if not self.cfg.viz.auto_resize_enabled:
            return

        h, w = frame_bgr.shape[:2]
        mw = int(self.cfg.viz.auto_resize_max_width)
        mh = int(self.cfg.viz.auto_resize_max_height)
        if mw <= 0 or mh <= 0:
            return
        if w <= mw and h <= mh:
            return

        scale = min(mw / float(w), mh / float(h), 1.0)
        new_w = max(320, int(round(w * scale)))
        new_h = max(240, int(round(h * scale)))
        self._resize_to = (new_w, new_h)

    def run(self) -> None:
        os.makedirs(self.cfg.logs_dir, exist_ok=True)
        os.makedirs(self.cfg.clips_dir, exist_ok=True)
        if self.cfg.snapshot.enabled:
            os.makedirs(self.cfg.snapshot.snapshots_dir, exist_ok=True)

        vs = VideoSource({**self.cfg.VIDEO_SOURCE, "fps_fallback": self.cfg.clip.fps_fallback})
        vs.open()
        try:
            cap = vs.cap
            if cap is None:
                raise RuntimeError("Video source not initialized.")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1e-3 or fps != fps:
                fps = float(self.cfg.clip.fps_fallback)

            frame_idx = 0
            last_frame: Optional[np.ndarray] = None
            source_type = (self.cfg.VIDEO_SOURCE.get("type") or "").lower().strip()
            self._setup_mouse_and_trackbar(cap=cap, source_type=source_type)

            while True:
                # apply pending seek (file only)
                if self._pending_seek_frame is not None and source_type == "file":
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(self._pending_seek_frame))
                    except Exception:
                        pass
                    frame_idx = int(self._pending_seek_frame)
                    last_frame = None
                    self._pending_seek_frame = None
                    self._reset_temporal_state_after_seek()
                    # If user is paused and scrubbing, still compute metrics for the new frame once.
                    self._force_process_once = True

                snapshot_native: Optional[np.ndarray] = None
                if not self._paused or last_frame is None:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if source_type == "file" and self.cfg.loop_file_on_end:
                            try:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            except Exception:
                                pass
                            self._reset_temporal_state_after_seek()
                            continue
                        if (
                            source_type == "file"
                            and last_frame is not None
                            and self.cfg.viz.show
                            and not self._file_eof_overlay_loop(last_frame, cap)
                        ):
                            continue
                        break
                    snapshot_native = frame.copy()
                    last_frame = frame
                    frame_idx += 1
                else:
                    frame = last_frame

                process = (not self._paused) or self._force_process_once

                # Auto resize (applies to processing, not just display) to keep windows usable.
                if process:
                    self._maybe_set_auto_resize(frame)
                    if self._resize_to is not None:
                        frame = cv2.resize(frame, self._resize_to)
                        last_frame = frame

                if process:
                    # Only push frames to clip recorder when playing (avoid side-effects while scrubbing)
                    if not self._paused:
                        self.clip.push_frame(frame, fps=fps)

                    if self.mode == "FIGHT":
                        # Keep model buffer moving for live playback. While paused+scrubbing we still append,
                        # but the score will only become valid after enough consecutive frames.
                        self._clip_buf.append(frame)
                        s = self._fight_model.predict_score_if_ready(self._clip_buf)
                        if s is not None:
                            self._fight_model_score = float(s)
                            self._fight_model_score_valid = True

                    dets = self.pose.detect(frame)
                    tracks = self.tracker.update(self._detections_from_pose(dets))

                    scene_motion = self._scene_motion_score(frame)

                    # interaction/clustering using track-id stability across frames
                    centers_now_by_id = {t.track_id: t.center for t in tracks}
                    inter = self.interaction.compute_tracked(centers_now_by_id, self._prev_centers_by_id)
                    self._prev_centers_by_id = centers_now_by_id

                    # per-track compute, then aggregate to frame-level
                    movement_scores = []
                    agitation_scores = []
                    inactivity_scores = []
                    posture_scores = []
                    self_harm_scores = []
                    collapse_scores = []
                    still_seconds_list = []

                    for t in tracks:
                        mv = self.movement.compute(t.history_centers)
                        movement_scores.append(mv.movement_score)

                        prev_k = self._prev_kpts_by_id.get(t.track_id)
                        ag = self.aggression.compute(
                            t.keypoints_xy,
                            t.keypoints_conf,
                            prev_k[0] if prev_k else None,
                            prev_k[1] if prev_k else None,
                        )
                        agitation_scores.append(ag.pose_agitation_score)
                        self_harm_scores.append(ag.self_harm_risk_score)

                        ia = self.inactivity.compute(t.history_centers, fps=fps)
                        inactivity_scores.append(ia.inactivity_score)
                        still_seconds_list.append(ia.still_seconds)

                        po = self.posture.compute(t.xyxy, t.keypoints_xy, t.keypoints_conf)
                        posture_scores.append(po.posture_score)
                        collapse_scores.append(po.collapse_score)

                        # store last kpts
                        if t.keypoints_xy is not None and t.keypoints_conf is not None:
                            self._prev_kpts_by_id[t.track_id] = (t.keypoints_xy.copy(), t.keypoints_conf.copy())

                    # aggregate (simple max to focus on worst-case)
                    movement_score = float(max(movement_scores) if movement_scores else 0.0)
                    agitation_score = float(max(agitation_scores) if agitation_scores else 0.0)
                    inactivity_score = float(max(inactivity_scores) if inactivity_scores else 0.0)
                    posture_score = float(max(posture_scores) if posture_scores else 0.0)
                    self_harm_score = float(max(self_harm_scores) if self_harm_scores else 0.0)
                    collapse_score = float(max(collapse_scores) if collapse_scores else 0.0)
                    still_seconds_max = float(max(still_seconds_list) if still_seconds_list else 0.0)

                    if self.mode == "FIGHT":
                        # Bonus rule tuned for demo sensitivity (still heuristic)
                        bonus = ((inter.interaction_score > 35.0) and (agitation_score > 55.0)) or (
                            (movement_score > 70.0) and (agitation_score > 55.0)
                        )
                        rr: FightRiskResult = self.fight_engine.compute(
                            movement_score=movement_score,
                            agitation_score=agitation_score,
                            interaction_score=inter.interaction_score,
                            clustering_score=inter.clustering_score,
                            scene_motion_score=scene_motion,
                            fight_model_score=float(self._fight_model_score),
                            bonus_flag=bonus,
                            active_interactions=inter.active_interactions,
                        )
                        risk_score = rr.risk_score
                        level = rr.level
                        main_reason = rr.main_reason
                        components = rr.components

                        # Direct rule helpers for 1v1 fights
                        if len(tracks) >= 2:
                            if self._fight_model.enabled and self._fight_model_score >= self.cfg.fight_direct_alarm_model_thr:
                                level = "ALARM"
                                main_reason = "direct_model_alarm"
                                risk_score = max(float(risk_score), float(self.fight_engine.alarm_thr))
                                components["direct_model_alarm"] = float(self._fight_model_score)
                            else:
                                signals = 0
                                if self._fight_model.enabled and self._fight_model_score >= self.cfg.fight_direct_warning_model_thr:
                                    signals += 1
                                if float(inter.interaction_score) >= float(self.cfg.fight_direct_warning_interaction_thr):
                                    signals += 1
                                if float(agitation_score) >= float(self.cfg.fight_direct_warning_agitation_thr):
                                    signals += 1
                                if float(movement_score) >= float(self.cfg.fight_direct_warning_movement_thr):
                                    signals += 1
                                if float(inter.clustering_score) >= float(self.cfg.fight_direct_warning_clustering_thr):
                                    signals += 1

                                if level != "ALARM" and signals >= int(self.cfg.fight_direct_min_signals_alarm):
                                    level = "ALARM"
                                    main_reason = "direct_multi_signal_alarm"
                                    risk_score = max(float(risk_score), float(self.fight_engine.alarm_thr))
                                    components["direct_multi_signal_alarm"] = float(signals)
                                elif level == "NORMAL" and signals >= int(self.cfg.fight_direct_min_signals_warning):
                                    level = "UYARI"
                                    main_reason = "direct_multi_signal_warning"
                                    risk_score = max(float(risk_score), float(self.fight_engine.warning_thr))
                                    components["direct_multi_signal_warning"] = float(signals)
                    else:
                        mr: MonitoringRiskResult = self.monitor_engine.compute(
                            inactivity_score=inactivity_score,
                            agitation_score=agitation_score,
                            posture_score=posture_score,
                            self_harm_risk_score=self_harm_score,
                            collapse_score=collapse_score,
                            still_seconds_max=still_seconds_max,
                        )
                        risk_score = mr.risk_score
                        level = mr.level
                        main_reason = mr.main_reason
                        components = mr.components

                    # Cache last computed state for pause/scrub UI
                    self._last_risk_score = float(risk_score)
                    self._last_level = str(level)
                    self._last_main_reason = str(main_reason)
                    self._last_components = {str(k): float(v) for k, v in (components or {}).items()}

                    self._force_process_once = False
                else:
                    # paused: keep last computed values (do not zero-out)
                    tracks = list(self.tracker.tracks.values())
                    scene_motion = 0.0
                    risk_score = float(self._last_risk_score)
                    level = str(self._last_level)
                    main_reason = str(self._last_main_reason)
                    components = dict(self._last_components)

                clip_path = None
                if (not self._paused) and level == "ALARM" and self.clip.can_trigger():
                    self.clip.trigger(reason=main_reason)

                clip_path = self.clip.pop_last_clip_path()

                snapshot_path: Optional[str] = None
                if (
                    self.snapshot is not None
                    and (not self._paused)
                    and level in ("UYARI", "ALARM")
                    and snapshot_native is not None
                ):
                    snapshot_path = self.snapshot.maybe_save(
                        snapshot_native,
                        level=level,
                        risk_score=float(risk_score),
                        main_reason=str(main_reason),
                        frame_idx=frame_idx,
                    )

                # logging on UYARI/ALARM (MVP)
                if (not self._paused) and level in ("UYARI", "ALARM"):
                    event = {
                        "timestamp": now_iso(),
                        "mode": self.mode,
                        "risk_score": float(risk_score),
                        "level": level,
                        "person_count": int(len(tracks)),
                        "active_interactions": int(inter.active_interactions),
                        "main_reason": str(main_reason),
                        "clip_path": clip_path,
                        "snapshot_path": snapshot_path,
                        "components": {k: float(v) for k, v in components.items()},
                    }
                    self.logger.log_event(event)

                if self.cfg.viz.show:
                    vis = self._apply_view(frame.copy())
                    # draw tracks
                    for t in tracks[: self.cfg.viz.max_skeleton_people]:
                        c = color_for_id(t.track_id)
                        if self.cfg.viz.draw_boxes:
                            draw_bbox(vis, t.xyxy, f"ID {t.track_id}", c)
                        if self.cfg.viz.draw_pose and t.keypoints_xy is not None and t.keypoints_conf is not None:
                            draw_skeleton(vis, t.keypoints_xy, t.keypoints_conf, c)

                    if self.cfg.viz.draw_panel:
                        lines = [
                            (f"MOD: {self.mode}", (220, 220, 220)),
                            (f"KISI: {len(tracks)}", (220, 220, 220)),
                            (f"DURUM: {level}", level_color(level)),
                            (f"NEDEN: {main_reason}", (200, 200, 200)),
                            (f"PAUSE: {'EVET' if self._paused else 'HAYIR'}", (170, 170, 170)),
                            (f"ZOOM: {self._zoom:0.2f}x", (170, 170, 170)),
                        ]
                        if self.mode == "FIGHT":
                            if self._fight_model.enabled and self._fight_model_score_valid:
                                lines.append((f"MODEL(FIGHT): {self._fight_model_score:5.1f}", (170, 170, 170)))
                            elif self._fight_model.enabled:
                                lines.append(("MODEL(FIGHT): ...", (170, 170, 170)))
                        if self.cfg.debug:
                            # top few components
                            for k, v in sorted(components.items(), key=lambda kv: kv[1], reverse=True)[:6]:
                                lines.append((f"{k}: {v:5.1f}", (170, 170, 170)))
                        vis = draw_hud_panel(
                            vis,
                            panel_width=self.cfg.viz.panel_width,
                            title="Risk Paneli",
                            lines=lines,
                            risk_score=float(risk_score),
                            level=str(level),
                        )

                    cv2.imshow(self.cfg.window_name, vis)

                    # Update seek trackbar with current position (file sources)
                    if source_type == "file":
                        try:
                            self._trackbar_internal_set = True
                            cv2.setTrackbarPos("Seek", self.cfg.window_name, max(0, int(frame_idx)))
                        except Exception:
                            pass
                        finally:
                            self._trackbar_internal_set = False
                    wait_ms = 50 if self._paused else 1
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key != 255:
                        if self._handle_keys(key, cap, fps):
                            break
        finally:
            vs.close()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

