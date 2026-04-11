from __future__ import annotations

import argparse
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import AppConfig
from app.analytics.aggression_analyzer import AggressionAnalyzer
from app.analytics.inactivity_analyzer import InactivityAnalyzer
from app.analytics.interaction_analyzer import InteractionAnalyzer
from app.analytics.movement_analyzer import MovementAnalyzer
from app.analytics.posture_analyzer import PostureAnalyzer
from app.detection.pose_detector import PoseDetector
from app.models.fight_classifier import FightClipClassifier, FightModelConfig
from app.risk.fight_risk_engine import FightRiskEngine
from app.risk.monitoring_risk_engine import MonitoringRiskEngine
from app.tracking.tracker import CentroidTracker


@dataclass
class FrameStats:
    frame_idx: int
    person_count: int
    risk_score: float
    level: str
    main_reason: str
    fight_model_score: float
    components: Dict[str, float]


def _scene_motion_score(frame_bgr: np.ndarray, prev_gray: Optional[np.ndarray]) -> Tuple[float, np.ndarray]:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 0.0, g
    diff = cv2.absdiff(g, prev_gray)
    mean = float(diff.mean())
    score = float(min(100.0, (mean / 15.0) * 100.0))
    return score, g


def _pct(x: int, total: int) -> float:
    return (100.0 * x / total) if total > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline debug report for a single video file.")
    ap.add_argument("--video", type=str, required=True, help="Path to video file")
    ap.add_argument("--mode", type=str, default="FIGHT", choices=["FIGHT", "MONITORING"])
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame (default: 1)")
    ap.add_argument("--max_frames", type=int, default=0, help="Stop after N processed frames (0=all)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO conf threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    args = ap.parse_args()

    path = os.path.abspath(args.video)
    if not os.path.exists(path):
        raise SystemExit(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3 or fps != fps:
        fps = 25.0

    detector = PoseDetector(conf=float(args.conf), iou=float(args.iou), imgsz=int(args.imgsz))
    tracker = CentroidTracker(max_distance_px=80.0, max_age_frames=30, min_hits=2)
    movement = MovementAnalyzer(movement_window=15)
    interaction = InteractionAnalyzer(interaction_distance_px=120.0, clustering_distance_px=110.0)
    inactivity = InactivityAnalyzer(inactivity_seconds=15.0)
    posture = PostureAnalyzer(collapse_angle_threshold=0.55)
    aggression = AggressionAnalyzer(self_harm_head_radius_px=60.0)

    fight_engine = FightRiskEngine(
        weights={
            "movement": 1.0,
            "agitation": 1.2,
            "interaction": 1.0,
            "clustering": 0.8,
            "scene_motion": 0.6,
            "bonus_fast_close_and_agitated": 12.0,
            "fight_model": 1.0,
        },
        warning_thr=40.0,
        alarm_thr=70.0,
    )
    monitor_engine = MonitoringRiskEngine(
        weights={
            "inactivity": 1.2,
            "agitation": 0.8,
            "posture": 1.0,
            "self_harm": 1.5,
            "collapse": 2.0,
        },
        warning_thr=40.0,
        alarm_thr=70.0,
    )

    fight_model = FightClipClassifier(FightModelConfig())
    clip_buf: deque[np.ndarray] = deque(maxlen=int(fight_model.cfg.clip_len))
    fight_model_score = 0.0
    cfg = AppConfig(mode=args.mode)

    prev_gray: Optional[np.ndarray] = None
    prev_centers_by_id: Dict[int, Tuple[float, float]] = {}
    prev_kpts_by_id: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    total = 0
    frames_read = 0
    warnings = 0
    alarms = 0
    direct_warn = 0
    direct_alarm = 0
    persons_ge2 = 0
    persons_ge3 = 0
    model_enabled = bool(fight_model.enabled)

    max_risk = FrameStats(0, 0, 0.0, "NORMAL", "n/a", 0.0, {})
    max_model = 0.0
    max_interaction = 0.0
    max_agitation = 0.0
    max_movement = 0.0
    max_clustering = 0.0
    frames_common2 = 0
    track_frame_counts: Counter[int] = Counter()

    reason_counts: Counter[str] = Counter()
    top_frames: List[FrameStats] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames_read += 1
            if int(args.stride) > 1 and (frames_read % int(args.stride)) != 0:
                continue

            total += 1
            if int(args.max_frames) > 0 and total > int(args.max_frames):
                break

            if args.mode == "FIGHT":
                clip_buf.append(frame)
                s = fight_model.predict_score_if_ready(clip_buf)
                if s is not None:
                    fight_model_score = float(s)

            dets = detector.detect(frame)
            detections = [
                {"xyxy": d.xyxy, "conf": float(d.conf), "keypoints_xy": d.keypoints_xy, "keypoints_conf": d.keypoints_conf}
                for d in dets
            ]
            tracks = tracker.update(detections)
            person_count = int(len(tracks))

            if person_count >= 2:
                persons_ge2 += 1
            if person_count >= 3:
                persons_ge3 += 1

            scene_motion, prev_gray = _scene_motion_score(frame, prev_gray)

            centers_now_by_id = {t.track_id: t.center for t in tracks}
            for tid in centers_now_by_id.keys():
                track_frame_counts[int(tid)] += 1
            if len(set(centers_now_by_id.keys()) & set(prev_centers_by_id.keys())) >= 2:
                frames_common2 += 1
            inter = interaction.compute_tracked(centers_now_by_id, prev_centers_by_id)
            prev_centers_by_id = centers_now_by_id
            max_interaction = max(max_interaction, float(inter.interaction_score))
            max_clustering = max(max_clustering, float(inter.clustering_score))

            movement_scores: List[float] = []
            agitation_scores: List[float] = []
            inactivity_scores: List[float] = []
            posture_scores: List[float] = []
            self_harm_scores: List[float] = []
            collapse_scores: List[float] = []
            still_seconds_list: List[float] = []

            for t in tracks:
                mv = movement.compute(t.history_centers)
                movement_scores.append(float(mv.movement_score))

                prev_k = prev_kpts_by_id.get(t.track_id)
                ag = aggression.compute(
                    t.keypoints_xy,
                    t.keypoints_conf,
                    prev_k[0] if prev_k else None,
                    prev_k[1] if prev_k else None,
                )
                agitation_scores.append(float(ag.pose_agitation_score))
                self_harm_scores.append(float(ag.self_harm_risk_score))

                ia = inactivity.compute(t.history_centers, fps=fps)
                inactivity_scores.append(float(ia.inactivity_score))
                still_seconds_list.append(float(ia.still_seconds))

                po = posture.compute(t.xyxy, t.keypoints_xy, t.keypoints_conf)
                posture_scores.append(float(po.posture_score))
                collapse_scores.append(float(po.collapse_score))

                if t.keypoints_xy is not None and t.keypoints_conf is not None:
                    prev_kpts_by_id[t.track_id] = (t.keypoints_xy.copy(), t.keypoints_conf.copy())

            movement_score = float(max(movement_scores) if movement_scores else 0.0)
            agitation_score = float(max(agitation_scores) if agitation_scores else 0.0)
            max_movement = max(max_movement, movement_score)
            max_agitation = max(max_agitation, agitation_score)
            inactivity_score = float(max(inactivity_scores) if inactivity_scores else 0.0)
            posture_score = float(max(posture_scores) if posture_scores else 0.0)
            self_harm_score = float(max(self_harm_scores) if self_harm_scores else 0.0)
            collapse_score = float(max(collapse_scores) if collapse_scores else 0.0)
            still_seconds_max = float(max(still_seconds_list) if still_seconds_list else 0.0)

            if args.mode == "FIGHT":
                bonus = (float(inter.interaction_score) > 55.0) and (agitation_score > 60.0)
                rr = fight_engine.compute(
                    movement_score=movement_score,
                    agitation_score=agitation_score,
                    interaction_score=float(inter.interaction_score),
                    clustering_score=float(inter.clustering_score),
                    scene_motion_score=float(scene_motion),
                    fight_model_score=float(fight_model_score),
                    bonus_flag=bool(bonus),
                    active_interactions=int(inter.active_interactions),
                )
                risk_score = float(rr.risk_score)
                level = str(rr.level)
                main_reason = str(rr.main_reason)
                components = dict(rr.components)

                # Mirror the pipeline "direct multi-signal" rule for quick diagnostics
                if person_count >= 2:
                    signals = 0
                    if fight_model.enabled and fight_model_score >= float(cfg.fight_direct_warning_model_thr):
                        signals += 1
                    if float(inter.interaction_score) >= float(cfg.fight_direct_warning_interaction_thr):
                        signals += 1
                    if float(agitation_score) >= float(cfg.fight_direct_warning_agitation_thr):
                        signals += 1
                    if float(movement_score) >= float(cfg.fight_direct_warning_movement_thr):
                        signals += 1
                    if float(inter.clustering_score) >= float(cfg.fight_direct_warning_clustering_thr):
                        signals += 1

                    if fight_model.enabled and fight_model_score >= float(cfg.fight_direct_alarm_model_thr):
                        direct_alarm += 1
                    elif signals >= int(cfg.fight_direct_min_signals_alarm):
                        direct_alarm += 1
                    elif signals >= int(cfg.fight_direct_min_signals_warning):
                        direct_warn += 1
            else:
                mr = monitor_engine.compute(
                    inactivity_score=inactivity_score,
                    agitation_score=agitation_score,
                    posture_score=posture_score,
                    self_harm_risk_score=self_harm_score,
                    collapse_score=collapse_score,
                    still_seconds_max=still_seconds_max,
                )
                risk_score = float(mr.risk_score)
                level = str(mr.level)
                main_reason = str(mr.main_reason)
                components = dict(mr.components)

            reason_counts[main_reason] += 1
            if level == "UYARI":
                warnings += 1
            elif level == "ALARM":
                alarms += 1

            if risk_score > max_risk.risk_score:
                max_risk = FrameStats(
                    frame_idx=frames_read,
                    person_count=person_count,
                    risk_score=risk_score,
                    level=level,
                    main_reason=main_reason,
                    fight_model_score=float(fight_model_score),
                    components={k: float(v) for k, v in components.items()},
                )

            max_model = max(max_model, float(fight_model_score))

            top_frames.append(
                FrameStats(
                    frame_idx=frames_read,
                    person_count=person_count,
                    risk_score=risk_score,
                    level=level,
                    main_reason=main_reason,
                    fight_model_score=float(fight_model_score),
                    components={k: float(v) for k, v in components.items()},
                )
            )

    finally:
        cap.release()

    top_frames.sort(key=lambda s: s.risk_score, reverse=True)
    top_frames = top_frames[:5]

    print("=== Debug Report ===")
    print("video:", path)
    print("mode:", args.mode)
    print("fps:", f"{fps:0.2f}", "| stride:", int(args.stride), "| processed_frames:", total, "| read_frames:", frames_read)
    print("fight_model_enabled:", model_enabled, "| max_model_score:", f"{max_model:0.1f}")
    if args.mode == "FIGHT":
        print(
            "max_movement:",
            f"{max_movement:0.1f}",
            "| max_agitation:",
            f"{max_agitation:0.1f}",
            "| max_interaction:",
            f"{max_interaction:0.1f}",
            "| max_clustering:",
            f"{max_clustering:0.1f}",
        )
    print(
        "persons>=2 frames:",
        f"{persons_ge2}/{total} ({_pct(persons_ge2,total):0.1f}%)",
        "| persons>=3 frames:",
        f"{persons_ge3}/{total} ({_pct(persons_ge3,total):0.1f}%)",
    )
    if args.mode == "FIGHT":
        print("frames_with_common_ids>=2:", f"{frames_common2}/{total} ({_pct(frames_common2,total):0.1f}%)")
        if track_frame_counts:
            most = track_frame_counts.most_common(5)
            print("top_track_persistence (frames):", ", ".join([f"id{tid}={cnt}" for tid, cnt in most]))
    print("UYARI frames:", f"{warnings}/{total} ({_pct(warnings,total):0.1f}%)", "| ALARM frames:", f"{alarms}/{total} ({_pct(alarms,total):0.1f}%)")
    if args.mode == "FIGHT":
        print("direct_rule would warn:", f"{direct_warn}/{total} ({_pct(direct_warn,total):0.1f}%)", "| direct_rule would alarm:", f"{direct_alarm}/{total} ({_pct(direct_alarm,total):0.1f}%)")
    print("max_risk:", f"{max_risk.risk_score:0.1f}", "| level:", max_risk.level, "| frame:", max_risk.frame_idx, "| persons:", max_risk.person_count, "| reason:", max_risk.main_reason)

    print("\nTop reasons:")
    for k, v in reason_counts.most_common(8):
        print(f"  - {k}: {v} ({_pct(v,total):0.1f}%)")

    print("\nTop 5 frames by risk:")
    for s in top_frames:
        print(f"  - f{s.frame_idx} risk={s.risk_score:0.1f} level={s.level} persons={s.person_count} model={s.fight_model_score:0.1f} reason={s.main_reason}")


if __name__ == "__main__":
    main()
