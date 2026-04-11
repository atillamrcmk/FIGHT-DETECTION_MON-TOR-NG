# Copilot Instructions for Fight Detection Monitoring

- Work from the `project` folder. The runtime entrypoint is `python -m app.main` and the UI is launched from `project/app/main.py`.
- Configuration is centralized in `project/app/config.py` via `AppConfig`. Important runtime settings are:
  - `mode` (`FIGHT` or `MONITORING`)
  - `VIDEO_SOURCE` (`file`, `webcam`, or `rtsp`)
  - output directories: `logs_dir`, `clips_dir`, `snapshot` settings
- The core processing pipeline lives in `project/app/pipeline.py`.
  - `Pipeline.run()` opens the video source, reads frames, runs pose detection, tracking, feature extraction, and risk scoring.
  - The pipeline writes event logs, clip files, and optional alert snapshots.
- Key subsystem boundaries:
  - Detection: `project/app/detection/pose_detector.py`
  - Tracking: `project/app/tracking/tracker.py`
  - Feature analytics: `project/app/analytics/*.py`
  - Risk scoring: `project/app/risk/fight_risk_engine.py` and `project/app/risk/monitoring_risk_engine.py`
  - Alerts: `project/app/alerts/clip_recorder.py`, `project/app/alerts/snapshot_saver.py`
- The UI dashboard is in `project/app/ui/dashboard.py` and constructs `AppConfig` for the selected source. Note it uses `object.__setattr__` to mutate the frozen config for `VIDEO_SOURCE`.
- Video input logic is in `project/app/utils/video_io.py`. Use absolute file paths on Windows. Supported source types:
  - `file`: `path`
  - `webcam`: `index`
  - `rtsp`: `url`
- Behavior and state details to preserve:
  - paused frames reuse the last frame and keep risk state stable
  - `loop_file_on_end` controls whether file sources restart automatically
  - UI controls are handled in `pipeline._handle_keys` (Space, q/ESC, r, +/- zoom, WASD pan)
- There is no separate build/test framework in the repo. Install dependencies from `project/requirements.txt`.
- When changing detection/tracking/risk behavior, keep the existing data flow:
  1. pose detection
  2. track update
  3. per-track analytics
  4. frame-level risk engine
  5. snapshot/clip triggering and logging

If any part of the dashboard, config variables, or mode-specific scoring is unclear, ask for clarification before changing the pipeline.