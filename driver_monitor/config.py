"""Central configuration. Every tunable value lives here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@dataclass(frozen=True)
class Config:
    # --- Camera -----------------------------------------------------------
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # --- Eyes (Eye Aspect Ratio) -----------------------------------------
    ear_threshold: float = 0.22        # below this the eye is considered closed
    eyes_closed_seconds: float = 1.0   # closed this long -> drowsy

    # --- Yawning (Mouth Aspect Ratio) ------------------------------------
    mar_threshold: float = 0.60
    yawn_seconds: float = 1.5

    # --- Head pose, in degrees relative to the calibrated posture --------
    yaw_limit: float = 25.0
    pitch_limit: float = 20.0
    roll_limit: float = 25.0
    distraction_seconds: float = 1.5

    # --- Face visibility --------------------------------------------------
    no_face_seconds: float = 3.0

    # --- Alerting ---------------------------------------------------------
    alarm_cooldown: float = 5.0        # minimum gap between alarm sounds
    alert_hold_seconds: float = 3.0    # keep the DANGER banner on screen
    alarm_path: Path = ASSETS_DIR / "audio.mp3"

    # --- Signal processing ------------------------------------------------
    smoothing: float = 0.4             # EMA factor, 1.0 = no smoothing
    calibration_seconds: float = 2.0
