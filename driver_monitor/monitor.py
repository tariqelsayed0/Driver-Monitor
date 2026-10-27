"""Detection engine: turns camera frames into a driver-state Result."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import Config
from .metrics import (
    LEFT_EYE,
    RIGHT_EYE,
    eye_aspect_ratio,
    head_pose,
    mouth_aspect_ratio,
)


class Status(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CALIBRATING = "calibrating"
    NO_FACE = "no_face"


# BGR colours used for the on-video overlay
OVERLAY_COLORS = {
    Status.SAFE: (96, 200, 63),
    Status.WARNING: (0, 170, 255),
    Status.DANGER: (73, 81, 248),
    Status.CALIBRATING: (253, 166, 88),
    Status.NO_FACE: (150, 150, 150),
}


@dataclass
class Result:
    status: Status = Status.NO_FACE
    reasons: List[str] = field(default_factory=list)
    fresh: bool = False  # True while the danger condition is actually present
    ear: float = 0.0
    mar: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    calibration_progress: float = 0.0


class DriverMonitor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._since: Dict[str, float] = {}
        self._ema: Dict[str, float] = {}
        self._raw_pose = np.zeros(3)
        self._baseline = np.zeros(3)
        self._calib_start: Optional[float] = None
        self._calib_samples: List[np.ndarray] = []
        self._hold_until = 0.0
        self._held_reasons: List[str] = []

    # ------------------------------------------------------------------ API
    def start_calibration(self) -> None:
        """Record the driver's neutral posture for `calibration_seconds`."""
        self._calib_start = time.monotonic()
        self._calib_samples = []
        self._since.clear()

    def close(self) -> None:
        self._mesh.close()

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Result]:
        now = time.monotonic()
        h, w = frame.shape[:2]
        out = self._mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not out.multi_face_landmarks:
            return frame, self._no_face(now)
        self._since.pop("no_face", None)

        lm = out.multi_face_landmarks[0].landmark
        ear = self._smooth(
            "ear",
            (eye_aspect_ratio(lm, LEFT_EYE, w, h) + eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2,
        )
        mar = self._smooth("mar", mouth_aspect_ratio(lm, w, h))

        pose = head_pose(lm, w, h)
        if pose is not None:
            self._raw_pose = np.array(
                [self._smooth(k, v) for k, v in zip(("yaw", "pitch", "roll"), pose)]
            )

        if self._calib_start is not None:
            result = self._calibrate(now, ear, mar)
        else:
            yaw, pitch, roll = self._raw_pose - self._baseline
            result = self._evaluate(now, ear, mar, yaw, pitch, roll)

        self._draw(frame, lm, w, h, result)
        return frame, result

    # ------------------------------------------------------------- internals
    def _smooth(self, key: str, value: float) -> float:
        prev = self._ema.get(key)
        new = value if prev is None else prev + self.cfg.smoothing * (value - prev)
        self._ema[key] = new
        return new

    def _elapsed(self, key: str, active: bool, now: float) -> float:
        """Seconds a condition has been continuously true (0 if inactive)."""
        if not active:
            self._since.pop(key, None)
            return 0.0
        return now - self._since.setdefault(key, now)

    def _calibrate(self, now: float, ear: float, mar: float) -> Result:
        elapsed = now - (self._calib_start or now)
        self._calib_samples.append(self._raw_pose.copy())
        if elapsed >= self.cfg.calibration_seconds:
            self._baseline = np.mean(self._calib_samples, axis=0)
            self._calib_start = None
            self._since.clear()
            return self._evaluate(now, ear, mar, 0.0, 0.0, 0.0)
        return Result(
            Status.CALIBRATING,
            ear=ear,
            mar=mar,
            calibration_progress=min(elapsed / self.cfg.calibration_seconds, 1.0),
        )

    def _no_face(self, now: float) -> Result:
        for key in ("eyes", "yawn", "head"):
            self._since.pop(key, None)
        missing = self._elapsed("no_face", True, now)
        if missing >= self.cfg.no_face_seconds:
            return Result(Status.DANGER, ["Face not visible"], fresh=True)
        return Result(Status.NO_FACE)

    def _evaluate(
        self, now: float, ear: float, mar: float, yaw: float, pitch: float, roll: float
    ) -> Result:
        c = self.cfg
        eyes_t = self._elapsed("eyes", ear < c.ear_threshold, now)
        yawn_t = self._elapsed("yawn", mar > c.mar_threshold, now)
        off_axis = (
            abs(yaw) > c.yaw_limit or abs(pitch) > c.pitch_limit or abs(roll) > c.roll_limit
        )
        head_t = self._elapsed("head", off_axis, now)

        values = dict(ear=ear, mar=mar, yaw=yaw, pitch=pitch, roll=roll)

        reasons = []
        if eyes_t >= c.eyes_closed_seconds:
            reasons.append("Eyes closed")
        if yawn_t >= c.yawn_seconds:
            reasons.append("Yawning")
        if head_t >= c.distraction_seconds:
            reasons.append("Distracted")

        if reasons:
            self._hold_until = now + c.alert_hold_seconds
            self._held_reasons = reasons
            return Result(Status.DANGER, reasons, fresh=True, **values)

        if now < self._hold_until:
            return Result(Status.DANGER, list(self._held_reasons), **values)

        early = []
        if eyes_t > 0.3:
            early.append("Eyes closing")
        if yawn_t > 0.3:
            early.append("Mouth open")
        if head_t > 0.3:
            early.append("Looking away")
        if early:
            return Result(Status.WARNING, early, **values)

        return Result(Status.SAFE, **values)

    # --------------------------------------------------------------- overlay
    def _draw(self, frame: np.ndarray, lm, w: int, h: int, r: Result) -> None:
        color = OVERLAY_COLORS[r.status]

        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        _brackets(
            frame,
            int(min(xs)) - 14, int(min(ys)) - 14,
            int(max(xs)) + 14, int(max(ys)) + 14,
            color,
        )

        for idx in (LEFT_EYE, RIGHT_EYE):
            pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in idx], np.int32)
            cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

        nose = (int(lm[1].x * w), int(lm[1].y * h))
        cv2.circle(frame, nose, 3, color, -1, cv2.LINE_AA)
        tip = (nose[0] + int(r.yaw * 4), nose[1] + int(r.pitch * 4))
        if tip != nose:
            cv2.arrowedLine(frame, nose, tip, color, 2, cv2.LINE_AA, tipLength=0.3)

        if r.status is Status.CALIBRATING:
            msg = "Look straight ahead"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(
                frame, msg, ((w - tw) // 2, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
            )


def _brackets(img, x1, y1, x2, y2, color, thickness=2, length=26) -> None:
    for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + dx * length, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * length), color, thickness, cv2.LINE_AA)
