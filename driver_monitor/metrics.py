"""Pure geometry helpers: EAR, MAR and head pose from FaceMesh landmarks."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# FaceMesh landmark indices -------------------------------------------------
# Order: outer corner, upper-1, upper-2, inner corner, lower-2, lower-1
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
# left corner, right corner, upper inner lip, lower inner lip
MOUTH = (78, 308, 13, 14)
# nose tip, chin, eye outer corners, mouth corners
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)

# Generic 3D face model (mm). Y points up, Z points out of the face.
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -330.0, -65.0),     # chin
        (-225.0, 170.0, -135.0),  # eye corner, image left
        (225.0, 170.0, -135.0),   # eye corner, image right
        (-150.0, -150.0, -125.0), # mouth corner, image left
        (150.0, -150.0, -125.0),  # mouth corner, image right
    ],
    dtype=np.float64,
)

_FLIP_YZ = np.diag([1.0, -1.0, -1.0])


def _pt(lm, i: int, w: int, h: int) -> np.ndarray:
    return np.array([lm[i].x * w, lm[i].y * h])


def eye_aspect_ratio(lm, idx: Sequence[int], w: int, h: int) -> float:
    p = [_pt(lm, i, w, h) for i in idx]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return float(vertical / (2.0 * horizontal + 1e-6))


def mouth_aspect_ratio(lm, w: int, h: int) -> float:
    left, right, upper, lower = (_pt(lm, i, w, h) for i in MOUTH)
    return float(np.linalg.norm(upper - lower) / (np.linalg.norm(left - right) + 1e-6))


def head_pose(lm, w: int, h: int) -> Optional[Tuple[float, float, float]]:
    """Return (yaw, pitch, roll) in degrees, or None if PnP fails.

    Conventions (as seen on screen, un-mirrored image):
      yaw   > 0  face turned towards the right of the image
      pitch > 0  head tilted down
      roll  > 0  head tilted counter-clockwise
    """
    image_points = np.array(
        [(lm[i].x * w, lm[i].y * h) for i in POSE_LANDMARKS], dtype=np.float64
    )
    camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    # Undo the model(y-up, z-out) -> camera(y-down, z-in) convention so the
    # Euler angles are small around a frontal pose instead of wrapping at 180.
    rmat = _FLIP_YZ @ rmat

    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    return yaw, pitch, roll
