"""Head pose from the 5 face landmarks, and a physical model of a head and of a photo of one.

``estimate_yaw`` is shared by enrollment (which asks for several views) and liveness (which asks for a
head turn). ``simulate_landmarks`` projects a 3D head, or a flat photo of it, through a pinhole camera.
Tests and the liveness self-test use it, so the thresholds rest on geometry rather than on guesses.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def estimate_yaw(landmarks: np.ndarray) -> Optional[float]:
    """Head turn from YuNet's 5 landmarks (right eye, left eye, nose, mouth right, mouth left).

    0 = frontal. Negative = turned toward the person's RIGHT (their nose moves to the image-left,
    because the subject's right eye is on the image-left); positive = toward their left.

    The nose is measured against BOTH the eye line and the mouth line, and the two readings are
    averaged: on one real face under different brightness and scale this halved the noise (std 0.018
    instead of 0.034) and removed a frontal bias. Distances are taken ALONG THE EYE AXIS, so the
    reading is unchanged when the head (or a photo) is tilted sideways like a steering wheel.
    None if the eyes coincide.
    """
    right_eye, left_eye, nose, mouth_right, mouth_left = (np.asarray(p, dtype=np.float64) for p in landmarks[:5])
    axis = left_eye - right_eye
    eye_dist = float(np.hypot(*axis))
    if eye_dist < 1e-3:
        return None
    axis = axis / eye_dist
    from_eyes = float(np.dot(nose - (right_eye + left_eye) / 2, axis))
    from_mouth = float(np.dot(nose - (mouth_right + mouth_left) / 2, axis))
    return (from_eyes + from_mouth) / 2 / eye_dist


# ---- a physical model, in centimetres --------------------------------------------------------------
# x = image-right, y = down, z = away from the camera. The nose tip sticks out about 3 cm in front of the
# eye plane, which is exactly what a photo lacks. Order: right eye, left eye, nose, mouth right, mouth left.
_HEAD_CM = np.array([[-3.2, -3.0, 0.0],
                     [+3.2, -3.0, 0.0],
                     [0.0, 0.0, -3.0],
                     [-2.4, +3.5, -0.8],
                     [+2.4, +3.5, -0.8]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def simulate_landmarks(turn_right_deg: float = 0.0, *, planar: bool = False, roll_deg: float = 0.0,
                       pitch_deg: float = 0.0, distance_cm: float = 50.0, focal_px: float = 554.0,
                       noise_px: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Pixel positions of the 5 landmarks of a head (or, with ``planar=True``, a flat photo of it).

    ``turn_right_deg``: positive = turned toward the person's right (``estimate_yaw`` then goes negative).
    ``roll_deg``: tilt sideways (steering wheel); ``pitch_deg``: nod. Defaults match a 640x480 webcam
    (about 60 degrees wide) with the face 50 cm away. ``noise_px`` adds landmark-detector jitter.
    """
    pts = _HEAD_CM.copy()
    if planar:
        pts[:, 2] = 0.0  # a photo: every landmark lies in one flat plane
    pivot = np.array([0.0, 0.0, 9.0])  # the neck axis, behind the face
    rot = _rot_z(np.radians(roll_deg)) @ _rot_x(np.radians(pitch_deg)) @ _rot_y(np.radians(turn_right_deg))
    pts = (pts - pivot) @ rot.T + pivot
    depth = distance_cm + pts[:, 2]
    px = np.stack([focal_px * pts[:, 0] / depth + 320.0, focal_px * pts[:, 1] / depth + 240.0], axis=1)
    if noise_px:
        px = px + (rng or np.random.default_rng()).normal(scale=noise_px, size=px.shape)
    return px.astype(np.float32)
