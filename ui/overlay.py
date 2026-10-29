"""Drawing helpers for the live camera window (OpenCV)."""
import textwrap
from typing import Optional, Sequence

import cv2

from vision.tracker import UNKNOWN, TrackState

GREEN, YELLOW, RED = (0, 220, 0), (0, 220, 255), (0, 0, 255)


def draw(img, tracks: Sequence[TrackState], status: str, banner: Optional[str] = None):
    """Boxes + names for each track, a status line on top, and an optional wrapped banner."""
    for t in tracks:
        top, right, bottom, left = t.bbox
        if t.identity is None:
            color, label = YELLOW, "verifying..."
        elif t.identity == UNKNOWN:
            color, label = RED, "Unknown"
        else:
            color, label = GREEN, f"{t.identity.replace('_', ' ')} {t.score:.2f}"
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom), (right, bottom + 24), color, -1)
        cv2.putText(img, label, (left + 4, bottom + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(img, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    if banner:
        lines = textwrap.wrap(banner, 42)[:3]
        y0 = img.shape[0] - 16 - 28 * (len(lines) - 1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (12, y0 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    return img
