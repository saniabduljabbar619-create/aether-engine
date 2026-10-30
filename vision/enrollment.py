"""Enrollment quality: decide which face samples are worth keeping.

Good enrollment means a few DIFFERENT, clear views of one person, not 10 copies of one frame.
A session therefore
  * accepts a sample only when exactly one clear, large-enough face is in view,
  * wants a mix of poses (straight on, a little to each side) and prompts for what is missing,
  * rejects near-identical frames, so 10 samples carry 10 slightly different views,
  * and at the end checks that the samples agree with each other (one person, not two).

Brightness is deliberately NOT a rejection rule. An absolute brightness cut-off penalizes darker
skin tones and dim rooms; the detector's own confidence and the consistency check already catch
unusable images. Dim light only produces a warning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from vision.pose import estimate_yaw

CENTER, USER_RIGHT, USER_LEFT = "center", "user_right", "user_left"
POSES = (CENTER, USER_RIGHT, USER_LEFT)
YAW_THRESHOLD = 0.12  # nose offset / eye distance beyond which the head counts as turned

PROMPTS = {
    CENTER: "Look straight at the camera.",
    USER_RIGHT: "Slowly turn your head a little to your right.",
    USER_LEFT: "Now a little to your left.",
}
DIM_WARNING = "It's dim, so more light will help."
DIM_BELOW = 60  # mean pixel value of the frame; a warning only, never a rejection


def pose_of(yaw: float) -> str:
    if yaw < -YAW_THRESHOLD:
        return USER_RIGHT
    if yaw > YAW_THRESHOLD:
        return USER_LEFT
    return CENTER


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cohesion(samples: Sequence[np.ndarray]) -> float:
    """Mean pairwise cosine similarity. Samples of one person agree; two people mixed do not."""
    if len(samples) < 2:
        return 1.0
    m = np.stack([_unit(s) for s in samples])
    sims = m @ m.T
    n = len(samples)
    return float((sims.sum() - n) / (n * (n - 1)))


@dataclass(frozen=True)
class Feedback:
    text: str  # what to tell the person right now
    accepted: bool = False  # this frame became a sample
    done: bool = False  # enough good samples


class EnrollmentSession:
    def __init__(self, name: str, *, needed: int = 10, min_center: int = 4, min_side: int = 2,
                 min_height: int = 90, min_score: float = 0.9, min_gap: float = 0.4,
                 duplicate_cos: float = 0.96, relax_after: float = 18.0) -> None:
        self.name = name
        self.needed, self.min_center, self.min_side = needed, min_center, min_side
        self.min_height, self.min_score, self.min_gap = min_height, min_score, min_gap
        self.duplicate_cos, self.relax_after = duplicate_cos, relax_after
        self.samples: List[np.ndarray] = []
        self.poses: List[str] = []
        self.started: Optional[float] = None
        self._last_accept = float("-inf")

    # ---- state -----------------------------------------------------------
    @property
    def counts(self) -> Dict[str, int]:
        return {p: self.poses.count(p) for p in POSES}

    def _minimum(self, pose: str) -> int:
        return self.min_center if pose == CENTER else self.min_side

    def _next_pose_wanted(self, now: float) -> Optional[str]:
        """First pose still below its minimum; None once variety is satisfied or waived."""
        if self.started is not None and now - self.started >= self.relax_after:
            return None  # the person may not manage every angle; don't wait forever
        counts = self.counts
        return next((p for p in POSES if counts[p] < self._minimum(p)), None)

    def _prompt(self, now: float) -> str:
        wanted = self._next_pose_wanted(now)
        return PROMPTS[wanted] if wanted else "Hold still, nearly done."

    def _complete(self, now: float) -> bool:
        return len(self.samples) >= self.needed and self._next_pose_wanted(now) is None

    # ---- the one entry point ---------------------------------------------
    def offer(self, faces: Sequence, brightness: float, now: float) -> Feedback:
        """Consider one frame's detections. ``faces`` are FaceDetection objects."""
        if self.started is None:
            self.started = now
        if not faces:
            return Feedback("I can't see a face. Step in front of the camera.")
        if len(faces) > 1:
            return Feedback("Only the person enrolling should be in view.")
        face = faces[0]
        top, _, bottom, _ = face.bbox
        if face.score < self.min_score:
            return Feedback("Your face isn't clear. Face the camera.")
        if bottom - top < self.min_height:
            return Feedback("Move a little closer to the camera.")
        yaw = estimate_yaw(face.landmarks)
        if yaw is None:
            return Feedback("I can't make out your face clearly. Face the camera.")

        hint = self._prompt(now)
        dim = f" {DIM_WARNING}" if brightness < DIM_BELOW else ""
        if now - self._last_accept < self.min_gap:
            return Feedback(hint + dim)

        pose = pose_of(yaw)
        wanted = self._next_pose_wanted(now)
        if wanted is not None and pose != wanted and self.counts[pose] >= self._minimum(pose):
            return Feedback(hint + dim)  # we already have enough of this pose; ask for the missing one
        emb = _unit(np.asarray(face.embedding, dtype=np.float32))
        if any(float(emb @ _unit(s)) >= self.duplicate_cos for s in self.samples):
            return Feedback("Move your head a little, I need different views." + dim)

        self.samples.append(np.asarray(face.embedding, dtype=np.float32))
        self.poses.append(pose)
        self._last_accept = now
        return Feedback(self._prompt(now) + dim, accepted=True, done=self._complete(now))
