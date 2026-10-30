"""Liveness check: is this a real, present person, not a photo held up to the camera?

Method: ask for a head turn in a RANDOM direction and measure how the nose moves against the eyes and mouth.
A real nose sticks out about 3 cm in front of the eye plane, so turning the head slides it sideways across
the face (a 15 degree turn changes the reading by about 0.14). A flat photo can only move rigidly, so its
nose cannot slide: rotated, tilted, spun and moved through every pose in the physical model, the reading
never changed by more than 0.035 (see ``vision/pose.py`` and ``tests/test_pose.py``). The turn must clear
``min_turn`` = 0.11, three times the worst photo, and be held for a few consecutive frames so one
detector glitch cannot pass it. The direction is chosen at random each time, so a recording cannot
simply be prepared in advance.

What this does NOT stop: a video of the person turning in the requested direction, a 3D mask, or a
virtual camera feeding fake frames. It raises the bar from "hold up a photo" to "have a live-looking
recording or a prop"; real protection against those needs depth or infrared hardware.

``LivenessSession`` knows nothing about identity. The caller checks that the face that turned is the
person being verified (the orchestrator does this frame by frame).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

from vision.enrollment import USER_LEFT, USER_RIGHT
from vision.pose import estimate_yaw

MIN_TURN = 0.11  # reading change that counts as a turn: ~11 degrees on a typical face, 3x the worst photo
BASELINE_FRAMES = 4  # consecutive steady frames to learn "looking straight"
BASELINE_SPREAD = 0.06  # how steady "steady" is (landmark noise alone is about 0.02)
HOLD_FRAMES = 3  # consecutive frames beyond the threshold (a single glitch cannot pass)
TIMEOUT_S = 12.0
LOST_AFTER_S = 1.5  # the face may vanish for this long (a fast turn can lose the detector briefly)


class Phase(str, Enum):
    BASELINE = "baseline"
    TURN = "turn"
    PASSED = "passed"
    FAILED = "failed"


DIRECTION_WORDS = {USER_RIGHT: "right", USER_LEFT: "left"}


@dataclass(frozen=True)
class LivenessStatus:
    phase: Phase
    hint: str  # what to tell the person right now
    turn: float = 0.0  # how far the head has turned in the requested direction (reading units)
    reason: str = ""  # why it failed

    @property
    def passed(self) -> Optional[bool]:
        return True if self.phase is Phase.PASSED else False if self.phase is Phase.FAILED else None


class LivenessSession:
    def __init__(self, direction: Optional[str] = None, *, min_turn: float = MIN_TURN,
                 baseline_frames: int = BASELINE_FRAMES, baseline_spread: float = BASELINE_SPREAD,
                 hold_frames: int = HOLD_FRAMES, timeout: float = TIMEOUT_S, lost_after: float = LOST_AFTER_S,
                 rng: Optional[random.Random] = None) -> None:
        self.direction = direction or (rng or random.SystemRandom()).choice([USER_RIGHT, USER_LEFT])
        self.min_turn, self.timeout, self.lost_after = min_turn, timeout, lost_after
        self.baseline_frames, self.baseline_spread, self.hold_frames = baseline_frames, baseline_spread, hold_frames
        self.phase = Phase.BASELINE
        self.baseline: Optional[float] = None
        self.best_turn = 0.0  # the largest turn seen in the requested direction
        self.readings: List[float] = []  # every reading, for diagnostics
        self._steady: List[float] = []
        self._beyond = 0
        self._started: Optional[float] = None
        self._last_seen: Optional[float] = None
        self._result: Optional[LivenessStatus] = None

    @property
    def side(self) -> str:
        return DIRECTION_WORDS[self.direction]

    @property
    def instruction(self) -> str:
        return f"Look at the camera, then slowly turn your head to your {self.side}."

    def _sign(self) -> float:
        """The estimator is negative toward the person's right."""
        return -1.0 if self.direction == USER_RIGHT else 1.0

    def _finish(self, phase: Phase, hint: str, reason: str = "") -> LivenessStatus:
        self.phase = phase
        self._result = LivenessStatus(phase, hint, self.best_turn, reason)
        return self._result

    def offer(self, faces: Sequence, now: float) -> LivenessStatus:
        """Feed one frame's detections (FaceDetection objects; may be empty). Returns the current status."""
        if self._result is not None:
            return self._result  # decided: further frames change nothing
        if self._started is None:
            self._started = self._last_seen = now
        if now - self._started > self.timeout:
            return self._finish(Phase.FAILED, "I couldn't verify you.", "no head turn in time")

        if len(faces) > 1:
            return LivenessStatus(self.phase, "Only the person being verified should be in view.", self.best_turn)
        if not faces:
            if now - (self._last_seen or now) > self.lost_after:
                return self._finish(Phase.FAILED, "I lost sight of you.", "face lost")
            return LivenessStatus(self.phase, self._prompt(), self.best_turn)
        self._last_seen = now

        yaw = estimate_yaw(faces[0].landmarks)
        if yaw is None:
            return LivenessStatus(self.phase, "Face the camera so I can see your eyes.", self.best_turn)
        self.readings.append(yaw)

        if self.phase is Phase.BASELINE:
            self._steady = (self._steady + [yaw])[-self.baseline_frames:]
            if (len(self._steady) == self.baseline_frames
                    and max(self._steady) - min(self._steady) <= self.baseline_spread):
                self.baseline = float(np.median(self._steady))
                self.phase = Phase.TURN
            return LivenessStatus(self.phase, self._prompt(), self.best_turn)

        turn = (yaw - self.baseline) * self._sign()  # positive = toward the requested side
        self.best_turn = max(self.best_turn, turn)
        if turn >= self.min_turn:
            self._beyond += 1
            if self._beyond >= self.hold_frames:
                return self._finish(Phase.PASSED, "Verified.")
        else:
            self._beyond = 0
        if -turn >= self.min_turn:
            return LivenessStatus(self.phase, f"The other way, please: to your {self.side}.", self.best_turn)
        return LivenessStatus(self.phase, self._prompt(), self.best_turn)

    def _prompt(self) -> str:
        if self.phase is Phase.BASELINE:
            return "Look straight at the camera."
        return f"Now slowly turn your head to your {self.side}."
