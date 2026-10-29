"""Face tracker: follows each face across frames and stabilizes its identity.

Pure NumPy, no OpenCV. Detections are matched to existing tracks by IoU (bounding-box
overlap), so two unknown people never share a track. Each track keeps
* an EMA-smoothed box, so drawn rectangles don't jitter, and
* a sliding window of identity votes; the identity only changes when a label wins
  a clear majority, so one bad frame can't flip "alice" to "bob".
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import count
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # (top, right, bottom, left), as used across the project
Detection = Tuple[BBox, str, float]  # (bbox, label, similarity score)
UNKNOWN = "Unknown"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4); boxes are (x1, y1, x2, y2). Returns (N,M)."""
    a, b = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def _xyxy(bbox: BBox) -> np.ndarray:
    top, right, bottom, left = bbox
    return np.array([left, top, right, bottom], dtype=np.float32)


@dataclass
class _Track:
    id: int
    box: np.ndarray  # smoothed (x1, y1, x2, y2)
    last_seen: float
    votes: Deque[Tuple[str, float]]
    identity: Optional[str] = None  # stable identity; None while still verifying
    score: float = 0.0
    announced: Optional[str] = None


@dataclass(frozen=True)
class TrackState:
    track_id: int
    bbox: BBox
    identity: Optional[str]  # None = still verifying, "Unknown" = confidently unknown
    score: float
    newly_identified: bool  # True once, when a known name becomes stable


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, alpha: float = 0.5, ttl_seconds: float = 1.0,
                 window: int = 7, min_votes: int = 3, majority: float = 0.6) -> None:
        self.iou_threshold = iou_threshold
        self.alpha = alpha
        self.ttl_seconds = ttl_seconds
        self.window = window
        self.min_votes = min_votes
        self.majority = majority
        self._tracks: Dict[int, _Track] = {}
        self._ids = count(1)

    def update(self, now: float, detections: Sequence[Detection]) -> List[TrackState]:
        """Feed one frame's detections (may be empty). Returns state of every visible track."""
        self._tracks = {i: t for i, t in self._tracks.items()
                        if now - t.last_seen <= self.ttl_seconds}
        assigned = self._match(detections)

        states = []
        for det_idx, (bbox, label, score) in enumerate(detections):
            box = _xyxy(bbox)
            track = assigned.get(det_idx)
            if track is None:
                track = _Track(next(self._ids), box, now, deque(maxlen=self.window))
                self._tracks[track.id] = track
            else:
                track.box = self.alpha * box + (1 - self.alpha) * track.box
            track.last_seen = now
            track.votes.append((label, score))
            states.append(self._state(track))
        return states

    def _match(self, detections: Sequence[Detection]) -> Dict[int, _Track]:
        """Greedy IoU assignment: best-overlapping (track, detection) pairs first."""
        if not self._tracks or not detections:
            return {}
        tracks = list(self._tracks.values())
        iou = iou_matrix(np.stack([t.box for t in tracks]),
                         np.stack([_xyxy(d[0]) for d in detections]))
        assigned: Dict[int, _Track] = {}
        used_tracks = set()
        for flat in np.argsort(iou, axis=None)[::-1]:
            ti, di = divmod(int(flat), iou.shape[1])
            if iou[ti, di] < self.iou_threshold:
                break
            if ti in used_tracks or di in assigned:
                continue
            used_tracks.add(ti)
            assigned[di] = tracks[ti]
        return assigned

    def _state(self, t: _Track) -> TrackState:
        stable = self._vote(t)
        if stable is not None:
            t.identity = stable
            scores = [s for lbl, s in t.votes if lbl == stable]
            t.score = float(np.mean(scores))
        newly = bool(t.identity and t.identity != UNKNOWN and t.identity != t.announced)
        if newly:
            t.announced = t.identity
        x1, y1, x2, y2 = (int(v) for v in t.box)
        return TrackState(t.id, (y1, x2, y2, x1), t.identity, t.score, newly)

    def _vote(self, t: _Track) -> Optional[str]:
        """Label holding a clear majority of the window, else None (keep previous identity)."""
        label, n = Counter(lbl for lbl, _ in t.votes).most_common(1)[0]
        return label if n >= self.min_votes and n / len(t.votes) >= self.majority else None
