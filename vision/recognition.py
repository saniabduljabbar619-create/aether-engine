"""Recognition service: FACE_DETECTED -> match -> track -> FACES_UPDATED / FACE_RECOGNIZED."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.event_bus import Event, EventBus, Subscription, Topic
from vision.capture import Frame
from vision.detector import FaceEvent
from vision.spatial_engine import VectorIdentityMatcher
from vision.tracker import UNKNOWN, FaceTracker, TrackState


@dataclass(frozen=True)
class FacesUpdate:
    """Payload of Topic.FACES_UPDATED: every tracked face in the frame, for overlays."""
    frame: Frame
    tracks: List[TrackState]
    inference_ms: float


@dataclass(frozen=True)
class Recognition:
    """Payload of Topic.FACE_RECOGNIZED: a known person has just been confirmed."""
    name: str
    score: float
    track_id: int
    timestamp: float


class RecognitionService:
    def __init__(self, bus: EventBus, matcher: VectorIdentityMatcher,
                 tracker: Optional[FaceTracker] = None, greet_cooldown: float = 30.0) -> None:
        self._bus = bus
        self._matcher = matcher
        self._tracker = tracker or FaceTracker()
        self._cooldown = greet_cooldown
        self._last_announced: Dict[str, float] = {}
        self._sub: Optional[Subscription] = None

    def start(self) -> None:
        self._sub = self._bus.subscribe(Topic.FACE_DETECTED, self._on_faces, max_queue=1)

    def stop(self) -> None:
        if self._sub:
            self._bus.unsubscribe(self._sub)
            self._sub = None

    async def _on_faces(self, event: Event) -> None:
        ev: FaceEvent = event.payload
        now = ev.frame.timestamp
        detections = []
        if ev.faces:
            results = self._matcher.search_batch(np.stack([f.embedding for f in ev.faces]))
            detections = [(f.bbox, label or UNKNOWN, score)
                          for f, (label, score) in zip(ev.faces, results)]
        tracks = self._tracker.update(now, detections)  # also called when empty, to expire tracks
        self._bus.publish(Topic.FACES_UPDATED, FacesUpdate(ev.frame, tracks, ev.inference_ms))

        for t in tracks:
            if t.newly_identified and self._cooldown_ok(t.identity, now):
                self._last_announced[t.identity] = now
                self._bus.publish(Topic.FACE_RECOGNIZED, Recognition(t.identity, t.score, t.track_id, now))

    def _cooldown_ok(self, name: str, now: float) -> bool:
        last = self._last_announced.get(name)
        return last is None or now - last >= self._cooldown
