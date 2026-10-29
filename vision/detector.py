"""Face detection and embedding extraction.

Uses the two ONNX models that ship with OpenCV's zoo, so ``opencv-python`` is the
only dependency:

* YuNet  - fast CPU face detector (bounding box, 5 landmarks, score)
* SFace  - 128-D face embeddings; its recommended cosine threshold is ~0.363

``FaceAnalyzer`` is plain blocking code. ``DetectorService`` runs it off the event
loop and wires it to the bus: FRAME_CAPTURED in, FACE_DETECTED out.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

import numpy as np

from core.event_bus import Event, EventBus, Subscription, Topic
from vision.capture import Frame

log = logging.getLogger("aether.detector")

BBox = Tuple[int, int, int, int]  # (top, right, bottom, left), same as spatial_engine

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MODELS = {
    "yunet.onnx": (
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "sface.onnx": (
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


@dataclass(frozen=True)
class FaceDetection:
    bbox: BBox
    score: float
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose, right mouth, left mouth
    embedding: np.ndarray  # (128,) float32, not yet L2-normalized


@dataclass(frozen=True)
class FaceEvent:
    """Payload of Topic.FACE_DETECTED."""
    frame: Frame
    faces: List[FaceDetection]
    inference_ms: float


class Analyzer(Protocol):
    def analyze(self, image: Any) -> List[FaceDetection]: ...


def ensure_models(model_dir: "str | Path" = "models") -> Tuple[Path, Path]:
    """Download missing models (once) and verify their SHA-256. Blocking."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (url, sha) in MODELS.items():
        path = model_dir / name
        if not (path.exists() and _sha256(path) == sha):
            log.info("downloading %s", name)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if _sha256(tmp) != sha:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch for {name}; download corrupted")
            os.replace(tmp, path)
        paths.append(path)
    return paths[0], paths[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FaceAnalyzer:
    """Detect faces and compute one embedding per face. Not thread-safe: use one thread."""

    def __init__(self, model_dir: "str | Path" = "models", score_threshold: float = 0.8,
                 nms_threshold: float = 0.3, top_k: int = 10) -> None:
        import cv2  # lazy so the rest of the package imports without OpenCV

        self._cv2 = cv2
        yunet, sface = ensure_models(model_dir)
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 320), score_threshold, nms_threshold, top_k)
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        self._size: Optional[Tuple[int, int]] = None

    def analyze(self, image: np.ndarray) -> List[FaceDetection]:
        h, w = image.shape[:2]
        if self._size != (w, h):
            self._detector.setInputSize((w, h))
            self._size = (w, h)
        _, rows = self._detector.detect(image)
        if rows is None:
            return []
        faces = []
        for row in rows:
            x, y, bw, bh = row[:4]
            bbox = (max(int(y), 0), min(int(x + bw), w), min(int(y + bh), h), max(int(x), 0))
            aligned = self._recognizer.alignCrop(image, row)
            emb = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
            faces.append(FaceDetection(bbox, float(row[-1]), row[4:14].reshape(5, 2).copy(), emb))
        return faces


class DetectorService:
    """Bus adapter: analyzes the newest captured frame and publishes FACE_DETECTED."""

    def __init__(self, bus: EventBus, analyzer: Analyzer) -> None:
        self._bus = bus
        self._analyzer = analyzer
        self._sub: Optional[Subscription] = None

    def start(self) -> None:
        # max_queue=1 + drop_oldest: while a frame is being analyzed, only the newest waits.
        self._sub = self._bus.subscribe(Topic.FRAME_CAPTURED, self._on_frame, max_queue=1)

    def stop(self) -> None:
        if self._sub:
            self._bus.unsubscribe(self._sub)
            self._sub = None

    async def _on_frame(self, event: Event) -> None:
        frame: Frame = event.payload
        t0 = time.perf_counter()
        faces = await asyncio.to_thread(self._analyzer.analyze, frame.image)
        ms = (time.perf_counter() - t0) * 1000
        self._bus.publish(Topic.FACE_DETECTED, FaceEvent(frame, faces, ms))
