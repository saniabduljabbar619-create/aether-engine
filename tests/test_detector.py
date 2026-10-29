import asyncio
from pathlib import Path

import numpy as np
import pytest

from core.event_bus import EventBus, Topic
from vision.capture import Frame
from vision.detector import DetectorService, FaceDetection, FaceEvent

DATA = Path(__file__).parent / "data"
MODELS = Path(__file__).parent.parent / "models"


def _face(score=0.9):
    return FaceDetection((10, 50, 60, 5), score, np.zeros((5, 2), np.float32), np.ones(128, np.float32))


class FakeAnalyzer:
    def __init__(self, delay=0.0, fail=False):
        self.delay, self.fail, self.seen = delay, fail, []

    def analyze(self, image):
        import time
        time.sleep(self.delay)
        self.seen.append(image)
        if self.fail:
            raise RuntimeError("model crashed")
        return [_face()]


@pytest.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


def _frame(i):
    return Frame(i, float(i), np.zeros((4, 4, 3), np.uint8))


async def test_publishes_face_detected(bus):
    got = []

    async def h(e):
        got.append(e.payload)

    bus.subscribe(Topic.FACE_DETECTED, h)
    DetectorService(bus, FakeAnalyzer()).start()
    bus.publish(Topic.FRAME_CAPTURED, _frame(1))
    await asyncio.sleep(0.05)
    assert len(got) == 1 and isinstance(got[0], FaceEvent)
    assert got[0].frame.frame_id == 1 and len(got[0].faces) == 1 and got[0].inference_ms >= 0


async def test_skips_stale_frames_while_busy(bus):
    got = []

    async def h(e):
        got.append(e.payload.frame.frame_id)

    bus.subscribe(Topic.FACE_DETECTED, h)
    DetectorService(bus, FakeAnalyzer(delay=0.05)).start()
    for i in range(1, 11):
        bus.publish(Topic.FRAME_CAPTURED, _frame(i))
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.3)
    assert got[0] == 1 and got[-1] == 10 and len(got) < 6  # newest processed, middle skipped


async def test_detection_rate_is_capped(bus):
    an = FakeAnalyzer()
    DetectorService(bus, an, max_fps=20).start()  # at most one analysis per 50 ms
    for i in range(40):  # the camera delivers 100 frames/s for 0.4 s
        bus.publish(Topic.FRAME_CAPTURED, _frame(i))
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    assert 3 <= len(an.seen) <= 12  # about 8, not 40


async def test_after_resting_the_newest_frame_is_analyzed(bus):
    got = []

    async def h(e):
        got.append(e.payload.frame.frame_id)

    bus.subscribe(Topic.FACE_DETECTED, h)
    DetectorService(bus, FakeAnalyzer(), max_fps=10).start()
    for i in range(1, 11):
        bus.publish(Topic.FRAME_CAPTURED, _frame(i))
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.3)
    assert got[0] == 1 and got[-1] == 10  # never a stale middle frame last


async def test_analyzer_failure_does_not_kill_service(bus):
    got = []

    async def h(e):
        got.append(e)

    bus.subscribe(Topic.FACE_DETECTED, h)
    an = FakeAnalyzer(fail=True)
    DetectorService(bus, an).start()
    bus.publish(Topic.FRAME_CAPTURED, _frame(1))
    await asyncio.sleep(0.05)
    an.fail = False
    bus.publish(Topic.FRAME_CAPTURED, _frame(2))
    await asyncio.sleep(0.05)
    assert len(got) == 1


needs_models = pytest.mark.skipif(
    not (MODELS / "yunet.onnx").exists() or not (MODELS / "sface.onnx").exists(),
    reason="run `python -c 'from vision.detector import ensure_models; ensure_models()'` first",
)


@needs_models
def test_real_models_find_face_and_embed():
    cv2 = pytest.importorskip("cv2")
    from vision.detector import FaceAnalyzer

    img = cv2.imread(str(DATA / "astronaut.jpg"))
    an = FaceAnalyzer(MODELS)
    faces = an.analyze(img)
    assert len(faces) == 1
    f = faces[0]
    top, right, bottom, left = f.bbox
    assert 0 <= top < bottom <= 512 and 0 <= left < right <= 512 and f.score > 0.8
    assert f.embedding.shape == (128,) and f.landmarks.shape == (5, 2)

    again = an.analyze(img)[0].embedding  # deterministic
    cos = float(f.embedding @ again / (np.linalg.norm(f.embedding) * np.linalg.norm(again)))
    assert cos > 0.999
    assert an.analyze(np.zeros((480, 640, 3), np.uint8)) == []  # no face, no crash


@needs_models
def test_enroll_then_recognize_modified_photo(tmp_path):
    """End to end with real models: same person, different pixels, still matches."""
    cv2 = pytest.importorskip("cv2")
    from storage.face_store import FaceStore
    from vision.detector import FaceAnalyzer
    from vision.spatial_engine import VectorIdentityMatcher

    img = cv2.imread(str(DATA / "astronaut.jpg"))
    an = FaceAnalyzer(MODELS)
    store = FaceStore(tmp_path)
    store.add("astro", an.analyze(img)[0].embedding)

    m = VectorIdentityMatcher()
    m.register_identities(store.load_all())
    variant = cv2.convertScaleAbs(cv2.resize(img, (400, 400)), alpha=0.8, beta=25)  # smaller, darker
    emb = an.analyze(variant)[0].embedding
    (label, score), = m.search_batch(emb[None])
    assert label == "astro" and score > 0.8
