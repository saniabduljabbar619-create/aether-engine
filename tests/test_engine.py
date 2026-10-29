import numpy as np

from config.settings import Settings
from core.engine import Engine
from storage.face_store import FaceStore


class ClosedCamera:
    def isOpened(self):
        return False

    def release(self): ...


class Analyzer:
    def analyze(self, image):
        return []


def cfg(tmp_path):
    return Settings(faces_dir=str(tmp_path / "faces"), db_path=str(tmp_path / "a.db"))


async def test_starts_and_stops_with_everything_off(tmp_path):
    e = Engine(cfg(tmp_path), camera=False, voice=False, tts=False)
    await e.start()
    assert e.cap is None and e.mic is None and e.warnings == []
    assert await e.orch._h_status(None) == "The camera is off. I know 0 people."
    await e.stop()


async def test_camera_failure_is_a_warning_not_a_crash(tmp_path):
    e = Engine(cfg(tmp_path), camera=True, voice=False, tts=False,
               analyzer=Analyzer(), camera_source_factory=ClosedCamera)
    await e.start()
    assert e.cap is None and "Camera unavailable" in e.warnings[0]
    await e.stop()


async def test_loads_enrolled_people_into_the_matcher(tmp_path):
    emb = np.random.default_rng(2).normal(size=128).astype(np.float32)
    FaceStore(cfg(tmp_path).faces_dir).add("Alice", emb)
    e = Engine(cfg(tmp_path), camera=False, voice=False, tts=False)
    await e.start()
    assert e.matcher.search_batch(emb[None])[0][0] == "Alice"
    await e.stop()
