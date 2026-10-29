import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

import core.orchestrator as orch_mod
from audio.stt_engine import Transcript
from config.settings import Settings
from core.event_bus import Topic
from storage.face_store import FaceStore
from ui.main_window import MainWindow, VideoWidget
from ui.qt_bridges import EngineController, QtBridge
from vision.detector import FaceDetection
from vision.tracker import TrackState

EMB = np.random.default_rng(7).normal(size=128).astype(np.float32)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_for(app, cond, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    app.processEvents()
    return cond()


class FakeAnalyzer:
    def analyze(self, image):
        return [FaceDetection((60, 260, 260, 60), 0.95, np.zeros((5, 2), np.float32), EMB)]


class FakeCamera:
    def isOpened(self):
        return True

    def read(self):
        time.sleep(1 / 30)
        return True, np.full((240, 320, 3), 90, np.uint8)

    def release(self): ...


def test_video_widget_paints_a_box_in_the_right_place(qapp):
    w = VideoWidget()
    w.resize(640, 480)
    img = QImage(640, 480, QImage.Format.Format_RGB888)
    img.fill(QColor(40, 40, 40))
    w.set_frame(img, [TrackState(1, (100, 300, 300, 100), "Alice", 0.9, False)])  # top,right,bottom,left
    out = w.grab().toImage()
    edge = QColor(out.pixel(100, 200))  # left edge of the box
    assert edge.green() > 150 and edge.red() < 120  # accent green, not the grey frame
    inside = QColor(out.pixel(200, 200))
    assert abs(inside.red() - 40) < 6  # image visible inside the box


def test_video_widget_placeholder(qapp):
    w = VideoWidget()
    w.resize(300, 200)
    w.set_message("Camera is off")
    assert not w.grab().isNull()


def test_frame_mailbox_never_builds_a_backlog(qapp):
    bridge, emitted = QtBridge(), []
    bridge.frame_available.connect(lambda: emitted.append(1), Qt.ConnectionType.DirectConnection)
    imgs = [QImage(4, 4, QImage.Format.Format_RGB888) for _ in range(100)]
    for i, im in enumerate(imgs):
        bridge._push_frame(im, i)
    assert len(emitted) == 1  # 100 frames, GUI never asked: still one pending signal
    image, info = bridge.take_frame()
    assert info == 99 and image is imgs[99]  # and it is the newest
    assert bridge.take_frame() is None
    bridge._push_frame(imgs[0], 0)
    assert len(emitted) == 2  # consuming re-arms the signal


def test_full_gui_flow(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "ENROLL_MIN_GAP", 0.0)
    cfg = Settings(faces_dir=str(tmp_path / "faces"), db_path=str(tmp_path / "a.db"))
    FaceStore(cfg.faces_dir).add("Alice", EMB)

    ctl = EngineController(dict(cfg=cfg, camera=True, voice=False, tts=False,
                                analyzer=FakeAnalyzer(), camera_source_factory=FakeCamera))
    win = MainWindow(ctl)
    win.show()
    ctl.start()
    try:
        chat = lambda: win.chat.toPlainText()
        # startup, live video, recognition + greeting
        assert wait_for(qapp, lambda: "Ready." in chat())
        assert wait_for(qapp, lambda: win.video._image is not None)
        assert wait_for(qapp, lambda: "Hello Alice." in chat())
        assert win.people.item(0).text().startswith("Alice")
        assert win.lbl_voice.text() == "Mic off" and not win.chk_listen.isEnabled()

        # typed command goes through the same path as speech
        win.input.setText("what time is it")
        win._send()
        assert wait_for(qapp, lambda: "It's " in chat())
        assert win.input.text() == ""

        # speech without the wake word is shown, but marked as ignored
        ctl.engine.bus.publish_threadsafe(
            Topic.SPEECH_TRANSCRIBED, Transcript("lunch later", "en", 1.0, 1.0, time.monotonic()))
        assert wait_for(qapp, lambda: "lunch later (ignored: no wake word)" in chat())

        # the activity tab fills from the audit log
        assert wait_for(qapp, lambda: win.table.rowCount() >= 2)
        kinds = {win.table.item(r, 1).text() for r in range(win.table.rowCount())}
        assert {"face recognized", "command"} <= kinds

        # enrolling from the UI: progress is visible, then the list updates by itself
        assert ctl.begin_enrollment("Bob").result(timeout=5).startswith("Okay Bob")
        assert wait_for(qapp, lambda: win.enroll_bar.isVisible() is False and "Bob" in " ".join(
            win.people.item(i).text() for i in range(win.people.count())), timeout=12)
        assert "I'll remember you, Bob." in chat()
        with pytest.raises(ValueError):
            ctl.begin_enrollment("not valid!").result(timeout=5)

        # forgetting
        assert ctl.forget("Bob").result(timeout=5) == orch_mod.ENROLL_SAMPLES
        assert wait_for(qapp, lambda: win.people.count() == 1 and win.people.item(0).text().startswith("Alice"))
    finally:
        t0 = time.monotonic()
        win.close()
        assert time.monotonic() - t0 < 8  # engine shuts down promptly
        assert ctl._thread._thread.is_alive() is False
