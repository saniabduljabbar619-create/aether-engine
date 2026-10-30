import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from tests._optional import optional_module

optional_module("PySide6")
from PySide6.QtCore import QPoint, QRect, Qt
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

EMB, BOB = (np.random.default_rng(7).normal(size=128).astype(np.float32),
             np.random.default_rng(8).normal(size=128).astype(np.float32))


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


YAWS = [0.0, 0.0, 0.0, 0.0, -0.3, -0.3, -0.3, 0.3, 0.3, 0.3]


class FakeAnalyzer:
    """One face whose pose and exact embedding change every frame, like a real person moving."""

    def __init__(self):
        self.emb, self.n, self.rng = EMB, 0, np.random.default_rng(11)

    def analyze(self, image):
        self.n += 1
        yaw = YAWS[self.n % len(YAWS)]
        lm = np.array([[80, 100], [120, 100], [100 + yaw * 40, 125], [88, 150], [112, 150]], np.float32)
        emb = (self.emb + self.rng.normal(size=128) * 0.7).astype(np.float32)
        return [FaceDetection((60, 260, 260, 60), 0.95, lm, emb)]


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

    analyzer = FakeAnalyzer()
    ctl = EngineController(dict(cfg=cfg, camera=True, voice=False, tts=False,
                                analyzer=analyzer, camera_source_factory=FakeCamera))
    win = MainWindow(ctl)
    win.show()
    ctl.start()
    try:
        def chat():
            return win.chat.toPlainText()

        # startup, live video, recognition + greeting
        assert wait_for(qapp, lambda: "Ready." in chat())
        assert wait_for(qapp, lambda: win.video._image is not None)
        assert wait_for(qapp, lambda: "Hello Alice." in chat())
        assert win.people.item(0).text().startswith("Alice")
        assert win.lbl_voice.text() == "Mic off" and not win.chk_listen.isEnabled()
        assert abs(win._brightness - 90) < 1  # measured from the frames (the fake camera is mid-grey)

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

        # enrolling from the UI (a different person steps in): progress is visible, then the list updates
        analyzer.emb = BOB
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


def test_window_fits_a_small_laptop_screen(qapp):
    win = MainWindow(EngineController())  # not started: layout only
    avail = QRect(0, 0, 1366, 728)  # 1366x768 minus the taskbar
    win.fit_to_screen(avail)
    win.show()
    qapp.processEvents()
    assert win.width() <= avail.width() and win.height() <= avail.height()
    assert win.minimumSizeHint().height() <= 640  # the layout itself is compact enough to fit
    for w in (win.input, win.btn_enroll):  # the bottom controls are inside the window, not clipped
        assert w.mapTo(win, QPoint(0, w.height())).y() <= win.height()
    win.close()


def test_dim_light_hint(qapp):
    win = MainWindow(EngineController())
    win.show()
    win._has_camera = True
    for brightness, shown in ((20.0, True), (140.0, False)):
        win._brightness = brightness
        win._poll_state()
        assert win.lbl_hint.isVisible() is shown
    win._has_camera = False
    win._brightness = 20.0
    win._poll_state()
    assert not win.lbl_hint.isVisible()  # no camera, no hint
    win.close()


def _start(qapp, tmp_path, store_setup=None):
    cfg = Settings(faces_dir=str(tmp_path / "faces"), db_path=str(tmp_path / "a.db"), _env_file=None)
    if store_setup:
        store_setup(FaceStore(cfg.faces_dir))
    analyzer = FakeAnalyzer()
    ctl = EngineController(dict(cfg=cfg, camera=True, voice=False, tts=False,
                                analyzer=analyzer, camera_source_factory=FakeCamera))
    win = MainWindow(ctl)
    win.show()
    ctl.start()
    return win, ctl, analyzer


def test_warns_when_one_person_is_enrolled_under_two_names(qapp, tmp_path):
    def two_names(store):
        for name in ("Asak", "YourName"):  # the same face, enrolled twice
            for _ in range(4):
                store.add(name, (EMB + np.random.default_rng(3).normal(size=128) * 0.5).astype(np.float32))

    win, ctl, _ = _start(qapp, tmp_path, two_names)
    try:
        assert wait_for(qapp, lambda: "Ready." in win.chat.toPlainText())
        assert "look like the same person" in win.chat.toPlainText()
        assert win.lbl_overlap.isVisible() and "Asak and YourName" in win.lbl_overlap.text()
    finally:
        win.close()


def test_add_samples_button_and_live_hint(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "ENROLL_MIN_GAP", 0.0)
    win, ctl, _ = _start(qapp, tmp_path, lambda store: store.add("Alice", EMB))
    try:
        assert wait_for(qapp, lambda: win.people.count() == 1 and win.people.item(0).text().startswith("Alice"))
        win.people.setCurrentRow(0)
        win._poll_state()
        assert win.btn_add.isEnabled()
        win._add_samples()
        assert wait_for(qapp, lambda: not win.btn_add.isEnabled())  # busy: can't start a second enrollment
        seen_hint = wait_for(qapp, lambda: "turn your head" in win.enroll_lbl.text() or "straight" in win.enroll_lbl.text(), 3)
        assert wait_for(qapp, lambda: ctl.enrolled().get("Alice") == 1 + orch_mod.ENROLL_SAMPLES, timeout=12)
        assert seen_hint or "Done." in win.chat.toPlainText()
    finally:
        win.close()


def test_talk_button_opens_a_listening_window_that_is_visible(qapp, tmp_path):
    win, ctl, _ = _start(qapp, tmp_path)
    try:
        assert wait_for(qapp, lambda: "Ready." in win.chat.toPlainText())
        assert not win.btn_talk.isEnabled()  # no microphone in this setup
        win._has_mic = True  # pretend there is one
        win.btn_talk.setEnabled(True)
        win._talk()
        assert wait_for(qapp, lambda: ctl.attention is not None and ctl.attention[0] == "talk")
        win._poll_state()
        assert win.lbl_voice.text().startswith("Your turn")
        # speech without a wake word is now accepted
        ctl.engine.bus.publish_threadsafe(
            Topic.SPEECH_TRANSCRIBED, Transcript("what time is it", "en", 1.0, 1.0, time.monotonic()))
        assert wait_for(qapp, lambda: "It's " in win.chat.toPlainText())
        assert "(ignored" not in win.chat.toPlainText()
        win._poll_state()
        assert win.lbl_voice.text().startswith("Follow-up")  # the conversation continues, and it says so
    finally:
        win.close()


def test_ready_message_names_the_wake_word_and_f9(qapp, tmp_path):
    win, ctl, _ = _start(qapp, tmp_path)
    try:
        assert wait_for(qapp, lambda: "Ready." in win.chat.toPlainText())
        assert 'say "Aether, help"' in win.chat.toPlainText() and "F9" in win.chat.toPlainText()
    finally:
        win.close()


def test_late_refreshes_after_closing_do_not_touch_the_closed_database(qapp, tmp_path):
    """Regression: saying "goodbye" closed the window and the audit database, and a refresh timer that
    was already scheduled then raised 'Cannot operate on a closed database'."""
    win, ctl, _ = _start(qapp, tmp_path)
    assert wait_for(qapp, lambda: "Ready." in win.chat.toPlainText())
    ctl.send_text("goodbye")  # shuts the engine down from the voice/typed command, closing the window
    assert wait_for(qapp, lambda: win._stopped)
    win.refresh_activity()  # the late timers
    win.refresh_people()
    assert ctl.recent_activity() == [] and ctl.enrolled() == {}


def test_verify_button_explains_itself_when_nobody_is_recognized(qapp, tmp_path):
    win, ctl, _ = _start(qapp, tmp_path)
    try:
        assert wait_for(qapp, lambda: "Ready." in win.chat.toPlainText())
        assert win.btn_verify.isEnabled() and win.btn_verify.text() == "Verify me"
        errors = []
        ctl.bridge.error.connect(errors.append)
        win._verify()
        assert wait_for(qapp, lambda: any("recognize" in e for e in errors))
        assert ctl.liveness_hint == "" and ctl.verified_names == []
    finally:
        win.close()
