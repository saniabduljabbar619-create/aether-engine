"""Bridges between the asyncio engine and the Qt GUI.

Threads:  Qt main thread (all widgets)  <-- signals -->  engine thread (asyncio loop, all I/O).

* ``EngineThread``   runs an asyncio loop in a background thread.
* ``QtBridge``       subscribes to the event bus and re-emits events as Qt signals. Emitting a
                     signal from another thread is safe: Qt queues it to the GUI thread.
* ``EngineController`` is the only object the window talks to; every method is safe to call
                     from the GUI thread (work that must happen on the loop is submitted to it).

Video frames are the one high-rate stream, so they go through a mailbox: the bridge keeps only
the newest frame and emits ``frame_available`` only when the GUI has consumed the previous one.
A slow GUI therefore skips frames instead of building a backlog.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Awaitable, List, Optional, Tuple, TypeVar

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from audio.stt_engine import TYPED, Transcript
from core.engine import Engine
from core.event_bus import Event, Topic
from vision.recognition import FacesUpdate

T = TypeVar("T")
TYPED_PREFIX = "Aether, "


@dataclass(frozen=True)
class FrameInfo:
    tracks: list
    inference_ms: float
    brightness: float = 255.0  # mean pixel value 0-255, to warn about dim light


class EngineThread:
    """An asyncio event loop living in its own thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="aether-engine", daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> None:
        self._thread.start()

    def submit(self, coro: Awaitable[T]) -> "Future[T]":
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        if not self._thread.is_alive():  # never started, or already stopped
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


class QtBridge(QObject):
    frame_available = Signal()
    reply = Signal(str)
    heard = Signal(str, bool)  # text, addressed to the assistant?
    voice_state = Signal(str)  # "listening" | "hearing" | "transcribing"
    people_changed = Signal()
    activity = Signal()
    error = Signal(str)
    status = Signal(str)
    shutdown_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._mailbox: Optional[Tuple[QImage, FrameInfo]] = None
        self._pending = False

    # ---- frame mailbox ---------------------------------------------------
    def take_frame(self) -> Optional[Tuple[QImage, FrameInfo]]:
        """Called by the GUI thread: newest frame, and re-arms the next ``frame_available``."""
        with self._lock:
            item, self._mailbox, self._pending = self._mailbox, None, False
        return item

    def _push_frame(self, image: QImage, info: FrameInfo) -> None:
        with self._lock:
            self._mailbox = (image, info)
            if self._pending:
                return
            self._pending = True
        self.frame_available.emit()

    # ---- bus -> signals (run on the engine thread) -----------------------
    def attach(self, engine: Engine) -> None:
        sub = engine.bus.subscribe
        sub(Topic.FACES_UPDATED, self._on_view, max_queue=1)
        sub(Topic.SPEECH_STARTED, lambda e: self._emit(self.voice_state, "hearing"))
        sub(Topic.SPEECH_CAPTURED, lambda e: self._emit(self.voice_state, "transcribing"))
        sub(Topic.SPEECH_TRANSCRIBED, lambda e: self._on_heard(engine, e), max_queue=16)
        sub(Topic.PEOPLE_CHANGED, lambda e: self._emit(self.people_changed))
        for topic in (Topic.COMMAND_ISSUED, Topic.FACE_RECOGNIZED, Topic.PEOPLE_CHANGED):
            sub(topic, lambda e: self._emit(self.activity))
        sub(Topic.SYSTEM_ERROR, lambda e: self._emit(self.error, str(e.payload)))

    @staticmethod
    async def _emit(signal, *args) -> None:  # handlers must be coroutines
        signal.emit(*args)

    async def _on_view(self, e: Event) -> None:
        u: FacesUpdate = e.payload
        img = u.frame.image
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[:, :, ::-1])  # BGR -> RGB
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()  # own the pixels
        self._push_frame(qimg, FrameInfo(u.tracks, u.inference_ms, float(img[::16, ::16].mean())))

    async def _on_heard(self, engine: Engine, e: Event) -> None:
        t: Transcript = e.payload
        text = t.text[len(TYPED_PREFIX):] if t.language == TYPED else t.text
        addressed = engine.orch.router.parse(t.text) is not None
        self.heard.emit(text, addressed)
        self.voice_state.emit("listening")


class EngineController(QObject):
    """Everything the window needs, with thread-safe methods."""

    started = Signal(list, bool, bool)  # warnings, has camera, has microphone

    def __init__(self, engine_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        self.bridge = QtBridge()
        self._kwargs = engine_kwargs or {}
        self._thread = EngineThread()
        self.engine: Optional[Engine] = None
        self._stopped = False
        self._watch: Optional[asyncio.Task] = None

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._thread.start()
        fut = self._thread.submit(self._boot())
        fut.add_done_callback(self._boot_done)

    async def _boot(self) -> None:
        self.engine = Engine(on_reply=self.bridge.reply.emit, on_status=self.bridge.status.emit, **self._kwargs)
        self.bridge.attach(self.engine)
        await self.engine.start()
        self._watch = asyncio.create_task(self._watch_shutdown())

    async def _watch_shutdown(self) -> None:
        await self.engine.orch.shutdown.wait()  # voice "goodbye"
        self.bridge.shutdown_requested.emit()

    async def _shutdown_engine(self) -> None:
        if self._watch:
            self._watch.cancel()
        await self.engine.stop()

    def _boot_done(self, fut: Future) -> None:
        exc = fut.exception()
        if exc is not None:
            self.bridge.error.emit(f"Failed to start: {exc}")
            return
        e = self.engine
        self.started.emit(list(e.warnings), e.cap is not None, e.mic is not None)

    def stop(self) -> None:
        if self._stopped:  # safe to call from both the window's close and the app's exit
            return
        self._stopped = True
        if self.engine is not None:
            try:
                self._thread.submit(self._shutdown_engine()).result(timeout=10)
            except Exception:
                pass
        self._thread.stop()

    # ---- actions (GUI thread -> engine thread) ---------------------------
    def send_text(self, text: str) -> None:
        """Run a typed command as if it had been spoken after the wake word."""
        if self.engine is None or not text.strip():
            return
        t = Transcript(TYPED_PREFIX + text.strip(), TYPED, 0.0, 0.0, time.monotonic())
        self.engine.bus.publish_threadsafe(Topic.SPEECH_TRANSCRIBED, t)

    def begin_enrollment(self, name: str) -> "Future[str]":
        async def go() -> str:
            return self.engine.orch.begin_enrollment(name)

        return self._thread.submit(go())

    def forget(self, name: str) -> "Future[int]":
        async def go() -> int:
            return self.engine.orch.forget_person(name)

        return self._thread.submit(go())

    def set_listening(self, on: bool) -> None:
        if self.engine:
            self.engine.orch.listening = on

    def set_muted(self, muted: bool) -> None:
        if self.engine and self.engine.speaker:
            self.engine.speaker.muted = muted

    # ---- read-only state (safe to poll from the GUI thread) --------------
    @property
    def fps(self) -> float:
        return self.engine.cap.fps if self.engine and self.engine.cap else 0.0

    @property
    def mic_level_db(self) -> float:
        return self.engine.mic.level_db if self.engine and self.engine.mic else -100.0

    @property
    def enrollment_progress(self):
        return self.engine.orch.enrollment_progress if self.engine and self.engine.orch else None

    def enrolled(self) -> dict:
        """{name: number of samples}"""
        if self.engine is None:
            return {}
        return {name: len(v) for name, v in self.engine.store.load_all().items()}

    def recent_activity(self, n: int = 100) -> List[Tuple[str, str, str]]:
        return self.engine.audit.recent(n) if self.engine else []
