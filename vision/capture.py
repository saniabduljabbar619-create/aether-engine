"""Non-blocking camera capture.

A dedicated thread reads frames as fast as the camera delivers them, so slow
consumers (face models, UI) can never lower the capture rate. Every frame is
published to the bus as ``Topic.FRAME_CAPTURED``; per-subscriber queues on the
bus decide what slow consumers get to see (newest wins by default).
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple

from core.event_bus import EventBus, Topic

log = logging.getLogger("aether.capture")


class CameraError(RuntimeError):
    """Camera could not be opened."""


class FrameSource(Protocol):
    """The subset of cv2.VideoCapture we rely on, so tests can fake it."""

    def isOpened(self) -> bool: ...
    def read(self) -> Tuple[bool, Any]: ...
    def release(self) -> None: ...


@dataclass(frozen=True)
class Frame:
    frame_id: int
    timestamp: float  # time.monotonic() at capture
    image: Any  # numpy array, BGR, shape (H, W, 3)


def open_opencv_camera(index: int, width: int, height: int) -> FrameSource:
    """Open a real camera. DirectShow on Windows opens far faster than the default MSMF."""
    import cv2  # lazy: only needed with real hardware

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't let the driver queue stale frames
    return cap


class FrameCapture:
    def __init__(
        self,
        bus: EventBus,
        *,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        max_failures: int = 30,
        reconnect_delay: float = 1.0,
        source_factory: Optional[Callable[[], FrameSource]] = None,
    ) -> None:
        self._bus = bus
        self._factory = source_factory or (lambda: open_opencv_camera(camera_index, width, height))
        self._max_failures = max_failures
        self._reconnect_delay = reconnect_delay
        self._source: Optional[FrameSource] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._fps = 0.0
        self._frame_id = 0

    # ---- public API ------------------------------------------------------
    async def start(self) -> None:
        """Open the camera (off the event loop) and start the capture thread."""
        if self._thread:
            return
        self._source = await asyncio.to_thread(self._open)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="aether-capture", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._stop.set()
        if self._thread:
            await asyncio.to_thread(self._thread.join, 5.0)
            self._thread = None
        self._release()

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    @property
    def fps(self) -> float:
        return self._fps

    # ---- internals -------------------------------------------------------
    def _open(self) -> FrameSource:
        src = self._factory()
        if not src.isOpened():
            src.release()
            raise CameraError("could not open camera (in use by another app, or wrong index?)")
        return src

    def _release(self) -> None:
        if self._source is not None:
            self._source.release()
            self._source = None

    def _run(self) -> None:
        failures = 0
        last = time.monotonic()
        while not self._stop.is_set():
            ok, image = self._source.read()
            if not ok:
                failures += 1
                if failures >= self._max_failures and not self._reconnect():
                    return
                time.sleep(0.01)
                continue
            failures = 0

            now = time.monotonic()
            inst = 1.0 / max(now - last, 1e-6)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst  # EMA
            last = now

            self._frame_id += 1
            frame = Frame(self._frame_id, now, image)
            with self._lock:
                self._latest = frame
            self._bus.publish_threadsafe(Topic.FRAME_CAPTURED, frame)

    def _reconnect(self) -> bool:
        """Camera stopped delivering frames: report it, then retry until stopped."""
        self._bus.publish_threadsafe(Topic.SYSTEM_ERROR, "camera lost, reconnecting")
        log.warning("camera lost, reconnecting")
        self._release()
        while not self._stop.wait(self._reconnect_delay):
            try:
                self._source = self._open()
                log.info("camera reconnected")
                return True
            except Exception:
                log.debug("reconnect failed", exc_info=True)
        return False
