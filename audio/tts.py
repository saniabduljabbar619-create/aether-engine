"""Text-to-speech (pyttsx3: Windows SAPI5, macOS NSSpeech, Linux espeak) in its own thread.

``say`` never blocks. ``was_speaking_at`` lets the orchestrator ignore microphone audio that
was captured while we were talking, otherwise the assistant would hear and obey itself.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import List, Optional, Protocol, Tuple

log = logging.getLogger("aether.tts")

_LEAD, _TAIL = 0.3, 0.8  # seconds of margin before/after speech (speaker latency, room echo)


class Speaker(Protocol):
    def say(self, text: str) -> None: ...
    def was_speaking_at(self, t: float) -> bool: ...
    def close(self) -> None: ...


class NullSpeaker:
    """Used with --no-tts, or when no voice engine is available."""

    def say(self, text: str) -> None: ...
    def was_speaking_at(self, t: float) -> bool:
        return False

    def close(self) -> None: ...


class Pyttsx3Speaker:
    def __init__(self, rate: int = 175) -> None:
        self._rate = rate
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lock = threading.Lock()
        self._spans: List[Tuple[float, Optional[float]]] = []  # (start, end or None while speaking)
        self._ready = threading.Event()
        self.available = False
        self._thread = threading.Thread(target=self._run, name="aether-tts", daemon=True)
        self._thread.start()
        self._ready.wait(10)

    def say(self, text: str) -> None:
        if self.available:
            self._q.put(text)

    def was_speaking_at(self, t: float) -> bool:
        with self._lock:
            return any(s - _LEAD <= t <= (e if e is not None else float("inf")) + _TAIL
                       for s, e in self._spans[-20:])

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=3)

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()  # must be created in the thread that uses it (Windows COM)
            engine.setProperty("rate", self._rate)
            self.available = True
        except Exception as exc:
            log.warning("text-to-speech unavailable (%s); replies will be text only", exc)
            return
        finally:
            self._ready.set()
        while (text := self._q.get()) is not None:
            with self._lock:
                self._spans.append((time.monotonic(), None))
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                log.exception("speech failed")
            with self._lock:
                self._spans[-1] = (self._spans[-1][0], time.monotonic())
