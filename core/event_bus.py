"""Asynchronous pub/sub event bus.

Design
------
* Every subscriber owns a bounded ``asyncio.Queue`` and a worker task, so a slow
  subscriber can never block the publisher or other subscribers.
* When a queue is full the subscriber's ``overflow`` policy decides what happens
  (``drop_oldest`` suits video frames: newest data wins).
* Handler exceptions are isolated, counted and logged; they never kill the bus.
* ``publish_threadsafe`` lets non-async threads (camera capture, audio callback)
  push events into the loop.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

log = logging.getLogger("aether.bus")

WILDCARD = "*"
_ids = itertools.count(1)


class Topic(str, Enum):
    FRAME_CAPTURED = "vision.frame_captured"
    FACE_DETECTED = "vision.face_detected"
    FACE_RECOGNIZED = "vision.face_recognized"
    FACES_UPDATED = "vision.faces_updated"
    SPEECH_STARTED = "audio.speech_started"
    SPEECH_TRANSCRIBED = "audio.speech_transcribed"
    COMMAND_ISSUED = "system.command_issued"
    SYSTEM_ERROR = "system.error"


class Overflow(str, Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


@dataclass(frozen=True)
class Event:
    topic: str
    payload: Any = None
    timestamp: float = field(default_factory=time.monotonic)
    id: int = field(default_factory=lambda: next(_ids))


Handler = Callable[[Event], Awaitable[None]]


@dataclass
class Subscription:
    topic: str
    handler: Handler
    queue: "asyncio.Queue[Optional[Event]]"
    overflow: Overflow
    task: Optional[asyncio.Task] = None
    processed: int = 0
    dropped: int = 0
    errors: int = 0

    @property
    def name(self) -> str:
        return getattr(self.handler, "__qualname__", repr(self.handler))


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Subscription]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._running = True
        for sub in self.subscriptions():
            self._spawn(sub)

    async def stop(self, drain: bool = True) -> None:
        """Stop workers. With ``drain`` queued events are handled first."""
        self._running = False
        subs = self.subscriptions()
        for sub in subs:
            if not sub.task:
                continue
            if drain:
                await sub.queue.put(None)  # sentinel; waits for room, handled after pending events
            else:
                sub.task.cancel()
        tasks = [s.task for s in subs if s.task]
        await asyncio.gather(*tasks, return_exceptions=True)
        for sub in subs:
            sub.task = None

    # ---- subscribe -------------------------------------------------------
    def subscribe(
        self,
        topic: "Topic | str",
        handler: Handler,
        *,
        max_queue: int = 100,
        overflow: Overflow = Overflow.DROP_OLDEST,
    ) -> Subscription:
        key = topic.value if isinstance(topic, Topic) else topic
        sub = Subscription(key, handler, asyncio.Queue(max_queue), overflow)
        self._subs.setdefault(key, []).append(sub)
        if self._running:
            self._spawn(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subs.get(sub.topic, []).remove(sub)
        if sub.task:
            sub.task.cancel()

    def subscriptions(self) -> List[Subscription]:
        return [s for subs in self._subs.values() for s in subs]

    # ---- publish ---------------------------------------------------------
    def publish(self, topic: "Topic | str", payload: Any = None) -> Event:
        """Non-blocking publish; call from the event-loop thread."""
        key = topic.value if isinstance(topic, Topic) else topic
        event = Event(key, payload)
        for sub in (*self._subs.get(key, ()), *self._subs.get(WILDCARD, ())):
            self._enqueue(sub, event)
        return event

    def publish_threadsafe(self, topic: "Topic | str", payload: Any = None) -> None:
        """Publish from any thread (camera capture, audio callback)."""
        if self._loop is None:
            raise RuntimeError("EventBus.start() must be awaited first")
        self._loop.call_soon_threadsafe(self.publish, topic, payload)

    # ---- internals -------------------------------------------------------
    def _enqueue(self, sub: Subscription, event: Event) -> None:
        q = sub.queue
        if q.full():
            sub.dropped += 1
            if sub.overflow is Overflow.DROP_NEWEST:
                return
            q.get_nowait()  # DROP_OLDEST
        q.put_nowait(event)

    def _spawn(self, sub: Subscription) -> None:
        if sub.task is None:
            sub.task = asyncio.create_task(self._worker(sub), name=f"bus:{sub.name}")

    async def _worker(self, sub: Subscription) -> None:
        while True:
            event = await sub.queue.get()
            if event is None:
                return
            try:
                await sub.handler(event)
                sub.processed += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                sub.errors += 1
                log.exception("handler %s failed on %s", sub.name, event.topic)

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {
            f"{s.topic} -> {s.name}": {
                "processed": s.processed, "dropped": s.dropped,
                "errors": s.errors, "queued": s.queue.qsize(),
            }
            for s in self.subscriptions()
        }
