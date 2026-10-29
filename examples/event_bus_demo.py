"""Simulates a camera thread and a mic thread publishing into the bus.

Run from the repo root:  python -m examples.event_bus_demo
Shows: topic routing, a slow subscriber dropping old frames, error isolation.
"""
import asyncio
import threading
import time

from core.event_bus import WILDCARD, EventBus, Event, Topic


def camera_thread(bus: EventBus) -> None:
    for i in range(30):  # ~100 FPS burst
        bus.publish_threadsafe(Topic.FRAME_CAPTURED, f"frame#{i}")
        if i == 12:
            bus.publish_threadsafe(Topic.FACE_RECOGNIZED, {"name": "alice", "score": 0.91})
        time.sleep(0.01)


def mic_thread(bus: EventBus) -> None:
    time.sleep(0.1)
    bus.publish_threadsafe(Topic.SPEECH_TRANSCRIBED, "lock the front door")


async def main() -> None:
    bus = EventBus()
    t0 = time.monotonic()

    def stamp(msg: str) -> None:
        print(f"[{time.monotonic() - t0:5.2f}s] {msg}")

    async def greeter(e: Event) -> None:
        stamp(f"GREETER   hello {e.payload['name']} (score {e.payload['score']})")

    async def commander(e: Event) -> None:
        stamp(f"COMMANDER executing: {e.payload!r}")
        bus.publish(Topic.COMMAND_ISSUED, e.payload)

    async def slow_ui(e: Event) -> None:
        stamp(f"SLOW UI   rendering {e.payload}")
        await asyncio.sleep(0.08)  # slower than the camera, so frames get dropped

    async def flaky(e: Event) -> None:
        raise RuntimeError("audit logger crashed")

    async def audit(e: Event) -> None:
        stamp(f"AUDIT     {e.topic}")

    bus.subscribe(Topic.FACE_RECOGNIZED, greeter)
    bus.subscribe(Topic.SPEECH_TRANSCRIBED, commander)
    ui = bus.subscribe(Topic.FRAME_CAPTURED, slow_ui, max_queue=2)  # drop_oldest
    bus.subscribe(Topic.FACE_RECOGNIZED, flaky)
    bus.subscribe(Topic.COMMAND_ISSUED, audit)

    await bus.start()
    threads = [threading.Thread(target=camera_thread, args=(bus,)),
               threading.Thread(target=mic_thread, args=(bus,))]
    for t in threads:
        t.start()
    for t in threads:
        await asyncio.to_thread(t.join)
    await asyncio.sleep(0.3)
    await bus.stop()

    print("\n--- stats ---")
    for name, s in bus.stats().items():
        print(f"{name:55s} {s}")
    print(f"\nframes published: 30 | UI rendered: {ui.processed} | dropped: {ui.dropped}")


if __name__ == "__main__":
    asyncio.run(main())
