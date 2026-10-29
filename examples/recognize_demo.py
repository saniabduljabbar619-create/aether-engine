"""Live face recognition.   Run: python -m examples.recognize_demo     (q to quit)
Enroll first:  python -m examples.enroll <your-name>
Green = recognized, yellow = verifying, red = unknown.
"""
import argparse
import asyncio
import time

import cv2

from config.settings import get_settings
from core.event_bus import Event, EventBus, Topic
from storage.face_store import FaceStore
from vision.capture import CameraError, FrameCapture
from vision.detector import DetectorService, FaceAnalyzer
from vision.recognition import FacesUpdate, Recognition, RecognitionService
from vision.spatial_engine import VectorIdentityMatcher
from ui.overlay import draw

async def main(args) -> None:
    cfg = get_settings()
    matcher = VectorIdentityMatcher(cfg.similarity_threshold)
    db = FaceStore(cfg.faces_dir).load_all()
    if not db:
        print("No enrolled faces yet. Run:  python -m examples.enroll <your-name>")
    matcher.register_identities(db)

    bus = EventBus()
    cap = FrameCapture(bus, camera_index=args.camera)
    print("loading models...")
    analyzer = await asyncio.to_thread(FaceAnalyzer)
    done, banner = asyncio.Event(), {"text": None, "until": 0.0}

    async def on_recognized(e: Event) -> None:
        r: Recognition = e.payload
        print(f"Hello {r.name}! (similarity {r.score:.2f})")
        banner.update(text=f"Hello {r.name}!", until=time.monotonic() + 3)

    async def show(e: Event) -> None:
        u: FacesUpdate = e.payload
        status = f"cam {cap.fps:4.1f} FPS | detect {u.inference_ms:3.0f} ms | {len(u.tracks)} face(s)"
        text = banner["text"] if time.monotonic() < banner["until"] else None
        cv2.imshow("Aether recognition", draw(u.frame.image.copy(), u.tracks, status, text))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            done.set()

    bus.subscribe(Topic.FACES_UPDATED, show, max_queue=1)
    bus.subscribe(Topic.FACE_RECOGNIZED, on_recognized)
    await bus.start()
    DetectorService(bus, analyzer).start()
    RecognitionService(bus, matcher).start()
    try:
        await cap.start()
    except CameraError as exc:
        print(f"Camera error: {exc}")
        await bus.stop()
        return
    await done.wait()
    await cap.stop()
    await bus.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
