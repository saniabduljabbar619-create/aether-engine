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
from vision.tracker import UNKNOWN

GREEN, YELLOW, RED = (0, 220, 0), (0, 220, 255), (0, 0, 255)


def draw(img, tracks, status, banner=None):
    for t in tracks:
        top, right, bottom, left = t.bbox
        if t.identity is None:
            color, label = YELLOW, "verifying..."
        elif t.identity == UNKNOWN:
            color, label = RED, "Unknown"
        else:
            color, label = GREEN, f"{t.identity} {t.score:.2f}"
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom), (right, bottom + 24), color, -1)
        cv2.putText(img, label, (left + 4, bottom + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(img, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    if banner:
        cv2.putText(img, banner, (12, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 3)
    return img


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
