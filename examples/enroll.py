"""Enroll a face.
  python -m examples.enroll alice                 # webcam: captures 10 samples, press q to cancel
  python -m examples.enroll alice --image a.jpg b.jpg
  python -m examples.enroll --list
  python -m examples.enroll --remove alice
Embeddings are stored locally in storage/reference_faces/<name>/ and never leave this machine.
"""
import argparse
import asyncio
import time

import cv2

from config.settings import get_settings
from core.event_bus import Event, EventBus, Topic
from storage.face_store import FaceStore
from vision.capture import CameraError, FrameCapture
from vision.detector import DetectorService, FaceAnalyzer, FaceEvent

SAMPLES = 10
MIN_GAP = 0.6  # seconds between samples, so they differ slightly
MIN_SCORE = 0.9
MIN_FACE_HEIGHT = 80  # px; too small gives poor embeddings


def enroll_images(store: FaceStore, name: str, paths) -> None:
    analyzer = FaceAnalyzer()
    for p in paths:
        img = cv2.imread(p)
        faces = analyzer.analyze(img) if img is not None else []
        if len(faces) != 1:
            print(f"skipped {p}: need exactly one face, found {len(faces)}")
            continue
        store.add(name, faces[0].embedding)
        print(f"added {p}")


async def enroll_webcam(store: FaceStore, name: str, camera: int) -> None:
    bus = EventBus()
    cap = FrameCapture(bus, camera_index=camera)
    print("loading models...")
    analyzer = await asyncio.to_thread(FaceAnalyzer)
    samples, state = [], {"last": 0.0, "cancel": False}
    done = asyncio.Event()

    async def on_faces(e: Event) -> None:
        ev: FaceEvent = e.payload
        img = ev.frame.image.copy()
        ok = len(ev.faces) == 1 and ev.faces[0].score >= MIN_SCORE
        if ok:
            top, right, bottom, left = ev.faces[0].bbox
            ok = (bottom - top) >= MIN_FACE_HEIGHT
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0) if ok else (0, 165, 255), 2)
        now = time.monotonic()
        if ok and now - state["last"] >= MIN_GAP:
            samples.append(ev.faces[0].embedding)
            state["last"] = now
        msg = (f"{len(samples)}/{SAMPLES} - turn your head slightly" if ok
               else "show exactly one face, closer to the camera")
        cv2.putText(img, f"Enrolling {name}: {msg}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Aether enroll", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            state["cancel"] = True
        if state["cancel"] or len(samples) >= SAMPLES:
            done.set()

    bus.subscribe(Topic.FACE_DETECTED, on_faces, max_queue=1)
    await bus.start()
    DetectorService(bus, analyzer).start()
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
    if state["cancel"]:
        print("cancelled, nothing saved")
        return
    for emb in samples:
        store.add(name, emb)
    print(f"saved {len(samples)} samples for {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--image", nargs="+")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove")
    a = ap.parse_args()
    store = FaceStore(get_settings().faces_dir)
    if a.list:
        for n, v in store.load_all().items():
            print(f"{n}: {len(v)} samples")
    elif a.remove:
        print(f"removed {store.remove(a.remove)} samples")
    elif not a.name:
        ap.error("give a name, or use --list / --remove")
    elif a.image:
        enroll_images(store, a.name, a.image)
    else:
        asyncio.run(enroll_webcam(store, a.name, a.camera))
