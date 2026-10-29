"""Live face detection.   Run: python -m examples.detect_demo     (q to quit)
First run downloads two small model files (~39 MB) into ./models.
--image PATH analyzes one image instead of the webcam; --save PATH writes the annotated result.
"""
import argparse
import asyncio

import cv2

from core.event_bus import Event, EventBus, Topic
from vision.capture import CameraError, FrameCapture
from vision.detector import DetectorService, FaceAnalyzer, FaceEvent


def annotate(img, faces, status):
    for f in faces:
        top, right, bottom, left = f.bbox
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, f"{f.score:.2f}", (left, max(top - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for x, y in f.landmarks:
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.putText(img, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def run_image(args) -> None:
    img = cv2.imread(args.image)
    an = FaceAnalyzer()
    faces = an.analyze(img)
    print(f"{len(faces)} face(s); embedding dim = {faces[0].embedding.shape[0] if faces else '-'}")
    out = annotate(img, faces, f"{len(faces)} face(s)")
    if args.save:
        cv2.imwrite(args.save, out)


async def run_camera(args) -> None:
    bus = EventBus()
    cap = FrameCapture(bus, camera_index=args.camera)
    print("loading models (first run downloads them)...")
    analyzer = await asyncio.to_thread(FaceAnalyzer)
    done = asyncio.Event()

    async def show(e: Event) -> None:
        ev: FaceEvent = e.payload
        status = f"cam {cap.fps:4.1f} FPS | {len(ev.faces)} face(s) | detect {ev.inference_ms:4.0f} ms"
        cv2.imshow("Aether detector", annotate(ev.frame.image.copy(), ev.faces, status))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            done.set()

    bus.subscribe(Topic.FACE_DETECTED, show, max_queue=1)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image")
    ap.add_argument("--save")
    a = ap.parse_args()
    run_image(a) if a.image else asyncio.run(run_camera(a))
