"""Live camera preview with FPS overlay.   Run: python -m examples.webcam_demo
Press q in the window to quit.  --synthetic uses a fake camera (no hardware needed);
--save PATH writes one annotated frame to PATH and exits.
"""
import argparse
import asyncio
import time

import cv2
import numpy as np

from core.event_bus import EventBus, Event, Topic
from vision.capture import CameraError, FrameCapture


class SyntheticCamera:
    """Moving test pattern at ~30 FPS."""

    def __init__(self, w=640, h=480):
        self.w, self.h, self.t = w, h, 0

    def isOpened(self):
        return True

    def read(self):
        time.sleep(1 / 30)
        self.t += 1
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[:] = (40, 30, 30)
        x = int((self.t * 7) % (self.w - 120))
        cv2.rectangle(img, (x, 180), (x + 120, 300), (80, 200, 120), -1)
        cv2.putText(img, "synthetic camera", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return True, img

    def release(self):
        pass


async def main(args) -> None:
    bus = EventBus()
    factory = (lambda: SyntheticCamera()) if args.synthetic else None
    cap = FrameCapture(bus, camera_index=args.camera, source_factory=factory)
    done = asyncio.Event()

    async def show(e: Event) -> None:
        img = e.payload.image.copy()
        cv2.putText(img, f"{cap.fps:5.1f} FPS  frame {e.payload.frame_id}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if args.save:
            if e.payload.frame_id >= 40:  # let the FPS estimate settle
                cv2.imwrite(args.save, img)
                done.set()
            return
        cv2.imshow("Aether camera", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            done.set()

    bus.subscribe(Topic.FRAME_CAPTURED, show, max_queue=1)  # newest frame only
    await bus.start()
    try:
        await cap.start()
    except CameraError as exc:
        print(f"Camera error: {exc}")
        await bus.stop()
        return
    await done.wait()
    await cap.stop()
    await bus.stop()
    if not args.save:
        cv2.destroyAllWindows()
    print(f"stopped. last FPS estimate: {cap.fps:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--save")
    asyncio.run(main(ap.parse_args()))
