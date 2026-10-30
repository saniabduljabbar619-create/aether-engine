"""Try the liveness check ("is this a real person, turning their head?") on your webcam.

    python -m examples.liveness_demo               # live: follow the instruction on screen (q quits)
    python -m examples.liveness_demo --selftest    # no camera: a simulated head and a simulated photo

Run it once normally (you should pass), then hold a photo of yourself on a phone in front of the camera
and turn the phone (you should NOT pass). Report the "turn" numbers if either result surprises you.
"""
import argparse
import time

from vision.enrollment import USER_RIGHT
from vision.liveness import LivenessSession
from vision.pose import simulate_landmarks
from vision.detector import FaceDetection
import numpy as np


def selftest() -> int:
    emb = np.zeros(128, np.float32)

    def run(planar: bool) -> LivenessSession:
        s = LivenessSession(direction=USER_RIGHT)
        turns = [0.0] * 6 + [8.0 * i for i in range(1, 6)]
        for i, deg in enumerate(turns * (1 if not planar else 2)):
            face = FaceDetection((0, 0, 100, 100), 0.95, simulate_landmarks(deg, planar=planar), emb)
            s.offer([face], i * 0.1)
        return s

    real, photo = run(False), run(True)
    print(f"real head : {real.phase.name:7s} best turn {real.best_turn:.2f} (needs {real.min_turn})")
    print(f"flat photo: {photo.phase.name:7s} best turn {photo.best_turn:.2f} (needs {photo.min_turn})")
    return 0 if real.phase.name == "PASSED" and photo.phase.name != "PASSED" else 1


def live(camera: int) -> int:
    import cv2

    from vision.detector import FaceAnalyzer

    analyzer = FaceAnalyzer()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Could not open the camera.")
        return 1
    session = LivenessSession()
    print(session.instruction)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        st = session.offer(analyzer.analyze(frame), time.monotonic())
        color = (0, 200, 0) if st.passed else (0, 0, 255) if st.passed is False else (0, 200, 255)
        cv2.putText(frame, st.hint if st.passed is None else (st.reason or st.hint), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"turn {st.turn:.2f} / {session.min_turn}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.imshow("Aether liveness", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if st.passed is not None:
            print("PASSED" if st.passed else f"FAILED: {st.reason}", f"(best turn {st.turn:.2f})")
            print("Press r to try again, q to quit.")
            if cv2.waitKey(0) & 0xFF != ord("r"):
                break
            session = LivenessSession()
            print(session.instruction)
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else live(a.camera))
