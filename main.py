"""Aether-Vision Engine: application bootstrap.

    python main.py                  camera + microphone + voice replies
    python main.py --no-voice       no microphone (camera only)
    python main.py --no-camera      no camera (voice only)
    python main.py --no-window      no preview window
    python main.py --no-tts         print replies instead of speaking them

Say "Aether, help" to hear the commands. Press q in the window, or say "Aether, goodbye", to quit.
"""
import argparse
import asyncio
import logging
import time

from config.settings import get_settings
from core.event_bus import Event, EventBus, Topic
from core.logger import setup_logging
from core.orchestrator import Orchestrator
from storage.face_store import FaceStore
from storage.local_db import AuditLog
from vision.spatial_engine import VectorIdentityMatcher

log = logging.getLogger("aether.main")


async def run(args: argparse.Namespace) -> None:
    cfg = get_settings()
    bus = EventBus()
    await bus.start()

    store = FaceStore(cfg.faces_dir)
    matcher = VectorIdentityMatcher(cfg.similarity_threshold)
    matcher.register_identities(store.load_all())
    audit = AuditLog(cfg.db_path)

    speaker = None
    if not args.no_tts:
        from audio.tts import Pyttsx3Speaker
        speaker = Pyttsx3Speaker()

    cleanup = []  # callables, run in reverse order on exit
    banner = {"text": None, "until": 0.0}
    state = {"cap": None}

    def on_reply(text: str) -> None:
        print(f"Aether: {text}")
        banner.update(text=text, until=time.monotonic() + 5)

    def status() -> str:
        cap = state["cap"]
        return f"Camera is running at {cap.fps:.0f} frames per second." if cap else "The camera is off."

    orch = Orchestrator(bus, matcher, store, audit, speaker, status_fn=status, on_reply=on_reply)
    orch.start()

    # ---- load heavy models in parallel, without freezing the event loop ----
    loaders = {}
    if not args.no_camera:
        from vision.detector import FaceAnalyzer
        loaders["analyzer"] = asyncio.to_thread(FaceAnalyzer)
    if not args.no_voice:
        from audio.stt_engine import FasterWhisperEngine
        loaders["engine"] = asyncio.to_thread(
            FasterWhisperEngine, cfg.whisper_model, cfg.whisper_device, language=cfg.whisper_language)
    print("Loading models (the first run downloads them)...")
    models = dict(zip(loaders, await asyncio.gather(*loaders.values()))) if loaders else {}

    # ---- vision ----
    if "analyzer" in models:
        from vision.capture import CameraError, FrameCapture
        from vision.detector import DetectorService
        from vision.recognition import RecognitionService

        cap = FrameCapture(bus, camera_index=args.camera if args.camera is not None else cfg.camera_index)
        DetectorService(bus, models["analyzer"]).start()
        RecognitionService(bus, matcher).start()
        try:
            await cap.start()
            state["cap"] = cap
            cleanup.append(cap.stop)
        except CameraError as exc:
            print(f"Camera unavailable: {exc}\nContinuing without vision.")

        if not args.no_window and state["cap"]:
            import cv2
            from ui.overlay import draw

            async def show(e: Event) -> None:
                u = e.payload
                text = banner["text"] if time.monotonic() < banner["until"] else None
                status_line = f"{cap.fps:4.1f} FPS | {u.inference_ms:3.0f} ms | {len(u.tracks)} face(s)"
                cv2.imshow("Aether", draw(u.frame.image.copy(), u.tracks, status_line, text))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    orch.shutdown.set()

            bus.subscribe(Topic.FACES_UPDATED, show, max_queue=1)
            cleanup.append(cv2.destroyAllWindows)

    # ---- voice ----
    if "engine" in models:
        from audio.recorder import MicRecorder, MicrophoneError
        from audio.stt_engine import SttService

        SttService(bus, models["engine"]).start()
        mic = MicRecorder(bus, device=args.mic if args.mic is not None else cfg.mic_device)
        try:
            mic.start()
            cleanup.append(lambda: asyncio.to_thread(mic.stop))
        except MicrophoneError as exc:
            print(f"Microphone unavailable: {exc}\nContinuing without voice.")

    print("Ready. Say 'Aether, help'. Ctrl+C to quit.")
    try:
        await orch.shutdown.wait()
        await asyncio.sleep(1.5)  # let the goodbye be spoken
    finally:
        for fn in reversed(cleanup):
            try:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                log.exception("cleanup failed")
        if speaker:
            speaker.close()
        await bus.stop()
        audit.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aether-Vision Engine")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--mic", type=int, default=None)
    a = ap.parse_args()
    setup_logging(get_settings().log_level)
    try:
        asyncio.run(run(a))
    except KeyboardInterrupt:
        print("\nbye")
