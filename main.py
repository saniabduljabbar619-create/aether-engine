"""Aether-Vision Engine: application bootstrap.

    python main.py --gui            desktop window (PySide6)
    python main.py                  classic OpenCV window
    python main.py --no-voice       no microphone      (also: --no-camera, --no-window, --no-tts)

Say "Aether, help" to hear the commands. Press q in the classic window, or say "Aether, goodbye", to quit.
"""
import argparse
import asyncio
import logging
import time

from config.settings import get_settings
from core.engine import Engine
from core.event_bus import Event, Topic
from core.logger import setup_logging

log = logging.getLogger("aether.main")


async def run_classic(args: argparse.Namespace) -> None:
    banner = {"text": None, "until": 0.0}

    def on_reply(text: str) -> None:
        print(f"Aether: {text}")
        banner.update(text=text, until=time.monotonic() + 5)

    engine = Engine(camera=not args.no_camera, voice=not args.no_voice, tts=not args.no_tts,
                    camera_index=args.camera, mic_device=args.mic, on_reply=on_reply, on_status=print)
    print("Loading models (the first run downloads them)...")
    await engine.start()
    for warning in engine.warnings:
        print(warning)

    window = None
    if engine.cap and not args.no_window:
        import cv2
        from ui.overlay import draw

        async def show(e: Event) -> None:
            u = e.payload
            text = banner["text"] if time.monotonic() < banner["until"] else None
            status_line = f"{engine.cap.fps:4.1f} FPS | {u.inference_ms:3.0f} ms | {len(u.tracks)} face(s)"
            cv2.imshow("Aether", draw(u.frame.image.copy(), u.tracks, status_line, text))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                engine.orch.shutdown.set()

        engine.bus.subscribe(Topic.FACES_UPDATED, show, max_queue=1)
        window = cv2

    print(f"Ready. Say '{engine.orch.router.wake_name}, help'. Ctrl+C to quit.")
    try:
        await engine.orch.shutdown.wait()
        await asyncio.sleep(1.5)  # let the goodbye be spoken
    finally:
        if window:
            window.destroyAllWindows()
        await engine.stop()


async def run_server(args: argparse.Namespace) -> None:
    import uvicorn
    from server.api import create_app

    engine = Engine(
        camera=not args.no_camera,
        voice=not args.no_voice,
        tts=not args.no_tts,
        camera_index=args.camera,
        mic_device=args.mic,
        on_reply=lambda text: print(f"Aether: {text}"),
        on_status=print,
    )
    print("Starting Aether Engine...")
    await engine.start()
    for warning in engine.warnings:
        print(warning)

    app = create_app(engine)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    print(f"\nAether API Server running at http://{args.host}:{args.port}")
    print(f"Swagger API Documentation: http://localhost:{args.port}/docs")
    print(f"Live Camera Stream:        http://localhost:{args.port}/api/stream\n")
    try:
        await server.serve()
    finally:
        await engine.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aether-Vision Engine")
    ap.add_argument("--gui", action="store_true", help="desktop window (needs PySide6)")
    ap.add_argument("--server", action="store_true", help="run headless FastAPI server with REST and WebSockets")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="server bind host (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8000, help="server bind port (default: 8000)")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--mic", type=int, default=None)
    a = ap.parse_args()
    setup_logging(get_settings().log_level)
    if a.gui:
        from ui.app import run_gui

        raise SystemExit(run_gui(a))
    if a.server:
        try:
            asyncio.run(run_server(a))
        except KeyboardInterrupt:
            print("\nbye")
        raise SystemExit(0)
    try:
        asyncio.run(run_classic(a))
    except KeyboardInterrupt:
        print("\nbye")
