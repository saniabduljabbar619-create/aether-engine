"""Entry point for the desktop GUI:  python main.py --gui"""
import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.qt_bridges import EngineController


def run_gui(args) -> int:
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    controller = EngineController(dict(
        camera=not args.no_camera, voice=not args.no_voice, tts=not args.no_tts,
        camera_index=args.camera, mic_device=args.mic))
    window = MainWindow(controller)
    window.show()
    controller.start()
    code = app.exec()
    controller.stop()  # no-op if the window already did it
    return code
