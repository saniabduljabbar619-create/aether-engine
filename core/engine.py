"""The engine: builds and runs every component (bus, vision, voice, orchestrator).

Both front ends (the classic OpenCV window in main.py and the Qt GUI) use this class, so
there is one place that knows how the pieces fit. Everything here runs on one asyncio loop.
Components that cannot start (no camera, no microphone) are skipped and listed in ``warnings``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional

from audio.tts import NullSpeaker, SwitchableSpeaker
from config.settings import Settings, get_settings
from core.event_bus import EventBus
from core.orchestrator import Orchestrator
from storage.face_store import FaceStore
from storage.local_db import AuditLog
from vision.spatial_engine import VectorIdentityMatcher

log = logging.getLogger("aether.engine")


class Engine:
    def __init__(self, cfg: Optional[Settings] = None, *, camera: bool = True, voice: bool = True,
                 tts: bool = True, camera_index: Optional[int] = None, mic_device: Optional[int] = None,
                 on_reply: Optional[Callable[[str], None]] = None,
                 on_status: Optional[Callable[[str], None]] = None,
                 analyzer: Any = None, transcriber: Any = None, camera_source_factory: Any = None) -> None:
        self.cfg = cfg or get_settings()
        self._use_camera, self._use_voice, self._use_tts = camera, voice, tts
        self._camera_index = camera_index if camera_index is not None else self.cfg.camera_index
        self._mic_device = mic_device if mic_device is not None else self.cfg.mic_device
        self._on_reply = on_reply
        self._on_status = on_status or (lambda msg: None)
        # injection points (tests, or custom hardware)
        self._analyzer, self._transcriber, self._source_factory = analyzer, transcriber, camera_source_factory

        self.warnings: List[str] = []
        self.bus = EventBus()
        self.store = FaceStore(self.cfg.faces_dir)
        self.matcher = VectorIdentityMatcher(self.cfg.similarity_threshold)
        self.audit = AuditLog(self.cfg.db_path)
        self.speaker: Optional[SwitchableSpeaker] = None
        self.orch: Optional[Orchestrator] = None
        self.cap = None
        self.mic = None
        self._closers: List[Callable] = []

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        await self.bus.start()
        self.matcher.register_identities(self.store.load_all())

        inner = NullSpeaker()
        if self._use_tts:
            from audio.tts import Pyttsx3Speaker

            tts = await asyncio.to_thread(Pyttsx3Speaker)
            if tts.available:
                inner = tts
            else:
                self.warnings.append("Text-to-speech is unavailable; replies are text only.")
        self.speaker = SwitchableSpeaker(inner)
        self.orch = Orchestrator(self.bus, self.matcher, self.store, self.audit, self.speaker,
                                 status_fn=self._status_text, on_reply=self._on_reply)
        self.orch.start()

        await self._load_models()
        if self._analyzer is not None:
            await self._start_vision()
        if self._transcriber is not None:
            await self._start_voice()
        self._on_status("Ready")

    async def stop(self) -> None:
        for fn in reversed(self._closers):
            try:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                log.exception("cleanup failed")
        self._closers.clear()
        if self.speaker:
            self.speaker.close()
        await self.bus.stop()
        self.audit.close()

    # ---- internals -------------------------------------------------------
    def _status_text(self) -> str:
        return (f"Camera is running at {self.cap.fps:.0f} frames per second."
                if self.cap else "The camera is off.")

    async def _load_models(self) -> None:
        jobs = {}
        if self._use_camera and self._analyzer is None:
            from vision.detector import FaceAnalyzer

            self._on_status("Loading face models...")
            jobs["analyzer"] = asyncio.to_thread(FaceAnalyzer)
        if self._use_voice and self._transcriber is None:
            from audio.stt_engine import FasterWhisperEngine

            self._on_status("Loading speech model (the first run downloads it)...")
            jobs["transcriber"] = asyncio.to_thread(
                FasterWhisperEngine, self.cfg.whisper_model, self.cfg.whisper_device,
                language=self.cfg.whisper_language, prompt=self.cfg.whisper_prompt,
                threads=self.cfg.whisper_threads)
        if jobs:
            for name, model in zip(jobs, await asyncio.gather(*jobs.values())):
                setattr(self, f"_{name}", model)

    async def _start_vision(self) -> None:
        from vision.capture import CameraError, FrameCapture
        from vision.detector import DetectorService
        from vision.recognition import RecognitionService

        DetectorService(self.bus, self._analyzer, max_fps=self.cfg.detect_max_fps).start()
        RecognitionService(self.bus, self.matcher).start()
        cap = FrameCapture(self.bus, camera_index=self._camera_index, source_factory=self._source_factory)
        try:
            await cap.start()
        except CameraError as exc:
            self.warnings.append(f"Camera unavailable: {exc}")
            return
        self.cap = cap
        self._closers.append(cap.stop)

    async def _start_voice(self) -> None:
        from audio.recorder import MicRecorder, MicrophoneError
        from audio.stt_engine import SttService

        SttService(self.bus, self._transcriber).start()
        mic = MicRecorder(self.bus, device=self._mic_device)
        try:
            mic.start()
        except MicrophoneError as exc:
            self.warnings.append(f"Microphone unavailable: {exc}")
            return
        self.mic = mic
        self._closers.append(lambda: asyncio.to_thread(mic.stop))
