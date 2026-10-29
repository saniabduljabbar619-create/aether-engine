"""Microphone capture with voice activity detection (VAD).

Pipeline: PortAudio callback -> queue -> worker thread -> UtteranceSegmenter -> bus.
The callback only copies samples, so audio never glitches however slow the consumers are.

``EnergyVAD`` and ``UtteranceSegmenter`` are pure NumPy (no audio hardware needed), which
is what makes them unit-testable. ``sounddevice`` is imported lazily by ``MicRecorder``.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Union

import numpy as np

from core.event_bus import EventBus, Topic

log = logging.getLogger("aether.audio")

SAMPLE_RATE = 16_000  # what Whisper expects
FRAME_MS = 30
FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples


class MicrophoneError(RuntimeError):
    """The microphone could not be opened."""


@dataclass(frozen=True)
class Utterance:
    """Payload of Topic.SPEECH_CAPTURED: one spoken phrase, float32 mono 16 kHz."""
    audio: np.ndarray
    started_at: float  # time.monotonic()

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


@dataclass(frozen=True)
class SpeechStart:
    """Payload of Topic.SPEECH_STARTED."""
    timestamp: float


def level_db(frame: np.ndarray) -> float:
    """RMS level in dBFS (0 = full scale, about -60 = quiet room)."""
    return float(20 * np.log10(np.sqrt(np.mean(np.square(frame))) + 1e-10))


class EnergyVAD:
    """Speech = clearly louder than the room. The noise floor adapts, so a fan switching on is
    absorbed after a few seconds instead of being treated as speech forever."""

    def __init__(self, margin_db: float = 12.0, min_db: float = -55.0, floor_db: float = -60.0):
        self.margin_db = margin_db
        self.min_db = min_db
        self.floor_db = floor_db

    def is_speech(self, frame: np.ndarray) -> bool:
        db = level_db(frame)
        speech = db > max(self.floor_db + self.margin_db, self.min_db)
        if db < self.floor_db:
            rate = 0.3  # quieter than thought: drop fast
        else:
            rate = 0.001 if speech else 0.02  # rise slowly; barely at all during speech
        self.floor_db += rate * (db - self.floor_db)
        return speech


class UtteranceSegmenter:
    """Cuts a continuous stream into utterances.

    * pre-roll: the 300 ms before speech was confirmed is kept, so first syllables aren't clipped
    * start debounce: needs ~90 ms of speech in a row, so a click or cough-tick doesn't start one
    * hangover: speech ends after 700 ms of silence, so short pauses stay inside one phrase
    * too-short utterances are discarded, and very long ones are split at ``max_seconds``
    """

    def __init__(self, vad: Optional[EnergyVAD] = None, preroll_ms: int = 300, start_ms: int = 90,
                 hangover_ms: int = 700, min_ms: int = 300, max_seconds: float = 15.0) -> None:
        self.vad = vad or EnergyVAD()
        self._preroll: Deque[np.ndarray] = deque(maxlen=max(preroll_ms // FRAME_MS, 0) or None)
        self._start_frames = max(start_ms // FRAME_MS, 1)
        self._hang_frames = max(hangover_ms // FRAME_MS, 1)
        self._min_voiced = max(min_ms // FRAME_MS, 1)
        self._max_frames = int(max_seconds * 1000 // FRAME_MS)
        self._carry = np.empty(0, np.float32)
        self._pending: List[np.ndarray] = []
        self._frames: List[np.ndarray] = []
        self._active = False
        self._silence = 0
        self._voiced = 0

    @property
    def active(self) -> bool:
        return self._active

    def push(self, samples: np.ndarray, now: float) -> List[Union[SpeechStart, Utterance]]:
        """Feed any amount of audio. ``now`` is the monotonic time at the end of ``samples``."""
        data = np.concatenate([self._carry, samples.astype(np.float32, copy=False)])
        n = len(data) // FRAME
        self._carry = data[n * FRAME:]
        events: List[Union[SpeechStart, Utterance]] = []
        for i in range(n):
            self._step(data[i * FRAME:(i + 1) * FRAME], now, events)
        return events

    def _step(self, frame: np.ndarray, now: float, events: list) -> None:
        speech = self.vad.is_speech(frame)
        if not self._active:
            if speech:
                self._pending.append(frame)
                if len(self._pending) >= self._start_frames:
                    self._active = True
                    self._frames = [*self._preroll, *self._pending]
                    self._voiced, self._silence = len(self._pending), 0
                    self._preroll.clear()
                    self._pending = []
                    events.append(SpeechStart(now))
            else:
                self._preroll.extend(self._pending)  # false alarm: it was just noise
                self._pending = []
                self._preroll.append(frame)
            return

        self._frames.append(frame)
        if speech:
            self._voiced += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self._hang_frames or len(self._frames) >= self._max_frames:
            audio = np.concatenate(self._frames)
            if self._voiced >= self._min_voiced:
                events.append(Utterance(audio, now - len(audio) / SAMPLE_RATE))
            self._frames, self._active, self._voiced, self._silence = [], False, 0, 0


class MicRecorder:
    def __init__(self, bus: EventBus, *, device: Optional[int] = None,
                 segmenter: Optional[UtteranceSegmenter] = None) -> None:
        self._bus = bus
        self._device = device
        self._seg = segmenter or UtteranceSegmenter()
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=500)  # ~15 s of audio
        self._stop = threading.Event()
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self.level_db = -100.0

    def start(self) -> None:
        import sounddevice as sd  # lazy: needs audio hardware libraries

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME,
                device=self._device, callback=self._on_audio)
            self._stream.start()
        except Exception as exc:
            raise MicrophoneError(f"could not open microphone: {exc}") from exc
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="aether-mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            log.warning("audio status: %s", status)
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            log.warning("audio worker is behind; dropping a block")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.level_db = level_db(chunk)
            for ev in self._seg.push(chunk, time.monotonic()):
                topic = Topic.SPEECH_STARTED if isinstance(ev, SpeechStart) else Topic.SPEECH_CAPTURED
                self._bus.publish_threadsafe(topic, ev)
