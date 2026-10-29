"""Local speech-to-text with faster-whisper (CTranslate2), no network after the first model download.

``FasterWhisperEngine`` is blocking; ``SttService`` runs it off the event loop:
SPEECH_CAPTURED in, SPEECH_TRANSCRIBED out.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

from audio.recorder import Utterance
from core.event_bus import Event, EventBus, Subscription, Topic

log = logging.getLogger("aether.stt")

TYPED = "typed"  # Transcript.language for commands typed in the UI (no microphone involved)


@dataclass(frozen=True)
class Transcript:
    """Payload of Topic.SPEECH_TRANSCRIBED."""
    text: str
    language: str
    audio_seconds: float
    inference_ms: float
    started_at: float


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        """float32 mono 16 kHz -> (text, language code)."""


# What Whisper tends to "hear" in near-silence or noise. Only dropped when it is the WHOLE text, so a
# real command such as "Aether, thank you" is never touched.
_PHANTOMS = {"thank you", "thanks", "thanks for watching", "thank you for watching", "you", "bye",
             "i'm sorry", "i am sorry", "sorry", "okay", "ok", "so", "uh", "um", "hmm", "mm", "oh"}


def is_phantom(text: str) -> bool:
    return re.sub(r"[^a-z' ]+", "", text.lower()).strip() in _PHANTOMS


def default_threads() -> int:
    """Half of the CPU threads (at least 1): speech never starves the camera and the UI."""
    return max(1, (os.cpu_count() or 2) // 2)


class FasterWhisperEngine:
    def __init__(self, model: str = "base", device: str = "auto", compute_type: str = "int8",
                 language: Optional[str] = "en", prompt: Optional[str] = None, threads: int = 0,
                 _model=None) -> None:
        if _model is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows, just noisy
            from faster_whisper import WhisperModel  # lazy: heavy import

            _model = WhisperModel(model, device=device, compute_type=compute_type,
                                  cpu_threads=threads or default_threads())
        self._model = _model
        self._language = language
        self._prompt = prompt or None

    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        segments, info = self._model.transcribe(
            audio, language=self._language, initial_prompt=self._prompt, beam_size=1, temperature=0.0,
            without_timestamps=True,
            vad_filter=False,  # already segmented by our own VAD
            condition_on_previous_text=False)  # stops one hallucination feeding the next
        kept = [s.text.strip() for s in segments if _trustworthy(s)]
        text = " ".join(t for t in kept if t).strip()
        return ("" if is_phantom(text) else text), info.language


def _trustworthy(seg) -> bool:
    """Drop segments Whisper itself doubts: probably no speech, or looping/garbled output."""
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
        return False
    return seg.compression_ratio <= 2.4


class SttService:
    def __init__(self, bus: EventBus, engine: Transcriber) -> None:
        self._bus = bus
        self._engine = engine
        self._sub: Optional[Subscription] = None

    def start(self) -> None:
        # Phrases are precious (unlike video frames), so a deeper queue; oldest dropped if swamped.
        self._sub = self._bus.subscribe(Topic.SPEECH_CAPTURED, self._on_speech, max_queue=8)

    def stop(self) -> None:
        if self._sub:
            self._bus.unsubscribe(self._sub)
            self._sub = None

    async def _on_speech(self, event: Event) -> None:
        utt: Utterance = event.payload
        t0 = time.perf_counter()
        text, lang = await asyncio.to_thread(self._engine.transcribe, utt.audio)
        if text:
            ms = (time.perf_counter() - t0) * 1000
            self._bus.publish(Topic.SPEECH_TRANSCRIBED, Transcript(text, lang, utt.duration, ms, utt.started_at))
