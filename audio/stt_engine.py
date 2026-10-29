"""Local speech-to-text with faster-whisper (CTranslate2), no network after the first model download.

``FasterWhisperEngine`` is blocking; ``SttService`` runs it off the event loop:
SPEECH_CAPTURED in, SPEECH_TRANSCRIBED out.
"""
from __future__ import annotations

import asyncio
import logging
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


class FasterWhisperEngine:
    def __init__(self, model: str = "base", device: str = "auto", compute_type: str = "int8",
                 language: Optional[str] = None) -> None:
        from faster_whisper import WhisperModel  # lazy: heavy import

        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language

    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        segments, info = self._model.transcribe(
            audio, language=self._language, beam_size=1, temperature=0.0,
            vad_filter=False,  # already segmented by our own VAD
            condition_on_previous_text=False)  # stops one hallucination feeding the next
        # Whisper invents text ("Thank you.") for near-silence; drop segments it itself doubts.
        kept = [s.text.strip() for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
        return " ".join(t for t in kept if t).strip(), info.language


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
