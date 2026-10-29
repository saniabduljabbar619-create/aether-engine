"""Live voice -> text.   Run: python -m examples.voice_demo     (Ctrl+C to quit)
  --list-devices        show microphones and their numbers
  --device N            use microphone N (default: system default)
  --wav FILE            transcribe a WAV file instead of the microphone
First run downloads the Whisper model (base: ~145 MB) from Hugging Face.
"""
import argparse
import asyncio
import sys
import wave

import numpy as np

from audio.recorder import SAMPLE_RATE, MicRecorder, MicrophoneError, Utterance, UtteranceSegmenter
from audio.stt_engine import FasterWhisperEngine, SttService, Transcript
from config.settings import get_settings
from core.event_bus import Event, EventBus, Topic


def load_engine() -> FasterWhisperEngine:
    cfg = get_settings()
    print(f"loading Whisper '{cfg.whisper_model}' (first run downloads it)...")
    return FasterWhisperEngine(cfg.whisper_model, cfg.whisper_device, language=cfg.whisper_language)


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit("only 16-bit PCM WAV files are supported")
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:  # crude linear resample; fine for speech
        x = np.interp(np.linspace(0, len(x), int(len(x) * SAMPLE_RATE / rate), endpoint=False),
                      np.arange(len(x)), x).astype(np.float32)
    return x


def run_wav(path: str) -> None:
    audio, seg, engine = read_wav(path), UtteranceSegmenter(), load_engine()
    silence = np.zeros(SAMPLE_RATE * 2, np.float32)  # flush the final hangover
    for ev in seg.push(np.concatenate([audio, silence]), now=0.0):
        if isinstance(ev, Utterance):
            text, lang = engine.transcribe(ev.audio)
            print(f"[{ev.duration:4.1f}s, {lang}] {text}")


async def run_mic(device) -> None:
    bus = EventBus()
    await bus.start()
    mic = MicRecorder(bus, device=device)
    engine = await asyncio.to_thread(load_engine)
    SttService(bus, engine).start()

    async def on_start(e: Event) -> None:
        print("\r● listening...                                   ")

    async def on_text(e: Event) -> None:
        t: Transcript = e.payload
        print(f"\r\"{t.text}\"   [{t.language}, {t.audio_seconds:.1f}s audio, {t.inference_ms:.0f} ms]")

    bus.subscribe(Topic.SPEECH_STARTED, on_start)
    bus.subscribe(Topic.SPEECH_TRANSCRIBED, on_text)
    try:
        mic.start()
    except MicrophoneError as exc:
        print(f"Microphone error: {exc}\nTry --list-devices, and check Windows Settings > Privacy > Microphone.")
        await bus.stop()
        return
    print("Speak. Ctrl+C to quit.")
    try:
        while True:
            bar = "#" * max(int((mic.level_db + 60) / 2), 0)
            print(f"\r{mic.level_db:6.1f} dB |{bar:<30}|", end="", flush=True)
            await asyncio.sleep(0.1)
    finally:
        mic.stop()
        await bus.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--wav")
    a = ap.parse_args()
    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
    elif a.wav:
        run_wav(a.wav)
    else:
        try:
            asyncio.run(run_mic(a.device if a.device is not None else get_settings().mic_device))
        except KeyboardInterrupt:
            print("\nbye")
