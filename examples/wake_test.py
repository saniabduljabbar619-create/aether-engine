"""Measure how well the wake word is heard on YOUR microphone, in YOUR room.

    python -m examples.wake_test                      # 10 trials with the settings from .env
    python -m examples.wake_test --model small        # try a bigger Whisper model
    python -m examples.wake_test --wake computer      # try a different wake word
    python -m examples.wake_test --phrase "who am I"  # say something else

For each trial you say "<wake word>, <phrase>". It reports what Whisper heard, whether the wake word
was recognized, and whether the whole command was understood, then summarizes. Use it to compare
models and wake words with real numbers instead of guessing.
"""
import argparse
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import numpy as np

from audio.recorder import FRAME, SAMPLE_RATE, Utterance, UtteranceSegmenter
from audio.stt_engine import FasterWhisperEngine
from config.settings import get_settings
from core.commands import CommandRouter


@dataclass(frozen=True)
class Trial:
    heard: str  # what Whisper wrote ("" if nothing was said)
    woke: bool  # wake word recognized
    intent: Optional[str]  # command understood
    seconds: float  # transcription time


def evaluate(clips: Iterable[Optional[np.ndarray]], engine, router: CommandRouter) -> List[Trial]:
    """Transcribe each clip and check it the way the real system would. None = nothing was heard."""
    trials = []
    for clip in clips:
        if clip is None:
            trials.append(Trial("", False, None, 0.0))
            continue
        t0 = time.perf_counter()
        text, _ = engine.transcribe(clip)
        seconds = time.perf_counter() - t0
        cmd = router.parse(text)
        trials.append(Trial(text, cmd is not None, cmd.intent if cmd else None, seconds))
    return trials


def summarize(trials: List[Trial], wake_name: str, expected_intent: str) -> str:
    n = len(trials)
    woke = sum(t.woke for t in trials)
    understood = sum(t.woke and t.intent == expected_intent for t in trials)
    avg = np.mean([t.seconds for t in trials if t.seconds]) if any(t.seconds for t in trials) else 0.0
    lines = [f"Wake word '{wake_name}' recognized:   {woke}/{n}",
             f"Whole command understood:    {understood}/{n}   (average transcription {avg:.1f} s)"]
    missed = [t.heard or "(nothing heard)" for t in trials if not t.woke]
    if missed:
        lines.append("When it was missed, Whisper wrote: " + " | ".join(f'"{m}"' for m in missed[:5]))
    rate = woke / n if n else 0.0
    if rate >= 0.9:
        lines.append("Verdict: excellent. The wake word works reliably here.")
    elif rate >= 0.7:
        lines.append("Verdict: workable. The follow-up window and the Talk button (F9) cover the misses.")
    else:
        lines.append("Verdict: poor. Try a more common wake word (--wake computer), a bigger model "
                     "(--model small), or move the microphone closer. The Talk button (F9) needs no wake word.")
    return "\n".join(lines)


def record_trials(n: int, device: Optional[int], phrase: str, wake_name: str) -> Iterator[Optional[np.ndarray]]:
    import sounddevice as sd  # needs audio hardware

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME, device=device) as stream:
        for i in range(n):
            print(f"\nTrial {i + 1}/{n}: say  \"{wake_name}, {phrase}\"")
            seg, clip, deadline = UtteranceSegmenter(), None, time.monotonic() + 10
            while clip is None and time.monotonic() < deadline:
                data, _ = stream.read(FRAME)
                for ev in seg.push(data[:, 0].copy(), time.monotonic()):
                    if isinstance(ev, Utterance):
                        clip = ev.audio
            yield clip
            time.sleep(1.0)


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--model", default=cfg.whisper_model)
    ap.add_argument("--wake", default=cfg.wake_word_list[0])
    ap.add_argument("--phrase", default="what time is it")
    ap.add_argument("--device", type=int, default=cfg.mic_device)
    a = ap.parse_args()

    router = CommandRouter(wake_words=[a.wake])
    prompt = f"Hey {router.wake_name}, who is at the camera? {router.wake_name}, what time is it?"
    print(f"Loading Whisper '{a.model}' (language {cfg.whisper_language or 'auto'})...")
    engine = FasterWhisperEngine(a.model, cfg.whisper_device, language=cfg.whisper_language,
                                 prompt=prompt, threads=cfg.whisper_threads)
    expected = router.parse(f"{a.wake} {a.phrase}")
    if expected is None or expected.intent == "unknown":
        raise SystemExit(f"'{a.phrase}' is not a command I know; try 'what time is it' or 'who am I'.")

    heard = []
    for clip in record_trials(a.trials, a.device, a.phrase, router.wake_name):
        trial = evaluate([clip], engine, router)[0]
        heard.append(trial)
        mark = "wake word OK" if trial.woke else "wake word MISSED"
        print(f"   heard: \"{trial.heard or '(nothing)'}\"  ->  {mark}" + (f", command: {trial.intent}" if trial.woke else ""))
    print("\n" + summarize(heard, router.wake_name, expected.intent))


if __name__ == "__main__":
    main()
