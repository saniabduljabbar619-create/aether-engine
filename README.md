# Aether-Vision Engine

Async, event-driven multimodal AI system: face recognition (vectorized embedding
matching + temporal stabilization) and local speech (VAD + faster-whisper),
coordinated by an asyncio event bus, with a PySide6 desktop UI.

## Layout

| Path | Responsibility |
|------|----------------|
| `config/` | Pydantic settings (env-driven) |
| `core/` | Event bus, logging |
| `vision/` | Capture worker, detector/embedder, spatial engine (matcher + filter) |
| `audio/` | Mic recorder + VAD, Whisper STT |
| `storage/` | SQLite audit log, reference face assets |
| `ui/` | PySide6 window and asyncio-to-Qt bridges |
| `tests/` | Unit and concurrency tests |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
```

## Install

Python 3.10+. Use a virtual environment.

| Stage | Command | Packages |
|-------|---------|----------|
| Now (bus, config, spatial engine) | `pip install numpy pydantic pydantic-settings` | math, settings |
| Dev / tests | `pip install pytest pytest-asyncio ruff` | test runner, async tests, linter |
| Vision (next) | `pip install opencv-python mediapipe deepface` | camera, face detection, embeddings |
| Audio | `pip install faster-whisper sounddevice webrtcvad-wheels` | local STT, mic capture, VAD |
| UI | `pip install PySide6` | desktop GUI |

Or install by group: `pip install -e ".[dev]"`, `".[vision]"`, `".[audio]"`, `".[ui]"`.
(`sounddevice` also needs the system PortAudio library, e.g. `apt install libportaudio2`.)

## Event bus

![demo](docs/screenshots/event_bus_demo.png)

Run it: `python -m examples.event_bus_demo`. Tests: `pytest`.

![tests](docs/screenshots/event_bus_tests.png)

## Camera capture

`pip install opencv-python`, then `python -m examples.webcam_demo` (press `q` to quit).
No camera? `python -m examples.webcam_demo --synthetic`.

![capture](docs/screenshots/capture_synthetic.png)
*Synthetic test pattern, not a real webcam.*

## Face detection

`python -m examples.detect_demo` (first run downloads two models, ~39 MB, into `models/`).
Only `opencv-python` is needed: YuNet detects faces, SFace produces 128-D embeddings.
Analyze a single image: `python -m examples.detect_demo --image photo.jpg --save out.png`.

![detector](docs/screenshots/detector_astronaut.png)
*NASA public-domain photo (scikit-image sample data).*

Note: SFace's recommended cosine threshold is **0.363**, which is now the default.

## Face recognition

```bash
python -m examples.enroll <your-name>      # webcam, 10 samples; or --image a.jpg b.jpg
python -m examples.enroll --list           # who is enrolled
python -m examples.recognize_demo          # live recognition, q to quit
```

![known](docs/screenshots/recognition_known.png)
*Simulated camera showing a public-domain photo, enrolled as "astronaut".*

How it works: each face embedding is matched against all enrolled embeddings with one matrix
multiply (`vision/spatial_engine.py`). `vision/tracker.py` follows faces between frames by box
overlap (IoU), smooths boxes, and only commits to a name after a majority of recent frames agree,
so a single bad frame can't flip an identity. `FACE_RECOGNIZED` fires once per person, with a
30 s cooldown against repeat greetings.

Privacy: embeddings are stored locally in `storage/reference_faces/` and git-ignored.

## Voice

```bash
pip install faster-whisper sounddevice
python -m examples.voice_demo --list-devices
python -m examples.voice_demo            # speak; Ctrl+C to quit
python -m examples.voice_demo --wav clip.wav
```

`audio/recorder.py` cuts the mic stream into phrases (adaptive-noise-floor VAD, pre-roll so first
syllables aren't clipped, 700 ms hangover so pauses don't split a phrase). `audio/stt_engine.py`
transcribes each phrase locally with faster-whisper (default model `base`, int8 on CPU).

![voice](docs/screenshots/audio_pipeline.png)
*Simulated microphone and a stand-in transcriber: shows the plumbing and timing, not Whisper accuracy.*

## Desktop window

`python main.py --gui` (needs `pip install PySide6`). Live camera with boxes and names, system status
and mic level, enrolled people (enroll with a button, forget with a button), a conversation log, an
activity tab fed by the audit log, and a box for typing commands when you would rather not talk.

![gui](docs/screenshots/gui_conversation.png)
![enrolling](docs/screenshots/gui_enrolling.png)
![activity](docs/screenshots/gui_activity.png)
*Simulated camera (public-domain photo) with the real face models; real Qt window rendered offscreen.*

Design notes: the asyncio engine runs on its own thread and the window only talks to an
`EngineController` (`ui/qt_bridges.py`). Video frames pass through a latest-frame mailbox, so a slow UI
skips frames instead of lagging. Enrolling from the window is allowed without a recognized person in
view, because whoever sits at the keyboard is the administrator; the voice rules still apply to voice.

## Run everything

```bash
pip install -e ".[vision,audio]" pyttsx3 pytest pytest-asyncio
python main.py --gui           # desktop window (recommended)
python main.py                 # classic OpenCV window
python main.py --no-voice      # camera only        (also: --no-camera, --no-window, --no-tts)
```

Say the wake word first, so ordinary conversation is ignored: **"Aether, ..."**

| Say | What happens |
|-----|--------------|
| who is at the camera / who am I | describes who is in view |
| enroll me as `<name>` | captures 8 samples of the one face in view, then remembers it |
| forget me | deletes the recognized person's stored face data |
| who do you know / status / recent activity | lists enrolled people / system status / last audit events |
| what time is it / what's the date | |
| help / goodbye | lists commands / shuts down |

![conversation](docs/screenshots/orchestrator_conversation.png)
![frame](docs/screenshots/orchestrator_e2e.png)
*Simulated camera and microphone with the real face models and a scripted stand-in for Whisper.*

**Safety rules** (this system handles biometric data): audio captured while the assistant is speaking
is ignored, so it never obeys itself; enrolling needs an already-recognized person in view (unless
nobody is enrolled yet); samples can only be added to a name by that person; "forget me" only deletes
the recognized person's own data. Events are logged to a local SQLite file (`storage/aether.db`, git-ignored).

Add your own commands:

```python
async def lock(cmd):
    my_door.lock()
    return "Front door locked."

orchestrator.register_handler("lock_door", [r"\block\b.*\bdoor\b"], lock)
```

## Build order (done)

event bus, settings, spatial engine, camera capture, face detector, recognition, audio, orchestrator.
PySide6 desktop UI.
