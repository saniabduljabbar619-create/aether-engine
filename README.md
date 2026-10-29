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

## Build order

1. `core/event_bus.py` 2. `config/settings.py` 3. `vision/spatial_engine.py`
4. `vision/capture.py` + `detector.py` 5. `audio/*` 6. `storage/local_db.py` 7. `ui/*` 8. `main.py`
