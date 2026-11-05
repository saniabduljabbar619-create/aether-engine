# Aether Edge AI Engine - Commercial Buyer Handover Guide

Welcome to the **Aether Edge AI Engine** codebase. This guide is designed for the new owner, developer, or agency team acquiring this intellectual property. It provides everything required to rebrand, deploy, customize, and monetize this system.

---

## 1. Quick Start (Run in Under 3 Minutes)

### On Windows
Double-click `start_server.bat` or run:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[vision,audio,server,dev]"
python main.py --server --port 8000
```
Open **`http://localhost:8000`** in any browser for the live Web Portal, or **`http://localhost:8000/docs`** for interactive Swagger API documentation.

### On Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[vision,audio,server,dev]"
python main.py --server --port 8000
```

### Docker 1-Click Startup
```bash
docker compose up -d
```
Spins up both the Aether Engine and a local Ollama instance with persistence.

---

## 2. White-Labeling & Rebranding in 3 Minutes

To resell this engine under your own company or product name (e.g., *"ApexVision"*, *"FrontDesk AI"*, *"GateKeeper Edge"*):

1. **Change App Name & Wake Word (`.env`):**
   ```env
   AETHER_WAKE_WORDS=apex,computer
   ```
2. **Change Web Portal Title & Logo (`server/dashboard.html`):**
   - Line 7: Update `<title>` tag.
   - Line 120: Change `<span>👁️ Aether Edge AI Engine</span>` to your brand name and custom SVG logo.
3. **Change PySide6 Desktop GUI Branding (`ui/main_window.py`):**
   - Update window title `self.setWindowTitle("ApexVision Kiosk")`.

---

## 3. High-Margin Monetization Models for the Buyer

### Model A: On-Premise Enterprise Kiosk (B2B SaaS / One-Time License)
* **Target Audience:** Medical clinics, dental offices, legal practices, private clubs, coworking spaces.
* **Pricing Model:** \$1,500 – \$3,500 upfront per location + \$99/month for maintenance and remote dashboard support.
* **Why Clients Buy It:** 100% on-premise guarantee. Zero facial biometrics or audio ever go to the cloud, making it fully compliant with HIPAA, GDPR, and Illinois BIPA biometric privacy laws.

### Model B: Smart Receptionist & Attendance Kiosk (Turnkey Appliance)
* **Target Audience:** Corporate offices, warehouses, shared office hubs.
* **Setup:** Package Aether onto an inexpensive Mini PC (Intel NUC or Beelink \$180-\$250) + a 1080p USB webcam + a small touchscreen display.
* **Selling Price:** \$2,500 - \$4,500 per turn-key hardware unit.
* **Features:** Automatically recognizes employees, logs attendance check-in to local SQLite/CSV, and sends a Slack/Discord webhook alert when a VIP or client arrives.

### Model C: Access Control & Security Add-On
* **Target Audience:** Gyms, member clubs, gated facilities.
* **Feature Used:** Head-turn challenge-response liveness (`AETHER_BUSINESS_PROFILE=access_control`).
* **Integration:** Connect Aether's outbound webhook to an IoT smart lock relay (e.g. Shelly or Raspberry Pi GPIO) to trigger door access only for verified live humans.

---

## 4. Hardware Sizing & Device Compatibility

| Component | Minimum Specification | Recommended Specification |
|---|---|---|
| **CPU** | Intel Core i3 (6th gen+) or AMD Ryzen 3 | Intel Core i5 / AMD Ryzen 5 (4+ physical cores) |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 2 GB free SSD space | 10 GB SSD space (for multiple Ollama models) |
| **Camera** | Standard 720p USB Webcam (UVC compliant) | 1080p wide-angle USB camera (e.g., Logitech C920) |
| **Microphone** | Standard PC microphone | USB boundary / conference microphone with echo reduction |
| **GPU** | None required (CPU int8 inference optimized) | NVIDIA GTX 1650+ or better if running 7B+ LLMs |

---

## 5. Webhook Integration Guide

Connect Aether to your existing tools without coding by configuring `AETHER_WEBHOOK_URL` in `.env`:

### Webhook JSON Payload Schema:
```json
{
  "source": "aether-engine",
  "event": "face_recognized",
  "timestamp": 1730820000.123,
  "data": {
    "identity": "Alice_Smith",
    "display_name": "Alice Smith",
    "confidence": 0.89
  }
}
```

### Connect to Zapier / Make / Webhook Services:
1. Create a "Catch Hook" trigger in Zapier or Make.
2. Paste the provided Webhook URL into `.env`:
   ```env
   AETHER_WEBHOOK_URL=https://hooks.zapier.com/hooks/catch/12345/abcde/
   ```
3. Test a recognized face or command—Zapier will immediately trigger SMS, email, Slack message, or database inserts.

---

## 6. Architecture & Code Structure

* **`core/event_bus.py`**: Non-blocking asynchronous pub/sub messaging bus with bounded queues.
* **`core/engine.py`**: Central coordinator managing lifecycle across vision, voice, and API services.
* **`core/orchestrator.py`**: State machine controlling greetings, enrollment coaching, liveness verification, and intent routing.
* **`core/webhooks.py`**: Non-blocking HTTP POST dispatcher with secret header authentication.
* **`core/profiles.py`**: Turnkey presets (`receptionist`, `access_control`, `vip_greeter`, `default`).
* **`server/api.py`**: Commercial FastAPI service with Swagger documentation, WebSocket feeds, and MJPEG video streaming.
* **`vision/`**: OpenCV YuNet face detection, SFace 128-D vector embeddings, head pose estimation, and IoU face tracking.
* **`audio/`**: VAD segmentation, faster-whisper local transcription, and pyttsx3 speech synthesis.
* **`tests/`**: 301 automated pytest test cases ensuring zero regressions.
