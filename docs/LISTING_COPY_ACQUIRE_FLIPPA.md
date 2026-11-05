# Ready-to-Use Listing Copy for Acquire.com, Flippa & SideProjectors

---

## 🏷️ Listing Title Options (Choose one)

* **Option 1 (Best for Acquire.com):**
  > **Commercial Edge AI Vision & Voice Microservice: Biometric Access, Smart Receptionist & On-Prem Multimodal Engine**
* **Option 2 (Best for Flippa):**
  > **Turnkey Edge AI Kiosk & Receptionist Software (FastAPI + Biometrics + Local Voice STT/TTS + Docker)**
* **Option 3 (Best for SideProjectors / Indie Hackers):**
  > **100% Private, On-Prem Multimodal AI Engine with REST API, Web Dashboard & Liveness Biometrics**

---

## 💰 Recommended Pricing & Deal Structure

* **Buyout Asking Price:** **\$3,500 – \$6,500 USD** (Optimal sweet spot for fast closing with no financing delays)
* **Minimum Reserve (if auction):** **\$3,000 USD**
* **Deal Type:** Full Asset & IP Transfer (Codebase, Documentation, Docker, Commercial License)

---

## 📄 Executive Summary (Copy & Paste into Listing Description)

**Aether-Vision Engine** is an enterprise-grade, event-driven multimodal AI system combining real-time face recognition, active challenge-response liveness verification, local voice recognition (VAD + Whisper), text-to-speech, local LLM assistance (Ollama), and a headless FastAPI microservice + Web Admin Portal.

Unlike cloud-dependent AI tools that charge per-API call and raise serious biometric privacy compliance issues, Aether runs **100% locally and on-premise**. Zero facial biometric data or microphone audio ever leaves the host machine—making it compliant with GDPR, HIPAA, and biometric privacy regulations (BIPA) out-of-the-box.

### 🌟 Key Commercial Assets Included in Sale:
1. **Full Intellectual Property & Commercial License:** Perpetual, unencumbered rights to white-label, rebrand, resell, or bundle into proprietary hardware.
2. **Headless REST & WebSocket API:** Production-ready FastAPI service with automated OpenAPI / Swagger documentation (`/docs`) and real-time WebSocket event feeds (`/ws/events`).
3. **Live MJPEG Video Stream (`/api/stream`):** Embed real-time annotated camera video directly into any web or mobile app with a simple `<img>` tag.
4. **Outbound Webhook Dispatcher:** Integrates with Zapier, Slack, Discord, Make, or Home Assistant on key events (face recognized, stranger detected, liveness passed).
5. **Modern Browser Admin Portal:** Responsive dark-mode web dashboard for camera monitoring, presence checks, face management, and live audit log inspection.
6. **PySide6 Desktop Application:** Native cross-platform desktop UI for dedicated kiosk screens.
7. **Production Docker Deployment:** 1-click `docker compose up` configuration.
8. **Automated Test Suite:** 301 comprehensive pytest tests with zero external cloud dependencies.

---

## 🎯 Multiple Pre-Built Commercial Verticals (Monetization Avenues)

The engine features instant business profile switching via a single environment variable:
* **🏢 Smart Receptionist (`receptionist`):** Automated office check-in, greeting employees/visitors by name, logging attendance in SQLite, and alerting team channels via webhooks.
* **🛡️ Security & Access Control (`access_control`):** Head-turn liveness challenge prevents photo spoofing; fires door-unlock webhooks to smart relays.
* **🛍️ VIP Retail & Hospitality (`vip_greeter`):** Identifies VIP clients upon entering a store or hotel, notifying staff in real time.
* **🤖 Air-Gapped Private Voice Assistant (`default`):** Local LLM answering via Ollama with zero cloud data transmission.

---

## 💻 Tech Stack & Sizing

* **Backend & API:** Python 3.10+, FastAPI, Uvicorn, AsyncIO Event Bus, Pydantic v2.
* **Vision & Biometrics:** OpenCV YuNet (face detection), SFace (128-D facial vector embeddings), IoU Tracker, 3D Pose Geometry.
* **Speech & Audio:** Local Faster-Whisper (int8 CPU optimized), WebRTC VAD, Pyttsx3 TTS.
* **Persistence:** Thread-safe SQLite audit logging, persistent notes, and reminder scheduling.
* **Deployment:** Standalone executables/scripts (`.bat` and `.sh`) and Docker / Docker-Compose.

---

## ❓ Frequently Asked Due-Diligence Questions (Buyer FAQ)

**Q: Are there any monthly cloud API costs or recurring fees to run this?**
* **A:** None. Everything (face detection, vector similarity, speech-to-text, TTS, and LLM chat) runs 100% locally on the host machine. Operating costs are \$0.

**Q: Can I rebrand this and resell it to my own clients?**
* **A:** Yes. The sale includes a complete commercial asset transfer license allowing unlimited white-labeling, sublicensing, and resale under your own brand name and pricing.

**Q: Does it require high-end GPUs?**
* **A:** No. The models are optimized for standard multi-core CPUs (e.g. Intel Core i3/i5 or AMD Ryzen), using int8 quantization and vectorized NumPy cosine math.

**Q: What is the transfer process?**
* **A:** Upon completion of escrow (via Acquire.com or Flippa Escrow), full GitHub repository ownership, commits, and all handover documentation will be transferred immediately.
