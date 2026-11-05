"""Turnkey Business Profiles for Aether Edge AI Engine.

Configures specialized behaviors, greetings, and business automation presets:
- receptionist: Greets visitors/staff, logs office check-in, fires notification webhooks.
- access_control: Strict liveness verification kiosk, door relay webhooks, security audits.
- vip_greeter: High-priority VIP recognition, hospitality alerts.
- default: General conversational assistant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class BusinessProfile:
    name: str
    description: str
    greeting_template: str
    require_liveness: bool
    webhook_event_type: str


PROFILES: Dict[str, BusinessProfile] = {
    "default": BusinessProfile(
        name="default",
        description="General Edge AI conversational and multimodal assistant",
        greeting_template="Hello {name}.",
        require_liveness=False,
        webhook_event_type="face_recognized",
    ),
    "receptionist": BusinessProfile(
        name="receptionist",
        description="Smart Office Receptionist: visitor & staff check-in, arrival notifications",
        greeting_template="Welcome to the office, {name}. Your arrival has been checked in.",
        require_liveness=False,
        webhook_event_type="office_checkin",
    ),
    "access_control": BusinessProfile(
        name="access_control",
        description="Security & Access Control Kiosk: mandatory liveness check and door unlock signaling",
        greeting_template="Identity confirmed: {name}. Please turn your head to complete access verification.",
        require_liveness=True,
        webhook_event_type="access_granted",
    ),
    "vip_greeter": BusinessProfile(
        name="vip_greeter",
        description="Hospitality & Retail VIP Greeter: alerts staff and offers premium welcome",
        greeting_template="Welcome back, {name}. It is our pleasure to have you with us today.",
        require_liveness=False,
        webhook_event_type="vip_arrival",
    ),
}


def get_profile(name: Optional[str]) -> BusinessProfile:
    """Return the profile by name, falling back to 'default' if unrecognized."""
    if not name:
        return PROFILES["default"]
    return PROFILES.get(name.strip().lower(), PROFILES["default"])
