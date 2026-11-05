"""Tests for Turnkey Business Profiles."""
import pytest
from core.profiles import PROFILES, get_profile


def test_profile_lookup_and_fallback():
    assert get_profile("receptionist").name == "receptionist"
    assert get_profile("access_control").require_liveness is True
    assert get_profile("vip_greeter").name == "vip_greeter"
    assert get_profile("non_existent").name == "default"
    assert get_profile(None).name == "default"


def test_greeting_templates():
    rec = get_profile("receptionist")
    assert "Alice" in rec.greeting_template.format(name="Alice")
    assert "office" in rec.greeting_template

    vip = get_profile("vip_greeter")
    assert "pleasure" in vip.greeting_template.format(name="Bob")
