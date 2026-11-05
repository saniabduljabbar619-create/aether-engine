"""System and hardware status utilities (battery, CPU, memory). Cross-platform with pure stdlib."""
from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path


def get_battery_status() -> str:
    """Return a plain English summary of battery status, suitable for TTS."""
    if platform.system() == "Windows":
        class _SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("Reserved", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = _SYSTEM_POWER_STATUS()
        try:
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                pct = status.BatteryLifePercent
                ac = status.ACLineStatus
                if pct == 255:
                    return "Running on AC power, no battery detected."
                state = "plugged in and charging" if ac == 1 else "discharging"
                return f"Battery is at {pct}%, currently {state}."
        except Exception:
            pass

    elif platform.system() == "Linux":
        p = Path("/sys/class/power_supply")
        if p.exists():
            for bat in p.glob("BAT*"):
                cap_file = bat / "capacity"
                stat_file = bat / "status"
                if cap_file.exists():
                    try:
                        pct = cap_file.read_text().strip()
                        state = stat_file.read_text().strip().lower() if stat_file.exists() else "in use"
                        return f"Battery is at {pct}%, currently {state}."
                    except Exception:
                        pass

    return "Battery status is not available on this machine."


def get_system_resources() -> str:
    """Return memory and CPU summary, suitable for TTS."""
    cpus = os.cpu_count() or 1
    cpu_desc = f"{cpus} CPU core" if cpus == 1 else f"{cpus} CPU cores"

    if platform.system() == "Windows":
        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem = _MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(mem)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                used_pct = mem.dwMemoryLoad
                avail_gb = round(mem.ullAvailPhys / (1024 ** 3), 1)
                total_gb = round(mem.ullTotalPhys / (1024 ** 3), 1)
                return f"System load: {used_pct}% memory used, with {avail_gb} GB free out of {total_gb} GB across {cpu_desc}."
        except Exception:
            pass

    elif platform.system() == "Linux":
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                data = {}
                for line in meminfo.read_text().splitlines():
                    parts = line.split(":")
                    if len(parts) == 2:
                        data[parts[0].strip()] = parts[1].strip().split()[0]
                total_kb = float(data.get("MemTotal", 0))
                avail_kb = float(data.get("MemAvailable", 0))
                if total_kb > 0:
                    used_pct = round(100.0 * (1.0 - avail_kb / total_kb))
                    avail_gb = round(avail_kb / (1024 ** 2), 1)
                    total_gb = round(total_kb / (1024 ** 2), 1)
                    return f"System load: {used_pct}% memory used, with {avail_gb} GB free out of {total_gb} GB across {cpu_desc}."
            except Exception:
                pass

    return f"System is running with {cpu_desc} available."
