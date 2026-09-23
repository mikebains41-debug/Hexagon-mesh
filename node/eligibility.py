"""Step 1 of the node agent: should this phone take work right now?

Rules: charger plugged in, battery above a minimum, temperature under the
limit (with a cool-down gap so it doesn't flip on and off), and on Wi-Fi.
Needs the termux-api package and the Termux:API app.
Test:  python -m node.eligibility
"""
import fcntl
import json
import os
import socket
import struct
import subprocess
import time

PAUSE_AT_C = float(os.environ.get("HM_PAUSE_AT_C", "38"))        # stop taking work
RESUME_BELOW_C = float(os.environ.get("HM_RESUME_BELOW_C", "35"))  # start again
MIN_BATTERY = int(os.environ.get("HM_MIN_BATTERY", "50"))          # let fast charging finish first
REQUIRE_WIFI = os.environ.get("HM_REQUIRE_WIFI", "1") == "1"


def _run_json(cmd, timeout=15):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return json.loads(out.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def battery():
    return _run_json(["termux-battery-status"])


def wlan_ip(ifname=b"wlan0"):
    """IPv4 address of the Wi-Fi interface, or None if not connected.
    Works on Play Store Termux, where termux-wifi-connectioninfo is unavailable."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            data = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", ifname[:15]))
            return socket.inet_ntoa(data[20:24])
        finally:
            s.close()
    except OSError:
        return None


def wifi_connected():
    info = _run_json(["termux-wifi-connectioninfo"])
    if info:
        return (info.get("supplicant_state") == "COMPLETED"
                and info.get("ip") not in (None, "", "0.0.0.0"))
    return wlan_ip() is not None


class Gate:
    """Remembers if the phone got too warm, and waits until it cools off."""

    def __init__(self):
        self.hot = False

    def check(self):
        b = battery()
        if b is None:
            return False, "termux-api not working (install termux-api + Termux:API app)"
        temp = float(b.get("temperature", 99))
        pct = int(b.get("percentage", 0))
        plugged = b.get("plugged", "UNPLUGGED")

        if plugged == "UNPLUGGED":
            return False, "not charging (%.1fC, %d%%)" % (temp, pct)
        if pct < MIN_BATTERY:
            return False, "battery %d%%, waiting for %d%%" % (pct, MIN_BATTERY)
        if temp >= PAUSE_AT_C:
            self.hot = True
        elif temp < RESUME_BELOW_C:
            self.hot = False
        if self.hot:
            return False, "cooling down (%.1fC, resumes below %.0fC)" % (temp, RESUME_BELOW_C)
        if REQUIRE_WIFI and not wifi_connected():
            return False, "not on Wi-Fi"
        return True, "%.1fC, %d%%, %s" % (temp, pct, plugged)


if __name__ == "__main__":
    gate = Gate()
    for _ in range(int(os.environ.get("HM_CHECKS", "5"))):
        ok, why = gate.check()
        print(time.strftime("%H:%M:%S"), "READY" if ok else "WAIT ", "-", why)
        time.sleep(3)
