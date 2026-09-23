"""Hexagon Mesh node agent: check in, pull tickets, run them, submit results.

Run:  python -m node.agent
Settings (environment variables):
  HM_URL          coordinator address (default http://127.0.0.1:8080)
  HM_WALLET       payout wallet address
  HM_NODE_NAME    identity name, lets several test agents run on one phone (default "main")
  HM_TEST=1       fake results, no model needed
  HM_SKIP_GATE=1  ignore charging/temperature/Wi-Fi checks (testing only)
  HM_NODE_KEY     API key, if the coordinator requires one
"""
import json
import os
import time
import urllib.error
import urllib.request
import uuid

from . import runner
from .eligibility import Gate

URL = os.environ.get("HM_URL", "http://127.0.0.1:8080").rstrip("/")
WALLET = os.environ.get("HM_WALLET", "")
NAME = os.environ.get("HM_NODE_NAME", "main")
TEST = runner.TEST
SKIP_GATE = os.environ.get("HM_SKIP_GATE", "0") == "1"
NODE_KEY = os.environ.get("HM_NODE_KEY")
POLL_SECONDS = 5
WAIT_SECONDS = 60          # pause between checks while the gate says WAIT
CHECKIN_EVERY = 600
TEST_MAX_IDLE = 3          # in test mode, exit after this many empty polls


def log(*msg):
    print(time.strftime("%H:%M:%S"), "[%s]" % NAME, *msg, flush=True)


def call(method, path, data=None):
    headers = {"Content-Type": "application/json"}
    if NODE_KEY:
        headers["X-API-Key"] = NODE_KEY
    req = urllib.request.Request(URL + path, method=method, headers=headers,
                                 data=json.dumps(data).encode() if data is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def node_id():
    """Stable ID for this phone, saved so it survives restarts."""
    path = os.path.expanduser("~/.hexagon/node-%s.json" % NAME)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)["node_id"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nid = uuid.uuid4().hex
    with open(path, "w") as fh:
        json.dump({"node_id": nid}, fh)
    return nid


def ram_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1048576, 1)
    except OSError:
        pass
    return 0


def checkin(nid):
    status, body = call("POST", "/nodes/checkin", {
        "node_id": nid, "wallet": WALLET, "ram_gb": ram_gb(), "models": runner.models()})
    if status != 200:
        raise RuntimeError("check-in failed: %s %s" % (status, body))
    log("checked in, models:", runner.models() or "none")


def main():
    nid = node_id()
    log("node", nid[:8], "->", URL, "(test mode)" if TEST else "")
    gate = Gate()
    checkin(nid)
    last_checkin, idle, done = time.time(), 0, 0

    while True:
        try:
            if time.time() - last_checkin > CHECKIN_EVERY:
                checkin(nid)
                last_checkin = time.time()

            if not SKIP_GATE:
                ok, why = gate.check()
                if not ok:
                    log("WAIT -", why)
                    time.sleep(WAIT_SECONDS)
                    continue

            status, body = call("POST", "/work/next", {"node_id": nid})
            if status == 403:
                log("suspended by coordinator, stopping")
                break
            ticket = body.get("ticket") if status == 200 else None
            if not ticket:
                idle += 1
                if TEST and idle >= TEST_MAX_IDLE:
                    log("no more work, stopping")
                    break
                time.sleep(POLL_SECONDS)
                continue
            idle = 0

            start = time.time()
            result = runner.run(ticket)
            took = time.time() - start
            if time.time() > ticket["deadline"]:
                log("ticket took too long (%.1fs), skipped" % took)
                continue
            status, body = call("POST", "/work/submit", {
                "node_id": nid, "assignment_id": ticket["assignment_id"], "result": result})
            done += 1
            log("ticket %s: %s (%.2fs)" % (ticket["ticket_id"][:8], body.get("status", status), took))

        except KeyboardInterrupt:
            break
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            log("connection problem:", e, "- retrying in 30s")
            time.sleep(30)
        except NotImplementedError as e:
            log(e)
            break

    _, bal = call("GET", "/nodes/%s/balance" % nid)
    log("finished %d tickets, balance: %s credits" % (done, bal.get("credits")))


if __name__ == "__main__":
    main()
