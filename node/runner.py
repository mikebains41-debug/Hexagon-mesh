"""Runs one ticket and returns the result.

HM_TEST=1 : fake but deterministic results (no model needed).
Otherwise : real embeddings from llama.cpp's llama-server, started automatically
            the first time a ticket needs it (or reused if already running).

Settings: HM_EMBED_MODEL (default ~/models/nomic-embed.gguf),
          HM_LLAMA_SERVER (default ~/llama.cpp/build/bin/llama-server),
          HM_LLAMA_PORT (default 8081), HM_THREADS (default 4).
"""
import atexit
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

TEST = os.environ.get("HM_TEST", "0") == "1"
TEST_MODEL = "sim-embed"
EMBED_MODEL_NAME = "nomic-embed"
EMBED_MODEL_PATH = os.path.expanduser(os.environ.get("HM_EMBED_MODEL", "~/models/nomic-embed.gguf"))
LLAMA_SERVER = os.path.expanduser(os.environ.get("HM_LLAMA_SERVER", "~/llama.cpp/build/bin/llama-server"))
LLAMA_PORT = int(os.environ.get("HM_LLAMA_PORT", "8081"))
THREADS = os.environ.get("HM_THREADS", "4")   # fewer threads = less heat
LLAMA_URL = "http://127.0.0.1:%d" % LLAMA_PORT
LOG_PATH = os.path.expanduser("~/.hexagon/llama-server.log")

_proc = None


def models():
    """Model names this phone can run; sent to the coordinator at check-in."""
    if TEST:
        return [TEST_MODEL]
    names = [m for m in os.environ.get("HM_MODELS", "").split(",") if m]
    if not names and os.path.exists(EMBED_MODEL_PATH):
        names = [EMBED_MODEL_NAME]
    return names


def _healthy():
    try:
        with urllib.request.urlopen(LLAMA_URL + "/health", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _post(path, data, timeout=180):
    req = urllib.request.Request(LLAMA_URL + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_server():
    """Start llama-server once and keep it running between tickets."""
    global _proc
    if _healthy():
        return
    if not os.path.exists(LLAMA_SERVER):
        raise RuntimeError("llama-server not found at " + LLAMA_SERVER)
    if not os.path.exists(EMBED_MODEL_PATH):
        raise RuntimeError("model not found at " + EMBED_MODEL_PATH)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log = open(LOG_PATH, "a")
    _proc = subprocess.Popen(
        [LLAMA_SERVER, "-m", EMBED_MODEL_PATH, "--embeddings",
         "--host", "127.0.0.1", "--port", str(LLAMA_PORT), "-t", THREADS,
         "-c", "2048", "-b", "2048", "-ub", "2048"],
        stdout=log, stderr=subprocess.STDOUT)
    atexit.register(stop_server)
    for _ in range(120):
        if _healthy():
            return
        if _proc.poll() is not None:
            raise RuntimeError("llama-server stopped; see " + LOG_PATH)
        time.sleep(1)
    raise RuntimeError("llama-server did not start in 2 minutes; see " + LOG_PATH)


def stop_server():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def _fake_embed(text, dim=16):
    h = hashlib.sha256(text.encode()).digest()
    return [(b - 128) / 128.0 for b in h[:dim]]


def run(ticket):
    job_type, payload = ticket["job_type"], ticket["payload"]
    if TEST:
        if job_type == "embeddings":
            return [_fake_embed(t) for t in payload["texts"]]
        if job_type == "text":
            return "test output for: " + payload["prompt"][:80]

    if job_type == "embeddings" and ticket["model"] == EMBED_MODEL_NAME:
        ensure_server()
        out = _post("/v1/embeddings", {"input": payload["texts"]})
        rows = sorted(out["data"], key=lambda d: d["index"])
        return [row["embedding"] for row in rows]

    raise NotImplementedError("this phone can't run %s / %s yet" % (job_type, ticket["model"]))
