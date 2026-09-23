"""Runs one ticket and returns the result.

HM_TEST=1 : fake but deterministic results (no model needed), for testing the pipeline.
Otherwise  : real inference via llama.cpp (added in Step 3).
"""
import hashlib
import os

TEST = os.environ.get("HM_TEST", "0") == "1"
TEST_MODEL = "sim-embed"


def models():
    """Model names this phone can run; sent to the coordinator at check-in."""
    if TEST:
        return [TEST_MODEL]
    return [m for m in os.environ.get("HM_MODELS", "").split(",") if m]


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
    raise NotImplementedError("real inference arrives in Step 3 (llama.cpp); use HM_TEST=1 for now")
