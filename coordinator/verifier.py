"""Decides whether two phones' results for the same ticket agree."""
import difflib
import math

EMBED_MIN_COSINE = 0.99   # embeddings must be near-identical
TEXT_MIN_RATIO = 0.90     # text outputs (temperature 0) must be very similar


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def agree(job_type, payload, a, b):
    try:
        if job_type == "embeddings":
            n = len(payload["texts"])
            if not (isinstance(a, list) and isinstance(b, list)):
                return False
            if len(a) != n or len(b) != n:
                return False
            for va, vb in zip(a, b):
                if not va or len(va) != len(vb):
                    return False
                if _cosine(va, vb) < EMBED_MIN_COSINE:
                    return False
            return True

        if job_type == "text":
            if not (isinstance(a, str) and isinstance(b, str)) or not a or not b:
                return False
            return difflib.SequenceMatcher(None, a, b).ratio() >= TEXT_MIN_RATIO

    except (TypeError, ValueError, KeyError):
        return False
    return False
