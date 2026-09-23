"""Turns one customer job into many small tickets a phone can finish quickly."""

EMBED_BATCH = 32  # texts per embeddings ticket


def est_tokens(text):
    """Rough token estimate (about 4 characters per token). Computed server-side
    so phones cannot inflate their own pay."""
    return max(1, len(text) // 4)


def split(job_type, data):
    """Return a list of (payload, estimated_tokens)."""
    if job_type == "embeddings":
        texts = [t for t in (data.get("texts") or []) if isinstance(t, str) and t]
        out = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            out.append(({"texts": batch}, sum(est_tokens(t) for t in batch)))
        return out

    if job_type == "text":
        prompts = [p for p in (data.get("prompts") or []) if isinstance(p, str) and p]
        max_tokens = int(data.get("max_tokens", 256))
        return [
            ({"prompt": p, "max_tokens": max_tokens, "temperature": 0},
             est_tokens(p) + max_tokens)
            for p in prompts
        ]

    raise ValueError("unsupported job_type: %s" % job_type)


def billable_tokens(job_type, payload, result):
    """Tokens to credit for a verified ticket."""
    if job_type == "embeddings":
        return sum(est_tokens(t) for t in payload["texts"])
    if job_type == "text":
        out = result if isinstance(result, str) else ""
        return est_tokens(payload["prompt"]) + est_tokens(out)
    return 0
