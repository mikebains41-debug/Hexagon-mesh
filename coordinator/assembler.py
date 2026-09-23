"""Stitches verified ticket results back into one output, in order."""
import json


def assemble(job_type, tickets):
    tickets = sorted(tickets, key=lambda t: t["idx"])
    results = [json.loads(t["result"]) for t in tickets]
    if job_type == "embeddings":
        return {"vectors": [v for r in results for v in r]}
    if job_type == "text":
        return {"outputs": results}
    raise ValueError("unsupported job_type: %s" % job_type)
