"""End-to-end test with fake phones (no models needed).

Two honest phones and one cheater pull tickets for an embeddings job.
Expected: job completes, honest phones earn credits, cheater gets suspended.
Start the coordinator first, then run:  python tools/simulate.py
"""
import hashlib
import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("HM_URL", "http://127.0.0.1:8080")
MODEL = "sim-embed"


def call(method, path, data=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code}


def fake_embed(text, dim=16):
    h = hashlib.sha256(text.encode()).digest()
    return [(b - 128) / 128.0 for b in h[:dim]]


def main():
    texts = ["sample document %d" % i for i in range(100)]
    job = call("POST", "/jobs", {"job_type": "embeddings", "model": MODEL,
                                 "input": {"texts": texts}})
    print("job:", job)

    nodes = []
    for name, wallet, honest in [("phone-a", "wallet-a", True),
                                 ("phone-b", "wallet-b", True),
                                 ("cheater", "wallet-c", False)]:
        r = call("POST", "/nodes/checkin", {"node_id": name, "wallet": wallet,
                                            "ram_gb": 12, "models": [MODEL]})
        nodes.append((r["node_id"], honest))

    for _ in range(1000):
        if call("GET", "/jobs/" + job["job_id"]).get("status") != "running":
            break
        for node_id, honest in nodes:
            work = call("POST", "/work/next", {"node_id": node_id}).get("ticket")
            if not work:
                continue
            batch = work["payload"]["texts"]
            result = [fake_embed(t) if honest else [0.5] * 16 for t in batch]
            call("POST", "/work/submit", {"node_id": node_id,
                                          "assignment_id": work["assignment_id"],
                                          "result": result})

    status = call("GET", "/jobs/" + job["job_id"])
    print("final:", {k: v for k, v in status.items() if k != "result"})
    if "result" in status:
        print("vectors returned:", len(status["result"]["vectors"]))
    for node_id, _ in nodes:
        print(node_id, call("GET", "/nodes/%s/balance" % node_id))


if __name__ == "__main__":
    main()
