"""Customer side for testing: submit a job or check on one.

  python tools/submit_job.py            submit a 100-text embeddings job
  python tools/submit_job.py ID         show that job's status
"""
import json
import os
import sys
import urllib.request

URL = os.environ.get("HM_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("HM_MODEL", "sim-embed")


def call(method, path, data=None):
    req = urllib.request.Request(URL + path, method=method,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


if len(sys.argv) > 1:
    s = call("GET", "/jobs/" + sys.argv[1])
    if "result" in s:
        s["result"] = "%d vectors" % len(s["result"].get("vectors", s["result"].get("outputs", [])))
    print(s)
else:
    n = int(os.environ.get("HM_TEXTS", "100"))
    texts = ["sample document %d" % i for i in range(n)]
    print(call("POST", "/jobs", {"job_type": "embeddings", "model": MODEL, "input": {"texts": texts}}))
