"""Hexagon Mesh coordinator.

Customers submit jobs. Phones check in, pull tickets, and submit results.
Each ticket goes to 2 phones; matching results are accepted and credited.
Run from the repo root:  python -m coordinator.app
"""
import json
import os
import time
import uuid

from flask import Flask, abort, jsonify, request

from . import assembler, db, splitter, verifier

REPLICAS = int(os.environ.get("HM_REPLICAS", "2"))   # phones per ticket (1 = solo testing)
MAX_REPLICAS = 4             # give up after this many disagreeing results
LEASE_SECONDS = int(os.environ.get("HM_LEASE", "300"))
CREDITS_PER_1K_TOKENS = 1.0
STRIKE_LIMIT = 5             # rejected results before a node is suspended
CUSTOMER_KEY = os.environ.get("HM_CUSTOMER_KEY")  # unset = open (dev only)
NODE_KEY = os.environ.get("HM_NODE_KEY")          # unset = open (dev only)

app = Flask(__name__)
db.init_db()


def require(key):
    if key and request.headers.get("X-API-Key") != key:
        abort(401)


def body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400)
    return data


# ---------------------------------------------------------------- customers

@app.post("/jobs")
def create_job():
    require(CUSTOMER_KEY)
    d = body()
    job_type, model = d.get("job_type"), d.get("model")
    if not job_type or not model:
        return jsonify(error="job_type and model are required"), 400
    try:
        pieces = splitter.split(job_type, d.get("input") or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if not pieces:
        return jsonify(error="empty input"), 400

    job_id = uuid.uuid4().hex
    now = time.time()
    with db.tx() as c:
        c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?)",
                  (job_id, job_type, model, "running", len(pieces), now))
        for i, (payload, est) in enumerate(pieces):
            c.execute(
                "INSERT INTO tickets (id, job_id, idx, payload, est_tokens, needed, status) "
                "VALUES (?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, job_id, i, json.dumps(payload), est, REPLICAS, "open"))
    return jsonify(job_id=job_id, tickets=len(pieces)), 201


@app.get("/jobs/<job_id>")
def job_status(job_id):
    require(CUSTOMER_KEY)
    with db.tx() as c:
        job = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            abort(404)
        counts = {r["status"]: r["n"] for r in c.execute(
            "SELECT status, COUNT(*) AS n FROM tickets WHERE job_id=? GROUP BY status", (job_id,))}
        out = {"job_id": job_id, "status": job["status"], "total": job["total_tickets"],
               "done": counts.get("done", 0), "failed": counts.get("failed", 0)}
        if job["status"] == "complete":
            rows = [dict(r) for r in c.execute(
                "SELECT idx, result FROM tickets WHERE job_id=?", (job_id,))]
            out["result"] = assembler.assemble(job["job_type"], rows)
    return jsonify(out)


# -------------------------------------------------------------------- nodes

@app.post("/nodes/checkin")
def checkin():
    require(NODE_KEY)
    d = body()
    node_id = d.get("node_id") or uuid.uuid4().hex
    with db.tx() as c:
        c.execute(
            "INSERT INTO nodes (id, wallet, ram_gb, models, last_seen) VALUES (?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET wallet=excluded.wallet, ram_gb=excluded.ram_gb, "
            "models=excluded.models, last_seen=excluded.last_seen",
            (node_id, d.get("wallet", ""), float(d.get("ram_gb", 0)),
             json.dumps(d.get("models") or []), time.time()))
    return jsonify(node_id=node_id)


@app.post("/work/next")
def next_work():
    require(NODE_KEY)
    node_id = body().get("node_id")
    now = time.time()
    with db.tx() as c:
        node = c.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not node:
            abort(404)
        if node["strikes"] >= STRIKE_LIMIT:
            return jsonify(error="node suspended"), 403
        c.execute("UPDATE nodes SET last_seen=? WHERE id=?", (now, node_id))
        # Leases that ran out go back to the pool.
        c.execute("UPDATE assignments SET status='expired' "
                  "WHERE status='assigned' AND deadline < ?", (now,))

        models = json.loads(node["models"] or "[]")
        if not models:
            return jsonify(ticket=None)
        marks = ",".join("?" * len(models))
        row = c.execute(f"""
            SELECT t.*, j.job_type, j.model FROM tickets t
            JOIN jobs j ON j.id = t.job_id
            WHERE t.status = 'open' AND j.model IN ({marks})
              AND (SELECT COUNT(*) FROM assignments a
                   WHERE a.ticket_id = t.id AND a.status IN ('assigned', 'submitted')) < t.needed
              AND NOT EXISTS (
                   SELECT 1 FROM assignments a JOIN nodes n ON n.id = a.node_id
                   WHERE a.ticket_id = t.id AND a.status <> 'expired'
                     AND (a.node_id = ? OR (n.wallet <> '' AND n.wallet = ?)))
            ORDER BY j.created_at, t.idx
            LIMIT 1""", (*models, node_id, node["wallet"])).fetchone()
        if not row:
            return jsonify(ticket=None)

        cur = c.execute(
            "INSERT INTO assignments (ticket_id, node_id, assigned_at, deadline, status) "
            "VALUES (?,?,?,?,?)", (row["id"], node_id, now, now + LEASE_SECONDS, "assigned"))
        ticket = {"assignment_id": cur.lastrowid, "ticket_id": row["id"],
                  "job_type": row["job_type"], "model": row["model"],
                  "payload": json.loads(row["payload"]), "deadline": now + LEASE_SECONDS}
    return jsonify(ticket=ticket)


@app.post("/work/submit")
def submit():
    require(NODE_KEY)
    d = body()
    node_id, aid = d.get("node_id"), d.get("assignment_id")
    now = time.time()
    with db.tx() as c:
        a = c.execute("SELECT * FROM assignments WHERE id=? AND node_id=?",
                      (aid, node_id)).fetchone()
        if not a:
            abort(404)
        if a["status"] != "assigned" or a["deadline"] < now:
            return jsonify(error="assignment not active"), 409
        c.execute("UPDATE assignments SET status='submitted', result=? WHERE id=?",
                  (json.dumps(d.get("result")), aid))
        outcome = resolve_ticket(c, a["ticket_id"], now)
    return jsonify(status=outcome)


@app.get("/nodes/<node_id>/balance")
def balance(node_id):
    require(NODE_KEY)
    with db.tx() as c:
        total = c.execute("SELECT COALESCE(SUM(amount), 0) FROM ledger WHERE node_id=?",
                          (node_id,)).fetchone()[0]
    return jsonify(node_id=node_id, credits=round(total, 6))


# -------------------------------------------------------------- verification

def resolve_ticket(c, ticket_id, now):
    t = c.execute("SELECT t.*, j.job_type FROM tickets t JOIN jobs j ON j.id = t.job_id "
                  "WHERE t.id=?", (ticket_id,)).fetchone()
    if t["status"] != "open":
        return t["status"]
    payload = json.loads(t["payload"])
    subs = c.execute("SELECT * FROM assignments WHERE ticket_id=? AND status='submitted' "
                     "ORDER BY id", (ticket_id,)).fetchall()
    if len(subs) < min(2, t["needed"]):
        return "waiting"

    results = [(s, json.loads(s["result"])) for s in subs]
    winner = results[0][1] if t["needed"] == 1 else None
    for i in range(len(results)):
        for k in range(i + 1, len(results)):
            if verifier.agree(t["job_type"], payload, results[i][1], results[k][1]):
                winner = results[i][1]
                break
        if winner is not None:
            break

    if winner is None:
        if len(subs) >= t["needed"]:
            if t["needed"] >= MAX_REPLICAS:
                c.execute("UPDATE tickets SET status='failed' WHERE id=?", (ticket_id,))
                finish_job_if_done(c, t["job_id"])
                return "failed"
            c.execute("UPDATE tickets SET needed = needed + 1 WHERE id=?", (ticket_id,))
        return "disputed"

    credits = CREDITS_PER_1K_TOKENS * splitter.billable_tokens(
        t["job_type"], payload, winner) / 1000.0
    c.execute("UPDATE tickets SET status='done', result=? WHERE id=?",
              (json.dumps(winner), ticket_id))
    for s, r in results:
        if verifier.agree(t["job_type"], payload, winner, r):
            c.execute("UPDATE assignments SET status='accepted' WHERE id=?", (s["id"],))
            c.execute("INSERT INTO ledger (node_id, amount, reason, ticket_id, created_at) "
                      "VALUES (?,?,?,?,?)", (s["node_id"], credits, "verified_work", ticket_id, now))
        else:
            c.execute("UPDATE assignments SET status='rejected' WHERE id=?", (s["id"],))
            c.execute("UPDATE nodes SET strikes = strikes + 1 WHERE id=?", (s["node_id"],))
    finish_job_if_done(c, t["job_id"])
    return "verified"


def finish_job_if_done(c, job_id):
    open_left = c.execute("SELECT COUNT(*) FROM tickets WHERE job_id=? AND status='open'",
                          (job_id,)).fetchone()[0]
    if open_left:
        return
    failed = c.execute("SELECT COUNT(*) FROM tickets WHERE job_id=? AND status='failed'",
                       (job_id,)).fetchone()[0]
    c.execute("UPDATE jobs SET status=? WHERE id=?",
              ("partial" if failed else "complete", job_id))


if __name__ == "__main__":
    app.run(host=os.environ.get("HM_HOST", "127.0.0.1"),
            port=int(os.environ.get("HM_PORT", "8080")), threaded=True)
