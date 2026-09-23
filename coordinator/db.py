"""SQLite storage for jobs, tickets, nodes, assignments and the credit ledger."""
import os
import sqlite3
import threading
from contextlib import contextmanager

DB_PATH = os.environ.get("HM_DB", "hexagon_mesh.db")
LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    wallet TEXT DEFAULT '',
    ram_gb REAL DEFAULT 0,
    models TEXT DEFAULT '[]',
    last_seen REAL,
    strikes INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT,
    model TEXT,
    status TEXT,
    total_tickets INTEGER,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    idx INTEGER,
    payload TEXT,
    est_tokens INTEGER,
    needed INTEGER DEFAULT 2,
    status TEXT DEFAULT 'open',
    result TEXT
);
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT,
    node_id TEXT,
    assigned_at REAL,
    deadline REAL,
    status TEXT,
    result TEXT
);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT,
    amount REAL,
    reason TEXT,
    ticket_id TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_assign_ticket ON assignments(ticket_id);
CREATE INDEX IF NOT EXISTS idx_ledger_node ON ledger(node_id);
"""


def connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def tx():
    """One locked transaction. Simple and safe for a single coordinator process."""
    with LOCK:
        conn = connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db():
    with tx() as c:
        c.executescript(SCHEMA)
