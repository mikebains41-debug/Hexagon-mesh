"""Hexagon Mesh benchmark: sustained real embedding work on this phone.

Feeds realistic-length documents to the model for several minutes and logs
speed, temperature, battery and (when unplugged) power draw to a CSV.
Run from ~/Hexagon-mesh:   python -m bench.bench

Settings:
  HM_BENCH_MINUTES     how long to run (default 10)
  HM_BENCH_STOP_C      stop early if battery reaches this temperature (default 42)
  HM_BASELINE_SECONDS  idle measurement before starting, unplugged only (default 30)
  HM_THREADS           CPU threads for the model (default 4)
Tip: run UNPLUGGED to measure power and energy per token.
     Run PLUGGED IN to measure overnight-style speed and heat.
"""
import csv
import os
import random
import statistics
import time

from node import runner
from node.eligibility import battery

MINUTES = float(os.environ.get("HM_BENCH_MINUTES", "10"))
STOP_C = float(os.environ.get("HM_BENCH_STOP_C", "42"))
BASELINE_S = int(os.environ.get("HM_BASELINE_SECONDS", "30"))
BATCH = 32
NOMINAL_V = 3.85  # typical phone battery voltage, used for power estimates

WORDS = """the a an and or but of to in on for with from by at as is are was were be been
network phone data model energy system report customer market service order price value
growth team product design process result policy research study analysis support update
security privacy device battery signal cloud server compute memory storage speed quality
city region company project meeting plan budget schedule contract partner supply demand
review summary document translation transcript article email ticket request response
people family school health travel weather music video image story history science
quickly carefully often usually recently clearly strongly slowly directly easily
important new large small local global public private final early recent common
increase reduce improve provide include require create develop measure deliver share
""".split()


def make_docs(n, seed):
    """Deterministic documents of about 180-320 words (roughly 250-420 tokens)."""
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        target, words = rng.randint(180, 320), []
        while len(words) < target:
            s = [rng.choice(WORDS) for _ in range(rng.randint(8, 20))]
            s[0] = s[0].capitalize()
            words.extend(s[:-1] + [s[-1] + "."])
        docs.append(" ".join(words))
    return docs


def current_ma(b):
    """Battery current in mA. Android reports microamps or milliamps depending on the phone."""
    v = b.get("current")
    if v is None:
        return None
    v = abs(float(v))
    return v / 1000.0 if v > 20000 else v


def sample():
    b = battery() or {}
    return {"temp_c": float(b.get("temperature", 0)), "battery_pct": int(b.get("percentage", 0)),
            "plugged": b.get("plugged", "UNKNOWN"), "current_ma": current_ma(b)}


def avg(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def main():
    if runner.TEST:
        print("Unset HM_TEST: the benchmark needs the real model.")
        return
    start_state = sample()
    unplugged = start_state["plugged"] == "UNPLUGGED"
    print("Hexagon Mesh benchmark: %.0f min, %s threads, %s, %.1fC, %d%%" % (
        MINUTES, runner.THREADS, "UNPLUGGED (measuring power)" if unplugged else "plugged in",
        start_state["temp_c"], start_state["battery_pct"]))
    runner.ensure_server()

    idle_ma = []
    if unplugged and BASELINE_S > 0:
        print("Measuring idle power for %ds, leave the phone alone..." % BASELINE_S)
        end = time.time() + BASELINE_S
        while time.time() < end:
            idle_ma.append(sample()["current_ma"])
            time.sleep(3)

    os.makedirs("bench/results", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    csv_path = "bench/results/bench-%s.csv" % stamp
    fields = ["t_s", "docs", "tokens", "batch_s", "tok_per_s", "temp_c", "battery_pct", "plugged", "current_ma"]
    rows, t_start, last_print, i = [], time.time(), 0, 0
    stop_reason = "time limit reached"

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        try:
            while time.time() - t_start < MINUTES * 60:
                docs = make_docs(BATCH, seed=i)
                t0 = time.time()
                out = runner._post("/v1/embeddings", {"input": docs})
                dt = time.time() - t0
                tokens = (out.get("usage") or {}).get("prompt_tokens") or sum(len(d) // 4 for d in docs)
                row = {"t_s": round(time.time() - t_start, 1), "docs": len(docs), "tokens": tokens,
                       "batch_s": round(dt, 3), "tok_per_s": round(tokens / dt, 1)}
                row.update(sample())
                w.writerow(row)
                fh.flush()
                rows.append(row)
                i += 1
                if time.time() - last_print > 30:
                    last_print = time.time()
                    print("%5.0fs  %7.0f tok/s  %.1fC  %d%%  %s" % (
                        row["t_s"], row["tok_per_s"], row["temp_c"], row["battery_pct"],
                        "" if row["current_ma"] is None else "%.0f mA" % row["current_ma"]))
                if row["temp_c"] >= STOP_C:
                    stop_reason = "stopped at %.1fC (safety limit %.0fC)" % (row["temp_c"], STOP_C)
                    break
        except KeyboardInterrupt:
            stop_reason = "stopped by you"

    if not rows:
        print("No data collected.")
        return

    elapsed = rows[-1]["t_s"]
    total_tokens = sum(r["tokens"] for r in rows)
    work_s = sum(r["batch_s"] for r in rows)
    tps = total_tokens / work_s
    early = [r["tok_per_s"] for r in rows if r["t_s"] <= 120]
    late = [r["tok_per_s"] for r in rows if r["t_s"] >= elapsed - 120]
    lines = [
        "HEXAGON MESH BENCHMARK  %s" % stamp,
        "Result: %s after %.1f min" % (stop_reason, elapsed / 60),
        "Documents: %d  Tokens: %d" % (len(rows) * BATCH, total_tokens),
        "Average speed: %.0f tokens/sec" % tps,
        "First 2 min: %.0f tok/s   Last 2 min: %.0f tok/s   (change %+.0f%%)" % (
            avg(early), avg(late), (avg(late) / avg(early) - 1) * 100),
        "Temperature: %.1fC start, %.1fC peak, %.1fC end" % (
            start_state["temp_c"], max(r["temp_c"] for r in rows), rows[-1]["temp_c"]),
        "Battery: %d%% -> %d%%" % (start_state["battery_pct"], rows[-1]["battery_pct"]),
        "If sustained for a 6-hour night: ~%.0f million tokens" % (tps * 6 * 3600 / 1e6),
    ]
    active_ma = avg([r["current_ma"] for r in rows])
    if unplugged and active_ma:
        idle = avg(idle_ma) or 0
        net_w = max(active_ma - idle, 0) / 1000 * NOMINAL_V
        lines.append("Power: %.0f mA working, %.0f mA idle -> ~%.2f W for AI work (at %.2fV nominal)" % (
            active_ma, idle, net_w, NOMINAL_V))
        if net_w > 0:
            lines.append("Energy: ~%.2f millijoules per token" % (net_w / tps * 1000))
    elif not unplugged:
        lines.append("Power: run unplugged to measure energy per token")
    lines.append("Data: %s" % csv_path)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open("bench/results/summary-%s.txt" % stamp, "w") as fh:
        fh.write(summary + "\n")


if __name__ == "__main__":
    main()
