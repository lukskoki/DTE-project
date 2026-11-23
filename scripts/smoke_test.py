"""Smoke test of a running stack (docker compose up): every service does its job.

    python scripts/smoke_test.py [--shadow http://localhost:8000] [--dashboard http://localhost:8501]

Checks, waiting up to --timeout seconds:
  1. the shadow API answers and the replay clock is moving (replay -> MQTT -> shadow)
  2. sensors and residents' rooms update in the shadow
  3. both residents get activity estimates from the model (inference -> MQTT -> shadow)
  4. the day timeline has estimated segments next to the labels (shadow history in SQLite)
  5. routine statuses arrive (routine checker)
  6. the dashboard serves its page
"""
from __future__ import annotations

import argparse
import sys
import time

import requests


def wait(what: str, check, timeout: float) -> object:
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        try:
            out = check()
            if out:
                print(f"ok    {what}  ({time.time() - t0:.0f}s)")
                return out
        except requests.RequestException as exc:
            last = exc
        time.sleep(2)
    print(f"FAIL  {what}  (after {timeout:.0f}s; last error: {last})")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", default="http://localhost:8000")
    ap.add_argument("--dashboard", default="http://localhost:8501")
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args()
    get = lambda path, **kw: requests.get(a.shadow + path, params=kw, timeout=5).json()  # noqa: E731

    first = wait("shadow API answers with a replay clock", lambda: get("/health").get("sim_ts"), a.timeout)
    wait("replay clock is moving", lambda: get("/health")["sim_ts"] != first, a.timeout)
    wait("sensor messages reach the shadow",
         lambda: sum(s["last_ts"] is not None for s in get("/state")["sensors"].values()) >= 5, a.timeout)
    wait("residents have a room from BLE",
         lambda: all(r["room"] != "unknown" for r in get("/state")["residents"].values()), a.timeout)
    state = wait("both residents have activity estimates",
                 lambda: (s := get("/state")) and all(r["activity"] for r in s["residents"].values()) and s,
                 a.timeout)
    for u, r in state["residents"].items():
        print(f"      {u}: {r['room']}, estimate {r['activity']} ({r['confidence']:.0%}), label {r['true_activity']}")
    tl = wait("timeline has estimated segments", lambda: (t := get("/timeline", user="user1")) and t["predicted"]
              and t, a.timeout)
    print(f"      {len(tl['predicted'])} estimated and {len(tl['actual'])} labelled segments on {tl['date']}")
    wait("routine statuses arrive", lambda: get("/state")["routines"], a.timeout)
    wait("dashboard is up", lambda: requests.get(a.dashboard + "/_stcore/health", timeout=5).ok
         and requests.get(a.dashboard, timeout=5).ok, a.timeout)
    print("all good")


if __name__ == "__main__":
    main()
