"""Work out where each ambient sensor is, and write config/home.yaml.

    python -m pipeline.infer_sensor_rooms

The dataset does not say which sensor id sits where. Two kinds of evidence:

1. Room vote. For every "activity" event of a sensor (motion, door opened,
   vibration, a power or light jump) look up the room the residents were in
   (BLE beacons). Only unambiguous moments vote: one resident home, or both in
   the same room. Votes are normalised by how much time people spend in each
   room, otherwise the bedroom would win everything.
2. Activity lift. Which labelled activity is going on when the sensor fires,
   compared with how common that activity is. A contact sensor that opens
   mostly during TAKE MEDS is the medicine drawer.

config/sensor_objects.yaml (hand-curated from that evidence and the paper)
names each sensor's object and overrides the room where the vote is weak.
"""
from __future__ import annotations

import re
import sys

import pandas as pd
import yaml

from shadowhome.schema import (
    ACTIVITIES, EVENTS_FILE, HOME_CONFIG, LABELS_FILE, ROOMS, ROOT, USERS,
)

OBJECTS_FILE = ROOT / "config" / "sensor_objects.yaml"
MIN_VOTES = 20

# Schematic floor plan for the dashboard: [x, y, w, h] on a 100 x 60 canvas.
# The paper publishes no floor plan, so this is a layout for display only.
LAYOUT = {
    "bedroom":     [0, 0, 34, 32],
    "study":       [34, 0, 22, 32],
    "kitchen":     [56, 0, 44, 26],
    "bathroom":    [0, 32, 22, 28],
    "hallway":     [22, 32, 22, 28],
    "living_room": [44, 26, 56, 34],
}


def activity_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Rows that mean 'someone did something near this sensor'."""
    t = ev["sensor_type"].astype(str)
    out = [ev[(t == "movement") & (ev["value"] == 1)],
           ev[(t == "contact") & (ev["value"] == 0)],
           ev[t == "vibration"]]
    for kind in ("power", "illuminance"):
        e = ev[t == kind].sort_values("ts")
        jump = e.groupby("sensor_id", observed=True)["value"].transform(
            lambda v: v > 1.5 * v.rolling(20, min_periods=1).median().shift().fillna(v.median()) + 1)
        out.append(e[jump])
    acts = pd.concat(out).sort_values("ts")[["ts", "sensor_id", "sensor_type"]]
    acts["sensor_id"] = acts["sensor_id"].astype(str)
    return acts


def room_votes(acts: pd.DataFrame, ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    pres = ev[ev["sensor_type"].astype(str) == "presence"][["ts", "sensor_id", "room"]]
    m = acts
    for user in USERS:
        p = pres[pres["sensor_id"].astype(str) == user][["ts", "room"]].sort_values("ts")
        m = pd.merge_asof(m, p.rename(columns={"room": user}), on="ts",
                          direction="backward", tolerance=pd.Timedelta(minutes=2))
    rooms = m[USERS].astype(str).where(m[USERS].astype(str).isin(ROOMS))
    unique = rooms.nunique(axis=1) == 1
    m["room"] = rooms.bfill(axis=1).iloc[:, 0]
    v = m[unique & m["room"].notna()]
    return v, v["room"].value_counts(normalize=True)


def activity_lift(acts: pd.DataFrame) -> dict[str, list[str]]:
    labels = pd.read_parquet(LABELS_FILE)
    rows = []
    for user in USERS:
        lab = labels[labels["user"] == user].sort_values("start")
        lab = lab[["start", "end", "activity"]].rename(columns={"start": "ts"})
        m = pd.merge_asof(acts, lab, on="ts", direction="backward")
        rows.append(m[m["ts"] < m["end"]])
    m = pd.concat(rows)
    m = m[m["activity"] != "OUT HOME"]  # a sensor firing while someone is out says nothing
    base = m["activity"].value_counts(normalize=True)
    top = {}
    for sid, g in m.groupby("sensor_id"):
        vc = g["activity"].value_counts()
        lift = ((vc / vc.sum()) / base.reindex(vc.index)).sort_values(ascending=False)
        top[sid] = [f"{a} x{lift[a]:.1f} (n={vc[a]})" for a in lift.index if vc[a] >= 5][:3]
    return top


def sort_key(sid: str):
    return re.sub(r"\d", "", sid), int(re.sub(r"\D", "", sid) or 0)


def main() -> int:
    ev = pd.read_parquet(EVENTS_FILE)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    acts = activity_events(ev)
    votes, base = room_votes(acts, ev)
    lifts = activity_lift(acts)
    curated = yaml.safe_load(open(OBJECTS_FILE)) if OBJECTS_FILE.exists() else {}

    ambient = ev[~ev["sensor_type"].astype(str).isin(["presence", "bpm", "steps"])]
    sensors = {}
    for sid, kinds in ambient.groupby("sensor_id", observed=True)["sensor_type"]:
        sid = str(sid)
        kind = sorted(set(map(str, kinds)))
        kind = "temphum" if "temperature" in kind else kind[0]
        counts = votes[votes["sensor_id"] == sid]["room"].astype(str).value_counts()
        voted, share, lift = "unknown", 0.0, 0.0
        if counts.sum() >= MIN_VOTES:
            ratio = (counts / counts.sum()) / base.reindex(counts.index)
            voted = str(ratio.idxmax())
            share = round(float(counts[voted] / counts.sum()), 2)
            lift = round(float(ratio.max()), 1)
        cur = curated.get(sid, {}) or {}
        entry = {
            "type": kind,
            "room": cur.get("room", voted),
            "object": cur.get("object", "unknown"),
            "room_source": "curated" if "room" in cur else "ble-vote",
            "ble_vote": {"room": voted, "share": share, "lift": lift, "votes": int(counts.sum())},
            "top_activities": lifts.get(sid, []),
        }
        if cur.get("evidence"):
            entry["evidence"] = cur["evidence"]
        if kind == "power":  # "in use" threshold for the dashboard
            entry["active_above_w"] = round(max(5.0, 1.5 * float(
                ambient[(ambient["sensor_id"].astype(str) == sid)]["value"].median())), 1)
        sensors[sid] = entry
        print(f"{sid:>4} {kind:<12} vote={voted:<12} share={share:.2f} lift={lift:<4} "
              f"-> {entry['room']:<12} {entry['object']}")

    cfg = {
        "rooms": {r: {"box": LAYOUT[r]} for r in ROOMS},
        "residents": USERS,
        "activities": ACTIVITIES,
        "sensors": dict(sorted(sensors.items(), key=lambda kv: sort_key(kv[0]))),
    }
    with open(HOME_CONFIG, "w") as f:
        f.write("# Generated by `python -m pipeline.infer_sensor_rooms` from the data and\n"
                "# config/sensor_objects.yaml. Edit sensor_objects.yaml, then re-run.\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, width=120,
                       default_flow_style=None)
    print(f"wrote {HOME_CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
