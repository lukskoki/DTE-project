"""The digital shadow: an in-memory model of the home, kept in sync with MQTT.

Entities (the representation model)
  Home      -> Rooms -> Sensors (attached to household objects)
            -> Residents (room, vitals, estimated activity, true activity)
            -> Alerts, simulated clock

It is a *shadow*, not a twin: data flows one way, from the (replayed) house
into the model. Nothing is sent back to the house.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from shadowhome.schema import ROOMS

AWAY_AFTER_S = 10 * 60      # no BLE fix for 10 simulated minutes -> resident is away
VIBRATION_ACTIVE_S = 60     # a vibration counts as "active" for a minute
MOTION_ACTIVE_S = 120       # motion without a "clear" message expires after 2 minutes


def parse_ts(s: str) -> float:
    return pd.Timestamp(s).timestamp()


@dataclass
class Sensor:
    id: str
    type: str
    room: str
    object: str
    value: float | None = None
    state: str | None = None
    last_ts: float | None = None
    last_active_ts: float | None = None
    active_above_w: float | None = None

    def update(self, ts: float, value: float, state: str | None) -> None:
        self.value, self.state, self.last_ts = value, state, ts
        if self._is_activation(value):
            self.last_active_ts = ts

    def _is_activation(self, value: float) -> bool:
        if self.type == "movement":
            return value == 1
        if self.type == "contact":
            return value == 0  # 0 = open
        if self.type == "vibration":
            return True
        if self.type == "power":
            return value > (self.active_above_w or 5)
        return False

    def active(self, now: float) -> bool:
        if self.last_ts is None:
            return False
        if self.type == "movement":
            return self.value == 1 and now - self.last_ts < MOTION_ACTIVE_S
        if self.type == "contact":
            return self.value == 0
        if self.type == "vibration":
            return self.last_active_ts is not None and now - self.last_active_ts < VIBRATION_ACTIVE_S
        if self.type == "power":
            return (self.value or 0) > (self.active_above_w or 5)
        return False


@dataclass
class Resident:
    id: str
    room: str = "unknown"
    room_since: float | None = None
    distance: float | None = None
    last_seen: float | None = None
    bpm: float | None = None
    steps: float | None = None
    activity: str | None = None
    confidence: float | None = None
    probs: dict = field(default_factory=dict)
    activity_since: float | None = None
    model_version: str | None = None
    true_activity: str | None = None


class GroundTruth:
    """Looks up the labelled activity of a resident at a given time."""

    def __init__(self, labels: pd.DataFrame | None):
        self.by_user = {}
        if labels is None or labels.empty:
            return
        for user, g in labels.sort_values("start").groupby("user"):
            self.by_user[user] = (
                g["start"].map(pd.Timestamp.timestamp).to_numpy(),
                g["end"].map(pd.Timestamp.timestamp).to_numpy(),
                g["activity"].to_numpy(),
            )

    def at(self, user: str, ts: float) -> str | None:
        if user not in self.by_user:
            return None
        start, end, act = self.by_user[user]
        i = int(np.searchsorted(start, ts, side="right")) - 1
        return str(act[i]) if i >= 0 and ts < end[i] else None


class HomeShadow:
    def __init__(self, home_cfg: dict, labels: pd.DataFrame | None = None):
        self.lock = threading.RLock()
        self.cfg = home_cfg
        self.truth = GroundTruth(labels)
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.sensors = {
                sid: Sensor(sid, s["type"], s.get("room", "unknown"), s.get("object", "unknown"),
                            active_above_w=s.get("active_above_w"))
                for sid, s in self.cfg["sensors"].items()
            }
            self.residents = {u: Resident(u) for u in self.cfg["residents"]}
            self.alerts: deque = deque(maxlen=200)
            self.routines: dict = {}
            self.clock = {"sim_ts": None, "speed": None, "day": None, "paused": None}
            self.sim_now: float | None = None

    # --- updates from MQTT --------------------------------------------------
    def on_clock(self, p: dict) -> tuple[bool, list[tuple[str, float]]]:
        """Advance the clock. Returns (jumped_back, residents who just went away).

        jumped_back is True after a seek backwards: history after it is stale.
        """
        with self.lock:
            new = parse_ts(p["sim_ts"])
            jumped_back = self.sim_now is not None and new < self.sim_now - 60
            if jumped_back:
                self.reset()
            self.clock = p
            self.sim_now = new
            went_away = []
            for r in self.residents.values():
                r.true_activity = self.truth.at(r.id, new)
                if r.last_seen is not None and new - r.last_seen > AWAY_AFTER_S and r.room != "away":
                    r.room, r.room_since = "away", r.last_seen
                    went_away.append((r.id, r.last_seen))
            return jumped_back, went_away

    def on_sensor(self, p: dict) -> None:
        with self.lock:
            s = self.sensors.get(p["sensor_id"])
            if s is None:  # a sensor the config does not know: still keep it
                s = self.sensors[p["sensor_id"]] = Sensor(p["sensor_id"], p["sensor_type"],
                                                          p.get("room") or "unknown", "unknown")
            s.update(parse_ts(p["ts"]), p.get("value"), p.get("state"))

    def on_location(self, p: dict) -> bool:
        """Returns True when the resident changed room."""
        with self.lock:
            r = self.residents.setdefault(p["user"], Resident(p["user"]))
            ts = parse_ts(p["ts"])
            r.last_seen, r.distance = ts, p.get("distance")
            room = p.get("room") or "unknown"
            if room != r.room:
                r.room, r.room_since = room, ts
                return True
            return False

    def on_wearable(self, user: str, kind: str, p: dict) -> None:
        with self.lock:
            r = self.residents.setdefault(user, Resident(user))
            setattr(r, kind, p.get("value"))

    def on_activity(self, p: dict) -> None:
        with self.lock:
            r = self.residents.setdefault(p["user"], Resident(p["user"]))
            ts = parse_ts(p["ts"])
            if p.get("activity") != r.activity:
                r.activity_since = ts
            r.activity, r.confidence = p.get("activity"), p.get("confidence")
            r.probs, r.model_version = p.get("probs") or {}, p.get("model_version")

    def on_alert(self, p: dict) -> None:
        with self.lock:
            self.alerts.appendleft(p)

    def on_routine(self, user: str, p: dict) -> None:
        with self.lock:
            self.routines[user] = p

    # --- read side ---------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            now = self.sim_now or 0.0
            rooms = {}
            for name in ROOMS:
                in_room = [s for s in self.sensors.values() if s.room == name]
                rooms[name] = {
                    "occupants": [r.id for r in self.residents.values() if r.room == name],
                    "active_sensors": [s.id for s in in_room if s.active(now)],
                    "sensors": [s.id for s in in_room],
                    "box": self.cfg["rooms"][name]["box"],
                }
            return {
                "clock": self.clock,
                "rooms": rooms,
                "sensors": {sid: {**asdict(s), "active": s.active(now)} for sid, s in self.sensors.items()},
                "residents": {u: asdict(r) for u, r in self.residents.items()},
                "alerts": list(self.alerts)[:20],
                "routines": self.routines,
            }
