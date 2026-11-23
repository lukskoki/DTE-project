"""Replay the recorded SDHAR-HOME events onto MQTT in simulated time.

    python -m services.replay.main

Each recorded event is published when the simulated clock passes its
timestamp, so the rest of the system sees the home "live". The simulated clock
runs SPEED times faster than real time (300 = one day in under 5 minutes).

Environment
  MQTT_HOST / MQTT_PORT  broker (default localhost:1883)
  REPLAY_SPEED           simulated seconds per real second (default 300)
  REPLAY_DAYS            dataset days to play, e.g. "48-62" (default: the test days)
  REPLAY_LOOP            1 = start again after the last day (default 1)

Control at runtime by publishing JSON to home/replay/control:
  {"action": "seek", "day": 55}   {"action": "speed", "value": 600}
  {"action": "pause"}             {"action": "resume"}
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np
import pandas as pd

from shadowhome import mqtt as bus
from shadowhome.config import sensor_rooms
from shadowhome.schema import (
    CLOCK_TOPIC, CONTROL_TOPIC, EVENTS_FILE, location_topic, sensor_topic, wearable_topic,
)

log = logging.getLogger("replay")
CLOCK_EVERY = 0.5  # real seconds between clock messages


def parse_days(spec: str) -> tuple[int, int]:
    a, _, b = spec.partition("-")
    return int(a), int(b or a)


def iso(ns: int) -> str:
    return pd.Timestamp(ns, tz="UTC").isoformat().replace("+00:00", "Z")


class Replayer:
    def __init__(self, events: pd.DataFrame, client, speed: float, days: tuple[int, int], loop: bool):
        rooms = sensor_rooms()
        self.client, self.speed, self.loop = client, speed, loop
        self.first_day, self.last_day = days
        self.ts = events["ts"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()  # ns, UTC
        self.day = events["day"].to_numpy()
        self.sid = events["sensor_id"].astype(str).to_numpy()
        self.kind = events["sensor_type"].astype(str).to_numpy()
        self.value = events["value"].to_numpy()
        self.state = events["state"].astype(object).where(events["state"].notna(), None).to_numpy()
        room = events["room"].astype(object).where(events["room"].notna(), None).to_numpy()
        self.room = np.array([r if r is not None else rooms.get(s, "unknown")
                              for r, s in zip(room, self.sid)], dtype=object)
        # Each dataset day file is cut at UTC midnight.
        self.day_start = {int(d): int(pd.Timestamp(t).tz_convert("UTC").floor("D").as_unit("ns").value)
                          for d, t in events.groupby("day")["ts"].min().items()}
        self.lock = threading.Lock()
        self.paused = False
        self.seek(self.first_day)

    # --- control ----------------------------------------------------------
    def seek(self, day: int) -> None:
        day = min(max(day, min(self.day_start)), max(self.day_start))
        with self.lock:
            self.sim_ns = self.day_start.get(day, self.day_start[min(self.day_start)])
            self.i = int(np.searchsorted(self.ts, self.sim_ns))
            self.anchor_wall, self.anchor_sim = time.monotonic(), self.sim_ns
        log.info("seek to day %s (%s)", day, iso(self.sim_ns))
        self.publish_clock()

    def set_speed(self, speed: float) -> None:
        with self.lock:
            self.anchor_wall, self.anchor_sim = time.monotonic(), self.sim_ns
            self.speed = max(1.0, float(speed))
        self.publish_clock()

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self.paused = paused
            self.anchor_wall, self.anchor_sim = time.monotonic(), self.sim_ns
        self.publish_clock()

    def on_control(self, client, userdata, msg) -> None:
        cmd = bus.decode(msg) or {}
        action = cmd.get("action")
        log.info("control: %s", cmd)
        if action == "seek" and "day" in cmd:
            self.seek(int(cmd["day"]))
        elif action == "speed" and "value" in cmd:
            self.set_speed(float(cmd["value"]))
        elif action in ("pause", "resume"):
            self.set_paused(action == "pause")

    # --- publishing -------------------------------------------------------
    def current_day(self) -> int:
        i = min(self.i, len(self.day) - 1)
        return int(self.day[i])

    def publish_clock(self) -> None:
        bus.publish(self.client, CLOCK_TOPIC, {
            "sim_ts": iso(self.sim_ns), "speed": self.speed, "day": self.current_day(),
            "paused": self.paused, "days": [self.first_day, self.last_day],
        }, retain=True)

    def publish_event(self, i: int) -> None:
        kind, sid, ts = self.kind[i], self.sid[i], iso(self.ts[i])
        if kind == "presence":
            bus.publish(self.client, location_topic(sid),
                        {"ts": ts, "user": sid, "room": self.room[i], "distance": float(self.value[i])})
        elif kind in ("bpm", "steps"):
            bus.publish(self.client, wearable_topic(sid, kind),
                        {"ts": ts, "user": sid, "value": float(self.value[i])})
        else:
            bus.publish(self.client, sensor_topic(kind, sid), {
                "ts": ts, "sensor_id": sid, "sensor_type": kind, "room": self.room[i],
                "value": float(self.value[i]), "state": self.state[i],
            })

    def advance_to(self, sim_ns: int) -> bool:
        """Publish every event up to sim_ns. Returns True once the last day is done."""
        self.sim_ns = sim_ns
        while self.i < len(self.ts) and self.ts[self.i] <= sim_ns and self.day[self.i] <= self.last_day:
            self.publish_event(self.i)
            self.i += 1
        return self.i >= len(self.ts) or self.day[self.i] > self.last_day

    def run(self) -> None:
        last_clock = 0.0
        while True:
            now = time.monotonic()
            with self.lock:
                if self.paused:
                    finished = self.i >= len(self.ts) or self.day[self.i] > self.last_day
                else:
                    finished = self.advance_to(int(self.anchor_sim + (now - self.anchor_wall) * self.speed * 1e9))
            if now - last_clock >= CLOCK_EVERY:
                self.publish_clock()
                last_clock = now
            if finished and not self.paused:
                if self.loop:
                    log.info("reached the end of day %s, looping", self.last_day)
                    self.seek(self.first_day)
                else:
                    self.set_paused(True)
            time.sleep(0.02)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    speed = float(os.getenv("REPLAY_SPEED", "300"))
    days = parse_days(os.getenv("REPLAY_DAYS", "48-62"))
    loop = os.getenv("REPLAY_LOOP", "1") == "1"
    events = pd.read_parquet(EVENTS_FILE)
    events["ts"] = pd.to_datetime(events["ts"], utc=True)
    events = events.sort_values("ts", kind="stable").reset_index(drop=True)
    log.info("loaded %s events, days %s-%s, speed %sx", len(events), *days, speed)
    replayer: Replayer | None = None

    def on_message(client, userdata, msg):
        if replayer is not None:
            replayer.on_control(client, userdata, msg)

    client = bus.connect("replay", on_message=on_message, subscribe=[CONTROL_TOPIC])
    replayer = Replayer(events, client, speed, days, loop)
    replayer.run()


if __name__ == "__main__":
    main()
