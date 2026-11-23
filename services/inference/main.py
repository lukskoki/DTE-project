"""Inference service: every simulated minute, estimate each resident's activity
and check their routines.

    python -m services.inference.main

Listens to the replayed sensor, location and wristband topics, keeps a rolling
buffer of recent events, builds the same features as in training
(shadowhome.features), runs the LightGBM model and publishes
  home/residents/{user}/activity   {ts, user, activity, confidence, probs, model_version}
  home/residents/{user}/routine    {ts, user, items: [{activity, window, date, status, ...}]}
  home/alerts                      {ts, user, kind, expected_at, date, message}
"""
from __future__ import annotations

import json
import logging
import threading

import lightgbm as lgb
import numpy as np
import pandas as pd

from shadowhome import mqtt as bus
from shadowhome.config import load_home, sensor_rooms
from shadowhome.features import LOOKBACK_MIN, build_features
from shadowhome.routines import RoutineChecker, routines_from_json
from shadowhome.schema import ALERTS_TOPIC, ALL_TOPICS, CLOCK_TOPIC, ROOT, activity_topic
from services.inference.buffer import EventBuffer

log = logging.getLogger("inference")
MODEL_FILE = ROOT / "models" / "activity_lgbm.txt"
META_FILE = ROOT / "models" / "activity_meta.json"
ROUTINES_FILE = ROOT / "models" / "routines.json"
DRAWER = "c2"                 # medicine drawer (see config/sensor_objects.yaml)
RESET_GAP = pd.Timedelta(hours=6)


class InferenceService:
    def __init__(self):
        self.model = lgb.Booster(model_file=str(MODEL_FILE))
        self.meta = json.loads(META_FILE.read_text())
        self.classes = np.array(self.meta["classes"])
        self.rooms = sensor_rooms()
        self.drawer_room = load_home()["sensors"].get(DRAWER, {}).get("room", "bathroom")
        self.routines = routines_from_json(json.loads(ROUTINES_FILE.read_text())) if ROUTINES_FILE.exists() else {}
        self.lock = threading.Lock()
        self.buffer = EventBuffer(keep_minutes=LOOKBACK_MIN + 5)
        self.client = None
        self.reset(None)

    def reset(self, minute: pd.Timestamp | None) -> None:
        self.buffer.clear()
        self.drawer_open: list[pd.Timestamp] = []
        self.last_minute = minute
        self.checker = RoutineChecker(self.routines, started_at=minute.timestamp()) if minute is not None else None

    # --- MQTT -------------------------------------------------------------------
    def on_message(self, client, userdata, msg) -> None:
        p = bus.decode(msg)
        if p is None:
            return
        parts = msg.topic.split("/")
        with self.lock:
            try:
                if msg.topic == CLOCK_TOPIC:
                    self.on_clock(pd.Timestamp(p["sim_ts"]))
                elif parts[1] == "sensors":
                    ts = pd.Timestamp(p["ts"])
                    self.buffer.add(ts, p["sensor_id"], p["sensor_type"], p.get("room"), p.get("value"),
                                    p.get("state"))
                    if p["sensor_id"] == DRAWER and p.get("value") == 0:
                        self.drawer_open.append(ts)
                elif parts[1] == "residents" and len(parts) == 4:
                    user, kind = parts[2], parts[3]
                    if kind == "location":
                        self.buffer.add(pd.Timestamp(p["ts"]), user, "presence", p.get("room"),
                                        p.get("distance"), p.get("room"))
                    elif kind in ("bpm", "steps"):
                        self.buffer.add(pd.Timestamp(p["ts"]), user, kind, None, p.get("value"), None)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("bad message on %s: %s", msg.topic, exc)

    def on_clock(self, now: pd.Timestamp) -> None:
        minute = now.floor("min")
        if self.last_minute is None or minute < self.last_minute or minute - self.last_minute > RESET_GAP:
            if self.last_minute is not None:
                log.info("replay jumped to %s: starting fresh", minute)
            # events of the new position that arrived just before this clock message
            keep = [r for r in self.buffer.rows if abs(r[0] - now) <= pd.Timedelta(minutes=2)]
            self.reset(minute)
            for r in keep:
                self.buffer.add(*r)
            return
        if minute <= self.last_minute:
            return
        minutes = pd.date_range(self.last_minute + pd.Timedelta(minutes=1), minute, freq="1min")
        self.process(minutes)
        self.last_minute = minute
        self.buffer.trim(minute)

    # --- the work -----------------------------------------------------------------
    def process(self, minutes: pd.DatetimeIndex) -> None:
        feat = build_features(self.buffer.frame(), minutes, self.rooms)
        probs = self.model.predict(feat[self.meta["features"]])
        best = probs.argmax(1)
        for i, row in enumerate(feat.itertuples()):
            top = np.argsort(probs[i])[::-1][:3]
            bus.publish(self.client, activity_topic(row.user), {
                "ts": row.minute.isoformat(), "user": row.user,
                "activity": str(self.classes[best[i]]), "confidence": round(float(probs[i, best[i]]), 3),
                "probs": {str(self.classes[j]): round(float(probs[i, j]), 3) for j in top},
                "model_version": self.meta.get("version", "unknown"),
            })
        feat["pred"] = self.classes[best]
        for minute, rows in feat.groupby("minute", sort=True):
            ts = minute.timestamp()
            for r in rows.itertuples():
                self.checker.observe_minute(r.user, ts, r.pred, away=bool(r.me_away))
            opened = [t for t in self.drawer_open if minute - pd.Timedelta(minutes=1) < t <= minute]
            if opened:
                here = [r.user for r in rows.itertuples() if getattr(r, f"me_{self.drawer_room}", 0) == 1]
                home = [r.user for r in rows.itertuples() if r.me_away == 0]
                self.checker.observe_drawer(here or home, ts)
            for alert in self.checker.tick(ts):
                log.info("ALERT %s", alert["message"])
                bus.publish(self.client, ALERTS_TOPIC, alert)
        self.drawer_open = [t for t in self.drawer_open if t > minutes[-1]]
        for user in self.routines:
            bus.publish(self.client, f"home/residents/{user}/routine",
                        {"ts": minutes[-1].isoformat(), "user": user, "items": self.checker.status(user)})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    svc = InferenceService()
    log.info("model %s, %d features, routines for %s", svc.meta.get("version"), len(svc.meta["features"]),
             {u: [w.label() + " " + w.activity for w in ws] for u, ws in svc.routines.items()})
    svc.client = bus.connect("inference", on_message=svc.on_message, subscribe=[ALL_TOPICS])
    threading.Event().wait()


if __name__ == "__main__":
    main()
