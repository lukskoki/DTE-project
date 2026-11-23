"""Digital shadow service: subscribes to the home's MQTT topics, keeps the
model up to date, stores history, and serves it over HTTP and WebSocket.

    uvicorn services.shadow.app:app --port 8000

Endpoints
  GET  /state                         current snapshot of rooms, sensors, residents, alerts
  GET  /config                        rooms, floor plan, sensor -> room/object map
  GET  /days                          local dates that have data
  GET  /timeline?user=user2&date=...  estimated vs labelled activity, rooms, alerts for one day
  GET  /usual?user=user2              the resident's usual day: mean labelled hours per activity, train days
  GET  /alerts                        recent alerts
  POST /replay                        forward a control command to the replayer
  WS   /stream                        pushes /state once per second
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from shadowhome import mqtt as bus
from shadowhome.config import load_home
from shadowhome.schema import (
    ALERTS_TOPIC, ALL_TOPICS, CLOCK_TOPIC, CONTROL_TOPIC, FIRST_DAY, LABELS_FILE, TIMEZONE, TRAIN_DAYS,
)
from services.shadow.model import HomeShadow, parse_ts
from services.shadow.store import Store

log = logging.getLogger("shadow")
DB_PATH = os.getenv("SHADOW_DB", "/tmp/shadow.db")


def load_labels() -> pd.DataFrame | None:
    if not LABELS_FILE.exists():
        log.warning("no %s: ground truth will be empty", LABELS_FILE)
        return None
    lab = pd.read_parquet(LABELS_FILE)
    lab["start"] = pd.to_datetime(lab["start"], utc=True)
    lab["end"] = pd.to_datetime(lab["end"], utc=True)
    return lab


class ShadowService:
    def __init__(self):
        self.labels = load_labels()
        self.shadow = HomeShadow(load_home(), self.labels)
        self.store = Store(DB_PATH)
        self.client = None

    def start(self) -> None:
        self.client = bus.connect("shadow", on_message=self.on_message, subscribe=[ALL_TOPICS])

    # topic layout: home/sensors/{type}/{id}, home/residents/{user}/{kind},
    # home/replay/clock, home/alerts
    def on_message(self, client, userdata, msg) -> None:
        p = bus.decode(msg)
        if p is None:
            return
        parts = msg.topic.split("/")
        try:
            if msg.topic == CLOCK_TOPIC:
                jumped_back, went_away = self.shadow.on_clock(p)
                if jumped_back:
                    log.info("clock jumped back to %s: clearing later history", p["sim_ts"])
                    self.store.truncate_after(parse_ts(p["sim_ts"]))
                for user, ts in went_away:
                    self.store.write("location_log", (ts, user, "away"))
            elif parts[1] == "sensors":
                self.shadow.on_sensor(p)
                self.store.write("sensor_log", (parse_ts(p["ts"]), p["sensor_id"], p["sensor_type"],
                                                p.get("room"), p.get("value"), p.get("state")))
            elif parts[1] == "residents" and len(parts) == 4:
                user, kind = parts[2], parts[3]
                if kind == "location":
                    if self.shadow.on_location(p):
                        self.store.write("location_log", (parse_ts(p["ts"]), user, p.get("room")))
                elif kind in ("bpm", "steps"):
                    self.shadow.on_wearable(user, kind, p)
                    if kind == "bpm":
                        self.store.write("vitals_log", (parse_ts(p["ts"]), user, kind, p.get("value")))
                elif kind == "activity":
                    self.shadow.on_activity(p)
                    self.store.write("activity_log", (parse_ts(p["ts"]), user, p.get("activity"),
                                                      p.get("confidence"), p.get("model_version")))
                elif kind == "routine":
                    self.shadow.on_routine(user, p)
            elif msg.topic == ALERTS_TOPIC:
                self.shadow.on_alert(p)
                self.store.write("alert_log", (parse_ts(p["ts"]), p.get("user"), p.get("kind"),
                                               p.get("expected_at"), p.get("message")))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("bad message on %s: %s", msg.topic, exc)

    # --- timeline ------------------------------------------------------------
    @staticmethod
    def day_bounds(date: str) -> tuple[float, float]:
        start = pd.Timestamp(date).tz_localize(TIMEZONE)
        return start.timestamp(), (start + pd.Timedelta(days=1)).timestamp()

    @staticmethod
    def segments(rows: list[tuple], key: int, t0: float, t1: float, extra=None,
                 max_len: float = float("inf")) -> list[dict]:
        """Rows (ts, value, ...) sorted by ts -> [{start, end, value}], merging repeats.

        A row lasts until the next row, at most max_len seconds (so a gap in
        the estimates does not stretch the previous one over it)."""
        out = []
        for i, row in enumerate(rows):
            ts, val = row[0], row[key]
            nxt = rows[i + 1][0] if i + 1 < len(rows) else t1
            end = min(nxt, ts + max_len, t1)
            if out and out[-1]["value"] == val and abs(out[-1]["end"] - ts) < 1e-6:
                out[-1]["end"] = end
                continue
            seg = {"start": max(ts, t0), "end": min(end, t1), "value": val}
            if extra:
                seg.update(extra(row))
            out.append(seg)
        return out

    def timeline(self, user: str, date: str) -> dict:
        t0, t1 = self.day_bounds(date)
        pred = self.store.query(
            "SELECT ts, activity, confidence FROM activity_log WHERE user=? AND ts>? AND ts<=? ORDER BY ts",
            (user, t0, t1))
        loc_before = self.store.query(
            "SELECT ts, room FROM location_log WHERE user=? AND ts<? ORDER BY ts DESC LIMIT 1", (user, t0))
        loc = self.store.query(
            "SELECT ts, room FROM location_log WHERE user=? AND ts>=? AND ts<? ORDER BY ts", (user, t0, t1))
        alerts = self.store.query(
            "SELECT ts, kind, expected_at, message FROM alert_log WHERE user=? AND ts>=? AND ts<? ORDER BY ts",
            (user, t0, t1))
        actual = self.labelled(user, t0, t1)
        loc_rows = [(t0, loc_before[0][1])] + loc if loc_before else loc
        # an estimate stamped at minute m describes the minute (m - 1 min, m]
        predicted = self.segments([(ts - 60, act, conf) for ts, act, conf in pred], 1, t0, t1, max_len=60)
        return {
            "user": user, "date": date, "timezone": TIMEZONE, "start": t0, "end": t1,
            "predicted": predicted,
            "actual": actual,
            "location": self.segments(loc_rows, 1, t0, t1),
            "alerts": [{"ts": a[0], "kind": a[1], "expected_at": a[2], "message": a[3]} for a in alerts],
            "hours_predicted": self.hours(predicted),
            "hours_actual": self.hours(actual),
        }

    def labelled(self, user: str, t0: float, t1: float) -> list[dict]:
        """Ground-truth activity segments of a resident, clipped to [t0, t1)."""
        if self.labels is None:
            return []
        lab = self.labels[self.labels["user"] == user]
        t0_ts, t1_ts = pd.Timestamp(t0, unit="s", tz="UTC"), pd.Timestamp(t1, unit="s", tz="UTC")
        lab = lab[(lab["end"] > t0_ts) & (lab["start"] < t1_ts)]
        return [{"start": max(r.start.timestamp(), t0), "end": min(r.end.timestamp(), t1), "value": r.activity}
                for r in lab.itertuples()]

    @functools.lru_cache(maxsize=8)
    def usual(self, user: str) -> dict:
        """The resident's usual day: mean labelled hours per activity over the full local
        days inside the training days (the first local day starts at 02:00, so it is skipped)."""
        first = pd.Timestamp(FIRST_DAY) + pd.Timedelta(days=1)
        dates = pd.date_range(first, periods=TRAIN_DAYS[-1] - TRAIN_DAYS[0], freq="D").strftime("%Y-%m-%d")
        total: dict[str, float] = {}
        for d in dates:
            t0, t1 = self.day_bounds(d)
            for act, h in self.hours(self.labelled(user, t0, t1)).items():
                total[act] = total.get(act, 0.0) + h
        n = len(dates)
        return {"user": user, "from": dates[0], "to": dates[-1], "days": n,
                "hours": {a: round(h / n, 2) for a, h in sorted(total.items(), key=lambda kv: -kv[1])}}

    @staticmethod
    def hours(segs: list[dict]) -> dict:
        out: dict[str, float] = {}
        for s in segs:
            out[s["value"]] = out.get(s["value"], 0.0) + (s["end"] - s["start"]) / 3600
        return {k: round(v, 2) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    def days(self) -> list[str]:
        if self.labels is None:
            return []
        local = self.labels["start"].dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d")
        return sorted(local.unique().tolist())


service = ShadowService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.start()
    yield


app = FastAPI(title="SDHAR-HOME digital shadow", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True, "sim_ts": service.shadow.clock.get("sim_ts")}


@app.get("/state")
def state():
    return service.shadow.snapshot()


@app.get("/config")
def config():
    return load_home()


@app.get("/days")
def days():
    return service.days()


@app.get("/timeline")
def timeline(user: str, date: str | None = None):
    if date is None:
        sim = service.shadow.clock.get("sim_ts")
        date = (pd.Timestamp(sim).tz_convert(TIMEZONE) if sim else pd.Timestamp.now(TIMEZONE)).strftime("%Y-%m-%d")
    return service.timeline(user, date)


@app.get("/usual")
def usual(user: str):
    return service.usual(user)


@app.get("/alerts")
def alerts(limit: int = 50):
    rows = service.store.query(
        "SELECT ts, user, kind, expected_at, message FROM alert_log ORDER BY ts DESC LIMIT ?", (limit,))
    return [{"ts": r[0], "user": r[1], "kind": r[2], "expected_at": r[3], "message": r[4]} for r in rows]


@app.post("/replay")
def replay(cmd: dict):
    if service.client is None:
        raise HTTPException(503, "the shadow is not connected to MQTT")
    bus.publish(service.client, CONTROL_TOPIC, cmd)
    return {"sent": cmd}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(service.shadow.snapshot())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
