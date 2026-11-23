"""Features for activity recognition, one row per resident per minute.

The same function is used offline (training on all 62 days) and online (the
inference service runs it on a buffer of recent events), so the model sees
identical features in both places. tests/test_features.py checks that.

Row (user u, minute m) describes the window (m - 1 min, m]:
  time       hour of day as sin/cos (local time), weekend flag
  location   u's room from BLE (one-hot, "away" if no fix for 10 min), minutes
             since the last fix, minutes in the current room; the other
             resident's room; whether both are in the same room
  ambient    per sensor (motion, door opened, vibration): activations in the
             last 1, 5 and 15 minutes and minutes since it last fired; doors
             open now; power draw of the TV and washing machine; temperature,
             humidity and their change over 15 min; light level; activations
             per room and in u's own room
  wearable   u's heart rate and steps in the last 10 minutes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shadowhome.schema import ROOMS, TIMEZONE, USERS

LOCATIONS = ROOMS + ["away"]
AWAY_AFTER_MIN = 10
LOOKBACK_MIN = 240       # history needed before the first requested minute
AGO_CAP_MIN = 240
CONTINUOUS = ["powe_p1", "powe_p2", "temp_th1", "temp_th2", "humi_th1", "humi_th2", "illu_l1", "illu_l2"]


def activation_sensors(sensor_rooms: dict[str, str]) -> list[str]:
    ids = [s for s in sensor_rooms if s[:1] in ("c", "m", "v") and s[1:].isdigit()]
    return sorted(ids, key=lambda s: (s[0], int(s[1:])))


def _bins(ts: pd.Series) -> pd.Series:
    return ts.dt.ceil("min")


def _last_on_grid(ev: pd.DataFrame, grid: pd.DatetimeIndex, limit: int | None) -> pd.DataFrame:
    """Per key, the last value at or before each grid minute, at most `limit` minutes old."""
    if ev.empty:
        return pd.DataFrame(index=grid)
    last = (ev.assign(minute=_bins(ev["ts"])).groupby(["key", "minute"], observed=True)["value"].last()
              .reset_index().sort_values("minute"))
    g = pd.DataFrame({"minute": grid})
    tol = pd.Timedelta(minutes=limit) if limit is not None else None
    out = {}
    for key, k in last.groupby("key", observed=True):
        m = pd.merge_asof(g, k[["minute", "value"]], on="minute", direction="backward", tolerance=tol)
        out[str(key)] = m["value"].to_numpy()
    return pd.DataFrame(out, index=grid)


def ambient_features(ev: pd.DataFrame, grid: pd.DatetimeIndex, sensor_rooms: dict[str, str]) -> pd.DataFrame:
    t = ev["sensor_type"].astype(str)
    ids = activation_sensors(sensor_rooms)
    out = {}

    act = ev[((t == "movement") & (ev["value"] == 1)) | ((t == "contact") & (ev["value"] == 0))
             | (t == "vibration")]
    counts = (act.assign(bin=_bins(act["ts"]), sid=act["sensor_id"].astype(str))
                 .groupby(["bin", "sid"]).size().unstack("sid"))
    counts = counts.reindex(index=grid, columns=ids).fillna(0)
    c5 = counts.rolling(5, min_periods=1).sum()
    c15 = counts.rolling(15, min_periods=1).sum()
    pos = pd.Series(np.arange(len(grid)), index=grid, dtype=float)
    for s in ids:
        out[f"n1_{s}"] = counts[s].to_numpy()
        out[f"n5_{s}"] = c5[s].to_numpy()
        out[f"n15_{s}"] = c15[s].to_numpy()
        last = pos.where(counts[s] > 0).ffill()
        out[f"ago_{s}"] = (pos - last).clip(upper=AGO_CAP_MIN).fillna(AGO_CAP_MIN).to_numpy()
    for room in ROOMS:
        cols = [s for s in ids if sensor_rooms.get(s) == room]
        out[f"room5_{room}"] = c5[cols].sum(axis=1).to_numpy() if cols else np.zeros(len(grid))

    contact = ev[t == "contact"].assign(key=ev.loc[t == "contact", "sensor_id"].astype(str))
    doors = _last_on_grid(contact, grid, limit=None)
    for s in [x for x in ids if x.startswith("c")]:
        col = doors[s] if s in doors else pd.Series(np.nan, index=grid)
        out[f"open_{s}"] = (col == 0).astype(float).to_numpy()

    cont = ev[t.isin(["power", "temperature", "humidity", "illuminance"])]
    cont = cont.assign(key=cont["sensor_type"].astype(str).str[:4] + "_" + cont["sensor_id"].astype(str))
    vals = _last_on_grid(cont, grid, limit=180)
    for key in CONTINUOUS:
        v = vals[key] if key in vals else pd.Series(np.nan, index=grid)
        out[key] = v.to_numpy()
        if key[:4] in ("temp", "humi"):
            out[f"d15_{key}"] = (v - v.shift(15)).to_numpy()
    return pd.DataFrame(out, index=grid)


def location_features(ev: pd.DataFrame, grid: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Per user on the grid: room ('away' after 10 min without a fix), minutes since fix, minutes in room."""
    pres = ev[ev["sensor_type"].astype(str) == "presence"]
    res = {}
    g = pd.DataFrame({"minute": grid})
    for user in USERS:
        p = pres[pres["sensor_id"].astype(str) == user]
        fixes = (p.assign(minute=_bins(p["ts"])).groupby("minute")["room"].last().astype(object)
                  .reset_index().rename(columns={"room": "fix_room"}))
        fixes["fix_minute"] = fixes["minute"]
        if fixes.empty:
            m = g.assign(fix_room=None, fix_minute=pd.Series(pd.NaT, index=g.index, dtype=grid.dtype))
        else:
            m = pd.merge_asof(g, fixes, on="minute", direction="backward")
        since = (m["minute"] - m["fix_minute"]) / pd.Timedelta(minutes=1)
        room = m["fix_room"].where(since <= AWAY_AFTER_MIN).fillna("away")
        room = room.where(room.isin(LOCATIONS), "away")
        run = (room != room.shift()).cumsum()
        res[user] = pd.DataFrame({
            "room": room.to_numpy(),
            "since_fix": since.clip(upper=120).fillna(120).to_numpy(),
            "in_room": room.groupby(run).cumcount().clip(upper=240).to_numpy(),
        }, index=grid)
    return res


def wearable_features(ev: pd.DataFrame, grid: pd.DatetimeIndex, user: str) -> pd.DataFrame:
    w = ev[(ev["sensor_id"].astype(str) == user) & ev["sensor_type"].astype(str).isin(["bpm", "steps"])]
    w = w.assign(key=w["sensor_type"].astype(str))
    bpm = _last_on_grid(w[w["key"] == "bpm"], grid, limit=5)
    steps = _last_on_grid(w[w["key"] == "steps"], grid, limit=30)
    s = steps["steps"] if "steps" in steps else pd.Series(np.nan, index=grid)
    d = s - s.shift(10)
    return pd.DataFrame({
        "bpm": (bpm["bpm"] if "bpm" in bpm else pd.Series(np.nan, index=grid)).to_numpy(),
        "steps10": d.where(d >= 0).to_numpy(),
    }, index=grid)


def build_features(ev: pd.DataFrame, minutes: pd.DatetimeIndex, sensor_rooms: dict[str, str]) -> pd.DataFrame:
    """Feature rows for every user and every minute in `minutes` (UTC, on the minute).

    `ev` holds the events up to the last minute. It should cover the
    LOOKBACK_MIN minutes before minutes[0]; older events only give the last
    known state (door open, power draw, room).
    """
    # One time unit everywhere (parsing ISO strings and reading Parquet can give us or ns).
    minutes = pd.DatetimeIndex(minutes).as_unit("ns")
    ev = ev.assign(ts=pd.to_datetime(ev["ts"], utc=True).astype("datetime64[ns, UTC]"))
    ev = ev[ev["ts"] <= minutes[-1]]
    grid = pd.date_range(minutes[0] - pd.Timedelta(minutes=LOOKBACK_MIN), minutes[-1], freq="1min", unit="ns")
    amb = ambient_features(ev, grid, sensor_rooms)
    loc = location_features(ev, grid)
    local = grid.tz_convert(TIMEZONE)
    hour = local.hour + local.minute / 60
    base = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekend": (local.dayofweek >= 5).astype(float),
    }, index=grid)
    rows = []
    for user in USERS:
        other = [u for u in USERS if u != user][0]
        me, them = loc[user], loc[other]
        f = pd.concat([base, amb], axis=1)
        f["is_user1"] = float(user == "user1")
        for r in LOCATIONS:
            f[f"me_{r}"] = (me["room"] == r).astype(float)
            f[f"other_{r}"] = (them["room"] == r).astype(float)
        f["same_room"] = ((me["room"] == them["room"]) & (me["room"] != "away")).astype(float)
        f["since_fix"] = me["since_fix"]
        f["in_room"] = me["in_room"]
        own = np.zeros(len(grid))
        for r in ROOMS:
            own += np.where(me["room"].to_numpy() == r, amb[f"room5_{r}"].to_numpy(), 0)
        f["own_room5"] = own
        f = pd.concat([f, wearable_features(ev, grid, user)], axis=1).reindex(minutes)
        f.insert(0, "user", user)
        f.insert(1, "minute", minutes)
        rows.append(f.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


FEATURE_EXCLUDE = {"user", "minute", "activity", "day"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]
