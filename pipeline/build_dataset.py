"""Turn the raw SDHAR-HOME day zips into two clean tables.

    python -m pipeline.build_dataset

Output
  data/clean/events.parquet  one row per sensor event, all 62 days, sorted by ts
  data/clean/labels.parquet  activity intervals per resident (user, start, end, activity)

Choices made here (worth a line in the report):
  * gyro.csv from the wristbands is skipped (~95% of the raw size, not needed
    for room-level activity recognition).
  * Duplicate rows (same sensor, same value, a few ms apart) are dropped.
  * Ambient sensors keep their dataset id (c1, m3, v7, ...); the room comes
    from config/home.yaml once pipeline.infer_sensor_rooms has run.
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from shadowhome.schema import (
    CLEAN_DIR, DATA_DIR, EVENTS_FILE, EVENT_COLUMNS, LABELS_FILE, ROOM_NAMES,
)

ZIP_DIR = DATA_DIR / "raw" / "SDHAR-HOME"
MAX_LABEL_HOURS = 14  # an interval longer than this means a gap in labelling


def _read(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    raw = zf.read(name)
    if len(raw) < 10:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    if "time" not in df.columns or df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["time"], utc=True, format="ISO8601", errors="coerce")
    return df.dropna(subset=["ts"])


def _rows(df, sensor_id, sensor_type, value, state=None, room=None) -> pd.DataFrame:
    out = pd.DataFrame({
        "ts": df["ts"].values,
        "sensor_id": sensor_id,
        "sensor_type": sensor_type,
        "room": room,
        "value": pd.to_numeric(value, errors="coerce").astype(float).values,
        "state": (state.astype(str).values if state is not None else None),
    })
    return out.dropna(subset=["value"])


def _ambient(df: pd.DataFrame, folder: str, sid: str) -> list[pd.DataFrame]:
    if folder == "Contact":
        closed = pd.to_numeric(df["contact"], errors="coerce")
        return [_rows(df, sid, "contact", closed, closed.map({1.0: "closed", 0.0: "open"}))]
    if folder == "Movement":
        occ = pd.to_numeric(df["occupancy"], errors="coerce")
        return [_rows(df, sid, "movement", occ, occ.map({1.0: "motion", 0.0: "clear"}))]
    if folder == "Vibration":
        act = df.get("action_str", pd.Series(dtype=str)).astype(str)
        hit = df[act.isin(["vibration", "tilt", "drop"])]
        return [_rows(hit, sid, "vibration", pd.Series(1.0, index=hit.index), act[hit.index])]
    if folder == "Power":
        return [_rows(df, sid, "power", df["power"])]
    if folder == "Illuminance":
        return [_rows(df, sid, "illuminance", df["illuminance_lux"])]
    if folder == "TempHum":
        return [_rows(df, sid, "temperature", df["temperature"]),
                _rows(df, sid, "humidity", df["humidity"])]
    return []


def read_day(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = int(re.search(r"DAY_(\d+)", zip_path.name).group(1))
    events, labels = [], []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv") or name.endswith("gyro.csv"):
                continue
            parts = Path(name).parts  # DAY_N/<folder>/<file>.csv
            if len(parts) < 3:
                continue
            folder, stem = parts[-2], Path(name).stem
            df = _read(zf, name)
            if df.empty:
                continue
            if folder == "Activities" and stem.startswith("activity_user_"):
                user = "user" + stem.rsplit("_", 1)[-1]
                labels.append(pd.DataFrame({"user": user, "ts": df["ts"],
                                            "activity": df.iloc[:, 1].astype(str).str.strip()}))
            elif folder == "Presences":
                user = stem.split("_")[0]
                room = df["state"].map(ROOM_NAMES).fillna("unknown")
                events.append(_rows(df, user, "presence", df["distance"], room, room=room))
            elif folder.startswith("BAND_"):
                user = folder.split("_", 1)[1]
                if stem in ("bpm", "steps"):
                    vals = pd.to_numeric(df[stem], errors="coerce")
                    if stem == "bpm":  # the band logs -123 and other junk when it loses contact
                        df, vals = df[vals.between(25, 230)], vals[vals.between(25, 230)]
                    events.append(_rows(df, user, stem, vals))
            else:
                events.extend(_ambient(df, folder, stem))
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=EVENT_COLUMNS)
    ev["day"] = day
    lb = pd.concat(labels, ignore_index=True) if labels else pd.DataFrame(columns=["user", "ts", "activity"])
    lb["day"] = day
    return ev, lb


def dedupe(ev: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated rows: same sensor and value as the previous row within 1 s."""
    ev = ev.sort_values(["sensor_id", "sensor_type", "ts"], kind="stable")
    same = (
        (ev["sensor_id"] == ev["sensor_id"].shift())
        & (ev["sensor_type"] == ev["sensor_type"].shift())
        & (ev["value"] == ev["value"].shift())
        & (ev["state"].fillna("") == ev["state"].shift().fillna(""))
        & ((ev["ts"] - ev["ts"].shift()) < pd.Timedelta(seconds=1))
    )
    return ev[~same]


def to_intervals(points: pd.DataFrame) -> pd.DataFrame:
    """Change-point labels -> intervals that last until the next change."""
    out = []
    for user, g in points.sort_values("ts").groupby("user"):
        g = g.drop_duplicates("ts", keep="last")
        start = g["ts"].values
        end = np.append(start[1:], start[-1] + np.timedelta64(1, "h"))
        iv = pd.DataFrame({"user": user, "start": start, "end": end,
                           "activity": g["activity"].values, "day": g["day"].values})
        too_long = (iv["end"] - iv["start"]) > pd.Timedelta(hours=MAX_LABEL_HOURS)
        iv.loc[too_long, "end"] = iv.loc[too_long, "start"] + pd.Timedelta(hours=MAX_LABEL_HOURS)
        out.append(iv)
    iv = pd.concat(out, ignore_index=True)
    iv["start"] = pd.to_datetime(iv["start"], utc=True)
    iv["end"] = pd.to_datetime(iv["end"], utc=True)
    return iv


def main() -> int:
    zips = sorted(ZIP_DIR.glob("DAY_*.zip"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    if not zips:
        print(f"No zips in {ZIP_DIR}. Run scripts/download_data.sh first.", file=sys.stderr)
        return 1
    all_ev, all_lb = [], []
    for z in zips:
        ev, lb = read_day(z)
        all_ev.append(ev)
        all_lb.append(lb)
        print(f"{z.name}: {len(ev):>7} events, {len(lb):>3} label changes")
    ev = pd.concat(all_ev, ignore_index=True)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    ev = dedupe(ev)
    ev = ev.sort_values("ts", kind="stable")[EVENT_COLUMNS].reset_index(drop=True)
    for col in ("sensor_id", "sensor_type", "room", "state"):
        ev[col] = ev[col].astype("category")
    labels = to_intervals(pd.concat(all_lb, ignore_index=True))
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(EVENTS_FILE, index=False)
    labels.to_parquet(LABELS_FILE, index=False)
    print(f"\n{len(ev):,} events -> {EVENTS_FILE.relative_to(DATA_DIR.parent)}")
    print(f"{len(labels):,} label intervals -> {LABELS_FILE.relative_to(DATA_DIR.parent)}")
    print(labels.assign(h=(labels.end - labels.start).dt.total_seconds() / 3600)
          .groupby(["user", "activity"])["h"].sum().round(1).unstack(0).fillna(0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
