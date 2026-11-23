"""Learn the residents' routines on the training days and test the alerts on the test days.

    python -m pipeline.train_routines        (after pipeline.train_model)

Writes
  models/routines.json    routine windows per resident (used by the inference service)
  reports/alerts.json     how the routine check did on test days 48-62:
      real     every routine window on every test day, our status vs the labels
      injected for each medication the residents really took, the medicine-drawer
               events are deleted and the model re-run, to see whether the
               system then flags the dose as missed (a simulated missed dose)
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.train_model import FEATURES_CACHE, META_FILE, MODEL_FILE, REPORT_DIR, TEST_DAYS, TRAIN_DAYS
from shadowhome.config import load_home, sensor_rooms
from shadowhome.features import build_features
from shadowhome.routines import (
    AWAY_SHARE, EARLY_MIN, RoutineChecker, learn_routines, routines_to_json,
)
from shadowhome.schema import EVENTS_FILE, LABELS_FILE, ROOT, USERS

ROUTINES_FILE = ROOT / "models" / "routines.json"
DRAWER = "c2"


def predict(model, meta, feat: pd.DataFrame) -> np.ndarray:
    return np.array(meta["classes"])[model.predict(feat[meta["features"]]).argmax(1)]


def drawer_users(feat_minute: pd.DataFrame, drawer_room: str) -> list[str]:
    """Residents to credit for a drawer opening: those in its room, else those at home."""
    here = [r.user for r in feat_minute.itertuples() if getattr(r, f"me_{drawer_room}") == 1]
    return here or [r.user for r in feat_minute.itertuples() if r.me_away == 0]


def run_checker(routines, feat: pd.DataFrame, pred: np.ndarray, drawer_minutes: set, drawer_room: str,
                start: float, meds_from_model: bool = False) -> RoutineChecker:
    chk = RoutineChecker(routines, started_at=start, meds_from_model=meds_from_model)
    cols = ["minute", "user", "pred", "me_away", f"me_{drawer_room}"]   # only what is read below (fast)
    feat = feat.assign(pred=pred)[cols].sort_values(["minute", "user"])
    for minute, rows in feat.groupby("minute", sort=True):
        ts = minute.timestamp()
        for r in rows.itertuples():
            chk.observe_minute(r.user, ts, r.pred, away=bool(r.me_away))
        if minute in drawer_minutes:
            chk.observe_drawer(drawer_users(rows, drawer_room), ts)
        chk.tick(ts)
    chk.tick(feat["minute"].max().timestamp() + 61)  # close windows whose deadline is the last minute
    return chk


def truth_status(inst, labels: pd.DataFrame) -> str:
    lab = labels[labels["user"] == inst.user]
    s, d = pd.Timestamp(inst.start, unit="s", tz="UTC"), pd.Timestamp(inst.deadline, unit="s", tz="UTC")
    act = lab[(lab["activity"] == inst.window.activity)
              & (lab["start"] >= s - pd.Timedelta(minutes=EARLY_MIN)) & (lab["start"] <= d)]
    if len(act):
        return "done"
    e = pd.Timestamp(inst.end, unit="s", tz="UTC")
    out = lab[(lab["activity"] == "OUT HOME") & (lab["end"] > s) & (lab["start"] < e)]
    away = sum((min(r.end, e) - max(r.start, s)).total_seconds() for r in out.itertuples())
    return "away" if away / max((e - s).total_seconds(), 1) >= AWAY_SHARE else "missed"


def main() -> int:
    labels = pd.read_parquet(LABELS_FILE)
    routines = learn_routines(labels[labels["day"].isin(TRAIN_DAYS)], USERS)
    ROUTINES_FILE.write_text(json.dumps(routines_to_json(routines), indent=2))
    for u, ws in routines.items():
        for w in ws:
            print(f"{u} {w.activity:<9} {w.label()}  on {w.support:.0%} of training days")

    model = lgb.Booster(model_file=str(MODEL_FILE))
    meta = json.loads(META_FILE.read_text())
    feat = pd.read_parquet(FEATURES_CACHE)
    test = feat[feat["day"].isin(TEST_DAYS)].reset_index(drop=True)
    pred = predict(model, meta, test)
    ev = pd.read_parquet(EVENTS_FILE)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    drawer = ev[(ev["sensor_id"].astype(str) == DRAWER) & (ev["value"] == 0)]
    drawer_minutes = set(drawer["ts"].dt.ceil("min"))
    drawer_room = load_home()["sensors"][DRAWER]["room"]
    start = test["minute"].min().timestamp()

    last = test["minute"].max().timestamp()
    rooms = sensor_rooms()
    report = {"routines": routines_to_json(routines)}

    def alert_scores(df):
        tp = int(((df.ours == "missed") & (df.truth == "missed")).sum())
        fp = int(((df.ours == "missed") & (df.truth != "missed")).sum())
        fn = int(((df.ours != "missed") & (df.truth == "missed")).sum())
        return {"windows": len(df), "alerts": tp + fp, "true_misses": tp + fn,
                "correct_alerts": tp, "false_alerts": fp, "missed_misses": fn,
                "status_agreement": round(float((df.ours == df.truth).mean()), 3)}

    for variant, from_model in (("drawer_only", False), ("drawer_or_model", True)):
        chk = run_checker(routines, test, pred, drawer_minutes, drawer_room, start, from_model)
        rows = []
        for inst in chk.instances.values():
            if inst.deadline > last:
                continue
            rows.append({"user": inst.user, "activity": inst.window.activity, "date": inst.date,
                         "window": inst.window.label(), "ours": inst.status,
                         "truth": truth_status(inst, labels)})
        real = pd.DataFrame(rows)
        print(f"\n== {variant}")
        print(pd.crosstab([real["activity"], real["truth"]], real["ours"]))

        # Simulated missed doses: delete the drawer events of each real dose, re-run the model.
        caught, trials = 0, []
        doses = [i for i in chk.instances.values() if i.window.activity == "TAKE MEDS"
                 and truth_status(i, labels) == "done" and i.deadline <= last]
        for inst in doses:
            s = pd.Timestamp(inst.start, unit="s", tz="UTC") - pd.Timedelta(minutes=EARLY_MIN)
            d = pd.Timestamp(inst.deadline, unit="s", tz="UTC")
            keep = ev[(ev["ts"] > s - pd.Timedelta(hours=5)) & (ev["ts"] <= d)]
            keep = keep[~((keep["sensor_id"].astype(str) == DRAWER) & (keep["ts"] >= s))]
            minutes = pd.date_range(s.ceil("min"), d.floor("min"), freq="1min")
            f = build_features(keep, minutes, rooms)
            p = predict(model, meta, f)
            sim = run_checker({inst.user: [inst.window]}, f, p, set(), drawer_room, s.timestamp() - 1,
                              from_model)
            status = [i.status for i in sim.instances.values() if i.user == inst.user and i.date == inst.date]
            flagged = status == ["missed"]
            caught += flagged
            trials.append({"user": inst.user, "date": inst.date, "flagged": flagged})
        print(f"simulated missed doses flagged: {caught} of {len(trials)}")
        report[variant] = {
            "real": {act: alert_scores(g) for act, g in real.groupby("activity")},
            "injected_missed_doses": {"trials": len(trials), "flagged": caught},
            "windows": rows,
        }
        print(json.dumps(report[variant]["real"]))
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "alerts.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
