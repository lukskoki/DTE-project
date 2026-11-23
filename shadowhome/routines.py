"""Daily routines: learn when each resident usually takes medication and eats,
then check, day by day, whether it happened.

Learning (from the training days' labels):
  start times of each activity, as local minutes of the day, are grouped into
  clusters (times closer than 60 min belong together; the circle is cut at
  its largest gap, so 23:50 and 00:10 are neighbours). A cluster that occurs
  on at least 30% of days is a routine. Its window runs from the 10th to the
  90th percentile of start times, widened by 15 minutes on each side; the
  deadline is 30 minutes after the window.

Checking (online, in the inference service, or offline for evaluation):
  each window on each day is "pending" until evidence arrives between 30 min
  before the window and the deadline, then "done". Past the deadline with no
  evidence it is "missed" and raises an alert, unless the resident was away
  for most of the window ("away").
  Evidence for TAKE MEDS: the medicine drawer opens while the resident is in
  its room, or the model says TAKE MEDS. (On the test days both rules give the
  same result: the model's TAKE MEDS output leans on the drawer sensor.)
  Evidence for EAT: the model says EAT for at least 3 minutes in total.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from shadowhome.schema import TIMEZONE

TRACKED = ("TAKE MEDS", "EAT")
GAP_MIN = 60
MIN_SUPPORT = 0.3
WIDEN_MIN = 15
GRACE_MIN = 30
EARLY_MIN = 30
EAT_MIN_MINUTES = 3
AWAY_SHARE = 0.8


@dataclass
class Window:
    activity: str
    start_min: int        # local minutes after midnight; may exceed 1440 (crosses midnight)
    end_min: int
    deadline_min: int
    support: float        # share of days on which it happened
    n: int

    def label(self) -> str:
        f = lambda m: f"{(m // 60) % 24:02d}:{m % 60:02d}"  # noqa: E731
        return f"{f(self.start_min)}-{f(self.end_min)}"


def _clusters(mins: np.ndarray) -> list[np.ndarray]:
    """Cluster minutes-of-day on a circle; returns unwrapped arrays (values may exceed 1440)."""
    if len(mins) == 0:
        return []
    v = np.sort(mins)
    gaps = np.diff(np.append(v, v[0] + 1440))
    cut = int(np.argmax(gaps))                  # cut the circle at the largest gap
    v = np.concatenate([v[cut + 1:], v[:cut + 1] + 1440]) if cut + 1 < len(v) else v
    if v.min() >= 1440:
        v = v - 1440
    groups, cur = [], [v[0]]
    for a, b in zip(v[:-1], v[1:]):
        if b - a > GAP_MIN:
            groups.append(np.array(cur))
            cur = []
        cur.append(b)
    groups.append(np.array(cur))
    return groups


def learn_routines(labels: pd.DataFrame, users: list[str]) -> dict[str, list[Window]]:
    out = {}
    for user in users:
        lab = labels[labels["user"] == user]
        local = lab["start"].dt.tz_convert(TIMEZONE)
        n_days = local.dt.date.nunique()
        wins = []
        for act in TRACKED:
            sel = lab["activity"] == act
            times = local[sel]
            mins = (times.dt.hour * 60 + times.dt.minute).to_numpy()
            for g in _clusters(mins):
                # A window may cross midnight: count days from the window's own start.
                in_g = np.isin(mins, g % 1440)
                days = {(t - pd.Timedelta(minutes=int(g.min()))).date() for t in times[in_g]}
                support = len(days) / max(n_days, 1)
                if support < MIN_SUPPORT:
                    continue
                lo, hi = np.percentile(g, 10), np.percentile(g, 90)
                start, end = int(lo) - WIDEN_MIN, int(np.ceil(hi)) + WIDEN_MIN
                wins.append(Window(act, start, end, end + GRACE_MIN, round(support, 2), int(len(g))))
        out[user] = sorted(wins, key=lambda w: w.start_min)
    return out


def routines_to_json(r: dict[str, list[Window]]) -> dict:
    return {u: [asdict(w) for w in ws] for u, ws in r.items()}


def routines_from_json(d: dict) -> dict[str, list[Window]]:
    return {u: [Window(**w) for w in ws] for u, ws in d.items()}


@dataclass
class Instance:
    user: str
    window: Window
    date: str                       # local date the window belongs to
    start: float                    # epoch seconds
    end: float
    deadline: float
    status: str = "pending"         # pending | done | missed | away
    done_at: float | None = None
    eat_minutes: int = 0
    away_minutes: int = 0
    seen_minutes: int = 0
    alerted: bool = False

    def public(self) -> dict:
        return {"activity": self.window.activity, "window": self.window.label(), "date": self.date,
                "deadline": self.deadline, "status": self.status, "done_at": self.done_at}


class RoutineChecker:
    """Feed it the model's per-minute output and drawer openings; it reports alerts."""

    def __init__(self, routines: dict[str, list[Window]], started_at: float, meds_from_model: bool = True):
        self.routines = routines
        self.meds_from_model = meds_from_model
        self.started_at = started_at          # windows that began before this are not judged
        self.instances: dict[tuple, Instance] = {}

    def _ensure(self, now: float) -> None:
        today = pd.Timestamp(now, unit="s", tz="UTC").tz_convert(TIMEZONE).normalize()
        for d in (today - pd.Timedelta(days=1), today):
            for user, wins in self.routines.items():
                for w in wins:
                    key = (user, w.activity, w.start_min, d.strftime("%Y-%m-%d"))
                    if key in self.instances:
                        continue
                    start = (d + pd.Timedelta(minutes=w.start_min)).timestamp()
                    if start - EARLY_MIN * 60 < self.started_at:
                        continue
                    self.instances[key] = Instance(
                        user, w, key[3], start, (d + pd.Timedelta(minutes=w.end_min)).timestamp(),
                        (d + pd.Timedelta(minutes=w.deadline_min)).timestamp())

    def _open(self, user: str, activity: str, ts: float):
        for inst in self.instances.values():
            if (inst.user == user and inst.window.activity == activity and inst.status == "pending"
                    and inst.start - EARLY_MIN * 60 <= ts <= inst.deadline):
                yield inst

    def observe_minute(self, user: str, ts: float, activity: str, away: bool) -> None:
        """One minute of model output for one resident."""
        self._ensure(ts)
        for inst in self.instances.values():
            if inst.user == user and inst.status == "pending" and inst.start <= ts <= inst.end:
                inst.seen_minutes += 1
                inst.away_minutes += int(away)
        if activity == "TAKE MEDS" and self.meds_from_model:
            for inst in self._open(user, "TAKE MEDS", ts):
                inst.status, inst.done_at = "done", ts
        if activity == "EAT":
            for inst in self._open(user, "EAT", ts):
                inst.eat_minutes += 1
                if inst.eat_minutes >= EAT_MIN_MINUTES:
                    inst.status, inst.done_at = "done", ts

    def observe_drawer(self, users_present: list[str], ts: float) -> None:
        """The medicine drawer opened; users_present = residents in its room (or home)."""
        self._ensure(ts)
        for user in users_present:
            for inst in self._open(user, "TAKE MEDS", ts):
                inst.status, inst.done_at = "done", ts

    def tick(self, now: float) -> list[dict]:
        """Close windows whose deadline passed; returns new alerts."""
        self._ensure(now)
        alerts = []
        for inst in self.instances.values():
            if inst.status != "pending" or now <= inst.deadline:
                continue
            if inst.seen_minutes and inst.away_minutes / inst.seen_minutes >= AWAY_SHARE:
                inst.status = "away"
                continue
            inst.status = "missed"
            if not inst.alerted:
                inst.alerted = True
                what = "medication" if inst.window.activity == "TAKE MEDS" else "meal"
                alerts.append({
                    "ts": pd.Timestamp(inst.deadline, unit="s", tz="UTC").isoformat(),
                    "user": inst.user,
                    "kind": "missed_meds" if inst.window.activity == "TAKE MEDS" else "missed_meal",
                    "expected_at": inst.window.label(),
                    "date": inst.date,
                    "message": f"{inst.user}: no {what} detected in the usual window "
                               f"{inst.window.label()} on {inst.date}. Please check in.",
                })
        return alerts

    def status(self, user: str) -> list[dict]:
        rows = [i for i in self.instances.values() if i.user == user]
        rows.sort(key=lambda i: i.start)
        return [i.public() for i in rows[-8:]]

