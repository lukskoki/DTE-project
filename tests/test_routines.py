"""Routine learning and the missed-meal / missed-medication checker, on synthetic data."""
import numpy as np
import pandas as pd

from shadowhome.routines import EAT_MIN_MINUTES, RoutineChecker, Window, _clusters, learn_routines
from shadowhome.schema import TIMEZONE


def local(s: str) -> float:
    return pd.Timestamp(s, tz=TIMEZONE).timestamp()


def labels_every_day(activity: str, hhmm: list[str], days: int = 10, minutes: int = 20,
                     first: int = 1) -> pd.DataFrame:
    rows = []
    for d in range(first, first + days):
        for t in hhmm:
            start = pd.Timestamp(f"2022-05-{d:02d} {t}", tz=TIMEZONE).tz_convert("UTC")
            rows.append({"user": "user1", "activity": activity, "start": start,
                         "end": start + pd.Timedelta(minutes=minutes)})
    return pd.DataFrame(rows)


def test_clusters_join_across_midnight():
    groups = _clusters(np.array([23 * 60 + 50, 10, 20]))
    assert len(groups) == 1
    assert sorted(groups[0] % 1440) == [10, 20, 23 * 60 + 50]


def test_learn_routines_finds_one_window_per_meal():
    lab = pd.concat([labels_every_day("EAT", ["13:00"]), labels_every_day("EAT", ["20:30"], days=2)])
    r = learn_routines(lab, ["user1"])["user1"]
    assert len(r) == 1                      # dinner on 2 of 10 days is below the 30% support
    w = r[0]
    assert w.activity == "EAT" and w.support == 1.0
    assert w.start_min <= 13 * 60 <= w.end_min
    assert w.deadline_min == w.end_min + 30


def test_medication_window_crossing_midnight():
    lab = labels_every_day("TAKE MEDS", ["23:50"], days=5)                  # 1-5 May, 23:50
    lab = pd.concat([lab, labels_every_day("TAKE MEDS", ["00:10"], days=5, first=2)])  # the nights after
    r = learn_routines(lab, ["user1"])["user1"]
    assert len(r) == 1
    assert r[0].start_min < 1440 < r[0].end_min      # 23:35-00:25 is one window
    assert r[0].label().startswith("23:")


LUNCH = Window("EAT", 12 * 60, 13 * 60, 13 * 60 + 30, 1.0, 10)
MEDS = Window("TAKE MEDS", 22 * 60, 23 * 60, 23 * 60 + 30, 1.0, 10)


def run_day(chk: RoutineChecker, user: str, activity_at: dict[str, str], away: bool = False,
            drawer_at: str | None = None, day: str = "2022-06-14") -> list[dict]:
    alerts = []
    for minute in pd.date_range(f"{day} 10:00", f"{day} 23:59", freq="1min", tz=TIMEZONE):
        ts = minute.timestamp()
        chk.observe_minute(user, ts, activity_at.get(minute.strftime("%H:%M"), "OTHER"), away=away)
        if drawer_at and minute.strftime("%H:%M") == drawer_at:
            chk.observe_drawer([user], ts)
        alerts += chk.tick(ts)
    return alerts


def test_meal_done_after_three_minutes_of_eat():
    chk = RoutineChecker({"user1": [LUNCH]}, started_at=local("2022-06-14 00:00"))
    alerts = run_day(chk, "user1", {f"12:{m:02d}": "EAT" for m in range(10, 10 + EAT_MIN_MINUTES)})
    inst = next(iter(chk.instances.values()))
    assert inst.status == "done" and not alerts


def test_meal_missed_raises_one_alert():
    chk = RoutineChecker({"user1": [LUNCH]}, started_at=local("2022-06-14 00:00"))
    alerts = run_day(chk, "user1", {"12:10": "EAT", "12:11": "EAT"})       # 2 minutes is not a meal
    assert [a["kind"] for a in alerts] == ["missed_meal"]
    assert alerts[0]["expected_at"] == "12:00-13:00"
    assert pd.Timestamp(alerts[0]["ts"]).timestamp() == local("2022-06-14 13:30")


def test_no_alert_when_away():
    chk = RoutineChecker({"user1": [LUNCH]}, started_at=local("2022-06-14 00:00"))
    alerts = run_day(chk, "user1", {}, away=True)
    assert not alerts
    assert next(iter(chk.instances.values())).status == "away"


def test_medication_from_drawer_or_model():
    for kwargs, expect in (({"drawer_at": "22:20"}, "done"), ({}, "missed")):
        chk = RoutineChecker({"user1": [MEDS]}, started_at=local("2022-06-14 00:00"), meds_from_model=False)
        alerts = run_day(chk, "user1", {}, **kwargs)
        inst = [i for i in chk.instances.values() if i.date == "2022-06-14"][0]
        assert inst.status == expect
        assert len(alerts) == (expect == "missed")
    chk = RoutineChecker({"user1": [MEDS]}, started_at=local("2022-06-14 00:00"), meds_from_model=True)
    run_day(chk, "user1", {"22:05": "TAKE MEDS"})
    assert [i.status for i in chk.instances.values() if i.date == "2022-06-14"] == ["done"]


def test_windows_before_start_are_not_judged():
    chk = RoutineChecker({"user1": [LUNCH]}, started_at=local("2022-06-14 11:45"))
    assert not run_day(chk, "user1", {})
    assert all(i.date != "2022-06-14" for i in chk.instances.values())
