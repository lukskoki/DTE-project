"""The dashboard's figures build from the shadow API's JSON."""
import pandas as pd

from services.dashboard import charts as ch
from shadowhome.config import load_home


def ts(iso: str) -> float:
    return pd.Timestamp(iso).timestamp()


def test_floor_plan_places_residents_and_sensors():
    cfg = load_home()
    state = {
        "rooms": {"kitchen": {"occupants": ["user1"]}},
        "sensors": {"c6": {"id": "c6", "type": "contact", "value": 0, "active": True, "last_ts": ts("2022-06-14T20:00Z")}},
        "residents": {"user1": {"room": "kitchen", "activity": "COOK", "confidence": 0.8},
                      "user2": {"room": "away", "activity": "OUT HOME", "confidence": 0.99}},
    }
    fig = ch.floor_plan(state, cfg)
    people = [t for t in fig.data if t.marker.size == 20]
    assert len(people) == 2
    x, y, w, h = cfg["rooms"]["kitchen"]["box"]
    assert x <= people[0].x[0] <= x + w and y <= people[0].y[0] <= y + h
    assert people[1].y[0] > 60                                    # user2 in the "away" strip
    pos = ch.sensor_positions(cfg["rooms"], cfg["sensors"])
    assert set(pos) == set(cfg["sensors"])                        # every sensor gets a spot in its room


def test_timeline_and_agreement():
    t0 = ts("2022-06-13T22:00Z")                                  # 2022-06-14 00:00 in Madrid
    tl = {"user": "user2", "start": t0, "end": t0 + 86400, "alerts": [],
          "actual": [{"start": t0, "end": t0 + 3600, "value": "SLEEP"}],
          "predicted": [{"start": t0, "end": t0 + 1800, "value": "SLEEP"},
                        {"start": t0 + 1800, "end": t0 + 3600, "value": "OTHER"}],
          "location": [{"start": t0, "end": t0 + 3600, "value": "bedroom"}]}
    agree, n = ch.minute_agreement(tl)
    assert n == 60 and agree == 0.5
    fig = ch.day_timeline(tl)
    assert {t.name for t in fig.data} == {"Label (ground truth)", "Model estimate", "Room (BLE)"}


def test_hours_bars_and_names():
    fig = ch.hours_bars([("a", {"SLEEP": 7.0, "WATCH TV": 1.0}, "#2a78d6"), ("b", {"SLEEP": 6.0}, ch.TRUTH)])
    assert list(fig.data[-1].y) == ["Sleep", "Watch TV"] and fig.data[-1].name == "a"
    sensors = load_home()["sensors"]
    assert ch.describe_feature("powe_p1", sensors) == "television power (p1)"
    assert ch.describe_feature("me_away", sensors) == "resident away"
    assert ch.describe_feature("ago_c6", sensors).startswith("refrigerator door")
