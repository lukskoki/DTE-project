"""The replayer publishes recorded events in simulated time, on the right topics."""
import pandas as pd

from services.replay.main import Replayer, parse_days
from tests.conftest import Msg, RecordingClient


def events() -> pd.DataFrame:
    return pd.DataFrame({
        "ts": pd.to_datetime(["2022-04-23T06:00:00Z", "2022-04-23T06:00:05Z", "2022-04-24T07:00:00Z"], utc=True),
        "day": [1, 1, 2],
        "sensor_id": ["m1", "user1", "c2"],
        "sensor_type": ["movement", "presence", "contact"],
        "room": [None, "bathroom", None],
        "value": [1.0, 0.8, 0.0],
        "state": ["occupied", "bathroom", "open"],
    })


def test_parse_days():
    assert parse_days("48-62") == (48, 62) and parse_days("55") == (55, 55)


def test_seek_publishes_a_retained_clock_at_day_start():
    c = RecordingClient()
    Replayer(events(), c, speed=60, days=(1, 2), loop=False)
    topic, payload, retain = c.sent[-1]
    assert topic == "home/replay/clock" and retain
    assert payload["sim_ts"] == "2022-04-23T00:00:00Z" and payload["day"] == 1


def test_events_go_out_in_time_order_on_their_topics():
    c = RecordingClient()
    r = Replayer(events(), c, speed=60, days=(1, 1), loop=False)
    c.sent.clear()
    assert not r.advance_to(pd.Timestamp("2022-04-23T06:00:00Z").value)
    assert [t for t, _, _ in c.sent] == ["home/sensors/movement/m1"]
    assert c.sent[0][1]["room"] == "bathroom"                     # filled in from config/home.yaml
    assert r.advance_to(pd.Timestamp("2022-04-24T23:00:00Z").value)   # day 2 is past the last day
    assert [t for t, _, _ in c.sent] == ["home/sensors/movement/m1", "home/residents/user1/location"]
    assert c.sent[1][1] == {"ts": "2022-04-23T06:00:05Z", "user": "user1", "room": "bathroom", "distance": 0.8}


def test_control_messages():
    c = RecordingClient()
    r = Replayer(events(), c, speed=60, days=(1, 2), loop=False)
    r.on_control(None, None, Msg("home/replay/control", {"action": "seek", "day": 2}))
    assert r.sim_ns == pd.Timestamp("2022-04-24T00:00:00Z").value and r.i == 2
    r.on_control(None, None, Msg("home/replay/control", {"action": "speed", "value": 3600}))
    r.on_control(None, None, Msg("home/replay/control", {"action": "pause"}))
    assert r.speed == 3600 and r.paused
    assert c.sent[-1][1]["paused"] and c.sent[-1][1]["speed"] == 3600
