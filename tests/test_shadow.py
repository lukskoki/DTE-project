"""The digital shadow's model of the home, and its HTTP API (without a broker)."""
import pandas as pd
from fastapi.testclient import TestClient

from shadowhome.schema import ROOMS
from services.shadow.model import GroundTruth, HomeShadow
from tests.conftest import Msg

CFG = {
    "rooms": {r: {"box": [0, 0, 10, 10]} for r in ROOMS},
    "residents": ["user1", "user2"],
    "sensors": {
        "c2": {"type": "contact", "room": "bathroom", "object": "medicine drawer"},
        "th1": {"type": "temphum", "room": "bathroom", "object": "temp/humidity"},
        "m1": {"type": "movement", "room": "bathroom", "object": "motion"},
        "p1": {"type": "power", "room": "living_room", "object": "television", "active_above_w": 60},
    },
}


def clock(shadow: HomeShadow, iso: str):
    return shadow.on_clock({"sim_ts": iso, "speed": 60, "day": 53, "paused": False})


def sensor(sid, typ, value, ts="2022-06-14T10:00:00Z", state=None):
    return {"ts": ts, "sensor_id": sid, "sensor_type": typ, "value": value, "state": state}


def test_sensors_keep_state_per_quantity():
    s = HomeShadow(CFG)
    clock(s, "2022-06-14T10:00:00Z")
    s.on_sensor(sensor("c2", "contact", 0, state="open"))
    s.on_sensor(sensor("th1", "temperature", 24.5))
    s.on_sensor(sensor("th1", "humidity", 61))
    s.on_sensor(sensor("p1", "power", 80))
    snap = s.snapshot()
    assert snap["sensors"]["c2"]["active"] and snap["sensors"]["p1"]["active"]
    assert snap["sensors"]["th1"]["readings"] == {"temperature": 24.5, "humidity": 61}
    assert snap["rooms"]["bathroom"]["active_sensors"] == ["c2"]
    s.on_sensor(sensor("c2", "contact", 1, state="closed"))
    assert not s.snapshot()["sensors"]["c2"]["active"]


def test_motion_expires_after_two_minutes():
    s = HomeShadow(CFG)
    clock(s, "2022-06-14T10:00:00Z")
    s.on_sensor(sensor("m1", "movement", 1, state="occupied"))
    clock(s, "2022-06-14T10:01:00Z")
    assert s.snapshot()["sensors"]["m1"]["active"]
    clock(s, "2022-06-14T10:03:00Z")
    assert not s.snapshot()["sensors"]["m1"]["active"]


def test_resident_is_away_after_ten_minutes_without_ble():
    s = HomeShadow(CFG)
    clock(s, "2022-06-14T10:00:00Z")
    assert s.on_location({"ts": "2022-06-14T10:00:00Z", "user": "user1", "room": "kitchen", "distance": 1.2})
    assert not s.on_location({"ts": "2022-06-14T10:00:30Z", "user": "user1", "room": "kitchen", "distance": 1.0})
    assert s.snapshot()["rooms"]["kitchen"]["occupants"] == ["user1"]
    assert clock(s, "2022-06-14T10:05:00Z") == (False, [])
    jumped, away = clock(s, "2022-06-14T10:11:00Z")
    assert away == [("user1", pd.Timestamp("2022-06-14T10:00:30Z").timestamp())]
    assert s.snapshot()["residents"]["user1"]["room"] == "away"


def test_seek_back_clears_the_shadow():
    s = HomeShadow(CFG)
    clock(s, "2022-06-14T10:00:00Z")
    s.on_alert({"ts": "2022-06-14T10:00:00Z", "user": "user2", "kind": "missed_meal"})
    jumped, _ = clock(s, "2022-06-13T08:00:00Z")
    assert jumped and s.snapshot()["alerts"] == []


def test_ground_truth_lookup():
    lab = pd.DataFrame({"user": ["user1", "user1"], "activity": ["COOK", "EAT"],
                        "start": pd.to_datetime(["2022-06-14T10:00Z", "2022-06-14T10:30Z"]),
                        "end": pd.to_datetime(["2022-06-14T10:30Z", "2022-06-14T11:00Z"])})
    gt = GroundTruth(lab)
    at = lambda iso: gt.at("user1", pd.Timestamp(iso).timestamp())  # noqa: E731
    assert at("2022-06-14T10:10Z") == "COOK" and at("2022-06-14T10:45Z") == "EAT"
    assert at("2022-06-14T11:30Z") is None and gt.at("user2", 0) is None


def test_segments_merge_repeats_and_cap_gaps():
    from services.shadow.app import ShadowService
    rows = [(0, "EAT"), (60, "EAT"), (120, "SLEEP"), (1000, "SLEEP")]
    segs = ShadowService.segments(rows, 1, 0, 2000, max_len=120)
    assert [(s["start"], s["end"], s["value"]) for s in segs] == [
        (0, 120, "EAT"), (120, 240, "SLEEP"), (1000, 1120, "SLEEP")]


def test_api_serves_state_and_timeline():
    from services.shadow import app as shadow_app
    svc = shadow_app.service
    client = TestClient(shadow_app.app)       # no `with`: the lifespan (MQTT connect) does not run
    feed = [
        ("home/replay/clock", {"sim_ts": "2022-06-14T10:00:00Z", "speed": 60, "day": 53, "paused": False}),
        ("home/residents/user1/location", {"ts": "2022-06-14T10:00:00Z", "user": "user1", "room": "kitchen",
                                           "distance": 1.0}),
        ("home/sensors/contact/c6", {"ts": "2022-06-14T10:00:10Z", "sensor_id": "c6", "sensor_type": "contact",
                                     "room": "kitchen", "value": 0, "state": "open"}),
        ("home/residents/user1/bpm", {"ts": "2022-06-14T10:00:20Z", "user": "user1", "value": 71}),
        ("home/residents/user1/activity", {"ts": "2022-06-14T10:01:00+00:00", "user": "user1", "activity": "COOK",
                                           "confidence": 0.8, "probs": {"COOK": 0.8}, "model_version": "t"}),
        ("home/residents/user1/activity", {"ts": "2022-06-14T10:02:00+00:00", "user": "user1", "activity": "COOK",
                                           "confidence": 0.7, "probs": {"COOK": 0.7}, "model_version": "t"}),
        ("home/alerts", {"ts": "2022-06-14T10:02:00+00:00", "user": "user1", "kind": "missed_meal",
                         "expected_at": "07:00-08:00", "date": "2022-06-14", "message": "test"}),
        ("home/sensors/contact/c6", {"ts": "bad"}),                       # malformed: logged and ignored
    ]
    for topic, payload in feed:
        svc.on_message(None, None, Msg(topic, payload))
    st = client.get("/state").json()
    assert st["residents"]["user1"]["room"] == "kitchen"
    assert st["residents"]["user1"]["activity"] == "COOK" and st["residents"]["user1"]["bpm"] == 71
    assert "c6" in st["rooms"]["kitchen"]["active_sensors"]
    tl = client.get("/timeline", params={"user": "user1", "date": "2022-06-14"}).json()
    assert [s["value"] for s in tl["predicted"]] == ["COOK"]
    assert tl["predicted"][0]["end"] - tl["predicted"][0]["start"] == 120
    assert [a["kind"] for a in tl["alerts"]] == ["missed_meal"]
    assert client.get("/alerts").json()[0]["user"] == "user1"
    assert client.post("/replay", json={"action": "pause"}).status_code == 503
    assert client.get("/health").json()["ok"]
