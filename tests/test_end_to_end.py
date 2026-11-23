"""The whole message flow in one process, without a broker: the replayer publishes a
recorded morning, the inference service estimates activities and checks routines, the
shadow keeps the state and history. Needs the cleaned data and the trained model."""
import time

import pandas as pd
import pytest

from shadowhome.schema import EVENTS_FILE, ROOT
from tests.conftest import LoopbackClient

pytestmark = pytest.mark.skipif(not (EVENTS_FILE.exists() and (ROOT / "models" / "activity_lgbm.txt").exists()),
                                reason="needs data/clean (pipeline) and models/ (training)")

DAY = 53                                   # 2022-06-14: user2 skipped breakfast (labels agree)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from services.dashboard.charts import minute_agreement
    from services.inference.main import InferenceService
    from services.replay.main import Replayer
    from services.shadow import app as shadow_app

    shadow_app.DB_PATH = str(tmp_path_factory.mktemp("e2e") / "shadow.db")
    shadow = shadow_app.ShadowService()
    inference = InferenceService()
    bus = LoopbackClient([shadow.on_message, inference.on_message])
    shadow.client = inference.client = bus

    ev = pd.read_parquet(EVENTS_FILE)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    ev = ev[ev["day"].isin([DAY])].sort_values("ts", kind="stable").reset_index(drop=True)
    replay = Replayer(ev, bus, speed=600, days=(DAY, DAY), loop=False)
    t0 = time.time()
    end = pd.Timestamp("2022-06-14T10:41:00Z").value              # 12:41 in Madrid
    while replay.sim_ns < end:
        replay.advance_to(replay.sim_ns + 120 * 10**9)            # two simulated minutes per tick
        replay.publish_clock()
    elapsed = time.time() - t0
    tl = shadow.timeline("user2", "2022-06-14")
    return {"bus": bus, "shadow": shadow, "timeline": tl, "agreement": minute_agreement(tl), "elapsed": elapsed}


def test_every_minute_gets_an_estimate(run):
    acts = [p for t, p, _ in run["bus"].sent if t.endswith("/activity")]
    minutes = {(p["user"], p["ts"]) for p in acts}
    assert len(minutes) >= 2 * 8 * 60                            # both residents, 02:00-12:41 local
    assert all(0 <= p["confidence"] <= 1 and p["model_version"] for p in acts)


def test_estimates_mostly_match_the_labels(run):
    agree, n = run["agreement"]
    assert n > 500 and agree > 0.7, (agree, n)


def test_missed_breakfast_raises_an_alert(run):
    alerts = [p for t, p, _ in run["bus"].sent if t == "home/alerts"]
    assert any(a["user"] == "user2" and a["kind"] == "missed_meal" and a["expected_at"] == "07:53-12:08"
               for a in alerts), alerts
    assert [a["kind"] for a in run["timeline"]["alerts"]] == ["missed_meal"]


def test_shadow_state_follows_the_replay(run):
    snap = run["shadow"].shadow.snapshot()
    assert snap["clock"]["sim_ts"].startswith("2022-06-14T10:4")
    assert all(r["activity"] for r in snap["residents"].values())
    assert sum(s["last_ts"] is not None for s in snap["sensors"].values()) >= 20
    routine = snap["routines"]["user2"]["items"]
    assert any(i["status"] == "missed" and i["window"] == "07:53-12:08" for i in routine)
