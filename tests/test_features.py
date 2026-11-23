"""Online features (small rolling buffer) must equal offline features (whole dataset)."""
import numpy as np
import pandas as pd
import pytest

from shadowhome.config import sensor_rooms
from shadowhome.features import LOOKBACK_MIN, build_features, feature_columns
from shadowhome.schema import EVENTS_FILE
from services.inference.buffer import EventBuffer

pytestmark = pytest.mark.skipif(not EVENTS_FILE.exists(), reason="run the data pipeline first")


@pytest.fixture(scope="module")
def events():
    ev = pd.read_parquet(EVENTS_FILE)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    return ev


@pytest.mark.parametrize("when", ["2022-06-10 06:31", "2022-06-12 21:07", "2022-06-15 13:45"])
def test_online_equals_offline(events, when):
    rooms = sensor_rooms()
    m = pd.Timestamp(when, tz="UTC")
    day = events[(events["ts"] > m - pd.Timedelta(days=1)) & (events["ts"] <= m + pd.Timedelta(hours=1))]
    offline = build_features(day, pd.date_range(m - pd.Timedelta(hours=2), m, freq="1min"), rooms)
    offline = offline[offline["minute"] == m].reset_index(drop=True)

    buf = EventBuffer(keep_minutes=LOOKBACK_MIN + 5)
    for row in events[events["ts"] <= m].tail(400_000).itertuples(index=False):
        buf.add(row.ts, str(row.sensor_id), str(row.sensor_type), row.room, row.value, row.state)
    online = build_features(buf.frame(), pd.DatetimeIndex([m]), rooms)

    cols = feature_columns(offline)
    a, b = offline[cols].to_numpy(float), online[cols].to_numpy(float)
    bad = [c for c, x, y in zip(cols, a.T, b.T) if not np.allclose(x, y, equal_nan=True)]
    assert not bad, f"features differ: {bad}"
