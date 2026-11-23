"""Rolling buffer of recent events for online feature building."""
from __future__ import annotations

from collections import deque

import pandas as pd

COLUMNS = ["ts", "sensor_id", "sensor_type", "room", "value", "state"]


class EventBuffer:
    """Events of the last `keep_minutes`, plus, for every (sensor, type), the
    newest event older than that, so the feature builder knows the state
    (door open, power draw, last room) at the start of its window."""

    def __init__(self, keep_minutes: int):
        self.keep = pd.Timedelta(minutes=keep_minutes)
        self.rows: deque = deque()
        self.prior: dict = {}

    def add(self, ts: pd.Timestamp, sensor_id: str, sensor_type: str, room, value, state) -> None:
        self.rows.append((ts, sensor_id, sensor_type, room, value, state))

    def trim(self, now: pd.Timestamp) -> None:
        cutoff = now - self.keep
        while self.rows and self.rows[0][0] < cutoff:
            row = self.rows.popleft()
            self.prior[(row[1], row[2])] = row

    def clear(self) -> None:
        self.rows.clear()
        self.prior.clear()

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self.prior.values()) + list(self.rows), columns=COLUMNS)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.sort_values("ts", kind="stable").reset_index(drop=True)
