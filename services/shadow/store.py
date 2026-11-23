"""History of the shadow in SQLite: what happened, when (simulated time)."""
from __future__ import annotations

import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_log   (ts REAL, sensor_id TEXT, type TEXT, room TEXT, value REAL, state TEXT);
CREATE TABLE IF NOT EXISTS location_log (ts REAL, user TEXT, room TEXT);
CREATE TABLE IF NOT EXISTS vitals_log   (ts REAL, user TEXT, kind TEXT, value REAL);
CREATE TABLE IF NOT EXISTS activity_log (ts REAL, user TEXT, activity TEXT, confidence REAL, model_version TEXT);
CREATE TABLE IF NOT EXISTS alert_log    (ts REAL, user TEXT, kind TEXT, expected_at TEXT, message TEXT);
CREATE INDEX IF NOT EXISTS i_sensor   ON sensor_log(ts);
CREATE INDEX IF NOT EXISTS i_location ON location_log(user, ts);
CREATE INDEX IF NOT EXISTS i_vitals   ON vitals_log(user, ts);
CREATE INDEX IF NOT EXISTS i_activity ON activity_log(user, ts);
CREATE INDEX IF NOT EXISTS i_alert    ON alert_log(ts);
"""
TABLES = ["sensor_log", "location_log", "vitals_log", "activity_log", "alert_log"]


class Store:
    def __init__(self, path: str):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.execute("PRAGMA journal_mode=WAL")
        threading.Thread(target=self._committer, daemon=True).start()

    def _committer(self) -> None:
        while True:
            time.sleep(1)
            with self.lock:
                self.db.commit()

    def write(self, table: str, row: tuple) -> None:
        marks = ",".join("?" * len(row))
        with self.lock:
            self.db.execute(f"INSERT INTO {table} VALUES ({marks})", row)

    def truncate_after(self, ts: float) -> None:
        """A seek backwards: forget everything recorded after the new simulated time."""
        with self.lock:
            for t in TABLES:
                self.db.execute(f"DELETE FROM {t} WHERE ts > ?", (ts,))
            self.db.commit()

    def query(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()
