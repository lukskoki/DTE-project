"""Shared contracts: activities, rooms, sensor types, MQTT topics, file paths.

Every service imports from here so the replay, shadow, inference and dashboard
agree on names. Keep this file in sync with the "Sučelja" section of the plan.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
EVENTS_FILE = CLEAN_DIR / "events.parquet"
LABELS_FILE = CLEAN_DIR / "labels.parquet"
HOME_CONFIG = ROOT / "config" / "home.yaml"

USERS = ["user1", "user2"]

# Activity classes: the 18 from the authors' prepare_data.py, plus MEDITATE,
# which appears in a few labels.
ACTIVITIES = [
    "BATHROOM ACTIVITY", "CHORES", "COOK", "DISHWASHING", "DRESS", "EAT",
    "LAUNDRY", "MAKE SIMPLE FOOD", "MEDITATE", "OUT HOME", "PET", "READ",
    "RELAX", "SHOWER", "SLEEP", "TAKE MEDS", "WATCH TV", "WORK", "OTHER",
]

# Timestamps in the dataset are UTC; the home is in Spain. Show local time.
TIMEZONE = "Europe/Madrid"

# BLE beacon room names in the dataset (Spanish) -> names we use everywhere.
ROOM_NAMES = {
    "banho": "bathroom",
    "cocina": "kitchen",
    "dormitorio": "bedroom",
    "estudio": "study",
    "pasillo": "hallway",
    "salon1": "living_room",
    "not_home": "away",
}
ROOMS = [r for r in ROOM_NAMES.values() if r != "away"]

# Ambient sensor folders in each DAY_N directory.
AMBIENT_TYPES = {
    "Contact": "contact",
    "Movement": "movement",
    "Vibration": "vibration",
    "Power": "power",
    "TempHum": "temphum",
    "Illuminance": "illuminance",
}

# Clean event table columns (one row per event, sorted by ts).
EVENT_COLUMNS = ["ts", "day", "sensor_id", "sensor_type", "room", "value", "state"]


# --- MQTT topics -----------------------------------------------------------
def sensor_topic(sensor_type: str, sensor_id: str) -> str:
    return f"home/sensors/{sensor_type}/{sensor_id}"


def location_topic(user: str) -> str:
    return f"home/residents/{user}/location"


def wearable_topic(user: str, kind: str) -> str:
    return f"home/residents/{user}/{kind}"  # kind: bpm | steps


def activity_topic(user: str) -> str:
    return f"home/residents/{user}/activity"


CLOCK_TOPIC = "home/replay/clock"        # retained: {sim_ts, speed, day, paused}
CONTROL_TOPIC = "home/replay/control"    # {action: seek|speed|pause|resume, ...}
ALERTS_TOPIC = "home/alerts"
ALL_TOPICS = "home/#"
