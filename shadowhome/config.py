"""Load config/home.yaml (rooms, floor-plan boxes, sensor -> room map)."""
from __future__ import annotations

from functools import lru_cache

import yaml

from shadowhome.schema import HOME_CONFIG


@lru_cache(maxsize=1)
def load_home() -> dict:
    with open(HOME_CONFIG) as f:
        return yaml.safe_load(f)


def sensor_rooms() -> dict[str, str]:
    return {sid: s.get("room", "unknown") for sid, s in load_home()["sensors"].items()}
