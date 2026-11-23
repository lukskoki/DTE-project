# SDHAR-HOME digital shadow

Digital shadow of a two-resident smart home for home-care monitoring (DTE, UvA, group 14).
Sensor recordings from [SDHAR-HOME](https://github.com/raugom13/SDHAR-HOME-A-Sensor-Dataset-for-Human-Activity-Recognition-at-Home)
are replayed through an MQTT broker; a shadow service keeps the state of rooms, sensors and residents;
a classifier estimates each resident's activity; a routine checker flags missed meals and medication;
a dashboard shows the home, the activity timeline and alerts.

## Quick start

```bash
pip install -r requirements.txt
./scripts/download_data.sh              # ~900 MB, 62 day zips into data/raw/SDHAR-HOME
python -m pipeline.build_dataset        # -> data/clean/events.parquet, labels.parquet (~40 s)
python -m pipeline.infer_sensor_rooms   # -> config/home.yaml (sensor -> room map)
docker compose up --build
```

## Layout

| Path | What |
| --- | --- |
| `shadowhome/schema.py` | Shared names: activities, rooms, MQTT topics, file paths |
| `pipeline/build_dataset.py` | Raw zips -> one clean event table + activity intervals |
| `pipeline/infer_sensor_rooms.py` | Infers each sensor's room from residents' BLE positions |
| `config/home.yaml` | Rooms, floor-plan boxes, sensor -> room map (generated, hand-editable) |
| `services/` | replay, shadow, inference, dashboard (one folder each) |

## Data notes

* 62 days (2022-04-23 to 2022-06-23), 2 residents, 33 ambient sensors, BLE room per resident, wristband bpm/steps.
* Wristband `gyro.csv` is skipped: ~95% of the raw size and not needed for room-level activities.
* Labels are change points; they become intervals lasting until the next change (capped at 14 h).
* The dataset has no sensor -> room map. `infer_sensor_rooms` votes with the residents' BLE room at the moment a
  sensor fires, counting only unambiguous moments, and normalises by how much time people spend in each room.
  `vote_share` in `config/home.yaml` shows how confident each mapping is.

## MQTT topics

| Topic | Publisher | Payload |
| --- | --- | --- |
| `home/sensors/{type}/{id}` | replay | `{ts, sensor_id, sensor_type, room, value, state}` |
| `home/residents/{user}/location` | replay | `{ts, user, room, distance}` |
| `home/residents/{user}/bpm`, `/steps` | replay | `{ts, user, value}` |
| `home/replay/clock` | replay | `{sim_ts, speed, day}` |
| `home/residents/{user}/activity` | inference | `{ts, user, activity, confidence, probs, model_version}` |
| `home/alerts` | inference (routine checker) | `{ts, user, kind, expected_at, message}` |
