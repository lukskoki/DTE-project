"""Small paho-mqtt helpers shared by the services."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)


def broker() -> tuple[str, int]:
    return os.getenv("MQTT_HOST", "localhost"), int(os.getenv("MQTT_PORT", "1883"))


def connect(name: str, on_message=None, subscribe: list[str] | None = None) -> mqtt.Client:
    """Connect (retrying until the broker is up), subscribe, start the network loop."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"{name}-{uuid.uuid4().hex[:6]}")
    topics = list(subscribe or [])

    def _on_connect(c, userdata, flags, reason_code, properties):
        log.info("%s connected to MQTT (%s)", name, reason_code)
        for t in topics:  # (re)subscribe on every connect
            c.subscribe(t, qos=0)

    client.on_connect = _on_connect
    if on_message:
        client.on_message = on_message
    host, port = broker()
    while True:
        try:
            client.connect(host, port, keepalive=30)
            break
        except OSError as exc:
            log.warning("%s: broker %s:%s not reachable (%s), retrying", name, host, port, exc)
            time.sleep(2)
    client.loop_start()
    return client


def publish(client: mqtt.Client, topic: str, payload: dict, retain: bool = False) -> None:
    client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=0, retain=retain)


def decode(msg) -> dict | None:
    try:
        return json.loads(msg.payload)
    except (ValueError, UnicodeDecodeError):
        return None
