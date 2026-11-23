"""Shared test helpers. The shadow service opens its SQLite file at import, so point it at a temp file."""
import json
import os
import tempfile
from collections import deque

os.environ.setdefault("SHADOW_DB", os.path.join(tempfile.mkdtemp(prefix="shadow-test-"), "shadow.db"))


class Msg:
    """Stand-in for a paho MQTT message."""

    def __init__(self, topic: str, payload: dict):
        self.topic = topic
        self.payload = json.dumps(payload).encode()


class RecordingClient:
    """Stand-in for a paho client: records what was published."""

    def __init__(self):
        self.sent: list[tuple[str, dict, bool]] = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.sent.append((topic, json.loads(payload), retain))


class LoopbackClient(RecordingClient):
    """An in-process broker: delivers every publish to the services (all subscribe to home/#),
    in order, after the current delivery returns, as a real broker would (a service that
    publishes while handling a message must not be called back re-entrantly)."""

    def __init__(self, handlers):
        super().__init__()
        self.handlers = handlers
        self.queue: deque = deque()
        self.delivering = False

    def publish(self, topic, payload, qos=0, retain=False):
        super().publish(topic, payload, qos, retain)
        self.queue.append(Msg(topic, json.loads(payload)))
        if self.delivering:
            return
        self.delivering = True
        try:
            while self.queue:
                msg = self.queue.popleft()
                for h in self.handlers:
                    h(self, None, msg)
        finally:
            self.delivering = False
