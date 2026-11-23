"""Plotly figures for the dashboard. Pure functions of the shadow API's JSON, so
they can be built (and tested) without Streamlit.

Colour rules: each resident keeps one colour everywhere (user1 blue, user2
orange); ground truth and reference values are neutral grey; red is kept for
alerts only. Text never takes a series colour.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shadowhome.schema import TIMEZONE

# --- palette (validated: user1/user2 pass the CVD and contrast checks on SURFACE) ---
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OCCUPIED = "#f0efec"
TRUTH = "#cbcac3"             # labelled (ground-truth) bands
CRITICAL = "#d03b3b"          # alerts only
GOOD = "#0ca30c"
USER_COLORS = {"user1": "#2a78d6", "user2": "#eb6834"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK2, size=12)

# Activities in a daily order (night, morning care, meals, work, leisure, chores, away)
ACTIVITY_ORDER = [
    "SLEEP", "BATHROOM ACTIVITY", "SHOWER", "DRESS", "TAKE MEDS", "MAKE SIMPLE FOOD", "COOK", "EAT",
    "DISHWASHING", "WORK", "READ", "WATCH TV", "RELAX", "MEDITATE", "CHORES", "LAUNDRY", "PET",
    "OUT HOME", "OTHER",
]
ROOM_ORDER = ["bedroom", "bathroom", "hallway", "kitchen", "living_room", "study", "away"]
SYMBOLS = {"contact": "square", "movement": "circle", "vibration": "diamond", "power": "star",
           "temphum": "triangle-up", "illuminance": "hexagon"}
TYPE_NAMES = {"contact": "door / drawer", "movement": "motion", "vibration": "vibration", "power": "power",
              "temphum": "temp / humidity", "illuminance": "light"}
AWAY_BOX = [0, 63, 100, 11]     # strip under the house


def nice(act: str | None) -> str:
    """Activity name for people: sentence case, TV kept upper case."""
    if not act:
        return "–"
    return act.capitalize().replace(" tv", " TV")


def pretty_room(name: str) -> str:
    return name.replace("_", " ").capitalize()


def user_color(user: str) -> str:
    return USER_COLORS.get(user, INK2)


def local(ts: float | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    return pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(TIMEZONE)


def hhmm(ts: float | None) -> str:
    t = local(ts)
    return t.strftime("%H:%M") if t is not None else "-"


def _base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    kw.setdefault("margin", dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=FONT,
                      hoverlabel=dict(bgcolor="white", font_size=12), **kw)
    return fig


# --- floor plan ---------------------------------------------------------------
def sensor_positions(rooms_cfg: dict, sensors_cfg: dict) -> dict[str, tuple[float, float]]:
    """Place each room's sensors on a grid along the bottom of the room."""
    pos = {}
    for name, r in rooms_cfg.items():
        x, y, w, h = r["box"]
        ids = sorted([s for s, c in sensors_cfg.items() if c.get("room") == name],
                     key=lambda s: (s[0], int("".join(ch for ch in s if ch.isdigit()) or 0)))
        cols = max(1, int((w - 4) // 5.5) + 1)
        for i, sid in enumerate(ids):
            row, col = divmod(i, cols)
            pos[sid] = (x + 3 + col * 5.5, y + h - 4 - row * 6.5)
    return pos


def _sensor_text(s: dict, cfg: dict) -> str:
    t = s.get("type")
    if t == "contact":
        state = "open" if s.get("value") == 0 else "closed" if s.get("value") == 1 else "no data"
    elif t == "movement":
        state = "motion" if s.get("active") else "clear"
    elif t == "vibration":
        state = "vibrating" if s.get("active") else "still"
    elif t == "power":
        state = f"{s.get('value') or 0:.0f} W"
    elif t == "temphum":
        r = s.get("readings") or {}
        parts = []
        if "temperature" in r:
            parts.append(f"{r['temperature']:.1f} °C")
        if "humidity" in r:
            parts.append(f"{r['humidity']:.0f} % RH")
        state = " · ".join(parts) or "no data"
    elif t == "illuminance":
        state = f"{s.get('value') or 0:.0f} lx" if s.get("value") is not None else "no data"
    else:
        state = str(s.get("value"))
    obj = cfg.get("object", s.get("object", ""))
    return (f"<b>{s['id']}</b> · {obj}<br>{TYPE_NAMES.get(t, t)}: {state}"
            f"<br>room: {pretty_room(cfg.get('room', s.get('room', '?')))}"
            f"<br>last message {hhmm(s.get('last_ts'))}")


def floor_plan(state: dict, cfg: dict, height: int = 440) -> go.Figure:
    fig = go.Figure()
    rooms_cfg, sensors_cfg = cfg["rooms"], cfg["sensors"]
    rooms_state = state.get("rooms", {})
    residents = state.get("residents", {})

    for name, r in rooms_cfg.items():
        x, y, w, h = r["box"]
        occupied = bool(rooms_state.get(name, {}).get("occupants"))
        fig.add_shape(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, layer="below",
                      line=dict(color=AXIS, width=1.5), fillcolor=OCCUPIED if occupied else SURFACE)
        fig.add_annotation(x=x + 1.4, y=y + 1.6, text=pretty_room(name), showarrow=False, xanchor="left",
                           yanchor="top", font=dict(size=13, color=INK if occupied else INK2))
    ax, ay, aw, ah = AWAY_BOX
    fig.add_shape(type="rect", x0=ax, y0=ay, x1=ax + aw, y1=ay + ah, layer="below",
                  line=dict(color=GRID, width=1.5), fillcolor=PAGE)
    fig.add_annotation(x=ax + 1.4, y=ay + ah / 2, text="Away", showarrow=False, xanchor="left",
                       yanchor="middle", font=dict(size=13, color=INK2))

    # sensors: one trace per type (legend = shape), filled ink when active now
    pos = sensor_positions(rooms_cfg, sensors_cfg)
    sensors = state.get("sensors", {})
    for typ, symbol in SYMBOLS.items():
        ids = [sid for sid, c in sensors_cfg.items() if c.get("type") == typ and sid in pos]
        if not ids:
            continue
        live = [sensors.get(sid, {"id": sid, "type": typ}) for sid in ids]
        active = [bool(s.get("active")) for s in live]
        fig.add_trace(go.Scatter(
            x=[pos[s][0] for s in ids], y=[pos[s][1] for s in ids], mode="markers+text",
            text=ids, textposition="bottom center", textfont=dict(size=9, color=MUTED),
            marker=dict(symbol=[symbol if a else f"{symbol}-open" for a in active], size=12,
                        color=[INK if a else MUTED for a in active], line=dict(width=1.5)),
            hovertext=[_sensor_text({**s, "id": sid}, sensors_cfg[sid]) for sid, s in zip(ids, live)],
            hoverinfo="text", name=TYPE_NAMES[typ], showlegend=False,
        ))

    # residents: a dot per resident in their room, stacked, direct-labelled with name and estimate
    placed: dict[str, int] = {}
    for user, r in residents.items():
        room = r.get("room") or "unknown"
        if room not in rooms_cfg:
            room = "away"
        k = placed.get(room, 0)
        placed[room] = k + 1
        if room == "away":
            px, py = AWAY_BOX[0] + 14 + 30 * k, AWAY_BOX[1] + AWAY_BOX[3] / 2
        else:
            x, y, _, _ = rooms_cfg[room]["box"]
            px, py = x + 4, y + 9 + 7.5 * k
        act = r.get("activity") or "?"
        conf = r.get("confidence")
        label = f"<b>{user}</b> {nice(act)}" + (f" {conf:.0%}" if conf is not None else "")
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers+text", text=[label], textposition="middle right",
            textfont=dict(size=11, color=INK),
            marker=dict(size=20, color=user_color(user), line=dict(color="white", width=2)),
            hovertext=[f"<b>{user}</b> in {pretty_room(room)} since {hhmm(r.get('room_since'))}"
                       f"<br>model: {act} ({conf or 0:.0%})<br>label: {r.get('true_activity') or '-'}"
                       f"<br>heart rate: {r.get('bpm') or '-'} bpm"],
            hoverinfo="text", showlegend=False,
        ))

    fig.update_xaxes(range=[-1, 101], visible=False, fixedrange=True)
    fig.update_yaxes(range=[75, -1], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    return _base_layout(fig, height, showlegend=False)


# --- day timeline -------------------------------------------------------------
def _bars(segs: list[dict], width: float, color: str, name: str, show: bool = True, extra: str = "") -> go.Bar:
    start = [local(s["start"]).tz_localize(None) for s in segs]
    dur = [(s["end"] - s["start"]) * 1000 for s in segs]
    custom = [[hhmm(s["start"]), hhmm(s["end"]), round((s["end"] - s["start"]) / 60)] for s in segs]
    return go.Bar(base=start, x=dur, y=[s["value"] for s in segs], orientation="h", width=width,
                  marker=dict(color=color, line=dict(width=0)), name=name, showlegend=show, customdata=custom,
                  hovertemplate=f"<b>%{{y}}</b><br>%{{customdata[0]}}–%{{customdata[1]}} "
                                f"(%{{customdata[2]}} min){extra}<extra>{name}</extra>")


def day_timeline(tl: dict, height: int = 620) -> go.Figure:
    user = tl["user"]
    tl = {**tl, "actual": [{**s, "value": nice(s["value"])} for s in tl["actual"]],
          "predicted": [{**s, "value": nice(s["value"])} for s in tl["predicted"]]}
    order = [nice(a) for a in ACTIVITY_ORDER]
    acts = {s["value"] for s in tl["actual"]} | {s["value"] for s in tl["predicted"]}
    rows = [a for a in order if a in acts] + sorted(acts - set(order))
    rooms = [r for r in ROOM_ORDER if r in {s["value"] for s in tl["location"]}]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[max(len(rows), 1), max(len(rooms), 1) * 0.8])
    if tl["actual"]:
        fig.add_trace(_bars(tl["actual"], 0.78, TRUTH, "Label (ground truth)"), row=1, col=1)
    if tl["predicted"]:
        fig.add_trace(_bars(tl["predicted"], 0.34, user_color(user), "Model estimate"), row=1, col=1)
    if tl["location"]:
        loc = [{**s, "value": s["value"]} for s in tl["location"]]
        fig.add_trace(_bars(loc, 0.55, MUTED, "Room (BLE)"), row=2, col=1)
    t0 = local(tl["start"]).tz_localize(None)
    t1 = local(tl["end"]).tz_localize(None)
    for a in tl.get("alerts", []):
        x = local(a["ts"]).tz_localize(None)
        fig.add_vline(x=x, line=dict(color=CRITICAL, width=1.5))
        fig.add_annotation(x=x, y=1.0, yref="paper", text=f"⚠ {a['kind'].replace('_', ' ')}", showarrow=False,
                           yanchor="bottom", font=dict(size=11, color=CRITICAL), bgcolor=SURFACE)
    fig.update_yaxes(categoryorder="array", categoryarray=rows[::-1], row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=rooms[::-1], row=2, col=1,
                     ticktext=[pretty_room(r) for r in rooms[::-1]], tickvals=rooms[::-1])
    fig.update_xaxes(type="date", range=[t0, t1], dtick=2 * 3600 * 1000, tickformat="%H:%M",
                     gridcolor=GRID, linecolor=AXIS, showline=True)
    fig.update_yaxes(gridcolor=SURFACE, linecolor=AXIS, tickfont=dict(size=11, color=INK2))
    fig.update_layout(barmode="overlay", bargap=0.1, barcornerradius=2)
    return _base_layout(fig, height, legend=dict(orientation="h", y=1.06, x=0, yanchor="bottom"),
                        margin=dict(l=8, r=8, t=40, b=8))


def minute_agreement(tl: dict) -> tuple[float | None, int]:
    """Share of estimated minutes whose estimate equals the label at that minute."""
    if not tl["predicted"] or not tl["actual"]:
        return None, 0
    t0 = tl["start"]
    n = int((tl["end"] - t0) // 60)

    def per_minute(segs):
        out = np.full(n, None, dtype=object)
        for s in segs:
            a, b = int((s["start"] - t0) // 60), int(math.ceil((s["end"] - t0) / 60))
            out[max(a, 0):min(b, n)] = s["value"]
        return out

    p, a = per_minute(tl["predicted"]), per_minute(tl["actual"])
    both = (p != None) & (a != None)  # noqa: E711
    if not both.any():
        return None, 0
    return float((p[both] == a[both]).mean()), int(both.sum())


# --- hours per activity -------------------------------------------------------
def hours_bars(series: list[tuple[str, dict, str]], height: int | None = None) -> go.Figure:
    """Grouped horizontal bars of hours per activity; series = [(name, {activity: hours}, colour)]."""
    acts = set()
    for _, h, _ in series:
        acts |= {a for a, v in h.items() if v >= 0.05}
    rows = [a for a in ACTIVITY_ORDER if a in acts]
    fig = go.Figure()
    for name, h, color in series[::-1]:   # horizontal groups stack upwards: add the first series last
        fig.add_trace(go.Bar(y=[nice(a) for a in rows], x=[h.get(a, 0) for a in rows], orientation="h", name=name,
                             marker=dict(color=color, line=dict(width=0)),
                             hovertemplate="<b>%{y}</b><br>%{x:.1f} h<extra>" + name + "</extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=[nice(a) for a in rows[::-1]],
                     tickfont=dict(size=11, color=INK2))
    fig.update_xaxes(title_text="hours", gridcolor=GRID, zeroline=True, zerolinecolor=AXIS, rangemode="tozero")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, barcornerradius=4)
    return _base_layout(fig, height or max(260, 34 * len(rows) + 80),
                        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", traceorder="reversed"),
                        margin=dict(l=8, r=8, t=36, b=8))


# --- model quality ------------------------------------------------------------
def per_class_f1(per_class: pd.DataFrame, height: int = 520) -> go.Figure:
    d = per_class.sort_values("f1")
    fig = go.Figure(go.Bar(
        y=d["activity"].map(nice), x=d["f1"], orientation="h", marker=dict(color=USER_COLORS["user1"], line=dict(width=0)),
        customdata=np.stack([d["precision"], d["recall"], d["support_min"]], axis=1),
        hovertemplate="<b>%{y}</b><br>F1 %{x:.2f}<br>precision %{customdata[0]:.2f}, recall %{customdata[1]:.2f}"
                      "<br>%{customdata[2]:,} labelled minutes<extra></extra>",
        text=[f"{v:.2f}" for v in d["f1"]], textposition="outside", textfont=dict(size=10, color=INK2),
        cliponaxis=False))
    fig.update_xaxes(range=[0, 1.08], gridcolor=GRID, title_text="F1 on test days 48–62")
    fig.update_yaxes(tickfont=dict(size=11, color=INK2))
    fig.update_layout(barcornerradius=4, bargap=0.3)
    return _base_layout(fig, height, showlegend=False)


def confusion(cm: pd.DataFrame, height: int = 560) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm.values, x=[nice(c) for c in cm.columns], y=[nice(c) for c in cm.index], zmin=0, zmax=1,
        colorscale=[[0, "#f3f7fd"]] + [[(i + 1) / len(BLUES), c] for i, c in enumerate(BLUES)],
        xgap=2, ygap=2, colorbar=dict(title=dict(text="share of row", side="right"), thickness=10),
        hovertemplate="labelled <b>%{y}</b><br>estimated <b>%{x}</b><br>%{z:.0%} of the labelled minutes<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10), title_text="labelled")
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10), title_text="estimated")
    return _base_layout(fig, height)


FEATURE_PREFIX = {"n1": "activations, last 1 min", "n5": "activations, last 5 min",
                  "n15": "activations, last 15 min", "ago": "minutes since last activation", "open": "open now"}
QUANTITY = {"powe": "power", "temp": "temperature", "humi": "humidity", "illu": "light level"}


def describe_feature(name: str, sensors: dict) -> str:
    """Readable name of a model feature (see shadowhome/features.py for how each is built)."""
    fixed = {"hour_sin": "time of day (sin)", "hour_cos": "time of day (cos)", "weekend": "weekend",
             "is_user1": "which resident", "same_room": "both residents in one room",
             "since_fix": "minutes since last BLE fix", "in_room": "minutes in current room",
             "own_room5": "activations in own room, 5 min", "bpm": "heart rate", "steps10": "steps, last 10 min"}
    if name in fixed:
        return fixed[name]
    head, _, rest = name.partition("_")
    if head in ("me", "other"):
        who = "resident" if head == "me" else "other resident"
        return f"{who} {'away' if rest == 'away' else 'in ' + pretty_room(rest).lower()}"
    if head == "room5":
        return f"activations in {pretty_room(rest).lower()}, 5 min"
    if head == "d15":
        q, _, sid = rest.partition("_")
        return f"{QUANTITY.get(q, q)} change in 15 min, {pretty_room(sensors.get(sid, {}).get('room', '?')).lower()} ({sid})"
    if head in QUANTITY:
        s = sensors.get(rest, {})
        if head == "powe":
            return f"{s.get('object', rest)} power ({rest})"
        return f"{QUANTITY[head]}, {pretty_room(s.get('room', '?')).lower()} ({rest})"
    if head in FEATURE_PREFIX:
        obj = sensors.get(rest, {}).get("object", rest)
        return f"{obj}: {FEATURE_PREFIX[head]} ({rest})"
    return name


def importance(imp: pd.DataFrame, sensors: dict | None = None, top: int = 15, height: int = 440) -> go.Figure:
    d = imp.head(top).iloc[::-1]
    names = [describe_feature(f, sensors or {}) for f in d["feature"]]
    fig = go.Figure(go.Bar(y=names, x=d["share"], orientation="h", customdata=d["feature"],
                           marker=dict(color=USER_COLORS["user1"], line=dict(width=0)),
                           hovertemplate="<b>%{y}</b><br>feature %{customdata}<br>%{x:.1%} of total gain"
                                         "<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", gridcolor=GRID, title_text="share of the model's total gain")
    fig.update_yaxes(tickfont=dict(size=11, color=INK2))
    fig.update_layout(barcornerradius=4, bargap=0.3)
    return _base_layout(fig, height, showlegend=False)
