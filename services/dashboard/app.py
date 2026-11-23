"""Dashboard of the SDHAR-HOME digital shadow.

    streamlit run services/dashboard/app.py        (SHADOW_URL, default http://localhost:8000)

Pages
  Live home      floor plan with residents and active sensors, resident cards
                 (estimated vs labelled activity, heart rate, today's routines),
                 room climate, alerts
  Day timeline   estimated vs labelled activity and room for one resident and day
  Compare days   hours per activity: a day against the resident's usual day or another day
  Model quality  offline test results of the classifier and the routine alerts

Live data comes only from the shadow service's API. The model page reads the
evaluation files in reports/ written by the training pipeline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `streamlit run`

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from services.dashboard import charts as ch  # noqa: E402
from shadowhome.schema import ROOT, TIMEZONE, USERS  # noqa: E402

SHADOW_URL = os.getenv("SHADOW_URL", "http://localhost:8000").rstrip("/")
REPORTS = ROOT / "reports"
SPEEDS = [60, 120, 300, 600, 1200, 1800, 3600]
PLOT_CONFIG = {"displayModeBar": False}

CSS = f"""
<style>
.block-container {{padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1500px;}}
.chip {{display:inline-block; width:12px; height:12px; border-radius:6px; margin-right:6px; vertical-align:-1px;}}
.card-head {{font-size:1.05rem; color:{ch.INK}; margin-bottom:.35rem;}}
.card-head .where {{color:{ch.INK2}; font-weight:400;}}
.kv {{display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:.6rem; margin:.25rem 0 .4rem;}}
.kv .k {{font-size:.75rem; color:{ch.INK2};}}
.kv .v {{font-size:1.05rem; color:{ch.INK}; font-weight:600; line-height:1.25; overflow-wrap:anywhere;}}
.kv .s {{font-size:.75rem; color:{ch.MUTED};}}
.match {{font-size:.8rem; color:{ch.INK2}; margin-bottom:.3rem;}}
.routine {{font-size:.85rem; color:{ch.INK}; padding:.12rem 0;}}
.routine .when {{color:{ch.INK2};}}
.tiles {{display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.5rem;}}
.tile {{background:{ch.SURFACE}; border:1px solid rgba(11,11,11,.10); border-radius:8px; padding:.45rem .6rem;}}
.tile .k {{font-size:.72rem; color:{ch.INK2};}}
.tile .v {{font-size:1.1rem; color:{ch.INK}; font-weight:600;}}
.alert {{border-left:3px solid {ch.CRITICAL}; background:{ch.SURFACE}; padding:.4rem .6rem; margin-bottom:.4rem;
         border-radius:0 6px 6px 0; font-size:.85rem; color:{ch.INK};}}
.alert .t {{color:{ch.CRITICAL}; font-weight:600;}}
.ok {{border-left:3px solid {ch.GOOD}; background:{ch.SURFACE}; padding:.4rem .6rem; border-radius:0 6px 6px 0;
      font-size:.85rem; color:{ch.INK};}}
.legend {{font-size:.8rem; color:{ch.INK2}; margin:-.4rem 0 .9rem;}}
.clock {{font-size:1.15rem; color:{ch.INK}; margin:.2rem 0 .6rem;}}
.clock .meta {{color:{ch.INK2}; font-size:.9rem;}}
@media (max-width: 640px) {{ .tiles {{grid-template-columns:repeat(2, minmax(0, 1fr));}} }}
</style>
"""


# --- shadow API ---------------------------------------------------------------
def fetch(path: str, **params):
    r = requests.get(SHADOW_URL + path, params=params, timeout=5)
    r.raise_for_status()
    return r.json()


def send(cmd: dict) -> None:
    try:
        requests.post(SHADOW_URL + "/replay", json=cmd, timeout=5).raise_for_status()
        st.toast(f"Sent to replay: {cmd}")
    except requests.RequestException as exc:
        st.toast(f"Replay command failed: {exc}")


@st.cache_data(ttl=600, show_spinner=False)
def home_config() -> dict:
    return fetch("/config")


@st.cache_data(ttl=120, show_spinner=False)
def day_list() -> list[str]:
    return fetch("/days")


@st.cache_data(ttl=3600, show_spinner=False)
def usual_day(user: str) -> dict:
    return fetch("/usual", user=user)


def offline(exc: Exception) -> None:
    st.error(f"The shadow service at {SHADOW_URL} is not reachable ({exc.__class__.__name__}). "
             "Start it with `docker compose up` or `uvicorn services.shadow.app:app`.")


def sim_now(state: dict | None) -> pd.Timestamp | None:
    ts = (state or {}).get("clock", {}).get("sim_ts")
    return pd.Timestamp(ts).tz_convert(TIMEZONE) if ts else None


# --- small renderers ------------------------------------------------------------
title = ch.nice


def meal_name(window: str, activity: str) -> str:
    if activity == "TAKE MEDS":
        return "Medication"
    hour = int(window[:2])
    return "Breakfast" if hour < 11 else "Lunch" if hour < 17 else "Dinner"


ROUTINE_STATUS = {
    "done": ("✅", "done"),
    "pending": ("⏳", "pending"),
    "missed": ("⚠️", "missed, alert sent"),
    "away": ("🚶", "away during the window"),
}


def routine_lines(items: list[dict], today: str) -> str:
    rows = []
    for it in items:
        if it["date"] != today and it["status"] != "pending":
            continue
        icon, text = ROUTINE_STATUS.get(it["status"], ("•", it["status"]))
        if it["status"] == "done" and it.get("done_at"):
            text = f"done at {ch.hhmm(it['done_at'])}"
        elif it["status"] == "pending":
            text = f"pending, deadline {ch.hhmm(it['deadline'])}"
        name = meal_name(it["window"], it["activity"]) + (" (last night)" if it["date"] != today else "")
        rows.append(f"<div class='routine'>{icon} <b>{name}</b> "
                    f"<span class='when'>{it['window'].replace('-', '–')}</span> · {text}</div>")
    return "".join(rows) or "<div class='routine'><span class='when'>No routine windows today yet.</span></div>"


def resident_card(user: str, state: dict, today: str) -> None:
    r = state.get("residents", {}).get(user, {})
    room = ch.pretty_room(r.get("room") or "unknown")
    since = ch.hhmm(r.get("room_since"))
    act, truth, conf = r.get("activity"), r.get("true_activity"), r.get("confidence")
    if act and truth:
        match = "✓ estimate matches the label" if act == truth else "✗ estimate differs from the label"
    else:
        match = "waiting for the first estimate"
    probs = " · ".join(f"{title(a)} {p:.0%}" for a, p in (r.get("probs") or {}).items())
    bpm = f"{r['bpm']:.0f} bpm" if r.get("bpm") else "–"
    items = (state.get("routines", {}).get(user) or {}).get("items", [])
    with st.container(border=True):
        st.markdown(
            f"<div class='card-head'><span class='chip' style='background:{ch.user_color(user)}'></span>"
            f"<b>{user}</b> <span class='where'>· {room} since {since}</span></div>"
            f"<div class='kv'>"
            f"<div><div class='k'>Model estimate</div><div class='v'>{title(act)}</div>"
            f"<div class='s'>{conf or 0:.0%} confidence</div></div>"
            f"<div><div class='k'>True label</div><div class='v'>{title(truth)}</div></div>"
            f"<div><div class='k'>Heart rate</div><div class='v'>{bpm}</div>"
            f"<div class='s'>{r.get('steps') or 0:.0f} steps today</div></div></div>"
            f"<div class='match'>{match}{' · top 3: ' + probs if probs else ''}</div>"
            f"<div class='k' style='font-size:.75rem;color:{ch.INK2};margin-top:.2rem'>Routines today</div>"
            f"{routine_lines(items, today)}",
            unsafe_allow_html=True)


def climate_tiles(state: dict) -> None:
    s = state.get("sensors", {})

    def reading(sid: str, key: str, fmt: str) -> str:
        v = (s.get(sid, {}).get("readings") or {}).get(key)
        return fmt.format(v) if v is not None else "–"

    tiles = [
        ("Bathroom temperature", reading("th1", "temperature", "{:.1f} °C")),
        ("Bathroom humidity", reading("th1", "humidity", "{:.0f} %")),
        ("Kitchen temperature", reading("th2", "temperature", "{:.1f} °C")),
        ("Kitchen humidity", reading("th2", "humidity", "{:.0f} %")),
        ("TV power", reading("p1", "power", "{:.0f} W")),
        ("Washing machine", reading("p2", "power", "{:.0f} W")),
        ("Bathroom light", reading("l1", "illuminance", "{:.0f} lx")),
        ("Bedroom light", reading("l2", "illuminance", "{:.0f} lx")),
    ]
    st.markdown("<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in tiles) + "</div>",
        unsafe_allow_html=True)


def alert_list(alerts: list[dict]) -> None:
    if not alerts:
        st.markdown("<div class='ok'>✓ No alerts in this replay so far: every routine window that closed "
                    "had a meal or medication, or the resident was away.</div>", unsafe_allow_html=True)
        return
    html = []
    for a in alerts[:6]:
        kind = "Missed medication" if a.get("kind") == "missed_meds" else "Missed meal"
        html.append(f"<div class='alert'><span class='t'>⚠ {kind}</span> · <b>{a.get('user')}</b> · "
                    f"usual window {a.get('expected_at', '').replace('-', '–')} on {a.get('date')}"
                    f"<br>{a.get('message', '')}</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# --- sidebar --------------------------------------------------------------------
@st.fragment(run_every=2)
def clock_panel() -> None:
    try:
        state = fetch("/state")
    except requests.RequestException as exc:
        st.caption(f"Shadow offline ({exc.__class__.__name__})")
        return
    now, clock = sim_now(state), state.get("clock", {})
    if now is None:
        st.caption("Waiting for the replay clock…")
        return
    status = "paused" if clock.get("paused") else "playing"
    st.markdown(f"**{now:%a %d %b %Y}**  \n**{now:%H:%M}** Madrid time  \n"
                f"dataset day {clock.get('day')} · {clock.get('speed', 0):.0f}× · {status}")


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Replay")
        clock_panel()
        if "speed" not in st.session_state:      # the controls start at the replay's current setting
            try:
                clock = fetch("/state").get("clock") or {}
            except requests.RequestException:
                clock = {}
            st.session_state.speed = min(SPEEDS, key=lambda s: abs(s - float(clock.get("speed") or 600)))
            st.session_state.day = int(clock.get("day") or 48)
        c1, c2 = st.columns(2)
        c1.button("⏸ Pause", on_click=send, args=({"action": "pause"},), width="stretch")
        c2.button("▶ Play", on_click=send, args=({"action": "resume"},), width="stretch")
        speed = st.select_slider("Speed (simulated seconds per second)", SPEEDS, key="speed")
        st.button("Set speed", on_click=lambda: send({"action": "speed", "value": st.session_state.speed}),
                  width="stretch")
        day = st.number_input("Dataset day (test days are 48–62)", min_value=1, max_value=62, step=1, key="day")
        st.button("Jump to day", on_click=send, args=({"action": "seek", "day": int(day)},),
                  width="stretch", type="primary")
        st.caption(f"Speed {speed}× plays one day in {86400 / speed / 60:.0f} min. "
                   "A jump back clears the history after that point.")
        st.divider()
        st.caption(f"Shadow API: {SHADOW_URL}  \nData: SDHAR-HOME, 2 residents, 62 days, replayed over MQTT.")


# --- pages ------------------------------------------------------------------------
@st.fragment(run_every=2)
def live_view() -> None:
    try:
        state, cfg = fetch("/state"), home_config()
    except requests.RequestException as exc:
        offline(exc)
        return
    now, clock = sim_now(state), state.get("clock", {})
    if now is None:
        st.info("Waiting for the replay to start publishing…")
        return
    today = now.strftime("%Y-%m-%d")
    st.markdown(f"<div class='clock'><b>{now:%A %d %B %Y, %H:%M}</b> "
                f"<span class='meta'>Madrid time · dataset day {clock.get('day')} · "
                f"replay {clock.get('speed', 0):.0f}× {'(paused)' if clock.get('paused') else ''}</span></div>",
                unsafe_allow_html=True)
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.plotly_chart(ch.floor_plan(state, cfg), width="stretch", theme=None, config=PLOT_CONFIG,
                        key="plan")
        st.markdown("<div class='legend'>Sensors: □ door or drawer · ○ motion · ◇ vibration · ☆ power · "
                    "△ temperature and humidity · ⬡ light. <b>Filled</b> = active now. A shaded room has someone "
                    "in it. Hover a sensor for its object and last value.</div>", unsafe_allow_html=True)
        st.markdown("**Room climate and appliances**")
        climate_tiles(state)
    with right:
        for user in USERS:
            resident_card(user, state, today)
    st.markdown("**Alerts**")
    alert_list(state.get("alerts", []))


def live_page() -> None:
    st.markdown("### Live home")
    st.caption("The shadow's current picture of the home: where each resident is, which sensors fired, what the "
               "model thinks they are doing, and whether today's meals and medication happened.")
    live_view()


def render_timeline(user: str, date: str) -> None:
    try:
        tl = fetch("/timeline", user=user, date=date)
    except requests.RequestException as exc:
        offline(exc)
        return
    agree, n = ch.minute_agreement(tl)
    hp, ha = tl["hours_predicted"], tl["hours_actual"]
    c = st.columns(4)
    c[0].metric("Estimate matches label", f"{agree:.0%}" if agree is not None else "–",
                help="Share of the estimated minutes where the model's activity equals the labelled one.")
    c[1].metric("Minutes estimated", f"{n:,}" if n else "–")
    c[2].metric("Sleep, estimated / labelled", f"{hp.get('SLEEP', 0):.1f} h / {ha.get('SLEEP', 0):.1f} h")
    c[3].metric("Alerts", len(tl["alerts"]))
    if not tl["predicted"]:
        st.info("No model estimates for this day yet. The replay plays the test days 48–62: jump to this day in "
                "the sidebar and the estimates fill in as it plays.")
    st.plotly_chart(ch.day_timeline(tl), width="stretch", theme=None, config=PLOT_CONFIG, key="timeline")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("**Hours per activity**")
        series = [("Model estimate", hp, ch.user_color(user)), ("Label (ground truth)", ha, ch.TRUTH)]
        st.plotly_chart(ch.hours_bars([s for s in series if s[1]]), width="stretch", theme=None,
                        config=PLOT_CONFIG, key="hours")
    with right:
        st.markdown("**Alerts this day**")
        if tl["alerts"]:
            st.dataframe(pd.DataFrame([{"time": ch.hhmm(a["ts"]), "kind": a["kind"].replace("_", " "),
                                        "usual window": a["expected_at"].replace("-", "–")}
                                       for a in tl["alerts"]]), hide_index=True, width="stretch")
        else:
            st.caption("None.")


def timeline_page() -> None:
    st.markdown("### Day timeline")
    st.caption("Grey bands are the labelled activities (ground truth), the coloured bars inside them the model's "
               "estimate, minute by minute. The lower panel is the room from the wristband's BLE position.")
    try:
        days, state = day_list(), fetch("/state")
    except requests.RequestException as exc:
        offline(exc)
        return
    now = sim_now(state)
    sim_date = now.strftime("%Y-%m-%d") if now is not None else None
    c1, c2, c3 = st.columns([1, 1, 1.4])
    user = c1.radio("Resident", USERS, horizontal=True, key="tl_user")
    default = days.index(sim_date) if sim_date in days else len(days) - 1
    date = c2.selectbox("Day (local date)", days, index=default, key="tl_date")
    follow = c3.toggle("Follow the replay (refresh every 5 s)", value=date == sim_date, key="tl_follow")
    st.fragment(render_timeline, run_every=5 if follow else None)(user, date)


def compare_page() -> None:
    st.markdown("### Compare days")
    st.caption("How a day differed from the resident's usual day (mean of the labelled training days 1–40) "
               "or from another day. Large differences are what a carer would ask about.")
    try:
        days, state = day_list(), fetch("/state")
    except requests.RequestException as exc:
        offline(exc)
        return
    now = sim_now(state)
    sim_date = now.strftime("%Y-%m-%d") if now is not None else days[-1]
    usual_label = "Usual day (mean of days 1–40)"
    c1, c2, c3, c4 = st.columns([0.9, 1, 1.3, 1.1])
    user = c1.radio("Resident", USERS, horizontal=True, key="cmp_user")
    date = c2.selectbox("Day", days, index=days.index(sim_date) if sim_date in days else len(days) - 1,
                        key="cmp_date")
    other = c3.selectbox("Compare with", [usual_label] + days, key="cmp_other")
    source = c4.radio("Hours of this day from", ["Labels", "Model estimate"], horizontal=True, key="cmp_src")
    try:
        tl = fetch("/timeline", user=user, date=date)
        if other == usual_label:
            u = usual_day(user)
            ref, ref_name = u["hours"], f"Usual day ({u['days']} days, labels)"
        else:
            tl2 = fetch("/timeline", user=user, date=other)
            ref = tl2["hours_actual"] if source == "Labels" else tl2["hours_predicted"]
            ref_name = f"{other} ({source.lower()})"
    except requests.RequestException as exc:
        offline(exc)
        return
    mine = tl["hours_actual"] if source == "Labels" else tl["hours_predicted"]
    if not mine:
        st.info("No model estimates for this day: it has not been replayed yet. Choose “Labels”, or jump to the "
                "day in the sidebar.")
        return
    if not ref:
        st.info(f"No hours for {other} from the {source.lower()}.")
        return
    acts = sorted(set(mine) | set(ref), key=lambda a: -abs(mine.get(a, 0) - ref.get(a, 0)))
    diff = pd.DataFrame([{"activity": ch.nice(a), "this day (h)": mine.get(a, 0.0), "compared (h)": ref.get(a, 0.0),
                          "Δ h": round(mine.get(a, 0.0) - ref.get(a, 0.0), 2)} for a in acts])
    top = [f"{r.activity} {r._4:+.1f} h" for r in diff.head(3).itertuples() if abs(r._4) >= 0.25]
    if top:
        st.markdown(f"**Biggest differences:** {', '.join(top)}")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        fig = ch.hours_bars([(f"{date} ({source.lower()})", mine, ch.user_color(user)), (ref_name, ref, ch.TRUTH)])
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key="cmp")
    with right:
        st.dataframe(diff.round(2), hide_index=True, width="stretch",
                     column_config={"Δ h": st.column_config.NumberColumn(format="%+.2f")})
        st.caption(f"Compared with: {ref_name}.")


@st.cache_data(show_spinner=False)
def load_reports() -> dict:
    out = {}
    for name in ("metrics", "alerts"):
        p = REPORTS / f"{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else None
    for name in ("per_class", "feature_importance"):
        p = REPORTS / f"{name}.csv"
        out[name] = pd.read_csv(p) if p.exists() else None
    p = REPORTS / "confusion_test.csv"
    out["confusion"] = pd.read_csv(p, index_col=0) if p.exists() else None
    return out


def window_label(w: dict) -> str:
    f = lambda m: f"{(m // 60) % 24:02d}:{m % 60:02d}"  # noqa: E731
    return f"{f(w['start_min'])}–{f(w['end_min'])}"


def model_page() -> None:
    st.markdown("### Model quality")
    rep = load_reports()
    m = rep["metrics"]
    if m is None:
        st.info("No reports yet. Run `python -m pipeline.train_model` and `python -m pipeline.train_routines`.")
        return
    st.caption("Trained on days 1–40, tuned on 41–47, tested on the unseen days 48–62 (split by day, never "
               "shuffled). The dataset authors' own predictions are the baseline, scored on the same minutes.")
    ours, auth = m["ours_on_same_minutes"], m["authors_predictions"]
    c = st.columns(4)
    c[0].metric("Accuracy", f"{ours['accuracy']:.1%}", f"{ours['accuracy'] - auth['accuracy']:+.1%} vs authors' "
                f"{auth['accuracy']:.1%}")
    c[1].metric("Macro-F1 (all 19 activities equal)", f"{ours['macro_f1']:.3f}",
                f"{ours['macro_f1'] - auth['macro_f1']:+.3f} vs authors' {auth['macro_f1']:.3f}")
    c[2].metric("Accuracy within ±10 min (authors' rule)", f"{ours['accuracy_pm10min']:.1%}",
                f"{ours['accuracy_pm10min'] - auth['accuracy_pm10min']:+.1%} vs authors' "
                f"{auth['accuracy_pm10min']:.1%}")
    c[3].metric("Test minutes compared", f"{ours['minutes']:,}",
                f"of {m['ours']['minutes']:,}; the authors' file covers {auth['coverage']:.0%}",
                delta_color="off", delta_arrow="off")
    st.caption(f"Accuracy per resident, all {m['ours']['minutes']:,} test minutes: "
               f"user1 {m['per_user']['user1']['accuracy']:.1%}, user2 {m['per_user']['user2']['accuracy']:.1%}.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**F1 per activity**")
        if rep["per_class"] is not None:
            st.plotly_chart(ch.per_class_f1(rep["per_class"]), width="stretch", theme=None,
                            config=PLOT_CONFIG)
    with right:
        st.markdown("**Confusion matrix** (rows: labelled activity, share of its minutes)")
        if rep["confusion"] is not None:
            st.plotly_chart(ch.confusion(rep["confusion"]), width="stretch", theme=None, config=PLOT_CONFIG)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Most useful features**")
        if rep["feature_importance"] is not None:
            try:
                sensors = home_config()["sensors"]
            except requests.RequestException:
                sensors = {}
            st.plotly_chart(ch.importance(rep["feature_importance"], sensors), width="stretch", theme=None,
                            config=PLOT_CONFIG)
    with right:
        a = rep["alerts"]
        if a is not None:
            st.markdown("**Learned routines** (from the labels of days 1–40)")
            st.dataframe(pd.DataFrame([{"resident": u, "routine": meal_name(window_label(w), w["activity"]),
                                        "activity": ch.nice(w["activity"]), "usual window": window_label(w),
                                        "on % of days": f"{w['support']:.0%}"}
                                       for u, ws in a["routines"].items() for w in ws]),
                         hide_index=True, width="stretch")
            st.markdown("**Routine alerts on the test days**")
            ev = a.get("drawer_or_model") or a["drawer_only"]
            st.dataframe(pd.DataFrame([{"routine": "Medication" if act == "TAKE MEDS" else "Meals",
                                        "windows": s["windows"], "misses": s["true_misses"],
                                        "alerts": s["alerts"], "correct": s["correct_alerts"],
                                        "false": s["false_alerts"], "unflagged": s["missed_misses"]}
                                       for act, s in ev["real"].items()]), hide_index=True, width="stretch")
            st.caption("A window is a resident's usual meal or medication time on one day. An alert is raised "
                       "when nothing was detected by the deadline (30 min after the window) and the resident "
                       "was at home. A meal is at least 3 minutes the model calls Eat (chosen on the validation "
                       "days); medication is the medicine drawer opening, or the model saying Take meds.")
            inj = ev["injected_missed_doses"]
            st.caption(f"Simulated missed doses (medicine-drawer events deleted, model re-run): "
                       f"{inj['flagged']} of {inj['trials']} flagged.")


# --- app --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="SDHAR-HOME digital shadow", page_icon="🏠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    pages = [
        st.Page(live_page, title="Live home", icon="🏠", default=True),
        st.Page(timeline_page, title="Day timeline", icon="📅", url_path="timeline"),
        st.Page(compare_page, title="Compare days", icon="⚖️", url_path="compare"),
        st.Page(model_page, title="Model quality", icon="📊", url_path="model"),
    ]
    nav = st.navigation(pages, position="top")
    sidebar()
    nav.run()


main()
