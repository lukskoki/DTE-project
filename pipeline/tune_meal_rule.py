"""Choose the meal rule on the validation days: how many minutes of EAT count as a meal.

    python -m pipeline.tune_meal_rule        (after pipeline.train_model and pipeline.train_routines)

For each candidate rule the routine checker runs over the model's output on the validation days
(41-47) and its meal alerts are scored against the labels. The rule is picked by F0.5 on the
validation days (precision counts double: a false alert sends a carer to check on someone who
ate). Test days are shown for reference only. Also checks whether evidence from 30 minutes before
the window (EARLY_MIN) matters. Writes reports/meal_rule.json.
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import pandas as pd

import shadowhome.routines as routines_mod
from pipeline.train_model import FEATURES_CACHE, META_FILE, MODEL_FILE, REPORT_DIR
from pipeline.train_routines import ROUTINES_FILE, predict, run_checker, truth_status
from shadowhome.config import load_home
from shadowhome.routines import routines_from_json
from shadowhome.schema import LABELS_FILE, TEST_DAYS, VAL_DAYS

MINUTES = (3, 5, 8, 10, 15, 20)
EARLY = (30, 0)


def score(chk, labels: pd.DataFrame, last: float) -> dict:
    rows = [(i.status, truth_status(i, labels)) for i in chk.instances.values() if i.deadline <= last]
    tp = sum(o == "missed" and t == "missed" for o, t in rows)
    fp = sum(o == "missed" and t != "missed" for o, t in rows)
    fn = sum(o != "missed" and t == "missed" for o, t in rows)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f05 = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    return {"windows": len(rows), "real_misses": tp + fn, "alerts": tp + fp, "correct": tp, "false": fp,
            "not_flagged": fn, "precision": round(p, 2), "recall": round(r, 2), "f0.5": round(f05, 2)}


def main() -> int:
    labels = pd.read_parquet(LABELS_FILE)
    meals = {u: [w for w in ws if w.activity == "EAT"]
             for u, ws in routines_from_json(json.loads(ROUTINES_FILE.read_text())).items()}
    model = lgb.Booster(model_file=str(MODEL_FILE))
    meta = json.loads(META_FILE.read_text())
    feat = pd.read_parquet(FEATURES_CACHE)
    drawer_room = load_home()["sensors"]["c2"]["room"]
    defaults = routines_mod.EAT_MIN_MINUTES, routines_mod.EARLY_MIN
    results = []
    for split, days in (("validation", VAL_DAYS), ("test", TEST_DAYS)):
        f = feat[feat["day"].isin(days)].reset_index(drop=True)
        pred = predict(model, meta, f)
        start, last = f["minute"].min().timestamp(), f["minute"].max().timestamp()
        for early in EARLY:
            for k in MINUTES:
                routines_mod.EAT_MIN_MINUTES, routines_mod.EARLY_MIN = k, early
                chk = run_checker(meals, f, pred, set(), drawer_room, start)
                routines_mod.EAT_MIN_MINUTES, routines_mod.EARLY_MIN = defaults
                results.append({"split": split, "eat_minutes": k, "early_min": early,
                                **score(chk, labels, last)})
                print(json.dumps(results[-1]))
    val = [r for r in results if r["split"] == "validation" and r["early_min"] == defaults[1]]
    best = max(val, key=lambda r: (r["f0.5"], -r["eat_minutes"]))
    print(f"\nchosen on validation: {best['eat_minutes']} minutes of EAT (F0.5 {best['f0.5']}); "
          f"in use: {defaults[0]}")
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "meal_rule.json").write_text(json.dumps(
        {"chosen_eat_minutes": best["eat_minutes"], "in_use": {"eat_minutes": defaults[0],
         "early_min": defaults[1]}, "criterion": "F0.5 of meal alerts on validation days 41-47",
         "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
