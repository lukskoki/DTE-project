"""Train the activity classifier on earlier days and test it on later days.

    python -m pipeline.train_model

Split by day, never shuffled, so the test days are truly "the future":
  train       days 1-40
  validation  days 41-47 (early stopping)
  test        days 48-62 (also the days the replay plays by default)

Writes
  models/activity_lgbm.txt     the LightGBM model
  models/activity_meta.json    feature list, classes, split, smoothing
  reports/metrics.json         test metrics for our model and the authors' predictions
  reports/per_class.csv        precision / recall / F1 per activity (test days)
  reports/confusion_test.csv   confusion matrix, rows = true, normalised per row
  reports/feature_importance.csv
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from shadowhome.config import sensor_rooms
from shadowhome.features import build_features, feature_columns
from shadowhome.schema import (
    ACTIVITIES, CLEAN_DIR, EVENTS_FILE, FIRST_DAY as FIRST_DAY_DATE, LABELS_FILE, ROOT, TEST_DAYS, TRAIN_DAYS,
    USERS, VAL_DAYS,
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_FILE = MODEL_DIR / "activity_lgbm.txt"
META_FILE = MODEL_DIR / "activity_meta.json"
FEATURES_CACHE = CLEAN_DIR / "features.parquet"
AUTHOR_PRED_FILE = CLEAN_DIR / "author_predictions.parquet"

SMOOTH_MIN = 3          # ablation only: averaging probabilities over 3 min lowered macro-F1
TOLERANCE_MIN = 10      # the authors' rule: correct if the activity occurs within +-10 min
FIRST_DAY = pd.Timestamp(FIRST_DAY_DATE, tz="UTC")  # DAY_1, files are cut at UTC midnight

PARAMS = {
    "objective": "multiclass",
    "num_class": len(ACTIVITIES),
    "learning_rate": 0.05,
    "num_leaves": 31,          # chosen on the validation days (macro-F1), see README
    "min_data_in_leaf": 100,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
}


def attach_labels(feat: pd.DataFrame, labels: pd.DataFrame, col: str = "activity") -> pd.DataFrame:
    out = []
    for user in USERS:
        f = feat[feat["user"] == user].sort_values("minute")
        lab = labels[labels["user"] == user].sort_values("start")[["start", "end", "activity"]]
        m = pd.merge_asof(f, lab.rename(columns={"start": "minute", "activity": col}),
                          on="minute", direction="backward")
        m.loc[~(m["minute"] < m["end"]), col] = None
        out.append(m.drop(columns="end"))
    return pd.concat(out, ignore_index=True)


def load_features() -> pd.DataFrame:
    if FEATURES_CACHE.exists() and FEATURES_CACHE.stat().st_mtime > EVENTS_FILE.stat().st_mtime:
        return pd.read_parquet(FEATURES_CACHE)
    t0 = time.time()
    ev = pd.read_parquet(EVENTS_FILE)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    minutes = pd.date_range(ev["ts"].min().ceil("min"), ev["ts"].max().floor("min"), freq="1min")
    feat = build_features(ev, minutes, sensor_rooms())
    feat["day"] = ((feat["minute"] - pd.Timedelta(seconds=1)).dt.floor("D") - FIRST_DAY).dt.days + 1
    feat.to_parquet(FEATURES_CACHE, index=False)
    print(f"built {len(feat):,} feature rows x {len(feature_columns(feat))} features in {time.time() - t0:.0f}s")
    return feat


def smooth(probs: np.ndarray, frame: pd.DataFrame, k: int) -> np.ndarray:
    """Causal moving average of class probabilities per user (rows sorted by user, minute)."""
    out = np.empty_like(probs)
    for user in frame["user"].unique():
        idx = np.flatnonzero(frame["user"].to_numpy() == user)
        out[idx] = pd.DataFrame(probs[idx]).rolling(k, min_periods=1).mean().to_numpy()
    return out


def tolerant_accuracy(true: np.ndarray, pred: np.ndarray, frame: pd.DataFrame, tol: int) -> float:
    """Share of minutes whose predicted activity occurs in the labels within +-tol minutes."""
    hits = np.zeros(len(true), bool)
    classes = {a: i for i, a in enumerate(ACTIVITIES)}
    for user in frame["user"].unique():
        idx = np.flatnonzero(frame["user"].to_numpy() == user)
        onehot = np.zeros((len(idx), len(ACTIVITIES)))
        onehot[np.arange(len(idx)), [classes[a] for a in true[idx]]] = 1
        near = pd.DataFrame(onehot).rolling(2 * tol + 1, center=True, min_periods=1).max().to_numpy()
        hits[idx] = near[np.arange(len(idx)), [classes[a] for a in pred[idx]]] > 0
    return float(hits.mean())


def scores(true: np.ndarray, pred: np.ndarray, frame: pd.DataFrame) -> dict:
    return {
        "minutes": int(len(true)),
        "accuracy": round(accuracy_score(true, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(true, pred), 4),
        "macro_f1": round(f1_score(true, pred, average="macro", labels=sorted(set(true))), 4),
        "weighted_f1": round(f1_score(true, pred, average="weighted"), 4),
        "accuracy_pm10min": round(tolerant_accuracy(true, pred, frame, TOLERANCE_MIN), 4),
    }


def main() -> int:
    feat = load_features()
    labels = pd.read_parquet(LABELS_FILE)
    data = attach_labels(feat, labels).dropna(subset=["activity"])
    data = data[data["activity"].isin(ACTIVITIES)].sort_values(["user", "minute"]).reset_index(drop=True)
    cols = feature_columns(feat)
    y = data["activity"].map({a: i for i, a in enumerate(ACTIVITIES)}).to_numpy()

    split = {name: data["day"].isin(days).to_numpy()
             for name, days in (("train", TRAIN_DAYS), ("val", VAL_DAYS), ("test", TEST_DAYS))}
    for name, mask in split.items():
        print(f"{name:<5} {mask.sum():>7,} labelled minutes")

    # Rare activities get more weight (square root of inverse frequency, capped).
    freq = np.bincount(y[split["train"]], minlength=len(ACTIVITIES)).astype(float)
    cw = np.sqrt(freq.max() / np.maximum(freq, 1))
    cw = np.minimum(cw, 30)
    train = lgb.Dataset(data.loc[split["train"], cols], y[split["train"]], weight=cw[y[split["train"]]])
    val = lgb.Dataset(data.loc[split["val"], cols], y[split["val"]], weight=cw[y[split["val"]]], reference=train)
    model = lgb.train(PARAMS, train, num_boost_round=2000, valid_sets=[val],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    print(f"best iteration {model.best_iteration}")

    test = data[split["test"]].reset_index(drop=True)
    probs = model.predict(test[cols], num_iteration=model.best_iteration)
    true = test["activity"].to_numpy()
    raw = np.array(ACTIVITIES)[probs.argmax(1)]
    smoothed = np.array(ACTIVITIES)[smooth(probs, test, SMOOTH_MIN).argmax(1)]

    metrics = {
        "split": {"train_days": [1, 40], "val_days": [41, 47], "test_days": [48, 62]},
        "best_iteration": model.best_iteration,
        "ours": scores(true, raw, test),
        f"ablation_smoothed_{SMOOTH_MIN}min": scores(true, smoothed, test),
        "per_user": {u: scores(true[test["user"] == u], raw[test["user"] == u],
                               test[test["user"] == u].reset_index(drop=True)) for u in USERS},
    }
    # Baseline: the dataset authors' own predictions on the same test minutes.
    if AUTHOR_PRED_FILE.exists():
        ap = attach_labels(test[["user", "minute"]], pd.read_parquet(AUTHOR_PRED_FILE), col="author")
        ap = test[["user", "minute", "activity"]].merge(ap, on=["user", "minute"], how="left")
        has = ap["author"].notna().to_numpy()
        sub = ap[has].reset_index(drop=True)
        metrics["authors_predictions"] = scores(sub["activity"].to_numpy(), sub["author"].to_numpy(), sub)
        metrics["authors_predictions"]["coverage"] = round(float(has.mean()), 4)
        ours_sub = raw[has]
        metrics["ours_on_same_minutes"] = scores(sub["activity"].to_numpy(), ours_sub, sub)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    present = [a for a in ACTIVITIES if a in set(true) | set(raw)]
    p, r, f, s = precision_recall_fscore_support(true, raw, labels=present, zero_division=0)
    per_class = pd.DataFrame({"activity": present, "precision": p.round(3), "recall": r.round(3),
                              "f1": f.round(3), "support_min": s}).sort_values("support_min", ascending=False)
    cm = confusion_matrix(true, raw, labels=present).astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    imp = pd.DataFrame({"feature": cols, "gain": model.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False)
    imp["share"] = (imp["gain"] / imp["gain"].sum()).round(4)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    model.save_model(str(MODEL_FILE), num_iteration=model.best_iteration)
    META_FILE.write_text(json.dumps({
        "version": time.strftime("lgbm-%Y%m%d-%H%M"),
        "features": cols, "classes": ACTIVITIES,
        "train_days": [1, 40], "class_weights": dict(zip(ACTIVITIES, cw.round(2).tolist())),
    }, indent=2))
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    per_class.to_csv(REPORT_DIR / "per_class.csv", index=False)
    pd.DataFrame(cm.round(3), index=present, columns=present).to_csv(REPORT_DIR / "confusion_test.csv")
    imp[["feature", "share"]].to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print(per_class.to_string(index=False))
    print(imp.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
