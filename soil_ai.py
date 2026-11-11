"""Soil quality AI: estimates a 0-100 soil quality score and a grade
(Poor / Fair / Good / Excellent) from basic soil measurements.

Usage:
    python soil_ai.py train                      # train on synthetic data
    python soil_ai.py train --data my_soil.csv   # train on your own lab data
    python soil_ai.py predict --ph 6.5 --organic_carbon 2.1 --nitrogen 0.15 \
        --phosphorus 25 --potassium 180 --ec 0.8 --moisture 25 --bulk_density 1.3 \
        --cec 18 --clay 25 --sand 40
    python soil_ai.py predict --csv new_samples.csv

Your own CSV must contain the FEATURES columns plus `soil_quality_score` (0-100).
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = "soil_model.joblib"
FEATURES = ["ph", "organic_carbon", "nitrogen", "phosphorus", "potassium",
            "ec", "moisture", "bulk_density", "cec", "clay", "sand"]
# units: organic_carbon %, nitrogen %, phosphorus ppm, potassium ppm,
# ec dS/m, moisture %, bulk_density g/cm3, cec cmol/kg, clay %, sand %
GRADES = [(80, "Excellent"), (60, "Good"), (40, "Fair"), (0, "Poor")]


def _tri(x, lo, opt_lo, opt_hi, hi):
    """1.0 inside the optimal band, falling linearly to 0 at lo / hi."""
    x = np.asarray(x, dtype=float)
    up = np.clip((x - lo) / (opt_lo - lo), 0, 1)
    down = np.clip((hi - x) / (hi - opt_hi), 0, 1)
    return np.minimum(up, down)


def agronomic_score(d):
    """Rule-based soil quality index (0-100) from published-style optimal ranges.
    Used to label synthetic data; replace with real lab/field labels when you have them."""
    texture = 1 - np.abs(d["clay"] - 25) / 60 - np.abs(d["sand"] - 40) / 120
    parts = {
        "ph": (_tri(d["ph"], 4.0, 6.0, 7.5, 9.0), 0.17),
        "organic_carbon": (np.clip(d["organic_carbon"] / 3.0, 0, 1), 0.17),
        "nitrogen": (np.clip(d["nitrogen"] / 0.2, 0, 1), 0.10),
        "phosphorus": (_tri(d["phosphorus"], 2, 15, 60, 150), 0.09),
        "potassium": (_tri(d["potassium"], 40, 120, 300, 600), 0.08),
        "ec": (_tri(d["ec"], -0.5, 0.0, 1.5, 6.0), 0.08),
        "moisture": (_tri(d["moisture"], 3, 18, 35, 55), 0.07),
        "bulk_density": (_tri(d["bulk_density"], 0.8, 1.0, 1.4, 1.8), 0.09),
        "cec": (np.clip(d["cec"] / 20, 0, 1), 0.08),
        "texture": (np.clip(texture, 0, 1), 0.07),
    }
    return 100 * sum(v * w for v, w in parts.values())


def grade(score):
    return next(name for cut, name in GRADES if score >= cut)


def synthetic_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    clay = rng.uniform(3, 60, n)
    sand = np.clip(rng.uniform(5, 90, n) * (1 - clay / 100), 2, 92)
    oc = rng.gamma(2.0, 0.9, n).clip(0.1, 8)
    df = pd.DataFrame({
        "ph": rng.normal(6.7, 1.1, n).clip(3.8, 9.5),
        "organic_carbon": oc,
        "nitrogen": (oc * 0.09 + rng.normal(0, 0.02, n)).clip(0.01, 0.6),
        "phosphorus": rng.lognormal(3.0, 0.9, n).clip(1, 200),
        "potassium": rng.lognormal(5.1, 0.6, n).clip(30, 800),
        "ec": rng.gamma(1.5, 0.7, n).clip(0.05, 8),
        "moisture": rng.normal(24, 9, n).clip(2, 60),
        "bulk_density": rng.normal(1.35, 0.17, n).clip(0.8, 1.9),
        "cec": (clay * 0.5 + oc * 4 + rng.normal(0, 3, n)).clip(1, 50),
        "clay": clay,
        "sand": sand,
    })
    df["soil_quality_score"] = (agronomic_score(df) + rng.normal(0, 2, n)).clip(0, 100)
    return df


def train(data_path=None):
    if data_path:
        df = pd.read_csv(data_path)
        missing = [c for c in FEATURES + ["soil_quality_score"] if c not in df.columns]
        if missing:
            sys.exit(f"CSV is missing columns: {missing}")
        source = data_path
    else:
        df = synthetic_data()
        source = "synthetic data"
    X, y = df[FEATURES], df["soil_quality_score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=0)
    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Trained on {source} ({len(df)} samples)")
    print(f"CV R2: {cv.mean():.3f}   Test R2: {r2_score(y_te, pred):.3f}   "
          f"Test MAE: {mean_absolute_error(y_te, pred):.2f} points")
    grade_acc = np.mean([grade(a) == grade(b) for a, b in zip(y_te, pred)])
    print(f"Grade accuracy: {grade_acc:.1%}")
    print("Feature importance:")
    for f, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {f:15s} {imp:.3f}")
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def predict(rows):
    model = joblib.load(MODEL_PATH)
    scores = np.clip(model.predict(rows[FEATURES]), 0, 100)
    out = rows.copy()
    out["predicted_score"] = scores.round(1)
    out["grade"] = [grade(s) for s in scores]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", help="CSV with FEATURES + soil_quality_score")
    p = sub.add_parser("predict")
    p.add_argument("--csv", help="CSV with FEATURES columns")
    for f in FEATURES:
        p.add_argument(f"--{f}", type=float)
    a = ap.parse_args()

    if a.cmd == "train":
        train(a.data)
        return
    if a.csv:
        rows = pd.read_csv(a.csv)
    else:
        vals = {f: getattr(a, f) for f in FEATURES}
        if any(v is None for v in vals.values()):
            sys.exit("Provide all of: " + ", ".join(f"--{f}" for f in FEATURES) + " (or --csv)")
        rows = pd.DataFrame([vals])
    print(predict(rows).to_string(index=False))


if __name__ == "__main__":
    main()
