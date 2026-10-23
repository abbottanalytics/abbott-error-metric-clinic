"""Real KDD Cup 1998 predictions for the clinic's fundraising example.

Re-runs the model grid from kddcup1998/run_target_b_metriccorr_experiment.py
(120 logistic regressions + 360 decision trees on one fixed split) and keeps
what that experiment throws away: every model's score on every test record,
plus the actual gift (TARGET_D), so the dashboard can value each responder at
what they really gave rather than a constant.

    python3 metric_clinic_kdd98.py                  # predictions + profile bank
    python3 metric_clinic_kdd98.py --profiles-only  # rebuild profiles from saved predictions

Output (.npz):
    y        TARGET_B on the test records
    value    TARGET_D (the actual gift; 0 for non-responders)
    scores   float32 (n_models, n_records) predicted probabilities
    names    model ids
    configs  JSON list of each model's configuration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS = os.path.dirname(HERE)
KDD_DIR = os.path.join(PROJECTS, "kddcup1998")
DEFAULT_OUT = os.path.join(HERE, "data", "kdd98_predictions.npz")
DEFAULT_PROFILES = os.path.join(HERE, "data", "kdd98_profiles.npz")

PREP_ID = "prep_6c8725699af8"
SPLIT_ID = "split_86b7d7947fdc"
SEEDS = [13, 19, 25, 27, 33, 39, 45, 49, 51, 57]
SAMPLE_FRACS = [0.50, 0.75, 1.00]


def build(data_dir, prep_id=PREP_ID, split_id=SPLIT_ID):
    # the experiment's own helpers, so the grid stays identical to the original;
    # imported here so the dashboard can load saved predictions without kddcup1998
    sys.path.insert(0, PROJECTS)
    from kddcup1998.run_target_b_metriccorr_experiment import sample_training, select_features

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    df = pd.read_parquet(os.path.join(data_dir, "prepared", f"{prep_id}.parquet"))
    df = df.merge(pd.read_parquet(os.path.join(data_dir, "splits", f"{split_id}.parquet")),
                  on="RECORD_ID")
    train_df, test_df = df[df.is_test == 0], df[df.is_test == 1]
    target = "TARGET_B"
    drop = ["RECORD_ID", "is_test", target, "TARGET_D"]
    feature_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X_test_full = test_df[feature_cols]

    scores, configs = [], []
    t0 = time.time()

    def keep(model, X_test, cfg):
        scores.append(model.predict_proba(X_test)[:, 1].astype(np.float32))
        configs.append(cfg)
        if len(configs) % 40 == 0:
            print(f"  {len(configs)} models, {time.time() - t0:.0f}s", flush=True)

    for frac in SAMPLE_FRACS:
        for seed in SEEDS:
            sample = sample_training(train_df, frac, seed)
            X_full, y_train = sample[feature_cols], sample[target].values
            for k in [25, 100]:
                cols = select_features(X_full, y_train, k)
                for penalty, C in [("l2", 1.0), ("l1", 0.3)]:
                    m = LogisticRegression(penalty=penalty, C=C, solver="liblinear", max_iter=1000)
                    m.fit(X_full[cols], y_train)
                    keep(m, X_test_full[cols], {"algo": "logreg", "frac": frac, "seed": seed,
                                                "k": k, "penalty": penalty, "C": C})

    for frac in SAMPLE_FRACS:
        for seed in SEEDS:
            sample = sample_training(train_df, frac, seed)
            X_train, y_train = sample[feature_cols], sample[target].values
            for w in [1, 10, 30]:
                for depth in [4, None]:
                    for min_leaf in [25, 100]:
                        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                                   class_weight={0: 1, 1: w}, random_state=seed)
                        m.fit(X_train, y_train)
                        keep(m, X_test_full, {"algo": "tree", "frac": frac, "seed": seed,
                                              "class_weight": w, "max_depth": depth,
                                              "min_leaf": min_leaf})

    names = [f"{c['algo'][:2].upper()}{i + 1:03d}" for i, c in enumerate(configs)]
    return dict(y=test_df[target].values.astype(np.int8), value=test_df.TARGET_D.values,
                scores=np.vstack(scores), names=np.array(names),
                configs=np.array(json.dumps(configs)))


def load(path=DEFAULT_OUT):
    """-> y, value, scores, names, configs DataFrame (indexed by model name)."""
    z = np.load(path)
    names = list(z["names"])
    configs = pd.DataFrame(json.loads(str(z["configs"])), index=pd.Index(names, name="model"))
    return z["y"], z["value"], z["scores"], names, configs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.path.join(KDD_DIR, "data"))
    ap.add_argument("--prep-id", default=PREP_ID)
    ap.add_argument("--split-id", default=SPLIT_ID)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--profiles", default=DEFAULT_PROFILES)
    ap.add_argument("--profiles-only", action="store_true",
                    help="skip training; rebuild the profile bank from --output")
    args = ap.parse_args()
    if not args.profiles_only:
        out = build(args.data_dir, args.prep_id, args.split_id)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        np.savez_compressed(args.output, **out)
        print(f"Saved {out['scores'].shape[0]} models x {out['scores'].shape[1]:,} records to {args.output}")
    write_profiles(args.output, args.profiles)


def write_profiles(predictions=DEFAULT_OUT, path=DEFAULT_PROFILES):
    from metric_clinic_profiles import extract_profiles, save_bank
    y, value, scores, names, configs = load(predictions)
    bank = extract_profiles(y, value, scores, names, configs, source="KDD Cup 1998 TARGET_B")
    save_bank(bank, path)
    print(f"Saved {len(bank)} model profiles to {path}")


if __name__ == "__main__":
    main()
