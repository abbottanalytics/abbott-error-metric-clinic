"""Real KDD Cup 1998 gift-amount (TARGET_D) models for the clinic's regression shapes.

Same prepared data and split as the TARGET_B grid (metric_clinic_kdd98.py), two
scopes, each matching one regression preset:

    responders  train and score only donors who gave (TARGET_B == 1):
                a positive, right-skewed target -- like customer lifetime value
    all         train and score every record: ~95% zeros plus skewed gifts --
                predicting the donation directly

Diversity comes from the choices real modelers make: algorithm (ridge, lasso,
regression tree), raw vs log1p target (log models under-predict big gifts),
feature count, tree depth/leaf size, and the training sample.

    python3 metric_clinic_kdd98_target_d.py            # both scopes, predictions + profiles
    python3 metric_clinic_kdd98_target_d.py --profiles-only

Writes data/kdd98_target_d_<scope>.npz (y, scores, names, configs) and
data/kdd98_target_d_<scope>_profiles.npz.

You don't need to run this: the resulting files ship in data/. Retraining the models
needs the author's KDD Cup 98 data-preparation project (kddcup1998, not public), so
on a standalone copy only --profiles-only works, which rebuilds the profile banks
from the saved predictions.
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

from metric_clinic_kdd98 import KDD_DIR, PREP_ID, PROJECTS, SAMPLE_FRACS, SEEDS, SPLIT_ID

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPES = ("responders", "all")
LASSO_ALPHA = {"raw": 0.1, "log1p": 0.01}


def predictions_path(scope):
    return os.path.join(HERE, "data", f"kdd98_target_d_{scope}.npz")


def profiles_path(scope):
    return os.path.join(HERE, "data", f"kdd98_target_d_{scope}_profiles.npz")


def _configs():
    for k in (10, 30):
        for transform in ("raw", "log1p"):
            for penalty in ("ridge", "lasso"):
                yield {"algo": "linear", "k": k, "transform": transform, "penalty": penalty}
    for depth in (4, None):
        for min_leaf in (25, 100):
            for transform in ("raw", "log1p"):
                yield {"algo": "tree", "max_depth": depth, "min_leaf": min_leaf, "transform": transform}


def _model(cfg, seed):
    # scikit-learn is only needed to retrain, so the dashboard runs without it
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    if cfg["algo"] == "tree":
        return DecisionTreeRegressor(max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_leaf"],
                                     random_state=seed)
    reg = Ridge(alpha=1.0) if cfg["penalty"] == "ridge" else \
        Lasso(alpha=LASSO_ALPHA[cfg["transform"]], max_iter=5000)
    return make_pipeline(StandardScaler(), reg)


def build(scope, data_dir, prep_id=PREP_ID, split_id=SPLIT_ID):
    from sklearn.exceptions import ConvergenceWarning
    sys.path.insert(0, PROJECTS)
    from kddcup1998.run_target_b_metriccorr_experiment import sample_training, select_features

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    df = pd.read_parquet(os.path.join(data_dir, "prepared", f"{prep_id}.parquet"))
    df = df.merge(pd.read_parquet(os.path.join(data_dir, "splits", f"{split_id}.parquet")),
                  on="RECORD_ID")
    if scope == "responders":
        df = df[df.TARGET_B == 1]
    train_df, test_df = df[df.is_test == 0], df[df.is_test == 1]
    drop = ["RECORD_ID", "is_test", "TARGET_B", "TARGET_D"]
    feature_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    scores, configs = [], []
    t0 = time.time()
    for frac in SAMPLE_FRACS:
        for seed in SEEDS:
            sample = sample_training(train_df, frac, seed)
            X_full, y_raw = sample[feature_cols], sample.TARGET_D.values.astype(float)
            selected = {k: select_features(X_full, y_raw, k) for k in (10, 30)}
            for cfg in _configs():
                cols = selected[cfg.get("k", 30)]
                y_fit = np.log1p(y_raw) if cfg["transform"] == "log1p" else y_raw
                m = _model(cfg, seed).fit(X_full[cols], y_fit)
                pred = m.predict(test_df[cols])
                if cfg["transform"] == "log1p":
                    # cap at the training range before back-transforming, as a
                    # modeler would; otherwise a few extrapolations explode
                    pred = np.expm1(np.clip(pred, y_fit.min(), y_fit.max()))
                scores.append(pred.astype(np.float32))
                configs.append({**cfg, "frac": frac, "seed": seed})
            print(f"  {scope}: {len(configs)} models, {time.time() - t0:.0f}s", flush=True)

    tag = {"linear": "LN", "tree": "RT"}
    names = [f"{tag[c['algo']]}{i + 1:03d}" for i, c in enumerate(configs)]
    return dict(y=test_df.TARGET_D.values.astype(float), scores=np.vstack(scores),
                names=np.array(names), configs=np.array(json.dumps(configs, default=str)))


def load(scope):
    """-> y, scores, names, configs DataFrame (indexed by model name)."""
    z = np.load(predictions_path(scope))
    names = list(z["names"])
    configs = pd.DataFrame(json.loads(str(z["configs"])), index=pd.Index(names, name="model"))
    return z["y"], z["scores"], names, configs


def write_profiles(scope):
    from metric_clinic_profiles import extract_regression_profiles, save_bank
    y, scores, names, configs = load(scope)
    bank = extract_regression_profiles(y, scores, names, configs,
                                       source=f"KDD Cup 1998 TARGET_D ({scope})")
    save_bank(bank, profiles_path(scope))
    print(f"Saved {len(bank)} {scope} profiles to {profiles_path(scope)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.path.join(KDD_DIR, "data"))
    ap.add_argument("--scope", choices=SCOPES, nargs="*", default=list(SCOPES))
    ap.add_argument("--profiles-only", action="store_true")
    args = ap.parse_args()
    for scope in args.scope:
        if not args.profiles_only:
            out = build(scope, args.data_dir)
            os.makedirs(os.path.dirname(predictions_path(scope)), exist_ok=True)
            np.savez_compressed(predictions_path(scope), **out)
            print(f"Saved {out['scores'].shape[0]} models x {out['scores'].shape[1]:,} records "
                  f"to {predictions_path(scope)}")
        write_profiles(scope)


if __name__ == "__main__":
    main()
