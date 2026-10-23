"""Empirical model profiles: real models' decile-to-decile behavior, reusable in simulation.

A parametric knob (concentration, value tilt) can only produce the asymmetries
someone thought to build in. Real models make messier trade-offs -- a tree that
over-weights frequent small donors, a sparse logistic model that finds the
big givers but misses the rest. A profile captures that shape from real
predictions so the simulator can impose it on any population.

Profile of one model (scores split into `n_bins` equal-size rank bins, 1 = top):
    joint      (n_value_buckets, n_bins) share of positives in each value bucket
               that land in each score bin. Rows: low -> high value.
               This one matrix carries both the lift curve (column sums) and
               *which* positives the model finds (how rows differ).
    calib      mean predicted probability per bin / overall positive rate,
               i.e. the model's own (mis)calibration.

    extract_profiles(y, value, scores, names, configs) -> ProfileBank
    save_bank / load_bank                               -> .npz on disk
    profile_scores(pop, profile, strength, calib_bias, rng) -> probabilities
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metric_clinic_metrics import rank_order

N_BINS = 20
N_VALUE_BUCKETS = 10


@dataclass
class ProfileBank:
    names: list
    joint: np.ndarray          # (n_models, n_value_buckets, n_bins)
    calib: np.ndarray          # (n_models, n_bins) classification: p / rate; regression: pred / mean(y)
    summary: pd.DataFrame      # one row per model: config + headline metrics
    source: str = ""
    task: str = "classification"
    curve: np.ndarray | None = None    # (n_models, n_bins) lift (cls) or mean actual / overall mean (reg)
    level_z: np.ndarray | None = None  # regression: (pred - mean(y)) / sd(y) per bin

    def __len__(self):
        return len(self.names)

    def lift(self, i):
        """Lift per bin: rate of 1s (classification) or mean actual (regression) vs overall."""
        if self.curve is not None:
            return self.curve[i]
        return self.joint[i].mean(axis=0) * self.joint.shape[2]


def _rank_bins(score, n_bins):
    order = rank_order(score)
    bins = np.empty(len(score), dtype=int)
    bins[order] = np.arange(len(score)) * n_bins // len(score)
    return bins


def _equal_count_buckets(v, n_buckets):
    pos = np.empty(len(v), dtype=int)
    pos[rank_order(v)] = np.arange(len(v) - 1, -1, -1)
    return pos * n_buckets // len(v)


def _value_buckets(v, n_buckets):
    """Buckets of v, low -> high. Equal-count, except that a dominant minimum (the
    zeros of a zero-inflated target) gets bucket 0 to itself, so non-zero records
    never share a bucket -- and so a profile row -- with zeros."""
    v = np.asarray(v, float)
    at_min = v == v.min()
    if at_min.mean() <= 1 / n_buckets or at_min.all():
        return _equal_count_buckets(v, n_buckets)
    out = np.zeros(len(v), dtype=int)
    out[~at_min] = 1 + _equal_count_buckets(v[~at_min], n_buckets - 1)
    return out


def extract_profiles(y, value, scores, names, configs=None, n_bins=N_BINS,
                     n_value_buckets=N_VALUE_BUCKETS, source="") -> ProfileBank:
    y = np.asarray(y).astype(bool)
    vb = _value_buckets(np.asarray(value)[y], n_value_buckets)
    rate = y.mean()
    joint = np.zeros((len(scores), n_value_buckets, n_bins))
    calib = np.zeros((len(scores), n_bins))
    rows = []
    for m, s in enumerate(scores):
        bins = _rank_bins(np.asarray(s, float), n_bins)
        counts = np.zeros((n_value_buckets, n_bins))
        np.add.at(counts, (vb, bins[y]), 1)
        counts += 0.5                                   # light smoothing for empty cells
        joint[m] = counts / counts.sum(axis=1, keepdims=True)
        calib[m] = np.bincount(bins, weights=s, minlength=n_bins) / np.bincount(bins, minlength=n_bins) / rate
        lift = np.bincount(bins[y], minlength=n_bins) / y.sum() * n_bins
        rows.append({"top_bin_lift": lift[0], "top_decile_lift": lift[: n_bins // 10].mean(),
                     "value_tilt": float(np.corrcoef(vb, -bins[y])[0, 1])})
    summary = pd.DataFrame(rows, index=pd.Index(list(names), name="model"))
    if configs is not None:
        summary = pd.concat([configs.reset_index(drop=True).set_index(summary.index), summary], axis=1)
    return ProfileBank(list(names), joint, calib, summary, source)


def extract_regression_profiles(y, preds, names, configs=None, n_bins=N_BINS,
                                n_value_buckets=N_VALUE_BUCKETS, source="") -> ProfileBank:
    """Regression analogue: rows are buckets of the *actual* target, columns are
    prediction ventiles; plus each ventile's prediction level so the simulator
    can reproduce bias, shrinkage and spread, not just the ranking."""
    y = np.asarray(y, float)
    vb = _value_buckets(y, n_value_buckets)
    m_y, sd_y = y.mean(), y.std()
    k = len(preds)
    joint = np.zeros((k, n_value_buckets, n_bins))
    ratio, level_z, curve = (np.zeros((k, n_bins)) for _ in range(3))
    rows = []
    for m, pred in enumerate(preds):
        pred = np.asarray(pred, float)
        bins = _rank_bins(pred, n_bins)
        counts = np.zeros((n_value_buckets, n_bins))
        np.add.at(counts, (vb, bins), 1)
        counts += 0.5
        joint[m] = counts / counts.sum(axis=1, keepdims=True)
        size = np.bincount(bins, minlength=n_bins)
        mean_pred = np.bincount(bins, weights=pred, minlength=n_bins) / size
        ratio[m] = mean_pred / m_y
        level_z[m] = (mean_pred - m_y) / sd_y
        curve[m] = np.bincount(bins, weights=y, minlength=n_bins) / size / m_y
        rows.append({"top_bin_value_lift": curve[m][0], "top_decile_value_lift": curve[m][: n_bins // 10].mean(),
                     "spearman": pd.Series(pred).corr(pd.Series(y), method="spearman"),
                     "mean_error": float((pred - y).mean())})
    summary = pd.DataFrame(rows, index=pd.Index(list(names), name="model"))
    if configs is not None:
        summary = pd.concat([configs.reset_index(drop=True).set_index(summary.index), summary], axis=1)
    return ProfileBank(list(names), joint, ratio, summary, source, "regression", curve, level_z)


def save_bank(bank: ProfileBank, path):
    extra = {k: v for k, v in (("curve", bank.curve), ("level_z", bank.level_z)) if v is not None}
    np.savez_compressed(path, names=np.array(bank.names), joint=bank.joint, calib=bank.calib,
                        summary=np.array(bank.summary.reset_index().to_json(orient="records")),
                        source=np.array(bank.source), task=np.array(bank.task), **extra)


def load_bank(path) -> ProfileBank:
    z = np.load(path)
    summary = pd.DataFrame(json.loads(str(z["summary"]))).set_index("model")
    task = str(z["task"]) if "task" in z else "classification"
    return ProfileBank(list(z["names"]), z["joint"], z["calib"], summary, str(z["source"]), task,
                       z["curve"] if "curve" in z else None, z["level_z"] if "level_z" in z else None)


def profile_scores(pop, joint, calib, strength, calib_bias, rng):
    """Probabilities for `pop` that reproduce a real model's bin-by-bin behavior.

    strength   1 = the real model's shape; 0 = random ranking; >1 sharpens it.
    calib_bias extra log-odds shift on top of the real model's own calibration.
    """
    n_vb, n_bins = joint.shape
    y = pop.y.astype(bool)
    n, n_pos = len(y), int(y.sum())
    # positives: bin drawn from the profile row for their value bucket
    rows = joint ** strength if strength > 0 else np.ones_like(joint)
    rows = rows / rows.sum(axis=1, keepdims=True)
    vb = _value_buckets(pop.value[y], n_vb) if pop.spec.value_cv > 0 else \
        rng.integers(0, n_vb, n_pos)
    cum = np.cumsum(rows, axis=1)
    pos_bin = (rng.random(n_pos)[:, None] > cum[vb]).sum(axis=1)
    pos_bin = np.minimum(pos_bin, n_bins - 1)
    # negatives fill whatever room is left in each bin
    cap = np.full(n_bins, n / n_bins)
    room = np.clip(cap - np.bincount(pos_bin, minlength=n_bins), 0, None)
    neg_bin = rng.choice(n_bins, size=n - n_pos, p=room / room.sum())
    bins = np.empty(n, dtype=int)
    bins[y], bins[~y] = pos_bin, neg_bin
    # position down the list: bin + random order within the bin (0 = very top)
    t = bins + rng.random(n)
    # probability: the real model's calibration ratio per bin at this prevalence,
    # forced non-increasing and interpolated so it never contradicts the ranking
    p_bin = np.clip(np.minimum.accumulate(calib) * y.mean(), 1e-6, 1 - 1e-6)
    centers = np.arange(n_bins) + 0.5
    logit = np.interp(t, centers, np.log(p_bin / (1 - p_bin))) - 1e-6 * t
    return 1 / (1 + np.exp(-(logit + calib_bias)))


def _draw_bins(buckets, joint, strength, rng):
    """For each record, a score bin drawn from its bucket's row of the profile."""
    rows = joint ** strength if strength > 0 else np.ones_like(joint)
    rows = rows / rows.sum(axis=1, keepdims=True)
    cum = np.cumsum(rows, axis=1)
    bins = (rng.random(len(buckets))[:, None] > cum[buckets]).sum(axis=1)
    return np.minimum(bins, joint.shape[1] - 1)


def profile_predictions(pop, joint, ratio, level_z, strength, shrink, bias, rng):
    """Regression predictions for `pop` that reproduce a real model's ventile behavior.

    Each record's prediction ventile is drawn from the profile row for its actual
    value bucket; the prediction level for that ventile comes from the real model
    (as a ratio to the mean for skewed targets, in SD units for a normal target).
    strength  1 = the real model's ranking; 0 = random; >1 sharpens it.
    shrink    1 = the real model's spread of predictions; <1 squashes toward the mean.
    bias      added constant, in SDs of the target.
    """
    n_vb, n_bins = joint.shape
    y = pop.y
    bins = _draw_bins(_value_buckets(y, n_vb), joint, strength, rng)
    t = bins + rng.random(len(y))
    m_y, sd_y = y.mean(), y.std()
    if pop.spec.shape == "normal":
        pred = m_y + sd_y * _level_curve(t, level_z)
    else:
        pred = m_y * _level_curve(t, ratio)
    return m_y + shrink * (pred - m_y) + bias * sd_y


def _level_curve(t, levels):
    """Monotone (non-increasing) level at list position t, interpolated between bin
    centers and extrapolated half a bin at each end, with a hair of slope so no
    two records tie."""
    lv = np.minimum.accumulate(np.asarray(levels, float))
    n_bins = len(lv)
    x = np.r_[0.0, np.arange(n_bins) + 0.5, n_bins]
    y_ = np.r_[lv[0] + (lv[0] - lv[1]) / 2, lv, lv[-1] - (lv[-2] - lv[-1]) / 2]
    span = max(lv[0] - lv[-1], 1e-9)
    return np.interp(t, x, y_) - 1e-6 * span * t
