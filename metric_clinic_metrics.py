"""Statistical and business metrics for simulated (or real) model scores.

The evaluation functions only need arrays -- actual outcomes, model scores,
and (for classification) the $ value of each positive -- so they work the same
on real predictions as on simulated ones:

    evaluate(task, y, scores, names, biz, value=None) -> DataFrame (one row per model)
    choose(table, select_by, judge_by)                -> who wins, opportunity cost
    regret_by_metric(table, judge_by)                 -> opportunity cost for every selection metric
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from metric_clinic_sim import CLASSIFICATION


@dataclass
class BusinessSpec:
    cost: float = 0.68           # cost of acting on one record (mail, call, offer); the mean if cost_cv > 0
    cost_cv: float = 0.0         # spread of the per-record cost (lognormal CV); 0 = every record costs `cost`
    top_pct: float = 0.10        # capacity: share of the population you can act on
    top_n: int = 1000            # capacity as a record count ("we can call 1,000 people")
    gain_target: float = 0.90    # "how deep to go to capture this share of the 1s"
    threshold: float = 0.50      # probability cutoff for accuracy/precision/recall/F1


@dataclass(frozen=True)
class Metric:
    key: str
    label: str                   # may contain {top}, {n}, {gain}, {thr}
    family: str                  # see FAMILIES
    higher_better: bool
    money: bool = False

    def name(self, biz: BusinessSpec) -> str:
        return self.label.format(top=f"{biz.top_pct:.0%}", n=f"{biz.top_n:,}",
                                 gain=f"{biz.gain_target:.0%}", thr=f"{biz.threshold:g}")


# Statistical metrics score every record; the rank-ordered families only care
# about the order of the list, and mostly about its top.
FAMILIES = ["statistical", "top-focused statistical", "top N / top %", "depth to capture",
            "business $"]


CLASSIFICATION_METRICS = [
    Metric("auc", "ROC AUC", "statistical", True),
    Metric("accuracy", "Accuracy @ p>{thr}", "statistical", True),
    Metric("precision", "Precision @ p>{thr}", "statistical", True),
    Metric("recall", "Recall @ p>{thr}", "statistical", True),
    Metric("f1", "F1 @ p>{thr}", "statistical", True),
    Metric("log_loss", "Log loss", "statistical", False),
    Metric("brier", "Brier score", "statistical", False),
    Metric("pauc", "Partial AUC (FPR ≤ 10%)", "top-focused statistical", True),
    Metric("lift_top", "Lift @ top {top}", "top N / top %", True),
    Metric("gain_top", "% of 1s captured @ top {top}", "top N / top %", True),
    Metric("value_top", "% of $ captured @ top {top}", "top N / top %", True),
    Metric("precision_topn", "Hit rate (precision) @ top {n}", "top N / top %", True),
    Metric("depth_gain", "Depth to capture {gain} of 1s", "depth to capture", False),
    Metric("profit_top", "Profit acting on top {top}", "business $", True, True),
    Metric("profit_topn", "Profit acting on top {n}", "business $", True, True),
    Metric("roi_top", "ROI acting on top {top}", "business $", True),
    Metric("profit_max", "Max profit (best depth)", "business $", True, True),
    Metric("profit_rule", "Profit acting where p x avg value > cost", "business $", True, True),
]

REGRESSION_METRICS = [
    Metric("r2", "R²", "statistical", True),
    Metric("rmse", "RMSE", "statistical", False),
    Metric("mae", "MAE", "statistical", False),
    Metric("mean_error", "|Mean error| (bias)", "statistical", False),
    Metric("spearman", "Spearman rank corr", "top-focused statistical", True),
    Metric("value_top", "% of total value in top {top}", "top N / top %", True),
    Metric("value_topn", "% of total value in top {n}", "top N / top %", True),
    Metric("depth_gain", "Depth to capture {gain} of value", "depth to capture", False),
    Metric("profit_top", "Profit acting on top {top}", "business $", True, True),
    Metric("profit_topn", "Profit acting on top {n}", "business $", True, True),
    Metric("roi_top", "ROI acting on top {top}", "business $", True),
    Metric("profit_max", "Max profit (best depth)", "business $", True, True),
    Metric("profit_rule", "Profit acting where prediction > cost", "business $", True, True),
]


def metrics_for(task: str) -> list:
    return CLASSIFICATION_METRICS if task == CLASSIFICATION else REGRESSION_METRICS


def metric(task: str, key: str) -> Metric:
    return next(m for m in metrics_for(task) if m.key == key)


# ----------------------------------------------------------------------------
# Per-model evaluation
# ----------------------------------------------------------------------------

TIE_SEED = 20261023


def rank_order(score, seed=TIE_SEED):
    """Indices from highest to lowest score; ties broken by a fixed random permutation.

    Trees and rule models give thousands of records the same score. Breaking
    those ties by row order (what a plain sort does) lets data order leak into
    every top-N / depth metric.
    """
    perm = np.random.default_rng(seed).permutation(len(score))
    return perm[np.argsort(-np.asarray(score)[perm], kind="stable")]


def _ranks_from_order(order):
    ranks = np.empty(len(order))
    ranks[order] = np.arange(len(order), 0, -1)
    return ranks


def _depth_to_capture(cum, total, target):
    idx = np.searchsorted(cum, target * total - 1e-9)
    return (min(idx, len(cum) - 1) + 1) / len(cum)


COST_SEED = 2718  # fixed so every model (and every rerun) faces the same per-record costs


def record_costs(biz: BusinessSpec, n: int) -> np.ndarray:
    """Cost of acting on each record. Fixed at biz.cost, or lognormal around it."""
    if biz.cost_cv <= 0 or biz.cost <= 0:
        return np.full(n, float(biz.cost))
    sigma2 = np.log1p(biz.cost_cv ** 2)
    rng = np.random.default_rng(COST_SEED + n)
    return rng.lognormal(np.log(biz.cost) - sigma2 / 2, np.sqrt(sigma2), n)


def _roi(profit, spent):
    return profit / spent if spent > 0 else np.nan


def _classification_row(y, p, payoff, gift, biz, avg_value, cost):
    n, n_pos = len(y), y.sum()
    order = rank_order(p)
    k = max(1, int(np.ceil(biz.top_pct * n)))
    kn = int(np.clip(biz.top_n, 1, n))

    sorted_p = p[order]
    if np.any(sorted_p[1:] == sorted_p[:-1]):
        ranks = stats.rankdata(p)          # tied scores share their average rank
    else:
        ranks = _ranks_from_order(order)   # ascending rank, 1 = lowest score
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (n - n_pos))

    pred = p > biz.threshold
    tp = np.sum(pred & (y == 1))
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / n_pos
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    pc = np.clip(p, 1e-12, 1 - 1e-12)

    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    cum_gift = np.cumsum(gift[order])
    profit_curve = np.cumsum(payoff[order])
    spent = np.cumsum(cost[order])
    act = p * avg_value > cost
    # partial AUC: mean true-positive rate over the first 10% of negatives, scaled to 0..1
    tpr_at_neg = cum_pos[y_sorted == 0] / n_pos
    m = max(1, int(0.10 * len(tpr_at_neg)))
    return {
        "auc": auc,
        "accuracy": np.mean(pred == (y == 1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
        "brier": np.mean((p - y) ** 2),
        "pauc": tpr_at_neg[:m].mean(),
        "lift_top": (cum_pos[k - 1] / k) / (n_pos / n),
        "gain_top": cum_pos[k - 1] / n_pos,
        "value_top": cum_gift[k - 1] / cum_gift[-1],
        "precision_topn": cum_pos[kn - 1] / kn,
        "depth_gain": _depth_to_capture(cum_pos, n_pos, biz.gain_target),
        "profit_top": profit_curve[k - 1],
        "profit_topn": profit_curve[kn - 1],
        "roi_top": _roi(profit_curve[k - 1], spent[k - 1]),
        "profit_max": max(0.0, profit_curve.max()),
        "best_depth": (np.argmax(profit_curve) + 1) / n if profit_curve.max() > 0 else 0.0,
        "best_n": int(np.argmax(profit_curve) + 1) if profit_curve.max() > 0 else 0,
        "profit_rule": payoff[act].sum(),
        "share_acted_rule": act.mean(),
    }


def _regression_row(y, y_rank, pred, biz, cost):
    n = len(y)
    order = rank_order(pred)
    k = max(1, int(np.ceil(biz.top_pct * n)))
    kn = int(np.clip(biz.top_n, 1, n))
    err = pred - y
    y_sorted = y[order]
    cum_val = np.cumsum(y_sorted)
    profit_curve = np.cumsum(y_sorted - cost[order])
    spent = np.cumsum(cost[order])
    act = pred > cost
    total = y.sum()
    return {
        "r2": 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
        "rmse": np.sqrt(np.mean(err ** 2)),
        "mae": np.mean(np.abs(err)),
        "mean_error": abs(err.mean()),
        "spearman": np.corrcoef(y_rank, stats.rankdata(pred))[0, 1],
        "value_top": cum_val[k - 1] / total if total else np.nan,
        "value_topn": cum_val[kn - 1] / total if total else np.nan,
        "depth_gain": _depth_to_capture(cum_val, total, biz.gain_target) if total > 0 else np.nan,
        "profit_top": profit_curve[k - 1],
        "profit_topn": profit_curve[kn - 1],
        "roi_top": _roi(profit_curve[k - 1], spent[k - 1]),
        "profit_max": max(0.0, profit_curve.max()),
        "best_depth": (np.argmax(profit_curve) + 1) / n if profit_curve.max() > 0 else 0.0,
        "best_n": int(np.argmax(profit_curve) + 1) if profit_curve.max() > 0 else 0,
        "profit_rule": (y[act] - cost[act]).sum(),
        "share_acted_rule": act.mean(),
    }


def settle_ties(table: pd.DataFrame, digits: int = 10) -> pd.DataFrame:
    """Round each column to `digits` significant figures of its largest value, so
    floating-point noise (the same sum added in a different order) can't split true
    ties into ranks 1..50. E.g. every model's max profit is "mail everyone"."""
    out = table.copy()
    for c in out.columns:
        col = out[c]
        if not pd.api.types.is_float_dtype(col):
            continue
        top = np.nanmax(np.abs(col.to_numpy())) if col.notna().any() else 0.0
        if top > 0 and np.isfinite(top):
            out[c] = col.round(int(digits - 1 - np.floor(np.log10(top))))
    return out


def evaluate(task, y, scores, names, biz: BusinessSpec, value=None, params=None) -> pd.DataFrame:
    """One row per model: knob values (if given) then every metric."""
    rows = []
    if task == CLASSIFICATION:
        y = np.asarray(y).astype(int)
        value = np.ones(len(y)) if value is None else np.asarray(value, float)
        cost = record_costs(biz, len(y))
        gift = y * value
        payoff = gift - cost
        avg_value = value[y == 1].mean()
        for s in scores:
            rows.append(_classification_row(y, s, payoff, gift, biz, avg_value, cost))
    else:
        y = np.asarray(y, float)
        y_rank = stats.rankdata(y)
        cost = record_costs(biz, len(y))
        for s in scores:
            rows.append(_regression_row(y, y_rank, s, biz, cost))
    table = settle_ties(pd.DataFrame(rows, index=pd.Index(names, name="model")))
    if params is not None:
        table = pd.concat([pd.DataFrame(params, index=table.index).add_prefix("knob_"), table], axis=1)
    return table


# ----------------------------------------------------------------------------
# Selection and opportunity cost
# ----------------------------------------------------------------------------

def rank(table, task, key) -> pd.Series:
    m = metric(task, key)
    return table[key].rank(ascending=not m.higher_better, method="min", na_option="bottom").astype(int)


def best(table, task, key) -> str:
    m = metric(task, key)
    col = table[key]
    if col.isna().all():                 # e.g. ROI when nothing is spent
        return table.index[0]
    return col.idxmax() if m.higher_better else col.idxmin()


def choose(table, task, select_by, judge_by) -> dict:
    """The model the statistical metric picks vs. the model the business wants."""
    picked, ideal = best(table, task, select_by), best(table, task, judge_by)
    judge = metric(task, judge_by)
    gap = table.at[ideal, judge_by] - table.at[picked, judge_by]
    return {
        "picked": picked,
        "ideal": ideal,
        "picked_value": table.at[picked, judge_by],
        "ideal_value": table.at[ideal, judge_by],
        "opportunity_cost": gap if judge.higher_better else -gap,
        "picked_business_rank": int(rank(table, task, judge_by)[picked]),
        "ideal_select_rank": int(rank(table, task, select_by)[ideal]),
        "rank_corr": stats.spearmanr(rank(table, task, select_by), rank(table, task, judge_by)).statistic,
    }


def regret_by_metric(table, task, judge_by) -> pd.DataFrame:
    """For every metric you might select by: which model it picks and what that costs."""
    rows = []
    for m in metrics_for(task):
        if m.key not in table or table[m.key].isna().all():
            continue
        c = choose(table, task, m.key, judge_by)
        rows.append({"key": m.key, "family": m.family, "picks": c["picked"],
                     "business_value": c["picked_value"], "opportunity_cost": c["opportunity_cost"],
                     "business_rank": c["picked_business_rank"]})
    return pd.DataFrame(rows)


def agreement_matrix(table, task) -> pd.DataFrame:
    """Spearman correlation between metrics across models, signed so + means 'agree'."""
    cols = [m for m in metrics_for(task) if m.key in table]
    oriented = pd.DataFrame({m.key: table[m.key] * (1 if m.higher_better else -1) for m in cols})
    return oriented.corr(method="spearman")


# ----------------------------------------------------------------------------
# Curves for the decile / gains / profit charts
# ----------------------------------------------------------------------------

def decile_table(task, y, score, value=None, cost=0.0, bins=10) -> pd.DataFrame:
    y = np.asarray(y, float)
    order = rank_order(score)
    groups = np.array_split(order, bins)
    c = np.broadcast_to(np.asarray(cost, float), len(y))
    rows = []
    for i, g in enumerate(groups, 1):
        row = {"decile": i, "n": len(g), "mean_score": score[g].mean(), "mean_actual": y[g].mean()}
        if task == CLASSIFICATION:
            v = np.ones(len(y)) if value is None else value
            row["lift"] = y[g].mean() / y.mean()
            row["profit"] = (y[g] * v[g] - c[g]).sum()
        else:
            row["profit"] = (y[g] - c[g]).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def cumulative_curves(task, y, score, value=None, cost=0.0, points=200) -> pd.DataFrame:
    """Depth vs. cumulative share of 1s/value captured and cumulative profit."""
    y = np.asarray(y, float)
    order = rank_order(score)
    c = np.broadcast_to(np.asarray(cost, float), len(y))[order]
    if task == CLASSIFICATION:
        v = np.ones(len(y)) if value is None else value
        gain = np.cumsum(y[order]) / y.sum()
        profit = np.cumsum(y[order] * v[order] - c)
    else:
        gain = np.cumsum(y[order]) / y.sum()
        profit = np.cumsum(y[order] - c)
    n = len(y)
    idx = np.unique(np.linspace(0, n - 1, points).astype(int))
    return pd.DataFrame({"depth": np.r_[0, (idx + 1) / n],
                         "gain": np.r_[0, gain[idx]],
                         "profit": np.r_[0, profit[idx]]})


def roc_points(y, score, points=200) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    order = rank_order(score)
    tp = np.cumsum(y[order])
    fp = np.arange(1, len(y) + 1) - tp
    idx = np.unique(np.linspace(0, len(y) - 1, points).astype(int))
    return pd.DataFrame({"fpr": np.r_[0, fp[idx] / fp[-1]], "tpr": np.r_[0, tp[idx] / tp[-1]]})
