"""Simulated populations and model predictions for the Model Selection Clinic.

Two layers, kept separate so the population can be frozen while model
behavior changes:

    make_population(spec)         -> Population   (the records and their truth)
    make_models(population, spec) -> ModelSet     (scores/predictions per model)

Every model knob is built to move one family of metrics while leaving another
roughly fixed. That is what produces ranking reversals:

Classification knobs (per model, drawn uniformly from a range)
    auc            overall ranking strength (sets the binormal separation)
    concentration  0 = signal spread evenly over the positives; 1 = signal
                   piled onto ~25% "easy" positives, the rest barely separated.
                   AUC held roughly constant -> lift at the top goes UP,
                   depth needed to capture 90% of 1s goes UP (worse).
    value_tilt     >0 = model ranks high-value positives higher; <0 = it finds
                   the low-value positives. AUC unchanged, profit changes.
    calib_bias     log-odds shift on the probabilities. Ranking unchanged ->
                   AUC/lift/gain identical; accuracy/precision/recall/log loss
                   and any rule that thresholds the probability change.

Regression knobs (applied on the modeling scale: log1p for skewed targets)
    corr           correlation of prediction with actual (ranking strength)
    hetero         >0 = errors grow with the actual (noisy on high values);
                   <0 = precise on high values. Total error variance held
                   fixed, so RMSE barely moves while top-decile value does.
    shrink         1 = calibrated. <1 squashes predictions toward the mean
                   (under-predicts high values); >1 over-spreads them.
                   Ranking unchanged.
    bias           additive offset in SDs of the target. Ranking unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import stats

CLASSIFICATION = "classification"
REGRESSION = "regression"

EASY_FRACTION = 0.25      # share of positives that get the boosted separation
MAX_EASY_BOOST = 3.0      # extra separation for easy positives at concentration=1


# ----------------------------------------------------------------------------
# Population
# ----------------------------------------------------------------------------

@dataclass
class PopulationSpec:
    task: str = CLASSIFICATION
    n: int = 100_000
    target_name: str = "responded"
    seed: int = 0
    # classification
    prevalence: float = 0.05
    value_mean: float = 15.0     # mean $ value of a positive outcome
    value_cv: float = 0.8        # 0 -> every positive worth value_mean
    # regression
    shape: str = "lognormal"     # normal | lognormal | zero_inflated
    mean: float = 100.0          # mean of target (of the non-zero part if zero_inflated)
    sd: float = 25.0             # normal only
    cv: float = 1.0              # lognormal / zero_inflated spread
    nonzero_rate: float = 0.10   # zero_inflated only


@dataclass
class Population:
    spec: PopulationSpec
    y: np.ndarray                # 0/1 (classification) or the continuous target
    value: np.ndarray | None     # classification: $ value if the record is a 1
    rank_z: np.ndarray           # normal scores used to place asymmetries

    @property
    def task(self) -> str:
        return self.spec.task

    @property
    def n(self) -> int:
        return len(self.y)


def _lognormal(rng, mean, cv, size):
    sigma2 = np.log1p(cv ** 2)
    mu = np.log(mean) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2), size)


def _normal_scores(x):
    r = stats.rankdata(x)
    z = stats.norm.ppf((r - 0.5) / len(x))
    return (z - z.mean()) / (z.std() or 1.0)


def make_population(spec: PopulationSpec) -> Population:
    rng = np.random.default_rng(spec.seed)
    n = int(spec.n)
    if spec.task == CLASSIFICATION:
        y = (rng.random(n) < spec.prevalence).astype(np.int8)
        if spec.value_cv > 0:
            value = _lognormal(rng, spec.value_mean, spec.value_cv, n)
        else:
            value = np.full(n, float(spec.value_mean))
        rank_z = _normal_scores(np.log(value)) if spec.value_cv > 0 else np.zeros(n)
        return Population(spec, y, value, rank_z)

    if spec.shape == "normal":
        y = rng.normal(spec.mean, spec.sd, n)
    elif spec.shape == "lognormal":
        y = _lognormal(rng, spec.mean, spec.cv, n)
    elif spec.shape == "zero_inflated":
        y = np.where(rng.random(n) < spec.nonzero_rate,
                     _lognormal(rng, spec.mean, spec.cv, n), 0.0)
    else:
        raise ValueError(f"unknown shape {spec.shape!r}")
    return Population(spec, y, None, _normal_scores(y))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

@dataclass
class ModelSpec:
    n_models: int = 50
    seed: int = 0
    # each knob is a (low, high) range; models draw uniformly inside it
    auc: tuple = (0.60, 0.70)
    concentration: tuple = (0.0, 0.0)
    value_tilt: tuple = (0.0, 0.0)
    calib_bias: tuple = (0.0, 0.0)
    corr: tuple = (0.5, 0.7)
    hetero: tuple = (0.0, 0.0)
    shrink: tuple = (1.0, 1.0)
    bias: tuple = (0.0, 0.0)
    # regression, skewed targets: the scale each model is built on --
    # "log" (log1p, back-transformed), "raw" (dollars), or "mixed" (half of each)
    scale: str = "log"
    # empirical mode: borrow real models' decile-by-decile shapes (classification)
    profile_path: str = ""       # ProfileBank .npz (classification or regression); "" = parametric
    profile_algos: tuple = ()    # restrict to these algorithms; () = all
    strength: tuple = (1.0, 1.0)

    def knobs(self, task):
        if self.profile_path and task == CLASSIFICATION:
            return {"strength": self.strength, "calib_bias": self.calib_bias}
        if self.profile_path:
            return {"strength": self.strength, "shrink": self.shrink, "bias": self.bias}
        if task == CLASSIFICATION:
            return {"auc": self.auc, "concentration": self.concentration,
                    "value_tilt": self.value_tilt, "calib_bias": self.calib_bias}
        return {"corr": self.corr, "hetero": self.hetero,
                "shrink": self.shrink, "bias": self.bias}


@dataclass
class ModelSet:
    names: list
    params: list                 # one dict of knob values per model
    scores: np.ndarray           # (n_models, n): probability or prediction
    extra: dict = field(default_factory=dict)


def draw_params(spec: ModelSpec, task: str, rng) -> list:
    knobs = spec.knobs(task)
    out = []
    for _ in range(spec.n_models):
        out.append({k: float(rng.uniform(lo, hi)) if hi > lo else float(lo)
                    for k, (lo, hi) in knobs.items()})
    return out


def _separations(auc, concentration):
    """Binormal shifts (easy, hard) that keep the mixture AUC near `auc`."""
    a0 = np.sqrt(2) * stats.norm.ppf(np.clip(auc, 0.5001, 0.9999))
    a_easy = a0 + concentration * MAX_EASY_BOOST
    hard_auc = (auc - EASY_FRACTION * stats.norm.cdf(a_easy / np.sqrt(2))) / (1 - EASY_FRACTION)
    a_hard = np.sqrt(2) * stats.norm.ppf(np.clip(hard_auc, 0.5, 0.9999))
    return a_easy, a_hard


def _platt(s, y, iters=25):
    """Two-parameter logistic fit of y on s (Newton); returns (intercept, slope)."""
    X = np.column_stack([np.ones_like(s), s])
    b = np.array([np.log(y.mean() / (1 - y.mean())), 0.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def _classification_scores(pop: Population, p: dict, rng) -> np.ndarray:
    y = pop.y.astype(bool)
    a_easy, a_hard = _separations(p["auc"], p["concentration"])
    easy = rng.random(pop.n) < EASY_FRACTION
    shift = np.where(easy, a_easy, a_hard) + p["value_tilt"] * pop.rank_z
    s = rng.standard_normal(pop.n) + np.where(y, shift, 0.0)
    # calibrate on a subsample, then apply the model's miscalibration
    idx = rng.choice(pop.n, size=min(pop.n, 10_000), replace=False)
    b0, b1 = _platt(s[idx], pop.y[idx].astype(float))
    logit = b0 + b1 * s + p["calib_bias"]
    return 1 / (1 + np.exp(-logit))


def _spearman(a, b):
    return np.corrcoef(stats.rankdata(a), stats.rankdata(b))[0, 1]


def _raw_predictions(pop: Population, p: dict, rng) -> np.ndarray:
    """A model built on raw dollars, matched on overall ranking quality (Spearman)
    to the log-scale model with the same knobs, so the two differ only in where
    their errors fall: raw-scale noise is constant in dollars, so it swamps the
    small values and barely touches the big ones."""
    y = pop.y
    target = _spearman(_regression_predictions(pop, {**p, "scale": "log"}, rng), y)
    m, sd = y.mean(), y.std()
    pct = stats.norm.cdf(pop.rank_z)
    w = np.clip(1 + p["hetero"] * (2 * pct - 1), 0.05, None)
    w /= np.sqrt(np.mean(w ** 2))
    z = w * rng.standard_normal(pop.n)
    sub = rng.choice(pop.n, size=min(pop.n, 20_000), replace=False)
    lo, hi = -4.0, 4.0                       # search log10(noise / sd)
    for _ in range(30):
        mid = (lo + hi) / 2
        if _spearman(y[sub] + 10 ** mid * sd * z[sub], y[sub]) > target:
            lo = mid
        else:
            hi = mid
    k = 10 ** ((lo + hi) / 2)
    pred = m + p["shrink"] * (y - m + k * sd * z) / (1 + k ** 2)
    return pred + p["bias"] * sd


def _regression_predictions(pop: Population, p: dict, rng) -> np.ndarray:
    """Predictions built on the modeling scale (log1p for skewed targets).

    shrink=1 is a calibrated model (E[actual | prediction] ~ prediction on the
    modeling scale), which means its slope on the actual is corr**2.
    """
    if pop.spec.shape != "normal" and p.get("scale", "log") == "raw":
        return _raw_predictions(pop, p, rng)
    y = pop.y
    skewed = pop.spec.shape != "normal"
    t = np.log1p(y) if skewed else y
    m, sd = t.mean(), t.std()
    rho = np.clip(p["corr"], 0.01, 0.999)
    # error scale runs linearly from (1 - hetero) at the lowest actual to
    # (1 + hetero) at the highest, renormalized so total error variance is fixed
    pct = stats.norm.cdf(pop.rank_z)
    w = np.clip(1 + p["hetero"] * (2 * pct - 1), 0.05, None)
    w /= np.sqrt(np.mean(w ** 2))
    noise_sd = sd * np.sqrt(1 - rho ** 2) / rho
    pred_t = m + p["shrink"] * rho ** 2 * (t - m + noise_sd * w * rng.standard_normal(pop.n))
    if skewed:
        pred = np.expm1(pred_t)
        pred *= y.mean() / pred.mean()          # mean-calibrate the back-transform
    else:
        pred = pred_t
    return pred + p["bias"] * y.std()


@lru_cache(maxsize=4)
def _bank(path):
    from metric_clinic_profiles import load_bank
    return load_bank(path)


def _profile_models(pop: Population, spec: ModelSpec, rng) -> ModelSet:
    from metric_clinic_profiles import profile_predictions, profile_scores
    bank = _bank(spec.profile_path)
    if bank.task != pop.task:
        raise ValueError(f"{spec.profile_path} holds {bank.task} profiles; population is {pop.task}")
    pool = [i for i, n in enumerate(bank.names)
            if not spec.profile_algos or bank.summary.at[n, "algo"] in spec.profile_algos]
    chosen = rng.choice(pool, size=spec.n_models, replace=spec.n_models > len(pool))
    params, scores = [], []
    for p, i in zip(draw_params(spec, pop.task, rng), chosen):
        p = {"profile": bank.names[i], **p}
        if "algo" in bank.summary:
            p["algo"] = bank.summary.at[bank.names[i], "algo"]
        params.append(p)
        if pop.task == CLASSIFICATION:
            scores.append(profile_scores(pop, bank.joint[i], bank.calib[i], p["strength"],
                                         p["calib_bias"], rng))
        else:
            scores.append(profile_predictions(pop, bank.joint[i], bank.calib[i], bank.level_z[i],
                                              p["strength"], p["shrink"], p["bias"], rng))
    width = len(str(spec.n_models))
    return ModelSet([f"M{i + 1:0{width}d}" for i in range(spec.n_models)], params, np.vstack(scores))


def make_models(pop: Population, spec: ModelSpec) -> ModelSet:
    rng = np.random.default_rng(spec.seed + 10_007)
    if spec.profile_path:
        return _profile_models(pop, spec, rng)
    params = draw_params(spec, pop.task, rng)
    if pop.task != CLASSIFICATION and pop.spec.shape != "normal":
        for i, p in enumerate(params):
            p["scale"] = spec.scale if spec.scale != "mixed" else ("log", "raw")[i % 2]
    gen = _classification_scores if pop.task == CLASSIFICATION else _regression_predictions
    scores = np.vstack([gen(pop, p, rng) for p in params])
    width = len(str(spec.n_models))
    names = [f"M{i + 1:0{width}d}" for i in range(spec.n_models)]
    return ModelSet(names, params, scores)
