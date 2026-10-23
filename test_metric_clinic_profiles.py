"""A profile extracted from one model reproduces that model's shape in a new population."""
import numpy as np
import pytest

from metric_clinic_metrics import BusinessSpec, decile_table, evaluate
from metric_clinic_profiles import extract_profiles, load_bank, profile_scores, save_bank
from metric_clinic_sim import CLASSIFICATION, PopulationSpec, _classification_scores, make_population

BIZ = BusinessSpec(cost=0.68, top_pct=0.10)


def _value_lift(y, value, score):
    """Mean gift of responders in the top decile / overall mean gift."""
    top = np.argsort(-score)[: len(y) // 10]
    return value[top][y[top] == 1].mean() / value[y == 1].mean()


@pytest.fixture(scope="module")
def source():
    pop = make_population(PopulationSpec(n=80_000, prevalence=0.05, value_cv=0.8, seed=1))
    rng = np.random.default_rng(3)
    shapes = {"top_heavy_big_gifts": dict(auc=0.65, concentration=0.8, value_tilt=0.7, calib_bias=0),
              "flat_small_gifts": dict(auc=0.65, concentration=0.0, value_tilt=-0.7, calib_bias=0)}
    scores = np.vstack([_classification_scores(pop, p, rng) for p in shapes.values()])
    return pop, list(shapes), scores


@pytest.fixture(scope="module")
def bank(source):
    pop, names, scores = source
    return extract_profiles(pop.y, pop.value, scores, names)


def test_round_trip_keeps_lift_and_gift_pattern(source, bank):
    pop, names, scores = source
    target = make_population(PopulationSpec(n=80_000, prevalence=0.05, value_cv=0.8, seed=99))
    rng = np.random.default_rng(5)
    for i, name in enumerate(names):
        sim = profile_scores(target, bank.joint[i], bank.calib[i], 1.0, 0.0, rng)
        real_d = decile_table(CLASSIFICATION, pop.y, scores[i])
        sim_d = decile_table(CLASSIFICATION, target.y, sim)
        assert np.abs(real_d.lift - sim_d.lift).max() < 0.35, name
        assert abs(_value_lift(pop.y, pop.value, scores[i])
                   - _value_lift(target.y, target.value, sim)) < 0.15, name
    # and the two shapes stay distinguishable after the round trip
    t = evaluate(CLASSIFICATION, target.y, [profile_scores(target, bank.joint[i], bank.calib[i], 1, 0, rng)
                                            for i in range(2)], names, BIZ, target.value)
    assert t.at["top_heavy_big_gifts", "lift_top"] > t.at["flat_small_gifts", "lift_top"]
    assert t.at["top_heavy_big_gifts", "profit_max"] > t.at["flat_small_gifts", "profit_max"]


def test_strength_zero_is_random(source, bank):
    pop, _, _ = source
    s = profile_scores(pop, bank.joint[0], bank.calib[0], 0.0, 0.0, np.random.default_rng(1))
    auc = evaluate(CLASSIFICATION, pop.y, [s], ["m"], BIZ, pop.value).at["m", "auc"]
    assert abs(auc - 0.5) < 0.02


def test_probabilities_follow_the_ranking(source, bank):
    pop, _, _ = source
    s = profile_scores(pop, bank.joint[0], bank.calib[0], 1.0, 0.5, np.random.default_rng(2))
    assert np.all((s > 0) & (s < 1))
    base = profile_scores(pop, bank.joint[0], bank.calib[0], 1.0, 0.0, np.random.default_rng(2))
    assert np.array_equal(np.argsort(s), np.argsort(base))    # calibration bias never reorders


def test_save_load(bank, tmp_path):
    save_bank(bank, tmp_path / "b.npz")
    back = load_bank(tmp_path / "b.npz")
    assert back.names == bank.names and np.allclose(back.joint, bank.joint)


# ---------------------------------------------------------------- regression

from metric_clinic_metrics import BusinessSpec as _B  # noqa: E402
from metric_clinic_profiles import extract_regression_profiles, profile_predictions  # noqa: E402
from metric_clinic_sim import REGRESSION, _regression_predictions  # noqa: E402

REG_BIZ = _B(cost=300, top_pct=0.2)


@pytest.mark.parametrize("shape,kw", [("lognormal", dict(mean=500, cv=1.5)),
                                      ("zero_inflated", dict(mean=15.6, cv=0.8, nonzero_rate=0.05))])
def test_regression_round_trip(shape, kw):
    spec = dict(task=REGRESSION, n=80_000, shape=shape, **kw)
    src = make_population(PopulationSpec(seed=1, **spec))
    rng = np.random.default_rng(4)
    shapes = {"precise_top": dict(corr=0.6, hetero=-0.9, shrink=1.0, bias=0.0),
              "noisy_top_biased": dict(corr=0.6, hetero=0.9, shrink=0.6, bias=0.2)}
    preds = np.vstack([_regression_predictions(src, p, rng) for p in shapes.values()])
    bank = extract_regression_profiles(src.y, preds, list(shapes))
    assert bank.task == "regression"

    tgt = make_population(PopulationSpec(seed=77, **spec))
    sim = np.vstack([profile_predictions(tgt, bank.joint[i], bank.calib[i], bank.level_z[i],
                                         1.0, 1.0, 0.0, rng) for i in range(2)])
    biz = _B(cost=kw["mean"] * 0.6, top_pct=0.2)
    real_t = evaluate(REGRESSION, src.y, preds, list(shapes), biz)
    sim_t = evaluate(REGRESSION, tgt.y, sim, list(shapes), biz)
    for name, i in zip(shapes, range(2)):
        real_d = decile_table(REGRESSION, src.y, preds[i]).mean_actual / src.y.mean()
        sim_d = decile_table(REGRESSION, tgt.y, sim[i]).mean_actual / tgt.y.mean()
        assert np.abs(real_d - sim_d).max() < 0.35 * real_d.max(), name
        assert abs(real_t.at[name, "value_top"] - sim_t.at[name, "value_top"]) < 0.05, name
        # the prediction level (bias) carries over, in units of the target mean
        assert abs(real_t.at[name, "mean_error"] / src.y.mean()
                   - sim_t.at[name, "mean_error"] / tgt.y.mean()) < 0.1, name
    # the two shapes keep their order on the top-20% value
    assert (real_t.value_top.idxmax() == sim_t.value_top.idxmax())


def test_regression_shrink_and_bias_keep_ranking():
    pop = make_population(PopulationSpec(task=REGRESSION, n=40_000, shape="lognormal", mean=500, cv=1.5))
    p = _regression_predictions(pop, dict(corr=0.6, hetero=0, shrink=1, bias=0), np.random.default_rng(1))
    bank = extract_regression_profiles(pop.y, [p], ["m"])
    a = profile_predictions(pop, bank.joint[0], bank.calib[0], bank.level_z[0], 1, 1.0, 0.0,
                            np.random.default_rng(9))
    b = profile_predictions(pop, bank.joint[0], bank.calib[0], bank.level_z[0], 1, 0.5, 0.3,
                            np.random.default_rng(9))
    assert np.array_equal(np.argsort(a), np.argsort(b))
