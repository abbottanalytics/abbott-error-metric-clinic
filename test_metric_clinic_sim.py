"""Each model knob moves the metrics it should and leaves the others alone."""
import numpy as np
import pytest

from metric_clinic_metrics import BusinessSpec, evaluate
from metric_clinic_sim import (CLASSIFICATION, REGRESSION, PopulationSpec, _classification_scores,
                               _regression_predictions, make_population)

BIZ = BusinessSpec(cost=0.68, top_pct=0.10)


@pytest.fixture(scope="module")
def cls_pop():
    return make_population(PopulationSpec(n=60_000, prevalence=0.05, value_mean=15.6, value_cv=0.8))


@pytest.fixture(scope="module")
def reg_pop():
    return make_population(PopulationSpec(task=REGRESSION, n=60_000, shape="lognormal",
                                          mean=500, cv=1.5))


def _cls(pop, **kw):
    p = dict(auc=0.65, concentration=0.0, value_tilt=0.0, calib_bias=0.0) | kw
    s = _classification_scores(pop, p, np.random.default_rng(7))
    return evaluate(CLASSIFICATION, pop.y, [s], ["m"], BIZ, pop.value).iloc[0]


def _reg(pop, **kw):
    p = dict(corr=0.6, hetero=0.0, shrink=1.0, bias=0.0) | kw
    s = _regression_predictions(pop, p, np.random.default_rng(7))
    return evaluate(REGRESSION, pop.y, [s], ["m"], BusinessSpec(cost=300, top_pct=0.2)).iloc[0]


def test_auc_hits_target(cls_pop):
    assert abs(_cls(cls_pop)["auc"] - 0.65) < 0.02


def test_concentration_raises_top_lift_not_auc(cls_pop):
    base, conc = _cls(cls_pop), _cls(cls_pop, concentration=1.0)
    assert abs(conc["auc"] - base["auc"]) < 0.02
    assert conc["lift_top"] > base["lift_top"] * 1.2
    assert conc["depth_gain"] > base["depth_gain"]


def test_value_tilt_moves_profit_not_auc(cls_pop):
    hi, lo = _cls(cls_pop, value_tilt=0.8), _cls(cls_pop, value_tilt=-0.8)
    assert abs(hi["auc"] - lo["auc"]) < 0.02
    assert hi["profit_max"] > lo["profit_max"] * 1.5


def test_calibration_bias_leaves_ranking_alone(cls_pop):
    base, shifted = _cls(cls_pop), _cls(cls_pop, calib_bias=2.0)
    for k in ("auc", "lift_top", "profit_max", "depth_gain"):
        assert base[k] == pytest.approx(shifted[k])
    assert shifted["log_loss"] > base["log_loss"]
    assert shifted["accuracy"] < base["accuracy"]


def test_calibrated_regression_r2_near_corr_squared():
    pop = make_population(PopulationSpec(task=REGRESSION, n=60_000, shape="normal", mean=100, sd=25))
    assert abs(_reg(pop)["r2"] - 0.36) < 0.03


def test_hetero_moves_top_value_not_rmse(reg_pop):
    base, precise_top = _reg(reg_pop), _reg(reg_pop, hetero=-0.9)
    assert abs(precise_top["rmse"] / base["rmse"] - 1) < 0.05
    assert precise_top["profit_top"] > base["profit_top"] * 1.05


def test_shrink_and_bias_leave_ranking_alone(reg_pop):
    base = _reg(reg_pop)
    for kw in (dict(shrink=0.5), dict(bias=0.3)):
        other = _reg(reg_pop, **kw)
        assert other["spearman"] == pytest.approx(base["spearman"], abs=1e-9)
        assert other["profit_top"] == pytest.approx(base["profit_top"])
        assert other["rmse"] > base["rmse"]


def test_raw_and_log_models_matched_on_ranking():
    """Mixed scale: half the models are built on raw $ and half on log(1+$),
    with about the same overall rank correlation."""
    from dataclasses import replace as _replace
    import numpy as _np
    from metric_clinic_sim import ModelSpec, _spearman, make_models
    pop = make_population(PopulationSpec(task=REGRESSION, n=20_000, shape="lognormal", mean=500, cv=1.5))
    ms = make_models(pop, ModelSpec(n_models=6, seed=1, corr=(0.6, 0.6), scale="mixed"))
    scales = [p["scale"] for p in ms.params]
    assert scales == ["log", "raw"] * 3
    sp = _np.array([_spearman(s, pop.y) for s in ms.scores])
    assert abs(sp[0::2].mean() - sp[1::2].mean()) < 0.03
