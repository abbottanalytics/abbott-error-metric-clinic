

def test_float_noise_ties_share_a_rank():
    # Every model's best depth is "act on everyone", so max profit is identical up to
    # summation-order noise; they must tie, and the opportunity cost must be exactly 0.
    import numpy as np
    from metric_clinic_metrics import BusinessSpec, evaluate, rank, choose
    rng = np.random.default_rng(1)
    y = rng.lognormal(2.5, 0.8, 20_000)              # everyone worth far more than the cost
    scores = [y + rng.normal(0, s, len(y)) for s in (1, 5, 20, 80)]
    t = evaluate("regression", y, scores, ["A", "B", "C", "D"], BusinessSpec(cost=0.68))
    assert t["profit_max"].nunique() == 1
    assert set(rank(t, "regression", "profit_max")) == {1}
    assert choose(t, "regression", "r2", "profit_max")["opportunity_cost"] == 0
