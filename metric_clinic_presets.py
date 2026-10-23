"""Starting scenarios for the clinic. Audience cases start from the closest one.

Each preset is (PopulationSpec, ModelSpec, BusinessSpec, select_by, judge_by,
talking point). Edit freely -- the dashboard reads this dict at start-up.

Model seeds are the draw whose opportunity cost sits at the median of 20
replicate draws, so the opening picture is typical, not cherry-picked.
"""

from metric_clinic_kdd98 import DEFAULT_PROFILES as KDD98_PROFILES
from metric_clinic_kdd98_target_d import profiles_path as target_d_profiles
from metric_clinic_metrics import BusinessSpec
from metric_clinic_sim import CLASSIFICATION, REGRESSION, ModelSpec, PopulationSpec

PRESETS = {
    "Fundraising appeal (KDD Cup 98 style)": dict(
        pop=PopulationSpec(task=CLASSIFICATION, n=100_000, target_name="TARGET_B",
                           prevalence=0.051, value_mean=15.6, value_cv=0.8),
        models=ModelSpec(n_models=50, seed=6, auc=(0.60, 0.66), concentration=(0.0, 0.6),
                         value_tilt=(-0.6, 0.4), calib_bias=(-1.0, 1.0)),
        biz=BusinessSpec(cost=0.68, top_pct=0.50, gain_target=0.90, threshold=0.5),
        select_by="auc", judge_by="profit_max",
        note="5% respond, gifts are skewed, a mailing costs $0.68. Some models find "
             "frequent small donors, others find the big gifts -- AUC can't tell them apart.",
    ),
    "Fundraising with real KDD Cup 98 model shapes": dict(
        pop=PopulationSpec(task=CLASSIFICATION, n=100_000, target_name="TARGET_B",
                           prevalence=0.051, value_mean=15.6, value_cv=0.8),
        models=ModelSpec(n_models=50, seed=9, profile_path=KDD98_PROFILES,
                         strength=(0.8, 1.2), calib_bias=(0.0, 0.0)),
        biz=BusinessSpec(cost=0.68, top_pct=0.50, gain_target=0.90, threshold=0.05),
        select_by="auc", judge_by="profit_max",
        note="Each candidate borrows the decile-by-decile shape of one of 480 real KDD Cup 98 "
             "models: where it finds responders and which gift sizes it finds.",
    ),
    "Retention offer with limited capacity": dict(
        pop=PopulationSpec(task=CLASSIFICATION, n=100_000, target_name="churned",
                           prevalence=0.08, value_mean=250.0, value_cv=0.0),
        models=ModelSpec(n_models=50, seed=1, auc=(0.70, 0.78), concentration=(0.0, 1.0),
                         value_tilt=(0.0, 0.0), calib_bias=(-1.5, 1.5)),
        biz=BusinessSpec(cost=25.0, top_pct=0.05, top_n=2000, gain_target=0.90, threshold=0.5),
        select_by="auc", judge_by="profit_top",
        note="The team can only call 5% of customers. A model that nails the top "
             "few percent beats a model with better AUC across the whole list.",
    ),
    "Fraud review: catch 90% of fraud": dict(
        pop=PopulationSpec(task=CLASSIFICATION, n=100_000, target_name="fraud",
                           prevalence=0.01, value_mean=400.0, value_cv=1.2),
        models=ModelSpec(n_models=50, seed=3, auc=(0.85, 0.92), concentration=(0.0, 1.0),
                         value_tilt=(-0.4, 0.4), calib_bias=(-1.0, 1.0)),
        biz=BusinessSpec(cost=5.0, top_pct=0.01, top_n=500, gain_target=0.90, threshold=0.5),
        select_by="lift_top", judge_by="depth_gain",
        note="Lift in the top 1% and the review depth needed to catch 90% of fraud "
             "reward opposite model shapes.",
    ),
    "Customer lifetime value (regression)": dict(
        pop=PopulationSpec(task=REGRESSION, n=100_000, target_name="CLV", shape="lognormal",
                           mean=500.0, cv=1.5),
        models=ModelSpec(n_models=50, seed=9, corr=(0.55, 0.65), hetero=(-0.9, 0.9),
                         shrink=(0.5, 1.5), bias=(-0.3, 0.3)),
        biz=BusinessSpec(cost=300.0, top_pct=0.20, gain_target=0.80),
        select_by="rmse", judge_by="profit_top",
        note="RMSE is dominated by calibration and a few huge customers; the business "
             "only needs the top 20% ranked correctly.",
    ),
    "Customer lifetime value with real gift-model shapes": dict(
        pop=PopulationSpec(task=REGRESSION, n=100_000, target_name="CLV", shape="lognormal",
                           mean=500.0, cv=1.5),
        models=ModelSpec(n_models=50, seed=0, profile_path=target_d_profiles("responders"),
                         strength=(1.0, 1.0), shrink=(1.0, 1.0), bias=(0.0, 0.0)),
        biz=BusinessSpec(cost=300.0, top_pct=0.20, gain_target=0.80),
        select_by="rmse", judge_by="profit_top",
        note="Each candidate borrows the shape of one of 480 real models of donors' gift "
             "amounts (ridge, lasso, trees; raw or log target), rescaled to this CLV population.",
    ),
    "Donation amount (zero-inflated regression)": dict(
        pop=PopulationSpec(task=REGRESSION, n=100_000, target_name="TARGET_D",
                           shape="zero_inflated", mean=15.6, cv=0.8, nonzero_rate=0.051),
        models=ModelSpec(n_models=50, seed=2, corr=(0.55, 0.65), hetero=(-0.9, 0.9),
                         shrink=(0.5, 1.5), bias=(-0.2, 0.2)),
        biz=BusinessSpec(cost=0.68, top_pct=0.20, gain_target=0.80),
        select_by="r2", judge_by="profit_rule",
        note="Predicting the gift directly: mail anyone whose predicted gift beats "
             "$0.68. Bias and calibration now matter to the business.",
    ),
    "Donation amount with real KDD Cup 98 model shapes": dict(
        pop=PopulationSpec(task=REGRESSION, n=100_000, target_name="TARGET_D",
                           shape="zero_inflated", mean=15.6, cv=0.8, nonzero_rate=0.051),
        models=ModelSpec(n_models=50, seed=3, profile_path=target_d_profiles("all"),
                         strength=(1.0, 1.0), shrink=(1.0, 1.0), bias=(0.0, 0.0)),
        biz=BusinessSpec(cost=0.68, top_pct=0.20, gain_target=0.80),
        select_by="r2", judge_by="profit_rule",
        note="Candidates borrow the shapes of 480 real models that predict TARGET_D for every "
             "record. Mail anyone whose predicted gift beats $0.68.",
    ),
}
