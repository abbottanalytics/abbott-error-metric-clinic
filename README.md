# Model Selection Clinic: are you optimizing the right metric?

This is the live simulator for the Machine Learning Week EU clinic. You turn an audience
member's problem into a simulated population and a set of candidate models. The
dashboard then shows which model each metric would pick and what the wrong pick costs.

```bash
pip install streamlit plotly numpy pandas scipy
streamlit run metric_clinic_app.py
```

## Files

| File | What it does |
|---|---|
| `metric_clinic_sim.py` | `make_population(PopulationSpec)` and `make_models(population, ModelSpec)` |
| `metric_clinic_metrics.py` | `evaluate(...)` gives one row per model with every metric. Also `choose`, `regret_by_metric`, and the curves |
| `metric_clinic_kdd98.py` | Reruns the 480-model TARGET_B grid from `kddcup1998/run_target_b_metriccorr_experiment.py` and saves each model's score on each held-out donor, plus the actual gift, to `data/kdd98_predictions.npz` |
| `metric_clinic_kdd98_target_d.py` | Two 480-model TARGET_D grids (responders only / all records) with saved predictions and profile banks |
| `metric_clinic_profiles.py` | Empirical model profiles: extract from real predictions, simulate from them (classification and regression) |
| `metric_clinic_presets.py` | The eight starting scenarios (three borrow real KDD Cup 98 model shapes). Edit them or add your own |
| `metric_clinic_app.py` | The Streamlit dashboard: three simulation tabs that follow the conversation, plus the real KDD Cup 98 tab |
| `test_metric_clinic_*.py` | Checks that each knob does what it claims, plus a render test for every preset |

`evaluate()` only needs arrays (actuals, a score matrix, optionally a $ value per record).
To use it on real predictions, pass in your own `y` and model scores. You don't need the simulator.

## The real response example (tab ④)

Generate the predictions once (takes a few minutes; needs the kddcup1998 prepared data):

```bash
python3 metric_clinic_kdd98.py
```

The tab runs the same comparison on 480 real models (120 logistic regressions and 360
decision trees) scored on the same 47,706 held-out donors. You can value a responder at
their **actual gift (TARGET_D)** or at the original experiment's **constant $14.50**.
Switching between the two shows how much model ranking depends on how you value a response.

## The real gift-amount example (tab ⑤)

The same comparison for 480 real TARGET_D regression models, in two scopes:
- **Every record (mostly zeros):** mail anyone whose predicted gift beats $0.68. Every model's
  R² is near 0, but profit runs from $0 to about $7.4k. Selecting by MAE picks a model that mails
  no one, because MAE rewards predicting the median gift, which is $0. |Mean error| tracks this
  decision best (ρ ≈ 0.84), since the rule depends on the level of the predictions.
- **Donors only (gift amount):** illustrative use is a $10 upgrade ask to the top 20%.

Model types (linear/tree × raw/log target) get their own marker on the scatter.

## Real model shapes in the simulator

Parametric knobs only produce the asymmetries someone designed. On the Models tab, choose
**"Borrowed from real KDD Cup 98 models"** (or the *Fundraising with real KDD Cup 98 model
shapes* preset) and each candidate gets the decile-by-decile shape of one real model.
`metric_clinic_profiles.py` stores each model as a profile:
- a 10×20 table: for each gift-size bucket, the share of responders that land in each score
  ventile. It holds both the lift curve and *which* donors the model finds.
- the model's own calibration in each ventile.

Regression presets can borrow real shapes too. `metric_clinic_kdd98_target_d.py` trains two
480-model TARGET_D grids on the same split:
- **responders**: gift amount among donors; used for skewed targets such as CLV.
- **all records**: TARGET_D for everyone, about 95% zeros; used for zero-inflated targets.

The grids vary ridge, lasso and trees, raw vs log1p target (log-model predictions are capped at
the training range), feature count, depth/leaf size, and the training sample. A regression
profile maps each bucket of the *actual* value to prediction ventiles. It also stores each
ventile's prediction level, so bias and shrinkage carry over. When a target has many zeros,
the zeros get a bucket of their own.

```bash
python3 metric_clinic_kdd98_target_d.py      # about 3 minutes, both grids + profiles
```

Simulating from a profile reproduces both in any population (the round-trip test checks this).
A strength control flattens the shape (<1) or sharpens it (>1). To add another dataset, call
`extract_profiles(y, value, scores, names)` and save the result with `save_bank`.

## Metric families

| Family | Metrics |
|---|---|
| Statistical (all records) | AUC, accuracy, precision, recall, F1 at a threshold, log loss, Brier (R², RMSE, MAE, bias for regression) |
| Top-focused statistical | partial AUC (FPR ≤ 10%); Spearman for regression |
| Top N / top % | lift, % of 1s captured, % of $ captured at top %; response rate at top N records |
| Depth to capture | depth needed to reach X% of the 1s (or of the value) |
| Business $ (fixed cost, variable gain) | profit at top %, profit at top N, ROI at top %, max profit over depth, profit from a threshold rule |

Tied scores, common with trees, are broken by a fixed random permutation, so a
dataset's row order can never influence a top-N result. AUC gives tied scores
their average rank, which matches sklearn.

## The model knobs

Each knob moves one family of metrics and holds another roughly fixed. That separation
is what makes rankings reverse.

| Knob | Moves | Leaves alone |
|---|---|---|
| **Classification** | | |
| AUC | everything | – |
| Concentration at the top | lift at top %, depth to capture 90% | AUC |
| Value tilt | profit | AUC, lift |
| Calibration bias | accuracy, precision, recall, F1, log loss, threshold-rule profit | AUC, lift, gains, profit at fixed depth |
| **Regression** (applied on the log scale for skewed targets) | | |
| Correlation | everything | – |
| Where errors fall | top-decile value and profit | RMSE |
| Spread (1 = calibrated) | RMSE, R², threshold-rule profit | ranking, top-% profit |
| Bias | RMSE, MAE, threshold-rule profit | ranking, top-% profit |

## Running the clinic

1. **Population tab.** Pick the closest preset, then edit the population to fit the audience member's case. When they agree it looks right, turn on **Freeze population**.
2. **Model tab.** Set the ranges for how the candidate models behave. Collapse a range to a single value to turn that behavior off and see whether the reversal goes away.
3. **Business tab.** Set the **Economics** (benefit per 1, cost per record, each with a spread; the breakeven hit rate is cost ÷ benefit). Then choose the metric they select models by now, and what the business actually values. Only the capacity/threshold settings those two metrics use are shown; the rest are in the comparison-settings panel. The headline shows the opportunity cost. The bar chart shows the cost of selecting by every other metric.
4. **"Is this a fluke?"** Rerun with fresh model draws. It reports how often the statistical pick was also the business winner, and the typical shortfall.

Each preset's model seed is the draw whose opportunity cost is at the median of 20 draws,
so the opening example is typical rather than cherry-picked.
