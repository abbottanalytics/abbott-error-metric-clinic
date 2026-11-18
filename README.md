# Model Selection Clinic: are you optimizing the right metric?

An interactive simulator that shows **what it costs to pick a model by the wrong error metric**.

Data scientists build dozens of candidate models and deploy the one that scores best on a metric
such as AUC, accuracy, F1 or RMSE. The business, though, is paid for something else: profit from
the people it contacts, frauds caught by the 100 investigations it can afford, value in the top
of the list. This dashboard simulates a population and a set of candidate models that look like
*your* problem. For every metric it shows which model that metric would pick and how much business
value that pick leaves on the table: the **opportunity cost** of the metric.

It also runs the same comparison on **480 real models** built on the KDD Cup 1998 fundraising
data. With actual gift amounts, AUC and profit rank those models in nearly opposite order
(ρ ≈ −0.22). Give every donor the same $14.50 gift and they agree (ρ ≈ 0.96): it's the dollars,
not the models, that break AUC as a selector.

Created by **Dean Abbott, Abbott Analytics**, for the *Model Selection Clinic* at Machine Learning
Week EU 2026.

## Quick start

Python 3.10–3.12.

```bash
git clone https://github.com/abbottanalytics/abbott-error-metric-clinic.git
cd abbott-error-metric-clinic
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run metric_clinic_app.py
```

The dashboard opens at http://localhost:8501. No other data or accounts are needed: the real
KDD Cup 98 model results ship in `data/`.

No git? Use **Code → Download ZIP** on GitHub, unzip, and run the same commands from the
unzipped folder.

## How to use it

1. **① Population.** Pick the closest **scenario preset** in the sidebar, then edit the records
   to match your problem: classification or regression, base rate or target shape. When it looks
   right, turn on **Freeze population**.
2. **② Model behavior.** Set the ranges the candidate models are drawn from: ranking strength,
   concentration at the top, value tilt, calibration (or borrow real KDD Cup 98 model shapes).
3. **③ Business success.** Enter the **Economics** (what a 1 is worth, what acting costs, the
   capacity), the metric you select models by today, and what the business actually values. The
   headline gives the opportunity cost; the charts show the cost of selecting by *every* metric.
   **Is this a fluke?** reruns with fresh model draws.
4. **④ / ⑤ Real data.** The same comparison on 480 real response models and 480 real gift-amount
   models.

**[DOCUMENTATION.md](DOCUMENTATION.md)** describes every control, chart and table, and why
each one matters.

The sidebar's **Saved** list starts with four worked examples from the clinic: invoice fraud
with 100 investigations a month (counted, then with variable dollar benefit and cost) and an
email/CLV regression with no contact cost. Your own saved
and recent scenarios are kept in `data/clinic_state.json` on your machine.

## Use it on your own models

`evaluate()` only needs arrays, so you can run the same comparison on your own predictions
without the simulator:

```python
import numpy as np
from metric_clinic_metrics import BusinessSpec, evaluate, choose, regret_by_metric

# y: actuals (0/1 or numbers); scores: one row per model; value: $ per record (optional)
table = evaluate("classification", y, scores, model_names,
                 BusinessSpec(cost=0.68, top_pct=0.10, top_n=1000), value=gift_amounts)
print(choose(table, "classification", "auc", "profit_max"))      # pick vs best, and the gap
print(regret_by_metric(table, "classification", "profit_max"))   # the cost of every metric
```

## Files

| File | What it does |
|---|---|
| `metric_clinic_app.py` | The Streamlit dashboard |
| `metric_clinic_sim.py` | `make_population(PopulationSpec)` and `make_models(population, ModelSpec)` |
| `metric_clinic_metrics.py` | `evaluate(...)`: one row per model with every metric; also `choose`, `regret_by_metric`, curves |
| `metric_clinic_profiles.py` | Empirical model profiles: extract them from real predictions and simulate from them |
| `metric_clinic_presets.py` | The eight scenario presets. Edit them or add your own |
| `metric_clinic_saved.py` | Last session, recent and saved scenarios |
| `example_scenarios.json` | The worked examples loaded into **Saved** on first run |
| `metric_clinic_kdd98.py`, `metric_clinic_kdd98_target_d.py` | How the real-model files in `data/` were built (see below) |
| `data/*.npz` | Predictions of 480 TARGET_B and 2 × 480 TARGET_D models on held-out KDD Cup 98 records, plus their profile banks |
| `test_metric_clinic_*.py` | Tests: each knob does what it claims, round-trip of real profiles, a render test for every preset |

Run the tests with `pip install pytest` and then `python -m pytest`.

## How the simulation works

- **Population.** Classification: 1s at your base rate, each with a lognormal $ benefit (mean
  and spread you choose). Regression: a normal, lognormal or zero-inflated target.
- **Candidate models.** Each model draws its behavior at random from the ranges on ②. Each knob
  moves one family of metrics and holds another roughly fixed. That separation is what makes the
  rankings reverse:

  | Knob | Moves | Leaves alone |
  |---|---|---|
  | AUC / correlation | everything | – |
  | Concentration at the top | lift at top %, depth to capture | AUC |
  | Value tilt | profit, % of $ captured | AUC, lift |
  | Calibration bias | accuracy, precision, recall, F1, log loss, threshold-rule profit | AUC, lift, profit at a fixed depth |
  | Where errors fall (regression) | top-decile value and profit | RMSE |
  | Spread, bias (regression) | RMSE, R², MAE, threshold-rule profit | ranking, top-% profit |

- **Real shapes.** In "borrowed" mode each candidate takes the decile-by-decile shape of one real
  KDD Cup 98 model: which score ventiles its responders land in, which gift sizes it finds, and
  its calibration. Real models make trade-offs nobody designed.
- **Fair comparisons.** Every model faces the same per-record costs. Tied scores are broken by a
  fixed random permutation. Results that differ only by floating-point noise count as ties. Each
  preset's model seed is the median of 20 draws, so the opening example isn't cherry-picked.

## The real-model data

`data/` holds model *scores* and the held-out targets (TARGET_B, TARGET_D) for 47,706 records of
the KDD Cup 1998 learning set (Paralyzed Veterans of America donor data, from the
[UCI KDD Archive](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html)). No donor input fields
are included.
- Response models: 120 logistic regressions and 360 decision trees.
- Gift-amount models: ridge, lasso and regression trees, each fit on raw dollars or on
  log(1 + $), in two scopes (all records, donors only).

`metric_clinic_kdd98.py` and `metric_clinic_kdd98_target_d.py` document how they were built.
Retraining needs the author's data-preparation project, which isn't public. You don't need to
retrain to use anything in the dashboard. `--profiles-only` rebuilds the profile banks from the
shipped predictions.

## License and credit

- **Code:** MIT License (see [LICENSE](LICENSE)).
- **Documentation and content** (this README, DOCUMENTATION.md, the scenario designs and the
  simulation design they describe): [CC BY 4.0](LICENSE-docs.md). Reuse and adapt freely, with
  credit.

If you use the simulator, its scenarios or its findings in a talk, course, paper or product,
please credit:

> Model Selection Clinic simulator by Dean Abbott, Abbott Analytics,
> https://github.com/abbottanalytics/abbott-error-metric-clinic

GitHub's **Cite this repository** button (from [CITATION.cff](CITATION.cff)) gives a formatted
citation.
