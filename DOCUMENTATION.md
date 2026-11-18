# Model Selection Clinic: user guide

This guide describes every control, chart and table in the dashboard: what it does, what the
axes and columns mean, and **why it matters** when you choose a model.

Contents
- [The idea in one paragraph](#the-idea-in-one-paragraph)
- [Sidebar](#sidebar)
- [① Population](#-population)
- [② Model behavior](#-model-behavior)
- [③ Business success](#-business-success)
- [④ Real: who responds](#-real-who-responds)
- [⑤ Real: gift amount](#-real-gift-amount)
- [Metric reference](#metric-reference)
- [Scenario presets and example scenarios](#scenario-presets-and-example-scenarios)
- [Things to try](#things-to-try)

---

## The idea in one paragraph

Data scientists rarely build one model. They build dozens, then pick one by sorting the
candidates on an **error metric** (AUC, accuracy, F1, RMSE...). That metric decides which model
gets deployed. The business, though, is paid for something else: dollars from the people it
acts on, frauds caught by the investigations it can afford, value in the part of the list it
contacts. This dashboard simulates a population that looks like your problem and a set of
candidate models that behave the way real models do. For every metric it shows which model that
metric would pick, and how much **business value that pick leaves on the table** compared with
the model the business would have wanted. That shortfall is the **opportunity cost** of the metric.

The workflow follows the pages from left to right:
**① describe the records → ② describe how the candidate models behave → ③ say what the
business values and see the cost of each metric.** Pages ④ and ⑤ repeat the comparison on
480 *real* models built on the KDD Cup 1998 fundraising data.

---

## Sidebar

| Control | What it does | Why it matters |
|---|---|---|
| **Scenario preset** (dropdown) | Loads a complete starting scenario: population, model ranges, economics and the two metrics. The one-line note below it tells the business story. | Start from the closest preset and edit it. That is much faster than setting 30 controls from scratch, and each preset was built to show a specific failure mode. |
| **Freeze population** (toggle) | Locks the records (all ① controls are greyed out). | When you then change model behavior or economics, any change in the result is caused by that change alone and not by a new random population. |
| **New population** (button) | Draws a fresh random population with the same settings (disabled while frozen). | Shows whether a result depends on one lucky set of records. |
| **New model draws** (button) | Draws a fresh set of candidate models from the same ranges. | One draw of 50 models can be unusual. Use this, or the replicate check on ③, before trusting a result. |
| *population seed · model seed* (caption) | The random seeds behind the current population and models. | Two people with the same settings and seeds see exactly the same numbers. |
| 🕒 *Simulation results computed...* (caption) | When the current results were last computed. | Tells you whether you are looking at fresh numbers after a change. |
| ↩️ *Resumed your last session* (caption) | Appears when the app restored your previous settings at start-up. | Your work survives closing the browser. Pick a preset to start fresh. |
| **Save current settings as** (text box) + **Save** | Saves every setting under the name you type. | Keep scenarios you want to show again, for example your own problem. |
| **Recent** (dropdown) + **Load recent** | The last 10 distinct setting combinations you ran, newest first, each shown with time · target · rows · % 1s · select metric → judge metric. | Go back to something you looked at a few minutes ago without having to save it. |
| **Saved** (dropdown) + **Load** / **Delete** | Your named scenarios. On first run this list holds the bundled [example scenarios](#scenario-presets-and-example-scenarios). | Worked examples to start from; delete them when you no longer need them. |

Saved settings live in `data/clinic_state.json` (not part of the repository). Delete that file
to get the example scenarios back.

---

## ① Population

Describes the records the models will score, before any model exists. Inputs are on the left;
a summary of the population you described is on the right.

### Inputs

| Control | Applies to | What it does | Why it matters |
|---|---|---|---|
| **Problem type** (classification / regression) | both | Classification: the target is 1/0 (responded, churned, fraud). Regression: the target is a number (gift amount, CLV). | It sets which metrics are available. Classification models are judged on how they rank 1s; regression models on predicted numbers. |
| **Target name** (text) | both | Label used in titles and captions ("fraud", "TARGET_B"). | Makes the screens read in the audience's own words. Changes nothing in the math. |
| **Population size** | both | Number of records, 1,000 to 1,000,000. | Rare targets need more records to contain enough 1s. A million records takes about 25 s per change, so 100,000 is the practical size for live use. |
| **% of records that are 1s** | classification | Base rate of the target, 0.1% to 60%. | Rare targets are where accuracy, the 0.5 threshold and F1 fail most visibly, and where only the very top of the list matters. |
| **Shape of the target** | regression | *Normal* (symmetric), *Right-skewed (lognormal)*, or *Mostly zeros + skewed non-zeros* (zero-inflated). | Shape drives everything in regression. Skew lets a few huge values dominate RMSE; zeros make the median 0, so MAE rewards predicting 0 for everyone. |
| **Mean** / **Mean of the non-zero values** | regression | Average of the target (of the non-zero part when zero-inflated). | Sets the dollar scale and, together with cost on ③, whether acting on a record pays at all. |
| **Standard deviation** | normal target | Spread of a normal target. | How much the values differ, and so how much ranking them is worth. |
| **Skew (coefficient of variation)** | lognormal / zero-inflated | SD ÷ mean of the (non-zero) values, 0.1 to 4. | Higher skew means a few records hold most of the value. That is when "find the big ones" and "be accurate on average" stop agreeing. |
| **% of records that are non-zero** | zero-inflated | Share of records with a positive value. | With 5% non-zero, a model that predicts 0 everywhere looks fine on MAE yet makes no money. |

### Outputs

**Classification: four tiles.** *Records*; *[target] = 1* (count of 1s); *Rate of 1s*;
*Records per [target] = 1* (how many records you'd act on, at random, to find one 1). The info box
reminds you that dollar values are set under **Economics** on ③, because what a 1 is worth is a
business assumption, not a property of the data.

**Regression: four tiles.** *Records*, *Mean*, *Median*, *Share that is zero*. A big gap between
mean and median is the first sign that statistical and business metrics will disagree.

**Distribution of [target]** (histogram, regression only). x-axis: target value; y-axis: number of
records. For zero-inflated targets only the non-zero records are drawn; the top 0.5% are left
out so the bulk stays readable. *Why:* check that it looks like the audience member's real data
("is this your problem?") before going further.

**Summary row** (table, regression only). Columns: *SD*, *skewness*, *90th pct*, *99th pct*,
*max*, *top 10% of records hold* (share of the total value held by the highest 10% of records).
*Why:* the last column is the plainest statement of concentration. If the top 10% hold 60% of
the value, ranking the top correctly is worth far more than being close on average.

---

## ② Model behavior

Describes the **candidate models** (M01, M02, ...). Each model draws its behavior at random from
the ranges you set. A range is a two-handle slider: collapse both handles to one value to turn that
behavior off (every model gets the same value), then see whether a reversal disappears.

### Inputs common to both problem types

| Control | What it does | Why it matters |
|---|---|---|
| **Number of candidate models** (5–200) | How many models compete. | Real projects try many models. More candidates mean more chances for a metric to pick a model that wins on paper and loses in the business. |
| **How do the models behave?** (radio) | *Set by the controls below* (parametric knobs), or *Borrowed from real KDD Cup 98 models*. Borrowed is only offered when real-model shapes exist for this kind of target (classification, lognormal or zero-inflated regression). | Parametric knobs only create the trade-offs someone designed. Borrowed shapes carry the decile-by-decile quirks of real fitted models, which nobody designed. |

### Classification, parametric

| Control | What it moves | What it leaves alone | Why it matters |
|---|---|---|---|
| **Ranking strength (target AUC)** range | Overall ranking quality: every metric. | – | Sets how good the models are. If the range is narrow, the models are similar overall, and the question becomes *where* in the list they differ. |
| **Concentration of signal at the top** range | 0: every 1 is equally easy to find. 1: a quarter of the 1s are very easy to find and the rest are barely separated. AUC is held roughly fixed. | AUC | Two models with the same AUC can have very different top 1% or top 100. When capacity is small, concentration decides the winner and AUC can't see it. |
| **Value tilt** range (only when dollars matter) | +: the model ranks the high-value 1s higher. −: it finds the cheap 1s. | AUC, lift | AUC counts every 1 the same; the business doesn't. Tilt is why the best-AUC model can lose money in fundraising. |
| **Calibration bias (log-odds)** range | Shifts every probability up or down. | AUC, lift, gains, profit at a fixed depth | Ranking is unchanged, but accuracy, precision, recall, F1, log loss and threshold-rule profit all change. It shows which metrics measure *ranking* and which measure the *level* of the probabilities. |

### Regression, parametric

| Control | What it moves | What it leaves alone | Why it matters |
|---|---|---|---|
| **Models are built on** (radio; skewed targets only) | *log(1 + $)*: like a model fit on the log of the target and back-transformed; it tames big values and under-predicts them. *raw $*: like least squares on dollars; it chases big values and can predict below 0. *mixed*: half of each. Candidates are matched on rank correlation, so neither kind starts with an advantage. | – | Teams often log-transform skewed targets by habit. With *mixed*, the scatter on ③ marks each model's type, so you can see whether the log habit costs money. |
| **Ranking strength (correlation)** range | Overall ranking quality: every metric. | – | As for AUC above. |
| **Where errors fall** range (− precise on high values, + noisy on high values) | Where along the target the model makes its errors. Total error variance is held fixed. | RMSE (barely moves) | Two models with the same RMSE can be precise where the money is or noisy there. Top-decile value and profit see the difference; RMSE doesn't. |
| **Spread of predictions** range (1 = calibrated, <1 under-predicts high values) | Stretches or shrinks predictions around their mean. | ranking, top-% profit | Changes RMSE, R² and threshold-rule profit without changing who is at the top of the list. |
| **Bias (in SDs of the target)** range | Adds a constant to every prediction. | ranking, top-% profit | Matters whenever a decision compares the predicted number to a cost ("mail if predicted gift > $0.68"). |

### Borrowed from real KDD Cup 98 models

| Control | What it does | Why it matters |
|---|---|---|
| **Borrow shapes from** (multiselect) | Which kinds of real model to borrow from: *logistic regression* / *decision tree* (classification), *linear* / *tree* (regression). | Trees and linear models make different kinds of mistakes. Restricting to one shows how much of the effect comes from mixing algorithms. |
| **Strength relative to the real model** range (1 = as fitted) | Below 1 flattens each borrowed shape toward random; above 1 sharpens it. | Lets you test the same real trade-offs at a higher or lower overall accuracy. |
| **Calibration bias (log-odds, on top of the real model's own)** range (classification) | As in parametric mode, added to the real model's own calibration. | Same ranking, different probability levels. |
| **Spread of predictions** range (regression) | As in parametric mode, relative to the real model's own spread. Ranking unchanged. | |
| **Extra bias (in SDs of the target)** range (regression) | Added to the real model's own bias. Ranking unchanged. | |

### Outputs

**Models to inspect** (multiselect, up to 4). Defaults to the model the statistical metric picks
and the model the business would want (both are set on ③). Add others to compare. All charts on
this page use these models.

**[AUC or R²] across N candidate models** (histogram). x-axis: AUC (classification) or R²
(regression); y-axis: number of models. Vertical lines mark the inspected models.
*Why:* if the picked model and the business winner sit almost on top of each other, they're
statistically indistinguishable, yet they can be worth very different amounts.

**ROC curves** (classification). x-axis: false positive rate; y-axis: true positive rate; dotted
diagonal = random. The legend shows each model's AUC. *Why:* curves that cross mean one model is
better at the top of the list and the other further down. AUC averages over the whole curve and
hides that.

**Calibration by decile: mean predicted vs mean actual** (regression). One point per decile of
predictions. x-axis: mean predicted value; y-axis: mean actual value; dotted line = perfect
calibration. *Why:* points below the line mean the model over-predicts; a flat line means it
predicts a level without ranking. Calibration drives RMSE and any "predicted value > cost" rule,
but not a top-% decision.

**Performance by decile (1 = highest scores)** (grouped bars). x-axis: decile of the model's own
scores (1 = top 10%); y-axis: *lift* (rate of 1s in the decile ÷ overall rate) for classification,
or *mean actual [target]* for regression. Hover shows the decile's profit and mean score.
*Why:* this is where models differ in *shape*. One may win decile 1 and lose deciles 2–4. Whether
that matters depends on how deep the business goes.

**Real shapes being borrowed** (profile mode only). Faint grey lines: every real model in the
pool; bold lines: the real model each inspected candidate borrowed (legend: "M07 ← real model
TR113"). x-axis: depth into the list (%) in 5% steps (ventiles); y-axis: lift (classification) or
value lift (regression); dotted line at 1 = random. *Why:* shows how varied real models are
beyond their headline accuracy.

**Settings table.** One row per inspected model, one column per knob it drew (auc, concentration,
value_tilt, calib_bias...; or profile, algo, strength in profile mode). *Why:* explains *why* a
model behaves as it does, for example "the business winner has high tilt, the AUC pick has none".

---

## ③ Business success

The comparison page. You say how models are chosen today and what the business actually values.
The page shows the cost of the difference.

### Inputs

| Control | What it does | Why it matters |
|---|---|---|
| **Metric you select models by** (dropdown) | The metric the team uses to pick a model today. All metrics for the problem type are listed with their family (statistical, top-focused statistical, top N / top %, depth to capture, business $). | This is the metric being put on trial. |
| **What the business actually values** (dropdown) | The yardstick the picked model is judged by. | This is the business goal. If both dropdowns hold the same metric, there is nothing to lose, and an info box says so. |

**Economics** (bordered box): what a 1 is worth and what acting costs, in one place.

| Control | Applies to | What it does | Why it matters |
|---|---|---|---|
| **Dollars matter** (checkbox) | classification | Off: the business only counts 1s (e.g. how many frauds land in the 100 cases you can investigate). Dollar metrics (profit, ROI, % of $) are hidden. | Some problems really are count-based. Turning dollars off keeps the comparison honest for them. |
| **Benefit per 1, avg ($)** | classification | What acting on a 1 brings in: a gift, a saved customer, money recovered from a fraudulent invoice. | Together with cost, this sets the breakeven hit rate. |
| **Benefit spread (CV)** | classification | How much the benefit varies between 1s (lognormal, SD ÷ average). 0 = every 1 worth the same; 0.8 = a few large values; 2+ = a handful of 1s hold most of the money. Changing it does not change who the 1s are. | Variable benefit is what breaks AUC as a selector: AUC counts a $5 donor and a $500 donor the same. |
| **Cost per record, avg ($)** | both | Mailing, calling, investigating: paid on every record you act on, 1 or not. | Cost turns "how many 1s" into "is it worth it". At zero cost, only how much value lands at the top matters. |
| **Cost spread (CV)** | both | Each record gets its own cost (lognormal around the average). 0 = every record costs the same; 0.5 = most cost between about half and double the average. The costs are fixed by a seed, so every model faces the same ones. | Real costs vary (a long investigation vs a quick one), and a model that finds 1s among cheap records wins. |
| *Breakeven hit rate* (readout) | classification | Cost ÷ benefit. | A record is worth acting on only if its chance of being a 1 beats this. If a model's hit rate within your capacity is below it, acting on all of those records loses money. |

**Capacity and threshold settings.** Only the settings the two chosen metrics use are shown
under the dropdowns. All the others are in the **⚙️ Settings for the comparison metrics**
expander, because the bar chart, Spearman chart and ranking table also score every model on every
other metric.

| Control | Used by | What it does | Why it matters |
|---|---|---|---|
| **Capacity: act on top %** | the "top %" metrics | Capacity as a share of the list. | "We can call 5% of customers." The winner at 5% is often not the winner at 50%. |
| **Capacity: act on top N records** | the "top N" metrics | Capacity as a count. | "Investigators can open 100 invoices a month." With a million records, only the extreme tail matters. |
| **Capture target %** | depth to capture | The share of the 1s (or of the value) you need to find. | "We must catch 90% of fraud." This rewards a very different model shape than top-1% lift does. |
| **Probability threshold** + **0.5** / **base rate** buttons | accuracy, precision, recall, F1 | Records scored above the threshold count as predicted 1s. *0.5* sets the usual default; *base rate* sets it to the share of 1s. | Most software uses 0.5 silently. At a 1% base rate almost nothing scores above 0.5, so accuracy and F1 judge only a tiny tail of the list. |

If neither chosen metric needs any of these, a caption says so.

### Headline

Three tiles:
- **Picked by [select metric]**: the model that metric chooses; underneath, its business value and
  its rank among all models on the business metric.
- **Best on [business metric]**: the model the business would want; underneath, its business value
  and its rank on the select metric.
- **Opportunity cost of the statistical pick**: the business value lost by deploying the first
  model instead of the second, with the share of the available value (for $ metrics) or the
  rank correlation between the two metrics across all models.

Captions add context where needed:
- *No capacity limit* (max profit): how many records each of the two models would act on before
  net benefit peaks.
- *Top N hits* (hit rate @ top N): the number of 1s each model puts in the top N, and how many
  fewer you get by selecting on your metric.
- *Every model ties*: when all models score the same on the business metric (for example every
  record is worth more than its cost, so every model's best depth is "everyone"), the choice of
  model can't matter. The note suggests what to change.

*Why it matters:* this is the clinic's answer, in the business's units: dollars, frauds
caught, value captured.

### Charts

**Every candidate model: statistical score vs business result** (scatter). One point per model.
x-axis: the select metric; y-axis: the business metric (axis titles say whether higher or lower is
better). Colors: the statistical pick, the business winner, both (if they're the same model),
and other candidates. Symbols mark model type where that varies (built on log/raw; real algorithm;
real model type). Hover shows the model's name. *Why:* if the cloud slopes up, the metric is a
decent proxy. If it's flat or slopes down, selecting by that metric is close to random, or worse,
for the business.

**Cumulative curve** (lines, picked model vs business winner). x-axis: share of the population
acted on, in score order. y-axis: either cumulative **profit** ($), or the cumulative **share of 1s
(or value) captured**, depending on the business metric. The dotted vertical line marks your
capacity (top % or top N); a dotted horizontal line marks the capture target on gain curves.
*Why:* it shows *where* the two models differ. A pick that is worse only beyond your capacity
costs nothing; one that is worse inside it costs real money.

**What would each selection metric have cost?** (horizontal bars, "Select by each metric, judge by
[business metric]"). One bar per metric: the opportunity cost if you had selected by that metric.
The bar label gives the model it would pick and the cost; hover adds the metric family and that
model's business rank. *Why:* it answers "which metric should we have used?" in one picture. The
short bars are the metrics that agree with the business.

**Spearman ρ vs [business metric]** (horizontal bars, colored by metric family). For each metric,
the rank correlation across all candidate models between that metric and the business metric. +1:
it ranks the models exactly as the business would; 0: unrelated; negative: backwards. *Why:* the
opportunity-cost bars depend on the single best model; ρ shows whether the metric tracks the
business across the whole field. Real KDD Cup 98 response models give ρ ≈ −0.22 between AUC and
profit.

### Tables and expanders

**Model ranking table.** One row per model, sorted by business rank. Columns: *rank: [select
metric]*, *rank: [business metric]*, then every metric's value, then the knobs each model drew.
Tied models share a rank. **Download the table (CSV)** saves it with full precision. *Why:* the
detail behind every chart. Look at how far down the select metric's ranking the business winner
sits.

**Do the metrics agree with each other?** (expander, heatmap). Rank correlation between every pair
of metrics across the candidate models, signs aligned so + always means "agree" (even for
lower-is-better metrics). Color from −1 to +1; each cell is labeled. *Why:* metrics cluster into
families that agree with each other and disagree with other families. That shows which metrics are
interchangeable and which measure different things.

**Show the benefit distribution** (expander, classification with dollars and spread > 0).
Histogram of the $ benefit of each 1 (x: $ benefit; y: records) and a summary row: *total*,
*mean*, *median*, *90th pct*, *99th pct*, *top 10% of 1s hold* (their share of the value).
*Why:* checks that the benefit assumption is believable, and shows how concentrated the money is.

**Is this a fluke? Repeat with fresh model draws** (expander). **Replicates** slider (5–50) and
**Run replicates** button. It redraws the candidate models that many times on the same population
and economics. Results: *[metric] picked the business winner* (*k* of *n* draws), *Median
opportunity cost*, *Worst case*, plus a histogram of the opportunity cost across draws. *Why:* one
draw can be lucky or unlucky. If the metric rarely finds the winner and the median cost is large,
the problem is the metric, not chance. Each preset's seed was chosen as the median of 20 draws, so
the opening example is typical rather than cherry-picked.

---

## ④ Real: who responds

The same comparison on **480 real models** (120 logistic regressions and 360 decision trees from
one experiment grid) that predict who responds (TARGET_B) to the KDD Cup 1998 Paralyzed Veterans
of America mailing. All of them are scored on the same **47,706 held-out donors**. Nothing here is
simulated.

**Tiles:** *Held-out donors*; *Response rate*; *Median / mean gift* (among responders); *Profit
mailing everyone (actual gifts)*: the baseline any model has to beat.

| Control | What it does | Why it matters |
|---|---|---|
| **Metric you select models by** / **What the business actually values** | As on ③, from the classification metrics. | |
| **Cost per mailing ($)** | Cost of each mail piece (the competition used $0.68). | With a 5% response rate and $15 gifts, most mailings lose money, so ranking matters. |
| **Capacity: mail top %** | Capacity for the "top %" metrics. | |
| **Capture target %** | Target for "depth to capture". | |
| **Probability threshold** (0.01–0.99) | Threshold for accuracy, precision, recall, F1. Default 0.05 ≈ the response rate, as in the original experiment. | At 0.5 almost no donor is a predicted responder. |
| **Value of a responder** (radio) | *actual gift (TARGET_D)* or *constant $14.50* (what the original experiment assumed for every donor). | **The key switch.** With actual gifts, AUC and max profit rank the models with ρ ≈ −0.22. The AUC pick ranks about 415th of 480 on profit and barely beats mailing everyone. With a constant gift, ρ ≈ 0.96. The models didn't change; the dollars did. |
| **Algorithms** (multiselect) | Logistic regression and/or decision trees. | Shows whether the effect is driven by mixing algorithms. |
| **…or top N records** | Capacity for the "top N" metrics. | |

Below these come the same headline, scatter (symbols = algorithm), cumulative curve, cost-by-metric
bars, Spearman bars, ranking table and agreement heatmap described under
[③ Business success](#-business-success). The caption gives when the metrics were computed and the
date of the predictions file.

---

## ⑤ Real: gift amount

The same comparison for **480 real regression models** of the gift amount (TARGET_D): ridge, lasso
and regression trees, each fit on the raw dollars or on log(1 + $).

| Control | What it does | Why it matters |
|---|---|---|
| **Models trained and scored on** (radio) | *every record (mostly zeros)*: predict each person's gift (0 for about 95%) and mail anyone whose predicted gift beats the mailing cost. *donors only (gift amount)*: among people who gave, predict how much (illustrative use: a $10 upgrade ask to the top 20%). Switching scope resets the metrics and economics to that scope's story. | The two scopes fail in different ways. |
| **Metric you select models by** / **What the business actually values** | As on ③, from the regression metrics. | |
| **Cost per action ($)** | Cost of each action ($0.68 per mailing; $10 per upgrade ask). | |
| **Capacity: act on top %** / **…or top N records** | Capacity for the top-% / top-N metrics. | |
| **Capture target %** | Target for "depth to capture". | |
| **Algorithms** (multiselect) | Linear (ridge/lasso) and/or trees. | |
| **Target fitted as** (multiselect) | *raw $* and/or *log(1 + $)*. | Log-target models compress big gifts; back-transformed, they predict low for almost everyone. In the all-records scope they mail nobody and make $0. |

**Tiles:** *Held-out records*; *Share giving nothing*; *Median / mean gift (givers)*; *Profit
acting on everyone*.

Then the same charts and tables as ③, with symbols marking model type (algorithm · target form).
Findings worth showing:
- **All records:** every model's R² is about 0, yet threshold-rule profit runs from $0 to about
  $7.4k. Selecting by **MAE** picks a model that mails nobody, because MAE rewards predicting the
  median gift, which is $0. |Mean error| tracks this decision best (ρ ≈ 0.84), because the rule
  compares the *level* of the prediction to the cost.
- **Donors only:** the RMSE pick ranks about 211th of 480 but is only about 3% short. Regression
  gaps are often smaller than classification gaps. That is an honest result worth saying out loud.

---

## Metric reference

"Top %" is the capacity share, "top N" the capacity count, "p" the predicted probability, and
"threshold" the probability cutoff set on ③. Every rank-ordered metric sorts records from highest
to lowest score. Tied scores (common with trees) are broken by a fixed random permutation, so row
order can never influence a top-N result.

### Classification

| Metric | Family | Better | Definition | When it is the right yardstick |
|---|---|---|---|---|
| ROC AUC | statistical | higher | Probability that a random 1 scores above a random 0. | You act on most of the list and every 1 is worth the same. |
| Accuracy @ p>threshold | statistical | higher | Share of records classified correctly at the threshold. | Rarely; errors almost never cost the same, and at low base rates it ignores almost everyone. |
| Precision @ p>threshold | statistical | higher | Of the records predicted 1, the share that are 1. | False alarms are what cost you. |
| Recall @ p>threshold | statistical | higher | Of the real 1s, the share predicted 1. | Missing a 1 is what costs you. |
| F1 @ p>threshold | statistical | higher | Harmonic mean of precision and recall. | Seldom a business goal: it weighs both error types and every record equally. |
| Log loss | statistical | lower | Mean negative log-likelihood of the probabilities. | You use the probabilities themselves (pricing, expected value). |
| Brier score | statistical | lower | Mean squared error of the probabilities. | As log loss, less harsh on confident mistakes. |
| Partial AUC (FPR ≤ 10%) | top-focused statistical | higher | Mean true-positive rate over the first 10% of 0s in the ranking (0–1). | You only act on the top of the list but want a threshold-free score. |
| Lift @ top % | top N / top % | higher | Rate of 1s in the top % ÷ overall rate. | Fixed share of the list, every 1 worth the same. |
| % of 1s captured @ top % | top N / top % | higher | Share of all 1s found in the top %. | Same, phrased as coverage. |
| % of $ captured @ top % | top N / top % | higher | Share of all 1s' dollar value found in the top %. | Fixed capacity, 1s worth different amounts. |
| Hit rate (precision) @ top N | top N / top % | higher | Share of the top N records that are 1s. | Fixed count, e.g. 100 investigations. Ranks models the same way as recall @ top N. |
| Depth to capture X% of 1s | depth to capture | lower | Smallest share of the list you must act on to find X% of the 1s. | You must reach a coverage target ("catch 90% of fraud"). |
| Profit acting on top % | business $ | higher | Σ(benefit of 1s − cost) over the top %. | Fixed capacity with dollars. |
| Profit acting on top N | business $ | higher | Σ(benefit of 1s − cost) over the top N. | Fixed count with dollars. |
| ROI acting on top % | business $ | higher | Profit ÷ total cost over the top %. Hidden when cost is 0. | Budget is the constraint. |
| Max profit (best depth) | business $ | higher | Highest cumulative profit at any depth (0 if acting never pays). | No capacity limit: go as deep as it pays. |
| Profit acting where p × avg value > cost | business $ | higher | Profit from acting on every record whose expected benefit (p × average benefit) beats its cost. | You deploy a decision rule on the probabilities, so calibration matters. |

### Regression

| Metric | Family | Better | Definition | When it is the right yardstick |
|---|---|---|---|---|
| R² | statistical | higher | 1 − SSE ÷ total sum of squares. | You need accurate numbers everywhere. |
| RMSE | statistical | lower | Root mean squared error. | As R²; dominated by the biggest values. |
| MAE | statistical | lower | Mean absolute error. | Typical error matters, not the big ones. Rewards predicting the median, which is 0 when most values are 0. |
| \|Mean error\| (bias) | statistical | lower | Absolute value of the average error. | Decisions compare the predicted level to a cost. |
| Spearman rank corr | top-focused statistical | higher | Rank correlation between predictions and actuals. | Only the ordering matters. |
| % of total value in top % | top N / top % | higher | Share of all value held by the top % of predictions. | Fixed capacity: contact the most valuable. |
| % of total value in top N | top N / top % | higher | Same for the top N records. | Fixed count. |
| Depth to capture X% of value | depth to capture | lower | Share of the list needed to reach X% of the total value. | Coverage target. |
| Profit acting on top % / top N | business $ | higher | Σ(actual value − cost) over the top % / top N. | Fixed capacity with a cost per contact. |
| ROI acting on top % | business $ | higher | Profit ÷ total cost. Hidden when cost is 0. | Budget constraint. |
| Max profit (best depth) | business $ | higher | Highest cumulative profit at any depth. | No capacity limit. |
| Profit acting where prediction > cost | business $ | higher | Profit from acting on every record whose predicted value beats its cost. | A threshold rule on the predicted number: bias and spread matter. |

---

## Scenario presets and example scenarios

**Presets** (sidebar dropdown) are starting points. Each preset's model seed is the median of
20 draws.

| Preset | Selected by → judged by | The story |
|---|---|---|
| Fundraising appeal (KDD Cup 98 style) | AUC → max profit | 5% respond, gifts are skewed, a mailing costs $0.68. Some models find frequent small donors, others the big gifts, and AUC can't tell them apart. |
| Fundraising with real KDD Cup 98 model shapes | AUC → max profit | Candidates borrow the shapes of real models. |
| Retention offer with limited capacity | AUC → profit in top 5% | The team can only call 5%. A model that nails the top few percent beats one with better AUC. |
| Fraud review: catch 90% of fraud | Lift top 1% → depth to catch 90% | Two capacity-style goals that reward opposite model shapes. Here AUC is a good selector: AUC isn't always wrong. |
| Customer lifetime value (regression) | RMSE → profit in top 20% | RMSE is dominated by calibration and a few huge customers. |
| Customer lifetime value with real gift-model shapes | RMSE → profit in top 20% | Real regression shapes; gaps are smaller. |
| Donation amount (zero-inflated regression) | R² → threshold-rule profit | Mail if the predicted gift beats $0.68: bias and calibration now matter. |
| Donation amount with real KDD Cup 98 model shapes | R² → threshold-rule profit | The same with real model shapes. |

**Example scenarios** (sidebar → Saved, on first run) are worked cases built from the presets.
Load one, then run **Is this a fluke?** on ③ before drawing conclusions.

| Example | Selected by → judged by | What it shows (as it opens) |
|---|---|---|
| fraud counts 1 | ROC AUC → hit rate @ top 100 | Invoice fraud, counts only: 100,000 invoices, 0.1% fraud, investigators can open 100, dollars off. The AUC pick puts **2** frauds in its top 100; the best model puts **28** there. AUC and top-100 hits barely agree across models (ρ = 0.21). |
| fraud net benefit v1 | Accuracy @ p>0.001 → profit acting on top 100 | A recovered fraud is worth about $10,000 (CV 1.2); an investigation costs about $2,000 (CV 1.9); breakeven hit rate 20%. Accuracy's pick **loses $185,627** on its 100 investigations; the best model makes **$108,528**. Try the **0.5** threshold button: accuracy then judges only the tiny tail of high scores. |
| fraud net benefit v2 | Accuracy @ p>0.001 → profit acting on top 100 | As v1 with a wider benefit spread (CV 1.6) and a cheaper, steadier investigation (about $1,000, CV 0.8; breakeven 10%). Accuracy's pick loses $27,659; the best model makes **$227,092**. |
| CLV best 10000 | RMSE → value in the top 10,000 | Email / CLV regression: mostly-zero spend, no cost per email, models built on raw dollars. A small gap ($66k, 1%): an honest "it doesn't always matter much". Switch *Models are built on* to **mixed** on ② to put log-built and raw-built models in the same race. |

---

## Things to try

1. **Flip the value assumption.** On ④, switch *Value of a responder* between actual gift and a
   constant $14.50, and watch the Spearman bar for AUC go from negative to about +1.
2. **Collapse a knob.** On ②, set *Value tilt* to 0–0 in the fundraising preset. If the reversal
   disappears, value tilt was the cause.
3. **Move the threshold.** In the fraud example, set the threshold to 0.5, then to the base rate,
   and watch accuracy and F1's picks change.
4. **Change the capacity.** Slide *act on top %* from 5% to 50% in the retention preset. The best
   model changes, because the business question changed.
5. **Run the replicates** before believing any single result.
6. **Bring your own predictions.** `metric_clinic_metrics.evaluate(task, y, scores, names, biz,
   value)` takes plain arrays (actuals, one row of scores per model, optional $ value per record)
   and returns the full metric table, so you can run the same comparison on your own models
   without the simulator.

---

*Model Selection Clinic by Dean Abbott, Abbott Analytics. Documentation licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code MIT. Please credit when you reuse it.*
