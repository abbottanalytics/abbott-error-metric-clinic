"""Model Selection Clinic -- are you optimizing the right metric?

    streamlit run metric_clinic_app.py

Three tabs follow the conversation with the audience:
    1. Population   -- what the records and the target look like
    2. Models       -- how a family of candidate models behaves (and where it errs)
    3. Business     -- what the business values, who wins, and what picking by
                       the wrong metric costs
"""

from __future__ import annotations

import os
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from metric_clinic_metrics import (FAMILIES, BusinessSpec, agreement_matrix, choose, cumulative_curves,
                                   decile_table, evaluate, metric, metrics_for, rank,
                                   regret_by_metric, roc_points)
from metric_clinic_kdd98 import DEFAULT_OUT as KDD98_PATH, DEFAULT_PROFILES as KDD98_PROFILES
from metric_clinic_kdd98_target_d import load as load_target_d, predictions_path as target_d_path
from metric_clinic_kdd98_target_d import profiles_path as target_d_profiles
from metric_clinic_kdd98 import load as load_kdd98
from metric_clinic_presets import PRESETS
from metric_clinic_sim import (CLASSIFICATION, REGRESSION, ModelSpec, PopulationSpec,
                               make_models, make_population)

st.set_page_config(page_title="Model Selection Clinic", layout="wide", page_icon="🎯")

# Categorical slots from the reference palette: identity, never rank
COL_PICKED = "#2a78d6"     # model the statistical metric picks
COL_IDEAL = "#eb6834"      # model the business would want
COL_OTHER = "#a3a29b"      # every other candidate
COL_BOTH = "#1baf7a"       # same model wins both

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 17px; }
div[data-testid="stMetricValue"] { font-size: 2.1rem; }
.block-container { padding-top: 2.2rem; }
.note { color: var(--text-color); opacity: .75; font-size: .95rem; }
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# State: presets write straight into widget keys
# ----------------------------------------------------------------------------

POP_KEYS = {"task": "pop_task", "n": "pop_n", "target_name": "pop_target",
            "prevalence": "pop_prev", "value_mean": "pop_vmean", "value_cv": "pop_vcv",
            "shape": "pop_shape", "mean": "pop_mean", "sd": "pop_sd", "cv": "pop_cv",
            "nonzero_rate": "pop_nonzero"}
MODEL_KEYS = {"n_models": "m_n", "auc": "m_auc", "concentration": "m_conc",
              "value_tilt": "m_tilt", "calib_bias": "m_calib", "corr": "m_corr",
              "hetero": "m_het", "shrink": "m_shrink", "bias": "m_bias", "strength": "m_strength"}
BIZ_KEYS = {"cost": "b_cost", "top_pct": "b_top", "top_n": "b_topn", "gain_target": "b_gain",
            "threshold": "b_thr"}


PROFILE_ALGOS = {CLASSIFICATION: ["logreg", "tree"], REGRESSION: ["linear", "tree"]}
ALGO_NAMES = {"logreg": "logistic regressions", "tree": "decision / regression trees",
              "linear": "linear regressions (ridge, lasso)"}


def profile_bank_for(task, shape):
    """Which real-model bank fits this population: TARGET_B classifiers, or TARGET_D
    models trained on all records (mostly zeros) or on responders (skewed gifts)."""
    if task == CLASSIFICATION:
        return KDD98_PROFILES, "480 TARGET_B classifiers"
    if shape == "zero_inflated":
        return target_d_profiles("all"), "480 TARGET_D models trained on every record (mostly zeros)"
    return target_d_profiles("responders"), "480 TARGET_D models trained on donors' gift amounts"


def apply_preset():
    p = PRESETS[st.session_state.preset]
    for field, key in POP_KEYS.items():
        v = getattr(p["pop"], field)
        st.session_state[key] = v * 100 if field in ("prevalence", "nonzero_rate") else v
    for field, key in MODEL_KEYS.items():
        v = getattr(p["models"], field)
        st.session_state[key] = tuple(v) if isinstance(v, tuple) else v
    for field, key in BIZ_KEYS.items():
        v = getattr(p["biz"], field)
        st.session_state[key] = v * 100 if field in ("top_pct", "gain_target") else v
    st.session_state.m_mode = "profiles" if p["models"].profile_path else "parametric"
    st.session_state.m_palgos = list(p["models"].profile_algos) or \
        PROFILE_ALGOS[p["pop"].task]
    st.session_state.select_by = p["select_by"]
    st.session_state.judge_by = p["judge_by"]
    st.session_state.frozen = False
    st.session_state.pop_seed = p["pop"].seed
    st.session_state.model_seed = p["models"].seed


if "preset" not in st.session_state:
    st.session_state.preset = next(iter(PRESETS))
    apply_preset()


def bump(key):
    st.session_state[key] += 1


# ----------------------------------------------------------------------------
# Cached computation (population frozen independently of the models)
# ----------------------------------------------------------------------------

@st.cache_resource(max_entries=6, show_spinner="Simulating the population...")
def get_population(pop_kw: tuple):
    return make_population(PopulationSpec(**dict(pop_kw)))


@st.cache_resource(max_entries=6, show_spinner="Simulating candidate models...")
def get_models(pop_kw: tuple, model_kw: tuple):
    return make_models(get_population(pop_kw), ModelSpec(**dict(model_kw)))


@st.cache_data(max_entries=12, show_spinner="Scoring every model...")
def get_table(pop_kw: tuple, model_kw: tuple, biz_kw: tuple):
    pop, ms = get_population(pop_kw), get_models(pop_kw, model_kw)
    return evaluate(pop.task, pop.y, ms.scores, ms.names, BusinessSpec(**dict(biz_kw)),
                    pop.value, ms.params)


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------

PCT_KEYS = {"accuracy", "precision", "recall", "gain_top", "value_top", "depth_gain",
            "best_depth", "share_acted_rule"}


def fmt(task, key, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if key in PCT_KEYS:
        return f"{v:.1%}"
    if key.startswith("profit") or (task == CLASSIFICATION and key == "cost"):
        return f"${v:,.0f}"
    if key in ("rmse", "mae", "mean_error"):
        return f"{v:,.4g}"
    return f"{v:.3f}"


def fmt_gap(task, key, v):
    if key in PCT_KEYS:
        return f"{v * 100:.1f} pts"
    return fmt(task, key, v)


def label(task, key, biz):
    return metric(task, key).name(biz)


def style_fig(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=15), legend=dict(orientation="h", y=-0.18),
                      hoverlabel=dict(font_size=14))
    fig.update_xaxes(showgrid=False)
    return fig


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("🎯 Model Selection Clinic")
    st.selectbox("Scenario preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption(PRESETS[st.session_state.preset]["note"])
    st.divider()
    st.toggle("Freeze population", key="frozen",
              help="Lock the records once the audience agrees they look right, "
                   "so changes to model behavior are not confounded with a new population.")
    c1, c2 = st.columns(2)
    c1.button("New population", on_click=bump, args=("pop_seed",),
              disabled=st.session_state.frozen, use_container_width=True)
    c2.button("New model draws", on_click=bump, args=("model_seed",), use_container_width=True)
    st.caption(f"population seed {st.session_state.pop_seed} · model seed {st.session_state.model_seed}")
    st.divider()
    st.caption("Machine Learning Week EU · Dean Abbott, Abbott Analytics")

task = st.session_state.pop_task
frozen = st.session_state.frozen
is_cls = task == CLASSIFICATION

tab_pop, tab_models, tab_biz, tab_real, tab_real_d = st.tabs(
    ["① Population", "② Model behavior", "③ Business success",
     "④ Real: who responds", "⑤ Real: gift amount"])


# ----------------------------------------------------------------------------
# Tab 1 -- inputs (results drawn after everything is computed)
# ----------------------------------------------------------------------------

with tab_pop:
    in_col, out_col = st.columns([1, 2.4], gap="large")
    with in_col:
        st.radio("Problem type", [CLASSIFICATION, REGRESSION], key="pop_task",
                 horizontal=True, disabled=frozen)
        st.text_input("Target name", key="pop_target", disabled=frozen)
        st.number_input("Population size", 1_000, 1_000_000, step=10_000, key="pop_n",
                        disabled=frozen)
        if is_cls:
            st.number_input("% of records that are 1s", 0.1, 60.0, step=0.5, key="pop_prev",
                            format="%.1f", disabled=frozen)
            st.number_input("Average $ value of a 1", 0.0, 1e7, step=1.0, key="pop_vmean",
                            disabled=frozen)
            st.slider("Spread of that value (CV; 0 = every 1 worth the same)", 0.0, 3.0,
                      step=0.1, key="pop_vcv", disabled=frozen)
        else:
            st.selectbox("Shape of the target", ["normal", "lognormal", "zero_inflated"],
                         key="pop_shape", disabled=frozen,
                         format_func={"normal": "Normal (symmetric)",
                                      "lognormal": "Right-skewed (lognormal)",
                                      "zero_inflated": "Mostly zeros + skewed non-zeros"}.get)
            shape = st.session_state.pop_shape
            st.number_input("Mean" + (" of the non-zero values" if shape == "zero_inflated" else ""),
                            step=1.0,
                            key="pop_mean", disabled=frozen)
            if shape == "normal":
                st.number_input("Standard deviation", 0.001, 1e9, step=1.0, key="pop_sd",
                                disabled=frozen)
            else:
                st.slider("Skew (coefficient of variation)", 0.1, 4.0, step=0.1, key="pop_cv",
                          disabled=frozen)
            if shape == "zero_inflated":
                st.number_input("% of records that are non-zero", 0.1, 100.0, step=0.5,
                                key="pop_nonzero", format="%.1f", disabled=frozen)


# ----------------------------------------------------------------------------
# Tab 2 -- model inputs
# ----------------------------------------------------------------------------

with tab_models:
    in_col2, out_col2 = st.columns([1, 2.4], gap="large")
    with in_col2:
        st.slider("Number of candidate models", 5, 200, step=5, key="m_n")
        st.caption("Each model draws its behavior uniformly from these ranges. "
                   "Collapse a range to one value to switch that behavior off.")
        bank_path, bank_desc = profile_bank_for(task, st.session_state.pop_shape)
        has_profiles = os.path.exists(bank_path)
        st.radio("How do the models behave?", ["parametric", "profiles"], key="m_mode",
                 disabled=not has_profiles,
                 format_func={"parametric": "Set by the controls below",
                              "profiles": "Borrowed from real KDD Cup 98 models"}.get,
                 help="Real models make decile-by-decile trade-offs nobody designed. "
                      "'Borrowed' gives each candidate the shape of one real model: "
                      + ("its lift by ventile and which gift sizes it finds." if is_cls else
                         "which actual values land in each prediction ventile, and its "
                         "prediction level there (bias, shrinkage, spread)."))
        use_profiles = has_profiles and st.session_state.m_mode == "profiles"
        if use_profiles:
            st.caption(f"Shapes from {bank_desc}.")
            if not set(st.session_state.m_palgos) <= set(PROFILE_ALGOS[task]):
                st.session_state.m_palgos = PROFILE_ALGOS[task]
            st.multiselect("Borrow shapes from", PROFILE_ALGOS[task], key="m_palgos",
                           format_func=ALGO_NAMES.get)
            st.slider("Strength relative to the real model (1 = as fitted)", 0.0, 2.0, step=0.05,
                      key="m_strength", help="<1 flattens the shape toward random, >1 sharpens it.")
            if is_cls:
                st.slider("Calibration bias (log-odds, on top of the real model's own)", -3.0, 3.0,
                          step=0.1, key="m_calib")
            else:
                st.slider("Spread of predictions (1 = as the real model fitted)", 0.2, 2.0,
                          step=0.05, key="m_shrink", help="Ranking unchanged.")
                st.slider("Extra bias (in SDs of the target)", -1.0, 1.0, step=0.05, key="m_bias",
                          help="On top of the real model's own bias. Ranking unchanged.")
        elif is_cls:
            st.slider("Ranking strength (target AUC)", 0.50, 0.99, step=0.01, key="m_auc")
            st.slider("Concentration of signal at the top", 0.0, 1.0, step=0.05, key="m_conc",
                      help="0: every 1 is equally easy to find. 1: a quarter of the 1s are very easy, "
                           "the rest are barely separated. AUC held roughly fixed.")
            st.slider("Value tilt (finds high-value 1s ↔ low-value 1s)", -1.0, 1.0, step=0.05,
                      key="m_tilt", help="+ ranks the high-value 1s higher; − finds the cheap ones. "
                                         "AUC unchanged; profit is not.")
            st.slider("Calibration bias (log-odds)", -3.0, 3.0, step=0.1, key="m_calib",
                      help="Shifts every probability up or down. Ranking unchanged, so AUC, lift and "
                           "gains are identical; accuracy, precision, recall, log loss are not.")
        if not use_profiles and not is_cls:
            st.slider("Ranking strength (correlation)", 0.05, 0.99, step=0.01, key="m_corr")
            st.slider("Where errors fall (− precise on high values, + noisy on high values)",
                      -0.95, 0.95, step=0.05, key="m_het",
                      help="Total error variance is held fixed, so RMSE barely moves.")
            st.slider("Spread of predictions (1 = calibrated, <1 under-predicts high values)",
                      0.2, 2.0, step=0.05, key="m_shrink", help="Ranking unchanged.")
            st.slider("Bias (in SDs of the target)", -1.0, 1.0, step=0.05, key="m_bias",
                      help="Adds a constant. Ranking unchanged.")


# ----------------------------------------------------------------------------
# Tab 3 -- business inputs
# ----------------------------------------------------------------------------

keys = [m.key for m in metrics_for(task)]
for k, fallback in (("select_by", keys[0]), ("judge_by", "profit_max")):
    if st.session_state.get(k) not in keys:
        st.session_state[k] = fallback


def biz_from_state():
    return BusinessSpec(cost=st.session_state.b_cost, top_pct=st.session_state.b_top / 100,
                        top_n=int(st.session_state.b_topn), gain_target=st.session_state.b_gain / 100,
                        threshold=st.session_state.b_thr)


with tab_biz:
    r1 = st.columns([1.3, 1.3, 0.9, 1, 0.9, 1, 1])
    biz_preview = biz_from_state()
    r1[0].selectbox("Metric you select models by", keys, key="select_by",
                    format_func=lambda k: f"{label(task, k, biz_preview)}  ·  {metric(task, k).family}")
    r1[1].selectbox("What the business actually values", keys, key="judge_by",
                    format_func=lambda k: f"{label(task, k, biz_preview)}  ·  {metric(task, k).family}")
    r1[2].number_input("Cost per action ($)", 0.0, 1e7, step=0.01 if is_cls else 1.0,
                       key="b_cost", help="Cost of mailing / calling / reviewing one record.")
    r1[3].slider("Capacity: act on top %", 1.0, 100.0, step=1.0, key="b_top")
    r1[4].number_input("…or top N records", 1, 10_000_000, step=100, key="b_topn",
                       help="Count-based capacity, used by the 'top N' metrics.")
    r1[5].slider("Capture target %", 10.0, 100.0, step=5.0, key="b_gain",
                 help="Used by 'depth to capture X% of 1s / value'.")
    if is_cls:
        r1[6].slider("Probability threshold", 0.01, 0.99, step=0.01, key="b_thr",
                     help="Used by accuracy, precision, recall and F1.")


# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------

s = st.session_state
pop_spec = PopulationSpec(task=task, n=int(s.pop_n), target_name=s.pop_target, seed=s.pop_seed,
                          prevalence=s.pop_prev / 100, value_mean=s.pop_vmean, value_cv=s.pop_vcv,
                          shape=s.pop_shape, mean=s.pop_mean, sd=s.pop_sd, cv=s.pop_cv,
                          nonzero_rate=s.pop_nonzero / 100)
model_spec = ModelSpec(n_models=int(s.m_n), seed=s.model_seed, auc=tuple(s.m_auc),
                       concentration=tuple(s.m_conc), value_tilt=tuple(s.m_tilt),
                       calib_bias=tuple(s.m_calib), corr=tuple(s.m_corr), hetero=tuple(s.m_het),
                       shrink=tuple(s.m_shrink), bias=tuple(s.m_bias),
                       profile_path=bank_path if use_profiles else "",
                       profile_algos=tuple(s.m_palgos), strength=tuple(s.m_strength))
biz = biz_from_state()

pop_kw = tuple(asdict(pop_spec).items())
model_kw = tuple(asdict(model_spec).items())
biz_kw = tuple(asdict(biz).items())

pop = get_population(pop_kw)
ms = get_models(pop_kw, model_kw)
table = get_table(pop_kw, model_kw, biz_kw)

def make_roles(task, picked, ideal, select_by, judge_by, biz):
    """role(name) -> legend label; color(name) -> its fixed color."""
    def role(name):
        if name == picked and name == ideal:
            return "picked by both"
        if name == picked:
            return f"picked by {label(task, select_by, biz)}"
        if name == ideal:
            return f"best on {label(task, judge_by, biz)}"
        return "other candidates"

    def color(name):
        r = role(name)
        return {"picked by both": COL_BOTH, "other candidates": COL_OTHER}.get(
            r, COL_PICKED if r.startswith("picked") else COL_IDEAL)
    return role, color


select_by, judge_by = s.select_by, s.judge_by
c = choose(table, task, select_by, judge_by)
picked, ideal = c["picked"], c["ideal"]
idx = {name: i for i, name in enumerate(ms.names)}
target = pop_spec.target_name
role, role_color = make_roles(task, picked, ideal, select_by, judge_by, biz)

focus = [picked] if picked == ideal else [picked, ideal]


# ----------------------------------------------------------------------------
# Tab 1 -- results
# ----------------------------------------------------------------------------

with tab_pop, out_col:
    y = pop.y
    if is_cls:
        k = st.columns(4)
        k[0].metric("Records", f"{pop.n:,}")
        k[1].metric(f"{target} = 1", f"{int(y.sum()):,}")
        k[2].metric("Rate of 1s", f"{y.mean():.2%}")
        v = pop.value[y == 1]
        k[3].metric("Total value of the 1s", f"${v.sum():,.0f}")
        if pop_spec.value_cv > 0:
            fig = px.histogram(x=v, nbins=80, title=f"Value of each {target} = 1 record",
                               labels={"x": "$ value"}, color_discrete_sequence=[COL_PICKED])
            fig.update_traces(marker_line_width=1, marker_line_color="white")
            fig.update_yaxes(title="records")
            st.plotly_chart(style_fig(fig), use_container_width=True)
            top10 = np.sort(v)[-max(1, len(v) // 10):].sum() / v.sum()
            st.dataframe(pd.DataFrame([{
                "mean": f"${v.mean():,.2f}", "median": f"${np.median(v):,.2f}",
                "90th pct": f"${np.quantile(v, .9):,.2f}", "99th pct": f"${np.quantile(v, .99):,.2f}",
                "top 10% of 1s hold": f"{top10:.1%} of the value"}]),
                hide_index=True, use_container_width=True)
        else:
            st.info(f"Every {target} = 1 record is worth ${pop_spec.value_mean:,.2f}. "
                    "Raise the spread to make some 1s more valuable than others.")
    else:
        nz = y[y != 0] if pop_spec.shape == "zero_inflated" else y
        k = st.columns(4)
        k[0].metric("Records", f"{pop.n:,}")
        k[1].metric("Mean", f"{y.mean():,.2f}")
        k[2].metric("Median", f"{np.median(y):,.2f}")
        k[3].metric("Share that is zero", f"{np.mean(y == 0):.1%}")
        clip = np.quantile(nz, 0.995)
        fig = px.histogram(x=nz[nz <= clip], nbins=80, color_discrete_sequence=[COL_PICKED],
                           title=f"Distribution of {target}"
                                 + (" (non-zero records)" if pop_spec.shape == "zero_inflated" else "")
                                 + " · top 0.5% not drawn",
                           labels={"x": target})
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_yaxes(title="records")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        top10 = np.sort(y)[-pop.n // 10:].sum() / y.sum() if y.sum() else np.nan
        st.dataframe(pd.DataFrame([{
            "SD": f"{y.std():,.2f}", "skewness": f"{pd.Series(y).skew():.2f}",
            "90th pct": f"{np.quantile(y, .9):,.2f}", "99th pct": f"{np.quantile(y, .99):,.2f}",
            "max": f"{y.max():,.2f}", "top 10% of records hold": f"{top10:.1%}"}]),
            hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 2 -- results
# ----------------------------------------------------------------------------

with tab_models, out_col2:
    choices = st.multiselect("Models to inspect", ms.names, default=focus, max_selections=4,
                             help="Defaults to the statistical pick and the business winner.")
    stat_key = "auc" if is_cls else "r2"
    fig = px.histogram(table, x=stat_key, nbins=30, color_discrete_sequence=[COL_OTHER],
                       title=f"{label(task, stat_key, biz)} across {len(table)} candidate models",
                       labels={stat_key: label(task, stat_key, biz)})
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    fig.update_yaxes(title="models")
    for i, name in enumerate(sorted(choices, key=lambda n_: table.at[n_, stat_key])):
        fig.add_vline(x=table.at[name, stat_key], line_color=role_color(name), line_width=3,
                      annotation_text=name, annotation_position="top left" if i % 2 == 0 else "top right")
    a, b = st.columns(2)
    a.plotly_chart(style_fig(fig, 360), use_container_width=True)

    palette = [COL_PICKED, COL_IDEAL, "#1baf7a", "#6250d6"]
    colors = {n: (role_color(n) if n in focus else palette[2 + i % 2]) for i, n in enumerate(choices)}
    if is_cls:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                                 line=dict(color=COL_OTHER, dash="dot", width=1)))
        for n in choices:
            r = roc_points(pop.y, ms.scores[idx[n]])
            fig.add_trace(go.Scatter(x=r.fpr, y=r.tpr, mode="lines", line=dict(width=2.5, color=colors[n]),
                                     name=f"{n} (AUC {table.at[n, 'auc']:.3f})"))
        fig.update_layout(title="ROC curves", xaxis_title="false positive rate",
                          yaxis_title="true positive rate")
    else:
        fig = go.Figure()
        for n in choices:
            d = decile_table(task, pop.y, ms.scores[idx[n]])
            fig.add_trace(go.Scatter(x=d.mean_score, y=d.mean_actual, mode="lines+markers",
                                     marker=dict(size=9), line=dict(width=2.5, color=colors[n]), name=n))
        hi = max([max(tr.x.max(), tr.y.max()) for tr in fig.data] or [1])
        lo = min([min(tr.x.min(), tr.y.min()) for tr in fig.data] or [0])
        lim = [min(lo, 0), hi]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect calibration",
                                 line=dict(color=COL_OTHER, dash="dot", width=1)))
        fig.update_layout(title="Calibration by decile: mean predicted vs mean actual",
                          xaxis_title="mean predicted", yaxis_title="mean actual")
    b.plotly_chart(style_fig(fig, 360), use_container_width=True)

    rows = []
    for n in choices:
        d = decile_table(task, pop.y, ms.scores[idx[n]], pop.value, biz.cost)
        d["model"] = n
        rows.append(d)
    if rows:
        d = pd.concat(rows)
        ycol, ytitle = ("lift", f"lift (rate of 1s ÷ overall rate)") if is_cls else \
            ("mean_actual", f"mean actual {target}")
        fig = px.bar(d, x="decile", y=ycol, color="model", barmode="group",
                     color_discrete_map=colors, title="Performance by decile (1 = highest scores)",
                     labels={ycol: ytitle}, hover_data={"profit": ":$,.0f", "mean_score": ":.3f"})
        fig.update_traces(marker_line_width=2, marker_line_color="white")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

        if use_profiles:
            from metric_clinic_profiles import load_bank
            bank = load_bank(bank_path)
            gal = go.Figure()
            pool = [n_ for n_ in bank.names if bank.summary.at[n_, "algo"] in s.m_palgos]
            x_ = np.arange(1, bank.joint.shape[2] + 1) * 100 / bank.joint.shape[2]
            for n_ in pool:
                gal.add_trace(go.Scatter(x=x_, y=bank.lift(bank.names.index(n_)), mode="lines",
                                         line=dict(color=COL_OTHER, width=1), opacity=0.25,
                                         hoverinfo="skip", showlegend=False))
            for n in choices:
                prof = table.at[n, "knob_profile"]
                gal.add_trace(go.Scatter(x=x_, y=bank.lift(bank.names.index(prof)), mode="lines",
                                         line=dict(color=colors[n], width=3),
                                         name=f"{n} ← real model {prof}"))
            gal.add_hline(y=1, line_dash="dot", line_color=COL_OTHER)
            gal.update_layout(title=f"Real shapes being borrowed: {'lift' if is_cls else 'value lift'} "
                                    f"by ventile, {len(pool)} KDD Cup 98 models",
                              xaxis_title="depth into the list (%)",
                              yaxis_title="lift" if is_cls else "value lift")
            st.plotly_chart(style_fig(gal, 380), use_container_width=True)

        knob_cols = [c_ for c_ in table.columns if c_.startswith("knob_")]
        st.dataframe(table.loc[choices, knob_cols].rename(columns=lambda x: x[5:]).round(3),
                     use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 3 -- results
# ----------------------------------------------------------------------------

def render_selection(table, task, y, value, scores, idx, biz, select_by, judge_by, target,
                     key, symbol=None):
    """Headline, scatter, curves, cost-by-metric, ranking table and agreement heatmap.

    Works for any scored model set -- simulated or real.
    """
    is_cls = task == CLASSIFICATION
    c = choose(table, task, select_by, judge_by)
    picked, ideal = c["picked"], c["ideal"]
    role, role_color = make_roles(task, picked, ideal, select_by, judge_by, biz)
    focus = [picked] if picked == ideal else [picked, ideal]
    sel_m, jud_m = metric(task, select_by), metric(task, judge_by)
    sel_name, jud_name = sel_m.name(biz), jud_m.name(biz)
    n_models = len(table)

    k = st.columns(3)
    k[0].metric(f"Picked by {sel_name}", picked,
                f"{fmt(task, judge_by, c['picked_value'])} · ranks #{c['picked_business_rank']} of {n_models}",
                delta_color="off")
    k[1].metric(f"Best on {jud_name}", ideal,
                f"{fmt(task, judge_by, c['ideal_value'])} · ranks #{c['ideal_select_rank']} on {sel_name}",
                delta_color="off")
    gap = c["opportunity_cost"]
    rel = gap / abs(c["ideal_value"]) if c["ideal_value"] else np.nan
    k[2].metric("Opportunity cost of the statistical pick", fmt_gap(task, judge_by, gap),
                (f"{rel:.0%} of what was available" if jud_m.money and np.isfinite(rel) else
                 f"rank agreement across models ρ = {c['rank_corr']:.2f}"),
                delta_color="inverse" if gap > 0 else "off")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        df = table[[select_by, judge_by]].copy()
        df["role"] = [role(n) for n in df.index]
        df["model"] = df.index
        if symbol is not None:
            df[symbol.name] = symbol
        order = ["other candidates"] + [r for r in df.role.unique() if r != "other candidates"]
        cmap = {r: (COL_OTHER if r == "other candidates" else COL_BOTH if r == "picked by both"
                    else COL_PICKED if r.startswith("picked") else COL_IDEAL) for r in order}
        fig = px.scatter(df, x=select_by, y=judge_by, color="role", hover_name="model",
                         symbol=None if symbol is None else symbol.name,
                         category_orders={"role": order}, color_discrete_map=cmap,
                         labels={select_by: f"{sel_name}  ({'higher' if sel_m.higher_better else 'lower'} is better)",
                                 judge_by: f"{jud_name}  ({'higher' if jud_m.higher_better else 'lower'} is better)",
                                 "role": ""},
                         title="Every candidate model: statistical score vs business result")
        fig.update_traces(marker=dict(size=11, line=dict(width=2, color="white")))
        for n in focus:
            fig.add_annotation(x=table.at[n, select_by], y=table.at[n, judge_by], text=f"<b>{n}</b>",
                               showarrow=True, arrowhead=0, ax=25, ay=-25)
        if jud_m.money:
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(style_fig(fig, 470), use_container_width=True, key=f"{key}_chart0")

    with right:
        curves = []
        for n in focus:
            cc = cumulative_curves(task, y, scores[idx[n]], value, biz.cost)
            cc["model"] = f"{n} · {role(n)}"
            curves.append(cc)
        cc = pd.concat(curves)
        cmap2 = {f"{n} · {role(n)}": role_color(n) for n in focus}
        use_gain = judge_by in ("gain_top", "value_top", "depth_gain", "lift_top")
        ycol = "gain" if use_gain else "profit"
        fig = px.line(cc, x="depth", y=ycol, color="model", color_discrete_map=cmap2,
                      title=("Cumulative share of " + ("1s" if is_cls else target) + " captured")
                      if use_gain else "Cumulative profit as you act deeper into the list",
                      labels={"depth": "share of population acted on", ycol: ""})
        fig.update_traces(line=dict(width=2.5))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%" if use_gain else "$,.0f")
        fig.add_vline(x=biz.top_pct, line_dash="dot", line_color=COL_OTHER,
                      annotation_text=f"capacity {biz.top_pct:.0%}")
        if use_gain:
            fig.add_hline(y=biz.gain_target, line_dash="dot", line_color=COL_OTHER)
        st.plotly_chart(style_fig(fig, 470), use_container_width=True, key=f"{key}_chart1")

    st.subheader("What would each selection metric have cost?")
    left, right = st.columns([1.15, 1], gap="large")
    reg = regret_by_metric(table, task, judge_by)
    reg["metric"] = [label(task, k_, biz) for k_ in reg.key]
    reg = reg.sort_values("opportunity_cost")
    fig = px.bar(reg, x="opportunity_cost", y="metric", orientation="h",
                 color_discrete_sequence=[COL_PICKED],
                 hover_data={"picks": True, "business_rank": True, "family": True, "metric": False},
                 labels={"opportunity_cost": f"shortfall vs best model on {jud_name}", "metric": ""})
    fig.update_traces(marker_line_width=2, marker_line_color="white",
                      text=[f"{p_} · {fmt_gap(task, judge_by, v_)}" for p_, v_ in
                            zip(reg.picks, reg.opportunity_cost)], textposition="outside",
                      cliponaxis=False)
    if jud_m.money:
        fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    hi = reg.opportunity_cost.max()
    fig.update_xaxes(range=[0, (hi if hi > 0 else 1) * 1.8])
    fig.update_layout(title=f"Select by each metric, judge by {jud_name}")
    left.plotly_chart(style_fig(fig, 80 + 34 * len(reg)), use_container_width=True, key=f"{key}_chart2")

    agree = agreement_matrix(table, task)[judge_by].drop(judge_by)
    agree = agree.dropna().sort_values()
    fam = {m_.key: m_.family for m_ in metrics_for(task)}
    fig = px.bar(x=agree.values, y=[label(task, k_, biz) for k_ in agree.index], orientation="h",
                 color=[fam[k_] for k_ in agree.index],
                 color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"],
                 category_orders={"color": FAMILIES},
                 labels={"x": f"Spearman ρ between model ranks", "y": "", "color": ""},
                 title=f"Spearman ρ vs {jud_name}")
    fig.update_traces(marker_line_width=2, marker_line_color="white",
                      texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[min(-0.3, agree.min() - 0.35), 1.2], zeroline=True,
                     zerolinecolor=COL_OTHER)
    fig.update_yaxes(categoryorder="array",
                     categoryarray=[label(task, k_, biz) for k_ in agree.index])
    fig.update_layout(legend=dict(orientation="h", y=-0.12))
    right.plotly_chart(style_fig(fig, 80 + 34 * len(reg)), use_container_width=True, key=f"{key}_chart4")

    st.subheader("Model ranking table")
    show = [m.key for m in metrics_for(task)]
    t = table.copy()
    t.insert(0, f"rank: {sel_name}", rank(table, task, select_by))
    t.insert(1, f"rank: {jud_name}", rank(table, task, judge_by))
    t = t.sort_values(f"rank: {jud_name}")
    disp = t[[f"rank: {sel_name}", f"rank: {jud_name}"]].copy()
    for k_ in show:
        disp[label(task, k_, biz)] = [fmt(task, k_, v) for v in t[k_]]
    for kc in [c_ for c_ in t.columns if c_.startswith("knob_")]:
        disp[kc[5:]] = t[kc].round(3) if pd.api.types.is_numeric_dtype(t[kc]) else t[kc]
    st.dataframe(disp, use_container_width=True, height=380)
    st.download_button("Download the table (CSV)", t.to_csv().encode(), f"{key}_model_ranking.csv",
                       "text/csv", key=f"{key}_download")

    with st.expander("Do the metrics agree with each other? (rank correlation across models)"):
        am = agreement_matrix(table, task)
        names = [label(task, k_, biz) for k_ in am.columns]
        fig = go.Figure(go.Heatmap(z=am.values, x=names, y=names, zmin=-1, zmax=1,
                                   colorscale=[[0, COL_IDEAL], [0.5, "#e8e7e2"], [1, COL_PICKED]],
                                   text=np.round(am.values, 2), texttemplate="%{text}",
                                   hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
        fig.update_layout(title="+1: always rank models the same way · −1: opposite (signs aligned so + = agree)")
        st.plotly_chart(style_fig(fig, 620), use_container_width=True, key=f"{key}_chart3")

with tab_biz:
    render_selection(table, task, pop.y, pop.value, ms.scores, idx, biz, select_by, judge_by,
                     target, key="sim")
    sel_name, jud_name = label(task, select_by, biz), label(task, judge_by, biz)
    with st.expander("Is this a fluke? Repeat with fresh model draws"):
        reps = st.slider("Replicates", 5, 50, 10, step=5)
        if st.button("Run replicates"):
            gaps, hits = [], 0
            bar = st.progress(0.0)
            for r in range(reps):
                spec_r = replace(model_spec, seed=model_spec.seed + 1000 + r)
                ms_r = make_models(pop, spec_r)
                t_r = evaluate(task, pop.y, ms_r.scores, ms_r.names, biz, pop.value)
                c_r = choose(t_r, task, select_by, judge_by)
                gaps.append(c_r["opportunity_cost"])
                hits += c_r["picked"] == c_r["ideal"]
                bar.progress((r + 1) / reps)
            gaps = np.array(gaps)
            k = st.columns(3)
            k[0].metric(f"{sel_name} picked the business winner", f"{hits} of {reps}")
            k[1].metric("Median opportunity cost", fmt_gap(task, judge_by, np.median(gaps)))
            k[2].metric("Worst case", fmt_gap(task, judge_by, gaps.max()))
            fig = px.histogram(x=gaps, nbins=20, color_discrete_sequence=[COL_PICKED],
                               labels={"x": f"opportunity cost ({jud_name})"},
                               title="Opportunity cost across replicate model sets")
            fig.update_traces(marker_line_width=1, marker_line_color="white")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 4 -- the real fundraising example
# ----------------------------------------------------------------------------

KDD98_CONSTANT_GIFT = 14.50        # what the original experiment assumed for every donor
for k_, v_ in {"rk_select": "auc", "rk_judge": "profit_max", "rk_cost": 0.68, "rk_top": 50.0, "rk_topn": 1000,
               "rk_gain": 90.0, "rk_thr": 0.05, "rk_value": "actual", "rk_algos": ["logreg", "tree"]}.items():
    st.session_state.setdefault(k_, v_)


@st.cache_resource(show_spinner="Loading KDD Cup 98 predictions...")
def get_kdd98(path, mtime):
    return load_kdd98(path)


@st.cache_data(max_entries=12, show_spinner="Scoring the real models...")
def get_real_table(mtime, biz_kw, value_mode, algos):
    y, value, scores, names, configs = get_kdd98(KDD98_PATH, mtime)
    keep = np.flatnonzero(configs.algo.isin(algos).values)
    v = value if value_mode == "actual" else np.full(len(y), KDD98_CONSTANT_GIFT)
    return evaluate(CLASSIFICATION, y, scores[keep], [names[i] for i in keep],
                    BusinessSpec(**dict(biz_kw)), v, configs.iloc[keep].to_dict("records"))


with tab_real:
    if not os.path.exists(KDD98_PATH):
        st.info("No KDD Cup 98 predictions yet. Generate them once (about 10 minutes):")
        st.code("python3 metric_clinic_kdd98.py", language="bash")
    else:
        mtime = os.path.getmtime(KDD98_PATH)
        ry, rvalue, rscores, rnames, rconfigs = get_kdd98(KDD98_PATH, mtime)
        st.markdown(
            f"**{len(rnames)} real models** (logistic regressions and decision trees from one "
            f"experiment grid) scored on the same **{len(ry):,} held-out donors** from the "
            f"KDD Cup 1998 PVA mailing. TARGET_B = responded; TARGET_D = the actual gift.")
        gifts = rvalue[ry == 1]
        k = st.columns(4)
        k[0].metric("Held-out donors", f"{len(ry):,}")
        k[1].metric("Response rate", f"{ry.mean():.2%}")
        k[2].metric("Median / mean gift", f"${np.median(gifts):,.0f} / ${gifts.mean():,.2f}")
        k[3].metric("Profit mailing everyone (actual gifts)", f"${(rvalue - st.session_state.rk_cost).sum():,.0f}")

        ckeys = [m.key for m in metrics_for(CLASSIFICATION)]
        rbiz_preview = BusinessSpec(cost=st.session_state.rk_cost, top_pct=st.session_state.rk_top / 100,
                                    top_n=int(st.session_state.rk_topn),
                                    gain_target=st.session_state.rk_gain / 100,
                                    threshold=st.session_state.rk_thr)
        r1 = st.columns([1.3, 1.3, 0.9, 1, 1, 1])
        fmt_key = lambda k_: f"{label(CLASSIFICATION, k_, rbiz_preview)}  ·  {metric(CLASSIFICATION, k_).family}"
        r1[0].selectbox("Metric you select models by", ckeys, key="rk_select", format_func=fmt_key)
        r1[1].selectbox("What the business actually values", ckeys, key="rk_judge", format_func=fmt_key)
        r1[2].number_input("Cost per mailing ($)", 0.0, 100.0, step=0.01, key="rk_cost")
        r1[3].slider("Capacity: mail top %", 1.0, 100.0, step=1.0, key="rk_top")
        r1[4].slider("Capture target %", 10.0, 100.0, step=5.0, key="rk_gain")
        r1[5].slider("Probability threshold", 0.01, 0.99, step=0.01, key="rk_thr",
                     help="0.05 ≈ the response rate, as in the original experiment.")
        r2 = st.columns([2, 2, 1, 2])
        r2[0].radio("Value of a responder", ["actual", "constant"], key="rk_value", horizontal=True,
                    format_func={"actual": "actual gift (TARGET_D)",
                                 "constant": f"constant ${KDD98_CONSTANT_GIFT:.2f}"}.get)
        r2[1].multiselect("Algorithms", ["logreg", "tree"], key="rk_algos")
        r2[2].number_input("…or top N records", 1, 50_000, step=100, key="rk_topn")
        if not st.session_state.rk_algos:
            st.warning("Pick at least one algorithm.")
        else:
            rbiz = rbiz_preview
            rtable = get_real_table(mtime, tuple(asdict(rbiz).items()), st.session_state.rk_value,
                                    tuple(st.session_state.rk_algos))
            rvalue_used = rvalue if st.session_state.rk_value == "actual" else \
                np.full(len(ry), KDD98_CONSTANT_GIFT)
            ridx = {n: i for i, n in enumerate(rnames)}
            render_selection(rtable, CLASSIFICATION, ry, rvalue_used, rscores, ridx, rbiz,
                             st.session_state.rk_select, st.session_state.rk_judge, "TARGET_B",
                             key="kdd98", symbol=rtable["knob_algo"].rename("algorithm"))


# ----------------------------------------------------------------------------
# Tab 5 -- real gift-amount (regression) models
# ----------------------------------------------------------------------------

TD_SCOPES = {
    "all": dict(label="every record (mostly zeros)", select="r2", judge="profit_rule", cost=0.68,
                top=20.0, topn=5000,
                story="Predict each person's gift (0 for most) and mail anyone whose predicted "
                      "gift beats the mailing cost."),
    "responders": dict(label="donors only (gift amount)", select="rmse", judge="profit_top", cost=10.0,
                       top=20.0, topn=250,
                       story="Among people who gave, predict how much. Illustrative use: a $10 "
                             "personal upgrade ask to the top 20% of predicted givers."),
}


def apply_td_scope():
    d = TD_SCOPES[st.session_state.td_scope]
    st.session_state.update(td_select=d["select"], td_judge=d["judge"], td_cost=d["cost"],
                            td_top=d["top"], td_topn=d["topn"])


if "td_scope" not in st.session_state:
    st.session_state.update(td_scope="all", td_gain=80.0, td_algos=["linear", "tree"],
                            td_transforms=["raw", "log1p"])
    apply_td_scope()


@st.cache_resource(show_spinner="Loading TARGET_D predictions...")
def get_target_d(scope, mtime):
    return load_target_d(scope)


@st.cache_data(max_entries=12, show_spinner="Scoring the real gift models...")
def get_target_d_table(scope, mtime, biz_kw, algos, transforms):
    y, scores, names, configs = get_target_d(scope, mtime)
    keep = np.flatnonzero((configs["algo"].isin(algos) & configs["transform"].isin(transforms)).values)
    return evaluate(REGRESSION, y, scores[keep], [names[i] for i in keep],
                    BusinessSpec(**dict(biz_kw)), params=configs.iloc[keep].to_dict("records"))


with tab_real_d:
    st.radio("Models trained and scored on", list(TD_SCOPES), key="td_scope", horizontal=True,
             on_change=apply_td_scope, format_func=lambda k_: TD_SCOPES[k_]["label"])
    scope = st.session_state.td_scope
    path = target_d_path(scope)
    if not os.path.exists(path):
        st.info("No TARGET_D predictions yet. Generate them once (about 3 minutes):")
        st.code("python3 metric_clinic_kdd98_target_d.py", language="bash")
    else:
        mtime = os.path.getmtime(path)
        dy, dscores, dnames, dconfigs = get_target_d(scope, mtime)
        st.markdown(
            f"**{len(dnames)} real models** (ridge, lasso and regression trees; raw or log target) "
            f"scored on the same **{len(dy):,} held-out records**. {TD_SCOPES[scope]['story']}")
        k = st.columns(4)
        k[0].metric("Held-out records", f"{len(dy):,}")
        k[1].metric("Share giving nothing", f"{np.mean(dy == 0):.1%}")
        nz = dy[dy > 0]
        k[2].metric("Median / mean gift (givers)", f"${np.median(nz):,.0f} / ${nz.mean():,.2f}")
        k[3].metric("Profit acting on everyone", f"${(dy - st.session_state.td_cost).sum():,.0f}")

        rkeys = [m.key for m in metrics_for(REGRESSION)]
        dbiz = BusinessSpec(cost=st.session_state.td_cost, top_pct=st.session_state.td_top / 100,
                            top_n=int(st.session_state.td_topn),
                            gain_target=st.session_state.td_gain / 100)
        fmt_key = lambda k_: f"{label(REGRESSION, k_, dbiz)}  ·  {metric(REGRESSION, k_).family}"
        r1 = st.columns([1.3, 1.3, 0.9, 1, 0.9, 1])
        r1[0].selectbox("Metric you select models by", rkeys, key="td_select", format_func=fmt_key)
        r1[1].selectbox("What the business actually values", rkeys, key="td_judge", format_func=fmt_key)
        r1[2].number_input("Cost per action ($)", 0.0, 1000.0, step=0.01, key="td_cost")
        r1[3].slider("Capacity: act on top %", 1.0, 100.0, step=1.0, key="td_top")
        r1[4].number_input("…or top N records", 1, 50_000, step=50, key="td_topn")
        r1[5].slider("Capture target %", 10.0, 100.0, step=5.0, key="td_gain")
        r2 = st.columns([2, 2, 3])
        r2[0].multiselect("Algorithms", ["linear", "tree"], key="td_algos", format_func=ALGO_NAMES.get)
        r2[1].multiselect("Target fitted as", ["raw", "log1p"], key="td_transforms",
                          format_func={"raw": "raw $", "log1p": "log(1 + $)"}.get,
                          help="Log-target models compress big gifts; back-transformed, they "
                               "predict low for almost everyone.")
        if not st.session_state.td_algos or not st.session_state.td_transforms:
            st.warning("Pick at least one algorithm and one target form.")
        else:
            dtable = get_target_d_table(scope, mtime, tuple(asdict(dbiz).items()),
                                        tuple(st.session_state.td_algos),
                                        tuple(st.session_state.td_transforms))
            didx = {n: i for i, n in enumerate(dnames)}
            kind = (dtable.knob_algo + " · " + dtable.knob_transform).rename("model type")
            render_selection(dtable, REGRESSION, dy, None, dscores, didx, dbiz,
                             st.session_state.td_select, st.session_state.td_judge, "TARGET_D",
                             key=f"td_{scope}", symbol=kind)
