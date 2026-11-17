"""Smoke test: every preset renders without exceptions."""
import os
import tempfile

import pytest
from streamlit.testing.v1 import AppTest

from metric_clinic_kdd98 import DEFAULT_OUT as KDD98_PATH
from metric_clinic_kdd98_target_d import predictions_path as target_d_path
from metric_clinic_presets import PRESETS

# keep test runs out of the real saved-settings file
os.environ["METRIC_CLINIC_STATE"] = os.path.join(tempfile.mkdtemp(), "clinic_state.json")


@pytest.mark.parametrize("preset", list(PRESETS))
def test_preset_renders(preset):
    at = AppTest.from_file("metric_clinic_app.py", default_timeout=120)
    at.run()
    at.selectbox(key="preset").set_value(preset).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 3


@pytest.mark.skipif(not os.path.exists(KDD98_PATH), reason="run metric_clinic_kdd98.py first")
def test_kdd98_tab_renders():
    at = AppTest.from_file("metric_clinic_app.py", default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Held-out donors" for m in at.metric)
    at.radio(key="rk_value").set_value("constant").run()
    assert not at.exception, at.exception


@pytest.mark.skipif(not all(os.path.exists(target_d_path(s_)) for s_ in ("all", "responders")),
                    reason="run metric_clinic_kdd98_target_d.py first")
def test_target_d_tab_renders_both_scopes():
    at = AppTest.from_file("metric_clinic_app.py", default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Share giving nothing" for m in at.metric)
    at.radio(key="td_scope").set_value("responders").run()
    assert not at.exception, at.exception
    assert at.session_state.td_judge == "profit_top"
    at.multiselect(key="td_transforms").set_value(["raw"]).run()
    assert not at.exception, at.exception
