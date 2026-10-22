import json

import pytest
from streamlit.testing.v1 import AppTest

from credit_risk.data import ROOT


@pytest.mark.parametrize("language", ["Deutsch", "English", "Türkçe"])
def test_all_pages_and_predictions_in_each_language(language):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    app.sidebar.selectbox[0].select(language).run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["4,480", "0.780", "0.551", "0.133"]
    pages = app.sidebar.radio[0].options
    for page in pages:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        if page == pages[1]:
            assert app.metric[0].value == "6.6%"
            app.selectbox[0].set_value(2).run()
            assert not app.exception
            assert app.metric[0].value == "58.4%"
        if page == pages[3]:
            assert app.metric[1].value == "0.005"
            app.main.radio[0].set_value(app.main.radio[0].options[1]).run()
            assert not app.exception
            assert app.metric[1].value == "0.902"


def test_export_and_model_files_are_present():
    assert (ROOT / "data/batch_example.csv").is_file()
    assert (ROOT / "artifacts/model.joblib").is_file()
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    assert report["threshold"] == .22
