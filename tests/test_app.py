import pytest
from streamlit.testing.v1 import AppTest

from credit_risk.data import ROOT


@pytest.mark.parametrize("page", range(5))
def test_german_pages_render(page):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[page]).run()
    assert not app.exception


def test_english_customer_and_monitoring():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    app.sidebar.selectbox[0].select("English").run()
    app.sidebar.radio[0].set_value("Customer analysis").run()
    assert not app.exception
    app.selectbox[0].set_value(2).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Data monitoring").run()
    assert not app.exception
