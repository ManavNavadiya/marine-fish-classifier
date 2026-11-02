"""Smoke test for the Streamlit page (skipped if Streamlit or the model is missing)."""

from pathlib import Path

import pytest

from app.model import get_model_path

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILE = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")


@pytest.mark.skipif(not get_model_path().exists(), reason="trained model not found")
def test_streamlit_page_predicts():
    at = AppTest.from_file(APP_FILE, default_timeout=30).run()
    assert not at.exception

    at.selectbox[0].select("Clownfish").run()
    at.button[0].click().run()

    assert not at.exception
    assert any("Clownfish" in s.value for s in at.success)


@pytest.mark.skipif(not get_model_path().exists(), reason="trained model not found")
def test_streamlit_page_rejects_bad_input():
    at = AppTest.from_file(APP_FILE, default_timeout=30).run()
    at.number_input[1].set_value(-1.0).run()  # weight_kg
    at.button[0].click().run()

    assert not at.exception
    assert any("weight_kg" in e.value for e in at.error)
