"""Streamlit frontend for the marine fish classifier.

Uses the SAME model file and the SAME preprocessing code as the FastAPI app,
so predictions are identical everywhere (no training-serving skew).

Run locally:   streamlit run ui/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Make the project root importable (Streamlit Cloud runs this file directly).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model import FishPredictor, get_model_path  # noqa: E402
from app.preprocessing import VALID_RANGES, PreprocessingError, preprocess  # noqa: E402

PRESETS = {
    "Atlantic Salmon": {"length_cm": 72.0, "weight_kg": 4.1, "height_cm": 14.5, "fin_count": 8, "water_temp_c": 9.0},
    "Bluefin Tuna": {"length_cm": 210.0, "weight_kg": 230.0, "height_cm": 52.0, "fin_count": 9, "water_temp_c": 19.0},
    "Clownfish": {"length_cm": 9.5, "weight_kg": 0.024, "height_cm": 4.2, "fin_count": 7, "water_temp_c": 27.0},
    "Atlantic Mackerel": {"length_cm": 34.0, "weight_kg": 0.46, "height_cm": 6.8, "fin_count": 9, "water_temp_c": 12.0},
    "Atlantic Herring": {"length_cm": 29.0, "weight_kg": 0.22, "height_cm": 5.8, "fin_count": 7, "water_temp_c": 10.0},
    "European Sardine": {"length_cm": 19.0, "weight_kg": 0.07, "height_cm": 3.9, "fin_count": 7, "water_temp_c": 17.0},
}

st.set_page_config(page_title="Marine Fish Classifier", page_icon="🐟", layout="centered")


@st.cache_resource
def load_predictor() -> FishPredictor:
    """Load the model once per server process, not on every click."""
    return FishPredictor.from_path(get_model_path())


def load_model_card() -> dict:
    path = ROOT / "models" / "model_card.json"
    return json.loads(path.read_text()) if path.exists() else {}


st.title("🐟 Marine Fish Classifier")
st.caption("Enter a fish's measurements and the model predicts its species. Deployed automatically via GitHub Actions.")

try:
    predictor = load_predictor()
except Exception as exc:  # show the problem instead of crashing the page
    st.error(f"Model could not be loaded: {exc}")
    st.stop()

preset_name = st.selectbox("Start from a typical fish (optional)", ["Custom"] + list(PRESETS))
defaults = PRESETS.get(preset_name, PRESETS["Atlantic Salmon"])

col1, col2 = st.columns(2)
with col1:
    length_cm = st.number_input("Length (cm)", value=defaults["length_cm"], step=1.0, format="%.1f")
    weight_kg = st.number_input("Weight (kg)", value=defaults["weight_kg"], step=0.01, format="%.3f")
    height_cm = st.number_input("Body height (cm)", value=defaults["height_cm"], step=0.5, format="%.1f")
with col2:
    fin_count = st.number_input("Fin count", value=defaults["fin_count"], step=1, format="%d")
    water_temp_c = st.number_input("Water temperature (°C)", value=defaults["water_temp_c"], step=0.5, format="%.1f")

if st.button("Predict species", type="primary", use_container_width=True):
    values = {
        "length_cm": length_cm,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "fin_count": int(fin_count),
        "water_temp_c": water_temp_c,
    }
    try:
        batch = preprocess(values)
    except PreprocessingError as exc:
        st.error(f"Invalid input: {exc}")
    else:
        probs = predictor.predict(batch)
        best = int(probs.argmax())
        st.success(f"**{predictor.classes[best]}**, {probs[best]:.1%} confidence")
        st.subheader("Probability per species")
        for species, p in sorted(zip(predictor.classes, probs), key=lambda x: -x[1]):
            st.progress(float(p), text=f"{species}: {p:.1%}")

with st.expander("About the model"):
    card = load_model_card()
    if card:
        st.write(f"Model: **{card.get('selected_run')}** (scikit-learn {card.get('sklearn_version')})")
        st.write(f"Validation accuracy: **{card.get('val_accuracy', 0):.1%}**, "
                 f"test accuracy: **{card.get('test_accuracy', 0):.1%}**")
        st.write("Features: " + ", ".join(card.get("features", [])))
    st.write("Valid input ranges:")
    st.table({name: {"min": low, "max": high} for name, (low, high) in VALID_RANGES.items()})
