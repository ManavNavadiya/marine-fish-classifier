"""Tests for the shared feature code (used by training AND the API)."""

import math

import numpy as np
import pytest

from app.preprocessing import (
    MODEL_FEATURES,
    RAW_FEATURES,
    PreprocessingError,
    engineer_features,
    preprocess,
    valid_rows_mask,
)

SALMON = {"length_cm": 75, "weight_kg": 4.5, "height_cm": 15, "fin_count": 8, "water_temp_c": 10}


def test_feature_order_is_stable():
    # The saved model depends on this exact order.
    assert MODEL_FEATURES == RAW_FEATURES + ["condition_factor", "body_ratio"]


def test_engineered_features_values():
    out = engineer_features(np.array([[10.0, 0.01, 2.0, 7, 20.0]]))
    assert out.shape == (1, 7)
    assert out[0, 5] == pytest.approx(1.0)   # 100 * 10 g / 10^3
    assert out[0, 6] == pytest.approx(5.0)   # 10 / 2


def test_preprocess_valid_input():
    x = preprocess(SALMON)
    assert x.shape == (1, len(MODEL_FEATURES))
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("field,value", [
    ("length_cm", 0),
    ("length_cm", 9999),
    ("weight_kg", -1),
    ("fin_count", 50),
    ("water_temp_c", 80),
])
def test_preprocess_rejects_out_of_range(field, value):
    with pytest.raises(PreprocessingError, match=field):
        preprocess({**SALMON, field: value})


def test_preprocess_rejects_nan():
    with pytest.raises(PreprocessingError, match="finite"):
        preprocess({**SALMON, "weight_kg": math.nan})


def test_preprocess_rejects_taller_than_long():
    with pytest.raises(PreprocessingError, match="height_cm"):
        preprocess({**SALMON, "height_cm": 80})


def test_preprocess_rejects_missing_value():
    data = dict(SALMON)
    del data["fin_count"]
    with pytest.raises(PreprocessingError, match="fin_count"):
        preprocess(data)


def test_valid_rows_mask_matches_preprocess():
    rows = np.array([
        [75, 4.5, 15, 8, 10],       # ok
        [75, -1, 15, 8, 10],        # negative weight
        [75, 4.5, np.nan, 8, 10],   # missing
        [10, 0.02, 12, 7, 26],      # taller than long
    ], dtype=float)
    assert valid_rows_mask(rows).tolist() == [True, False, False, False]
