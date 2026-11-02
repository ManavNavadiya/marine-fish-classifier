"""Shared feature logic: used by BOTH the ETL/training code and the API.

Keeping one copy of this code is how we prevent training-serving skew:
the model sees features built in exactly the same way at training time
and at prediction time.
"""

import math

import numpy as np

# Raw measurements a user (or the survey CSV) provides, in this order.
RAW_FEATURES = ["length_cm", "weight_kg", "height_cm", "fin_count", "water_temp_c"]

# Features we derive from the raw ones.
#   condition_factor: Fulton's K = 100 * weight(g) / length(cm)^3, a standard
#                     fisheries measure of how "plump" a fish is.
#   body_ratio:       length / height, a simple body-shape measure.
ENGINEERED_FEATURES = ["condition_factor", "body_ratio"]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

# Class labels (alphabetical). Index i in this list == label i in training.
SPECIES = [
    "Atlantic Herring",
    "Atlantic Mackerel",
    "Atlantic Salmon",
    "Bluefin Tuna",
    "Clownfish",
    "European Sardine",
]

# Physically sensible ranges. The ETL drops rows outside them and the API
# rejects requests outside them, so both sides agree on the input domain.
VALID_RANGES = {
    "length_cm": (1.0, 400.0),
    "weight_kg": (0.001, 1000.0),
    "height_cm": (0.5, 120.0),
    "fin_count": (3, 20),
    "water_temp_c": (-2.0, 35.0),
}


class PreprocessingError(ValueError):
    """Input is well-formed but not usable (out of range, NaN, ...)."""


def engineer_features(raw: np.ndarray) -> np.ndarray:
    """(n, 5) raw measurements -> (n, 7) model features."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(RAW_FEATURES):
        raise PreprocessingError(
            f"Expected shape (n, {len(RAW_FEATURES)}), got {raw.shape}"
        )
    length, weight, height = raw[:, 0], raw[:, 1], raw[:, 2]
    condition_factor = 100.0 * (weight * 1000.0) / np.power(length, 3)
    body_ratio = length / height
    return np.column_stack([raw, condition_factor, body_ratio])


def valid_rows_mask(raw: np.ndarray) -> np.ndarray:
    """Boolean mask of rows that are finite, in range and plausible."""
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.all(np.isfinite(raw), axis=1)
    for i, name in enumerate(RAW_FEATURES):
        low, high = VALID_RANGES[name]
        mask &= (raw[:, i] >= low) & (raw[:, i] <= high)
    mask &= raw[:, 2] < raw[:, 0]  # a fish is longer than it is tall
    return mask


def preprocess(values: dict) -> np.ndarray:
    """One request's measurements -> (1, 7) array ready for the model.

    Raises PreprocessingError with a human-readable message if invalid.
    """
    row = []
    for name in RAW_FEATURES:
        if name not in values:
            raise PreprocessingError(f"Missing value: {name}")
        value = float(values[name])
        if not math.isfinite(value):
            raise PreprocessingError(f"{name} must be a finite number")
        low, high = VALID_RANGES[name]
        if not low <= value <= high:
            raise PreprocessingError(
                f"{name}={value:g} is outside the valid range {low:g} to {high:g}"
            )
        row.append(value)

    if row[2] >= row[0]:
        raise PreprocessingError("height_cm must be smaller than length_cm")

    return engineer_features(np.array([row]))
