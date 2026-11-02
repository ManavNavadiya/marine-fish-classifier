"""Load the trained scikit-learn bundle and run predictions."""

import logging
import os
from pathlib import Path

import numpy as np

from app.preprocessing import MODEL_FEATURES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fish_classifier.joblib"


def get_model_path() -> Path:
    """MODEL_PATH env var wins; otherwise use the Phase 1 output."""
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


class FishPredictor:
    def __init__(self, pipeline, classes: list[str]):
        self.pipeline = pipeline
        self.classes = list(classes)

    @classmethod
    def from_path(cls, path: Path) -> "FishPredictor":
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run Phase 1: python -m src.train"
            )
        # Imported here so the API (and its tests) start fast.
        import joblib
        import sklearn

        logger.info("Loading model from %s", path)
        bundle = joblib.load(path)

        # Guard against training-serving skew: the model must have been trained
        # on exactly the features this API builds.
        if list(bundle["features"]) != MODEL_FEATURES:
            raise ValueError(
                f"Model expects features {bundle['features']}, "
                f"but the API builds {MODEL_FEATURES}. Retrain the model."
            )
        if bundle.get("sklearn_version") != sklearn.__version__:
            logger.warning(
                "Model was saved with scikit-learn %s but %s is installed. "
                "Pin the same version in requirements-serve.txt.",
                bundle.get("sklearn_version"), sklearn.__version__,
            )
        return cls(bundle["pipeline"], bundle["classes"])

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_classes,), for a single fish."""
        probs = self.pipeline.predict_proba(batch)
        return np.asarray(probs, dtype=np.float64)[0]
