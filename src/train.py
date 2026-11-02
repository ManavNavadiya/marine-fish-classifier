"""Phase 1 / Exp 4: train several model configurations, track them with MLflow,
and promote the best one (by VALIDATION accuracy) to models/fish_classifier.joblib.

Run from the project root:   python -m src.train
Then compare runs:           mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
"""

import json
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.preprocessing import MODEL_FEATURES, SPECIES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fish.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "fish_classifier.joblib"
MODEL_CARD_PATH = PROJECT_ROOT / "models" / "model_card.json"
CANDIDATES_DIR = PROJECT_ROOT / "models" / "candidates"
EXPERIMENT_NAME = "marine-fish-classifier"
SEED = 42

BUILDERS = {
    "decision_tree": lambda p: DecisionTreeClassifier(random_state=SEED, **p),
    "logistic_regression": lambda p: LogisticRegression(random_state=SEED, **p),
    "random_forest": lambda p: RandomForestClassifier(random_state=SEED, n_jobs=-1, **p),
}

# Each entry becomes one MLflow run. The first is a deliberately weak baseline.
CONFIGS = [
    {
        "run_name": "baseline_tree_depth2",
        "builder": "decision_tree",
        "params": {"max_depth": 2},
    },
    {
        "run_name": "logistic_regression",
        "builder": "logistic_regression",
        "params": {"C": 1.0, "max_iter": 2000},
    },
    {
        "run_name": "random_forest_200",
        "builder": "random_forest",
        "params": {"n_estimators": 200, "min_samples_leaf": 1},
    },
]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", BUILDERS[cfg["builder"]](cfg["params"])),
    ])


def run_config(cfg: dict, data) -> dict:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    set_seed()
    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params({
            "model_type": cfg["builder"],
            **{f"hp_{k}": v for k, v in cfg["params"].items()},
            "features": ",".join(MODEL_FEATURES),
            "n_features": len(MODEL_FEATURES),
            "training_samples": len(y_train),
            "validation_samples": len(y_val),
            "test_samples": len(y_test),
            "seed": SEED,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        })

        # Learning curve: train on growing slices of the training set.
        # Logged with step = % of data, like per-epoch curves for a neural net.
        for pct in (10, 25, 50, 75, 100):
            if pct < 100:
                X_part, _, y_part, _ = train_test_split(
                    X_train, y_train, train_size=pct / 100, random_state=SEED, stratify=y_train
                )
            else:
                X_part, y_part = X_train, y_train
            model = build_pipeline(cfg).fit(X_part, y_part)
            mlflow.log_metric("curve_train_accuracy", accuracy_score(y_part, model.predict(X_part)), step=pct)
            mlflow.log_metric("curve_val_accuracy", accuracy_score(y_val, model.predict(X_val)), step=pct)

        # Final model on the full training set.
        start = time.perf_counter()
        model = build_pipeline(cfg).fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        metrics = {
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
            "val_accuracy": accuracy_score(y_val, model.predict(X_val)),
            "val_f1_macro": f1_score(y_val, model.predict(X_val), average="macro", zero_division=0),
            "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
            "test_f1_macro": f1_score(y_test, model.predict(X_test), average="macro", zero_division=0),
            "fit_seconds": fit_seconds,
        }

        y_pred = model.predict(X_test)
        mlflow.log_dict(
            {"labels": SPECIES, "matrix": confusion_matrix(y_test, y_pred).tolist()},
            "confusion_matrix_test.json",
        )
        mlflow.log_text(
            classification_report(y_test, y_pred, target_names=SPECIES, digits=3, zero_division=0),
            "classification_report_test.txt",
        )

        # The bundle is what the API loads: model + the contract it was trained with.
        bundle = {
            "pipeline": model,
            "classes": SPECIES,
            "features": MODEL_FEATURES,
            "sklearn_version": sklearn.__version__,
            "run_name": cfg["run_name"],
        }
        CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        path = CANDIDATES_DIR / f"{cfg['run_name']}.joblib"
        joblib.dump(bundle, path, compress=3)
        metrics["model_size_kb"] = path.stat().st_size / 1024
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(path), artifact_path="model")

        print(f"  {cfg['run_name']:<24} val_acc={metrics['val_accuracy']:.3f} "
              f"val_f1={metrics['val_f1_macro']:.3f} test_acc={metrics['test_accuracy']:.3f}")
        return {"run_id": run.info.run_id, "run_name": cfg["run_name"], "path": path, **metrics}


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("No processed data. Run first:  python -m src.data_preprocessing")
    data = np.load(DATA_PATH)

    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Training {len(CONFIGS)} configurations (MLflow experiment '{EXPERIMENT_NAME}'):")
    results = [run_config(cfg, data) for cfg in CONFIGS]

    # Pick by validation score; the test set stays an unbiased final check.
    best = max(results, key=lambda r: (r["val_accuracy"], r["val_f1_macro"]))

    client = MlflowClient()
    for r in results:
        client.set_tag(r["run_id"], "selected_for_deployment", str(r is best).lower())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best["path"], MODEL_PATH)
    MODEL_CARD_PATH.write_text(json.dumps({
        "selected_run": best["run_name"],
        "mlflow_run_id": best["run_id"],
        "val_accuracy": round(best["val_accuracy"], 4),
        "val_f1_macro": round(best["val_f1_macro"], 4),
        "test_accuracy": round(best["test_accuracy"], 4),
        "test_f1_macro": round(best["test_f1_macro"], 4),
        "features": MODEL_FEATURES,
        "classes": SPECIES,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    print(f"\nSelected: {best['run_name']} (val_acc={best['val_accuracy']:.3f}, "
          f"test_acc={best['test_accuracy']:.3f})")
    print(f"Saved to {MODEL_PATH.relative_to(PROJECT_ROOT)} and {MODEL_CARD_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
