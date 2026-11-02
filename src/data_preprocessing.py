"""Phase 1 / Exp 4: ETL pipeline -> feature store.

Extract   : produce a raw "fisheries survey" CSV (synthetic, with realistic
            dirt: missing values, duplicates, messy names, impossible values)
Transform : clean it, then build model features with the SHARED code in
            app/preprocessing.py
Load      : split into train/val/test and save to data/processed/fish.npz
            (our simple "feature store") plus an ETL report.

Run from the project root:   python -m src.data_preprocessing
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.preprocessing import (
    MODEL_FEATURES,
    RAW_FEATURES,
    SPECIES,
    engineer_features,
    valid_rows_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "fish_survey_raw.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "fish.npz"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "etl_report.json"

SEED = 42
ROWS_PER_SPECIES = 300

# Typical adult measurements per species:
# length_cm, weight_kg, height_cm, fin_count, water_temp_c
SPECIES_PROFILES = {
    "Atlantic Herring": (30, 0.25, 6, 7, 10),
    "Atlantic Mackerel": (35, 0.50, 7, 9, 13),
    "Atlantic Salmon": (75, 4.50, 15, 8, 10),
    "Bluefin Tuna": (200, 200.0, 50, 9, 18),
    "Clownfish": (9, 0.02, 4, 7, 26),
    "European Sardine": (20, 0.08, 4, 7, 17),
}


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
def extract(rows_per_species: int = ROWS_PER_SPECIES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for name, (length, weight, height, fins, temp) in SPECIES_PROFILES.items():
        n = rows_per_species
        scale = rng.normal(1.0, 0.18, n).clip(0.4, 1.8)  # how big this fish is
        frames.append(pd.DataFrame({
            "length_cm": length * scale,
            # weight grows with length cubed, plus individual variation
            "weight_kg": weight * scale ** 3 * rng.normal(1.0, 0.12, n).clip(0.5, 1.5),
            "height_cm": height * scale * rng.normal(1.0, 0.08, n).clip(0.7, 1.3),
            "fin_count": rng.integers(fins - 1, fins + 2, n),
            "water_temp_c": rng.normal(temp, 3.0, n),
            "species": name,
        }))
    df = pd.concat(frames, ignore_index=True)
    df = df.round({"length_cm": 1, "weight_kg": 3, "height_cm": 1, "water_temp_c": 1})

    # --- add the kind of mess real survey exports have ---
    n = len(df)
    numeric = ["length_cm", "weight_kg", "height_cm", "water_temp_c"]
    df[numeric] = df[numeric].astype("float64")
    for _ in range(int(n * 0.02)):  # 2 % missing cells
        df.loc[rng.integers(n), rng.choice(numeric)] = np.nan
    bad = rng.choice(n, int(n * 0.01), replace=False)  # 1 % impossible values
    df.loc[bad, "weight_kg"] = -1.0
    messy = rng.choice(n, int(n * 0.10), replace=False)  # 10 % messy labels
    df["species"] = df["species"].astype("object")
    df.loc[messy, "species"] = [
        rng.choice([s.lower(), s.upper(), f"  {s} "]) for s in df.loc[messy, "species"]
    ]
    unknown = rng.choice(n, 5, replace=False)
    df.loc[unknown, "species"] = "unknown"
    dupes = df.sample(int(n * 0.01), random_state=seed)  # 1 % duplicate rows
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def transform(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"raw_rows": int(len(raw))}
    df = raw.copy()

    df["species"] = df["species"].astype(str).str.strip().str.title()
    known = df["species"].isin(SPECIES)
    report["dropped_unknown_species"] = int((~known).sum())
    df = df[known]

    before = len(df)
    df = df.dropna(subset=RAW_FEATURES)
    report["dropped_missing_values"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicates"] = int(before - len(df))

    before = len(df)
    df = df[valid_rows_mask(df[RAW_FEATURES].to_numpy())]
    report["dropped_out_of_range"] = int(before - len(df))

    df = df.astype({"fin_count": int}).reset_index(drop=True)
    report["clean_rows"] = int(len(df))
    report["rows_per_species"] = {k: int(v) for k, v in df["species"].value_counts().sort_index().items()}
    return df, report


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load(df: pd.DataFrame, report: dict, seed: int = SEED) -> None:
    X = engineer_features(df[RAW_FEATURES].to_numpy())
    y = df["species"].map({name: i for i, name in enumerate(SPECIES)}).to_numpy()

    # 70 % train, 15 % validation (model selection), 15 % test (final, untouched)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(MODEL_FEATURES),
        class_names=np.array(SPECIES),
    )
    report["split"] = {"train": len(y_train), "val": len(y_val), "test": len(y_test)}
    report["features"] = MODEL_FEATURES
    REPORT_PATH.write_text(json.dumps(report, indent=2))


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = extract()
    raw.to_csv(RAW_PATH, index=False)
    print(f"[extract]   {len(raw)} raw rows -> {RAW_PATH.relative_to(PROJECT_ROOT)}")

    clean, report = transform(pd.read_csv(RAW_PATH))
    print(f"[transform] {report['clean_rows']} clean rows "
          f"(dropped: {report['dropped_unknown_species']} unknown species, "
          f"{report['dropped_missing_values']} missing, "
          f"{report['dropped_duplicates']} duplicates, "
          f"{report['dropped_out_of_range']} out of range)")

    load(clean, report)
    print(f"[load]      split {report['split']} -> {PROCESSED_PATH.relative_to(PROJECT_ROOT)}")
    print(f"            report -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
