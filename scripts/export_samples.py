"""Export held-out test rows as JSON request bodies, with the true label in the file name.

Usage:  python scripts/export_samples.py --count 20
Output: samples/000_atlantic_salmon.json, ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.preprocessing import RAW_FEATURES, SPECIES  # noqa: E402

DATA_FILE = ROOT / "data" / "processed" / "fish.npz"
OUT_DIR = ROOT / "samples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit("Run first:  python -m src.data_preprocessing")
    data = np.load(DATA_FILE)
    raw = data["X_test"][:, : len(RAW_FEATURES)]
    labels = data["y_test"]

    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("*.json"):
        old.unlink()

    for i in range(min(args.count, len(labels))):
        body = {name: round(float(v), 3) for name, v in zip(RAW_FEATURES, raw[i])}
        body["fin_count"] = int(body["fin_count"])
        slug = SPECIES[int(labels[i])].lower().replace(" ", "_")
        (OUT_DIR / f"{i:03d}_{slug}.json").write_text(json.dumps(body))

    print(f"Wrote {min(args.count, len(labels))} samples to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
