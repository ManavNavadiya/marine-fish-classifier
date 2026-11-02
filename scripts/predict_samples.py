"""Send every sample in samples/ to the API and print accuracy.

Usage:  python scripts/predict_samples.py
        python scripts/predict_samples.py --url http://<ec2-public-ip>
"""

import argparse
import json
from pathlib import Path

import httpx

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    files = sorted(SAMPLES_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No samples. Run:  python scripts/export_samples.py --count 20")

    correct = 0
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for path in files:
            truth = path.stem.split("_", 1)[1].replace("_", " ").title()
            res = client.post("/predict", json=json.loads(path.read_text()))
            res.raise_for_status()
            body = res.json()
            ok = body["species"] == truth
            correct += ok
            print(f"{'OK ' if ok else 'XX '} {path.name:<32} -> {body['species']:<18} "
                  f"({body['confidence']:.2f})")

    print(f"\nAccuracy: {correct}/{len(files)} = {correct / len(files):.1%}")


if __name__ == "__main__":
    main()
