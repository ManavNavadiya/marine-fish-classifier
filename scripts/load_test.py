"""Generate traffic so the Grafana dashboard has something to show.

Sends mostly valid predictions plus a share of deliberately bad requests
(non-JSON, out-of-range values, missing fields, broken JSON) to exercise
the error handling.

Usage:  python scripts/load_test.py --url http://<ec2-ip> --requests 300
"""

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"

TYPICAL_FISH = [
    {"length_cm": 30, "weight_kg": 0.25, "height_cm": 6, "fin_count": 7, "water_temp_c": 10},
    {"length_cm": 35, "weight_kg": 0.50, "height_cm": 7, "fin_count": 9, "water_temp_c": 13},
    {"length_cm": 75, "weight_kg": 4.50, "height_cm": 15, "fin_count": 8, "water_temp_c": 10},
    {"length_cm": 200, "weight_kg": 200.0, "height_cm": 50, "fin_count": 9, "water_temp_c": 18},
    {"length_cm": 9, "weight_kg": 0.02, "height_cm": 4, "fin_count": 7, "water_temp_c": 26},
    {"length_cm": 20, "weight_kg": 0.08, "height_cm": 4, "fin_count": 7, "water_temp_c": 17},
]

JSON = "application/json"


def _jitter(fish: dict) -> dict:
    """A typical fish with +-10 % natural variation."""
    out = {k: round(v * random.uniform(0.9, 1.1), 3) for k, v in fish.items()}
    out["fin_count"] = fish["fin_count"]
    out["water_temp_c"] = round(fish["water_temp_c"] + random.uniform(-2, 2), 1)
    return out


def build_requests(n: int, error_share: float):
    samples = [json.loads(p.read_text()) for p in sorted(SAMPLES_DIR.glob("*.json"))]
    jobs = []
    for _ in range(n):
        if random.random() < error_share:
            fish = random.choice(TYPICAL_FISH)
            jobs.append(random.choice([
                (b"not json at all", "text/plain"),                                  # -> 415
                (json.dumps({**fish, "weight_kg": -5}).encode(), JSON),             # -> 400
                (json.dumps({k: v for k, v in fish.items() if k != "length_cm"}).encode(), JSON),  # -> 422
                (b'{"length_cm": 30, ', JSON),                                       # -> 422
            ]))
        else:
            fish = random.choice(samples) if samples and random.random() < 0.5 else _jitter(
                random.choice(TYPICAL_FISH))
            jobs.append((json.dumps(fish).encode(), JSON))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="e.g. http://<public-ip>")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--errors", type=float, default=0.15, help="share of bad requests")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between requests per worker")
    args = parser.parse_args()

    jobs = build_requests(args.requests, args.errors)
    statuses = Counter()
    latencies = []

    with httpx.Client(base_url=args.url, timeout=15) as client:
        def send(job):
            body, ctype = job
            start = time.perf_counter()
            try:
                res = client.post("/predict", content=body, headers={"Content-Type": ctype})
                code = res.status_code
            except httpx.HTTPError as exc:
                code = type(exc).__name__
            latencies.append(time.perf_counter() - start)
            statuses[code] += 1
            time.sleep(args.delay)

        print(f"Sending {len(jobs)} requests to {args.url} with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(send, jobs))

    latencies.sort()
    print("\nStatus codes:", dict(statuses))
    print(f"Median latency: {latencies[len(latencies) // 2] * 1000:.0f} ms, "
          f"p95: {latencies[int(len(latencies) * 0.95)] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
