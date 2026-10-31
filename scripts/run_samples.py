"""Sends every image in sample_images/ to the gateway and saves the JSON
response next to a copy of the visualization info. Prints a summary table.

Usage: python scripts/run_samples.py [--url http://localhost:8000]
"""
import argparse
import json
import os
import time

import requests

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_images")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    images = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    print(f"{'image':<55} {'products':>8} {'groups':>7} {'detect_ms':>10} {'group_ms':>9} {'total_ms':>9}")
    for name in images:
        path = os.path.join(SAMPLE_DIR, name)
        with open(path, "rb") as f:
            start = time.perf_counter()
            resp = requests.post(f"{args.url}/api/analyze", files={"image": (name, f, "image/jpeg")})
            wall_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            print(f"{name:<55} FAILED: {resp.status_code} {resp.text}")
            continue

        data = resp.json()
        result_path = os.path.join(RESULTS_DIR, f"{os.path.splitext(name)[0]}.json")
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)

        lat = data["latency_ms"]
        print(
            f"{name:<55} {data['num_products']:>8} {data['num_groups']:>7} "
            f"{lat['detect']:>10.1f} {lat['group']:>9.1f} {wall_ms:>9.1f}"
        )

    print(f"\nSaved per-image JSON results to {RESULTS_DIR}")
    print("Visualizations are in outputs/ (see each result's visualization_url).")


if __name__ == "__main__":
    main()
