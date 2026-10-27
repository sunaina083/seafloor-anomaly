"""
score_tiles_baseline.py

Computes a naive "anomaly score" for every tile using simple intensity
statistics -- no learning involved. This is our baseline: the thing the
autoencoder needs to beat in order to justify using ML at all.

Baseline logic: sonar backscatter from solid objects (metal, wreckage)
tends to reflect much more strongly than soft sediment, producing bright
highlights (and often a dark acoustic shadow right behind the object).
So our baseline score = the tile's max pixel intensity -- a simple proxy
for "does this tile contain something unusually reflective."

Outputs a CSV with the same format as score_tiles.py, so the eval script
can compare both methods side by side.

Usage:
    python score_tiles_baseline.py --split test
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

TILES_ROOT = Path("data/tiles_drishti")
OUTPUT_ROOT = Path("../data/scores")


def score_folder(folder, true_label):
    rows = []
    paths = sorted(Path(folder).glob("*.png"))
    for path in paths:
        image = np.array(Image.open(path), dtype=np.float32) / 255.0  # normalize to 0-1, same scale as AE
        score = float(image.max())  # baseline score: brightest pixel in the tile
        rows.append((str(path), true_label, score))
    return rows


def main(split):
    normal_folder = TILES_ROOT / split / "normal"
    anomalous_folder = TILES_ROOT / split / "anomalous"

    print(f"Scoring normal tiles in {normal_folder} ...")
    normal_rows = score_folder(normal_folder, true_label=0)

    print(f"Scoring anomalous tiles in {anomalous_folder} ...")
    anomalous_rows = score_folder(anomalous_folder, true_label=1)

    all_rows = normal_rows + anomalous_rows

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{split}_baseline_scores.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_label", "baseline_score"])
        writer.writerows(all_rows)

    print(f"Scored {len(all_rows)} tiles ({len(normal_rows)} normal, {len(anomalous_rows)} anomalous).")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score tiles with the naive intensity baseline.")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    args = parser.parse_args()
    main(args.split)