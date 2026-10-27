"""
build_ranked_index.py

Computes final ensemble scores (normalized AE + baseline, averaged) for
every tile in a split, ranks them, and writes a single JSON index file
that the FastAPI backend serves directly -- avoids recomputing scores
on every API request.

Usage:
    python build_ranked_index.py --split test
"""

import argparse
import csv
import json
from pathlib import Path

SCORES_ROOT = Path("../data/scores")
OUTPUT_ROOT = Path("../data/index")


def load_scores(csv_path, score_column):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["filepath"], int(row["true_label"]), float(row[score_column])))
    return rows


def normalize(values):
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    return [(v - lo) / span for v in values]


def main(split):
    ae_rows = load_scores(SCORES_ROOT / f"{split}_ae_scores.csv", "ae_score")
    baseline_rows = load_scores(SCORES_ROOT / f"{split}_baseline_scores.csv", "baseline_score")

    # both CSVs were written from the same folder scan order, so they line up 1:1
    ae_scores_norm = normalize([s for _, _, s in ae_rows])
    baseline_scores_norm = normalize([s for _, _, s in baseline_rows])

    tiles = []
    for i, (filepath, true_label, ae_score) in enumerate(ae_rows):
        ensemble_score = (ae_scores_norm[i] + baseline_scores_norm[i]) / 2
        tiles.append({
            "tile_id": Path(filepath).stem,
            "filepath": filepath,
            "true_label": true_label,
            "ae_score": ae_scores_norm[i],
            "baseline_score": baseline_scores_norm[i],
            "ensemble_score": ensemble_score,
        })

    tiles.sort(key=lambda t: t["ensemble_score"], reverse=True)
    for rank, tile in enumerate(tiles, start=1):
        tile["rank"] = rank

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{split}_ranked.json"
    with open(output_path, "w") as f:
        json.dump(tiles, f, indent=2)

    print(f"Wrote {len(tiles)} ranked tiles to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    args = parser.parse_args()
    main(args.split)