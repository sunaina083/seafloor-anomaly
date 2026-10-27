"""
prepare_drishti.py

Converts the drishti-sss dataset into the same data/tiles/<split>/<normal|anomalous>/
format our existing pipeline (train.py, score_tiles.py, evaluate.py) already expects
-- so nothing downstream needs to change.

Unlike the old KLSG/Roboflow dataset, these images are already pre-cropped,
purpose-built detection tiles (not full sonar swaths), so each image becomes
ONE tile directly -- no grid slicing needed. We resize each to 64x64 to match
the existing autoencoder architecture.

Classes (from drishti.yaml):
    0: crab_pot          -> anomaly (lost fishing gear)
    1: submarine_pipeline -> normal (routine infrastructure, not debris)
    2: shipwreck          -> anomaly
    3: ghost_net          -> anomaly
    4: mine_cylinder      -> anomaly

Usage:
    python prepare_drishti.py --split train
    python prepare_drishti.py --split val
    python prepare_drishti.py --split test
"""

import argparse
from pathlib import Path

from PIL import Image

DRISHTI_ROOT = Path("../../data/raw_drishti")
OUTPUT_ROOT = Path("../data/tiles_drishti")  # separate from the old data/tiles/ -- keeps old data intact
TILE_SIZE = 64

ANOMALY_CLASS_IDS = {0, 2, 3, 4}  # crab_pot, shipwreck, ghost_net, mine_cylinder
# class 1 (submarine_pipeline) is intentionally left out -- treated as normal infrastructure

# The dataset uses "val", our existing scripts expect "valid" -- map the folder name.
SPLIT_NAME_MAP = {"train": "train", "val": "valid", "test": "test"}


def has_anomaly(label_path):
    """Return True if the label file contains any anomaly-class box."""
    if not label_path.exists():
        return False  # no label file = no boxes at all = normal
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in ANOMALY_CLASS_IDS:
                return True
    return False


def main(split):
    images_dir = DRISHTI_ROOT / split / "images"
    labels_dir = DRISHTI_ROOT / split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Couldn't find {images_dir} -- check DRISHTI_ROOT.")

    output_split = SPLIT_NAME_MAP[split]
    output_dir = OUTPUT_ROOT / output_split

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"Found {len(image_paths)} images in {split} split.")

    normal_count, anomaly_count = 0, 0
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        is_anomalous = has_anomaly(label_path)

        image = Image.open(image_path).convert("L").resize((TILE_SIZE, TILE_SIZE))

        label_folder = "anomalous" if is_anomalous else "normal"
        out_dir = output_dir / label_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_dir / f"{image_path.stem}.png")

        if is_anomalous:
            anomaly_count += 1
        else:
            normal_count += 1

    total = normal_count + anomaly_count
    print(f"Done. {output_split}: {normal_count} normal, {anomaly_count} anomalous "
          f"({anomaly_count / total:.1%} anomalous).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = parser.parse_args()
    main(args.split)