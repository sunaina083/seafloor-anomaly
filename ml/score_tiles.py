"""
score_tiles.py

Loads the trained autoencoder and scores every tile in a given split
(both normal and anomalous) by reconstruction error. Higher error =
more anomalous, according to the model.

Outputs a CSV with columns: filepath, true_label, ae_score
  - true_label is 0 for normal, 1 for anomalous (from our tiling step's
    folder split) -- used ONLY for evaluation, never for training.

Usage:
    python score_tiles.py --split test
"""

import argparse
import csv
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from models.conv_ae import ConvAutoencoder

TILES_ROOT = Path("data/tiles_drishti")
CHECKPOINT_PATH = Path("../data/checkpoints/conv_ae.pt")
OUTPUT_ROOT = Path("../data/scores")


def load_model(device):
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()  # inference mode -- disables dropout/batchnorm updates (we have neither, but good habit)
    return model


def score_folder(model, folder, true_label, transform, device):
    """Score every tile in `folder`, returning a list of (filepath, true_label, score) rows."""
    rows = []
    paths = sorted(Path(folder).glob("*.png"))
    with torch.no_grad():  # no gradients needed at inference -- saves memory and time
        for path in paths:
            image = Image.open(path)
            tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension: (1, 1, 80, 80)
            
            reconstruction = model(tensor)
            # Per-pixel squared error, then take the mean of the worst 10% of
            # pixels -- not the whole-tile mean. Debris/anomalies usually only
            # cover part of an 80x80 tile, so a whole-tile average dilutes a
            # small but badly-reconstructed region among thousands of
            # correctly-reconstructed background pixels. Focusing on the worst
            # pixels makes localized reconstruction failures stand out clearly.
            error_map = (reconstruction - tensor) ** 2
            flat_errors = error_map.flatten()
            k = max(1, int(0.10 * flat_errors.numel()))  # worst 10% of pixels
            top_k_errors, _ = torch.topk(flat_errors, k)
            score = top_k_errors.mean().item()
            rows.append((str(path), true_label, score))
    return rows


def main(split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    transform = transforms.ToTensor()

    normal_folder = TILES_ROOT / split / "normal"
    anomalous_folder = TILES_ROOT / split / "anomalous"

    print(f"Scoring normal tiles in {normal_folder} ...")
    normal_rows = score_folder(model, normal_folder, true_label=0, transform=transform, device=device)

    print(f"Scoring anomalous tiles in {anomalous_folder} ...")
    anomalous_rows = score_folder(model, anomalous_folder, true_label=1, transform=transform, device=device)

    all_rows = normal_rows + anomalous_rows

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{split}_ae_scores.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_label", "ae_score"])
        writer.writerows(all_rows)

    print(f"Scored {len(all_rows)} tiles ({len(normal_rows)} normal, {len(anomalous_rows)} anomalous).")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score tiles with the trained autoencoder.")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    args = parser.parse_args()
    main(args.split)