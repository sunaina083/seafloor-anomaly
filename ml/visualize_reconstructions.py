"""
visualize_reconstructions.py

Diagnostic tool: picks a handful of normal and anomalous tiles, runs them
through the trained autoencoder, and saves a side-by-side image showing:
  original tile | reconstruction | error map (brighter = higher error)

This helps answer: is the model actually reacting differently to debris
vs. plain seafloor, or is something else going wrong?

Usage:
    python visualize_reconstructions.py
"""

from pathlib import Path

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models.conv_ae import ConvAutoencoder

TILES_ROOT = Path("../data/tiles/test")
CHECKPOINT_PATH = Path("../data/checkpoints/conv_ae.pt")
OUTPUT_PATH = Path("../data/diagnostics/reconstructions.png")
N_SAMPLES = 6  # how many of each (normal / anomalous) to show


def main():
    device = torch.device("cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    transform = transforms.ToTensor()

    normal_paths = sorted((TILES_ROOT / "normal").glob("*.png"))[:N_SAMPLES]
    anomalous_paths = sorted((TILES_ROOT / "anomalous").glob("*.png"))[:N_SAMPLES]
    all_paths = [(p, "normal") for p in normal_paths] + [(p, "anomalous") for p in anomalous_paths]

    fig, axes = plt.subplots(len(all_paths), 3, figsize=(6, 2 * len(all_paths)))

    with torch.no_grad():
        for i, (path, label) in enumerate(all_paths):
            image = Image.open(path)
            tensor = transform(image).unsqueeze(0)
            reconstruction = model(tensor)
            error_map = (reconstruction - tensor) ** 2

            original_np = tensor.squeeze().numpy()
            recon_np = reconstruction.squeeze().numpy()
            error_np = error_map.squeeze().numpy()

            axes[i, 0].imshow(original_np, cmap="gray")
            axes[i, 0].set_title(f"{label}: original")
            axes[i, 1].imshow(recon_np, cmap="gray")
            axes[i, 1].set_title("reconstruction")
            axes[i, 2].imshow(error_np, cmap="hot")
            axes[i, 2].set_title(f"error (score={error_np.mean():.4f})")

            for ax in axes[i]:
                ax.axis("off")

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=120)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()