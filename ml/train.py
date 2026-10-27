"""
train.py

Trains the convolutional autoencoder on "normal" seafloor tiles ONLY.
Validates on a held-out set of normal tiles (never anomalous ones —
the model should never see anomalies during training or validation,
only at final evaluation time).

Usage:
    python train.py

ADJUST ME:
- EPOCHS / BATCH_SIZE / LEARNING_RATE below if training is too slow,
  underfitting, or overfitting.
- TILES_ROOT if your tiled data lives somewhere other than data/tiles/.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from models.conv_ae import ConvAutoencoder

# ---- Config ----
TILES_ROOT = Path("data/tiles_drishti")
CHECKPOINT_PATH = Path("../data/checkpoints/conv_ae.pt")
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


class NormalTilesDataset(Dataset):
    """Loads all .png tiles from a given 'normal' folder."""

    def __init__(self, folder, transform):
        self.paths = sorted(Path(folder).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])  # already grayscale from tiling step
        return self.transform(image)


def main():
    transform = transforms.ToTensor()  # converts PIL image (0-255) -> tensor (0-1)

    train_dataset = NormalTilesDataset(TILES_ROOT / "train" / "normal", transform)
    val_dataset = NormalTilesDataset(TILES_ROOT / "valid" / "normal", transform)

    print(f"Training on {len(train_dataset)} normal tiles, validating on {len(val_dataset)}.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ---- Training pass ----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_dataset)

        # ---- Validation pass (no gradient updates) ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                loss = loss_fn(reconstruction, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f}")

        # Save the model whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best model saved (val_loss={val_loss:.5f})")

    print(f"Training done. Best val_loss: {best_val_loss:.5f}. Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()