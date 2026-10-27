"""
conv_ae.py

A small convolutional autoencoder for 80x80 grayscale sonar tiles.

Encoder: 80x80 -> 40x40 -> 20x20 -> 10x10 -> 5x5, increasing channel depth
Decoder: mirrors the encoder in reverse to reconstruct the original tile

The model is trained ONLY on "normal" seafloor tiles. At inference time,
tiles it reconstructs poorly (high error) are flagged as anomalous.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels=16):
        super().__init__()

        # ---- Encoder ----
        # Each block halves spatial size and (usually) increases channels.
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #
        # IMPORTANT: the bottleneck (end of encoder) must be much SMALLER
        # than the input (80*80=6400 values), or the autoencoder can just
        # pass information straight through without learning what's
        # "typical" -- which would mean it reconstructs anomalies just as
        # well as normal tiles, giving us no anomaly signal at all.
        # Here: 5*5*16 = 400 values, a 16x compression from the 6400-value input.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),    # 80 -> 40
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),   # 40 -> 20
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),   # 20 -> 10
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, kernel_size=4, stride=2, padding=1),  # 10 -> 5
            nn.ReLU(inplace=True),
        )

        # ---- Decoder ----
        # ConvTranspose2d upsamples back up, mirroring the encoder.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1),  # 5 -> 10
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 10 -> 20
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 20 -> 40
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # 40 -> 80
            nn.Sigmoid(),  # output pixels squashed to [0, 1] to match normalized input
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction