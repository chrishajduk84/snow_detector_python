"""1D U-Net model for per-range-bin dielectric property estimation.

Input:  (batch, num_rx * 2, num_range_bins) — real + imaginary channels
Output: (batch, 3, num_range_bins) — [eps_r, tan_delta, presence] per bin

The architecture uses an encoder-decoder with skip connections (U-Net style)
to predict material properties at each range bin of the radar profile.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two Conv1d layers with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PropertyEstimator(nn.Module):
    """1D U-Net for dielectric property estimation.

    Args:
        in_channels: Number of input channels (default: 6 for 3 RX × 2 real/imag).
        base_filters: Number of filters in the first encoder level.
        num_range_bins: Expected number of range bins (for documentation; model is flexible).
    """

    def __init__(self, in_channels: int = 6, base_filters: int = 32):
        super().__init__()

        f = base_filters

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 4, f * 8)

        # Decoder (with skip connections)
        self.up3 = nn.ConvTranspose1d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)  # concat with enc3

        self.up2 = nn.ConvTranspose1d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)  # concat with enc2

        self.up1 = nn.ConvTranspose1d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)  # concat with enc1

        self.pool = nn.MaxPool1d(2)

        # Output heads
        self.eps_head = nn.Sequential(
            nn.Conv1d(f, f // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(f // 2, 1, kernel_size=1),
            nn.Softplus(),  # ensures positive output
        )

        self.tan_delta_head = nn.Sequential(
            nn.Conv1d(f, f // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(f // 2, 1, kernel_size=1),
            nn.Sigmoid(),  # bounded [0, 1]
        )

        self.presence_head = nn.Sequential(
            nn.Conv1d(f, f // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(f // 2, 1, kernel_size=1),
            nn.Sigmoid(),  # binary
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, num_range_bins)

        Returns:
            (batch, 3, num_range_bins) — [eps_r, tan_delta, presence]
        """
        # Pad input to be divisible by 8 (3 levels of pooling)
        orig_len = x.shape[2]
        pad_len = (8 - orig_len % 8) % 8
        if pad_len > 0:
            x = nn.functional.pad(x, (0, pad_len))

        # Encoder
        e1 = self.enc1(x)          # (B, f, L)
        e2 = self.enc2(self.pool(e1))   # (B, 2f, L/2)
        e3 = self.enc3(self.pool(e2))   # (B, 4f, L/4)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 8f, L/8)

        # Decoder
        d3 = self.up3(b)                    # (B, 4f, L/4)
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3)                  # (B, 4f, L/4)

        d2 = self.up2(d3)                   # (B, 2f, L/2)
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2)                  # (B, 2f, L/2)

        d1 = self.up1(d2)                   # (B, f, L)
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1)                  # (B, f, L)

        # Output heads
        eps_r = self.eps_head(d1) + 1.0     # shift so minimum is 1.0 (vacuum)
        tan_delta = self.tan_delta_head(d1)
        presence = self.presence_head(d1)

        out = torch.cat([eps_r, tan_delta, presence], dim=1)

        # Remove padding
        if pad_len > 0:
            out = out[:, :, :orig_len]

        return out

    @staticmethod
    def _match_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Concatenate upsampled and skip tensors, handling size mismatches."""
        diff = skip.shape[2] - upsampled.shape[2]
        if diff > 0:
            upsampled = nn.functional.pad(upsampled, (0, diff))
        elif diff < 0:
            upsampled = upsampled[:, :, :skip.shape[2]]
        return torch.cat([upsampled, skip], dim=1)
