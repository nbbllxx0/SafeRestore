from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, base: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base, dropout)
        self.enc2 = ConvBlock(base, base * 2, dropout)
        self.enc3 = ConvBlock(base * 2, base * 4, dropout)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, dropout)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.conv2(F.silu(self.conv1(x)))


class CompactRestorer(nn.Module):
    def __init__(self, channels: int = 32, blocks: int = 6) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.out_conv = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, bicubic: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.in_conv(bicubic))
        residual = self.out_conv(self.body(features))
        return torch.clamp(bicubic + residual, 0.0, 1.0)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, 1, bias=True)
        self.skip_conv = nn.Conv2d(skip_channels, inter_channels, 1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = torch.sigmoid(
            self.psi(F.silu(self.gate_conv(gate) + self.skip_conv(skip)))
        )
        return skip * attention


class AttentionUNet(nn.Module):
    """Attention-gated U-Net detector family (Oktay-style gates, compact width)."""

    def __init__(self, base: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base, dropout)
        self.enc2 = ConvBlock(base, base * 2, dropout)
        self.enc3 = ConvBlock(base * 2, base * 4, dropout)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = ConvBlock(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, base // 2 if base >= 2 else 1)
        self.dec1 = ConvBlock(base * 2, base, dropout)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        g2 = self.up2(e3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))
        return self.head(d1)


class UNetRestorer(nn.Module):
    """Encoder--decoder residual restorer; architecturally distinct from CompactRestorer."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec = ConvBlock(base * 2, base)
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, bicubic: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(bicubic)
        e2 = self.enc2(self.pool(e1))
        d1 = self.dec(torch.cat([self.up(e2), e1], dim=1))
        residual = self.out_conv(d1)
        return torch.clamp(bicubic + residual, 0.0, 1.0)


def build_detector(family: str, **kwargs: int | float) -> nn.Module:
    if family == "tiny_unet":
        return TinyUNet(base=int(kwargs.get("base", 16)), dropout=float(kwargs.get("dropout", 0.05)))
    if family == "attn_unet":
        return AttentionUNet(base=int(kwargs.get("base", 16)), dropout=float(kwargs.get("dropout", 0.05)))
    raise ValueError(f"unknown detector family: {family}")


def build_restorer(family: str, **kwargs: int) -> nn.Module:
    if family == "compact":
        return CompactRestorer(
            channels=int(kwargs.get("channels", 32)),
            blocks=int(kwargs.get("blocks", 6)),
        )
    if family == "unet":
        return UNetRestorer(base=int(kwargs.get("channels", 32)))
    raise ValueError(f"unknown restorer family: {family}")


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()
