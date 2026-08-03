from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def degrade_batch(
    high_resolution: torch.Tensor,
    *,
    scale: int = 2,
    blur_sigma: float = 1.2,
    noise_std: float = 0.025,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale < 2:
        raise ValueError("scale must be at least 2")
    kernel = max(3, int(round(blur_sigma * 6)) | 1)
    blurred = gaussian_blur(high_resolution, [kernel, kernel], [blur_sigma, blur_sigma])
    low = F.interpolate(blurred, scale_factor=1.0 / scale, mode="bicubic", align_corners=False)
    if noise_std > 0:
        noise = torch.randn(low.shape, device=low.device, dtype=low.dtype, generator=generator) * noise_std
        low = torch.clamp(low + noise, 0.0, 1.0)
    bicubic = F.interpolate(low, size=high_resolution.shape[-2:], mode="bicubic", align_corners=False)
    return low, torch.clamp(bicubic, 0.0, 1.0)


def restore_methods(low: torch.Tensor, target_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    nearest = F.interpolate(low, size=target_size, mode="nearest")
    bilinear = F.interpolate(low, size=target_size, mode="bilinear", align_corners=False)
    bicubic = torch.clamp(F.interpolate(low, size=target_size, mode="bicubic", align_corners=False), 0.0, 1.0)
    smooth = gaussian_blur(bicubic, [3, 3], [0.8, 0.8])
    blurred = gaussian_blur(bicubic, [5, 5], [1.0, 1.0])
    unsharp = torch.clamp(bicubic + 0.8 * (bicubic - blurred), 0.0, 1.0)
    return {
        "raw_nearest": nearest,
        "bilinear": bilinear,
        "bicubic": bicubic,
        "smooth_bicubic": smooth,
        "unsharp_bicubic": unsharp,
    }


def forward_consistency(restored: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    projected = F.interpolate(restored, size=low.shape[-2:], mode="bicubic", align_corners=False)
    return ((projected - low) ** 2).flatten(1).mean(dim=1)
