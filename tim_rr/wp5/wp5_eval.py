"""WP5 inference library: per-image x per-action records from saved checkpoints.

Read-only on tim_upgrade/, scripts/, src/, splits/ and the raw data.  Nothing
here writes files; callers write only under tim_rr/wp5/.  No training occurs:
detector.pt / restorer.pt and the stored detector threshold are loaded from
the frozen run folders.

Three indication modes share one evaluation loop:

* ``nominal``  - the published degradation schedule
  (``saferestore.degradation_schedule.degrade_mixed_batch``) and the published
  candidate actions.  With this mode the loop reproduces
  ``scripts/run_dataset_pipeline.evaluate_split`` (Carinthia-S) and
  ``tim_upgrade/scripts/run_ksdd2_pipeline.evaluate_split_ksdd2`` (KSDD2)
  column for column (verified by the callers against the frozen CSVs).
* ``native``   - no synthetic degradation: the acquired reference image is the
  indication.  Candidates: raw (the image itself), Gaussian smoothing and
  unsharp sharpening with the pipeline's kernels applied at native resolution,
  and the learned restorer applied to the native image (it was trained on
  bicubic-upsampled degraded -> reference pairs).
* ``family:<name>`` - the published severity schedule, but one component of
  the degradation operator is replaced (motion blur, defocus disk PSF,
  Poisson-Gaussian noise, JPEG at a fixed quality).  Candidates as nominal.

For KSDD2 the loop evaluates several annotation-tolerance radii in one pass;
radius-dependent quantities (clean region, activation label, secondary_peak)
are emitted per radius together with boundary-band columns.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision.transforms.functional import gaussian_blur

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tim_upgrade" / "scripts"))

import run_ksdd2_pipeline as rk  # noqa: E402
from scripts import run_dataset_pipeline as pipeline  # noqa: E402
from saferestore.degradation import forward_consistency, restore_methods  # noqa: E402
from saferestore.degradation_schedule import degrade_mixed_batch, level_for_sample  # noqa: E402

DATA = {
    "carinthia_s": {
        "root": Path("<path to your Carinthia-S data directory>"),
        "split_file": REPO / "splits" / "carinthia_s_v4.json",
        "run": REPO / "experiments" / "v4_carinthia_primary_trainseed{seed}",
    },
    "kolektor_sdd2": {
        "root": Path("<path to your KolektorSDD2 dataset directory>"),
        "split_file": REPO / "tim_upgrade" / "splits" / "kolektor_sdd2_v1.json",
        # rich = records regenerated from the final checkpoints (same detector.pt / restorer.pt)
        "run": REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed{seed}",
    },
}
SPLIT_FILES = {"detector_val": "detector_val_records.csv",
               "calibration": "calibration_records.csv",
               "test": "test_records.csv"}

STANDARD_COLUMNS = (
    "sample_id", "split", "severity", "method", "has_defect", "pixel_recall",
    "clean_fpr", "deletion_error", "invention_error", "entropy",
    "forward_consistency", "detector_disagreement", "evidence_shift",
    "restoration_residual", "high_probability_area",
)
KSDD2_EXTRA = ("peak_probability", "peak_disagreement", "secondary_peak", "active_components")
BAND_COLUMNS = ("total_calls", "clean_calls", "ring_calls", "ring_pixels", "clean_pixels",
                "strict_clean_fpr", "strict_invention_error")


# ----------------------------------------------------------------------------- models

def load_models(domain: str, seed: int, device: torch.device):
    folder = Path(str(DATA[domain]["run"]).format(seed=seed))
    meta = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    detector = pipeline.build_detector(meta["detector_family"], base=meta["detector_base"]).to(device)
    detector.load_state_dict(torch.load(folder / "detector.pt", weights_only=True, map_location=device))
    restorer = pipeline.build_restorer(meta["restorer_family"], channels=meta["restorer_channels"],
                                       blocks=meta["restorer_blocks"]).to(device)
    restorer.load_state_dict(torch.load(folder / "restorer.pt", weights_only=True, map_location=device))
    detector.eval()
    restorer.eval()
    return detector, restorer, float(meta["detector_threshold"]), meta, folder


def make_loader(domain: str, split: str, seed: int, batch_size: int = 8):
    args = Namespace(root=DATA[domain]["root"], split_file=DATA[domain]["split_file"],
                     seed=seed, batch_size=batch_size, dataset=domain)
    if domain == "kolektor_sdd2":
        return rk.make_ksdd2_loader(args, split)
    return pipeline.make_loader(args, split)


# ----------------------------------------------------------------------------- degradation families

def _hash_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _conv_reflect(image: torch.Tensor, kernel: np.ndarray) -> torch.Tensor:
    k = torch.as_tensor(kernel, dtype=image.dtype, device=image.device)[None, None]
    ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = F.pad(image, (pw, pw, ph, ph), mode="reflect")
    return F.conv2d(padded, k)


def motion_kernel(length: float, angle_deg: float, oversample: int = 64) -> np.ndarray:
    """Linear motion PSF of the given length (px), bilinear-splatted and normalised."""
    half = int(math.ceil(length / 2.0)) + 1
    size = 2 * half + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    theta = math.radians(angle_deg)
    n = max(oversample, int(length * oversample))
    for t in np.linspace(-length / 2.0, length / 2.0, n):
        x = half + t * math.cos(theta)
        y = half - t * math.sin(theta)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                kernel[y0 + dy, x0 + dx] += wx * wy
    return (kernel / kernel.sum()).astype(np.float32)


def disk_kernel(radius: float, oversample: int = 16) -> np.ndarray:
    """Uniform defocus disk PSF of the given radius (px), area-supersampled."""
    half = int(math.ceil(radius))
    size = 2 * half + 1
    offsets = (np.arange(oversample) + 0.5) / oversample - 0.5
    kernel = np.zeros((size, size), dtype=np.float64)
    for iy in range(size):
        for ix in range(size):
            yy = (iy - half) + offsets[:, None]
            xx = (ix - half) + offsets[None, :]
            kernel[iy, ix] = np.mean(xx ** 2 + yy ** 2 <= radius ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def _downsample(blurred: torch.Tensor) -> torch.Tensor:
    return F.interpolate(blurred, scale_factor=0.5, mode="bicubic", align_corners=False)


def _gaussian_blur_like_pipeline(image: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = max(3, int(round(sigma * 6)) | 1)
    return gaussian_blur(image, [kernel, kernel], [sigma, sigma])


def family_params(family: str, level) -> dict:
    """Matched-severity parameters per published level (mild/moderate/severe)."""
    sb, sn = level.blur_sigma, level.noise_std
    if family == "motion":
        return {"length": math.sqrt(12.0) * sb, "noise_std": sn}  # same 2nd moment along motion
    if family == "defocus":
        return {"radius": 2.0 * sb, "noise_std": sn}  # disk per-axis sigma = radius/2
    if family == "poisson_gaussian":
        return {"blur_sigma": sb, "a": sn ** 2, "b": sn ** 2 / 2.0}  # var = a x + b; = sn^2 at x = 0.5
    if family.startswith("jpeg"):
        return {"blur_sigma": sb, "quality": int(family[4:])}
    raise ValueError(family)


def degrade_family_batch(high: torch.Tensor, sample_ids, *, split: str, corruption_key: str, family: str):
    """Published severity schedule with one operator component replaced."""
    if family == "nominal":
        return degrade_mixed_batch(high, sample_ids, split=split, corruption_key=corruption_key)
    lows, bicubics, names = [], [], []
    for index, sample_id in enumerate(sample_ids):
        level = level_for_sample(sample_id, split=split, corruption_key=corruption_key)
        params = family_params(family, level)
        generator = torch.Generator(device=high.device).manual_seed(
            _hash_int(f"noise:{corruption_key}:{split}:{sample_id}:{family}"))
        image = high[index:index + 1]
        if family in ("motion", "defocus"):
            if family == "motion":
                angle = (_hash_int(f"angle:{corruption_key}:{split}:{sample_id}") % 18000) / 100.0
                kernel = motion_kernel(params["length"], angle)
            else:
                kernel = disk_kernel(params["radius"])
            low = _downsample(_conv_reflect(image, kernel))
            noise = torch.randn(low.shape, device=low.device, dtype=low.dtype, generator=generator)
            low = torch.clamp(low + noise * params["noise_std"], 0.0, 1.0)
        elif family == "poisson_gaussian":
            low = torch.clamp(_downsample(_gaussian_blur_like_pipeline(image, params["blur_sigma"])), 0.0, 1.0)
            counts = torch.poisson(low / params["a"], generator=generator)
            gauss = torch.randn(low.shape, device=low.device, dtype=low.dtype, generator=generator)
            low = torch.clamp(counts * params["a"] + gauss * math.sqrt(params["b"]), 0.0, 1.0)
        elif family.startswith("jpeg"):
            low = torch.clamp(_downsample(_gaussian_blur_like_pipeline(image, params["blur_sigma"])), 0.0, 1.0)
            array = (low[0, 0].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG", quality=params["quality"])
            buffer.seek(0)
            decoded = np.asarray(Image.open(buffer).convert("L"), dtype=np.float32) / 255.0
            low = torch.from_numpy(decoded.copy())[None, None].to(high.device)
        else:
            raise ValueError(family)
        bicubic = torch.clamp(F.interpolate(low, size=high.shape[-2:], mode="bicubic", align_corners=False), 0.0, 1.0)
        lows.append(low)
        bicubics.append(bicubic)
        names.append(level.name)
    return torch.cat(lows, 0), torch.cat(bicubics, 0), names


def native_candidates(high: torch.Tensor, restorer) -> dict[str, torch.Tensor]:
    """Pipeline smooth / unsharp kernels applied at native resolution (no degradation)."""
    smooth = gaussian_blur(high, [3, 3], [0.8, 0.8])
    blurred = gaussian_blur(high, [5, 5], [1.0, 1.0])
    unsharp = torch.clamp(high + 0.8 * (high - blurred), 0.0, 1.0)
    return {"raw_nearest": high, "smooth_bicubic": smooth, "unsharp_bicubic": unsharp,
            "learned": restorer(high)}


# ----------------------------------------------------------------------------- structure features

def structure_features_multi(probability: torch.Tensor, threshold: float, margins):
    """rk._structure_features evaluated for several margins with one labelling.

    Same calls as the original (3x3 structure, binary_dilation with
    iterations=margin; note scipy repeats the dilation to convergence when
    iterations < 1, so margin 0 masks the whole connected image).
    """
    prob_cpu = probability.detach().cpu().numpy()
    calls = prob_cpu >= threshold
    structure = np.ones((3, 3), dtype=bool)
    n = prob_cpu.shape[0]
    secondary = {m: np.zeros(n, dtype=np.float32) for m in margins}
    components = np.zeros(n, dtype=np.float32)
    for i in range(n):
        call = calls[i, 0]
        labels, count = ndimage.label(call, structure=structure)
        components[i] = float(count)
        if count == 0:
            continue
        sizes = ndimage.sum(call, labels, index=np.arange(1, count + 1))
        largest = int(np.argmax(sizes)) + 1
        base = labels == largest
        for m in margins:
            keep_out = ndimage.binary_dilation(base, structure=structure, iterations=m)
            outside = prob_cpu[i, 0][~keep_out]
            secondary[m][i] = float(outside.max()) if outside.size else 0.0
    return secondary, components


# ----------------------------------------------------------------------------- evaluation loop

@torch.no_grad()
def evaluate(domain: str, split: str, seed: int, detector, restorer, threshold: float,
             device: torch.device, *, mode: str = "nominal", radii=(8,),
             corruption_key: str = "v4corr:eval", deletion_floor: float = 0.25,
             invention_limit: float = 2e-3, batch_size: int = 8) -> dict:
    """Return {radius: rows}.  Carinthia-S ignores ``radii`` (clean region = ~mask), key None."""
    ksdd2 = domain == "kolektor_sdd2"
    radii = tuple(radii) if ksdd2 else (None,)
    loader = make_loader(domain, split, seed, batch_size)
    out = {r: [] for r in radii}
    for batch in loader:
        high = batch["image"].to(device)
        mask_float = batch["mask"].to(device)
        mask = mask_float.bool()
        ids = batch["sample_id"]
        if mode == "native":
            low = high
            severities = ["native"] * high.shape[0]
            candidates = native_candidates(high, restorer)
        else:
            family = "nominal" if mode == "nominal" else mode.split(":", 1)[1]
            low, bicubic, severities = degrade_family_batch(
                high, ids, split=split, corruption_key=corruption_key, family=family)
            candidates = restore_methods(low, tuple(high.shape[-2:]))
            candidates["learned"] = restorer(bicubic)
        raw_probability = torch.sigmoid(detector(candidates["raw_nearest"]))
        positive_pixels = mask.flatten(1).sum(1)
        excluded = {}
        for r in radii:
            if r is None or r == 0:
                excluded[r] = mask
            else:
                excluded[r] = F.max_pool2d(mask_float, kernel_size=2 * r + 1, stride=1, padding=r).bool()
        for method, candidate in candidates.items():
            probability = torch.sigmoid(detector(candidate))
            call = probability >= threshold
            true_positive = (call & mask).flatten(1).sum(1)
            recall = torch.where(positive_pixels > 0,
                                 true_positive.float() / positive_pixels.clamp_min(1),
                                 torch.ones_like(true_positive, dtype=torch.float32))
            deletion_error = (positive_pixels > 0) & (recall < deletion_floor)
            residual = (candidate - candidates["raw_nearest"]).abs().flatten(1).mean(1)
            disagreement_map = (probability - raw_probability).abs()
            disagreement = disagreement_map.flatten(1).mean(1)
            entropy = pipeline._binary_entropy(probability).flatten(1).mean(1)
            evidence_shift = (probability.flatten(1).mean(1) - raw_probability.flatten(1).mean(1)).abs()
            high_area = (probability >= 0.5).flatten(1).float().mean(1)
            forward = forward_consistency(candidate, low)
            mse = ((candidate - high) ** 2).flatten(1).mean(1)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            total_calls = call.flatten(1).sum(1)
            if ksdd2:
                peak_probability = rk._top_fraction_mean(probability)
                peak_disagreement = rk._top_fraction_mean(disagreement_map)
                secondary, components = structure_features_multi(
                    probability, threshold, [r for r in radii])
            for r in radii:
                exc = excluded[r]
                negative_pixels = (~exc).flatten(1).sum(1).clamp_min(1)
                false_positive = (call & ~exc).flatten(1).sum(1)
                clean_fpr = false_positive.float() / negative_pixels
                invention_error = clean_fpr > invention_limit
                ring = exc & ~mask
                ring_calls = (call & ring).flatten(1).sum(1)
                ring_pixels = ring.flatten(1).sum(1)
                strict_fpr = (false_positive + ring_calls).float() / negative_pixels
                strict_inv = strict_fpr > invention_limit
                for item in range(high.shape[0]):
                    row = {
                        "sample_id": ids[item], "split": split, "severity": severities[item],
                        "method": method,
                        "has_defect": int(positive_pixels[item].item() > 0),
                        "pixel_recall": float(recall[item].cpu()),
                        "clean_fpr": float(clean_fpr[item].cpu()),
                        "deletion_error": int(deletion_error[item].cpu()),
                        "invention_error": int(invention_error[item].cpu()),
                        "entropy": float(entropy[item].cpu()),
                        "forward_consistency": float(forward[item].cpu()),
                        "detector_disagreement": float(disagreement[item].cpu()),
                        "evidence_shift": float(evidence_shift[item].cpu()),
                        "restoration_residual": float(residual[item].cpu()),
                        "high_probability_area": float(high_area[item].cpu()),
                    }
                    if ksdd2:
                        row.update({
                            "peak_probability": float(peak_probability[item].cpu()),
                            "peak_disagreement": float(peak_disagreement[item].cpu()),
                            "secondary_peak": float(secondary[r][item]),
                            "active_components": float(components[item]),
                        })
                    row["psnr"] = float(psnr[item].cpu())
                    row.update({
                        "total_calls": int(total_calls[item].cpu()),
                        "clean_calls": int(false_positive[item].cpu()),
                        "ring_calls": int(ring_calls[item].cpu()),
                        "ring_pixels": int(ring_pixels[item].cpu()),
                        "clean_pixels": int(negative_pixels[item].cpu()),
                        "strict_clean_fpr": float(strict_fpr[item].cpu()),
                        "strict_invention_error": int(strict_inv[item].cpu()),
                    })
                    out[r].append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    pipeline.write_csv(path, rows)


def compare_records(new_rows: list[dict], frozen_path: Path, rel_tol: float = 0.0) -> dict:
    """Column-by-column comparison of regenerated rows with a frozen CSV."""
    import csv
    frozen = list(csv.DictReader(frozen_path.open(encoding="utf-8")))
    report = {"frozen": str(frozen_path), "n_new": len(new_rows), "n_frozen": len(frozen),
              "columns": {}, "label_mismatches": 0}
    if len(frozen) != len(new_rows):
        report["error"] = "row count differs"
        return report
    columns = [c for c in frozen[0].keys()]
    for col in columns:
        worst = 0.0
        mismatches = 0
        for a, b in zip(new_rows, frozen):
            va, vb = a[col], b[col]
            if col in ("sample_id", "split", "severity", "method"):
                if str(va) != vb:
                    mismatches += 1
                continue
            fa, fb = float(va), float(vb)
            if fa != fb:
                mismatches += 1
                worst = max(worst, abs(fa - fb))
        report["columns"][col] = {"mismatches": mismatches, "max_abs_diff": worst}
    report["label_mismatches"] = (report["columns"]["deletion_error"]["mismatches"]
                                  + report["columns"]["invention_error"]["mismatches"])
    report["exact"] = all(v["mismatches"] == 0 for v in report["columns"].values())
    return report
