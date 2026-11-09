"""Run the SafeRestore pipeline on KolektorSDD2 for the TIM upgrade.

Reuses the frozen training/evaluation logic from
``scripts/run_dataset_pipeline.py`` verbatim by substituting only the data
loader (KSDD2 dataset class plus a defect-balanced sampler driven by the
split manifest).  Nothing outside ``tim_upgrade/`` is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))          # scripts package
sys.path.insert(0, str(REPO / "src"))  # saferestore library
sys.path.insert(0, str(Path(__file__).resolve().parent))  # ksdd2_data

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ksdd2_data import Ksdd2SegmentationDataset
from scripts import run_dataset_pipeline as pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one KolektorSDD2 SafeRestore experiment.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--detector-epochs", type=int, default=12)
    parser.add_argument("--restorer-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--corruption-key", default="v4corr:eval")
    parser.add_argument("--detector-family", default="tiny_unet", choices=("tiny_unet", "attn_unet"))
    parser.add_argument("--restorer-family", default="compact", choices=("compact", "unet"))
    parser.add_argument("--detector-base", type=int, default=16)
    parser.add_argument("--restorer-channels", type=int, default=32)
    parser.add_argument("--restorer-blocks", type=int, default=6)
    parser.add_argument("--target-pixel-fpr", type=float, default=1e-3)
    parser.add_argument("--deletion-recall-floor", type=float, default=0.25)
    parser.add_argument("--invention-fpr-limit", type=float, default=2e-3)
    parser.add_argument("--risk-deletion", type=float, default=0.15)
    parser.add_argument("--risk-invention", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument(
        "--boundary-tolerance", type=int, default=8,
        help="Pixel radius of the annotation-boundary tolerance ring excluded "
             "from the clean region (KolektorSDD2 annotations are coarse; the "
             "dataset authors' own evaluation dilates ground truth).",
    )
    args = parser.parse_args()
    args.dataset = "kolektor_sdd2"
    return args


def make_ksdd2_loader(
    args: argparse.Namespace,
    split: str,
    *,
    augment: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    dataset = Ksdd2SegmentationDataset(
        dataset="kolektor_sdd2",
        root=args.root,
        split_file=args.split_file,
        split=split,
        augment=augment,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if augment and shuffle:
        weights = [
            8.0 if record["has_defect"] else 1.0 for record in dataset.records
        ]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


KSDD2_FEATURES = (
    "entropy",
    "forward_consistency",
    "detector_disagreement",
    "evidence_shift",
    "restoration_residual",
    "high_probability_area",
    "peak_probability",
    "peak_disagreement",
    "secondary_peak",
    "active_components",
)


def _structure_features(
    probability: torch.Tensor, threshold: float, margin: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Secondary-peak and component-count statistics of the detector call map.

    Defect evidence is typically one compact activation cluster, whereas
    clean-region activation is dispersed: ``secondary_peak`` is the maximum
    probability outside a margin around the largest call component, and
    ``active_components`` counts call components.  Both are computable at
    application time (no mask needed).
    """

    from scipy import ndimage

    prob_cpu = probability.detach().cpu().numpy()
    calls = prob_cpu >= threshold
    secondary = np.zeros(prob_cpu.shape[0], dtype=np.float32)
    components = np.zeros(prob_cpu.shape[0], dtype=np.float32)
    structure = np.ones((3, 3), dtype=bool)
    for i in range(prob_cpu.shape[0]):
        call = calls[i, 0]
        labels, count = ndimage.label(call, structure=structure)
        components[i] = float(count)
        if count == 0:
            secondary[i] = 0.0
            continue
        sizes = ndimage.sum(call, labels, index=np.arange(1, count + 1))
        largest = int(np.argmax(sizes)) + 1
        keep_out = ndimage.binary_dilation(
            labels == largest, structure=structure, iterations=margin
        )
        outside = prob_cpu[i, 0][~keep_out]
        secondary[i] = float(outside.max()) if outside.size else 0.0
    return (
        torch.from_numpy(secondary),
        torch.from_numpy(components),
    )


def _top_fraction_mean(values: torch.Tensor, fraction: float = 0.001) -> torch.Tensor:
    """Mean of the top `fraction` of pixels per image (localized statistic)."""

    flat = values.flatten(1)
    k = max(1, int(round(flat.shape[1] * fraction)))
    top = torch.topk(flat, k, dim=1).values
    return top.mean(dim=1)


@torch.no_grad()
def evaluate_split_ksdd2(args, *, split, detector, restorer, threshold, device):
    """pipeline.evaluate_split plus two localized peak features.

    Whole-image mean features are dominated by clean area when the defect
    occupies a very small pixel fraction, as on KolektorSDD2; the peak
    statistics summarize the strongest local detector response and the
    strongest local detector change, and remain reference-free.
    """

    import torch.nn.functional as F

    loader = make_ksdd2_loader(args, split)
    detector.eval()
    restorer.eval()
    tolerance = int(getattr(args, "boundary_tolerance", 8))
    rows = []
    for batch in loader:
        high = batch["image"].to(device)
        mask_float = batch["mask"].to(device)
        mask = mask_float.bool()
        low, bicubic, severities = pipeline.degrade_mixed_batch(
            high, batch["sample_id"], split=split, corruption_key=args.corruption_key
        )
        # Clean region excludes a tolerance ring around the annotation so that
        # boundary overrun of a correct detection is not counted as clean-area
        # activation; recall still uses the unmodified mask.
        if tolerance > 0:
            excluded = F.max_pool2d(
                mask_float, kernel_size=2 * tolerance + 1, stride=1, padding=tolerance
            ).bool()
        else:
            excluded = mask
        candidates = pipeline.restore_methods(low, tuple(high.shape[-2:]))
        candidates["learned"] = restorer(bicubic)
        raw_probability = torch.sigmoid(detector(candidates["raw_nearest"]))
        for method, candidate in candidates.items():
            probability = torch.sigmoid(detector(candidate))
            call = probability >= threshold
            positive_pixels = mask.flatten(1).sum(1)
            true_positive = (call & mask).flatten(1).sum(1)
            negative_pixels = (~excluded).flatten(1).sum(1).clamp_min(1)
            false_positive = (call & ~excluded).flatten(1).sum(1)
            recall = torch.where(
                positive_pixels > 0,
                true_positive.float() / positive_pixels.clamp_min(1),
                torch.ones_like(true_positive, dtype=torch.float32),
            )
            clean_fpr = false_positive.float() / negative_pixels
            deletion_error = (positive_pixels > 0) & (recall < args.deletion_recall_floor)
            invention_error = clean_fpr > args.invention_fpr_limit
            residual = (candidate - candidates["raw_nearest"]).abs().flatten(1).mean(1)
            disagreement_map = (probability - raw_probability).abs()
            disagreement = disagreement_map.flatten(1).mean(1)
            entropy = pipeline._binary_entropy(probability).flatten(1).mean(1)
            evidence_shift = (
                probability.flatten(1).mean(1) - raw_probability.flatten(1).mean(1)
            ).abs()
            high_area = (probability >= 0.5).flatten(1).float().mean(1)
            peak_probability = _top_fraction_mean(probability)
            peak_disagreement = _top_fraction_mean(disagreement_map)
            secondary_peak, active_components = _structure_features(
                probability, threshold, margin=tolerance
            )
            forward = pipeline.forward_consistency(candidate, low)
            mse = ((candidate - high) ** 2).flatten(1).mean(1)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            for item in range(high.shape[0]):
                rows.append(
                    {
                        "sample_id": batch["sample_id"][item],
                        "split": split,
                        "severity": severities[item],
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
                        "peak_probability": float(peak_probability[item].cpu()),
                        "peak_disagreement": float(peak_disagreement[item].cpu()),
                        "secondary_peak": float(secondary_peak[item]),
                        "active_components": float(active_components[item]),
                        "psnr": float(psnr[item].cpu()),
                    }
                )
    return rows


@torch.no_grad()
def validation_positive_dice(model, loader, device) -> float:
    """Dice restricted to defect-positive images.

    KolektorSDD2 validation is ~86% clean, so the all-image dice used for
    Carinthia checkpoint selection is dominated by clean images and selects
    near-random checkpoints; positive-image dice tracks defect detection.
    """

    model.eval()
    scores: list[float] = []
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        keep = mask.flatten(1).sum(1) > 0
        if not bool(keep.any()):
            continue
        prediction = torch.sigmoid(model(image[keep])) >= 0.5
        local = mask[keep].bool()
        intersection = (prediction * local).flatten(1).sum(1).float()
        denominator = prediction.flatten(1).sum(1) + local.flatten(1).sum(1)
        scores.extend(((2 * intersection + 1) / (denominator + 1)).cpu().tolist())
    return float(np.mean(scores)) if scores else 0.0


def train_ksdd2_detector(args, device):
    """pipeline.train_detector with positive-image dice checkpoint selection."""

    train = make_ksdd2_loader(args, "train", augment=True, shuffle=True)
    valid = make_ksdd2_loader(args, "detector_val")
    model = pipeline.build_detector(args.detector_family, base=args.detector_base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_state = None
    best_dice = -1.0
    for epoch in range(args.detector_epochs):
        model.train()
        losses: list[float] = []
        for batch in train:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = pipeline.segmentation_loss(model(image), mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dice = validation_positive_dice(model, valid, device)
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_positive_dice": dice}
        )
        if dice > best_dice:
            best_dice = dice
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    args = parse_args()
    pipeline.make_loader = make_ksdd2_loader  # only substitution: data source
    pipeline.seed_everything(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    detector, detector_history = train_ksdd2_detector(args, device)
    restorer, restorer_history = pipeline.train_restorer(args, device)
    threshold = pipeline.detector_threshold(
        detector,
        make_ksdd2_loader(args, "detector_val"),
        device,
        args.target_pixel_fpr,
    )
    score_training = evaluate_split_ksdd2(
        args, split="detector_val", detector=detector, restorer=restorer,
        threshold=threshold, device=device,
    )
    calibration_rows = evaluate_split_ksdd2(
        args, split="calibration", detector=detector, restorer=restorer,
        threshold=threshold, device=device,
    )
    test_rows = evaluate_split_ksdd2(
        args, split="test", detector=detector, restorer=restorer,
        threshold=threshold, device=device,
    )

    torch.save(detector.state_dict(), args.output / "detector.pt")
    torch.save(restorer.state_dict(), args.output / "restorer.pt")
    pipeline.write_csv(args.output / "detector_val_records.csv", score_training)
    pipeline.write_csv(args.output / "calibration_records.csv", calibration_rows)
    pipeline.write_csv(args.output / "test_records.csv", test_rows)
    result = {
        "dataset": "kolektor_sdd2",
        "seed": args.seed,
        "device": str(device),
        "detector_threshold": threshold,
        "target_pixel_fpr": args.target_pixel_fpr,
        "deletion_recall_floor": args.deletion_recall_floor,
        "invention_fpr_limit": args.invention_fpr_limit,
        "detector_base": args.detector_base,
        "restorer_channels": args.restorer_channels,
        "restorer_blocks": args.restorer_blocks,
        "detector_family": args.detector_family,
        "restorer_family": args.restorer_family,
        "corruption_key": args.corruption_key,
        "boundary_tolerance": int(getattr(args, "boundary_tolerance", 8)),
        "detector_history": detector_history,
        "restorer_history": restorer_history,
        "test_methods": pipeline.aggregate_methods(test_rows),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("dataset", "seed", "detector_threshold", "elapsed_seconds")}, indent=2))
    print(json.dumps(result["test_methods"], indent=2))


if __name__ == "__main__":
    main()
