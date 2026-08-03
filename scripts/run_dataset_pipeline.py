from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from saferestore.calibration import calibrate_sequential_router
from saferestore.degradation import degrade_batch, forward_consistency, restore_methods
from saferestore.degradation_schedule import degrade_mixed_batch
from saferestore.metrics import evaluate_routing
from saferestore.sampling import defect_balanced_sampler
from saferestore.router import route_sequential
from saferestore.torch_data import PublicSegmentationDataset
from saferestore.torch_models import (
    CompactRestorer,
    TinyUNet,
    build_detector,
    build_restorer,
    soft_dice_loss,
)
from saferestore.types import Action, RiskTargets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SafeRestore public-data experiment.")
    parser.add_argument("--dataset", choices=("carinthia_s", "kolektor_sdd"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--detector-epochs", type=int, default=5)
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
    parser.add_argument("--deletion-recall-floor", type=float, default=0.50)
    parser.add_argument("--invention-fpr-limit", type=float, default=2e-3)
    parser.add_argument("--risk-deletion", type=float, default=0.25)
    parser.add_argument("--risk-invention", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.10)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    args: argparse.Namespace,
    split: str,
    *,
    augment: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    dataset = PublicSegmentationDataset(
        dataset=args.dataset,
        root=args.root,
        split_file=args.split_file,
        split=split,
        augment=augment,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = defect_balanced_sampler(dataset, seed=args.seed) if augment and shuffle else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def segmentation_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positive = float(mask.sum().detach().cpu())
    negative = float(mask.numel() - positive)
    weight = min(80.0, max(1.0, negative / max(positive, 1.0)))
    bce = F.binary_cross_entropy_with_logits(
        logits,
        mask,
        pos_weight=torch.tensor(weight, device=logits.device),
    )
    return bce + soft_dice_loss(logits, mask)


@torch.no_grad()
def validation_dice(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores: list[float] = []
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        prediction = torch.sigmoid(model(image)) >= 0.5
        intersection = (prediction * mask.bool()).flatten(1).sum(1).float()
        denominator = prediction.flatten(1).sum(1) + mask.flatten(1).sum(1)
        scores.extend(((2 * intersection + 1) / (denominator + 1)).cpu().tolist())
    return float(np.mean(scores))


def train_detector(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    train = make_loader(args, "train", augment=True, shuffle=True)
    valid = make_loader(args, "detector_val")
    model = build_detector(
        args.detector_family, base=args.detector_base
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_dice = -1.0
    for epoch in range(args.detector_epochs):
        model.train()
        losses: list[float] = []
        for batch in train:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(image), mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dice = validation_dice(model, valid, device)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_dice": dice})
        if dice > best_dice:
            best_dice = dice
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def train_restorer(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    train = make_loader(args, "train", augment=True, shuffle=True)
    model = build_restorer(
        args.restorer_family,
        channels=args.restorer_channels,
        blocks=args.restorer_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history: list[dict[str, float]] = []
    for epoch in range(args.restorer_epochs):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(train):
            high = batch["image"].to(device)
            _, bicubic, _ = degrade_mixed_batch(
                high,
                batch["sample_id"],
                split="train",
                corruption_key=args.corruption_key,
                training_draw_seed=args.seed + epoch * 10_007,
            )
            restored = model(bicubic)
            l1 = F.l1_loss(restored, high)
            gx_r = restored[..., :, 1:] - restored[..., :, :-1]
            gx_h = high[..., :, 1:] - high[..., :, :-1]
            gy_r = restored[..., 1:, :] - restored[..., :-1, :]
            gy_h = high[..., 1:, :] - high[..., :-1, :]
            edge = F.l1_loss(gx_r, gx_h) + F.l1_loss(gy_r, gy_h)
            loss = l1 + 0.25 * edge
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return model, history


@torch.no_grad()
def detector_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_fpr: float,
) -> float:
    model.eval()
    negatives: list[np.ndarray] = []
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device).bool()
        probability = torch.sigmoid(model(image))
        negatives.append(probability[~mask].detach().cpu().numpy())
    values = np.concatenate(negatives)
    return float(np.quantile(values, 1.0 - target_fpr, method="higher"))


def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    p = probability.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


@torch.no_grad()
def evaluate_split(
    args: argparse.Namespace,
    *,
    split: str,
    detector: torch.nn.Module,
    restorer: torch.nn.Module,
    threshold: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    loader = make_loader(args, split)
    detector.eval()
    restorer.eval()
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        high = batch["image"].to(device)
        mask = batch["mask"].to(device).bool()
        low, bicubic, severities = degrade_mixed_batch(
            high,
            batch["sample_id"],
            split=split,
            corruption_key=args.corruption_key,
        )
        candidates = restore_methods(low, tuple(high.shape[-2:]))
        candidates["learned"] = restorer(bicubic)
        raw_probability = torch.sigmoid(detector(candidates["raw_nearest"]))

        for method, candidate in candidates.items():
            probability = torch.sigmoid(detector(candidate))
            call = probability >= threshold
            positive_pixels = mask.flatten(1).sum(1)
            true_positive = (call & mask).flatten(1).sum(1)
            negative_pixels = (~mask).flatten(1).sum(1).clamp_min(1)
            false_positive = (call & ~mask).flatten(1).sum(1)
            recall = torch.where(
                positive_pixels > 0,
                true_positive.float() / positive_pixels.clamp_min(1),
                torch.ones_like(true_positive, dtype=torch.float32),
            )
            clean_fpr = false_positive.float() / negative_pixels
            deletion_error = (positive_pixels > 0) & (recall < args.deletion_recall_floor)
            invention_error = clean_fpr > args.invention_fpr_limit
            residual = (candidate - candidates["raw_nearest"]).abs().flatten(1).mean(1)
            disagreement = (probability - raw_probability).abs().flatten(1).mean(1)
            entropy = _binary_entropy(probability).flatten(1).mean(1)
            evidence_shift = (probability.flatten(1).mean(1) - raw_probability.flatten(1).mean(1)).abs()
            high_area = (probability >= 0.5).flatten(1).float().mean(1)
            forward = forward_consistency(candidate, low)
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
                        "psnr": float(psnr[item].cpu()),
                    }
                )
    return rows


FEATURES = (
    "entropy",
    "forward_consistency",
    "detector_disagreement",
    "evidence_shift",
    "restoration_residual",
    "high_probability_area",
)


def fit_risk_score(rows: list[dict[str, Any]], method: str, seed: int) -> dict[str, Any]:
    subset = [row for row in rows if row["method"] == method]
    x = np.asarray([[row[name] for name in FEATURES] for row in subset], dtype=np.float64)
    result: dict[str, Any] = {"features": list(FEATURES), "method": method}
    for target in ("deletion_error", "invention_error"):
        y = np.asarray([row[target] for row in subset], dtype=np.int64)
        if np.unique(y).size < 2:
            result[target] = {"constant": float(y[0] if y.size else 1.0)}
        else:
            scaler = StandardScaler().fit(x)
            normalized = scaler.transform(x)
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
                solver="liblinear",
            ).fit(normalized, y)
            result[target] = {
                "coef": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
    return result


def predict_risk(model: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features = tuple(model.get("features", FEATURES))
    x = np.asarray([[row[name] for name in features] for row in rows], dtype=np.float64)
    predictions: list[np.ndarray] = []
    for target in ("deletion_error", "invention_error"):
        spec = model[target]
        if "constant" in spec:
            predictions.append(np.full(x.shape[0], spec["constant"], dtype=np.float64))
        else:
            normalized = (x - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
            logit = normalized @ np.asarray(spec["coef"]) + spec["intercept"]
            predictions.append(1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40))))
    return predictions[0], predictions[1]


def _rows_for_method(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["method"] == method]


def route_experiment(
    args: argparse.Namespace,
    *,
    score_training: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    restored_method: str = "learned",
) -> dict[str, Any]:
    raw_model = fit_risk_score(score_training, "raw_nearest", args.seed)
    restored_model = fit_risk_score(score_training, restored_method, args.seed)
    models = {Action.RAW: raw_model, Action.RESTORED: restored_model}

    def arrays(rows: list[dict[str, Any]]) -> tuple[dict[Action, np.ndarray], dict[Action, np.ndarray], dict[Action, np.ndarray]]:
        scores: dict[Action, np.ndarray] = {}
        deletion: dict[Action, np.ndarray] = {}
        invention: dict[Action, np.ndarray] = {}
        for action, method in ((Action.RAW, "raw_nearest"), (Action.RESTORED, restored_method)):
            subset = _rows_for_method(rows, method)
            p_del, p_inv = predict_risk(models[action], subset)
            scores[action] = np.maximum(
                p_del / max(args.risk_deletion, 1e-12),
                p_inv / max(args.risk_invention, 1e-12),
            )
            deletion[action] = np.asarray([row["deletion_error"] for row in subset], dtype=np.int8)
            invention[action] = np.asarray([row["invention_error"] for row in subset], dtype=np.int8)
        return scores, deletion, invention

    c_scores, c_del, c_inv = arrays(calibration_rows)
    targets = RiskTargets(
        deletion=args.risk_deletion,
        invention=args.risk_invention,
        delta=args.delta,
        min_accept=max(5, min(20, len(c_scores[Action.RAW]) // 8)),
    )
    calibration = calibrate_sequential_router(
        c_scores,
        c_del,
        c_inv,
        targets=targets,
        seed=args.seed,
    )
    t_scores, t_del, t_inv = arrays(test_rows)
    decisions = route_sequential(t_scores, calibration)
    metrics = evaluate_routing(decisions, t_del, t_inv)
    return {
        "restored_method": restored_method,
        "risk_models": {"raw": raw_model, "restored": restored_model},
        "calibration": {
            "seed": calibration.seed,
            "size": calibration.calibration_size,
            "targets": asdict(calibration.targets),
            "actions": {
                action.value: asdict(calibration.actions[action]) for action in calibration.order
            },
        },
        "test": metrics.to_dict(),
        "decisions": decisions.tolist(),
    }


def aggregate_methods(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        output[method] = {
            "n": len(subset),
            "mean_pixel_recall": float(np.mean([row["pixel_recall"] for row in subset])),
            "mean_clean_fpr": float(np.mean([row["clean_fpr"] for row in subset])),
            "deletion_error_rate": float(np.mean([row["deletion_error"] for row in subset])),
            "invention_error_rate": float(np.mean([row["invention_error"] for row in subset])),
            "mean_psnr": float(np.mean([row["psnr"] for row in subset])),
        }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    detector, detector_history = train_detector(args, device)
    restorer, restorer_history = train_restorer(args, device)
    threshold = detector_threshold(
        detector,
        make_loader(args, "detector_val"),
        device,
        args.target_pixel_fpr,
    )
    score_training = evaluate_split(
        args,
        split="detector_val",
        detector=detector,
        restorer=restorer,
        threshold=threshold,
        device=device,
    )
    calibration_rows = evaluate_split(
        args,
        split="calibration",
        detector=detector,
        restorer=restorer,
        threshold=threshold,
        device=device,
    )
    test_rows = evaluate_split(
        args,
        split="test",
        detector=detector,
        restorer=restorer,
        threshold=threshold,
        device=device,
    )
    shift_rows: list[dict[str, Any]] = []
    split_payload = json.loads(args.split_file.read_text(encoding="utf-8"))
    if "shift_test" in split_payload["counts"]:
        shift_rows = evaluate_split(
            args,
            split="shift_test",
            detector=detector,
            restorer=restorer,
            threshold=threshold,
            device=device,
        )

    torch.save(detector.state_dict(), args.output / "detector.pt")
    torch.save(restorer.state_dict(), args.output / "restorer.pt")
    write_csv(args.output / "detector_val_records.csv", score_training)
    write_csv(args.output / "calibration_records.csv", calibration_rows)
    write_csv(args.output / "test_records.csv", test_rows)
    if shift_rows:
        write_csv(args.output / "shift_test_records.csv", shift_rows)
    result = {
        "dataset": args.dataset,
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
        "detector_history": detector_history,
        "restorer_history": restorer_history,
        "test_methods": aggregate_methods(test_rows),
        "shift_methods": aggregate_methods(shift_rows) if shift_rows else {},
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()





