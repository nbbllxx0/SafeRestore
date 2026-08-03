"""Clean-proxy sensitivity for a certified SafeRestore gate.

Synthesize clean twins of defect-positive Carinthia images by inpainting the
defect support, then evaluate invention and acceptance under a fixed certified
gate. This is a proxy sensitivity, not morphology-matched validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from saferestore.certification import DualRiskTargets
from saferestore.degradation import forward_consistency, restore_methods
from saferestore.degradation_schedule import degrade_mixed_batch
from saferestore.torch_data import PublicSegmentationDataset
from saferestore.torch_models import build_detector, build_restorer
from scripts.analyze_router_pool import read_rows
from scripts.run_reported_analysis import (
    build_policy_arrays,
    fit_risk_models,
    stable_tuning_mask,
    subset,
)
from saferestore.certification import certify_threshold, choose_threshold

PROJECT = Path(__file__).resolve().parents[1]


def inpaint_defect(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill defect pixels with local clean neighborhood means (OpenCV-free)."""

    clean = image.copy()
    defect = mask.astype(bool)
    if not defect.any():
        return clean
    # Dilate defect slightly so boundary contamination is removed.
    kernel = np.ones((5, 5), dtype=bool)
    padded = np.pad(defect, 2, constant_values=False)
    dilated = np.zeros_like(defect)
    for dy in range(5):
        for dx in range(5):
            dilated |= padded[dy : dy + defect.shape[0], dx : dx + defect.shape[1]]
    # Iterative fill from exterior ring.
    unknown = dilated.copy()
    filled = clean.copy()
    for _ in range(64):
        if not unknown.any():
            break
        ys, xs = np.where(unknown)
        updates = []
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 2), min(clean.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(clean.shape[1], x + 3)
            window = filled[y0:y1, x0:x1]
            known = ~unknown[y0:y1, x0:x1]
            if not known.any():
                continue
            updates.append((y, x, float(window[known].mean())))
        if not updates:
            # Fallback: global clean mean.
            known = ~unknown
            fill_value = float(filled[known].mean()) if known.any() else 0.5
            filled[unknown] = fill_value
            break
        for y, x, value in updates:
            filled[y, x] = value
            unknown[y, x] = False
    return np.clip(filled, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT / "experiments" / "v4_carinthia_primary_trainseed211",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT / "data" / "raw" / "carinthia_s" / "data" / "data",
    )
    parser.add_argument("--split-file", type=Path, default=Path("splits/carinthia_s_v4.json"))
    parser.add_argument("--corruption-key", default="v4corr:eval")
    parser.add_argument("--tune-key", default="v4tunecert")
    parser.add_argument("--max-proxies", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "experiments" / "v4_clean_proxy_sensitivity_seed211",
    )
    args = parser.parse_args()

    meta = json.loads((args.checkpoint / "results.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = build_detector(
        meta.get("detector_family", "tiny_unet"),
        base=int(meta.get("detector_base", 16)),
    ).to(device)
    restorer = build_restorer(
        meta.get("restorer_family", "compact"),
        channels=int(meta.get("restorer_channels", 32)),
        blocks=int(meta.get("restorer_blocks", 6)),
    ).to(device)
    detector.load_state_dict(
        torch.load(args.checkpoint / "detector.pt", map_location=device, weights_only=True)
    )
    restorer.load_state_dict(
        torch.load(args.checkpoint / "restorer.pt", map_location=device, weights_only=True)
    )
    detector.eval()
    restorer.eval()
    threshold = float(meta["detector_threshold"])

    dataset = PublicSegmentationDataset(
        dataset="carinthia_s",
        root=args.data_root,
        split_file=args.split_file,
        split="test",
        augment=False,
    )
    positives = [item for item in dataset if float(item["mask"].sum()) > 0]
    positives = positives[: args.max_proxies]

    # Fit gate from checkpoint records under primary targets.
    targets = DualRiskTargets()
    models = fit_risk_models(
        read_rows(args.checkpoint / "detector_val_records.csv"),
        int(meta["seed"]),
        0.25,
    )
    calibration = build_policy_arrays(
        read_rows(args.checkpoint / "calibration_records.csv"),
        models,
        targets,
        0.25,
    )["action_pool"]
    tuning_mask = stable_tuning_mask(
        calibration["sample_id"].tolist(), tune_key=args.tune_key
    )
    tuning = subset(calibration, tuning_mask)
    certification = subset(calibration, ~tuning_mask)
    gate = choose_threshold(
        tuning["score"],
        tuning["positive"],
        tuning["deletion"],
        tuning["invention"],
        targets,
    )
    certificate = certify_threshold(
        certification["score"],
        gate,
        certification["positive"],
        certification["deletion"],
        certification["invention"],
        targets,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in positives:
            image = item["image"][0].numpy()
            mask = item["mask"][0].numpy() > 0.5
            proxy = inpaint_defect(image, mask)
            high = torch.from_numpy(proxy).unsqueeze(0).unsqueeze(0).to(device)
            zero_mask = torch.zeros_like(high)
            low, bicubic, severities = degrade_mixed_batch(
                high,
                [item["sample_id"]],
                split="proxy_clean",
                corruption_key=args.corruption_key,
            )
            candidates = restore_methods(low, tuple(high.shape[-2:]))
            candidates["learned"] = restorer(bicubic)
            # Score each action with the frozen action-risk models via records schema.
            local_rows = []
            raw_probability = torch.sigmoid(detector(candidates["raw_nearest"]))
            for method, candidate in candidates.items():
                probability = torch.sigmoid(detector(candidate))
                call = probability >= threshold
                false_positive = (call & ~zero_mask.bool()).flatten(1).sum(1)
                negative_pixels = zero_mask.numel()
                clean_fpr = float(false_positive.item() / max(negative_pixels, 1))
                invention_error = int(clean_fpr > 0.002)
                residual = float((candidate - candidates["raw_nearest"]).abs().mean().item())
                disagreement = float((probability - raw_probability).abs().mean().item())
                p = probability.clamp(1e-6, 1 - 1e-6)
                entropy = float((-(p * p.log() + (1 - p) * (1 - p).log())).mean().item())
                evidence_shift = float(
                    (probability.mean() - raw_probability.mean()).abs().item()
                )
                high_area = float((probability >= 0.5).float().mean().item())
                forward = float(forward_consistency(candidate, low).mean().item())
                local_rows.append(
                    {
                        "sample_id": f"proxy::{item['sample_id']}",
                        "method": method,
                        "has_defect": 0,
                        "pixel_recall": 1.0,
                        "clean_fpr": clean_fpr,
                        "deletion_error": 0,
                        "invention_error": invention_error,
                        "entropy": entropy,
                        "forward_consistency": forward,
                        "detector_disagreement": disagreement,
                        "evidence_shift": evidence_shift,
                        "restoration_residual": residual,
                        "high_probability_area": high_area,
                        "psnr": 0.0,
                        "severity": severities[0],
                    }
                )
            policy = build_policy_arrays(local_rows, models, targets, 0.25)["action_pool"]
            accepted = bool(policy["score"][0] <= gate)
            rows.append(
                {
                    "source_sample_id": item["sample_id"],
                    "severity": severities[0],
                    "selected_method": str(policy["method"][0]),
                    "score": float(policy["score"][0]),
                    "gate": float(gate),
                    "accepted": int(accepted),
                    "invention_incident": int(policy["invention"][0]),
                    "clean_fpr_selected": float(
                        next(
                            row["clean_fpr"]
                            for row in local_rows
                            if row["method"] == policy["method"][0]
                        )
                    ),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "proxy_clean_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "note": (
            "Morphology clean-proxy sensitivity under a frozen certified gate. "
            "Proxy cleans are inpainted defect twins, not morphology-matched acquisition."
        ),
        "checkpoint": str(args.checkpoint),
        "n_proxies": len(rows),
        "certificate_passes": bool(certificate["passes"]),
        "acceptance_rate": float(np.mean([row["accepted"] for row in rows])),
        "invention_rate_among_accepted": (
            float(
                np.mean(
                    [row["invention_incident"] for row in rows if row["accepted"] == 1]
                )
            )
            if any(row["accepted"] == 1 for row in rows)
            else None
        ),
        "invention_rate_all_proxies": float(
            np.mean([row["invention_incident"] for row in rows])
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
