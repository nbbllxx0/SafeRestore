"""Regenerate KSDD2 record CSVs from saved checkpoints with current features.

Loads detector.pt/restorer.pt and the stored detector threshold from an
existing run folder and re-evaluates the three data roles with the current
``evaluate_split_ksdd2`` feature set.  No training occurs; the policy inputs
(models, threshold) are unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import run_ksdd2_pipeline as rk
from scripts import run_dataset_pipeline as pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--boundary-tolerance", type=int, default=None,
        help="Override the stored annotation-boundary tolerance radius (px). "
             "Used to regenerate the tolerance sweep of the diagnostics "
             "figure; models and detector threshold are unchanged.",
    )
    args = parser.parse_args()

    meta = json.loads((args.source / "results.json").read_text(encoding="utf-8"))
    tolerance_override = args.boundary_tolerance
    for key in (
        "seed", "corruption_key", "detector_family", "restorer_family",
        "detector_base", "restorer_channels", "restorer_blocks",
        "deletion_recall_floor", "invention_fpr_limit", "boundary_tolerance",
    ):
        setattr(args, key, meta[key])
    if tolerance_override is not None:
        args.boundary_tolerance = tolerance_override
        meta["boundary_tolerance"] = tolerance_override
    args.batch_size = 8
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    detector = pipeline.build_detector(
        meta["detector_family"], base=meta["detector_base"]
    ).to(device)
    detector.load_state_dict(
        torch.load(args.source / "detector.pt", weights_only=True)
    )
    restorer = pipeline.build_restorer(
        meta["restorer_family"],
        channels=meta["restorer_channels"],
        blocks=meta["restorer_blocks"],
    ).to(device)
    restorer.load_state_dict(
        torch.load(args.source / "restorer.pt", weights_only=True)
    )
    threshold = float(meta["detector_threshold"])

    args.output.mkdir(parents=True, exist_ok=True)
    for split, filename in (
        ("detector_val", "detector_val_records.csv"),
        ("calibration", "calibration_records.csv"),
        ("test", "test_records.csv"),
    ):
        rows = rk.evaluate_split_ksdd2(
            args, split=split, detector=detector, restorer=restorer,
            threshold=threshold, device=device,
        )
        pipeline.write_csv(args.output / filename, rows)
    for name in ("results.json", "detector.pt", "restorer.pt"):
        shutil.copy2(args.source / name, args.output / name)
    meta["features"] = list(rk.KSDD2_FEATURES)
    meta["regenerated_from"] = str(args.source)
    (args.output / "results.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("regenerated", args.output)


if __name__ == "__main__":
    main()
