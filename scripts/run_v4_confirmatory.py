"""Run SafeRestore Carinthia, alternate-family, and Kolektor experiments."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
TRAIN_SEEDS = (201, 203, 207, 211, 223)
CORRUPTION_KEY = "v4corr:eval"
CARINTHIA_SPLIT = PROJECT / "splits" / "carinthia_s_v4.json"
KOLEKTOR_POWERED = PROJECT / "splits" / "kolektor_sdd_powered_v4.json"

FAMILIES = (
    {"tag": "primary", "detector_family": "tiny_unet", "restorer_family": "compact"},
    {"tag": "attn_unet", "detector_family": "attn_unet", "restorer_family": "unet"},
)


def run(cmd: list[str]) -> None:
    print("RUNNING", " ".join(cmd), flush=True)
    subprocess.run(
        cmd,
        check=True,
        cwd=str(PROJECT),
        env={**os.environ, "PYTHONPATH": "src;."},
    )


def pipeline_cmd(
    *,
    py: str,
    dataset: str,
    root: Path,
    split: Path,
    output: Path,
    seed: int,
    detector_family: str,
    restorer_family: str,
    device: str,
    detector_epochs: int,
    restorer_epochs: int,
) -> list[str]:
    return [
        py,
        str(PROJECT / "scripts" / "run_dataset_pipeline.py"),
        "--dataset",
        dataset,
        "--root",
        str(root),
        "--split-file",
        str(split),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--corruption-key",
        CORRUPTION_KEY,
        "--detector-family",
        detector_family,
        "--restorer-family",
        restorer_family,
        "--detector-epochs",
        str(detector_epochs),
        "--restorer-epochs",
        str(restorer_epochs),
        "--deletion-recall-floor",
        "0.25",
        "--invention-fpr-limit",
        "0.002",
        "--risk-deletion",
        "0.15",
        "--risk-invention",
        "0.15",
        "--delta",
        "0.10",
        "--device",
        device,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--carinthia-root",
        type=Path,
        default=PROJECT / "data" / "raw" / "carinthia_s" / "data" / "data",
    )
    parser.add_argument(
        "--kolektor-root",
        type=Path,
        default=PROJECT / "data" / "raw" / "kolektor_sdd" / "dataset",
    )
    parser.add_argument(
        "--kolektor-split-file",
        type=Path,
        default=PROJECT / "data" / "raw" / "kolektor_sdd" / "splits" / "split.pyb",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-epochs", type=int, default=5)
    parser.add_argument("--restorer-epochs", type=int, default=5)
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["primary", "family", "kolektor"],
        choices=("primary", "family", "kolektor"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    args = parser.parse_args()
    py = str(args.python)

    if not args.skip_split:
        # Prefer creating v4 manifests directly so powered Kolektor does not
        # depend on the official fold loader path being present.
        from scripts.prepare_splits import carinthia_split, kolektor_powered_split

        CARINTHIA_SPLIT.parent.mkdir(parents=True, exist_ok=True)
        car = carinthia_split(args.carinthia_root, seed=20260725, version=4)
        CARINTHIA_SPLIT.write_text(json.dumps(car, indent=2, sort_keys=True), encoding="utf-8")
        kol = kolektor_powered_split(
            args.kolektor_root, seed=20260725, min_calibration_positives=18
        )
        KOLEKTOR_POWERED.write_text(json.dumps(kol, indent=2, sort_keys=True), encoding="utf-8")
        print("wrote", CARINTHIA_SPLIT, car["counts"])
        print("wrote", KOLEKTOR_POWERED, kol["counts"], kol["positive_counts"])

    outputs: list[str] = []
    if "primary" in args.stages:
        family = FAMILIES[0]
        for seed in args.seeds:
            out = PROJECT / "experiments" / f"v4_carinthia_{family['tag']}_trainseed{seed}"
            if (out / "test_records.csv").exists():
                print("SKIP", out, flush=True)
            else:
                run(
                    pipeline_cmd(
                        py=py,
                        dataset="carinthia_s",
                        root=args.carinthia_root,
                        split=CARINTHIA_SPLIT,
                        output=out,
                        seed=seed,
                        detector_family=family["detector_family"],
                        restorer_family=family["restorer_family"],
                        device=args.device,
                        detector_epochs=args.detector_epochs,
                        restorer_epochs=args.restorer_epochs,
                    )
                )
            outputs.append(str(out))

    if "family" in args.stages:
        family = FAMILIES[1]
        for seed in args.seeds:
            out = PROJECT / "experiments" / f"v4_carinthia_{family['tag']}_trainseed{seed}"
            if (out / "test_records.csv").exists():
                print("SKIP", out, flush=True)
            else:
                run(
                    pipeline_cmd(
                        py=py,
                        dataset="carinthia_s",
                        root=args.carinthia_root,
                        split=CARINTHIA_SPLIT,
                        output=out,
                        seed=seed,
                        detector_family=family["detector_family"],
                        restorer_family=family["restorer_family"],
                        device=args.device,
                        detector_epochs=args.detector_epochs,
                        restorer_epochs=args.restorer_epochs,
                    )
                )
            outputs.append(str(out))

    if "kolektor" in args.stages:
        family = FAMILIES[0]
        for seed in args.seeds:
            out = (
                PROJECT
                / "experiments"
                / f"v4_kolektor_powered_{family['tag']}_trainseed{seed}"
            )
            if (out / "test_records.csv").exists():
                print("SKIP", out, flush=True)
            else:
                run(
                    pipeline_cmd(
                        py=py,
                        dataset="kolektor_sdd",
                        root=args.kolektor_root,
                        split=KOLEKTOR_POWERED,
                        output=out,
                        seed=seed,
                        detector_family=family["detector_family"],
                        restorer_family=family["restorer_family"],
                        device=args.device,
                        detector_epochs=args.detector_epochs,
                        restorer_epochs=args.restorer_epochs,
                    )
                )
            outputs.append(str(out))

    manifest = {
        "protocol": "confirmatory_protocol_v4",
        "corruption_key": CORRUPTION_KEY,
        "carinthia_split": str(CARINTHIA_SPLIT),
        "kolektor_powered_split": str(KOLEKTOR_POWERED),
        "seeds": list(args.seeds),
        "stages": list(args.stages),
        "outputs": outputs,
    }
    path = PROJECT / "experiments" / "v4_confirmatory_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", path)
    print("V4_CONFIRMATORY_DONE")


if __name__ == "__main__":
    main()
