"""WP6 MVTec AD study: the paper's KolektorSDD2 pipeline with an MVTec data source.

Only the dataset class is substituted (mvtec_data.MvtecSegmentationDataset);
training, threshold rule, degradation instrument, candidate actions, features
and endpoint definitions are the functions of
tim_upgrade/scripts/run_ksdd2_pipeline.py and scripts/run_dataset_pipeline.py,
called unchanged with the KolektorSDD2 settings (tiny U-Net, 32 base channels,
15 detector epochs, positive-image dice checkpointing, 5 restorer epochs,
pixel FPR 1e-3, r0 = 0.25, f0 = 0.002, 8-px tolerance ring).

  --mode dev      train detector and restorer, fix the pixel threshold, write
                  detector_val_records.csv and the models.  Reads train and
                  detector_val only.
  --mode confirm  load the dev-mode models of the seed and write
                  calibration_records.csv and test_records.csv.  Refused unless
                  FREEZE_MANIFEST.json verifies; each run is logged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "tim_upgrade" / "scripts", HERE):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

import mvtec_data  # noqa: E402
import run_ksdd2_pipeline as k  # noqa: E402
from scripts import run_dataset_pipeline as pipeline  # noqa: E402

SPLIT_FILE = HERE / "splits" / "mvtec_rr_v1.json"
# Dataset root: MVTEC_AD_ROOT, else <project parent>/data/raw/mvtec_ad/extracted.
DATA_ROOT = Path(os.environ.get("MVTEC_AD_ROOT", Path(__file__).resolve().parents[3].parent / "data" / "raw" / "mvtec_ad" / "extracted"))
MANIFEST = HERE / "FREEZE_MANIFEST.json"
ACCESS_LOG = HERE / "sealed_role_access_log.txt"
DEV_ROLES = {"train", "detector_val"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_freeze() -> dict:
    if not MANIFEST.exists():
        raise PermissionError("no FREEZE_MANIFEST.json: protocol not frozen, sealed roles stay closed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise PermissionError("manifest status is not FROZEN")
    for rel, digest in manifest["files"].items():
        actual = sha256_file(HERE / rel)
        if actual != digest:
            raise PermissionError(f"{rel} changed after freeze ({actual} != {digest})")
    return manifest


def install_data_source(allowed: set[str]) -> None:
    """Point the KolektorSDD2 loader at MVTec AD and refuse roles outside `allowed`."""

    def factory(*, dataset, root, split_file, split, augment=False):
        if split not in allowed:
            raise PermissionError(f"role {split} is sealed in this run")
        return mvtec_data.MvtecSegmentationDataset(dataset=dataset, root=root, split_file=split_file,
                                                   split=split, augment=augment)

    k.Ksdd2SegmentationDataset = factory
    pipeline.make_loader = k.make_ksdd2_loader


def install_unet4() -> None:
    """Make build_detector('unet4') return the four-level grayscale U-Net (base 32)."""

    sys.path.insert(0, str(HERE.parent))
    from unet4 import UNet4

    original = pipeline.build_detector

    def build(family, **kwargs):
        if family == "unet4":
            return UNet4(in_ch=1, base=int(kwargs.get("base", 32)), out_ch=1)
        return original(family, **kwargs)

    pipeline.build_detector = build


def portable_settings(args: argparse.Namespace) -> dict[str, str]:
    """Run settings with machine-local paths replaced (the file is hashed at freeze)."""

    out = {key: str(value) for key, value in vars(args).items()}
    out["root"] = "$MVTEC_AD_ROOT"
    for key in ("split_file", "output"):
        out[key] = Path(getattr(args, key)).resolve().relative_to(REPO).as_posix()
    return out


def make_args(seed: int, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        root=DATA_ROOT, split_file=SPLIT_FILE, output=output, seed=seed, dataset="kolektor_sdd2",
        detector_epochs=15, restorer_epochs=5, batch_size=8, device="cuda", corruption_key="v4corr:eval",
        detector_family="tiny_unet", restorer_family="compact", detector_base=32, restorer_channels=32,
        restorer_blocks=6, target_pixel_fpr=1e-3, deletion_recall_floor=0.25, invention_fpr_limit=2e-3,
        risk_deletion=0.15, risk_invention=0.15, delta=0.10, boundary_tolerance=8,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=("dev", "confirm"), required=True)
    ap.add_argument("--detector", choices=("tiny_unet", "unet4"), default="tiny_unet",
                    help="unet4: four-level U-Net (development iteration 2, PROTOCOL.md Section 11)")
    ap.add_argument("--detector-epochs", type=int, default=15)
    ap.add_argument("--tag", default="", help="run folder suffix of a development variant")
    cli = ap.parse_args()
    output = HERE / "runs" / f"seed{cli.seed}{cli.tag}"
    output.mkdir(parents=True, exist_ok=True)
    args = make_args(cli.seed, output)
    args.detector_family, args.detector_epochs = cli.detector, cli.detector_epochs
    if cli.detector == "unet4":
        install_unet4()
    pipeline.seed_everything(args.seed)
    device = torch.device("cuda")
    started = time.time()

    if cli.mode == "dev":
        install_data_source(DEV_ROLES)
        detector, det_hist = k.train_ksdd2_detector(args, device)
        restorer, res_hist = pipeline.train_restorer(args, device)
        threshold = pipeline.detector_threshold(detector, k.make_ksdd2_loader(args, "detector_val"), device,
                                                args.target_pixel_fpr)
        rows = k.evaluate_split_ksdd2(args, split="detector_val", detector=detector, restorer=restorer,
                                      threshold=threshold, device=device)
        torch.save(detector.state_dict(), output / "detector.pt")
        torch.save(restorer.state_dict(), output / "restorer.pt")
        pipeline.write_csv(output / "detector_val_records.csv", rows)
        meta = {"dataset": "mvtec_ad", "seed": args.seed, "detector_threshold": threshold,
                "split_sha256": sha256_file(SPLIT_FILE), "detector_history": det_hist,
                "restorer_history": res_hist, "settings": portable_settings(args),
                "elapsed_seconds": time.time() - started}
        (output / "dev_results.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps({x: meta[x] for x in ("seed", "detector_threshold", "elapsed_seconds")}), flush=True)
        return

    manifest = verify_freeze()
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with ACCESS_LOG.open("a", encoding="utf-8") as log:
        log.write(f"{stamp}\troles=calibration,test\tseed={cli.seed}\t"
                  f"manifest_sha256={sha256_file(MANIFEST)}\tfrozen_at={manifest['frozen_at']}\n")
    install_data_source(DEV_ROLES | {"calibration", "test"})
    meta = json.loads((output / "dev_results.json").read_text(encoding="utf-8"))
    detector = pipeline.build_detector(args.detector_family, base=args.detector_base).to(device)
    detector.load_state_dict(torch.load(output / "detector.pt", map_location=device))
    restorer = pipeline.build_restorer(args.restorer_family, channels=args.restorer_channels,
                                       blocks=args.restorer_blocks).to(device)
    restorer.load_state_dict(torch.load(output / "restorer.pt", map_location=device))
    threshold = meta["detector_threshold"]
    for split, name in (("calibration", "calibration_records.csv"), ("test", "test_records.csv")):
        rows = k.evaluate_split_ksdd2(args, split=split, detector=detector, restorer=restorer,
                                      threshold=threshold, device=device)
        pipeline.write_csv(output / name, rows)
    print(f"seed {cli.seed}: calibration and test records written", flush=True)


if __name__ == "__main__":
    main()
