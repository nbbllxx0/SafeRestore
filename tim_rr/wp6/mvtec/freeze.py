"""Write FREEZE_MANIFEST.json for the MVTec AD study.

Run once, after author sign-off of PROTOCOL.md.  Refuses if a manifest exists,
if PROTOCOL.md does not say "Status: FROZEN", if any sealed role was already
opened, or if any calibration or test record file already exists.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_mvtec  # noqa: E402

FILES = [
    "PROTOCOL.md",
    "mvtec_data.py",
    "run_mvtec.py",
    "analyze_mvtec.py",
    "make_split.py",
    "power_preview.py",
    "test_mvtec.py",
    "freeze.py",
    "splits/mvtec_rr_v1.json",
    "../../../tim_upgrade/scripts/run_ksdd2_pipeline.py",
    "../../../tim_upgrade/scripts/run_tim_analysis.py",
    "../../../tim_upgrade/scripts/familywise_certification.py",
    "../../../scripts/run_dataset_pipeline.py",
    "../../../scripts/run_reported_analysis.py",
    "../../../src/saferestore/certification.py",
    "../../../src/saferestore/torch_models.py",
    "../../wp3/wp3_methods.py",
    "../../wp3/wp3_data.py",
    "../../wp3/compare_records.py",
]


def main() -> None:
    if run_mvtec.MANIFEST.exists():
        raise SystemExit("FREEZE_MANIFEST.json already exists")
    if run_mvtec.ACCESS_LOG.exists():
        raise SystemExit("sealed roles were already opened; a freeze now would be retrospective")
    if list((HERE / "runs").glob("seed*/calibration_records.csv")) or list((HERE / "runs").glob("seed*/test_records.csv")):
        raise SystemExit("sealed-role records already exist")
    if "Status: FROZEN" not in (HERE / "PROTOCOL.md").read_text(encoding="utf-8"):
        raise SystemExit("PROTOCOL.md does not carry 'Status: FROZEN'")
    files = list(FILES)
    for seed_dir in sorted((HERE / "runs").glob("seed*")):
        for name in ("detector.pt", "restorer.pt", "dev_results.json", "detector_val_records.csv"):
            if (seed_dir / name).exists():
                files.append(f"runs/{seed_dir.name}/{name}")
    manifest = {"status": "FROZEN",
                "frozen_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "files": {rel: run_mvtec.sha256_file(HERE / rel) for rel in files}}
    run_mvtec.MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
