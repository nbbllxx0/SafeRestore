"""Write FREEZE_MANIFEST.json: hashes of the protocol, config, code and pre-freeze models.

Run once, after author sign-off of PROTOCOL.md.  Refuses if a manifest exists,
if PROTOCOL.md still says DRAFT, or if any sealed role was already opened.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

FILES = [
    "PROTOCOL.md",
    "protocol_config.json",
    "common.py",
    "train_models.py",
    "make_records.py",
    "analyze.py",
    "../pilot_v2.py",
    "../unet4.py",
    "../splits/m2ad_rr_v1.json",
    "../../../tim_upgrade/scripts/familywise_certification.py",
    "../../../src/saferestore/certification.py",
    "../../wp3/wp3_methods.py",
]


def main() -> None:
    if C.MANIFEST.exists():
        raise SystemExit("FREEZE_MANIFEST.json already exists")
    if C.ACCESS_LOG.exists():
        raise SystemExit("sealed roles were already opened; a freeze now would be retrospective")
    if "Status: FROZEN" not in (C.HERE / "PROTOCOL.md").read_text(encoding="utf-8"):
        raise SystemExit("PROTOCOL.md does not carry 'Status: FROZEN'")
    files = list(FILES)
    for seed_dir in sorted((C.HERE / "models").glob("seed*")):
        for name in ("detector.pt", "enhancer.pt", "calib.json"):
            if (seed_dir / name).exists():
                files.append(f"models/{seed_dir.name}/{name}")
    manifest = {
        "status": "FROZEN",
        "frozen_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": {rel: C.sha256_file(C.HERE / rel) for rel in files},
    }
    C.MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
