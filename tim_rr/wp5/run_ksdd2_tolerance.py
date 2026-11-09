"""Experiment A, step 1: regenerate KSDD2 records at several tolerance radii.

For each training seed the saved detector.pt / restorer.pt and the stored
detector threshold (tim_upgrade/experiments/ksdd2_rich_trainseed{S}) are
evaluated once per data role; the radius-dependent quantities are emitted for
every radius.  Output: tim_rr/wp5/ksdd2_tol{R}_seed{S}/{detector_val,
calibration,test}_records.csv + results.json (metadata only; checkpoints are
not copied).  Records carry the standard columns plus boundary-band columns.

Verification (written to ksdd2_tolerance_verification.json): every regenerated
8-px CSV is compared column by column with the frozen ksdd2_rich CSV, and the
seed-201 0-px and 4-px CSVs with ksdd2_tolerance{0,4}px_trainseed201.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import wp5_eval as E

HERE = Path(__file__).resolve().parent
UP = E.REPO / "tim_upgrade" / "experiments"
RADII = (0, 2, 4, 6, 8, 12, 16)
SEEDS = (201, 203, 207, 211, 223)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--radii", type=int, nargs="+", default=list(RADII))
    args = parser.parse_args()
    device = torch.device("cuda")
    verification_path = HERE / "ksdd2_tolerance_verification.json"
    verification = (json.loads(verification_path.read_text(encoding="utf-8"))
                    if verification_path.exists() else {})
    for seed in args.seeds:
        started = time.time()
        detector, restorer, threshold, meta, folder = E.load_models("kolektor_sdd2", seed, device)
        per_split = {}
        for split, filename in E.SPLIT_FILES.items():
            per_split[split] = E.evaluate("kolektor_sdd2", split, seed, detector, restorer,
                                          threshold, device, radii=args.radii)
        for radius in args.radii:
            out = HERE / f"ksdd2_tol{radius}_seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            for split, filename in E.SPLIT_FILES.items():
                E.write_csv(out / filename, per_split[split][radius])
            local = dict(meta)
            local["boundary_tolerance"] = radius
            local["checkpoint_source"] = str(folder)
            local["regenerated_by"] = "tim_rr/wp5/run_ksdd2_tolerance.py"
            (out / "results.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
        checks = {}
        refs = {8: UP / f"ksdd2_rich_trainseed{seed}"}
        if seed == 201:
            refs[0] = UP / "ksdd2_tolerance0px_trainseed201"
            refs[4] = UP / "ksdd2_tolerance4px_trainseed201"
        for radius, ref in refs.items():
            if radius not in args.radii:
                continue
            for split, filename in E.SPLIT_FILES.items():
                rep = E.compare_records(per_split[split][radius], ref / filename)
                checks[f"r{radius}_{split}"] = {"exact": rep.get("exact"), "frozen": rep["frozen"],
                                               "label_mismatches": rep["label_mismatches"],
                                               "rows": rep["n_new"]}
        verification[str(seed)] = checks
        verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
        print(seed, f"{time.time() - started:.0f}s", all(c["exact"] for c in checks.values()), flush=True)
        del detector, restorer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
