"""WP6 confirmatory study, step 2: image-action records for one seed and one role.

  --role detector_val   every view of every detector_val specimen (risk-model
                        fitting and the development dry run); open before freeze.
  --role tuning|holdout one view per specimen (common.select_units); refused
                        unless FREEZE_MANIFEST.json verifies, and every opening
                        is appended to sealed_role_access_log.txt.

Output: confirm/records/seed<N>/<role>_records.csv, one row per (view, action).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import common as C  # noqa: E402  (also puts tim_rr/wp6 and src on sys.path)
import pilot_v2 as P
from train_models import enhance
from unet4 import UNet4


def prepare(v: dict, photo: dict) -> dict:
    i07, i10 = C.rgb(v["I07"]), C.rgb(v["I10"])
    mask = C.load_mask(v, i10.shape[:2])
    clean = C.clean_region(C.foreground(i10), mask)
    cands = {m: C.apply_classical(m, i07, photo) for m in C.METHODS if m != "learned"}
    return {"i07": i07, "mask": mask, "clean": clean, "cands": cands}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--role", choices=("detector_val", "tuning", "holdout"), required=True)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N views only (detector_val)")
    ap.add_argument("--tag", default="", help="model/record folder suffix (development variants)")
    ap.add_argument("--pixel-fpr", type=float, default=C.PIXEL_FPR,
                    help="development: pixel-threshold FPR target (tau from tau_grid.json); f0 = 2 x this")
    args = ap.parse_args()
    if args.limit and args.role != "detector_val":
        raise SystemExit("--limit is for development smoke tests only")
    models = C.HERE / "models" / f"seed{args.seed}{args.tag}"
    calib = json.loads((models / "calib.json").read_text(encoding="utf-8"))
    tau, photo = calib["tau"], calib["photometric"]
    f0, out_tag = C.F0, args.tag
    if args.pixel_fpr != C.PIXEL_FPR:
        if args.role != "detector_val":
            raise SystemExit("--pixel-fpr is a development option (detector_val only)")
        tau = json.loads((models / "tau_grid.json").read_text(encoding="utf-8"))["tau"][f"{args.pixel_fpr:g}"]
        f0, out_tag = 2 * args.pixel_fpr, f"{args.tag}_px{args.pixel_fpr:g}"
    device = torch.device("cuda")
    detector = UNet4(3, calib["args"]["base"]).to(device)
    detector.load_state_dict(torch.load(models / "detector.pt", map_location=device))
    enhancer = UNet4(3, calib["args"]["base"], out_ch=3).to(device)
    enhancer.load_state_dict(torch.load(models / "enhancer.pt", map_location=device))
    detector.eval()
    enhancer.eval()

    views = C.build_views({args.role}, purpose=f"make_records seed={args.seed}")
    if args.role != "detector_val":
        views = C.select_units(views)
    if args.limit:
        views = views[: args.limit]
    out = C.HERE / "records" / f"seed{args.seed}{out_tag}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"{args.role}_records.csv" if not args.limit else f"{args.role}_smoke_records.csv")
    print(f"seed {args.seed} role {args.role}: {len(views)} views -> {path}", flush=True)

    t0 = time.time()
    writer = None
    with path.open("w", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(12) as pool:
        for start in range(0, len(views), args.chunk):
            batch = views[start:start + args.chunk]
            prepped = list(pool.map(lambda v: prepare(v, photo), batch))
            learned = enhance(enhancer, [p["i07"] for p in prepped], device)
            for p, img in zip(prepped, learned):
                p["cands"]["learned"] = img
            probs = {m: P.predict(detector, [p["cands"][m] for p in prepped], device) for m in C.METHODS}

            def rows_for(j: int) -> list[dict]:
                p = prepped[j]
                return [C.record_row(batch[j], m, p["cands"][m], p["i07"], probs[m][j], probs["raw"][j],
                                     tau, p["mask"], p["clean"], f0) for m in C.METHODS]

            for rows in pool.map(rows_for, range(len(batch))):
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
            done = start + len(batch)
            print(f"  {done}/{len(views)} views, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
