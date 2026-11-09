"""Development: detector pixel thresholds at several clean-pixel FPR targets.

Same rule as train_models.py (quantile of detector probabilities over the clean
region of detector_val views at I10, up to 20 000 sampled pixels per view), for
the FPR targets given.  Writes models/seed<N><tag>/tau_grid.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import common as C  # noqa: E402  (also puts tim_rr/wp6 and src on sys.path)
import pilot_v2 as P
from unet4 import UNet4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fprs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    args = ap.parse_args()
    folder = C.HERE / "models" / f"seed{args.seed}{args.tag}"
    calib = json.loads((folder / "calib.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    detector = UNet4(3, calib["args"]["base"]).to(device)
    detector.load_state_dict(torch.load(folder / "detector.pt", map_location=device))
    detector.eval()
    views = C.build_views({"detector_val"}, purpose="compute_tau")
    rng = np.random.default_rng(args.seed)
    neg = []
    for start in range(0, len(views), 64):
        batch = views[start:start + 64]
        with ThreadPoolExecutor(12) as pool:
            i10 = list(pool.map(lambda v: C.rgb(v["I10"]), batch))
        probs = P.predict(detector, i10, device)
        for v, img, p in zip(batch, i10, probs):
            vals = p[C.clean_region(C.foreground(img), C.load_mask(v, img.shape[:2]))]
            if len(vals):
                neg.append(vals[rng.choice(len(vals), size=min(20000, len(vals)), replace=False)])
    neg = np.concatenate(neg)
    grid = {f"{f:g}": float(np.quantile(neg, 1 - f, method="higher")) for f in args.fprs}
    out = {"seed": args.seed, "tag": args.tag, "calib_tau_1e-3": calib["tau"], "tau": grid}
    (folder / "tau_grid.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
