"""Experiment B, step 1: native-acquisition records (R2.1), both domains.

The acquired reference image itself is the indication: no synthetic
degradation.  Candidates (wp5_eval.native_candidates): raw image, Gaussian
smoothing (3x3, sigma 0.8) and unsharp sharpening (5x5, sigma 1.0, amount 0.8)
- the pipeline's smooth / sharp kernels applied at native resolution - and the
learned restorer applied to the native image (trained on bicubic-upsampled
degraded -> reference pairs; applying it to a native image is outside its
training distribution).  Same detector checkpoint and stored pixel threshold
per seed; same feature definitions (forward consistency is measured against
the native observation; residual and disagreement against the raw image).
KSDD2 uses the published 8-px tolerance ring.

Output: tim_rr/wp5/native_{domain}_seed{S}/{detector_val,calibration,test}_records.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import wp5_eval as E

HERE = Path(__file__).resolve().parent
SEEDS = (201, 203, 207, 211, 223)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=["carinthia_s", "kolektor_sdd2"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    device = torch.device("cuda")
    for domain in args.domains:
        for seed in args.seeds:
            started = time.time()
            detector, restorer, threshold, meta, folder = E.load_models(domain, seed, device)
            out = HERE / f"native_{domain}_seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            for split, filename in E.SPLIT_FILES.items():
                rows = E.evaluate(domain, split, seed, detector, restorer, threshold, device,
                                  mode="native", radii=(8,))
                key = 8 if domain == "kolektor_sdd2" else None
                E.write_csv(out / filename, rows[key])
            local = dict(meta)
            local.update({"indication": "native (no synthetic degradation)",
                          "checkpoint_source": str(folder),
                          "regenerated_by": "tim_rr/wp5/run_native.py"})
            (out / "results.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
            print(domain, seed, f"{time.time() - started:.0f}s", flush=True)
            del detector, restorer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
