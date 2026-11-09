"""WP2: per-seed sensitivity grid and alpha sweep on frozen records.

  grid : r0 in {0.10,0.25,0.50} x f0 in {0.001,0.002,0.005}
         x alpha (both endpoints) in {0.10,0.15,0.20,0.25} x delta in {0.05,0.10,0.20}
  sweep: delta = 0.10, r0 = 0.25, f0 = 0.002, alpha = 0.05..0.30 step 0.01

One row per (domain, seed, cell) with the flat summary of wp2_common.summarize_seed.
Usage: python -B run_grid.py --mode grid|sweep --domain carinthia_s|kolektor_sdd2
Writes grid_seed_<domain>.csv or sweep_seed_<domain>.csv next to this file.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import wp2_common as W

HERE = Path(__file__).resolve().parent

GRID = {
    "r0": (0.10, 0.25, 0.50),
    "f0": (0.001, 0.002, 0.005),
    "alpha": (0.10, 0.15, 0.20, 0.25),
    "delta": (0.05, 0.10, 0.20),
}
SWEEP_ALPHAS = tuple(round(0.05 + 0.01 * i, 2) for i in range(26))


def cells(mode: str):
    if mode == "grid":
        yield from itertools.product(GRID["r0"], GRID["f0"], GRID["alpha"], GRID["delta"])
    else:
        for a in SWEEP_ALPHAS:
            yield (0.25, 0.002, a, 0.10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("grid", "sweep"), required=True)
    ap.add_argument("--domain", choices=tuple(W.DOMAINS), required=True)
    args = ap.parse_args()
    out = HERE / f"{args.mode}_seed_{args.domain}.csv"
    rows = []
    t0 = time.time()
    # Loop order keeps (r0, f0) outermost so the score-model fits are reused.
    for seed in W.SEEDS:
        # Clear both caches together (fit memo keys on id(rows)).
        W._LABEL_CACHE.clear()
        W._FIT_CACHE.clear()
        W._RAW_CACHE.clear()
        for r0, f0, alpha, delta in cells(args.mode):
            res = W.analyze(args.domain, seed, r0, f0, alpha, delta)
            row = {"domain": args.domain, "seed": seed, "r0": r0, "f0": f0,
                   "alpha": alpha, "delta": delta}
            row.update(W.summarize_seed(res))
            rows.append(row)
        print(f"{args.domain} seed {seed} done, {time.time() - t0:.0f}s", flush=True)
    fields = list(rows[0])
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
