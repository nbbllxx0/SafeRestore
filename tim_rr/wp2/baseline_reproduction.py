"""WP2 step 0: reproduce the definitive TIM analysis through the WP2 harness.

For both domains, all five seeds and both recall floors (0.25, 0.50) at
f0 = 0.002, alpha = 0.15, delta = 0.10, the harness result (after json_safe)
is compared field by field with the definitive JSON in
tim_upgrade/experiments/tim_analysis/.  Writes baseline_reproduction.json.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import wp2_common as W

HERE = Path(__file__).resolve().parent


def diff(a, b, path="", out=None, tol=0.0):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: key missing on one side")
            else:
                diff(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
            if a != b:
                out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def main() -> None:
    report = {"settings": {"f0": 0.002, "alpha": 0.15, "delta": 0.10}, "runs": [], "summary": {}}
    t0 = time.time()
    for domain in W.DOMAINS:
        for floor in (0.25, 0.50):
            seeds_out = []
            for seed in W.SEEDS:
                res = W.analyze(domain, seed, floor, 0.002, 0.15, 0.10)
                mine = json.loads(json.dumps(W.reported.json_safe(res), allow_nan=False))
                ref_path = W.DEFINITIVE_DIR / f"{domain}_seed{seed}_floor{floor:.2f}.json"
                ref = json.loads(ref_path.read_text(encoding="utf-8"))
                d = diff(mine, ref)
                s = W.summarize_seed(res)
                report["runs"].append({
                    "domain": domain, "seed": seed, "floor": floor,
                    "reference": str(ref_path.relative_to(W.REPO)),
                    "n_differences": len(d), "differences_first20": d[:20],
                    "portfolio_selected": s["portfolio_selected"],
                })
                seeds_out.append(s)
            key = f"{domain}_floor{floor:.2f}"
            report["summary"][key] = {
                "portfolio_qualified": sum(s["portfolio_qualified"] for s in seeds_out),
                "portfolio_selections": [s["portfolio_selected"] for s in seeds_out],
                "familywise_single": {p: sum(s[f"fam_pass__{p}"] for s in seeds_out) for p in W.POLICIES},
                "marginal": {p: sum(s[f"marg_pass__{p}"] for s in seeds_out) for p in W.POLICIES},
                "tuning_selected_qualified": sum(s["tuning_selected_qualified"] for s in seeds_out),
            }
    report["all_identical"] = all(r["n_differences"] == 0 for r in report["runs"])
    report["seconds"] = round(time.time() - t0, 1)
    (HERE / "baseline_reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in report["summary"].items():
        print(k, json.dumps(v))
    print("all_identical", report["all_identical"], "seconds", report["seconds"])
    for r in report["runs"]:
        if r["n_differences"]:
            print(r["domain"], r["seed"], r["floor"], r["n_differences"], r["differences_first20"][:5])


if __name__ == "__main__":
    main()
