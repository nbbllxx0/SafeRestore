"""Experiment B, step 2: qualification on native-acquisition records (R2.1).

Inputs: tim_rr/wp5/native_{domain}_seed{S}/ (run_native.py).  The definitive
analysis is run unchanged except that the action set is the four native
candidates, so the policy family is raw, smoothed, sharpened, learned,
restoration pool {smoothed, sharpened, learned} and all-action pool - six
policies, portfolio allocation delta/6 per policy (a delta/8 sensitivity row,
as if the eight-policy family were kept, is also reported).  Score models are
fitted on the native detector-validation role (Carinthia-S class-balanced,
KSDD2 MLE, as published); tuning / hold-out from the qualification split; the
test role is descriptive.  Targets alpha = 0.15, delta = 0.10, r0 = 0.25,
f0 = 0.002.

Outputs (tim_rr/wp5/): exp_b_results/*.json, exp_b_incidence.csv,
exp_b_per_seed.csv, exp_b_summary.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import wp5_analysis as A
import wp5_eval as E

HERE = Path(__file__).resolve().parent
SEEDS = (201, 203, 207, 211, 223)
DOMAINS = ("carinthia_s", "kolektor_sdd2")
NATIVE_METHODS = ("raw_nearest", "smooth_bicubic", "unsharp_bicubic", "learned")
LABEL = {"raw_nearest": "raw", "smooth_bicubic": "smoothed", "unsharp_bicubic": "sharpened",
         "learned": "learned", "bilinear": "bilinear", "bicubic": "bicubic"}
FLOOR = 0.25


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def incidence(rows: list[dict], domain: str, seed: int, arm: str, role: str) -> list[dict]:
    out = []
    methods = sorted({r["method"] for r in rows}, key=lambda m: list(LABEL).index(m))
    for method in methods:
        local = [r for r in rows if r["method"] == method]
        pos = [r for r in local if r["has_defect"] == "1"]
        cln = [r for r in local if r["has_defect"] == "0"]
        out.append({
            "domain": domain, "seed": seed, "arm": arm, "role": role, "action": LABEL[method],
            "n_pos": len(pos), "n_clean": len(cln),
            "loss_e": sum(float(r["pixel_recall"]) < FLOOR for r in pos),
            "act_pos_e": sum(int(r["invention_error"]) for r in pos),
            "act_clean_e": sum(int(r["invention_error"]) for r in cln),
            "act_all_e": sum(int(r["invention_error"]) for r in local),
            "loss_rate": sum(float(r["pixel_recall"]) < FLOOR for r in pos) / len(pos) if pos else None,
            "act_rate_all": sum(int(r["invention_error"]) for r in local) / len(local),
            "mean_psnr_db": float(np.mean([float(r["psnr"]) for r in local])),
        })
    return out


def main() -> None:
    (HERE / "exp_b_results").mkdir(exist_ok=True)
    inc, per_seed = [], []
    for domain in DOMAINS:
        for seed in SEEDS:
            folder = HERE / f"native_{domain}_seed{seed}"
            nominal = Path(str(E.DATA[domain]["run"]).format(seed=seed))
            for role, filename in E.SPLIT_FILES.items():
                inc += incidence(read(folder / filename), domain, seed, "native", role)
                inc += incidence(read(nominal / filename), domain, seed, "nominal (published records)", role)
            for variant, delta in (("delta_over_6", 0.10), ("delta_over_8_portfolio_only", 0.10 * 6 / 8)):
                result = A.analyze(folder, domain, seed, methods=NATIVE_METHODS, delta=delta)
                if variant == "delta_over_6":
                    (HERE / "exp_b_results" / f"{domain}_seed{seed}.json").write_text(
                        json.dumps(A.json_safe(result), indent=1, allow_nan=False), encoding="utf-8")
                row = {"domain": domain, "seed": seed, "variant": variant}
                row.update(A.summarize(result))
                per_seed.append(row)
            print("done", domain, seed, flush=True)
            A.clear_cache()
    for name, rows in (("exp_b_incidence.csv", inc), ("exp_b_per_seed.csv", per_seed)):
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with (HERE / name).open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    summary = []
    for domain in DOMAINS:
        for arm in ("native", "nominal (published records)"):
            for role in ("calibration", "test"):
                for action in ("raw", "bilinear", "bicubic", "smoothed", "sharpened", "learned"):
                    local = [r for r in inc if r["domain"] == domain and r["arm"] == arm
                             and r["role"] == role and r["action"] == action]
                    if not local:
                        continue
                    summary.append({
                        "domain": domain, "arm": arm, "role": role, "action": action,
                        "n_pos": local[0]["n_pos"], "n_clean": local[0]["n_clean"],
                        "loss_rate_mean": float(np.mean([r["loss_rate"] for r in local])),
                        "act_pos_rate_mean": float(np.mean([r["act_pos_e"] / r["n_pos"] for r in local])),
                        "act_clean_rate_mean": (float(np.mean([r["act_clean_e"] / r["n_clean"] for r in local]))
                                                if local[0]["n_clean"] else None),
                        "act_all_rate_mean": float(np.mean([r["act_rate_all"] for r in local])),
                        "loss_e_by_seed": ";".join(str(r["loss_e"]) for r in local),
                        "act_all_e_by_seed": ";".join(str(r["act_all_e"]) for r in local),
                    })
    with (HERE / "exp_b_summary.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("wrote exp_b_*")


if __name__ == "__main__":
    main()
