"""Generate LaTeX tables for the TIM manuscript from tim_analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "tim_upgrade" / "experiments" / "tim_analysis"
TABLES = REPO / "tim_upgrade" / "manuscript" / "tables"

SEEDS = (201, 203, 207, 211, 223)
POLICY_LABELS = {
    "raw_nearest": "Raw display",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "smooth_bicubic": "Smoothed bicubic",
    "unsharp_bicubic": "Sharpened bicubic",
    "learned": "Learned",
    "restoration_pool": "Restoration-only pool",
    "action_pool": "All-action policy",
}
DOMAIN_LABELS = {"carinthia_s": "Carinthia-S", "kolektor_sdd2": "KolektorSDD2"}


def load(domain: str, seed: int, floor: float) -> dict | None:
    path = ANALYSIS / f"{domain}_seed{seed}_floor{floor:.2f}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.{digits}f}"


def policy_comparison_table(domain: str, floor: float) -> str:
    """Marginal vs familywise per-policy pass counts and coverage."""

    rows = []
    for name, label in POLICY_LABELS.items():
        legacy_passes = 0
        fam_count = 0
        fam_cov = []
        for seed in SEEDS:
            data = load(domain, seed, floor)
            if data is None:
                continue
            legacy = data["legacy_marginal"]["policies"][name]
            legacy_passes += int(bool(legacy["certificate"]["passes"]))
            fam = data["familywise_single_policy"][name]
            if fam["certified_threshold"] is not None:
                fam_count += 1
                fam_cov.append(float(fam["test"]["coverage"]))
        mean_cov = pct(sum(fam_cov) / len(fam_cov)) if fam_cov else "--"
        rows.append(
            f"{label} & {legacy_passes}/5 & {fam_count}/5 & {mean_cov} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "Policy & Marginal & Familywise & Test cov.\\ (\\%) \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def portfolio_table(floor: float, domain: str = "carinthia_s") -> str:
    """Per-seed portfolio selections and evidence with test corroboration."""

    rows = []
    for seed in SEEDS:
        data = load(domain, seed, floor)
        if data is None:
            continue
        pf = data["familywise_portfolio"]
        name = pf["selected_policy"]
        if name is None:
            rows.append(f"{seed} & --- (none certified) & -- & -- & -- & -- & -- & -- \\\\")
            continue
        ev = pf["policies"][name]
        test = pf.get("selected_test") or {}
        rows.append(
            f"{seed} & {POLICY_LABELS[name]} & "
            f"{ev['deletion_errors']}/{ev['deletion_denominator']} ({ev['deletion_ucb']:.3f}) & "
            f"{ev['invention_errors']}/{ev['invention_denominator']} ({ev['invention_ucb']:.3f}) & "
            f"{pct(ev['coverage'])} & {pct(test.get('coverage'))} & "
            f"{pct(test.get('deletion_risk'))} & {pct(test.get('invention_risk'))} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
        "Seed & Selected policy & Loss $e/n$ ($U$) & Act.\\ $e/n$ ($U$) & "
        "Cert.\\ cov.\\ (\\%) & Test cov.\\ (\\%) & Test loss (\\%) & "
        "Test act.\\ (\\%) \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def ksdd2_frontier_table(floor: float) -> str:
    """Per-seed best descriptive frontier level on KolektorSDD2."""

    rows = []
    for seed in SEEDS:
        data = load("kolektor_sdd2", seed, floor)
        if data is None:
            continue
        best_name, best = min(
            data["familywise_single_policy"].items(),
            key=lambda kv: kv[1].get("min_certifiable_alpha_descriptive", 9.0),
        )
        fail = best.get("first_failure", {})
        rows.append(
            f"{seed} & {POLICY_LABELS[best_name]} & "
            f"{best['min_certifiable_alpha_descriptive']:.3f} & "
            f"{fail.get('deletion_errors','--')}/{fail.get('deletion_denominator','--')} "
            f"({fail.get('deletion_ucb', float('nan')):.3f}) & "
            f"{fail.get('invention_errors','--')}/{fail.get('invention_denominator','--')} "
            f"({fail.get('invention_ucb', float('nan')):.3f}) \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}llccc@{}}\n\\toprule\n"
        "Seed & Best policy & $\\alpha^{*}$ & "
        "FF loss $e/n$ ($U$) & FF act.\\ $e/n$ ($U$) \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=float, default=0.25)
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    for domain in DOMAIN_LABELS:
        out = TABLES / f"tab_policy_comparison_{domain}.tex"
        out.write_text(policy_comparison_table(domain, args.floor), encoding="utf-8")
        print("wrote", out)
    out = TABLES / "tab_portfolio.tex"
    out.write_text(portfolio_table(args.floor), encoding="utf-8")
    print("wrote", out)
    out = TABLES / "tab_ksdd2_frontier.tex"
    out.write_text(ksdd2_frontier_table(args.floor), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
