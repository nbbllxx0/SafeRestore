"""Additional TIM figures: two-domain examples, policy-comparison chart,
and qualitative decision panel."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Dataset roots. Neither dataset is redistributed with this code; set these
# environment variables to your local copies (see configs/kolektor_sdd2.json).
CARINTHIA_ROOT = os.environ.get(
    "SAFERESTORE_CARINTHIA_ROOT", "<path to your Carinthia-S data directory>"
)
KSDD2_ROOT = os.environ.get(
    "SAFERESTORE_KSDD2_ROOT", "<path to your KolektorSDD2 dataset directory>"
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

FIGDIR = REPO / "tim_upgrade" / "manuscript" / "figures"
ANALYSIS = REPO / "tim_upgrade" / "experiments" / "tim_analysis"
SEEDS = (201, 203, 207, 211, 223)

BLUE = "#2b5f9e"
AMBER = "#c8811a"
CHARCOAL = "#3a3a3a"
GREEN = "#2e7d32"
LIGHT = "#eef2f7"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    # IEEE PDF eXpress rejects Type 3 fonts; 42 embeds TrueType instead.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ------------------------------------------------------------ two domains
def _load_pair(dataset: str, sample_id: str):
    if dataset == "carinthia_s":
        from saferestore.torch_data import PublicSegmentationDataset

        ds = PublicSegmentationDataset(
            dataset="carinthia_s",
            root=CARINTHIA_ROOT,
            split_file=REPO / "splits" / "carinthia_s_v4.json",
            split="test",
        )
    else:
        from ksdd2_data import Ksdd2SegmentationDataset

        ds = Ksdd2SegmentationDataset(
            dataset="kolektor_sdd2",
            root=KSDD2_ROOT,
            split_file=REPO / "tim_upgrade" / "splits" / "kolektor_sdd2_v1.json",
            split="test",
        )
    index = next(i for i, r in enumerate(ds.records) if r["sample_id"] == sample_id)
    item = ds[index]
    return item["image"], item["mask"]


def _first_positive(dataset: str, severity: str, records_csv: Path) -> str:
    from scripts.analyze_router_pool import read_rows

    rows = read_rows(records_csv)
    for r in rows:
        if (r["method"] == "raw_nearest" and int(r["has_defect"])
                and r["severity"] == severity):
            return r["sample_id"]
    raise RuntimeError("no positive found")


def fig_domains() -> None:
    from scripts.run_dataset_pipeline import degrade_mixed_batch

    cases = [
        ("carinthia_s", "Carinthia-S",
         REPO / "experiments" / "v4_carinthia_primary_trainseed211" / "test_records.csv"),
        ("kolektor_sdd2", "KolektorSDD2",
         REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed201" / "test_records.csv"),
    ]
    # Single-column figure (3.5 in): keep native aspect ratios via
    # height_ratios (Carinthia square, KSDD2 transposed 256x640).
    fig, axes = plt.subplots(
        2, 3, figsize=(3.5, 1.95),
        gridspec_kw={"hspace": 0.06, "wspace": 0.04,
                     "height_ratios": [2.5, 1.0]},
    )
    from saferestore.degradation import restore_methods

    titles = ["reference $Y$", "controlled\nobservation", "annotation $M$"]
    for row, (dataset, label, records) in enumerate(cases):
        sid = _first_positive(dataset, "severe", records)
        image, mask = _load_pair(dataset, sid)
        batch_img = image.unsqueeze(0)
        low, _, _ = degrade_mixed_batch(
            batch_img, [sid], split="test", corruption_key="v4corr:eval")
        display = restore_methods(low, tuple(batch_img.shape[-2:]))["raw_nearest"]
        panels = [image[0].numpy(), display[0, 0].numpy(), image[0].numpy()]
        for col, arr in enumerate(panels):
            ax = axes[row, col]
            if dataset == "kolektor_sdd2":
                arr = arr.T  # portrait -> landscape for layout
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1, aspect="auto")
            if col == 2:
                m = mask[0].numpy()
                if dataset == "kolektor_sdd2":
                    m = m.T
                ax.contour(m, levels=[0.5], colors=[BLUE], linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=7)
            if col == 0:
                ax.set_ylabel(label, fontsize=6.4)
    fig.savefig(FIGDIR / "fig_domains.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_domains.pdf")


# --------------------------------------------------- policy comparison
POLICY_LABELS = {
    "raw_nearest": "Raw",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "smooth_bicubic": "Smooth bic.",
    "unsharp_bicubic": "Sharp bic.",
    "learned": "Learned",
    "restoration_pool": "Resto. pool",
    "action_pool": "All-action",
}


def fig_policy_comparison() -> None:
    marg, fam = [], []
    for name in POLICY_LABELS:
        m = f = 0
        for seed in SEEDS:
            data = json.loads(
                (ANALYSIS / f"carinthia_s_seed{seed}_floor0.25.json").read_text())
            m += int(bool(data["legacy_marginal"]["policies"][name]["certificate"]["passes"]))
            f += int(data["familywise_single_policy"][name]["certified_threshold"] is not None)
        marg.append(m); fam.append(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    x = np.arange(len(POLICY_LABELS))
    ax.bar(x - 0.19, marg, 0.38, color=CHARCOAL, label="marginal tuned gate")
    ax.bar(x + 0.19, fam, 0.38, color=BLUE, label="familywise sequence")
    ax.set_xticks(x, list(POLICY_LABELS.values()), rotation=28, ha="right")
    ax.set_ylabel("certified repetitions (of 5)")
    ax.set_ylim(0, 5.6)
    ax.set_yticks(range(6))
    ax.set_title("(a) per-policy certifications, Carinthia-S", loc="left")
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    certcov, testcov, labels = [], [], []
    for seed in SEEDS:
        data = json.loads(
            (ANALYSIS / f"carinthia_s_seed{seed}_floor0.25.json").read_text())
        pf = data["familywise_portfolio"]
        name = pf["selected_policy"]
        certcov.append(pf["policies"][name]["coverage"] * 100)
        testcov.append(pf["selected_test"]["coverage"] * 100)
        labels.append(f"{seed}\n{POLICY_LABELS[name]}")
    x = np.arange(len(SEEDS))
    ax.bar(x - 0.19, certcov, 0.38, color=BLUE, label="certification role")
    ax.bar(x + 0.19, testcov, 0.38, color=GREEN, label="held-out test")
    ax.set_xticks(x, labels, fontsize=6.8)
    ax.set_ylabel("return coverage (%)")
    ax.set_ylim(0, 66)
    ax.set_title("(b) portfolio selection per repetition", loc="left")
    ax.legend(frameon=False, loc="upper left", ncols=2,
              columnspacing=0.9, borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGDIR / "fig_policy_comparison.pdf")
    plt.close(fig)
    print("wrote fig_policy_comparison.pdf")


# ------------------------------------------------------- qualitative
def fig_qualitative() -> None:
    """Carinthia seed-211 test decisions under the certified portfolio gate."""

    import scripts.run_reported_analysis as reported
    from scripts.analyze_router_pool import read_rows
    from saferestore.certification import DualRiskTargets
    from saferestore.torch_models import build_detector, build_restorer
    from scripts.run_dataset_pipeline import degrade_mixed_batch
    from saferestore.degradation import restore_methods

    targets = DualRiskTargets()
    folder = REPO / "experiments" / "v4_carinthia_primary_trainseed211"
    data = json.loads((ANALYSIS / "carinthia_s_seed211_floor0.25.json").read_text())
    pf = data["familywise_portfolio"]
    policy = pf["selected_policy"]
    threshold = pf["policies"][policy]["certified_threshold"]

    train = read_rows(folder / "detector_val_records.csv")
    test = read_rows(folder / "test_records.csv")
    models = reported.fit_risk_models(train, 211, 0.25)
    arrays = reported.build_policy_arrays(test, models, targets, 0.25)
    a = arrays[policy]
    accepted = np.asarray(a["score"]) <= threshold
    sids = np.asarray(a["sample_id"])
    methods = np.asarray(a["method"])
    positive = np.asarray(a["positive"])
    deletion = np.asarray(a["deletion"]).astype(bool)

    meta = json.loads((folder / "results.json").read_text())
    tau = float(meta["detector_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = build_detector("tiny_unet", base=16).to(device)
    detector.load_state_dict(torch.load(folder / "detector.pt", weights_only=True))
    detector.eval()
    restorer = build_restorer("compact", channels=32, blocks=6).to(device)
    restorer.load_state_dict(torch.load(folder / "restorer.pt", weights_only=True))
    restorer.eval()

    # Example A: accepted positive, non-raw action, no incident.
    # Example B: reviewed positive (gate score above threshold).
    idx_a = next(i for i in range(len(sids))
                 if accepted[i] and positive[i] and not deletion[i]
                 and methods[i] not in ("raw_nearest",))
    idx_b = next(i for i in range(len(sids))
                 if not accepted[i] and positive[i])

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9),
                             gridspec_kw={"hspace": 0.3, "wspace": 0.05})
    for row, (idx, tag) in enumerate([(idx_a, "returned"), (idx_b, "review")]):
        sid = str(sids[idx])
        sel_method = str(methods[idx])
        image, mask = _load_pair("carinthia_s", sid)
        batch_img = image.unsqueeze(0).to(device)
        with torch.no_grad():
            low, bicubic, _ = degrade_mixed_batch(
                batch_img, [sid], split="test", corruption_key="v4corr:eval")
            cands = restore_methods(low, tuple(batch_img.shape[-2:]))
            cands["learned"] = restorer(bicubic)
            shown = ["raw_nearest", "bicubic",
                     sel_method if sel_method not in ("raw_nearest", "bicubic")
                     else "learned"]
            panels = [(image[0].numpy(), None, "reference")]
            for m in shown:
                prob = torch.sigmoid(detector(cands[m]))[0, 0].cpu().numpy()
                img = cands[m][0, 0].detach().cpu().numpy()
                star = " (selected)" if m == sel_method else ""
                panels.append((img, prob, m.replace("_", " ") + star))
        for col, (arr, prob, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.contour(mask[0].numpy(), levels=[0.5], colors=[BLUE],
                       linewidths=0.9)
            if prob is not None:
                ax.contour(prob, levels=[tau], colors=[AMBER], linewidths=0.9,
                           linestyles="--")
            if row == 0:
                # Example A defects are small at print size: enlarge the
                # largest annotated component in a corner inset (stray
                # annotation specks would otherwise inflate the box).
                from scipy import ndimage

                m = mask[0].numpy()
                lab, ncomp = ndimage.label(m > 0.5)
                if ncomp:
                    sizes = ndimage.sum(m > 0.5, lab, range(1, ncomp + 1))
                    lab_main = 1 + int(np.argmax(sizes))
                    ys, xs = np.nonzero(lab == lab_main)
                else:
                    ys, xs = np.nonzero(m > 0.5)
                if len(ys):
                    pad = 24
                    y0 = max(int(ys.min()) - pad, 0)
                    y1 = min(int(ys.max()) + pad, arr.shape[0])
                    x0 = max(int(xs.min()) - pad, 0)
                    x1 = min(int(xs.max()) + pad, arr.shape[1])
                    axi = ax.inset_axes([0.55, 0.02, 0.43, 0.43])
                    axi.imshow(arr[y0:y1, x0:x1], cmap="gray",
                               vmin=0, vmax=1)
                    axi.contour(m[y0:y1, x0:x1], levels=[0.5],
                                colors=[BLUE], linewidths=0.9)
                    if prob is not None:
                        axi.contour(prob[y0:y1, x0:x1], levels=[tau],
                                    colors=[AMBER], linewidths=0.9,
                                    linestyles="--")
                    axi.set_xticks([]); axi.set_yticks([])
                    for s in axi.spines.values():
                        s.set_color("white"); s.set_linewidth(1.0)
                    ax.add_patch(mpatches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False,
                        ec="white", lw=0.7))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=7.6)
            if col == 0:
                decision = ("returned automatically" if tag == "returned"
                            else "routed to review")
                ax.set_ylabel(f"Example {'AB'[row]}:\n{decision}", fontsize=7.6)
    fig.savefig(FIGDIR / "fig_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_qualitative.pdf")


if __name__ == "__main__":
    fig_domains()
    fig_policy_comparison()
    fig_qualitative()
