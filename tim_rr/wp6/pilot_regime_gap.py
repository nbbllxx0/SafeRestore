"""WP6 step 1: development-only pilot on M2AD.

Question: with a detector trained on the reference regime (I10), does the
poor regime (I07) produce materially more detector evidence-loss incidents
(recall < r0) than I10 on held-aside development specimens, and do two fixed
application-time enhancements change that?

Reads ONLY specimens whose frozen role (splits/m2ad_rr_v1.json) is train or
detector_val; asserts this for every file opened. Writes only under
tim_rr/wp6/pilot/.

Endpoints (same definitions as the paper): r0 = 0.25 recall floor on
mask >= 128; activation = clean-foreground FPR > 0.002, clean foreground =
(I10 grey > 20/255) minus the mask; detector pixel threshold = 1 - 1e-3
quantile of clean-foreground probabilities on detector_val at I10.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.exposure import equalize_adapthist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from saferestore.torch_models import build_detector, soft_dice_loss  # noqa: E402

ALLOWED = {"train", "detector_val"}
R0, F0, PIXEL_FPR = 0.25, 0.002, 1e-3
CROP = 384


def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def build_index(root: Path, meta: dict, roles: dict, categories: list[str]) -> list[dict]:
    views: dict[tuple, dict] = {}
    for split in ("train", "test"):
        for cls in categories:
            for it in meta[split].get(cls, []):
                kind = it["img_path"].split("/")[1]
                role = roles[f"{cls}/{kind}/{it['object_name']}"]
                if role not in ALLOWED:
                    continue
                key = (cls, kind, it["object_name"], it["view"])
                v = views.setdefault(key, {"cls": cls, "kind": kind, "obj": it["object_name"],
                                           "view": it["view"], "role": role, "mask": None,
                                           "detectable": {}})
                if it["illumination"] in ("07", "10"):
                    v[f"I{it['illumination']}"] = root / it["img_path"]
                    v["detectable"][it["illumination"]] = it["detectable"]
                if it["image_anomaly"]:
                    v["mask"] = root / it["mask_path"]
    out = [v for v in views.values() if "I07" in v and "I10" in v]
    for v in out:
        assert v["role"] in ALLOWED
    return out


def load_view(v: dict) -> dict:
    assert v["role"] in ALLOWED, v
    d = {"I10": gray(v["I10"]), "I07": gray(v["I07"])}
    d["mask"] = (gray(v["mask"]) >= 128) if v["mask"] is not None else None
    return d


def sample_crop(img: np.ndarray, mask: np.ndarray | None, rng: np.random.Generator):
    h, w = img.shape
    if mask is not None and mask.any() and rng.random() < 0.5:
        ys, xs = np.nonzero(mask)
        j = rng.integers(len(ys))
        cy, cx = ys[j] + rng.integers(-CROP // 3, CROP // 3), xs[j] + rng.integers(-CROP // 3, CROP // 3)
    else:
        fg = np.argwhere(img > 20)
        cy, cx = fg[rng.integers(len(fg))] if len(fg) else (h // 2, w // 2)
    y0 = int(np.clip(cy - CROP // 2, 0, h - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, w - CROP))
    m = mask[y0:y0 + CROP, x0:x0 + CROP] if mask is not None else np.zeros((CROP, CROP), bool)
    return img[y0:y0 + CROP, x0:x0 + CROP], m


@torch.no_grad()
def predict(model, imgs: list[np.ndarray], device) -> list[np.ndarray]:
    out = []
    for i in range(0, len(imgs), 4):
        x = torch.from_numpy(np.stack(imgs[i:i + 4]).astype(np.float32) / 255.0)[:, None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x).float())
        out.extend(p[:, 0].cpu().numpy())
    return out


def positive_dice(model, data: list[dict], device) -> float:
    pos = [d for d in data if d["mask"] is not None and d["mask"].any()]
    probs = predict(model, [d["I10"] for d in pos], device)
    scores = []
    for d, p in zip(pos, probs):
        call = p >= 0.5
        inter = (call & d["mask"]).sum()
        scores.append(2 * inter / (call.sum() + d["mask"].sum() + 1e-9))
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path(__file__).with_name("splits") / "m2ad_rr_v1.json")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("pilot"))
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=301)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    meta = json.load(open(args.meta, encoding="utf-8"))
    roles = json.load(open(args.split, encoding="utf-8"))["roles"]
    index = build_index(args.root, meta, roles, args.categories)
    t0 = time.time()
    with ThreadPoolExecutor(16) as pool:
        data = list(pool.map(load_view, index))
    train = [d for d, v in zip(data, index) if v["role"] == "train"]
    val = [(d, v) for d, v in zip(data, index) if v["role"] == "detector_val"]
    print(f"loaded {len(train)} train / {len(val)} val views in {time.time()-t0:.0f}s", flush=True)

    # Fixed enhancement parameters from the train role only.
    def fg_mean(img, ref):
        fg = ref > 20
        return img[fg].mean() if fg.any() else np.nan
    gain = float(np.nanmean([fg_mean(d["I10"], d["I10"]) for d in train]) /
                 np.nanmean([fg_mean(d["I07"], d["I10"]) for d in train]))
    print(f"train-role I10/I07 foreground mean ratio (gain) = {gain:.3f}", flush=True)

    model = build_detector("tiny_unet", base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state, history = -1.0, None, []
    vdata = [d for d, _ in val]
    for step in range(1, args.steps + 1):
        model.train()
        xs, ms = [], []
        for _ in range(args.batch):
            d = train[rng.integers(len(train))]
            x, m = sample_crop(d["I10"], d["mask"], rng)
            if rng.random() < 0.5:
                x, m = x[:, ::-1], m[:, ::-1]
            xs.append(x.astype(np.float32) / 255.0)
            ms.append(m.astype(np.float32))
        x = torch.from_numpy(np.stack(xs))[:, None].to(device)
        m = torch.from_numpy(np.stack(ms))[:, None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        pos = float(m.sum())
        weight = min(80.0, max(1.0, (m.numel() - pos) / max(pos, 1.0)))
        loss = F.binary_cross_entropy_with_logits(
            logits, m, pos_weight=torch.tensor(weight, device=device)) + soft_dice_loss(logits, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            model.eval()
            dice = positive_dice(model, vdata, device)
            history.append({"step": step, "loss": float(loss), "val_positive_dice_I10": dice})
            print(history[-1], flush=True)
            if dice > best:
                best, best_state = dice, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, args.out / "detector_pilot.pt")

    # Pixel threshold on clean foreground of detector_val at I10.
    probs10 = predict(model, [d["I10"] for d in vdata], device)
    neg = []
    for d, p in zip(vdata, probs10):
        clean = d["I10"] > 20
        if d["mask"] is not None:
            clean &= ~d["mask"]
        vals = p[clean]
        neg.append(vals[rng.choice(len(vals), size=min(20000, len(vals)), replace=False)] if len(vals) else vals)
    tau = float(np.quantile(np.concatenate(neg), 1 - PIXEL_FPR, method="higher"))
    print(f"pixel threshold tau = {tau:.4f}", flush=True)

    transforms = {
        "I10_raw": lambda d: d["I10"],
        "I07_raw": lambda d: d["I07"],
        "I07_gain": lambda d: np.clip(d["I07"].astype(np.float32) * gain, 0, 255).astype(np.uint8),
        "I07_clahe": lambda d: (equalize_adapthist(d["I07"], clip_limit=0.01) * 255).astype(np.uint8),
    }
    results = {}
    per_view = defaultdict(dict)
    for name, fn in transforms.items():
        imgs = [fn(d) for d in vdata]
        probs = probs10 if name == "I10_raw" else predict(model, imgs, device)
        loss_inc, act_inc, recalls = [], [], []
        by_cls = defaultdict(lambda: [0, 0])
        det_loss = []
        for (d, v), p in zip(val, probs):
            call = p >= tau
            clean = d["I10"] > 20
            positive = bool(d["mask"] is not None and d["mask"].any())
            if positive:
                clean &= ~d["mask"]
                rec = float((call & d["mask"]).sum() / d["mask"].sum())
                recalls.append(rec)
                li = bool(rec < R0)
                loss_inc.append(li)
                by_cls[v["cls"]][0] += li
                by_cls[v["cls"]][1] += 1
                if v["detectable"].get("07") is True and v["detectable"].get("10") is True:
                    det_loss.append(li)
            fpr = float(call[clean].mean()) if clean.any() else 0.0
            act_inc.append(bool(fpr > F0))
            per_view[f"{v['cls']}/{v['kind']}/{v['obj']}/{v['view']}"][name] = {
                "positive": positive, "recall": recalls[-1] if positive else None, "clean_fpr": fpr}
        results[name] = {
            "positive_views": len(loss_inc),
            "evidence_loss_rate": float(np.mean(loss_inc)),
            "evidence_loss_rate_detectable_both": float(np.mean(det_loss)) if det_loss else None,
            "median_recall": float(np.median(recalls)),
            "excess_activation_rate": float(np.mean(act_inc)),
            "all_views": len(act_inc),
            "loss_rate_by_category": {c: round(a / b, 3) for c, (a, b) in sorted(by_cls.items())},
        }
        print(name, json.dumps(results[name]), flush=True)
    summary = {"categories": args.categories, "split_file": str(args.split), "seed": args.seed,
               "train_views": len(train), "val_views": len(val), "gain": gain, "tau": tau,
               "best_val_positive_dice_I10": best, "history": history, "results": results}
    (args.out / "pilot_results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (args.out / "pilot_per_view.json").write_text(json.dumps(per_view), encoding="utf-8")


if __name__ == "__main__":
    main()
