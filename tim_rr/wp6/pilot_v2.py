"""WP6 step 1 (v2): development-only pilot on M2AD.

Question: with a detector trained on the reference regime (I10), does the
poor regime (I07) produce materially more detector evidence-loss incidents
(recall < r0) than I10 on held-aside development specimens, and do two fixed
application-time enhancements change that?

v2 changes from pilot_regime_gap.py (stopped at step 1000, positive dice
0.005): RGB input, four-level U-Net (unet4.py), precomputed foreground
coordinates for crop sampling, cosine learning rate, fixed 300-view
validation subset for checkpointing.

Reads ONLY specimens whose frozen role (splits/m2ad_rr_v1.json) is train or
detector_val; asserts this for every file opened. Writes only under the
--out folder inside tim_rr/wp6/.

Endpoints (same definitions as the paper): r0 = 0.25 recall floor on
mask >= 128; activation = clean-foreground FPR > 0.002, clean foreground =
(I10 grey > 20) minus the mask; detector pixel threshold = 1 - 1e-3
quantile of clean-foreground probabilities on detector_val at I10.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.exposure import equalize_adapthist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from saferestore.torch_models import soft_dice_loss  # noqa: E402
from unet4 import UNet4  # noqa: E402

ALLOWED = {"train", "detector_val"}
R0, F0, PIXEL_FPR = 0.25, 0.002, 1e-3
CROP = 384


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


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
    i10, i07 = rgb(v["I10"]), rgb(v["I07"])
    fg = i10.mean(axis=2) > 20
    coords = np.argwhere(fg)
    rng = np.random.default_rng(zlib.crc32(f"{v['cls']}/{v['obj']}/{v['view']}".encode()))
    if len(coords) > 4000:
        coords = coords[rng.choice(len(coords), 4000, replace=False)]
    mask = (np.asarray(Image.open(v["mask"]).convert("L")) >= 128) if v["mask"] is not None else None
    mcoords = np.argwhere(mask) if mask is not None and mask.any() else None
    return {"I10": i10, "I07": i07, "fg": fg, "fg_coords": coords, "mask": mask, "mask_coords": mcoords}


def sample_crop(d: dict, rng: np.random.Generator):
    img, mask = d["I10"], d["mask"]
    h, w = img.shape[:2]
    if d["mask_coords"] is not None and rng.random() < 0.5:
        cy, cx = d["mask_coords"][rng.integers(len(d["mask_coords"]))]
        cy += rng.integers(-CROP // 3, CROP // 3)
        cx += rng.integers(-CROP // 3, CROP // 3)
    elif len(d["fg_coords"]):
        cy, cx = d["fg_coords"][rng.integers(len(d["fg_coords"]))]
    else:
        cy, cx = h // 2, w // 2
    y0 = int(np.clip(cy - CROP // 2, 0, h - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, w - CROP))
    m = mask[y0:y0 + CROP, x0:x0 + CROP] if mask is not None else np.zeros((CROP, CROP), bool)
    return img[y0:y0 + CROP, x0:x0 + CROP], m


def to_tensor(imgs: list[np.ndarray], device) -> torch.Tensor:
    return torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).float().div_(255.0)


@torch.no_grad()
def predict(model, imgs: list[np.ndarray], device) -> list[np.ndarray]:
    out = []
    for i in range(0, len(imgs), 4):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(to_tensor(imgs[i:i + 4], device)).float())
        out.extend(p[:, 0].cpu().numpy())
    return out


def positive_dice(model, data: list[dict], device) -> float:
    probs = predict(model, [d["I10"] for d in data], device)
    scores = []
    for d, p in zip(data, probs):
        call = p >= 0.5
        inter = (call & d["mask"]).sum()
        scores.append(2 * inter / (call.sum() + d["mask"].sum() + 1e-9))
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=HERE / "splits" / "m2ad_rr_v1.json")
    ap.add_argument("--out", type=Path, default=HERE / "pilot_v2")
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=301)
    args = ap.parse_args()
    assert args.out.resolve().is_relative_to(HERE), "outputs must stay inside tim_rr/wp6"
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
    vdata = [d for d, _ in val]
    vpos = [d for d in vdata if d["mask"] is not None and d["mask"].any()]
    vsub = [vpos[j] for j in rng.choice(len(vpos), size=min(300, len(vpos)), replace=False)]
    print(f"loaded {len(train)} train / {len(val)} val views ({len(vpos)} positive) in {time.time()-t0:.0f}s",
          flush=True)

    gain = float(np.mean([d["I10"][d["fg"]].mean() for d in train if d["fg"].any()]) /
                 np.mean([d["I07"][d["fg"]].mean() for d in train if d["fg"].any()]))
    print(f"train-role I10/I07 foreground mean ratio (gain) = {gain:.3f}", flush=True)

    model = UNet4(3, args.base).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    best, best_state, history = -1.0, None, []
    t1 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        xs, ms = [], []
        for _ in range(args.batch):
            x, m = sample_crop(train[rng.integers(len(train))], rng)
            if rng.random() < 0.5:
                x, m = x[:, ::-1], m[:, ::-1]
            if rng.random() < 0.5:
                x, m = x[::-1], m[::-1]
            xs.append(np.ascontiguousarray(x))
            ms.append(np.ascontiguousarray(m))
        x = to_tensor(xs, device).contiguous(memory_format=torch.channels_last)
        m = torch.from_numpy(np.stack(ms)).to(device)[:, None].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        pos = float(m.sum())
        weight = min(80.0, max(1.0, (m.numel() - pos) / max(pos, 1.0)))
        loss = F.binary_cross_entropy_with_logits(
            logits, m, pos_weight=torch.tensor(weight, device=device)) + soft_dice_loss(logits, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            model.eval()
            dice = positive_dice(model, vsub, device)
            history.append({"step": step, "loss": float(loss.detach()), "val_positive_dice_I10": dice,
                            "minutes": round((time.time() - t1) / 60, 1)})
            print(history[-1], flush=True)
            if dice > best:
                best, best_state = dice, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, args.out / "detector_pilot.pt")

    probs10 = predict(model, [d["I10"] for d in vdata], device)
    neg = []
    for d, p in zip(vdata, probs10):
        clean = d["fg"].copy()
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
        probs = probs10 if name == "I10_raw" else predict(model, [fn(d) for d in vdata], device)
        loss_inc, act_inc, recalls, det_loss = [], [], [], []
        by_cls = defaultdict(lambda: [0, 0])
        for (d, v), p in zip(val, probs):
            call = p >= tau
            clean = d["fg"].copy()
            positive = bool(d["mask"] is not None and d["mask"].any())
            rec = None
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
                "positive": positive, "recall": rec, "clean_fpr": fpr}
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
    summary = {"version": "pilot_v2", "categories": args.categories, "split_file": str(args.split),
               "seed": args.seed, "steps": args.steps, "train_views": len(train), "val_views": len(val),
               "gain": gain, "tau": tau, "best_val_positive_dice_I10_subset": best,
               "history": history, "results": results}
    (args.out / "pilot_results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (args.out / "pilot_per_view.json").write_text(json.dumps(per_view), encoding="utf-8")


if __name__ == "__main__":
    main()
