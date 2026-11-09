"""WP6 confirmatory study, step 1: train the per-seed detector and learned enhancer.

Reads only train and detector_val specimens.  For one seed it writes, under
confirm/models/seed<N>/:
  detector.pt   four-level U-Net trained on I10 (reference regime); the
                pilot_v2 recipe, checkpoint chosen by positive-view dice on a
                fixed 300-view detector_val subset at I10;
  enhancer.pt   four-level U-Net residual enhancer I07 -> I10, L1 loss, trained
                on train-role pairs, checkpoint chosen by foreground L1 on 60
                detector_val views;
  calib.json    pixel threshold tau (1 - 1e-3 clean-region quantile on
                detector_val at I10), gain and gamma fitted on train, histories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

import common as C  # noqa: E402  (also puts tim_rr/wp6 and src on sys.path)
import pilot_v2 as P
from saferestore.torch_models import soft_dice_loss
from unet4 import UNet4


def sample_pair(d: dict, rng: np.random.Generator):
    h, w = d["I10"].shape[:2]
    if len(d["fg_coords"]):
        cy, cx = d["fg_coords"][rng.integers(len(d["fg_coords"]))]
    else:
        cy, cx = h // 2, w // 2
    y0 = int(np.clip(cy - P.CROP // 2, 0, h - P.CROP))
    x0 = int(np.clip(cx - P.CROP // 2, 0, w - P.CROP))
    sl = (slice(y0, y0 + P.CROP), slice(x0, x0 + P.CROP))
    return d["I07"][sl], d["I10"][sl]


def train_detector(train, vsub, args, rng, device):
    model = UNet4(3, args.base).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.detector_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    best, best_state, history, t0 = -1.0, None, [], time.time()
    for step in range(1, steps + 1):
        model.train()
        xs, ms = [], []
        for _ in range(args.batch):
            d = train[rng.integers(len(train))]
            regime = args.detector_regimes[rng.integers(len(args.detector_regimes))]
            x, m = P.sample_crop({**d, "I10": d[regime]} if regime != "I10" else d, rng)
            if rng.random() < 0.5:
                x, m = x[:, ::-1], m[:, ::-1]
            if rng.random() < 0.5:
                x, m = x[::-1], m[::-1]
            xs.append(np.ascontiguousarray(x))
            ms.append(np.ascontiguousarray(m))
        x = P.to_tensor(xs, device)
        if args.photometric_aug:
            # per-crop global gamma (0.7-1.4) and gain (0.7-1.3) jitter, on the GPU
            g = torch.from_numpy(rng.uniform(0.7, 1.4, args.batch)).float().to(device)[:, None, None, None]
            a = torch.from_numpy(rng.uniform(0.7, 1.3, args.batch)).float().to(device)[:, None, None, None]
            x = (x.clamp(0, 1).pow(g) * a).clamp(0, 1)
        x = x.contiguous(memory_format=torch.channels_last)
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
            dice = float(np.mean([P.positive_dice(model, [{**d, "I10": d[r]} for d in vsub], device)
                                  for r in args.detector_regimes]))
            history.append({"step": step, "loss": float(loss.detach()), "val_positive_dice_I10": dice,
                            "minutes": round((time.time() - t0) / 60, 1)})
            print("detector", history[-1], flush=True)
            if dice > best:
                best, best_state = dice, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best_state, best, history


@torch.no_grad()
def enhance(model, imgs: list[np.ndarray], device) -> list[np.ndarray]:
    out = []
    for i in range(0, len(imgs), 4):
        x = P.to_tensor(imgs[i:i + 4], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = (x + model(x).float()).clamp_(0, 1)
        out.extend((y.permute(0, 2, 3, 1).float().cpu().numpy() * 255 + 0.5).astype(np.uint8))
    return out


def train_enhancer(train, venh, args, rng, device):
    model = UNet4(3, args.base, out_ch=3).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.enhancer_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    best, best_state, history, t0 = math.inf, None, [], time.time()
    for step in range(1, steps + 1):
        model.train()
        xs, ys = [], []
        for _ in range(args.batch):
            a, b = sample_pair(train[rng.integers(len(train))], rng)
            if rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            xs.append(np.ascontiguousarray(a))
            ys.append(np.ascontiguousarray(b))
        x = P.to_tensor(xs, device).contiguous(memory_format=torch.channels_last)
        y = P.to_tensor(ys, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = x + model(x).float()
        loss = F.l1_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            model.eval()
            outs = enhance(model, [d["I07"] for d in venh], device)
            l1 = float(np.mean([np.abs(o[d["fg"]].astype(np.float32) - d["I10"][d["fg"]]).mean() / 255
                                for o, d in zip(outs, venh)]))
            history.append({"step": step, "loss": float(loss.detach()), "val_fg_l1": l1,
                            "minutes": round((time.time() - t0) / 60, 1)})
            print("enhancer", history[-1], flush=True)
            if l1 < best:
                best, best_state = l1, {k: t.detach().clone() for k, t in model.state_dict().items()}
    return best_state, best, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--detector-steps", type=int, default=12000)
    ap.add_argument("--enhancer-steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--detector-regimes", nargs="+", choices=("I10", "I07"), default=["I10"],
                    help="illuminations the detector is trained on; checkpoint dice is averaged over them")
    ap.add_argument("--photometric-aug", action="store_true", help="gain/gamma jitter on detector crops")
    ap.add_argument("--tag", default="", help="suffix of the model folder (development variants)")
    args = ap.parse_args()
    out = C.HERE / "models" / f"seed{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    views = C.build_views(C.DEV_ROLES, purpose="train_models")
    t0 = time.time()
    with ThreadPoolExecutor(16) as pool:
        data = list(pool.map(P.load_view, views))
    train = [d for d, v in zip(data, views) if v["role"] == "train"]
    val = [(d, v) for d, v in zip(data, views) if v["role"] == "detector_val"]
    vdata = [d for d, _ in val]
    vpos = [d for d in vdata if d["mask"] is not None and d["mask"].any()]
    vsub = [vpos[j] for j in rng.choice(len(vpos), size=min(300, len(vpos)), replace=False)]
    venh = [vdata[j] for j in rng.choice(len(vdata), size=60, replace=False)]
    print(f"seed {args.seed}: {len(train)} train / {len(vdata)} val views in {time.time()-t0:.0f}s", flush=True)

    photo = C.fit_photometric([(d["I07"], d["I10"], d["fg"]) for d in train])
    print("photometric", photo, flush=True)

    detector, det_state, det_best, det_hist = train_detector(train, vsub, args, rng, device)
    torch.save(det_state, out / "detector.pt")
    probs10 = P.predict(detector, [d["I10"] for d in vdata], device)
    neg = []
    for d, p in zip(vdata, probs10):
        mask = d["mask"] if d["mask"] is not None else np.zeros(p.shape, bool)
        vals = p[C.clean_region(d["fg"], mask)]
        neg.append(vals[rng.choice(len(vals), size=min(20000, len(vals)), replace=False)] if len(vals) else vals)
    tau = float(np.quantile(np.concatenate(neg), 1 - C.PIXEL_FPR, method="higher"))
    print(f"tau = {tau:.5f}", flush=True)
    del detector
    torch.cuda.empty_cache()

    enh_state, enh_best, enh_hist = train_enhancer(train, venh, args, rng, device)
    torch.save(enh_state, out / "enhancer.pt")

    calib = {"seed": args.seed, "args": vars(args), "tau": tau, "photometric": photo,
             "detector_best_val_positive_dice_I10": det_best, "detector_history": det_hist,
             "enhancer_best_val_fg_l1": enh_best, "enhancer_history": enh_hist,
             "train_views": len(train), "detector_val_views": len(vdata)}
    (out / "calib.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
