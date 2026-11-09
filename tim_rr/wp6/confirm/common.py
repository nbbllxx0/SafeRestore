"""Shared definitions for the M2AD confirmatory study (WP6).

Everything that defines a record lives here: the unit rule (one view per
specimen), the six candidate actions, the endpoint definitions, and the nine
application-time ranking features.  The freeze manifest hashes this file, so
a change after the freeze is detectable.

Data roles come from the frozen specimen split splits/m2ad_rr_v1.json.
Roles "tuning" and "holdout" are opened only through open_roles(), which
refuses unless the protocol freeze manifest verifies.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
WP6 = HERE.parent
REPO = WP6.parents[1]
sys.path.insert(0, str(WP6))
sys.path.insert(0, str(REPO / "src"))

SPLIT_FILE = WP6 / "splits" / "m2ad_rr_v1.json"
SPLIT_SHA256 = "1b9670248da4f3bce9f3612e3d3da8bbe668355133d9823839f90674fcf127f3"
DATA_ROOT = Path("<path to your M2AD directory>")
META_FILE = DATA_ROOT / "jsons" / "meta_unsupervised.json"
CATEGORIES = ("Bird", "Car", "Cube", "Dice", "Doll", "Holder", "Motor", "Ring", "Teapot", "Tube")

DEV_ROLES = {"train", "detector_val"}
SEALED_ROLES = {"tuning", "holdout"}
MANIFEST = HERE / "FREEZE_MANIFEST.json"
ACCESS_LOG = HERE / "sealed_role_access_log.txt"

# Endpoints (identical definitions to the two-domain study).
R0 = 0.25            # evidence loss: recall < R0 on a defect-positive unit
F0 = 0.002           # excess activation: clean-foreground FPR > F0
PIXEL_FPR = 1e-3     # detector pixel threshold: 1 - PIXEL_FPR clean quantile at I10
TOLERANCE = 8        # annotation-boundary ring (px) excluded from the clean region
FG_LEVEL = 20        # object foreground: mean RGB of the I10 reference > FG_LEVEL

METHODS = ("raw", "gain", "gamma", "clahe", "gain_unsharp", "learned")
RESTORATION_METHODS = METHODS[1:]
POLICIES = METHODS + ("restoration_pool", "action_pool")
FEATURES = (
    "entropy",
    "detector_disagreement",
    "evidence_shift",
    "restoration_residual",
    "high_probability_area",
    "peak_probability",
    "peak_disagreement",
    "secondary_peak",
    "active_components",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_roles() -> dict[str, str]:
    digest = sha256_file(SPLIT_FILE)
    if digest != SPLIT_SHA256:
        raise RuntimeError(f"split file hash {digest} != frozen {SPLIT_SHA256}")
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["roles"]


def verify_freeze() -> dict:
    """Return the manifest if every hashed file matches; raise otherwise."""

    if not MANIFEST.exists():
        raise PermissionError("no FREEZE_MANIFEST.json: protocol not frozen, sealed roles stay closed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise PermissionError("manifest status is not FROZEN")
    for rel, digest in manifest["files"].items():
        actual = sha256_file(HERE / rel)
        if actual != digest:
            raise PermissionError(f"{rel} changed after freeze ({actual} != {digest})")
    return manifest


def open_roles(roles_wanted: set[str], purpose: str) -> None:
    """Gate for sealed roles: verify the freeze and append an access record."""

    sealed = roles_wanted & SEALED_ROLES
    unknown = roles_wanted - SEALED_ROLES - DEV_ROLES
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}")
    if not sealed:
        return
    manifest = verify_freeze()
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with ACCESS_LOG.open("a", encoding="utf-8") as log:
        log.write(f"{stamp}\troles={','.join(sorted(sealed))}\tpurpose={purpose}\t"
                  f"manifest_sha256={sha256_file(MANIFEST)}\tfrozen_at={manifest['frozen_at']}\n")


def view_key(cls: str, kind: str, obj: str, view: str) -> str:
    return f"{cls}/{kind}/{obj}/{view}"


def build_views(roles_wanted: set[str], *, purpose: str) -> list[dict]:
    """All (specimen, view) entries of the wanted roles with I07, I10 and mask paths."""

    open_roles(roles_wanted, purpose)
    roles = load_roles()
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    views: dict[tuple, dict] = {}
    for split in ("train", "test"):
        for cls in CATEGORIES:
            for it in meta[split].get(cls, []):
                kind = it["img_path"].split("/")[1]
                specimen = f"{cls}/{kind}/{it['object_name']}"
                role = roles[specimen]
                if role not in roles_wanted:
                    continue
                key = (cls, kind, it["object_name"], it["view"])
                v = views.setdefault(key, {
                    "cls": cls, "kind": kind, "obj": it["object_name"], "view": it["view"],
                    "specimen": specimen, "role": role, "mask": None, "anomalous": False,
                    "detectable": {},
                })
                if it["illumination"] in ("07", "10"):
                    v[f"I{it['illumination']}"] = DATA_ROOT / it["img_path"]
                    v["detectable"][it["illumination"]] = it["detectable"]
                if it["image_anomaly"]:
                    v["anomalous"] = True
                    v["mask"] = DATA_ROOT / it["mask_path"]
    out = sorted((v for v in views.values() if "I07" in v and "I10" in v),
                 key=lambda v: view_key(v["cls"], v["kind"], v["obj"], v["view"]))
    for v in out:
        assert v["role"] in roles_wanted
        v["sample_id"] = view_key(v["cls"], v["kind"], v["obj"], v["view"])
    return out


def select_units(views: list[dict]) -> list[dict]:
    """One view per specimen (the independent unit).

    Good specimens: any of their views.  NG specimens: one of their views
    flagged image_anomaly.  The choice is the minimum of
    sha256("m2ad-rr-v1-view:<specimen>/<view>") over the candidates, so it is
    fixed by identifiers alone.
    """

    by_specimen: dict[str, list[dict]] = {}
    for v in views:
        if v["kind"] == "NG" and not v["anomalous"]:
            continue
        by_specimen.setdefault(v["specimen"], []).append(v)
    units = []
    for specimen in sorted(by_specimen):
        cands = by_specimen[specimen]
        units.append(min(cands, key=lambda v: hashlib.sha256(
            f"m2ad-rr-v1-view:{v['sample_id']}".encode()).hexdigest()))
    return units


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(v: dict, shape: tuple[int, int]) -> np.ndarray:
    if v["mask"] is None:
        return np.zeros(shape, dtype=bool)
    return np.asarray(Image.open(v["mask"]).convert("L")) >= 128


def foreground(i10: np.ndarray) -> np.ndarray:
    return i10.mean(axis=2) > FG_LEVEL


def clean_region(fg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Foreground minus the mask dilated by the TOLERANCE ring (square ring, max filter)."""

    if not mask.any():
        return fg.copy()
    from scipy import ndimage

    ring = ndimage.maximum_filter(mask.astype(np.uint8), size=2 * TOLERANCE + 1).astype(bool)
    return fg & ~ring


# ----------------------------------------------------------------- actions
def fit_photometric(train_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """Gain and gamma fitted on train-role (I07, I10, fg) triples only."""

    m07 = float(np.mean([a[fg].mean() for a, _, fg in train_pairs if fg.any()])) / 255.0
    m10 = float(np.mean([b[fg].mean() for _, b, fg in train_pairs if fg.any()])) / 255.0
    return {"gain": m10 / m07, "gamma": float(np.log(m10) / np.log(m07)), "m07": m07, "m10": m10}


def apply_classical(name: str, i07: np.ndarray, photo: dict) -> np.ndarray:
    x = i07.astype(np.float32) / 255.0
    if name == "raw":
        return i07
    if name == "gain":
        y = x * photo["gain"]
    elif name == "gamma":
        y = np.power(x, photo["gamma"])
    elif name == "clahe":
        from skimage.exposure import equalize_adapthist

        y = equalize_adapthist(i07, clip_limit=0.01)
    elif name == "gain_unsharp":
        from scipy.ndimage import gaussian_filter

        # Unsharp mask written out (radius = Gaussian sigma 2, amount 1): scikit-image 0.26
        # unsharp_mask returned uninitialised values in this environment.
        g = np.clip(x * photo["gain"], 0, 1).astype(np.float64)
        y = g + 1.0 * (g - gaussian_filter(g, sigma=(2, 2, 0)))
    else:
        raise ValueError(name)
    return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- features
def structure_features(prob: np.ndarray, threshold: float, margin: int = TOLERANCE) -> tuple[float, float]:
    """Secondary peak outside the largest call component, and call-component count."""

    from scipy import ndimage

    call = prob >= threshold
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(call, structure=structure)
    if count == 0:
        return 0.0, 0.0
    sizes = ndimage.sum(call, labels, index=np.arange(1, count + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)
    keep_out = ndimage.binary_dilation(largest, structure=structure, iterations=margin)
    outside = prob[~keep_out]
    return (float(outside.max()) if outside.size else 0.0), float(count)


def top_fraction_mean(values: np.ndarray, fraction: float = 0.001) -> float:
    flat = values.ravel()
    k = max(1, int(round(flat.size * fraction)))
    return float(np.partition(flat, flat.size - k)[flat.size - k:].mean())


def binary_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def record_row(v: dict, method: str, cand: np.ndarray, i07: np.ndarray, prob: np.ndarray,
               prob_raw: np.ndarray, tau: float, mask: np.ndarray, clean: np.ndarray,
               f0: float = F0) -> dict:
    """Endpoints use the mask and the I10-derived clean region; features use only the
    candidate image, the acquired I07 image, and detector outputs (application-time)."""

    call = prob >= tau
    positive = bool(mask.any())
    recall = float((call & mask).sum() / mask.sum()) if positive else 1.0
    clean_fpr = float(call[clean].mean()) if clean.any() else 0.0
    diff = np.abs(prob - prob_raw)
    secondary, components = structure_features(prob, tau)
    return {
        "sample_id": v["sample_id"],
        "specimen": v["specimen"],
        "category": v["cls"],
        "role": v["role"],
        "method": method,
        "has_defect": int(positive),
        "pixel_recall": recall,
        "clean_fpr": clean_fpr,
        "deletion_error": int(positive and recall < R0),
        "invention_error": int(clean_fpr > f0),
        "detectable_07": v["detectable"].get("07"),
        "entropy": float(binary_entropy(prob).mean()),
        "detector_disagreement": float(diff.mean()),
        "evidence_shift": float(abs(prob.mean() - prob_raw.mean())),
        "restoration_residual": float(np.abs(cand.astype(np.float32) - i07.astype(np.float32)).mean() / 255.0),
        "high_probability_area": float((prob >= 0.5).mean()),
        "peak_probability": top_fraction_mean(prob),
        "peak_disagreement": top_fraction_mean(diff),
        "secondary_peak": secondary,
        "active_components": components,
    }
