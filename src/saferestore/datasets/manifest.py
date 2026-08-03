from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    source_url: str
    local_root: Path
    license_id: str
    license_verified: bool
    read_only: bool
    split_scheme: str
    inventory_sha256: str | None = None


def load_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    local_root = Path(payload["local_root"])
    if not local_root.is_absolute():
        local_root = (manifest_path.parent / local_root).resolve()
    return DatasetManifest(
        name=str(payload["name"]),
        source_url=str(payload["source_url"]),
        local_root=local_root,
        license_id=str(payload.get("license_id", "UNVERIFIED")),
        license_verified=bool(payload.get("license_verified", False)),
        read_only=bool(payload.get("read_only", True)),
        split_scheme=str(payload.get("split_scheme", "UNVERIFIED")),
        inventory_sha256=payload.get("inventory_sha256"),
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_manifest(
    manifest: DatasetManifest,
    *,
    protected_roots: Iterable[str | Path] = (),
    require_existing: bool = True,
) -> list[str]:
    issues: list[str] = []
    if not manifest.source_url.startswith(("https://", "http://")):
        issues.append("source_url must be an official HTTP(S) landing page")
    if require_existing and not manifest.local_root.exists():
        issues.append(f"local_root does not exist: {manifest.local_root}")
    if not manifest.license_verified:
        issues.append("dataset license has not been verified")
    if manifest.split_scheme == "UNVERIFIED":
        issues.append("split semantics have not been verified")
    for root in protected_roots:
        if _is_within(manifest.local_root, Path(root)) and not manifest.read_only:
            issues.append(f"protected source must be read-only: {root}")
    return issues
