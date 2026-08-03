from pathlib import Path

from saferestore.datasets.manifest import DatasetManifest, validate_manifest


def test_protected_manifest_must_be_read_only(tmp_path: Path):
    protected = tmp_path / "protected"
    protected.mkdir()
    manifest = DatasetManifest(
        name="example",
        source_url="https://example.org/data",
        local_root=protected,
        license_id="TEST",
        license_verified=True,
        read_only=False,
        split_scheme="official",
    )
    issues = validate_manifest(manifest, protected_roots=[protected])
    assert any("read-only" in issue for issue in issues)
