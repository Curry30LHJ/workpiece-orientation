import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from release_tools.portable_package import (
    PackageAuditError, audit_package, build_release_config,
    write_manifest, write_sha256, zip_package,
)
from src.model_fingerprint import model_directory_sha256


def minimal_stage(root: Path, edition: str = "gpu") -> Path:
    root.mkdir()
    (root / "backend").mkdir()
    (root / "backend" / "orientation_backend.exe").write_bytes(b"MZ-backend")
    if edition == "gpu":
        for name in ("paddle_inference.dll", "cudnn64_8.dll", "cublas64_11.dll", "cudart64_110.dll"):
            (root / "backend" / name).write_bytes(b"MZ-runtime")
    (root / "backend" / "resources").mkdir()
    (root / "backend" / "resources" / "inference_general.yaml").write_text("Global: {}\n")
    (root / "models" / "shitu_rec").mkdir(parents=True)
    (root / "models" / "shitu_rec" / "inference.pdmodel").write_bytes(b"model")
    (root / "models" / "shitu_rec" / "inference.pdiparams").write_bytes(b"params")
    (root / "models" / "shitu_rec" / "inference.pdiparams.info").write_bytes(b"info")
    (root / "data" / "workpieces").mkdir(parents=True)
    for name in ("rules", "cache", "logs", "temp"):
        (root / "data" / name).mkdir()
    (root / "data" / "data_layout.json").write_text(
        json.dumps({"layout_version": 1}), encoding="utf-8"
    )
    (root / "WorkpieceOrientation.exe").write_bytes(b"MZ-qt")
    (root / "Qt5Core.dll").write_bytes(b"MZ-qtcore")
    (root / "platforms").mkdir()
    (root / "platforms" / "qwindows.dll").write_bytes(b"MZ-platform")
    (root / "qt.conf").write_text("[Paths]\nPlugins=.\n", encoding="utf-8")
    (root / "third_party_licenses").mkdir()
    (root / "third_party_licenses" / "index.txt").write_text("licenses", encoding="utf-8")
    model_sha = model_directory_sha256(root / "models" / "shitu_rec")
    config = build_release_config(
        edition=edition, version="1.0.0", model_sha256=model_sha
    )
    (root / "app_config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "version.json").write_text(json.dumps({
        "version": "1.0.0", "edition": edition,
        "git_commit": "0" * 40, "build_utc": "2026-08-28T12:00:00Z",
        "python": "3.10.20", "paddle": "3.2.2", "paddleclas": "2.6.0",
        "pyinstaller": "6.22.2", "qt": "5.14.2",
        "model_sha256": model_sha,
    }), encoding="utf-8")
    (root / "THIRD_PARTY-NOTICES.txt").write_text("notices", encoding="utf-8")
    (root / "使用说明.txt").write_text("离线使用说明", encoding="utf-8")
    return root


def audit_synthetic(root: Path, edition: str = "gpu", forbidden_roots=()):
    return audit_package(
        root, edition=edition, version="1.0.0",
        forbidden_roots=list(forbidden_roots),
        dependency_checker=lambda executable, package_root: [],
    )


def test_generated_configs_are_relative_fast_and_device_specific():
    model_sha = "a" * 64
    gpu = build_release_config(edition="gpu", version="1.0.0", model_sha256=model_sha)
    cpu = build_release_config(edition="cpu", version="1.0.0", model_sha256=model_sha)
    assert gpu["backend_executable"] == "backend/orientation_backend.exe"
    assert gpu["compute_device"] == gpu["edition"] == "gpu"
    assert cpu["compute_device"] == cpu["edition"] == "cpu"
    assert gpu["inference_mode"] == cpu["inference_mode"] == "fast_geometry"
    assert gpu["model_sha256"] == cpu["model_sha256"] == model_sha
    assert gpu["startup_timeout_ms"] == 30000
    assert cpu["startup_timeout_ms"] == 60000
    assert all("E:/" not in str(value) for value in gpu.values())


@pytest.mark.parametrize("bad_name", ["leak.py", "stub.pyi"])
def test_audit_rejects_visible_python_source(tmp_path: Path, bad_name: str):
    root = minimal_stage(tmp_path / "package")
    (root / "backend" / bad_name).write_text("secret", encoding="utf-8")
    with pytest.raises(PackageAuditError, match="Python source"):
        audit_synthetic(root)


def test_audit_rejects_nonempty_workpiece_library(tmp_path: Path):
    root = minimal_stage(tmp_path / "package")
    (root / "data" / "workpieces" / "existing-piece").mkdir()
    with pytest.raises(PackageAuditError, match="workpieces"):
        audit_synthetic(root)


def test_audit_rejects_development_absolute_path(tmp_path: Path):
    root = minimal_stage(tmp_path / "package")
    (root / "bad.json").write_text(r'{"path":"E:\\Project\\wang\\pp_813"}')
    with pytest.raises(PackageAuditError, match="absolute path"):
        audit_synthetic(root, forbidden_roots=[Path(r"E:\Project\wang\pp_813")])


def test_cpu_audit_rejects_cuda_runtime(tmp_path: Path):
    root = minimal_stage(tmp_path / "package", edition="cpu")
    (root / "backend" / "cudnn64_8.dll").write_bytes(b"MZ")
    with pytest.raises(PackageAuditError, match="CUDA"):
        audit_synthetic(root, edition="cpu")


def test_manifest_and_zip_use_one_versioned_root(tmp_path: Path):
    root = minimal_stage(tmp_path / "WorkpieceOrientation-GPU")
    manifest = write_manifest(root, edition="gpu", version="1.0.0")
    assert manifest["version"] == "1.0.0"
    assert all("sha256" in item for item in manifest["files"])
    archive = zip_package(root, tmp_path / "WorkpieceOrientation-GPU-x64-1.0.0.zip")
    assert archive.is_file()
    with ZipFile(archive) as zipped:
        assert {Path(name).parts[0] for name in zipped.namelist()} == {
            "WorkpieceOrientation-GPU"
        }
    checksum = write_sha256(archive)
    digest, filename = checksum.read_text(encoding="ascii").strip().split("  ", 1)
    assert len(digest) == 64
    assert filename == archive.name
