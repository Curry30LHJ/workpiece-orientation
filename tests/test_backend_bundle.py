from __future__ import annotations

from pathlib import Path

import pytest

from release_tools.backend_bundle import (
    BundleEnvironmentError,
    edition_for,
    production_src_modules,
    pyinstaller_excludes,
    validate_installed_distributions,
)


def test_src_is_an_explicit_package():
    assert (Path(__file__).parents[1] / "src" / "__init__.py").is_file()


def test_production_src_module_list_contains_runtime_dependencies():
    from release_tools.backend_bundle import production_src_modules

    modules = set(production_src_modules())
    assert {
        "src.orientation_tcp_service",
        "src.orientation_classifier",
        "src.workpiece_catalog",
        "src.workpiece_library",
        "src.runtime_data",
        "src.fast_orientation",
        "src.fast_geometry",
        "src.fast_ridge",
        "src.geometry_calibration",
        "src.geometry_mask_profiles",
    } <= modules
    assert "src.aliked_lightglue_matcher" not in modules


def test_frozen_backend_archive_must_contain_production_modules(tmp_path: Path):
    archive = tmp_path / "PYZ-00.pyz"
    archive.write_bytes("\n".join(production_src_modules()).encode())

    from release_tools.backend_bundle import assert_frozen_backend_modules

    assert_frozen_backend_modules(tmp_path) is None


def test_frozen_backend_archive_reports_missing_production_modules(tmp_path: Path):
    (tmp_path / "PYZ-00.pyz").write_bytes(b"src.orientation_tcp_service")

    from release_tools.backend_bundle import assert_frozen_backend_modules

    with pytest.raises(RuntimeError, match="missing modules"):
        assert_frozen_backend_modules(tmp_path)


def test_frozen_backend_exact_target_cannot_be_masked_by_sibling_archive(tmp_path: Path):
    from release_tools.backend_bundle import assert_frozen_backend_modules

    target = tmp_path / "orientation_backend.exe"
    target.write_bytes(b"src.orientation_tcp_service")
    (tmp_path / "decoy.pyz").write_bytes("\n".join(production_src_modules()).encode())

    with pytest.raises(RuntimeError, match="orientation_backend.exe.*src.orientation_classifier"):
        assert_frozen_backend_modules(target)


def test_frozen_backend_exact_target_accepts_all_production_modules(tmp_path: Path):
    from release_tools.backend_bundle import assert_frozen_backend_modules

    target = tmp_path / "orientation_backend.exe"
    target.write_bytes("\n".join(production_src_modules()).encode())

    assert_frozen_backend_modules(target) is None


def _packages(edition: str = "gpu") -> dict[str, str]:
    paddle = "paddlepaddle-gpu" if edition == "gpu" else "paddlepaddle"
    return {paddle: "3.2.2", "paddleclas": "2.6.0", "pyinstaller": "6.22.2"}


def test_gpu_edition_is_cuda118_paddle_322():
    edition = edition_for("gpu")
    assert edition.paddle_distribution == "paddlepaddle-gpu"
    assert edition.paddle_version == "3.2.2"
    assert edition.compute_device == "gpu"
    assert "nvidia.cudnn" in edition.cuda_namespaces


def test_cpu_rejects_gpu_paddle_in_same_environment():
    packages = _packages("cpu")
    packages["paddlepaddle-gpu"] = "3.2.2"
    with pytest.raises(BundleEnvironmentError, match="paddlepaddle-gpu"):
        validate_installed_distributions(edition_for("cpu"), packages, python_version=(3, 10), is_64bit=True)


def test_fast_bundle_excludes_local_matcher_stack():
    excluded = set(pyinstaller_excludes())
    assert {
        "torch",
        "lightglue",
        "src.aliked_lightglue_matcher",
        "src.local_sift_matcher",
        "src.soft_center_matcher",
        "faiss",
        "visualdl",
        "scripts.benchmark_adaptive_local_search",
        "scripts.benchmark_geometry_rule_inference",
    } <= excluded


def test_validation_requires_exact_versions_and_python():
    with pytest.raises(BundleEnvironmentError, match="paddleclas"):
        validate_installed_distributions(edition_for("gpu"), {**_packages(), "paddleclas": "2.5.0"}, python_version=(3, 10), is_64bit=True)
    with pytest.raises(BundleEnvironmentError, match="Python 3.10"):
        validate_installed_distributions(edition_for("gpu"), _packages(), python_version=(3, 11), is_64bit=True)
