from __future__ import annotations

import pytest

from release_tools.backend_bundle import (
    BundleEnvironmentError,
    edition_for,
    pyinstaller_excludes,
    validate_installed_distributions,
)


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
    assert {"torch", "lightglue", "src.aliked_lightglue_matcher", "src.local_sift_matcher", "faiss", "visualdl"} <= excluded


def test_validation_requires_exact_versions_and_python():
    with pytest.raises(BundleEnvironmentError, match="paddleclas"):
        validate_installed_distributions(edition_for("gpu"), {**_packages(), "paddleclas": "2.5.0"}, python_version=(3, 10), is_64bit=True)
    with pytest.raises(BundleEnvironmentError, match="Python 3.10"):
        validate_installed_distributions(edition_for("gpu"), _packages(), python_version=(3, 11), is_64bit=True)

