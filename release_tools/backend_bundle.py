"""Metadata and validation helpers for the frozen backend editions.

The functions in this module deliberately have no Paddle/PyInstaller imports so
they can be used by tests and build scripts before an edition environment is
loaded.
"""

from __future__ import annotations

import platform
import struct
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence


class BundleEnvironmentError(RuntimeError):
    """Raised when a packaging environment does not match an edition."""


@dataclass(frozen=True)
class BundleEdition:
    name: str
    paddle_distribution: str
    paddle_version: str
    compute_device: str
    cuda_namespaces: Sequence[str]


EDITIONS: dict[str, BundleEdition] = {
    "gpu": BundleEdition(
        "gpu",
        "paddlepaddle-gpu",
        "3.2.2",
        "gpu",
        (
            "nvidia.cublas",
            "nvidia.cuda_nvrtc",
            "nvidia.cuda_runtime",
            "nvidia.cudnn",
            "nvidia.cufft",
            "nvidia.curand",
            "nvidia.cusolver",
            "nvidia.cusparse",
        ),
    ),
    "cpu": BundleEdition("cpu", "paddlepaddle", "3.2.2", "cpu", ()),
}


def edition_for(name: str) -> BundleEdition:
    """Return immutable metadata for ``gpu`` or ``cpu``."""

    key = str(name).strip().lower()
    try:
        return EDITIONS[key]
    except KeyError as exc:
        raise ValueError("edition must be 'gpu' or 'cpu'") from exc


def _normalise_version(value: object) -> str:
    # ``importlib.metadata.version`` values are strings; accepting a mapping
    # supplied by tests makes this helper easy to exercise without installing
    # heavyweight runtimes.
    return str(value).strip()


def _installed_distributions() -> dict[str, str]:
    from importlib import metadata

    result: dict[str, str] = {}
    for name in ("paddlepaddle", "paddlepaddle-gpu", "paddleclas", "pyinstaller"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return result


def validate_installed_distributions(
    edition: BundleEdition | str,
    installed: Mapping[str, object] | None = None,
    *,
    python_version: tuple[int, int] | None = None,
    is_64bit: bool | None = None,
    python_implementation: str | None = None,
) -> None:
    """Validate the exact runtime used to freeze one backend.

    ``installed`` maps distribution names to versions.  When omitted, package
    metadata from the current interpreter is queried.  The opposite Paddle
    distribution is rejected even when the requested distribution is present;
    this prevents accidentally producing a CPU package from a GPU environment
    (or vice versa).
    """

    target = edition if isinstance(edition, BundleEdition) else edition_for(edition)
    packages = {
        str(name).lower().replace("_", "-"): _normalise_version(version)
        for name, version in (installed if installed is not None else _installed_distributions()).items()
    }
    pyver = python_version or (sys.version_info.major, sys.version_info.minor)
    bits = is_64bit if is_64bit is not None else struct.calcsize("P") * 8 == 64
    implementation = python_implementation or platform.python_implementation()
    if implementation != "CPython":
        raise BundleEnvironmentError(f"CPython 3.10 64-bit is required, found {implementation}")
    if tuple(pyver) != (3, 10):
        raise BundleEnvironmentError(
            f"Python 3.10 64-bit is required, found Python {pyver[0]}.{pyver[1]}"
        )
    if not bits:
        raise BundleEnvironmentError("a 64-bit Python interpreter is required")

    opposite = "paddlepaddle" if target.paddle_distribution == "paddlepaddle-gpu" else "paddlepaddle-gpu"
    if opposite in packages:
        raise BundleEnvironmentError(
            f"opposite Paddle distribution {opposite} must not be installed for {target.name}"
        )
    expected = {
        target.paddle_distribution: target.paddle_version,
        "paddleclas": "2.6.0",
        "pyinstaller": "6.22.2",
    }
    for name, version in expected.items():
        actual = packages.get(name)
        if actual is None:
            raise BundleEnvironmentError(f"required distribution {name}=={version} is not installed")
        if actual != version:
            raise BundleEnvironmentError(f"{name}=={version} is required, found {actual}")


def pyinstaller_excludes() -> tuple[str, ...]:
    """Modules intentionally omitted from the fast-geometry frozen service."""

    return (
        "torch",
        "torch.*",
        "lightglue",
        "lightglue.*",
        "src.aliked_lightglue_matcher",
        "src.local_sift_matcher",
        "src.soft_center_matcher",
        "src.soft_center_matcher.*",
        "faiss",
        "faiss.*",
        "sklearn",
        "sklearn.*",
        "visualdl",
        "visualdl.*",
        "pytest",
        "tests",
        "training",
        "src.training",
        "src.train",
        "benchmark",
        "src.benchmark",
        "scripts.benchmark_*",
        "scripts.benchmark_adaptive_local_search",
        "scripts.benchmark_geometry_rule_inference",
        "scripts.benchmark_fast_geometry_inference",
        "scripts.benchmark_portable_service",
    )


__all__ = [
    "BundleEdition",
    "BundleEnvironmentError",
    "EDITIONS",
    "edition_for",
    "pyinstaller_excludes",
    "validate_installed_distributions",
]
