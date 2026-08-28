"""Dependency-free model directory fingerprinting."""

from __future__ import annotations

import hashlib
from pathlib import Path


def model_directory_sha256(path: Path) -> str:
    """Return a deterministic SHA-256 digest for model file paths and contents."""
    digest = hashlib.sha256()
    root = Path(path)
    for item in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
