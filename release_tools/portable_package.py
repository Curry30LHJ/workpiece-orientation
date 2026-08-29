"""Build, audit, and archive self-contained offline editions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import zipfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from src.model_fingerprint import model_directory_sha256


class PackageAuditError(RuntimeError):
    pass


_RUNTIME_DEV_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".obj", ".exp",
    ".lib", ".pdb", ".ilk", ".map", ".pyi", ".res",
}
_TEXT_AUDIT_SUFFIXES = {
    ".cfg", ".conf", ".ini", ".json", ".md", ".ps1", ".py", ".pyi",
    ".toml", ".txt", ".xml", ".yaml", ".yml",
}
_DEPENDENCY_LOADER_PREFIXES = ("backend/_internal/cv2/",)
_DEVELOPMENT_DIR_NAMES = {"test", "tests", "report", "reports", "fixture", "fixtures", "manual", "manuals"}
_MSVC_RUNTIME_REQUIRED = ("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")
_MSVC_RUNTIME_OPTIONAL = (
    "MSVCP140_1.dll", "MSVCP140_2.dll", "MSVCP140_ATOMIC_WAIT.dll",
    "MSVCP140_CODECvt_IDS.dll", "CONCRT140.dll", "VCCORLIB140.dll",
)
_MSVC_RUNTIME_ALL = _MSVC_RUNTIME_REQUIRED + _MSVC_RUNTIME_OPTIONAL


def _is_dependency_loader(rel: str) -> bool:
    """Return whether a visible Python file is a required third-party loader.

    The OpenCV wheel ships ``cv2/__init__.py`` and companion loader files as
    data.  They are not project source and removing them breaks ``import cv2``
    in the frozen service, so the audit has this explicit, narrow exception.
    Project modules and all other visible Python source remain forbidden.
    """

    return any(rel.startswith(prefix) for prefix in _DEPENDENCY_LOADER_PREFIXES)


def _should_copy_runtime_file(relative: str, *, backend: bool = False) -> bool:
    """Keep only files needed by an onedir runtime bundle."""

    path = Path(relative)
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in _RUNTIME_DEV_SUFFIXES or suffix == ".rc":
        return False
    if name.startswith(("makefile", "moc_", "ui_")):
        return False
    if suffix == ".py":
        return backend and _is_dependency_loader("backend/" + relative.lower())
    return True


def _is_reparse_point(path: Path) -> bool:
    """Detect symlinks and Windows junction/reparse points without resolving them."""

    path = Path(path)
    try:
        if path.is_symlink():
            return True
    except OSError:
        # An inaccessible entry must not be treated as a safe regular file.
        return True
    if os.name != "nt":
        return False
    try:
        import ctypes

        attributes = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attributes in {-1, 0xFFFFFFFF}:
            return False
        return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except (AttributeError, OSError):
        return False


def _assert_no_reparse_components(path: Path, floor: Path, *, label: str) -> None:
    """Reject a symlink/junction in any lexical component below ``floor``."""

    path = Path(path)
    floor = Path(floor)
    try:
        relative = path.relative_to(floor)
    except ValueError as exc:
        raise PackageAuditError(f"{label} is outside its allowed root: {path}") from exc
    current = floor
    for component in relative.parts:
        current = current / component
        if _is_reparse_point(current):
            raise PackageAuditError(f"{label} contains a reparse point: {current}")


def _iter_tree_entries(root: Path) -> Iterable[Path]:
    """Walk a tree without traversing symlinks/junctions."""

    root = Path(root)
    # Check the root before ``is_dir``: a dangling junction/symlink reports
    # false from ``is_dir`` and would otherwise make the walk silently empty.
    if _is_reparse_point(root):
        raise PackageAuditError(f"runtime tree contains a symlink/reparse point: {root}")
    if not root.is_dir():
        return
    _assert_no_reparse_components(root, root.parent, label="runtime tree")
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.sort()
        files.sort()
        for name in directories:
            item = current_path / name
            if _is_reparse_point(item):
                raise PackageAuditError(f"runtime tree contains a symlink/reparse point: {item.relative_to(root).as_posix()}")
            yield item
        for name in files:
            item = current_path / name
            if _is_reparse_point(item):
                raise PackageAuditError(f"runtime tree contains a symlink/reparse point: {item.relative_to(root).as_posix()}")
            yield item


def _iter_tree_files(root: Path) -> Iterable[Path]:
    for item in _iter_tree_entries(Path(root)):
        if item.is_file():
            yield item


def _looks_like_x64_runtime_dir(path: Path) -> bool:
    """Reject explicit x86/SysWOW64 runtime directories."""

    normalized = str(path).replace("\\", "/").lower().rstrip("/")
    parts = set(normalized.split("/"))
    if "syswow64" in parts or "x86" in parts or "hostx86" in parts:
        return False
    # System32 is the x64 system directory on a 64-bit Windows host.  Known
    # Visual Studio redist/tool paths are constrained to x64 by their glob.
    return True


def _runtime_dir_files(directory: Path) -> dict[str, Path]:
    """Return exact-name MSVC runtime files from one candidate directory."""

    directory = Path(directory)
    if not directory.is_dir() or not _looks_like_x64_runtime_dir(directory):
        return {}
    files = {}
    try:
        for item in directory.iterdir():
            if item.is_file() and not _is_reparse_point(item):
                files[item.name.lower()] = item
    except OSError:
        return {}
    return {name: files[name.lower()] for name in _MSVC_RUNTIME_ALL if name.lower() in files}


def _append_runtime_dir(candidates: list[Path], seen: set[str], value: object) -> None:
    if value is None or not str(value).strip():
        return
    try:
        path = Path(value)
        if path.is_file():
            path = path.parent
        key = str(path).replace("\\", "/").rstrip("/").lower()
        if key in seen or not path.is_dir() or not _looks_like_x64_runtime_dir(path):
            return
        if _is_reparse_point(path):
            raise PackageAuditError(f"MSVC runtime directory is a reparse point: {path}")
    except OSError:
        return
    seen.add(key)
    candidates.append(path)


def _msvc_runtime_candidates(*, explicit: Path | None = None,
                             source_roots: Iterable[Path] = ()) -> list[Path]:
    """Find likely x64 MSVC redist/toolchain directories in preference order."""

    candidates: list[Path] = []
    seen: set[str] = set()

    if explicit is not None:
        _append_runtime_dir(candidates, seen, explicit)

    # windeployqt may already have copied the compiler runtime into the Qt
    # release directory.  A frozen backend can also contain a coherent copy.
    for source in source_roots:
        source = Path(source)
        _append_runtime_dir(candidates, seen, source)
        try:
            for item in _iter_tree_files(source):
                if item.name.lower() in {name.lower() for name in _MSVC_RUNTIME_REQUIRED}:
                    _append_runtime_dir(candidates, seen, item.parent)
        except PackageAuditError:
            raise
        except OSError:
            continue

    env_values = (
        os.environ.get("VCToolsRedistDir"), os.environ.get("VCToolsInstallDir"),
        os.environ.get("VCINSTALLDIR"), os.environ.get("VSINSTALLDIR"),
    )
    for value in env_values:
        if not value:
            continue
        root = Path(value)
        _append_runtime_dir(candidates, seen, root)
        for pattern in (
            "*/x64/Microsoft.VC*.CRT", "*/Microsoft.VC*.CRT",
            "*/bin/Hostx64/x64", "*/bin/HostX64/x64",
        ):
            try:
                for item in sorted(root.glob(pattern), reverse=True):
                    _append_runtime_dir(candidates, seen, item)
            except OSError:
                continue

    # Standard VS 2017/2019/2022 layouts.  ProgramFiles(x86) is checked first
    # because VS is commonly installed there on this Windows toolchain.
    program_roots = [
        os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
    ]
    for base in program_roots:
        if not base:
            continue
        for year in ("2022", "2019", "2017"):
            root = Path(base) / "Microsoft Visual Studio" / year
            for pattern in (
                "*/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT",
                "*/VC/Tools/MSVC/*/bin/Hostx64/x64",
                "*/VC/Tools/MSVC/*/bin/HostX64/x64",
            ):
                try:
                    for item in sorted(root.glob(pattern), reverse=True):
                        _append_runtime_dir(candidates, seen, item)
                except OSError:
                    continue

    # Last-resort x64 system runtime.  This is intentionally after toolchain
    # candidates so a build never silently mixes an older system copy when a
    # matching redistributable is available.
    _append_runtime_dir(
        candidates, seen,
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
    )
    return candidates


def locate_msvc_runtime_dlls(*, explicit_dir: Path | None = None,
                             source_roots: Iterable[Path] = ()) -> dict[str, Path]:
    """Locate one coherent x64 MSVC runtime set for the Qt executable.

    The three DLLs listed in ``_MSVC_RUNTIME_REQUIRED`` are mandatory.  Any
    compatible side-by-side runtime DLLs found in the same directory are also
    returned so dependencies such as ``MSVCP140_1.dll`` remain self-contained.
    """

    required = {name.lower() for name in _MSVC_RUNTIME_REQUIRED}
    if explicit_dir is not None:
        explicit_path = Path(explicit_dir)
        files = _runtime_dir_files(explicit_path)
        if not required.issubset({name.lower() for name in files}):
            missing = ", ".join(
                name for name in _MSVC_RUNTIME_REQUIRED if name.lower() not in {key.lower() for key in files}
            )
            raise PackageAuditError(
                f"explicit MSVC runtime directory is missing: {missing}: {explicit_path}"
            )
        return files
    candidates = _msvc_runtime_candidates(source_roots=source_roots)
    searched: list[str] = []
    for directory in candidates:
        files = _runtime_dir_files(directory)
        searched.append(str(directory))
        if required.issubset({name.lower() for name in files}):
            return files
    missing = ", ".join(_MSVC_RUNTIME_REQUIRED)
    locations = "; ".join(searched[:12]) or "no candidate directories"
    raise PackageAuditError(
        f"MSVC runtime DLLs are unavailable (required: {missing}); searched: {locations}. "
        "Install the x64 Visual C++ toolchain/redist or pass --msvc-runtime-dir."
    )


def _copy_runtime_tree(src: Path, dst: Path, *, backend: bool = False) -> None:
    """Copy a tree without shipping build sources or reparse points."""

    src = Path(src)
    dst = Path(dst)
    _assert_no_reparse_components(src, src.parent, label="source bundle")
    if _is_reparse_point(dst):
        raise PackageAuditError(f"destination bundle is a reparse point: {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in _iter_tree_entries(src):
        relative = item.relative_to(src).as_posix()
        target = dst / relative
        if item.is_dir():
            _assert_no_reparse_components(target, dst.parent, label="destination bundle")
            target.mkdir(parents=True, exist_ok=True)
            continue
        if item.is_file() and _should_copy_runtime_file(relative, backend=backend):
            target.parent.mkdir(parents=True, exist_ok=True)
            _assert_no_reparse_components(target.parent, dst.parent, label="destination bundle")
            shutil.copy2(item, target)


def _is_system_dependency(name: str, system_roots: Iterable[Path] = ()) -> bool:
    base = Path(name).name.lower()
    if base.startswith(("api-ms-", "ext-ms-")): return True
    roots = [Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32", Path(os.environ.get("SystemRoot", r"C:\Windows")) / "SysWOW64", *map(Path, system_roots)]
    # MSVC runtime DLLs must be shipped beside the Qt executable.  Even when
    # the build host has them in System32, treating them as system dependencies
    # would let a release pass audit and then fail on an offline clean machine.
    if base.startswith(("msvcp", "vcruntime", "concrt", "vccorlib")):
        return False
    if any((r / base).is_file() for r in roots): return True
    return base in {"kernel32.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll", "combase.dll", "rpcrt4.dll", "imm32.dll", "version.dll", "ucrtbase.dll"}


@dataclass(frozen=True)
class PackageLayout:
    root: Path
    edition: str
    version: str

    @property
    def backend(self) -> Path:
        return self.root / "backend"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def data(self) -> Path:
        return self.root / "data"


def build_release_config(*, edition: str, version: str, model_sha256: str) -> dict[str, object]:
    edition = edition.lower()
    if edition not in {"gpu", "cpu"}:
        raise ValueError("edition must be gpu or cpu")
    return {
        "launch_mode": "packaged_executable", "backend_executable": "backend/orientation_backend.exe",
        "project_root": ".", "paddle_config": "backend/resources/inference_general.yaml",
        "model_dir": "models/shitu_rec", "data_root": "data", "model_sha256": model_sha256,
        "compute_device": edition, "edition": edition, "package_version": version,
        "local_search_mode": "adaptive", "inference_mode": "fast_geometry", "host": "127.0.0.1",
        "port": 37651, "startup_timeout_ms": 30000 if edition == "gpu" else 60000,
        "request_timeout_ms": 120000,
    }


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def stage_package(*, edition: str, version: str, qt_release_dir: Path, backend_dir: Path,
                  model_dir: Path, output_root: Path, repository_root: Path | None = None,
                  guide: Path | None = None, notices: Path | None = None, git_commit: str = "unknown",
                  paddle_config: Path | None = None,
                  msvc_runtime_dir: Path | None = None) -> PackageLayout:
    edition = edition.lower()
    if edition not in {"gpu", "cpu"} or not re.fullmatch(r"\d+\.\d+\.\d+", version): raise ValueError("invalid edition or version")
    for source in (qt_release_dir, backend_dir, model_dir):
        if not Path(source).is_dir(): raise NotADirectoryError(source)
    qt_release_dir, backend_dir, model_dir = map(Path, (qt_release_dir, backend_dir, model_dir))
    _assert_no_reparse_components(model_dir, model_dir.parent, label="model bundle")
    # Resolve this before creating the destination so an incomplete package is
    # never mistaken for a successful stage when windeployqt omitted VC DLLs.
    msvc_runtime = locate_msvc_runtime_dlls(
        explicit_dir=Path(msvc_runtime_dir) if msvc_runtime_dir is not None else None,
        source_roots=(qt_release_dir, backend_dir),
    )
    root = Path(output_root) / f"{edition}-{version}" / f"WorkpieceOrientation-{edition.upper()}"
    if root.exists():
        out = Path(output_root).resolve(); target = root.resolve()
        if out not in target.parents: raise PackageAuditError("staging path escapes output root")
        if _is_reparse_point(root): raise PackageAuditError("refusing to remove reparse-point staging path")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _copy_runtime_tree(qt_release_dir, root)
    # Keep VC runtime DLLs at the package root, beside WorkpieceOrientation.exe
    # (and duplicate only optional side-by-side files that were present in the
    # same coherent source directory).
    for name, source in msvc_runtime.items():
        target = root / name
        if _is_reparse_point(source):
            raise PackageAuditError(f"MSVC runtime source is a reparse point: {source}")
        shutil.copy2(source, target)
    # The package config is generated below; never carry a stale build copy.
    generated_config = root / "app_config.json"
    if generated_config.exists():
        generated_config.unlink()
    (root / "backend").mkdir(exist_ok=True)
    _copy_runtime_tree(backend_dir, root / "backend", backend=True)
    model_target = root / "models" / "shitu_rec"
    model_target.mkdir(parents=True)
    required = {"inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info"}
    for item in model_dir.iterdir():
        if item.name in required:
            shutil.copy2(item, model_target / item.name)
    missing = required - {p.name for p in model_target.iterdir()}
    if missing:
        raise FileNotFoundError(f"model files missing: {sorted(missing)}")
    resources = root / "backend" / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    config_candidates: list[Path] = []
    if paddle_config is not None:
        config_candidates.append(Path(paddle_config))
    config_candidates.extend([
        resources / "inference_general.yaml",
        backend_dir / "resources" / "inference_general.yaml",
    ])
    if repository_root is not None:
        repo = Path(repository_root)
        config_candidates.extend([
            repo / "deploy" / "configs" / "inference_general.yaml",
            repo / "deploy" / "inference_general.yaml",
            repo / "third_party" / "PaddleClas" / "deploy" / "configs" / "inference_general.yaml",
        ])
    config_source = next((candidate for candidate in config_candidates if candidate.is_file()), None)
    if config_source is None:
        discovered = sorted(backend_dir.rglob("inference_general.yaml"))
        config_source = discovered[0] if discovered else None
    if config_source is None:
        raise FileNotFoundError("inference_general.yaml is required for the portable backend")
    if config_source.is_symlink():
        raise PackageAuditError("inference_general.yaml must not be a symlink")
    config_target = resources / "inference_general.yaml"
    if config_source.resolve() != config_target.resolve():
        shutil.copy2(config_source, config_target)
    (root / "qt.conf").write_text("[Paths]\nPlugins=platforms\nLibraries=.\n", encoding="utf-8")
    data = root / "data"
    for name in ("workpieces", "rules", "cache", "logs", "temp"):
        (data / name).mkdir(parents=True, exist_ok=True)
    (data / "data_layout.json").write_text(json.dumps({"layout_version": 1}, indent=2), encoding="utf-8")
    licenses = root / "third_party_licenses"
    licenses.mkdir()
    (licenses / "index.txt").write_text("Third-party license texts are listed in THIRD_PARTY-NOTICES.txt.\n", encoding="utf-8")
    if not guide or not Path(guide).is_file(): raise FileNotFoundError("offline guide is required")
    if not notices or not Path(notices).is_file(): raise FileNotFoundError("third-party notices are required")
    shutil.copy2(guide, root / "使用说明.txt"); shutil.copy2(notices, root / "THIRD_PARTY-NOTICES.txt")
    model_sha = model_directory_sha256(model_target)
    (root / "app_config.json").write_text(json.dumps(build_release_config(edition=edition, version=version, model_sha256=model_sha), indent=2), encoding="utf-8")
    metadata = {"version": version, "edition": edition, "git_commit": git_commit, "build_utc": datetime.now(timezone.utc).isoformat(), "python": "3.10", "paddle": "3.2.2", "paddleclas": "2.6.0", "pyinstaller": "6.22.2", "qt": "5.14.2", "model_sha256": model_sha}
    (root / "version.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PackageLayout(root, edition, version)


def run_dumpbin(executable: Path, package_root: Path) -> list[str]:
    dumpbin = shutil.which("dumpbin")
    if not dumpbin:
        candidates = sorted(Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019").glob("*/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"))
        if candidates:
            dumpbin = str(candidates[-1])
    if not dumpbin:
        raise PackageAuditError("dumpbin is required to audit executable dependencies (install VS C++ tools)")
    try:
        headers = subprocess.run([dumpbin, "/HEADERS", str(executable)], capture_output=True, text=True, check=False)
        result = subprocess.run([dumpbin, "/DEPENDENTS", str(executable)], capture_output=True, text=True, check=False)
    except OSError:
        raise PackageAuditError(f"unable to execute dumpbin: {dumpbin}")
    if headers.returncode != 0 or result.returncode != 0:
        raise PackageAuditError(f"dumpbin failed for {executable.name}")
    machine = re.search(r"\b([0-9a-f]{3,4})\s+machine\b", headers.stdout, flags=re.IGNORECASE)
    if not machine: raise PackageAuditError(f"unable to determine machine type for {executable.name}")
    if machine.group(1).lower() not in {"8664"}:
        raise PackageAuditError(f"{executable.name} is not an x64 executable")
    return re.findall(r"^\s*([A-Za-z0-9_.-]+\.dll)\s*$", result.stdout, flags=re.MULTILINE | re.IGNORECASE)


def audit_package(root: Path, *, edition: str, version: str, forbidden_roots: Iterable[Path] = (), runtime_roots: Iterable[Path] = (), dependency_checker: Callable[[Path, Path], list[str]] = run_dumpbin) -> dict[str, object]:
    root = Path(root); edition = edition.lower(); errors: list[str] = []
    required = ["WorkpieceOrientation.exe", "backend/orientation_backend.exe", "backend/resources/inference_general.yaml", "app_config.json", "version.json", "models/shitu_rec/inference.pdmodel", "models/shitu_rec/inference.pdiparams", "models/shitu_rec/inference.pdiparams.info", "data/data_layout.json", "data/workpieces", "third_party_licenses/index.txt", "THIRD_PARTY-NOTICES.txt", "使用说明.txt", *_MSVC_RUNTIME_REQUIRED]
    for rel in required:
        if not (root / rel).exists():
            if rel in _MSVC_RUNTIME_REQUIRED:
                errors.append(f"MSVC runtime missing: {rel}")
            else:
                errors.append(f"missing required file: {rel}")
    cfg = {}
    ver = {}
    try: cfg = json.loads((root / "app_config.json").read_text(encoding="utf-8")); ver = json.loads((root / "version.json").read_text(encoding="utf-8"))
    except Exception as exc: errors.append(f"invalid metadata: {exc}")
    if cfg.get("edition") != edition or cfg.get("compute_device") != edition or cfg.get("package_version") != version: errors.append("edition/device/version mismatch")
    if ver.get("version") != version or ver.get("edition") != edition or not ver.get("git_commit") or ver.get("git_commit") == "unknown": errors.append("version metadata mismatch")
    if cfg.get("inference_mode") != "fast_geometry" or cfg.get("launch_mode") != "packaged_executable": errors.append("configuration must be packaged fast mode")
    model = root / "models" / "shitu_rec"
    if model.exists():
        digest = model_directory_sha256(model)
        if cfg.get("model_sha256") != digest or ver.get("model_sha256") != digest: errors.append("model fingerprint mismatch")
    wp = root / "data" / "workpieces"
    if wp.exists() and any(wp.iterdir()): errors.append("workpieces library must be empty")
    forbidden = [str(Path(p)).replace("\\", "/").rstrip("/").lower() for p in forbidden_roots if str(p).strip()]
    known_roots = forbidden + [str(Path(p)).replace("\\", "/").rstrip("/").lower() for p in runtime_roots if str(p).strip()]
    for candidate in (os.environ.get("USERPROFILE", ""), str(Path(os.sys.executable).parent)):
        normalized = str(candidate).replace("\\", "/").rstrip("/").lower()
        if normalized and normalized not in {".", "/"}:
            known_roots.append(normalized)
    try:
        package_entries = list(_iter_tree_entries(root))
    except PackageAuditError as exc:
        errors.append(str(exc))
        package_entries = []
    package_files = [entry for entry in package_entries if entry.is_file()]
    for file in package_files:
        if not file.is_file(): continue
        rel = file.relative_to(root).as_posix().lower(); name = file.name.lower()
        if file.suffix.lower() == ".py" and not _is_dependency_loader(rel): errors.append("Python source is not allowed: " + rel)
        if file.suffix.lower() == ".pyi" and not rel.startswith("backend/_internal/"): errors.append("Python source is not allowed: " + rel)
        is_internal_typing_stub = file.suffix.lower() == ".pyi" and rel.startswith("backend/_internal/")
        if (file.suffix.lower() in _RUNTIME_DEV_SUFFIXES and not is_internal_typing_stub) or file.suffix.lower() == ".rc" or any(part in _DEVELOPMENT_DIR_NAMES for part in rel.split("/")): errors.append("development artifact: " + rel)
        if edition == "cpu" and any(tok in name for tok in ("cuda", "cudnn", "cublas", "nvidia")): errors.append("CUDA runtime in CPU package: " + rel)
        data_rel = rel.startswith("data/")
        if data_rel and any(tok in rel.split("/") for tok in ("rules", "cache", "manifest")) and name != "data_layout.json": errors.append("shipped data artifact: " + rel)
        # Binary payloads routinely contain byte sequences that decode to
        # ``${``/``{{`` or drive-letter fragments.  Restrict textual checks to
        # formats where paths and template tokens are meaningful.
        if file.suffix.lower() in _TEXT_AUDIT_SUFFIXES:
            raw = file.read_bytes(); text = raw.decode("utf-8", errors="ignore").replace("\\", "/").lower(); text16 = raw.decode("utf-16", errors="ignore").replace("\\", "/").lower()
            normalized_text = text.replace("//", "/"); normalized_text16 = text16.replace("//", "/")
            if any(token and token in normalized_text for token in known_roots) or any(token and token in normalized_text16 for token in known_roots): errors.append("absolute path found: " + rel)
            if rel in {"app_config.json", "version.json"} and re.search(r"[a-z]:[/\\]", text): errors.append("absolute path found: " + rel)
            if rel in {"app_config.json", "version.json"} and (re.search(r"(^|[\"'])/(?!/)[^\s\"']+", text) or re.search(r"(^|[\"'])//[^\s\"']+", text) or re.search(r"\\\\[^\\/]+\\[^\"']+", raw.decode("utf-8", errors="ignore"))): errors.append("absolute path found: " + rel)
            if "${" in text or "{{" in text: errors.append("unresolved template token: " + rel)
    if edition == "gpu":
        names = {p.name.lower() for p in package_files if p.is_relative_to(root / "backend")}
        if not any("paddle" in n and Path(n).suffix in {".dll", ".pyd"} for n in names): errors.append("Paddle GPU runtime missing")
        for family in ("cudnn", "cublas", "cudart"):
            if not any(family in n for n in names): errors.append(f"NVIDIA runtime family missing: {family}")
    for exe in (root / "WorkpieceOrientation.exe", root / "backend" / "orientation_backend.exe"):
        if exe.exists():
            for dep in dependency_checker(exe, root) or []:
                d = dep.lower()
                if _is_system_dependency(d): continue
                if d not in {p.name.lower() for p in package_files}: errors.append(f"dependency absent from package: {dep}")
    if errors: raise PackageAuditError("; ".join(errors))
    return {"root": str(root), "edition": edition, "version": version, "files": len(package_files), "dependency_source_exceptions": sorted(_DEPENDENCY_LOADER_PREFIXES)}


def write_manifest(root: Path, *, edition: str, version: str) -> dict[str, object]:
    files = []
    for path in sorted(p for p in _iter_tree_files(Path(root)) if path_relative(p, root) != "manifest.json"):
        data = path.read_bytes(); files.append({"path": path.relative_to(root).as_posix(), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"version": version, "edition": edition, "files": files}
    (Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def path_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_licenses(destination: Path, distributions: Iterable[str] = ("pyinstaller", "paddlepaddle", "paddleclas", "numpy", "opencv-python"), paddleclas_license: Path | None = None, python_license: Path | None = None) -> Path:
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True); index = []
    for name in distributions:
        try: dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            raise PackageAuditError(f"required distribution missing: {name}")
        copied = []
        for f in dist.files or []:
            if Path(f).name.upper().startswith(("LICENSE", "COPYING")):
                src = Path(dist.locate_file(f)); target = destination / f"{name}-{src.name}"; shutil.copy2(src, target); copied.append(target.name)
        if copied: index.append(f"{name} {dist.version}: {', '.join(copied)}")
        elif name.lower() in {"paddlepaddle", "paddlepaddle-gpu", "paddleclas", "pyinstaller"}:
            raise PackageAuditError(f"missing license text for {name}")
    if paddleclas_license:
        if not Path(paddleclas_license).is_file(): raise PackageAuditError("PaddleClas license text is missing")
        shutil.copy2(paddleclas_license, destination / "PaddleClas-LICENSE")
    if python_license:
        if not Path(python_license).is_file(): raise PackageAuditError("Python license text is missing")
        shutil.copy2(python_license, destination / "Python-LICENSE"); index.append("Python: Python-LICENSE")
    (destination / "index.txt").write_text("\n".join(index) + "\n", encoding="utf-8"); return destination / "index.txt"


def zip_package(root: Path, archive: Path) -> Path:
    root, archive = Path(root), Path(archive); archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        entries = sorted(_iter_tree_entries(root))
        for p in entries:
            arcname = root.name + "/" + p.relative_to(root).as_posix()
            if p.is_dir():
                z.writestr(arcname.rstrip("/") + "/", "")
            elif p.is_file():
                z.write(p, arcname)
    return archive


def audit_zip_archive(archive: Path, *, edition: str, version: str,
                      forbidden_roots: Iterable[Path] = (),
                      runtime_roots: Iterable[Path] = (),
                      dependency_checker: Callable[[Path, Path], list[str]] = run_dumpbin) -> dict[str, object]:
    """Safely extract a generated archive, audit its actual top-level package.

    Extraction is intentionally performed in a new temporary directory beside
    the archive and is always removed on return.  ZIP entries are checked for
    traversal and symlink metadata before writing, so this verification step
    cannot escape its disposable root even if an archive is replaced.
    """
    archive = Path(archive)
    if not archive.is_file():
        raise PackageAuditError(f"ZIP archive is missing: {archive}")
    parent = archive.parent.resolve()
    temp_root: Path | None = None
    # Avoid tempfile.mkdtemp on Windows: on some managed hosts it creates a
    # directory with an ACL that prevents the current process from creating
    # extracted children.  Explicit mkdir inherits the archive directory's
    # normal permissions and the UUID makes collisions negligible.
    for _ in range(8):
        candidate = parent / f".zip-audit-{uuid.uuid4().hex}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        temp_root = candidate
        break
    if temp_root is None:
        raise PackageAuditError("unable to create a unique ZIP audit directory")
    try:
        with zipfile.ZipFile(archive) as zipped:
            root = temp_root.resolve()
            for info in zipped.infolist():
                name = info.filename.replace("\\", "/")
                destination = (temp_root / name).resolve()
                if destination != root and root not in destination.parents:
                    raise PackageAuditError(f"ZIP entry escapes extraction root: {name}")
                mode = (info.external_attr >> 16) & 0o170000
                if stat.S_ISLNK(mode):
                    raise PackageAuditError(f"ZIP symlink is not allowed: {name}")
                if info.is_dir() or name.endswith("/"):
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        top_entries = list(temp_root.iterdir())
        top_dirs = [item for item in top_entries if item.is_dir()]
        if len(top_dirs) != 1 or len(top_entries) != 1:
            raise PackageAuditError("ZIP must contain exactly one top-level package directory")
        return audit_package(top_dirs[0], edition=edition, version=version,
                             forbidden_roots=forbidden_roots,
                             runtime_roots=runtime_roots,
                             dependency_checker=dependency_checker)
    except zipfile.BadZipFile as exc:
        raise PackageAuditError(f"invalid ZIP archive: {exc}") from exc
    finally:
        shutil.rmtree(temp_root, ignore_errors=False)


def write_sha256(archive: Path) -> Path:
    archive = Path(archive); digest = hashlib.sha256(archive.read_bytes()).hexdigest(); out = archive.with_suffix(archive.suffix + ".sha256"); out.write_text(f"{digest}  {archive.name}\n", encoding="ascii"); return out


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True); stage = sub.add_parser("stage")
    for arg in ("edition", "version", "qt-release-dir", "backend-dir", "model-dir", "output-root"): stage.add_argument("--" + arg, required=True)
    stage.add_argument("--guide", required=True); stage.add_argument("--notices", required=True); stage.add_argument("--git-commit", required=True)
    stage.add_argument("--repository-root"); stage.add_argument("--paddle-config"); stage.add_argument("--msvc-runtime-dir")
    args = parser.parse_args()
    if args.command == "stage":
        result = stage_package(edition=args.edition, version=args.version, qt_release_dir=Path(args.qt_release_dir), backend_dir=Path(args.backend_dir), model_dir=Path(args.model_dir), output_root=Path(args.output_root), repository_root=Path(args.repository_root) if args.repository_root else None, guide=Path(args.guide), notices=Path(args.notices), git_commit=args.git_commit, paddle_config=Path(args.paddle_config) if args.paddle_config else None, msvc_runtime_dir=Path(args.msvc_runtime_dir) if args.msvc_runtime_dir else None); print(result.root)


if __name__ == "__main__": main()

__all__ = ["PackageAuditError", "PackageLayout", "audit_package", "audit_zip_archive", "build_release_config", "collect_licenses", "locate_msvc_runtime_dlls", "run_dumpbin", "stage_package", "write_manifest", "write_sha256", "zip_package"]
