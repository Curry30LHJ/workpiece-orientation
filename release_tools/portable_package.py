"""Build, audit, and archive self-contained offline editions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from src.model_fingerprint import model_directory_sha256


class PackageAuditError(RuntimeError):
    pass


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
                  guide: Path | None = None, notices: Path | None = None, git_commit: str = "unknown") -> PackageLayout:
    edition = edition.lower()
    if edition not in {"gpu", "cpu"} or not re.fullmatch(r"\d+\.\d+\.\d+", version): raise ValueError("invalid edition or version")
    for source in (qt_release_dir, backend_dir, model_dir):
        if not Path(source).is_dir(): raise NotADirectoryError(source)
    root = Path(output_root) / f"{edition}-{version}" / f"WorkpieceOrientation-{edition.upper()}"
    if root.exists():
        out = Path(output_root).resolve(); target = root.resolve()
        if out not in target.parents: raise PackageAuditError("staging path escapes output root")
        if root.is_symlink(): raise PackageAuditError("refusing to remove reparse-point staging path")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    qt_release_dir, backend_dir, model_dir = map(Path, (qt_release_dir, backend_dir, model_dir))
    for item in qt_release_dir.iterdir():
        if item.name.lower() == "app_config.json":
            continue
        target = root / item.name
        shutil.copytree(item, target, dirs_exist_ok=True) if item.is_dir() else shutil.copy2(item, target)
    (root / "backend").mkdir(exist_ok=True)
    for item in backend_dir.iterdir():
        target = root / "backend" / item.name
        shutil.copytree(item, target, dirs_exist_ok=True) if item.is_dir() else shutil.copy2(item, target)
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
    if not (resources / "inference_general.yaml").exists():
        candidates = list(backend_dir.rglob("inference_general.yaml"))
        if candidates: shutil.copy2(candidates[0], resources / "inference_general.yaml")
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
    try:
        headers = subprocess.run(["dumpbin", "/HEADERS", str(executable)], capture_output=True, text=True, check=False)
        result = subprocess.run(["dumpbin", "/DEPENDENTS", str(executable)], capture_output=True, text=True, check=False)
    except OSError:
        raise PackageAuditError("dumpbin is required to audit executable dependencies")
    if headers.returncode != 0 or result.returncode != 0:
        raise PackageAuditError(f"dumpbin failed for {executable.name}")
    machine = re.search(r"\b([0-9a-f]{3,4})\s+machine\b", headers.stdout, flags=re.IGNORECASE)
    if not machine: raise PackageAuditError(f"unable to determine machine type for {executable.name}")
    if machine.group(1).lower() not in {"8664"}:
        raise PackageAuditError(f"{executable.name} is not an x64 executable")
    return re.findall(r"^\s*([A-Za-z0-9_.-]+\.dll)\s*$", result.stdout, flags=re.MULTILINE | re.IGNORECASE)


def audit_package(root: Path, *, edition: str, version: str, forbidden_roots: Iterable[Path] = (), runtime_roots: Iterable[Path] = (), dependency_checker: Callable[[Path, Path], list[str]] = run_dumpbin) -> dict[str, object]:
    root = Path(root); edition = edition.lower(); errors: list[str] = []
    required = ["WorkpieceOrientation.exe", "backend/orientation_backend.exe", "backend/resources/inference_general.yaml", "app_config.json", "version.json", "models/shitu_rec/inference.pdmodel", "models/shitu_rec/inference.pdiparams", "models/shitu_rec/inference.pdiparams.info", "data/data_layout.json", "data/workpieces", "third_party_licenses/index.txt", "THIRD_PARTY-NOTICES.txt", "使用说明.txt"]
    for rel in required:
        if not (root / rel).exists(): errors.append(f"missing required file: {rel}")
    cfg = {}
    ver = {}
    try: cfg = json.loads((root / "app_config.json").read_text(encoding="utf-8")); ver = json.loads((root / "version.json").read_text(encoding="utf-8"))
    except Exception as exc: errors.append(f"invalid metadata: {exc}")
    if cfg.get("edition") != edition or cfg.get("compute_device") != edition or cfg.get("package_version") != version: errors.append("edition/device/version mismatch")
    if cfg.get("inference_mode") != "fast_geometry" or cfg.get("launch_mode") != "packaged_executable": errors.append("configuration must be packaged fast mode")
    model = root / "models" / "shitu_rec"
    if model.exists():
        digest = model_directory_sha256(model)
        if cfg.get("model_sha256") != digest or ver.get("model_sha256") != digest: errors.append("model fingerprint mismatch")
    wp = root / "data" / "workpieces"
    if wp.exists() and any(wp.iterdir()): errors.append("workpieces library must be empty")
    forbidden = [str(Path(p)).replace("\\", "/").rstrip("/").lower() for p in forbidden_roots]
    known_roots = forbidden + [str(Path(p)).replace("\\", "/").rstrip("/").lower() for p in runtime_roots] + [str(Path(os.environ.get("USERPROFILE", ""))).replace("\\", "/").rstrip("/").lower(), str(Path(os.sys.executable).parent).replace("\\", "/").rstrip("/").lower()]
    for file in root.rglob("*"):
        if not file.is_file(): continue
        rel = file.relative_to(root).as_posix().lower(); name = file.name.lower()
        if file.suffix.lower() == ".py" or (file.suffix.lower() == ".pyi" and "/_internal/" not in "/" + rel): errors.append("Python source is not allowed: " + rel)
        if file.suffix.lower() in {".pdb", ".obj"} or any(x in rel for x in ("test", "report", "fixture", "manual")): errors.append("development artifact: " + rel)
        if edition == "cpu" and any(tok in name for tok in ("cuda", "cudnn", "cublas", "nvidia")): errors.append("CUDA runtime in CPU package: " + rel)
        data_rel = rel.startswith("data/")
        if data_rel and any(tok in rel.split("/") for tok in ("rules", "cache", "manifest")) and name != "data_layout.json": errors.append("shipped data artifact: " + rel)
        raw = file.read_bytes(); text = raw.decode("utf-8", errors="ignore").replace("\\", "/").lower(); text16 = raw.decode("utf-16", errors="ignore").replace("\\", "/").lower()
        if any(token and token in text for token in known_roots) or any(token and token in text16 for token in known_roots): errors.append("absolute path found: " + rel)
        if rel in {"app_config.json", "version.json"} and re.search(r"[a-z]:[/\\]", text): errors.append("absolute path found: " + rel)
        if "${" in text or "{{" in text: errors.append("unresolved template token: " + rel)
    if edition == "gpu":
        names = {p.name.lower() for p in (root / "backend").rglob("*") if p.is_file()}
        if not any("paddle" in n and Path(n).suffix in {".dll", ".pyd"} for n in names): errors.append("Paddle GPU runtime missing")
        for family in ("cudnn", "cublas", "cudart"):
            if not any(family in n for n in names): errors.append(f"NVIDIA runtime family missing: {family}")
    for exe in (root / "WorkpieceOrientation.exe", root / "backend" / "orientation_backend.exe"):
        if exe.exists():
            for dep in dependency_checker(exe, root) or []:
                system = {"kernel32.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "ws2_32.dll", "gdi32.dll", "comdlg32.dll", "msvcp140.dll", "vcruntime140.dll"}
                if dep.lower() not in system and dep.lower() not in {p.name.lower() for p in root.rglob("*")}: errors.append(f"dependency absent from package: {dep}")
    if errors: raise PackageAuditError("; ".join(errors))
    return {"root": str(root), "edition": edition, "version": version, "files": len([p for p in root.rglob('*') if p.is_file()])}


def write_manifest(root: Path, *, edition: str, version: str) -> dict[str, object]:
    files = []
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file() and path_relative(p, root) != "manifest.json"):
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
        for p in sorted(x for x in root.rglob("*") if x.is_file()): z.write(p, root.name + "/" + p.relative_to(root).as_posix())
    return archive


def write_sha256(archive: Path) -> Path:
    archive = Path(archive); digest = hashlib.sha256(archive.read_bytes()).hexdigest(); out = archive.with_suffix(archive.suffix + ".sha256"); out.write_text(f"{digest}  {archive.name}\n", encoding="ascii"); return out


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True); stage = sub.add_parser("stage")
    for arg in ("edition", "version", "qt-release-dir", "backend-dir", "model-dir", "output-root"): stage.add_argument("--" + arg, required=True)
    stage.add_argument("--guide", required=True); stage.add_argument("--notices", required=True); stage.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if args.command == "stage":
        result = stage_package(edition=args.edition, version=args.version, qt_release_dir=Path(args.qt_release_dir), backend_dir=Path(args.backend_dir), model_dir=Path(args.model_dir), output_root=Path(args.output_root), guide=Path(args.guide), notices=Path(args.notices), git_commit=args.git_commit); print(result.root)


if __name__ == "__main__": main()

__all__ = ["PackageAuditError", "PackageLayout", "audit_package", "build_release_config", "collect_licenses", "run_dumpbin", "stage_package", "write_manifest", "write_sha256", "zip_package"]
