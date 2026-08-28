"""PyInstaller onedir specification for the minimal orientation TCP backend."""

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

from release_tools.backend_bundle import edition_for, pyinstaller_excludes, validate_installed_distributions


edition_name = os.environ.get("WORKPIECE_PACKAGE_EDITION", "").strip().lower()
if not edition_name:
    raise SystemExit("WORKPIECE_PACKAGE_EDITION must be gpu or cpu")
edition = edition_for(edition_name)
project_root_env = os.environ.get("WORKPIECE_PROJECT_ROOT", "").strip()
project_root = Path(project_root_env).resolve() if project_root_env else Path.cwd().resolve()
validate_installed_distributions(edition)

paddle_binaries = collect_dynamic_libs("paddle")
paddle_datas = collect_data_files("paddle", include_py_files=False)
cuda_binaries = []
for namespace in edition.cuda_namespaces:
    cuda_binaries += collect_dynamic_libs(namespace)

a = Analysis(
    [str(project_root / "src" / "orientation_tcp_service.py")],
    pathex=[str(project_root)],
    binaries=paddle_binaries + cuda_binaries,
    datas=paddle_datas,
    hiddenimports=[
        "paddle.base.core",
        "paddle.inference",
        "paddleclas.deploy.python.predict_rec",
        "paddleclas.deploy.utils.config",
        "paddleclas.deploy.utils.predictor",
    ],
    excludes=list(pyinstaller_excludes()),
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="orientation_backend",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="orientation_backend")
