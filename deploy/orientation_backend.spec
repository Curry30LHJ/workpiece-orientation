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

# Keep NumPy/Paddle extension loading functional when the package is extracted
# beneath a long Windows path. This is consumed by the frozen backend EXE.
LONG_PATH_MANIFEST = """
<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\" manifestVersion=\"1.0\">
  <application>
    <windowsSettings>
      <ws2:longPathAware xmlns:ws2=\"http://schemas.microsoft.com/SMI/2016/WindowsSettings\">true</ws2:longPathAware>
    </windowsSettings>
  </application>
</assembly>
"""

# Paddle is collected as binaries/data only; its Python implementation and
# the project modules are frozen into the PYZ archive, not delivered as loose
# source files.  OpenCV's wheel is the one intentional exception: its
# ``cv2/__init__.py`` loader is required beside the extension and is audited as
# a third-party dependency loader by ``portable_package``.
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
    manifest=LONG_PATH_MANIFEST,
)
coll = COLLECT(exe, a.binaries, a.datas, name="orientation_backend")
