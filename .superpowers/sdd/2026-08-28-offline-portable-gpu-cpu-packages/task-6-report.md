# Task 6 — Freeze Minimal GPU and CPU Backend Editions

## Implemented

- Added `release_tools.backend_bundle` with immutable `BundleEdition` metadata for GPU (Paddle GPU 3.2.2, CUDA 11.8 namespaces) and CPU (Paddle 3.2.2).
- Added strict environment validation: CPython 3.10, 64-bit, exact Paddle/PaddleClas/PyInstaller versions, and rejection of the opposite Paddle distribution.
- Added the fast-geometry PyInstaller exclusion list (Torch, LightGlue, local matchers, faiss, sklearn, visualdl, tests/training/benchmark modules).
- Added a shared PyInstaller onedir spec that freezes only `orientation_tcp_service.py` and runtime dependencies, with `console=False`; model/YAML/data remain external.
- Added pinned common/GPU/CPU requirements files.
- Added PowerShell scripts for creating the two packaging environments and building one edition into its owned `release_staging/backend-{edition}` directory. Build script verifies `orientation_backend.exe` and only clears its own work/dist paths.

## TDD evidence

RED (before implementation, backend module temporarily absent):

```text
python -m pytest tests/test_backend_bundle.py -q -p no:cacheprovider
ModuleNotFoundError: No module named 'release_tools.backend_bundle'
1 error in 0.34s
```

GREEN:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_backend_bundle.py -q -p no:cacheprovider
....                                                                     [100%]
4 passed in 0.03s
```

Additional checks:

```text
python -m py_compile release_tools/backend_bundle.py  # exit 0
PowerShell parser: scripts/create_packaging_envs.ps1  # parsed
PowerShell parser: scripts/build_portable_backend.ps1 # parsed
git diff --check                                      # clean
```

## Fix round 2 — packaging environment robustness

RED: a static script test failed because `create_packaging_envs.ps1` did not clear a globally inherited `PIP_NO_INDEX` value and had no supported-version guard for an already existing environment.

GREEN:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_packaging_env_script.py tests/test_backend_bundle.py -q -p no:cacheprovider
.....                                                                    [100%]
5 passed in 0.05s
PowerShell parser: scripts/create_packaging_envs.ps1  # parsed
git diff --check                                      # clean
```

The script now sets `$env:PIP_NO_INDEX = ''` before any child process, preserves the official cu118/cpu indexes, and rejects an existing environment unless its reported interpreter is Python 3.10.x.

## Environment/build attempt

The authorized `create_packaging_envs.ps1` attempt created the Python 3.10.20 Conda environments and installed the common dependencies into the GPU environment. The Paddle GPU download/install did not complete (the process was interrupted after prolonged network resolution/download); CPU dependency installation therefore did not start. Neither `paddlepaddle-gpu` nor `paddlepaddle` metadata is present in the respective environments after interruption. Consequently no PyInstaller build or EXE smoke test was run, and no build result is claimed.

## Files

- `release_tools/__init__.py`
- `release_tools/backend_bundle.py`
- `tests/test_backend_bundle.py`
- `deploy/orientation_backend.spec`
- `deploy/requirements/common.txt`
- `deploy/requirements/gpu.txt`
- `deploy/requirements/cpu.txt`
- `scripts/create_packaging_envs.ps1`
- `scripts/build_portable_backend.ps1`

## Self-review / concerns

- Real GPU/CPU freezing and smoke remain pending because the locked Paddle wheels were not available within the authorized build attempt. The build scripts are intentionally fail-fast when an edition environment is incomplete.
- `paddleclas==2.6.0` brings optional development packages (for example faiss/sklearn/visualdl); the spec excludes these modules from the frozen backend as required.

## Fix round 1 — exclusion coverage

Review identified missing exclusions for the soft-center matcher and two existing repository benchmark modules. The test was extended first and produced the expected RED failure (`src.soft_center_matcher`, `scripts.benchmark_adaptive_local_search`, and `scripts.benchmark_geometry_rule_inference` absent from the exclusion set). Added those exact modules plus safe wildcard forms (`src.soft_center_matcher.*`, `scripts.benchmark_*`).

GREEN:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_backend_bundle.py -q -p no:cacheprovider
....                                                                     [100%]
4 passed in 0.03s
python -m py_compile release_tools/backend_bundle.py  # exit 0
git diff --check                                      # clean
```
