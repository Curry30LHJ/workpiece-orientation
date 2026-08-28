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

## Built bundle audit and smoke evidence

The existing GPU onedir output was audited in place (no release artifact was added to Git): 3,287 files totaling 4,393,004,346 bytes (~4.09 GiB). Size is dominated by 2.94 GiB of NVIDIA CUDA libraries and 1.15 GiB of Paddle libraries. The CPU build completed successfully with 3,266 files totaling 649,934,304 bytes (~0.605 GiB) and contains no NVIDIA directory or CUDA DLLs.

Archive and filesystem scans found no Torch, LightGlue, faiss, sklearn, visualdl, soft-center matcher, ALIKED/local matcher, or benchmark runtime modules. The only name matches were Paddle compatibility header files under `paddle/include/.../compat/torch`; these are headers, not imported runtime code. GPU cuDNN train DLLs are shipped by the Paddle CUDA runtime alongside inference DLLs.

An external-model/data smoke was attempted against both EXEs using the real model directory and inference YAML, temporary writable data roots, structured hello/loading polling, and shutdown. Both editions reached `loading_model` then failed deterministically with `MODEL_LOAD_FAILED: No module named 'sklearn'`: PaddleClas 2.6.0 unconditionally imports `sklearn.metrics` during `RecPredictor` construction, while the required `sklearn` exclusion removes it from the frozen archive. No prediction or shutdown handshake could proceed after this startup failure; generated smoke data/logs remain under ignored `release_staging` only. This dependency/spec conflict requires a follow-up decision before claiming offline smoke success.

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

## Fix round 3 — PyInstaller spec project-root fallback

The first real GPU PyInstaller invocation reached spec evaluation and failed before analysis with `NameError: name '__file__' is not defined` at the project-root default expression (the build environment variable was set, but Python evaluated the default argument eagerly). The regression test was added first and failed on the old `Path(__file__)` expression. The spec now reads `WORKPIECE_PROJECT_ROOT` first and only falls back to `Path.cwd().resolve()`, so spec execution never references an undefined `__file__` while retaining the shared GPU/CPU spec and external model/data contract.

GREEN:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_backend_bundle.py tests/test_backend_spec.py tests/test_packaging_env_script.py -q -p no:cacheprovider
......                                                                   [100%]
6 passed in 0.08s
python -m py_compile release_tools/backend_bundle.py  # exit 0
git diff --check                                      # clean
```

## Environment/build attempt

The authorized `create_packaging_envs.ps1` attempt created the Python 3.10.20 Conda environments. After the initial Paddle download interruption, both locked Paddle distributions became available in their respective environments and the CPU PyInstaller build completed. Subsequent onedir audit and smoke results (including the startup dependency failure) are recorded below; no successful offline smoke is claimed.

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
