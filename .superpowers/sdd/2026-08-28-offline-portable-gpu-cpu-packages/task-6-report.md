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

## Historical first bundle audit and smoke attempt (resolved)

The existing GPU onedir output was audited in place (no release artifact was added to Git): 3,287 files totaling 4,393,004,346 bytes (~4.09 GiB). Size is dominated by 2.94 GiB of NVIDIA CUDA libraries and 1.15 GiB of Paddle libraries. The CPU build completed successfully with 3,266 files totaling 649,934,304 bytes (~0.605 GiB) and contains no NVIDIA directory or CUDA DLLs.

Archive and filesystem scans found no Torch, LightGlue, faiss, sklearn, visualdl, soft-center matcher, ALIKED/local matcher, or benchmark runtime modules. The only name matches were Paddle compatibility header files under `paddle/include/.../compat/torch`; these are headers, not imported runtime code. GPU cuDNN train DLLs are shipped by the Paddle CUDA runtime alongside inference DLLs.

An initial external-model/data smoke attempt against both EXEs reached `loading_model` and failed with `MODEL_LOAD_FAILED: No module named 'sklearn'`. This was a historical failure caused by PaddleClas' optional imports conflicting with the required `sklearn` exclusion; the import-only compatibility layer in fix round 4 resolved it. A subsequent attempt then exposed the legacy two-file model format (`inference.pdmodel` + `inference.pdiparams` without `inference.json`), which fix round 5 resolved. These failures are retained as diagnostic history only; they do not describe the final bundle status.

## Fix round 4 — import-only PaddleClas compatibility

To preserve the required `sklearn`/`faiss` PyInstaller exclusions without changing inference semantics, `src.paddleclas_inference_compat.py` now registers six failing sklearn metric/preprocessing callables and an empty faiss module only when those optional packages are unavailable. `OrientationClassifier.load` invokes the helper immediately before importing PaddleClas. A regression test removes both modules and blocks optional imports, verifies the stubs permit import and are cleaned up afterward, and confirms the exclusion metadata remains unchanged.

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

The authorized `create_packaging_envs.ps1` attempt created the Python 3.10.20 Conda environments. After an initial Paddle download interruption, both locked Paddle distributions became available and both editions were rebuilt successfully after the compatibility fixes. The historical download/startup failures above are superseded by the final fix-round-5 build and smoke evidence below.

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

- Real GPU/CPU freezing and smoke initially failed during dependency/model-format compatibility work; fix round 5 completed both builds and the end-to-end smoke. The build scripts remain intentionally fail-fast when an edition environment is incomplete.
- `paddleclas==2.6.0` brings optional development packages (for example faiss/sklearn/visualdl); the spec excludes these modules from the frozen backend as required.

The faiss placeholder now raises an explicit `RuntimeError` on any attribute access; the compatibility regression covers this contract.

## Final result — fix round 5 legacy two-file model loading

The supplied model is the legacy `inference.pdmodel` + `inference.pdiparams(.info)` format. `create_rec_predictor` now temporarily advertises Paddle 2.5 to PaddleClas so it selects `Config(model_file, params_file)`, and temporarily disables `ir_optim` only for that construction (restoring all values afterward). CPU MKLDNN remains enabled for the packaged configuration. Focused regression tests cover format detection, temporary version/config flags, restoration, and optional import stubs.

Both editions were rebuilt successfully after this change. Final onedir sizes are GPU 4,393,004,263 bytes (3,287 files) and CPU 649,936,161 bytes (3,266 files). The reproducible smoke runner is `release_staging/smoke_backend.py`; it was invoked as `E:\python\anaconda3\envs\shitu\python.exe release_staging/smoke_backend.py cpu <free-port>` and the equivalent `gpu` command. It uses the external model/YAML/data paths documented in the script and writes logs under `release_staging/smoke-data-cpu/logs/orientation-service.log` and `release_staging/smoke-data-gpu/logs/orientation-service.log`.

Both smoke runs passed: hello progressed through `loading_model` to `ready`, registration accepted front/back templates, predictions returned `front` and `back`, shutdown returned `ok: true`, and each process exited with code `0`. Temporary smoke data remained under `release_staging`; the shipped package data and model directory were not modified. `release_staging/` and `deploy/pyinstaller-work/` are intentionally untracked local build outputs and were not added to any commit.

## Historical fix round 4 build and smoke results (superseded)

Both editions were rebuilt from commit `9067c48` with Python 3.10.20/PyInstaller 6.22.2. The compatibility regression passed, but smoke then stopped at `loading_model` because Paddle 3.2.2 looked for `inference.json` in the supplied legacy model directory. This is the historical model-format failure fixed in round 5; the final round-5 smoke above is the authoritative result.

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
