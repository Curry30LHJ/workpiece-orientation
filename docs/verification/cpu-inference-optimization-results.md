# CPU inference optimization verification

Date: 2026-08-30  
Branch: `feature/20260830/cpu-inference-optimization`  
Code commit: `9ae717509d32711a7a5b067080ebaa613551d666`  
Report commit: `1fb1cf0` (prior report commit; the documentation commit necessarily has a new SHA)

## Method and data

Task 4 required the complete pytest suite, packaged CPU runs at thread candidates 1/2/4, and paired 1,000-iteration runs with `WORKPIECE_CPU_DEDUPLICATE_SLOTS=0` versus the default-enabled path. The acceptance input was checked read-only at `runtime_reports/fast-geometry-acceptance.json` (SHA-256 `45c73611e828b8bbd94650ee316fc970f859a04d2ad1c40a5e609d2b99572eef`, 120 queries across M1/M2/M7). The file is user-provided and was not changed or staged.

The requested canonical release ZIP was absent. To make a safe current-branch measurement, the current backend was built with `E:\python\anaconda3\envs\workpiece-package-cpu\python.exe`, filtered into a fresh copy of the authorized legacy package, and archived as `release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0-cpuopt-benchmark.zip` (SHA-256 `d4b288d1ed5e509a47254a98921c82a16ca7fdefb67eb2ca1e23b218a1d578c7`). The package metadata records code commit `9ae7175`; model fingerprint `1fab156fb025705a836ad6c28590fc32e1141ae5c530282369412a3078300b33` matches `app_config.json` and `version.json`. Strict directory and extracted-ZIP audits passed before execution.

## Commands and results

| Command | Result |
| --- | --- |
| `python -m pytest -q` | Collection blocked by `PermissionError: [WinError 5]` while scanning `qt_app/tests/build-test_geometryrulespage/pytest-task10-profiles`; pytest cache also lacked write access. |
| `python -m pytest tests/test_fast_orientation.py tests/test_orientation_classifier.py tests/test_orientation_tcp_service.py tests/test_workpiece_catalog.py -q` | `103 passed, 177 errors`; errors were chiefly pytest temporary-directory lock permissions under `C:\Users\Administrator\AppData\Local\Temp\pytest-of-Administrator`. No source was changed to work around this environment failure. |
| `scripts/build_portable_backend.ps1 -Edition cpu -Python E:\python\anaconda3\envs\workpiece-package-cpu\python.exe` | Exit 0; PyInstaller 6.22.2 completed and emitted `release_staging/backend-cpu/orientation_backend`. |
| Isolated backend replacement, manifest regeneration, strict package audit, ZIP audit | All passed; filtered package contained 1,171 files and metadata commit `9ae7175`. |
| One-request smoke attempt (`--warmup 0 --iterations 1`) | Exit 2 by benchmark CLI validation (`iterations must be at least 1000`); no backend was started by this attempt. |
| Current package, `WORKPIECE_CPU_DEDUPLICATE_SLOTS=0`, `--cpu-threads 1`, warmup 50/iterations 1000 | Exit 0; 1,000 measured requests, accuracy 1.0, review 0.0. |
| Current package, `WORKPIECE_CPU_DEDUPLICATE_SLOTS=0`, `--cpu-threads 2`, warmup 50/iterations 1000 | Exit 0; 1,000 measured requests, accuracy 1.0, review 0.0. |
| Current package, `WORKPIECE_CPU_DEDUPLICATE_SLOTS=0`, `--cpu-threads 4`, warmup 50/iterations 1000 | Exit 0; 1,000 measured requests, accuracy 1.0, review 0.0. |
| Current package, default dedup, `--cpu-threads 1`, warmup 50/iterations 1000 | Exit 0; 1,000 measured requests, accuracy 1.0, review 0.0. |
| Current package, default dedup, `--cpu-threads 2`, warmup 50/iterations 1000 | Exit 0; 1,000 measured requests, accuracy 1.0, review 0.0. |
| Current package, default dedup, `--cpu-threads 4`, warmup 50/iterations 1000 | Exit 0; 1,000 measured requests, accuracy 1.0, review 0.0. |

Benchmark JSON outputs and the benchmark-only package remain under gitignored `release_staging/` and `release_artifacts/`. Existing `runtime_reports/` files were not overwritten and were not added to git.

The exact benchmark invocations were:

```powershell
$env:WORKPIECE_CPU_DEDUPLICATE_SLOTS='0'; python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0-cpuopt-benchmark.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 1 --warmup 50 --iterations 1000 --timeout-seconds 900 --output release_staging/benchmarks/current-dedup0-threads1.json
$env:WORKPIECE_CPU_DEDUPLICATE_SLOTS='0'; python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0-cpuopt-benchmark.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 2 --warmup 50 --iterations 1000 --timeout-seconds 900 --output release_staging/benchmarks/current-dedup0-threads2.json
$env:WORKPIECE_CPU_DEDUPLICATE_SLOTS='0'; python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0-cpuopt-benchmark.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 4 --warmup 50 --iterations 1000 --timeout-seconds 900 --output release_staging/benchmarks/current-dedup0-threads4.json
Remove-Item Env:WORKPIECE_CPU_DEDUPLICATE_SLOTS -ErrorAction SilentlyContinue; python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0-cpuopt-benchmark.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 1 --warmup 50 --iterations 1000 --timeout-seconds 900 --output release_staging/benchmarks/current-dedup-default-threads1.json
Remove-Item Env:WORKPIECE_CPU_DEDUPLICATE_SLOTS -ErrorAction SilentlyContinue; python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0-cpuopt-benchmark.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 2 --warmup 50 --iterations 1000 --timeout-seconds 900 --output release_staging/benchmarks/current-dedup-default-threads2.json
Remove-Item Env:WORKPIECE_CPU_DEDUPLICATE_SLOTS -ErrorAction SilentlyContinue; python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0-cpuopt-benchmark.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 4 --warmup 50 --iterations 1000 --timeout-seconds 900 --output release_staging/benchmarks/current-dedup-default-threads4.json
```

## Historical reference: legacy CPU package

For supplemental context only, the sibling worktree package was run without copying it into this branch. Its manifest identifies package commit `d3d1fa0834f11964919124359921c2ebc5db27cc` (Paddle 3.2.2/PaddleClas 2.6.0), ZIP SHA-256 `4044820dcabbd1fcc8c517eb7d80b6b6b6a64478b4210640be0536052d51f2db`, and 50 warmup + 1,000 measured requests per run. These are historical measurements from the pre-optimization package and do not represent current code commit `9ae7175`.

| Legacy package CPU threads | startup ready (ms) | backend elapsed mean/P50/P95/P99/max (ms) | round trip mean/P50/P95/P99/max (ms) | accuracy / review |
| --- | ---: | --- | --- | --- |
| 1 | 18,132.10 | 172.52 / 171.25 / 199.49 / 212.32 / 301.42 | 173.24 / 171.97 / 200.21 / 213.79 / 302.29 | 1.0 / 0.0 |
| 2 | 13,896.07 | 162.70 / 161.69 / 179.76 / 192.84 / 223.60 | 163.45 / 162.39 / 180.81 / 193.68 / 224.32 | 1.0 / 0.0 |
| 4 | 14,404.12 | 160.59 / 159.93 / 177.39 / 185.44 / 240.97 | 161.32 / 160.62 / 178.13 / 186.61 / 242.34 | 1.0 / 0.0 |

The legacy package did not expose the new slot-dedup telemetry (`global_unique_slots` or model-call count), and no dedup-disabled/default pair was run against the current code. These results must not be used to claim the optimization target is met.

## Measurements

| Configuration | backend elapsed (mean/P50/P95/P99/max ms) | round trip (mean/P50/P95/P99/max ms) | accuracy | review | unique slots / call proxy | status |
| --- | --- | --- | --- | --- | --- | --- |
| Dedup disabled, thread 1 | 225.61 / 187.65 / 350.62 / 477.55 / 814.32 | 226.51 / 188.37 / 351.79 / 479.08 / 815.87 | 1.0 | 0.0 | 3→3; 3,000 slot-elements | completed current package |
| Dedup disabled, thread 2 | 159.02 / 155.61 / 177.98 / 244.34 / 404.88 | 159.89 / 156.41 / 178.93 / 245.62 / 405.49 | 1.0 | 0.0 | 3→3; 3,000 slot-elements | completed current package |
| Dedup disabled, thread 4 | 145.19 / 142.89 / 162.40 / 190.25 / 259.40 | 145.96 / 143.61 / 163.38 / 191.33 / 261.17 | 1.0 | 0.0 | 3→3; 3,000 slot-elements | completed current package |
| Dedup default enabled, thread 1 | 57.54 / 56.08 / 64.95 / 73.30 / 335.83 | 58.23 / 56.75 / 65.89 / 74.59 / 336.45 | 1.0 | 0.0 | 3→1; 1,000 slot-elements | completed current package |
| Dedup default enabled, thread 2 | 53.71 / 52.76 / 61.48 / 78.46 / 89.00 | 54.46 / 53.48 / 62.35 / 79.45 / 91.23 | 1.0 | 0.0 | 3→1; 1,000 slot-elements | completed current package |
| Dedup default enabled, thread 4 | 52.39 / 50.89 / 64.57 / 77.74 / 102.68 | 53.22 / 51.70 / 65.67 / 79.13 / 104.71 | 1.0 | 0.0 | 3→1; 1,000 slot-elements | completed current package |

Across all six runs, predictions were identical (0 differences across 120 identities), accuracy remained 1.0, and review rate remained 0. The default-dedup thread-2 run has the lowest round-trip P95 (62.35 ms) among default-enabled candidates and is the selected thread configuration. Deduplication reduced aggregate slot-elements from 3,000 to 1,000 per 1,000 requests (66.7% reduction); the protocol does not expose an authoritative model-call counter, so this is a predictor-batch-size proxy, not a claimed call count. The best measured current-branch P95 is 61.48 ms backend / 62.35 ms round trip, so P95 ≤ 25 ms is not achieved or claimed. PP-ShiTu/global inference remains the dominant cost.

## Limitations and risks

- The canonical release ZIP is missing; the current result uses an explicitly labeled benchmark-only package assembled in `release_staging/` and must be rebuilt through the normal release flow for shipping evidence.
- The active base Python 3.12.7 environment lacks PaddlePaddle/PaddleClas, but the packaged runtime used for these runs reports Python 3.10, Paddle 3.2.2, and PaddleClas 2.6.0.
- Each run exited cleanly and confirmed shutdown; the optional captured backend log was not UTF-8 decodable (`0xd0` at byte 0), so log text is unavailable even though protocol results and lifecycle status are valid.
- Full and targeted pytest runs are affected by Windows ACL/temporary-directory permissions and an inaccessible generated test directory. These failures were recorded, not “fixed” in unrelated code.
- A follow-up with the canonical release ZIP and writable pytest temp/cache directories should repeat the six configurations before release-signoff; these measurements do not establish the 25 ms target.
