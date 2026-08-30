# CPU inference optimization verification

Date: 2026-08-30  
Branch: `feature/20260830/cpu-inference-optimization`  
Code commit: `b5c2b47eb1bdafd123ea7374f683bf9cb17cddc9`  
Report commit: `381c9c2` (the prior report commit; this clarification commit necessarily has a new SHA)

## Method and data

Task 4 required the complete pytest suite, packaged CPU runs at thread candidates 1/2/4, and paired 1,000-iteration runs with `WORKPIECE_CPU_DEDUPLICATE_SLOTS=0` versus the default-enabled path. The acceptance input was checked read-only at `runtime_reports/fast-geometry-acceptance.json` (SHA-256 `45c73611e828b8bbd94650ee316fc970f859a04d2ad1c40a5e609d2b99572eef`, 120 queries across M1/M2/M7). The file is user-provided and was not changed or staged.

The workspace model directory is present and its three file hashes are recorded in the companion JSON, but the required `release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip` does not exist. Consequently no backend was started, no cold-start `ready_ms` was observed, and no CPU latency, accuracy, review, deduplication, or model-call measurements can be claimed.

## Commands and results

| Command | Result |
| --- | --- |
| `python -m pytest -q` | Collection blocked by `PermissionError: [WinError 5]` while scanning `qt_app/tests/build-test_geometryrulespage/pytest-task10-profiles`; pytest cache also lacked write access. |
| `python -m pytest tests/test_fast_orientation.py tests/test_orientation_classifier.py tests/test_orientation_tcp_service.py tests/test_workpiece_catalog.py -q` | `103 passed, 177 errors`; errors were chiefly pytest temporary-directory lock permissions under `C:\Users\Administrator\AppData\Local\Temp\pytest-of-Administrator`. No source was changed to work around this environment failure. |
| `python scripts/benchmark_portable_service.py ... --cpu-threads 1 ...` | Exit 2; `FileNotFoundError` for the required release ZIP. |
| Same command with `--cpu-threads 2` | Exit 2; same missing ZIP. |
| Same command with `--cpu-threads 4` | Exit 2; same missing ZIP. |

The three benchmark attempts wrote only temporary failure JSON outside the repository. Existing `runtime_reports/` files were not overwritten and were not added to git.

## Measurements

| Configuration | backend elapsed (mean/P50/P95/P99/max ms) | round trip (mean/P50/P95/P99/max ms) | accuracy | review | unique slots / calls | status |
| --- | --- | --- | --- | --- | --- | --- |
| Dedup disabled, thread 1 | — | — | — | — | — | unavailable: package missing |
| Dedup disabled, thread 2 | — | — | — | — | — | unavailable: package missing |
| Dedup disabled, thread 4 | — | — | — | — | — | unavailable: package missing |
| Dedup default enabled, thread 1 | — | — | — | — | — | unavailable: package missing |
| Dedup default enabled, thread 2 | — | — | — | — | — | unavailable: package missing |
| Dedup default enabled, thread 4 | — | — | — | — | — | unavailable: package missing |

No P95 target conclusion is possible for this branch. In particular, P95 ≤ 25 ms is not claimed without a real packaged measurement. The existing acceptance JSON contains a historical fast-geometry total P95 of 23.324 ms at a different commit and environment; it is reference evidence only, not this verification run.

## Limitations and risks

- The release ZIP is missing, so package contents, startup behavior, OneDNN thread configuration, and CPU-vs-GPU parity were not exercised.
- PaddlePaddle/PaddleClas are unavailable in the active Python environment (Python 3.12.7); versions are therefore `null` in JSON.
- Full and targeted pytest runs are affected by Windows ACL/temporary-directory permissions and an inaccessible generated test directory. These failures were recorded, not “fixed” in unrelated code.
- A follow-up on a machine with the release ZIP, writable pytest temp/cache directories, and the intended Paddle runtime must run all six benchmark configurations and fill the null metrics before making performance claims.
