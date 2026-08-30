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

## Historical reference: legacy CPU package

For supplemental context only, the sibling worktree package was run without copying it into this branch. Its manifest identifies package commit `d3d1fa0834f11964919124359921c2ebc5db27cc` (Paddle 3.2.2/PaddleClas 2.6.0), ZIP SHA-256 `4044820dcabbd1fcc8c517eb7d80b6b6b6a64478b4210640be0536052d51f2db`, and 50 warmup + 1,000 measured requests per run. These are historical measurements from the pre-optimization package and do not represent code commit `b5c2b47`.

| Legacy package CPU threads | startup ready (ms) | backend elapsed mean/P50/P95/P99/max (ms) | round trip mean/P50/P95/P99/max (ms) | accuracy / review |
| --- | ---: | --- | --- | --- |
| 1 | 18,132.10 | 172.52 / 171.25 / 199.49 / 212.32 / 301.42 | 173.24 / 171.97 / 200.21 / 213.79 / 302.29 | 1.0 / 0.0 |
| 2 | 13,896.07 | 162.70 / 161.69 / 179.76 / 192.84 / 223.60 | 163.45 / 162.39 / 180.81 / 193.68 / 224.32 | 1.0 / 0.0 |
| 4 | 14,404.12 | 160.59 / 159.93 / 177.39 / 185.44 / 240.97 | 161.32 / 160.62 / 178.13 / 186.61 / 242.34 | 1.0 / 0.0 |

The legacy package did not expose the new slot-dedup telemetry (`global_unique_slots` or model-call count), and no dedup-disabled/default pair was run against the current code. These results must not be used to claim the optimization target is met.

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
