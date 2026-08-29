# Task 9 final report

Task 9 is complete for the developer-host acceptance scope. The final artifacts and machine-readable evidence are recorded in `docs/verification/offline-portable-gpu-cpu-results.md` and `.json`.

## Evidence

- GPU ZIP: `release_artifacts/WorkpieceOrientation-GPU-x64-1.0.0.zip`, 2,537,515,338 bytes, SHA-256 `02d7e7fe1e958a50e8c6ceb7184559aa0893d8cdcff5655a59e9d1c850be1817`.
- CPU ZIP: `release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip`, 254,500,024 bytes, SHA-256 `4044820dcabbd1fcc8c517eb7d80b6b6b6a64478b4210640be0536052d51f2db`.
- Static package audit and fresh ZIP extraction audit passed for both editions. Both shipped `data/workpieces` roots are empty; no current workpiece library or geometry rules were copied into the packages.
- Real package smoke passed for both editions with asymmetric 5+10 templates, front/back predictions, recycle/restore, and graceful shutdown. Both processes exited with code 0 and temporary smoke data was removed.
- Direct packaged Qt smoke passed for both editions after commit `d3d1fa0834f11964919124359921c2ebc5db27cc`: GPU 9,856 ms and CPU 5,983 ms in the current cached developer environment, both exit code 0. The smoke entry point now keeps the event loop alive while the backend loads.
- GPU 1,000-request benchmark: 120/120 unique queries correct, sample accuracy 100%, review rate 0%, backend P95 14.530 ms, forced termination false.
- CPU 1,000-request benchmark: 120/120 unique queries correct, sample accuracy 100%, review rate 0%, backend P95 222.440 ms, forced termination false. GPU/CPU label differences: 0; review differences: 0.
- Python full suite: 726 passed, 5 skipped, 0 failed. Qt 5.14.2 offscreen suite: all 11 targets passed. Package-focused regression suite: 61 passed. Packaged Qt smoke: 2/2 editions passed.

## Environment

Windows 10 build 19045, Intel64 Family 6 Model 183, 32 GiB RAM, NVIDIA GeForce RTX 4060 Ti, driver 560.94. Package runtime versions are Python 3.10, Paddle 3.2.2, PaddleClas 2.6.0, PyInstaller 6.22.2 and Qt 5.14.2. Both rebuilt package metadata files embed build commit `d3d1fa0834f11964919124359921c2ebc5db27cc`.

## Scope limitation

No clean Windows 10/11 VM or Windows Sandbox was available, so clean-machine, network-disabled acceptance is still pending. The recorded smoke/benchmark runs are developer-host, offline-style package-copy runs; they do not prove compatibility with every NVIDIA driver or GPU. The CPU edition is functional on CPU-only hosts but does not meet the 25 ms GPU latency target.

The earlier `-platform offscreen` probe is not a supported user launch path: the delivery packages intentionally contain the Windows platform plugin but not Qt's `qoffscreen.dll`. The final packaged smoke uses the default Windows platform and now waits for the actual backend readiness result.
