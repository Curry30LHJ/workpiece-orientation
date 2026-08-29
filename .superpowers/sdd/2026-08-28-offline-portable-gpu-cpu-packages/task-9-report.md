# Task 9 final report

Task 9 is complete for the developer-host acceptance scope. The final artifacts and machine-readable evidence are recorded in `docs/verification/offline-portable-gpu-cpu-results.md` and `.json`.

## Evidence

- GPU ZIP: `release_artifacts/WorkpieceOrientation-GPU-x64-1.0.0.zip`, 2,537,514,980 bytes, SHA-256 `8818e1c17725a92b4a2097b3c761fc86b174e1c224001aa7b129310d471f33f4`.
- CPU ZIP: `release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip`, 254,500,483 bytes, SHA-256 `e29b2a95bc927920f51e025039935947a81fac9cc4b42287f473bb4a2b7520ce`.
- Static package audit and fresh ZIP extraction audit passed for both editions. Both shipped `data/workpieces` roots are empty; no current workpiece library or geometry rules were copied into the packages.
- Real package smoke passed for both editions with asymmetric 5+10 templates, front/back predictions, recycle/restore, and graceful shutdown. Both processes exited with code 0 and temporary smoke data was removed.
- GPU 1,000-request benchmark: 120/120 unique queries correct, sample accuracy 100%, review rate 0%, backend P95 14.530 ms, forced termination false.
- CPU 1,000-request benchmark: 120/120 unique queries correct, sample accuracy 100%, review rate 0%, backend P95 222.440 ms, forced termination false. GPU/CPU label differences: 0; review differences: 0.
- Python full suite: 726 passed, 5 skipped, 0 failed. Qt 5.14.2 offscreen suite: all 11 targets passed. Package-focused regression suite: 61 passed.

## Environment

Windows 10 build 19045, Intel64 Family 6 Model 183, 32 GiB RAM, NVIDIA GeForce RTX 4060 Ti, driver 560.94. Package runtime versions are Python 3.10, Paddle 3.2.2, PaddleClas 2.6.0, PyInstaller 6.22.2 and Qt 5.14.2. The GPU/CPU package metadata embeds build commit `d918723261bf7d211b752e2f1022ba120b6ebe63`; commit `1d3f185738b93a97b1994a842204cf76e98d481f` contains a test-only Qt UTF-8/mock-launcher correction made after the runtime package build.

## Scope limitation

No clean Windows 10/11 VM or Windows Sandbox was available, so clean-machine, network-disabled acceptance is still pending. The recorded smoke/benchmark runs are developer-host, offline-style package-copy runs; they do not prove compatibility with every NVIDIA driver or GPU. The CPU edition is functional on CPU-only hosts but does not meet the 25 ms GPU latency target.
