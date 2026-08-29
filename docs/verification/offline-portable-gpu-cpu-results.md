# Offline portable GPU/CPU acceptance

This report records the real Task 9 attempt at commit `61264e4d310a7a01ea70323efef5d986cb6b2597`.

The benchmark statistics/comparison suite passed: 12 tests. Related benchmark/package/smoke tests collected 32 passes but 38 setup errors caused by the host pytest temporary-directory lock (`PermissionError`); those errors are not treated as passes. Qt tests were not run.

The requested build used the GPU and CPU package environments, the externally supplied PP-ShiTu model directory, and `-SkipSmoke` because no valid acceptance smoke dataset is shipped. Both backend PyInstaller builds completed. An initial run exposed a missing `PYTHONPATH`; the release script now exports the worktree root and a focused regression test passes. The retry reached `portable_package` successfully but failed because the Qt release directory was absent: qmake could not resolve `qt_app/workpiece_orientation.pro` from this Chinese-path worktree. Consequently no final ZIP or SHA-256 file was produced, and ZIP audits, GPU/CPU 1000-request benchmarks, cross-edition comparison, and clean-machine offline startup are **not run**. No success, timing, accuracy, or hash is inferred.

The attempt did not modify `runtime_reports/`, external acceptance data, or any source package. Existing `release_staging/smoke-data-*` directories are prior artifacts and were not used as acceptance data. A rerun should use a path where the release script's Python subprocess can import the worktree package, then run `benchmark_portable_service.py` with the fixed M1/M2/M7 acceptance spec and at least 1000 measured requests. The 25 ms GPU backend gate remains unchanged.
