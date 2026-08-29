# Task 9 fix round

Implemented the acceptance hardening requested by review. `portable_benchmark` now locks the acceptance corpus to M1/M2/M7, validates source hashes, dimensions, roles and template/query overlap, rejects unsafe package-relative paths and ZIP symlink/path traversal entries, checks backend hello metadata/token/model fingerprint against `app_config`, and fails on invalid labels or missing timing fields. It records source hashes, `needs_review`, lifecycle timing and cross-edition review differences. Shutdown failures and non-zero backend exits are surfaced after kill-and-wait cleanup.

`build_qt5.ps1` now passes `..\workpiece_orientation.pro` to qmake, which resolves the Chinese-path qmake failure mode. Focused benchmark/build tests pass (15 tests). A real retry still cannot produce final ZIPs because the Qt build subsequently fails in compilation/runtime setup on this host; no ZIP, clean-machine, accuracy or latency success is claimed.
