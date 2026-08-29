# SDD ledger — plan: docs/superpowers/plans/2026-08-28-offline-portable-gpu-cpu-packages.md

Workspace: E:/Project/wang/ai_区分正反/.worktrees/offline-portable-packages
Implementation branch: feature/20260828/offline-portable-packages
Implementation base: 4a2bfee87bca4ae91bf27879ab8ac07d2ee1da23
Pre-flight: no task-to-task or plan/global-constraint conflicts found.
Task 1: minor (deferred): remove the unused `hashlib` import left in `src/orientation_classifier.py`; final whole-branch review must triage it.
Task 1: complete (commits 4a2bfee..fe8bf81, spec compliant; 1 deferred minor)
Task 2: fix round 1/5 (1 addressed, 0 open — nested Windows junction cleanup contained; commits 5db84a9..d78690b)
Task 2: complete (commits fe8bf81..d78690b, review clean)
Task 3: fix round 1/5 (1 Important open — enforce at most 14 dated diagnostic backups during startup pruning)
Task 3: minor (deferred): add a loader test using the real empty workpiece library; final whole-branch review must triage it.
Task 3: fix round 1/5 (1 addressed, 0 open — startup retains only the newest 14 dated backups and preserves unrelated files; commits a7350d3..1a60720)
Task 3: complete (commits d78690b..1a60720, spec compliant; 1 deferred minor)
Task 4: minor (deferred): extend packaged required-field test rows to cover backend executable, Paddle config, data root, model SHA, project root, and model directory; final whole-branch review must triage it.
Task 4: complete (commits 1a60720..a34af7b, spec compliant; 1 deferred minor)
Task 5: fix round 1/5 (5 Important open — graceful shutdown escalation, loading retry re-entrancy, unconditional packaged pre-launch conflict, Recovering preservation, stale process callback rejection)
Task 5: minor (deferred): Qt tests still emit known font/offscreen warnings; final whole-branch review must triage whether deterministic suppression is warranted.
Task 5: fix round 1/5 (4 addressed, 1 Important open — real application exit currently starts graceful shutdown only after the main event loop has ended)
Task 5: fix round 2/5 (1 Important open — initiate async owned-backend shutdown before window/event-loop exit and add not-Ready terminate-to-kill fallback)
Task 5: fix round 2/5 (original exit timing mostly addressed; 2 Important open — QProcess destructor can still kill an unverified child, and post-shutdown close can double-prompt active-task exit)
Task 5: fix round 3/5 (2 Important open — non-destructive release for unverified running QProcess; latch authorized close across asynchronous shutdown)
Task 5: fix round 3/5 (2 addressed, 0 open — detached unverified child teardown and single-prompt close intent; commits 876a01e..b11d35b)
Task 5: complete (commits a34af7b..b11d35b, spec compliant; 1 deferred minor)
Task 6: implementation committed (49426c2); metadata/spec/scripts green, but real GPU/CPU environment creation, EXE builds, and smoke are pending because locked Paddle GPU download stalled; review and a controlled retry are required before Task 6 can be marked complete.
Task 6: documentation fix (3de2a09..HEAD) — report now labels sklearn/inference.json failures as resolved historical attempts, records the final GPU/CPU smoke commands, log paths, exit code 0 evidence, and clarifies release_staging/deploy/pyinstaller-work remain untracked local artifacts.
Task 6: smoke evidence fix (8b2b4be) — reran staged GPU/CPU executables (runner exit code 0 each), captured complete compact stdout JSON including loading→ready, register front/back, predict front/back, shutdown, and process_exit, and committed the model/library-free transcript at docs/packaging/task-6-smoke-evidence.md.
Task 7: implementation — portable staging/audit/manifest/license/ZIP tooling, Qt staging-safe build changes, release orchestration, offline guide/notices, and synthetic audit tests added. Pytest tmp_path execution is blocked by sandbox WinError 5 creating temporary directories; config test and py_compile pass.
Task 7: fix round — commit ee90f40 removes production dependency-check bypass, enforces windeployqt and required guide/notices/licenses, handles recursive _internal runtime trees and stricter data/path audits.
Task 7: second fix round — commit 84e8e24 adds staging input/containment guards, strict dumpbin machine parsing and system DLL allowlist, and real Git commit metadata.
Task 7: third fix round — commit 4891d89 tightens exact-root/reparse containment, dependency-line parsing, runtime roots, and strict edition-aware license requirements.
Task 7: fourth fix round — commit ff70418 aligns license collection with locked environments and removes the invalid `python` distribution requirement.
Task 7: license follow-up — commit 6af0886 detects the locked environment's `LICENSE_PYTHON.txt` reliably and fails clearly if absent.
Task 7: final cleanup — commit a6a537a expands Windows/API-set system dependency filtering and documents PaddleClas license source fallback.
Task 8: implementation — added hidden Qt startup probe, injectable packaged protocol smoke client/CLI, focused tests, and Task 8 report. Py_compile passes; Qt runner is blocked by Unicode worktree path resolution and pytest temp creation by sandbox WinError 5. Real staged-package smoke remains environment-limited.
Task 8: fix rounds — hardened cross-edition destination/source lifecycle validation, matching instance-token shutdown, held-out query checks, complete success/failure logs and command responses, report-persistence error precedence, and release-script smoke gating (commits 87847d5..61264e4).
Task 8: complete (commits b5d892c..61264e4, independent review APPROVED; direct injected smoke/portability tests pass; pytest/Qt runner limitations documented above).
Task 9: fix round — benchmark now validates exact M1/M2/M7 inventory, source SHA/dimensions/roles and template/query disjointness; rejects unsafe config paths and ZIP symlinks/traversal; validates backend hello identity and prediction labels/timings, records review/source hashes, and reports shutdown/process failures. Qt qmake receives a relative project path. Focused tests: 15 passed; real package acceptance remains blocked by Qt build path/runtime constraints.
Task 9: complete — final GPU/CPU ZIPs built and audited; 5+10 asymmetric package smoke passed with clean shutdown; 1,000-request GPU/CPU benchmarks passed with 100% accuracy and zero cross-edition label/review differences. GPU backend P95 14.530 ms (25 ms gate passed); CPU backend P95 222.440 ms (gate not applicable). Full Python suite 726 passed/5 skipped; all 11 Qt5.14.2 offscreen targets passed. Commit d918723 is embedded in the packages; commit 1d3f185 adds a test-only UTF-8/mock-launcher fix after the runtime package build. Clean Windows VM acceptance remains pending.
