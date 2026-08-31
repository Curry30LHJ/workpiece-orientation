# CPU batch parallel inference verification results

Date: 2026-09-01  
Repository: `E:\Project\wang\pp_813`  
Branch: `feature/20260830/cpu-inference-optimization`  
Source and tooling verification commit: `77fb5c3` (`fix: tolerate loading batch handshakes`)
Runtime package build commit: `28833be561036bca140f16068d7ca1b891579c14` (`fix: coalesce refreshes after backend reconnect`)

## Scope

This record covers Task 8 verification for the CPU portable package after the Task 6 fix landed on `28833be561036bca140f16068d7ca1b891579c14`, with the packaged smoke and benchmark CLI revalidated after the Task 7 CLI fixes landed on `77fb5c3`. No production source or test files were modified during this verification pass. Only this document is intended to be committed.

Between `28833be561036bca140f16068d7ca1b891579c14` and `77fb5c3`, the changed paths were:

```text
docs/verification/cpu-batch-parallel-inference-results.md
scripts/benchmark_batch_inference.py
scripts/smoke_portable_package.py
tests/test_benchmark_portable_service.py
tests/test_smoke_portable_package.py
```

That means the CPU `1.1.0` ZIP validated here still contains runtime bytes built from `28833be561036bca140f16068d7ca1b891579c14`; the later `77fb5c3` work only changed documentation, verification scripts, and their tests.

## Repository state observed

The original Task 8 verification pass started with:

```text
?? .pytest-task8-tmp/
?? runtime_reports/
```

Earlier `git status` invocations also emitted ACL warnings for `.pytest-task8-tmp/` and `.pytest-tmp-task2/`. Per task constraints, `runtime_reports/` and `.pytest-tmp-task2/` were left untouched. `.pytest-task8-tmp/` was verified to resolve exactly to `E:\Project\wang\pp_813\.pytest-task8-tmp` and was removed after the test evidence was captured.

The follow-up evidence refresh on `77fb5c3` started with:

```text
?? runtime_reports/
```

## Full Python test suite

Command:

```text
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=E:\Project\wang\pp_813\.pytest-task8-tmp
```

Saved log:

```text
release_staging\task8-pytest-full-28833be.txt
```

Result:

```text
821 passed, 3 skipped, 1 warning in 67.13s (0:01:07)
```

The single warning was the existing Paddle `ccache` warning and did not indicate a functional failure.

## Full Qt test suite

Primary command:

```text
powershell -File scripts\run_qt5_tests.ps1
```

Saved script log:

```text
release_staging\task8-qt-full-28833be.txt
```

Because the script log did not preserve per-target totals cleanly under redirection, the built test executables were also rerun individually and saved under:

```text
release_staging\task8-qt-individual-28833be\
```

Verified totals:

```text
test_annotationmanager: Totals: 10 passed, 0 failed, 0 skipped, 0 blacklisted, 94ms
test_appconfig: Totals: 22 passed, 0 failed, 0 skipped, 0 blacklisted, 127ms
test_appfoundation: Totals: 16 passed, 0 failed, 0 skipped, 0 blacklisted, 37ms
test_backendclient: Totals: 31 passed, 0 failed, 0 skipped, 0 blacklisted, 5855ms
test_backendprocessmanager: Totals: 33 passed, 0 failed, 0 skipped, 0 blacklisted, 20936ms
test_geometryrulecanvas: Totals: 18 passed, 0 failed, 0 skipped, 0 blacklisted, 17ms
test_geometryrulespage: Totals: 78 passed, 0 failed, 0 skipped, 0 blacklisted, 1308ms
test_inspectionpage: Totals: 47 passed, 0 failed, 0 skipped, 0 blacklisted, 513ms
test_mainwindow: Totals: 127 passed, 0 failed, 0 skipped, 0 blacklisted, 23448ms
test_startupsmokecontroller: Totals: 4 passed, 0 failed, 0 skipped, 0 blacklisted, 26ms
test_workpiecelibrarypage: Totals: 40 passed, 0 failed, 0 skipped, 0 blacklisted, 583ms
```

The previously observed `inspectionpage` and `mainwindow` failures did not reproduce on `28833be561036bca140f16068d7ca1b891579c14`.

## CPU 1.1.0 portable package build

First build attempt used the standard script entry:

```text
scripts\build_portable_release.ps1 -Edition cpu -Version 1.1.0 -CpuPython E:\python\anaconda3\envs\workpiece-package-cpu\python.exe -SmokeDatasetRoot E:\Project\wang\pp_813\data\1_M1 -OutputRoot E:\Project\wang\pp_813\release_staging
```

Saved log:

```text
release_staging\task8-build-cpu-1.1.0-28833be.txt
```

That attempt failed in packaging with `NotADirectoryError: models\shitu_rec`.

The repository did contain the required model payload under:

```text
third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer
```

The build was rerun with that explicit existing model path, without any source changes:

```text
scripts\build_portable_release.ps1 -Edition cpu -Version 1.1.0 -CpuPython E:\python\anaconda3\envs\workpiece-package-cpu\python.exe -ModelPath third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer -SmokeDatasetRoot E:\Project\wang\pp_813\data\1_M1 -OutputRoot E:\Project\wang\pp_813\release_staging
```

Saved log:

```text
release_staging\task8-build-cpu-1.1.0-28833be-rerun.txt
```

Result: success, exit code `0`.

Produced staging directory:

```text
release_staging\cpu-1.1.0\WorkpieceOrientation-CPU
```

Produced release artifact:

```text
release_artifacts\WorkpieceOrientation-CPU-x64-1.1.0.zip
release_artifacts\WorkpieceOrientation-CPU-x64-1.1.0.zip.sha256
```

Artifact facts:

- ZIP size: `255148439` bytes
- ZIP SHA-256: `f170bff51e40da954c1a7b0e07924a09aa50325101a5bef6630dd5688809b1c9`
- Sidecar first token matched the artifact SHA-256 exactly
- Package version: `1.1.0`
- Edition: `cpu`
- Model fingerprint: `1fab156fb025705a836ad6c28590fc32e1141ae5c530282369412a3078300b33`

## Standard package smoke

Saved report:

```text
release_staging\reports\cpu-smoke.json
```

Verified from the saved report:

- Package boot reached `ready`
- Ready-state capabilities reported `predict_batch=true`, `batch_ready=true`, `batch_workers=4`, `batch_threads_per_worker=1`
- `register` succeeded with `front=5`, `back=10`
- Scalar `predict` succeeded for one front image and one back image
- Reported `workpiece_id`: `6a46c71c3c3c46b295f1cc7826a77127`
- Process exit code: `0`

## Exact ZIP verification

The exact release ZIP was extracted into:

```text
release_staging\task8-exact-zip-verify-28833be\
```

Saved static audit:

```text
release_staging\reports\task8-exact-zip-static-28833be.json
```

Verified from the saved audit:

- `qt_executable_exists=true`
- `backend_executable_exists=true`
- `manifest_file_count=1170`, meaning `manifest.json` lists 1170 payload files and does not count `manifest.json` itself; the ZIP therefore contains 1171 file entries in total
- `production_src_module_count=20`
- `forbidden_python_files=[]`
- Data directories `workpieces`, `rules`, `cache`, `logs`, and `temp` were empty before smoke usage
- `app_config.package_version=1.1.0`
- `app_config.compute_device=cpu`
- `app_config.model_sha256=1fab156fb025705a836ad6c28590fc32e1141ae5c530282369412a3078300b33`
- `version.git_commit=28833be561036bca140f16068d7ca1b891579c14`
- `version.build_utc=2026-08-31T20:50:33.687748+00:00`

The extracted backend also passed the frozen-module import verification used by the Task 8 flow.

## Exact ZIP retained smoke

Saved report:

```text
release_staging\reports\task8-exact-zip-smoke-28833be.json
```

Verified from the saved report:

- Smoke completed with `ok=true`
- Package root was the extracted exact ZIP payload
- Retained temporary package root:

```text
E:\Project\wang\pp_813\release_staging\task8-exact-zip-verify-28833be\.便携 smoke package-8139a84763f14a50a552cadc36ace1e4\xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- Ready-state capabilities reported `predict_batch=true`, `batch_ready=true`, `batch_workers=4`, `batch_threads_per_worker=1`
- `register` succeeded with `front=5`, `back=10`
- Reported `workpiece_id`: `8c68b62a8014428a9e8e0468ef5f3301`
- Process exit code: `0`

## Official exact ZIP smoke CLI revalidation on 77fb5c3

Command:

```text
E:\python\anaconda3\envs\shitu\python.exe scripts\smoke_portable_package.py --package-root E:\Project\wang\pp_813\release_staging\task8-exact-zip-verify-28833be\WorkpieceOrientation-CPU --dataset-root E:\Project\wang\pp_813\data\1_M1 --report E:\Project\wang\pp_813\release_staging\reports\task8-exact-zip-smoke-official-77fb5c3.json
```

Saved report:

```text
release_staging\reports\task8-exact-zip-smoke-official-77fb5c3.json
```

Verified from the saved report:

- Smoke completed with `ok=true`
- `package_root=E:\Project\wang\pp_813\release_staging\task8-exact-zip-verify-28833be\WorkpieceOrientation-CPU`
- `temp_package=E:\Project\wang\pp_813\release_staging\task8-exact-zip-verify-28833be\.便携 smoke package-1988c7efa4934855889387f98b5718d8\xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- `workpiece_id=f931ed23f39d4ca59b0f6ee9fb5df3bf`
- `template_counts.front=5`
- `template_counts.back=10`
- Final `hello` reported `ready=true`, `predict_batch=true`, `batch_workers=4`, `batch_threads_per_worker=1`
- `process_exit=0`

## Five-image batch smoke from different working directory

Ordered images used:

```text
front: E:\Project\wang\pp_813\data\1_M1\0\4763_1460_0_39_2026_05_25_07_33_29_4353.png
back:  E:\Project\wang\pp_813\data\1_M1\1\3753_1233_0_11_2026_05_25_07_36_20_2185.png
front: E:\Project\wang\pp_813\data\1_M1\0\2847_2531_0_14_2026_05_25_07_33_27_1613.png
back:  E:\Project\wang\pp_813\data\1_M1\1\1611_1394_0_32_2026_05_25_07_36_21_9885.png
front: E:\Project\wang\pp_813\data\1_M1\0\2706_1271_0_37_2026_05_25_07_33_29_2593.png
```

Expected label order:

```text
front, back, front, back, front
```

Saved report:

```text
release_staging\reports\task8-exact-zip-batch-smoke-28833be.json
```

Verified from the saved report:

- Backend launch working directory differed from the package root
- Ready-state capabilities reported `predict_batch=true`, `batch_ready=true`, `batch_workers=4`, `batch_threads_per_worker=1`
- `predict_batch` returned `ok=true`
- `worker_count=4`
- `fallback=null`
- Observed labels matched expected labels exactly:

```text
front, back, front, back, front
```

- Process exit code: `0`

## Official packaged benchmark CLI on this machine

These benchmark runs used the retained exact-package root and workpiece from the successful exact-ZIP retained smoke above, as requested:

```text
package_root=E:\Project\wang\pp_813\release_staging\task8-exact-zip-verify-28833be\.便携 smoke package-8139a84763f14a50a552cadc36ace1e4\xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
workpiece_id=8c68b62a8014428a9e8e0468ef5f3301
```

CLI form used for both runs:

```text
E:\python\anaconda3\envs\shitu\python.exe scripts\benchmark_batch_inference.py --package-root <retained package root> --workpiece-id 8c68b62a8014428a9e8e0468ef5f3301 --image <same ordered five-image batch> --warmup 5 --iterations 20 --workers <N> --threads-per-worker <M> --expected-label front --expected-label back --expected-label front --expected-label back --expected-label front --report <report path>
```

Saved reports:

```text
release_staging\benchmarks\task8-batch-4x1-official-77fb5c3.json
release_staging\benchmarks\task8-batch-2x2-official-77fb5c3.json
```

Environment recorded in both reports:

- Platform: `Windows-10-10.0.19045-SP0`
- CPU string: `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`
- Logical processors: `24`
- Warmup: `5` iterations
- Measured iterations: `20`
- Batch size: `5`
- Accuracy: `100/100 = 1.0`
- Fallback: `used=false`
- Actual worker count matched the requested worker count in both runs

Measured results:

| Config | Throughput (images/s) | Batch p50 (ms) | Batch p95 (ms) | Batch p99 (ms) | Batch max (ms) | Reported workers | Reported threads/worker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4×1 | 40.93 | 120.59 | 132.06 | 135.21 | 136.00 | 4 | 1 |
| 2×2 | 31.87 | 157.60 | 163.05 | 163.55 | 163.67 | 2 | 2 |

Observed labels matched the expected five-image order in both runs via the official benchmark CLI. On this development machine, `4×1` remained faster than `2×2` for the short run.

## Target i5 acceptance status

Status: `PENDING TARGET-HARDWARE MEASUREMENT`

The official CLI benchmark above is still only a development-machine check. The required target-hardware acceptance run still needs to be executed on the designated i5 machine against the same `1.1.0` CPU package.

Recommended target run parameters:

- Five-image ordered batch listed above
- `workers=4`, `threads_per_worker=1`
- `workers=2`, `threads_per_worker=2`
- warmup `50`
- measured iterations `200`

Recommended target flow on the i5 machine:

1. Extract the exact `WorkpieceOrientation-CPU-x64-1.1.0.zip`.
2. Run `scripts\smoke_portable_package.py` against the extracted package so the package creates a retained temp package and returns the new `workpiece_id`.
3. Use that retained package root and workpiece id directly with the benchmark CLI below.

Benchmark commands for the target i5 machine, run from the repository root:

```text
E:\python\anaconda3\envs\shitu\python.exe scripts\benchmark_batch_inference.py --package-root "<retained package root from the exact-ZIP smoke run>" --workpiece-id "<workpiece_id from the same smoke run>" --image "E:\Project\wang\pp_813\data\1_M1\0\4763_1460_0_39_2026_05_25_07_33_29_4353.png" --image "E:\Project\wang\pp_813\data\1_M1\1\3753_1233_0_11_2026_05_25_07_36_20_2185.png" --image "E:\Project\wang\pp_813\data\1_M1\0\2847_2531_0_14_2026_05_25_07_33_27_1613.png" --image "E:\Project\wang\pp_813\data\1_M1\1\1611_1394_0_32_2026_05_25_07_36_21_9885.png" --image "E:\Project\wang\pp_813\data\1_M1\0\2706_1271_0_37_2026_05_25_07_33_29_2593.png" --warmup 50 --iterations 200 --workers 4 --threads-per-worker 1 --expected-label front --expected-label back --expected-label front --expected-label back --expected-label front --report release_staging\benchmarks\task8-batch-4x1-i5.json
E:\python\anaconda3\envs\shitu\python.exe scripts\benchmark_batch_inference.py --package-root "<retained package root from the exact-ZIP smoke run>" --workpiece-id "<workpiece_id from the same smoke run>" --image "E:\Project\wang\pp_813\data\1_M1\0\4763_1460_0_39_2026_05_25_07_33_29_4353.png" --image "E:\Project\wang\pp_813\data\1_M1\1\3753_1233_0_11_2026_05_25_07_36_20_2185.png" --image "E:\Project\wang\pp_813\data\1_M1\0\2847_2531_0_14_2026_05_25_07_33_27_1613.png" --image "E:\Project\wang\pp_813\data\1_M1\1\1611_1394_0_32_2026_05_25_07_36_21_9885.png" --image "E:\Project\wang\pp_813\data\1_M1\0\2706_1271_0_37_2026_05_25_07_33_29_2593.png" --warmup 50 --iterations 200 --workers 2 --threads-per-worker 2 --expected-label front --expected-label back --expected-label front --expected-label back --expected-label front --report release_staging\benchmarks\task8-batch-2x2-i5.json
```

Interpretation rule: compare throughput and latency percentiles on the i5 box using the same five-image ordered batch, then choose the faster configuration only if accuracy remains `100%`, `fallback.used=false`, and `fallback.actual_worker_count` matches the requested worker count.

## Final verification conclusion

With source/tooling verified on `77fb5c3` and the runtime ZIP still built from `28833be561036bca140f16068d7ca1b891579c14`, the current repository state provides:

- full Python suite passing,
- full Qt suite passing,
- successful CPU `1.1.0` portable ZIP build,
- exact ZIP static verification passing,
- exact ZIP smoke passing,
- five-image different-working-directory batch smoke passing,
- official packaged benchmark CLI evidence showing `4×1` outperforming `2×2` on the development machine.

No new production-source defect was proven during this pass. The only open item is the required target-i5 measurement.
