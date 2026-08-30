# Task 2 report: CPU fast inference runtime

## Implementation summary

- Added `CPU_THREADS_ENV = WORKPIECE_CPU_THREADS` and `CPU_SLOT_DEDUP_ENV = WORKPIECE_CPU_DEDUPLICATE_SLOTS`.
- CPU thread override accepts only positive decimal integers and emits a warning for invalid values while retaining the YAML value.
- CPU-only thread assignment preserves GPU configuration; exact slot deduplication defaults on for CPU and is forced off for GPU. Boolean overrides accept `0/1/true/false/yes/no`; invalid values warn and use the device default.
- Exposed `OrientationClassifier.cpu_num_threads`, passed deduplication to `FastOrientationEngine`, and preserved the option when rebuilding the fast engine for background cache work.

## Changed files

- `src/orientation_classifier.py`
- `tests/test_orientation_classifier.py`
- `.superpowers/sdd/2026-08-30-cpu-inference-optimization/task-2-report.md`

## TDD and verification

RED attempt (sandbox):

```text
python -m pytest tests/test_orientation_classifier.py -k "cpu_load_applies or gpu_load_ignores or invalid_cpu_thread" -q
EEE; pytest could not create its Windows temporary-directory lock (PermissionError: [Errno 13] Permission denied).
```

RED was therefore blocked by the environment before test bodies ran. The same tests were rerun with approved elevated execution and then passed after implementation:

```text
python -m pytest tests/test_orientation_classifier.py -k "cpu_load_applies or gpu_load_ignores or invalid_cpu_thread" -q --basetemp .pytest-tmp-task2
...                                                                      [100%]
3 passed, 114 deselected, 1 warning in 3.76s
```

Related required tests:

```text
python -m pytest tests/test_orientation_classifier.py tests/test_fast_orientation.py -q --basetemp .pytest-tmp-task2-related2
139 passed, 1 failed, 1 warning in 18.04s
```

The sole failure is pre-existing environment dependency setup in `test_fast_load_does_not_construct_aliked_or_lightglue`: importing `src.aliked_lightglue_matcher` raises `ModuleNotFoundError: No module named 'lightglue'`. No unrelated code was changed.

## Unresolved risks

- Full related suite remains blocked by the missing optional `lightglue` package in this environment.
- The SciPy/NumPy compatibility warning is emitted by the installed environment (`SciPy` requires NumPy `<1.29`, while `2.5.2` is installed); it does not affect the three passing configuration tests.

Final focused regression:

```text
python -m pytest tests/test_orientation_classifier.py -k "cpu_load_applies or gpu_load_ignores or invalid_cpu_thread or load_configures_requested_paddle_device" -q --basetemp .pytest-tmp-task2-final
5 passed, 112 deselected, 1 warning in 0.43s
```
