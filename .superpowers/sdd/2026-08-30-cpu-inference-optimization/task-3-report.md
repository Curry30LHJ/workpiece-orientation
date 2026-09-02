# Task 3 Report: portable CPU benchmark thread override

## Summary

Added an optional positive `cpu_threads` override to `run_benchmark`. The
backend subprocess receives `WORKPIECE_CPU_THREADS` only for CPU packages when
the override is supplied. Reports now retain the existing shape and include
`settings.cpu_threads`; GPU packages do not receive the override. The CLI now
accepts `--cpu-threads` as a positive integer.

## Files

- `release_tools/portable_benchmark.py`
- `scripts/benchmark_portable_service.py`
- `tests/test_benchmark_portable_service.py`

## Tests

Focused new test:

```text
python -m pytest tests/test_benchmark_portable_service.py -k cpu_thread_override -q
.                                                                        [100%]
1 passed, 24 deselected in 0.21s
```

Required Task 3 tests:

```text
python -m pytest tests/test_benchmark_portable_service.py tests/test_benchmark_fast_geometry_inference.py -q
.......................................................................  [100%]
71 passed in 2.76s
```

An initial sandboxed run could not create pytest's temporary directory due to
managed Windows ACLs; the required tests were rerun successfully with the
approved elevated test execution.

## Risks / concerns

The override is validated strictly as `None` or a positive integer (booleans
are rejected). The subprocess environment is copied from the parent process,
so any pre-existing environment variables remain inherited; this change only
injects the CPU thread variable for CPU packages.
