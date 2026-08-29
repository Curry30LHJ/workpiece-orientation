# Task 6 staged EXE smoke evidence

Collected 2026-08-29 from the existing staged onedir executables (no rebuild in this evidence pass):

- `release_staging/backend-gpu/orientation_backend/orientation_backend.exe`
- `release_staging/backend-cpu/orientation_backend/orientation_backend.exe`

The runner performs hello polling through every loading phase, registers five front and five back templates, predicts one image from each side, requests shutdown, and reports the child process exit code. It uses external model/YAML/data paths configured inside the local runner; those inputs and all generated images remain outside this tracked evidence file.

## GPU

Command (working directory: repository root):

```text
E:\python\anaconda3\envs\shitu\python.exe release_staging\smoke_backend.py gpu 38777
```

Process exit code: `0`; stderr: empty. Complete runner stdout (single JSON line):

```json
{"edition": "gpu", "hello_phases": ["loading_model", "loading_model", "loading_model", "ready"], "ready": {"ok": true, "ready": true, "phase": "ready", "progress": 100, "compute_device": "gpu", "model_fingerprint": "1fab156fb025705a836ad6c28590fc32e1141ae5c530282369412a3078300b33", "instance_token": "smoke-gpu"}, "register": {"ok": true, "template_counts": {"front": 5, "back": 5}, "fast_cache_state": "ready"}, "predictions": [{"ok": true, "label": "front", "needs_review": false, "inference_engine": "fast_geometry"}, {"ok": true, "label": "back", "needs_review": false, "inference_engine": "fast_geometry"}], "shutdown": {"ok": true}, "process_exit": 0}
```

## CPU

Command (working directory: repository root):

```text
E:\python\anaconda3\envs\shitu\python.exe release_staging\smoke_backend.py cpu 38778
```

Process exit code: `0`; stderr: empty. Complete runner stdout (single JSON line):

```json
{"edition": "cpu", "hello_phases": ["loading_model", "loading_model", "ready"], "ready": {"ok": true, "ready": true, "phase": "ready", "progress": 100, "compute_device": "cpu", "model_fingerprint": "1fab156fb025705a836ad6c28590fc32e1141ae5c530282369412a3078300b33", "instance_token": "smoke-cpu"}, "register": {"ok": true, "template_counts": {"front": 5, "back": 5}, "fast_cache_state": "ready"}, "predictions": [{"ok": true, "label": "front", "needs_review": false, "inference_engine": "fast_geometry"}, {"ok": true, "label": "back", "needs_review": false, "inference_engine": "fast_geometry"}], "shutdown": {"ok": true}, "process_exit": 0}
```

The service log paths reported by hello were `release_staging/smoke-data-gpu/logs/orientation-service.log` and `release_staging/smoke-data-cpu/logs/orientation-service.log`; both files were present and zero bytes because the successful protocol transcript is emitted by the runner above. `release_staging/` and `deploy/pyinstaller-work/` are local, untracked build/test outputs and are intentionally excluded from commits.
