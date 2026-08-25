# Propagation Failure Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a single template-local propagation failure into an auditable pending-review result instead of `Internal server error`, while fatal local-model failures preserve stable state and return `MODEL_ERROR`.

**Execution status (2026-08-17):** Tasks 1–4 implemented and verified. Task 5 automated checks and Qt Release build passed; the manual field workflow remains an operator acceptance step.

**Architecture:** Template Evolution owns pair outcomes, target review state, and atomic draft publication. The classifier adapter identifies expected geometry failures and fatal device failures; TCP only translates stable outcomes; Qt renders progress and recovery feedback without deciding propagation policy.

**Tech Stack:** Python 3.10, existing OpenCV/NumPy/PyTorch ALIKED-LightGlue adapter, JSON-lines TCP, pytest, Qt 5.14.2 Widgets/QTest, qmake/MSVC release build.

## Global Constraints

- Do not change PP-ShiTuV2, ALIKED, LightGlue weights, device selection, input size, keypoint limit, or fusion thresholds.
- Do not require rebuilding an existing library or re-annotating current seed templates.
- Preserve TCP protocol version 1 and existing response fields; add only stable diagnostics and error mapping.
- Keep active annotation revision and prediction cache unchanged until an entire candidate group is safe and complete.
- Never treat failed projection or missing correspondence as confirmed absence.
- Use TDD: each behavior starts with a focused red test and ends with its green verification.
- Do not expose stack traces, full source paths, or CUDA details to TCP clients or Qt.

---

## File Structure

- `src/orientation_classifier.py`: classifies expected per-pair geometry failures and fatal local-model failures without altering prediction.
- `src/template_evolution.py`: isolates recoverable pair errors, builds target diagnostics, enforces an error budget, and preserves atomic commit rules.
- `src/orientation_tcp_service.py`: maps fatal propagation errors to `MODEL_ERROR` and configures local rotating diagnostics.
- `src/workpiece_catalog.py`: serializes safe target diagnostics in annotation snapshots.
- `qt_app/annotationmanager.h` and `qt_app/annotationmanager.cpp`: display persistent operation errors and localized target diagnostics.
- `qt_app/mainwindow.cpp`: clears busy state and forwards annotation failures to the manager.
- `tests/test_template_evolution.py`, `tests/test_orientation_tcp_service.py`, `qt_app/tests/test_annotationmanager.cpp`, and `qt_app/tests/test_mainwindow.cpp`: regression coverage.

### Task 1: Isolate recoverable pair errors at the Template Evolution seam

**Files:**

- Modify: `tests/test_template_evolution.py`
- Modify: `src/template_evolution.py`

**Interfaces:**

- Consumes: `TemplateEvolution.save_annotations(workpiece_id, groups, expected_revision, operation_id, progress_callback)` and `project_region_between_templates(source_path, target_path, region)`.
- Produces: target diagnostics with `reason_code`, `attempted_source_count`, `successful_projection_count`, `projection_failure_count`, and `failed_source_template_ids`.

- [ ] **Step 1: Write the failing recoverable-error test**

```python
def test_recoverable_projection_exception_marks_only_target_for_review_and_continues(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=4)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    first = save_front_seed(evolution, record, index=0, operation_id="seed-0")
    catalog.classifier.project_region_between_templates = project_except_for_target("front:02.png")
    result = save_second_seed(evolution, catalog.get(record.id), index=1, operation_id="seed-1")
    failed = target_by_id(result, "edge", "front:02.png")
    assert failed["state"] == "unresolved"
    assert failed["diagnostics"]["reason_code"] == "projection_failed"
    assert target_by_id(result, "edge", "front:03.png")["provenance"] == "automatic"
    assert result["active_annotation_revision"] == first["active_annotation_revision"]
```

- [ ] **Step 2: Run the test and confirm red**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_template_evolution.py -k recoverable_projection_exception -q`

Expected: FAIL because the pair exception currently escapes `_propagate_group`.

- [ ] **Step 3: Implement one-source isolation and terminal progress**

```python
try:
    projected = project(source_paths[source_index], target_path, region)
except PropagationModelError:
    raise
except Exception as exc:
    failure_count += 1
    failed_sources.append(source_template_id)
    logger.warning("propagation pair failed", exc_info=True, extra=diagnostic_context)
    continue
```

After all source attempts, create one unresolved target when fewer than two projections are usable. Include `projection_failed` only when one or more source calls threw. Emit one progress callback from a `finally` block for every target, including skipped seeds. Commit the normal draft only after every target reaches a terminal state.

- [ ] **Step 4: Run the test and confirm green**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_template_evolution.py -k recoverable_projection_exception -q`

Expected: PASS. The failed target is unresolved while later targets still propagate.

- [ ] **Step 5: Commit**

Run: `git add src/template_evolution.py tests/test_template_evolution.py; git commit -m "fix: isolate annotation propagation failures"`

### Task 2: Classify fatal local-model errors without changing recognition

**Files:**

- Modify: `tests/test_template_evolution.py`
- Modify: `src/orientation_classifier.py`
- Modify: `src/template_evolution.py`

**Interfaces:**

- Consumes: existing ALIKED extraction, LightGlue matching, and `project_region` geometry adapter.
- Produces: `dict | None` for usable/no-correspondence projection and `PropagationModelError("局部特征模型不可用，请检查后端日志后重试")` for fatal device/model state.

- [ ] **Step 1: Write failing invalid-geometry and fatal-device tests**

```python
def test_projection_treats_invalid_geometry_as_no_correspondence(classifier):
    classifier.matcher = matcher_with_bad_indices()
    assert classifier.project_region_between_templates(Path("a.png"), Path("b.png"), valid_region()) is None

def test_projection_converts_cuda_memory_failure_to_fatal_propagation_error(classifier):
    classifier.matcher = matcher_raising(RuntimeError("CUDA out of memory"))
    with pytest.raises(PropagationModelError, match="局部特征模型不可用"):
        classifier.project_region_between_templates(Path("a.png"), Path("b.png"), valid_region())
```

- [ ] **Step 2: Run the tests and confirm red**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_template_evolution.py -k "invalid_geometry or cuda_memory_failure" -q`

Expected: FAIL because malformed matcher data and device errors currently escape untyped.

- [ ] **Step 3: Implement narrow classifier behavior**

```python
def _is_fatal_local_model_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("cuda out of memory", "cuda error", "cudnn", "device-side"))
```

Return `None` for expected correspondence and geometry failures (`ValueError`, `TypeError`, `IndexError`, `KeyError`, and OpenCV geometry errors). Convert recognized device-wide failures to `PropagationModelError`. Let unknown exceptions use Task 1’s error budget with diagnostics; do not turn arbitrary programming faults into a normal no-match.

- [ ] **Step 4: Run classifier and Task 1 tests together**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_template_evolution.py -k "projection or propagation" -q`

Expected: PASS. Prediction behavior is unchanged because this path is propagation-only.

- [ ] **Step 5: Commit**

Run: `git add src/orientation_classifier.py src/template_evolution.py tests/test_template_evolution.py; git commit -m "fix: classify local propagation matcher failures"`

### Task 3: Expose stable TCP errors and durable diagnostics

**Files:**

- Modify: `tests/test_orientation_tcp_service.py`
- Modify: `src/orientation_tcp_service.py`
- Modify: `src/template_evolution.py`
- Modify: `src/workpiece_catalog.py`

**Interfaces:**

- Consumes: `PropagationModelError`, operation/workpiece/group/template identifiers, and current `save_workpiece_annotations` progress events.
- Produces: localized `MODEL_ERROR`, rotating local diagnostic records, and safe snapshot fields for unresolved targets.

- [ ] **Step 1: Write failing TCP translation tests**

```python
def test_annotation_save_returns_snapshot_after_recoverable_projection_failure(client, running_server):
    running_server.evolution.save_annotations = save_snapshot_with_projection_failure
    response = client.request("save_workpiece_annotations", **valid_annotation_request())
    assert response["ok"] is True
    assert response["annotations"]["groups"][0]["targets"][2]["diagnostics"]["reason_code"] == "projection_failed"

def test_annotation_save_maps_fatal_projection_failure_to_model_error(client, running_server):
    running_server.evolution.save_annotations = raise_propagation_model_error
    response = client.request("save_workpiece_annotations", **valid_annotation_request())
    assert response["ok"] is False
    assert response["error"]["code"] == "MODEL_ERROR"
```

- [ ] **Step 2: Run the tests and confirm red**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_tcp_service.py -k "annotation_save and projection" -q`

Expected: FAIL because fatal propagation currently reaches a generic error mapping.

- [ ] **Step 3: Implement mapping and local logging**

```python
except PropagationModelError as exc:
    return self._error(request_id, "MODEL_ERROR", str(exc))
```

Configure one rotating UTF-8 file handler under the configured library root during service startup. Log operation, workpiece, group, source, target, exception class, and traceback for recoverable pairs. Snapshot diagnostics retain only stable reason codes, counts, and copied template identifiers.

- [ ] **Step 4: Run TCP annotation and progress tests**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_tcp_service.py -k "annotation or progress" -q`

Expected: PASS. The server accepts another request after either outcome.

- [ ] **Step 5: Commit**

Run: `git add src/orientation_tcp_service.py src/template_evolution.py src/workpiece_catalog.py tests/test_orientation_tcp_service.py; git commit -m "fix: report propagation model failures clearly"`

### Task 4: Render propagation failure as recoverable Qt state

**Files:**

- Modify: `qt_app/tests/test_annotationmanager.cpp`
- Modify: `qt_app/tests/test_mainwindow.cpp`
- Modify: `qt_app/annotationmanager.h`
- Modify: `qt_app/annotationmanager.cpp`
- Modify: `qt_app/mainwindow.cpp`

**Interfaces:**

- Consumes: annotation diagnostics, `progressReceived`, and `requestFailed(code, message)`.
- Produces: `AnnotationManagerDialog::setOperationError(QString)` plus localized target diagnostic display and restored buttons after error.

- [ ] **Step 1: Write failing QTest cases**

```cpp
void managerShowsProjectionFailureDiagnostics() {
    AnnotationManagerDialog dialog;
    dialog.setSnapshot(snapshotWithTargetFailure("front:02.png", "projection_failed"));
    selectTemplate(&dialog, QStringLiteral("front:02.png"));
    QVERIFY(dialog.findChild<QTextEdit *>("annotationDiagnosticsText")
                ->toPlainText().contains(QStringLiteral("局部特征投影失败")));
}

void managerKeepsFailureVisibleAfterBusyClears() {
    AnnotationManagerDialog dialog;
    dialog.setBusy(true);
    dialog.setOperationError(QStringLiteral("局部特征模型不可用，请检查后端日志后重试"));
    dialog.setBusy(false);
    QVERIFY(dialog.findChild<QLabel *>("annotationOperationStatusLabel")
                ->text().contains(QStringLiteral("局部特征模型不可用")));
}
```

- [ ] **Step 2: Build and run to confirm red**

Run: `Set-Location qt_app\build-test-annotationmanager-repropagate-release; cmd.exe /d /s /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 && nmake /f Makefile.Release'; $env:QT_QPA_PLATFORM='offscreen'; .\release\test_annotationmanager.exe -o -,txt`

Expected: FAIL because the manager has no persistent operation-error status or localized projection-failure label.

- [ ] **Step 3: Implement the minimal status flow**

```cpp
void AnnotationManagerDialog::setOperationError(const QString &message) {
    operationStatusLabel_->setStyleSheet(QStringLiteral("color: #b00020;"));
    operationStatusLabel_->setText(message);
}
```

Keep progress transient but retain an operation error after `setBusy(false)`. Clear it only after a successful fresh annotation snapshot. Map `projection_failed`, `target_unreadable`, and insufficient-projection diagnostics to concise Chinese labels. In MainWindow’s annotation mutation failure branch, clear busy state and then call `setOperationError(message)`.

- [ ] **Step 4: Run focused Qt tests**

Run the annotation-manager binary above, then rebuild and run the existing `test_mainwindow` target with the same Qt 5.14.2/MSVC developer shell.

Expected: PASS. A fatal error leaves the manager operable and visibly explains recovery; a successful refresh clears stale status.

- [ ] **Step 5: Commit**

Run: `git add qt_app/annotationmanager.h qt_app/annotationmanager.cpp qt_app/mainwindow.cpp qt_app/tests/test_annotationmanager.cpp qt_app/tests/test_mainwindow.cpp; git commit -m "fix: show recoverable annotation propagation failures"`

### Task 5: Verify the original workflow and release artifact

**Files:**

- Create: `docs/verification/propagation-failure-resilience-results.md`

**Interfaces:**

- Consumes: Tasks 1–4 and the existing variable-template-count workpiece workflow.
- Produces: a reproducible verification record and Qt release executable.

- [ ] **Step 1: Write the verification checklist**

```markdown
- [ ] First seed saves as draft without propagation.
- [ ] Second seed reports every target progress value.
- [ ] One injected pair failure creates a pending-review target and remaining targets finish.
- [ ] An injected CUDA-memory failure returns MODEL_ERROR and preserves active revision.
- [ ] Existing 5+5 and variable-count manifests reopen without migration.
```

- [ ] **Step 2: Run static and Python verification**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m py_compile src\orientation_classifier.py src\template_evolution.py src\orientation_tcp_service.py; E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_template_evolution.py tests/test_orientation_tcp_service.py -q; git diff --check`

Expected: PASS with no whitespace errors.

- [ ] **Step 3: Build Qt 5.14.2 release**

Run: `Set-Location qt_app\build-release; cmd.exe /d /s /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 && nmake /f Makefile.Release'`

Expected: PASS and produce `release\workpiece_orientation.exe`.

- [ ] **Step 4: Run and record the manual field workflow**

```markdown
1. Restart the backend from the freshly built Qt executable.
2. Open a workpiece with two manually marked seeds.
3. Edit the second seed and confirm the region.
4. Verify progress advances beyond the second target.
5. Verify recoverable target trouble is pending review, not Internal server error.
6. Verify injected fatal model failure leaves the prior active annotation revision selected for prediction.
```

Record elapsed time, total templates, unresolved count, error code, active/draft revisions, and diagnostic-log location.

- [ ] **Step 5: Commit**

Run: `git add docs/verification/propagation-failure-resilience-results.md; git commit -m "test: verify propagation failure resilience"`

## Self-Review

- Spec coverage: Task 1 covers continuation and seed preservation; Task 2 classifies local-model failures; Task 3 provides protocol and diagnostics; Task 4 provides operator recovery; Task 5 validates original workflow and compatibility.
- Completeness scan: every task lists files, interfaces, test code, command, expected result, and implementation behavior; no task relies on an unfinished marker or generic error-handling instruction.
- Type consistency: `PropagationModelError` starts in Task 1, is classified in Task 2, translated in Task 3, and rendered in Task 4. `projection_failed` is produced in Task 1 and consumed in Tasks 3–4.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-17-propagation-failure-resilience.md`.

Two execution options:

1. Subagent-Driven (recommended) — dispatch a fresh subagent per task and review between tasks.
2. Inline Execution — execute tasks in this session using `executing-plans`, with review checkpoints.
