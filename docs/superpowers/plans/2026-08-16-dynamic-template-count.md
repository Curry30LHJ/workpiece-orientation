# Dynamic Template Counts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the production 5+5 template restriction so each orientation accepts any positive number of valid, unique images, persists and predicts with every selected image, reports progress in Qt, and restores old 5+5 libraries.

**Architecture:** Keep the model and fusion algorithms unchanged. `OrientationClassifier` owns variable-length global/local caches; `WorkpieceLibrary` validates, fingerprints, copies, manifests, and atomically commits all templates; the TCP layer exposes strict registration validation plus opt-in progress events; Qt renders dynamic counts and progress while `BackendClient` keeps one pending request until the final response.

**Tech Stack:** Python 3, pytest, OpenCV/numpy, loopback JSON-lines TCP, Qt 5.14.2/C++17, QTest.

## Global Constraints

- Each side must contain a positive number of templates; there is no maximum and the two sides may differ.
- Empty lists, non-string paths, invalid/unreadable images, same-path duplicates, and duplicate decoded image content are rejected; no selected item may be silently filtered or truncated.
- Templates with a total count over 30 produce a Qt performance warning only; one or two templates on either side produce a low-count warning only.
- PP-ShiTuV2, ALIKED, LightGlue, scoring, fusion thresholds, preprocessing, and training behavior remain unchanged.
- Old manifests without `template_counts`, including existing 5+5 libraries, remain recoverable.
- No production implementation is written before its new or changed test has been observed failing.

---

### Task 1: Make classifier caches and prediction independent of template count

**Files:**
- Modify: `tests/test_orientation_classifier.py`
- Modify: `src/orientation_classifier.py:22,155-171, predict local-candidate loop`

**Interfaces:**
- Consumes: existing `OrientationClassifier`, `TemplateCache`, fake predictor/extractor/matcher seams.
- Produces: `build_template_cache(front_paths, back_paths, progress_callback=None) -> TemplateCache`; both labels contain all supplied paths; callback receives `(label, completed_for_label, total_for_label)` after each template is fully extracted.

- [ ] **Step 1: Write failing classifier tests.**

Replace the fixture's hard-coded list construction only where needed and add focused tests:

```python
def test_build_template_cache_accepts_unequal_counts_and_reports_all_templates(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{i}.png", 1) for i in range(1)]
    back = [write_marker(tmp_path / f"back-{i}.png", 2) for i in range(12)]
    progress = []

    cache = classifier.build_template_cache(
        front, back, progress_callback=lambda label, done, total: progress.append((label, done, total))
    )

    assert cache.global_vectors["front"].shape == (1, 2)
    assert cache.global_vectors["back"].shape == (12, 2)
    assert len(cache.local_features["front"]) == 1
    assert len(cache.local_features["back"]) == 12
    assert progress[-1] == ("back", 12, 12)
    assert classifier.global_predictor.calls == 13
    assert classifier.extractor.calls == 13


def test_build_template_cache_rejects_an_empty_orientation(classifier, tmp_path):
    back = [write_marker(tmp_path / f"back-{i}.png", 2) for i in range(1)]

    with pytest.raises(ValueError, match="at least one"):
        classifier.build_template_cache([], back)


def test_prediction_scores_every_local_template(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{i}.png", 1) for i in range(10)]
    back = [write_marker(tmp_path / f"back-{i}.png", 2) for i in range(15)]
    scored = []

    def score(query, template, image_shape, matcher):
        scored.append(template["marker"])
        return {"score": 1.0}

    classifier._score_feature_pair = score
    classifier.set_template_cache("m", classifier.build_template_cache(front, back))
    classifier.predict("m", write_marker(tmp_path / "query.png", 3))

    assert len(scored) == 25
```

Keep the existing two fusion-decision tests; they remain 5+5 regression coverage, not a production constraint.

- [ ] **Step 2: Run the focused tests and verify RED.**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py -q
```

Expected: failures show the exact-five guard rejects 1+12, the empty-set error is absent, and the callback argument is unsupported. Fix test setup errors until the failures are caused by the missing behavior.

- [ ] **Step 3: Implement the minimal classifier change.**

Remove `TEMPLATE_COUNT` and the exact-five check. Reject either empty sequence with `ValueError("Each orientation requires at least one template")`. Add the optional callback and invoke it after each image's global vector and local feature are appended. In `predict`, replace the eager list comprehension that moves all local candidates to the device with a generator loop that moves, scores, and discards one candidate at a time; retain the existing `max`, score function, decision logic, and thresholds.

- [ ] **Step 4: Run classifier tests and verify GREEN.**

Run the same pytest command. Expected: all classifier tests pass and no fusion threshold changes are needed.

- [ ] **Step 5: Commit the classifier slice.**

```powershell
git add tests/test_orientation_classifier.py src/orientation_classifier.py
git commit -m "feat: support variable template cache sizes"
```

### Task 2: Validate, persist, recover, and stream arbitrary template sets

**Files:**
- Modify: `tests/test_workpiece_library.py`
- Modify: `src/workpiece_library.py:module constants, _validate_images, _copy_templates, _record_from_root, register, CacheBuilder`

**Interfaces:**
- Consumes: Task 1 cache-builder callback signature.
- Produces: `WorkpieceLibrary.register(..., progress_callback: Callable[[dict[str, object]], None] | None = None)` returning a record/cache containing all templates; manifests include `template_counts`; `recover` accepts both old and new manifests.

- [ ] **Step 1: Write failing library tests.**

Change the test helper to accept a requested count and add tests for unequal counts, performance-size counts, content duplicates, manifest compatibility, and progress:

```python
def image_set(tmp_path: Path, prefix: str, marker: int, count: int) -> list[Path]:
    return [write_image(tmp_path / f"{prefix}-{index}.png", marker + index) for index in range(count)]


def test_register_persists_unequal_counts_and_manifest_counts(tmp_path):
    library = WorkpieceLibrary(tmp_path / "库")
    front = image_set(tmp_path, "front", 10, 5)
    back = image_set(tmp_path, "back", 20, 12)

    record, cache = library.register("M7", front, back, False, fake_builder)

    manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["template_counts"] == {"front": 5, "back": 12}
    assert len(record.front_images) == len(cache.local_features["front"]) == 5
    assert len(record.back_images) == len(cache.local_features["back"]) == 12


def test_register_allows_more_than_thirty_templates_without_truncation(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    front = image_set(tmp_path, "front", 10, 31)
    back = image_set(tmp_path, "back", 200, 1)

    record, _ = library.register("M7", front, back, False, fake_builder)

    assert len(list((record.root / "0").iterdir())) == 31
    assert len(list((record.root / "1").iterdir())) == 1


def test_register_rejects_same_decoded_image_under_different_paths(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    front = image_set(tmp_path, "front", 10, 1)
    back = image_set(tmp_path, "back", 20, 1)
    shutil.copy2(front[0], tmp_path / "copy.png")

    with pytest.raises(InvalidTemplateSetError):
        library.register("M7", front, [tmp_path / "copy.png"], False, fake_builder)


def test_recover_accepts_old_manifest_without_template_counts(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    record, _ = library.register("M7", image_set(tmp_path, "front", 10, 5), image_set(tmp_path, "back", 20, 5), False, fake_builder)
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("template_counts")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(fake_builder)

    assert [(item.name, len(item.front_images), len(item.back_images)) for item, _ in recovered] == [("M7", 5, 5)]
```

Also add an invalid manifest count mismatch test, a 101+1 naming-order test, and assert a cache-builder failure leaves no staging directory. Update existing 5+5 helpers to pass an explicit count so no test helper silently implies a production limit.

- [ ] **Step 2: Run library tests and verify RED.**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_workpiece_library.py -q
```

Expected: exact-five validation, `TEMPLATE_COUNT * 2`, missing manifest counts, and fixed-width ordering cause failures. If Windows pytest temp ACL errors recur, run the same tests from a user-created writable temp base and record the environment issue separately from assertion failures.

- [ ] **Step 3: Implement validation and persistence.**

Replace exact-five checks with positive-length checks. Preserve input order and cardinality. During image validation, decode each image once for readability and compute a stable digest from shape, dtype, and bytes; reject duplicate canonical paths or digests across either side. Store `template_counts` in the manifest. Derive filename width from each side's count so 101+ files sort naturally. Keep transaction staging, replace backup, and rollback behavior unchanged.

Add an optional domain progress callback. Emit validation/copy phase activity and adapt the classifier callback to overall `features` progress without importing TCP/JSON types. Keep `recover` tolerant of missing `template_counts`, but reject malformed/mismatched present counts.

- [ ] **Step 4: Run library tests and verify GREEN.**

Run the focused pytest command, then rerun `tests/test_orientation_classifier.py` to ensure the changed cache-builder signature is consistent.

- [ ] **Step 5: Commit the library slice.**

```powershell
git add tests/test_workpiece_library.py src/workpiece_library.py
git commit -m "feat: persist arbitrary template sets"
```

### Task 3: Add strict TCP registration and opt-in progress events

**Files:**
- Modify: `tests/test_orientation_tcp_service.py`
- Modify: `src/orientation_tcp_service.py:OrientationCommandDispatcher.dispatch and OrientationTcpServer._handle_client`
- Modify: `tests/test_orientation_service_integration.py` only if its registration payload assumes five items

**Interfaces:**
- Consumes: Task 2 `register(..., progress_callback=None)` and Task 1 classifier callback.
- Produces: `dispatch(request, progress_callback: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]`; registration requests accept all string paths, reject any non-string item, optionally stream `event: progress`, and return `template_counts` plus backend `elapsed_ms` in the final success response.

- [ ] **Step 1: Write failing TCP tests.**

Add tests that use distinct unequal lists and verify strict handling:

```python
def test_register_rejects_non_string_template_item(client, running_server):
    assert client.request("hello")["ok"] is True
    response = client.request(
        "register", name="M7", front_images=["front.png", 7], back_images=["back.png"]
    )
    assert response["error"]["code"] == "INVALID_REQUEST"
    assert running_server.library.register_calls == []


def test_register_without_progress_flag_returns_one_final_response(client):
    assert client.request("hello")["ok"] is True
    response = client.request(
        "register", name="M7", front_images=["f1.png"], back_images=["b1.png"]
    )
    assert response["ok"] is True
    assert "event" not in response
    assert response["template_counts"] == {"front": 1, "back": 1}


def test_register_with_progress_flag_streams_before_final_response(client, running_server):
    assert client.request("hello")["ok"] is True
    client.send_raw(json.dumps({
        "version": 1, "request_id": "r", "command": "register",
        "name": "M7", "front_images": ["f1.png", "f2.png"],
        "back_images": ["b1.png"], "progress_events": True,
    }).encode() + b"\n")
    messages = []
    while True:
        messages.append(client.read())
        if messages[-1].get("ok") is True:
            break
    assert any(message.get("event") == "progress" for message in messages)
    assert messages[0]["request_id"] == "r"
    assert messages[-1]["template_counts"] == {"front": 2, "back": 1}
```

Update `FakeLibrary` and `FakeClassifier` to accept optional progress callbacks and make the fake final response expose actual counts. Add a test that progress callbacks cannot make a registration fail when the socket is closed.

- [ ] **Step 2: Run TCP tests and verify RED.**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py -q
```

Expected: non-string filtering currently accepts the request, final responses lack counts, and no progress event is emitted.

- [ ] **Step 3: Implement the protocol seam.**

Validate every list element with `isinstance(item, str)` before converting to `Path`; do not filter. Validate `progress_events` as an optional boolean. Add an optional `progress_callback` parameter to dispatcher dispatch and pass it into library registration. In the TCP client loop, create a callback only for opted-in registration and send `{version, request_id, event: "progress", command: "register", progress: ...}`; swallow notification `OSError` and disable later notifications. Measure registration elapsed time around the dispatcher operation and derive counts from the committed record.

- [ ] **Step 4: Run TCP and integration tests and verify GREEN.**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py tests\test_orientation_service_integration.py -q
```

- [ ] **Step 5: Commit the TCP slice.**

```powershell
git add tests/test_orientation_tcp_service.py tests/test_orientation_service_integration.py src/orientation_tcp_service.py
git commit -m "feat: stream registration progress over tcp"
```

### Task 4: Update Qt client and template-registration UI

**Files:**
- Modify: `qt_app/backendclient.h`, `qt_app/backendclient.cpp`
- Modify: `qt_app/mainwindow.h`, `qt_app/mainwindow.cpp`, `qt_app/mainwindow.ui`
- Modify: `qt_app/tests/test_backendclient.cpp`, `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Consumes: Task 3 progress frames and final response fields.
- Produces: `BackendClient::progressReceived(const QString &command, const QJsonObject &progress)`; Qt accepts arbitrary multi-selection and shows actual count/warning/progress/elapsed state.

- [ ] **Step 1: Write failing Qt tests.**

Add a `progressReceived` signal spy test that sends a progress frame and then a final response; assert the client remains Busy between frames. Add MainWindow tests replacing the existing exact-five test:

```cpp
void acceptsUnequalTemplateCountsAndShowsWarnings() {
    QTemporaryDir dir;
    BackendClient client;
    MainWindow window(&client, nullptr);
    QMetaObject::invokeMethod(&window, "onBackendReady", Qt::DirectConnection);
    window.setWorkpieceName(QStringLiteral("M7"));
    window.setTemplatePaths(writeImages(dir, QStringLiteral("front"), 1),
                            writeImages(dir, QStringLiteral("back"), 12));

    QCOMPARE(window.findChild<QLabel *>(QStringLiteral("frontTemplatesLabel"))->text(),
             QStringLiteral("正面已选择 1 张"));
    QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text().contains(QStringLiteral("模板较少")));
    QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration", Qt::DirectConnection));
}

void showsPerformanceWarningWithoutDisablingRegistration() {
    QTemporaryDir dir;
    BackendClient client;
    MainWindow window(&client, nullptr);
    QMetaObject::invokeMethod(&window, "onBackendReady", Qt::DirectConnection);
    window.setWorkpieceName(QStringLiteral("M7"));
    window.setTemplatePaths(writeImages(dir, QStringLiteral("front"), 31),
                            writeImages(dir, QStringLiteral("back"), 1));

    QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text().contains(QStringLiteral("耗时")));
    QVERIFY(window.findChild<QPushButton *>(QStringLiteral("registerButton"))->isEnabled());
}
```

Extend the registration fake server to emit a progress frame and final response containing `template_counts`/`elapsed_ms`; assert the completion message displays the actual counts. Keep the replace-confirmation test with explicit 5+5 data as a compatibility case, not a count requirement.

- [ ] **Step 2: Build and run Qt tests to verify RED.**

For each Qt test project, run from a VS x64 developer prompt:

```powershell
E:\QT\5.14\5.14.2\msvc2015_64\bin\qmake.exe qt_app\tests\test_backendclient.pro CONFIG+=release -o qt_app\build-test-backendclient-dynamic\Makefile
nmake /f qt_app\build-test-backendclient-dynamic\Makefile
```

Run the executable with `-platform offscreen` and the existing Qt plugin path. Expected: no progress signal exists, labels/buttons still use `/5`, and registration validation rejects 1+12.

- [ ] **Step 3: Implement `BackendClient` progress handling.**

Add the progress signal. In `handleResponse`, after validating version/request ID and before reading final `ok`, recognize `event == "progress"`, validate command/progress fields, emit the signal, restart `requestTimer_`, and leave `pending_`/Busy state intact. Leave handshake and non-progress final response behavior unchanged.

- [ ] **Step 4: Implement dynamic Qt selection and registration state.**

Change UI copy to count-only labels and generic multi-select buttons. Make `normalizedPaths` preserve every normalized entry. Enable registration when both lists are non-empty and the name is non-empty. `validateRegistration` checks non-empty lists, supported readable images, canonical-path duplicates, and decoded-content duplicates without any maximum. Add labels/status widgets for low-count/performance warnings, phase/progress, and elapsed time. Start a monotonic elapsed timer when registration begins; send `progress_events: true`; update progress/phase from `progressReceived`; stop the timer only on final success/failure. On success display backend `template_counts` and `elapsed_ms`; on failure preserve retry state. Never enable a second request while Busy.

- [ ] **Step 5: Build and run all Qt tests and verify GREEN.**

Build/run `test_backendclient`, `test_backendprocessmanager`, and `test_mainwindow` with the existing Qt 5.14.2 commands and `-platform offscreen`. Expected: all prior tests plus dynamic-count, warning, progress, and completion-count tests pass.

- [ ] **Step 6: Commit the Qt slice.**

```powershell
git add qt_app/backendclient.h qt_app/backendclient.cpp qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/tests/test_backendclient.cpp qt_app/tests/test_mainwindow.cpp
git commit -m "feat: support dynamic template selection in qt"
```

### Task 5: Remove production fixed-five copy, add performance evidence, and run regression suite

**Files:**
- Modify: `src/workpiece_library.py`, `src/orientation_classifier.py`, `qt_app/mainwindow.cpp`, `qt_app/mainwindow.ui`, and any affected production-facing docs/specs found by `rg`.
- Create: `tests/benchmark_dynamic_templates.py` only if no existing benchmark can record the required measurements.
- Modify: `tests/test_orientation_classifier.py`, `tests/test_workpiece_library.py`, `tests/test_orientation_tcp_service.py`, and Qt tests as needed for final coverage.

**Interfaces:**
- Consumes: Tasks 1–4 complete implementation and tests.
- Produces: a clean production-facing search result with no exact-five limitation/copy, test evidence, and a reproducible timing report.

- [ ] **Step 1: Search and remove only production fixed-five restrictions/copy.**

Run:

```powershell
rg -n "TEMPLATE_COUNT|requires exactly five|必须.*5 张|选择.*5 张|/5|== 5|\* 2" src qt_app tests docs openspec
```

Remove or rewrite every match that represents a production registration rule or UI copy. Retain historical compatibility assertions and explicitly labeled fixed-size offline baselines where needed for reproducibility; do not alter model/fusion threshold documentation.

- [ ] **Step 2: Add a repeatable performance measurement.**

Use the existing injectable fake seams for automated scaling checks (1+1, 5+10, 10+15, 31+31) and, when the configured model environment is available, run a real-model smoke benchmark with the same images, one build measurement per set, and five warmed prediction measurements reported by median. Record template counts, hardware, model/device, build milliseconds, prediction median, and memory observations in a new verification note; do not assert machine-dependent absolute times in pytest.

- [ ] **Step 3: Run the complete Python suite.**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q
```

If Windows pytest temporary-directory ACL failures recur, run the tests with an explicitly writable user temp base and report that infrastructure limitation separately; do not turn it into a code skip.

- [ ] **Step 4: Run the complete Qt suite and production smoke checks.**

Run all four Qt test executables (including `test_appconfig`) offscreen, then run the existing service smoke script with a real configured backend if available. Confirm an old 5+5 library recovers and a new unequal-count library can be selected and predicted.

- [ ] **Step 5: Review, diff-check, and commit verification evidence.**

Run `git diff --check`, inspect `git diff --stat`, verify only task-related files changed, and commit the benchmark/verification note and any final test updates:

Use `git add -- <new verification file>` for new files and `git add -p` for already-dirty files so unrelated earlier work in the shared worktree is not staged. Then commit the selected feature hunks:

```powershell
git diff --check
git diff --stat
git add -- docs/verification/<dynamic-template-report>.md
git add -p src/orientation_classifier.py src/workpiece_library.py src/orientation_tcp_service.py qt_app/mainwindow.cpp qt_app/mainwindow.h qt_app/mainwindow.ui qt_app/backendclient.cpp qt_app/backendclient.h tests qt_app/tests
git commit -m "test: verify dynamic template scaling and compatibility"
```
