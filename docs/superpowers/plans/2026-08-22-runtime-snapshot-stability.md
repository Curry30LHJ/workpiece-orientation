# Runtime Snapshot Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep online prediction available on an immutable published workpiece revision while confirmed templates are built in the background, preserve workpiece sidecars transactionally, and prevent Qt domain errors from being treated as backend failures.

**Architecture:** `WorkpieceCatalog` owns immutable active snapshots. Slow append preparation and all model work occur outside the catalog lock; a priority gate serializes GPU work and lets waiting online inference run before the next background template step. Prepared disk state and cache are committed with a short revision-checked swap. Qt separates command failure from transport failure, and backend restart becomes an ownership-aware asynchronous state machine.

**Tech Stack:** Python 3.10, pytest, OpenCV, PaddlePaddle, PyTorch, Qt 5.14.2 Widgets/Network/Test, MSVC 2019, JSON-line loopback protocol

## Global Constraints

- Work in `E:\Project\wang\pp_813` on `feature/20260815/workpiece-orientation-desktop-ui`.
- Preserve all unrelated and pre-existing working-tree changes; do not reformat adjacent code.
- Do not modify PP-ShiTu, ALIKED, LightGlue, image preprocessing, fusion policy, thresholds, or model weights.
- Do not introduce true concurrent Paddle/Torch model execution.
- Preserve old 5+5 libraries and arbitrary unequal front/back template counts.
- Preserve immutable geometry revisions and previews; only the staged profile library-revision pointer may change during append.
- Every production behavior change starts with a failing focused test and follows red, green, then focused regression.
- Because relevant source files already contain uncommitted work, do not create implementation commits that would capture pre-existing hunks. Keep the implementation as a reviewable working-tree diff and commit only newly created documentation files.

---

### Task 1: Add the priority model execution gate

**Files:**
- Create: `src/model_execution_gate.py`
- Create: `tests/test_model_execution_gate.py`

**Interfaces:**
- Produces: `PriorityModelGate.run_online(action: Callable[[], T]) -> T`
- Produces: `PriorityModelGate.run_background_step(action: Callable[[], T]) -> T`
- Guarantee: one active action; queued online work precedes the next background acquisition

- [ ] **Step 1: Write the failing serialization test**

```python
def test_gate_never_runs_two_model_actions_concurrently():
    gate = PriorityModelGate()
    active = 0
    maximum = 0
    guard = threading.Lock()

    def action():
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with guard:
            active -= 1

    threads = [
        threading.Thread(target=gate.run_background_step, args=(action,)),
        threading.Thread(target=gate.run_online, args=(action,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
    assert maximum == 1
```

- [ ] **Step 2: Write the failing priority-order test**

```python
def test_waiting_online_action_runs_before_next_background_step():
    gate = PriorityModelGate()
    first_started = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def first_background():
        order.append("background-1")
        first_started.set()
        assert release_first.wait(1)

    first = threading.Thread(target=gate.run_background_step, args=(first_background,))
    first.start()
    assert first_started.wait(1)
    online = threading.Thread(target=gate.run_online, args=(lambda: order.append("online"),))
    second = threading.Thread(target=gate.run_background_step, args=(lambda: order.append("background-2"),))
    online.start()
    second.start()
    release_first.set()
    for thread in (first, online, second):
        thread.join(timeout=1)
    assert order == ["background-1", "online", "background-2"]
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_model_execution_gate.py -q -p no:cacheprovider
```

Expected: collection fails because `src.model_execution_gate` does not exist.

- [ ] **Step 4: Implement the minimal condition-based gate**

```python
from threading import Condition
from typing import Callable, TypeVar

T = TypeVar("T")


class PriorityModelGate:
    def __init__(self) -> None:
        self._condition = Condition()
        self._active = False
        self._online_waiters = 0

    def run_online(self, action: Callable[[], T]) -> T:
        with self._condition:
            self._online_waiters += 1
            try:
                self._condition.wait_for(lambda: not self._active)
                self._active = True
            finally:
                self._online_waiters -= 1
        try:
            return action()
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    def run_background_step(self, action: Callable[[], T]) -> T:
        with self._condition:
            self._condition.wait_for(
                lambda: not self._active and self._online_waiters == 0
            )
            self._active = True
        try:
            return action()
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Step 3 command. Expected: both tests pass repeatedly with `--count=20` when pytest-repeat is available; otherwise run the command five times.

- [ ] **Step 6: Review the task diff without committing overlapping work**

Run:

```powershell
git diff --check -- src/model_execution_gate.py tests/test_model_execution_gate.py
git status --short -- src/model_execution_gate.py tests/test_model_execution_gate.py
```

Expected: no whitespace error; only the two new files are reported.

### Task 2: Make prediction consume an immutable captured cache

**Files:**
- Modify: `src/orientation_classifier.py`
- Modify: `src/workpiece_catalog.py`
- Modify: `tests/test_orientation_classifier.py`
- Modify: `tests/test_workpiece_catalog.py`

**Interfaces:**
- Produces: frozen `ActiveWorkpieceSnapshot(record: WorkpieceRecord, cache: TemplateCache)`
- Produces: `WorkpieceCatalog.capture_snapshot(workpiece_id: str) -> ActiveWorkpieceSnapshot`
- Produces: `OrientationClassifier.predict_with_cache(cache, image_path, *, library_revision=None)`
- Consumes: `PriorityModelGate`

- [ ] **Step 1: Add a failing classifier test proving explicit cache use**

Create two caches with opposite global/local fake outcomes, install one in the compatibility map, and call `predict_with_cache` with the other:

```python
def test_predict_with_cache_uses_supplied_snapshot_not_mutable_map(classifier, tmp_path):
    requested = cache_for_label("front")
    classifier.set_template_cache("m7", cache_for_label("back"))
    result = classifier.predict_with_cache(
        requested, image(tmp_path / "query.png", 19), library_revision=7
    )
    assert result["label"] == "front"
    assert result["library_revision"] == 7
```

- [ ] **Step 2: Add a failing catalog test proving the lock is released before inference**

Use a fake classifier whose `predict_with_cache` blocks on an event. While it is blocked, call `capture_snapshot` on another thread and assert that call completes before inference is released.

```python
def test_predict_releases_catalog_lock_before_classifier_runs(tmp_path):
    catalog, classifier, record = create_blocking_catalog(tmp_path)
    prediction = threading.Thread(target=catalog.predict, args=(record.id, query_path(tmp_path)))
    prediction.start()
    assert classifier.started.wait(1)
    captured: list[ActiveWorkpieceSnapshot] = []
    reader = threading.Thread(target=lambda: captured.append(catalog.capture_snapshot(record.id)))
    reader.start()
    reader.join(timeout=0.2)
    assert not reader.is_alive()
    assert captured[0].record.revision == record.revision
    classifier.release.set()
    prediction.join(timeout=1)
```

- [ ] **Step 3: Run the two focused tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py::test_predict_with_cache_uses_supplied_snapshot_not_mutable_map tests/test_workpiece_catalog.py::test_predict_releases_catalog_lock_before_classifier_runs -q -p no:cacheprovider
```

Expected: failures because the new APIs do not exist and prediction still holds the catalog lock.

- [ ] **Step 4: Add snapshot ownership to the catalog**

Implement the frozen dataclass and the two small helpers:

```python
@dataclass(frozen=True)
class ActiveWorkpieceSnapshot:
    record: WorkpieceRecord
    cache: TemplateCache


def _activate(self, record: WorkpieceRecord, cache: TemplateCache) -> None:
    self._snapshots[record.id] = ActiveWorkpieceSnapshot(record, cache)
    self.classifier.set_template_cache(record.id, cache)


def capture_snapshot(self, workpiece_id: str) -> ActiveWorkpieceSnapshot:
    with self._lock:
        return self._snapshots[workpiece_id]
```

Initialize `_snapshots` in the constructor and use `_activate` after register and recover. Remove snapshots when recycling or purging.

- [ ] **Step 5: Add explicit-cache prediction and gate online model work**

Refactor the existing prediction body into a private cache-based function. Public production calls become:

```python
def predict_with_cache(self, cache, image_path, *, library_revision=None):
    def run():
        started = time.perf_counter()
        image = _read_image(Path(image_path))
        result = (
            self._predict_geometry(image, cache, started)
            if cache.geometry_profile is not None
            else self._predict_baseline(image, cache, started)
        )
        if library_revision is not None:
            result["library_revision"] = int(library_revision)
        return result
    return self.model_gate.run_online(run)
```

`WorkpieceCatalog.predict()` captures the snapshot, releases its lock, and calls `predict_with_cache`. Preserve the old classifier `predict(workpiece_id, image_path)` only as a compatibility wrapper for untouched callers and tests.

- [ ] **Step 6: Route background feature extraction through bounded gate steps**

In `build_template_cache`, wrap one template's global and local extraction in one `run_background_step` call. Do not hold the gate during image decoding, filesystem copying, progress callbacks, or manifest writes. Route geometry leave-one-out and template projection through background steps as well.

- [ ] **Step 7: Run focused and classifier/catalog regressions**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_model_execution_gate.py tests/test_orientation_classifier.py tests/test_workpiece_catalog.py -q -p no:cacheprovider
```

Expected: all selected tests pass and existing fusion assertions remain unchanged.

### Task 3: Prepare and commit complete staged template updates

**Files:**
- Modify: `src/workpiece_library.py`
- Modify: `tests/test_workpiece_library.py`

**Interfaces:**
- Produces: `PreparedTemplateUpdate`
- Produces: `WorkpieceLibrary.prepare_append(base_record, front_images, back_images, build_cache, *, operation_id, progress_callback=None)`
- Produces: `WorkpieceLibrary.commit_prepared(prepared) -> tuple[WorkpieceRecord, Path | None]`
- Produces: `WorkpieceLibrary.abort_prepared(prepared) -> None`

- [ ] **Step 1: Write the failing sidecar preservation test**

Create `geometry_masks/profile.json`, `geometry_masks/revisions/1.json`, and `geometry_masks/previews/front-00.png` in a registered workpiece. Prepare and commit one extra front template. Assert immutable file hashes are unchanged and the old active directory is untouched before commit.

```python
def test_prepared_append_preserves_geometry_sidecars_until_commit(tmp_path):
    library, record = registered_library(tmp_path)
    hashes = write_geometry_sidecars(record.root)
    prepared = library.prepare_append(
        record,
        [image(tmp_path / "confirmed.png", 31)],
        [],
        fake_builder,
        operation_id="job-append-1",
    )
    assert sidecar_hashes(record.root) == hashes
    committed, backup = library.commit_prepared(prepared)
    assert immutable_sidecar_hashes(committed.root) == hashes["immutable"]
    assert len(committed.front_images) == len(record.front_images) + 1
    library.remove_retired(backup)
```

- [ ] **Step 2: Write failure and stale-owner tests**

- Builder exception removes staging and leaves active root byte-for-byte unchanged.
- `abort_prepared` is safe when called twice.
- `commit_prepared` rejects a prepared object whose formal workpiece root no longer matches the base record.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_library.py -k "prepared_append" -q -p no:cacheprovider
```

Expected: collection or assertion failure because prepared transactions do not exist and append drops sidecars.

- [ ] **Step 4: Implement the transaction value object and staging clone**

The prepared value stores operation ID, workpiece ID, base revision, staging root, staged record, base candidate cache, and item digests. Preparation must:

```python
staging = self.library_dir / f".staging-{uuid.uuid4().hex}"
shutil.copytree(base_record.root, staging)
shutil.rmtree(staging / "0")
shutil.rmtree(staging / "1")
(staging / TEMPLATE_CACHE_FILE_NAME).unlink(missing_ok=True)
```

Recreate both template directories, update counts and revision in the staged manifest, set schema version 3, and write `last_template_update` with the supplied operation ID and SHA-256 digests. Construct the staged record explicitly because its root name is temporary.

- [ ] **Step 5: Implement short commit, rollback, and retired cleanup**

Commit performs only checked renames and record-map replacement. Return the backup path instead of recursively deleting it while the catalog lock is held. `remove_retired` removes that exact `.backup-*` directory after the caller releases the lock. Existing startup recovery remains the final fallback for an undeleted backup.

- [ ] **Step 6: Keep the old append API as a compatibility wrapper**

`append_templates()` calls prepare, commit, and retired cleanup synchronously using a generated operation ID when no explicit ID is supplied. This keeps direct library tests and external callers working while the catalog moves to explicit prepare/commit orchestration.

- [ ] **Step 7: Run the complete library suite**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_library.py -q -p no:cacheprovider
```

Expected: all existing dynamic-count, invalid-image, duplicate, replacement, cache, recycle, restore, and recovery tests pass.

### Task 4: Integrate staged append with snapshots and active geometry

**Files:**
- Modify: `src/workpiece_catalog.py`
- Modify: `src/orientation_classifier.py`
- Modify: `src/geometry_mask_profiles.py`
- Modify: `tests/test_workpiece_catalog.py`
- Modify: `tests/test_orientation_classifier.py`
- Modify: `tests/test_geometry_mask_profiles.py`

**Interfaces:**
- Consumes: `PreparedTemplateUpdate`, `ActiveWorkpieceSnapshot`, `PriorityModelGate`
- Produces: optional `operation_id` on `WorkpieceCatalog.append_templates`
- Produces: explicit `base_cache` parameter for candidate geometry and legacy-mask preparation

- [ ] **Step 1: Write the failing non-blocking append test**

Block candidate building after it has captured revision `vN`. While append is blocked, call `catalog.predict()` and prove it returns the old revision before releasing the builder. After release, prove the next prediction returns `vN+1`.

- [ ] **Step 2: Write the failing same-base conflict test**

Prepare two updates from revision 1, commit the first, and assert the second raises the stable stale-revision exception without modifying counts or cache.

- [ ] **Step 3: Write the failing active-geometry append test**

Publish or fixture an active geometry profile, append one accepted template, and assert:

```python
snapshot = catalog.capture_snapshot(record.id)
assert snapshot.record.revision == record.revision + 1
assert snapshot.cache.geometry_profile_revision == active_revision
assert profile_document(snapshot.record)["library_revision"] == snapshot.record.revision
assert immutable_revision_hashes(snapshot.record.root) == original_hashes
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run the three exact new tests with `pytest -q -p no:cacheprovider`. Expected: append holds the lock, lacks operation metadata, or cannot prepare geometry from an explicit base cache.

- [ ] **Step 5: Refactor cache preparation to accept an explicit base cache**

Add optional keyword-only `base_cache` to `prepare_geometry_cache` and `prepare_template_masks`. The compatibility default reads the classifier map; the staged append path always supplies the prepared base cache. Never install a candidate in the compatibility map before commit.

- [ ] **Step 6: Implement staged active-cache preparation**

Add a geometry-profile helper that reads and copies profile state under its own lock, releases that lock, updates only the staged profile's library-revision pointer, and invokes classifier preparation from the explicit base cache. If no geometry profile is active, prepare the active legacy-mask cache from the explicit base cache. Persist the raw portion of the effective cache into staging before commit.

- [ ] **Step 7: Implement revision-checked snapshot publication**

`WorkpieceCatalog.append_templates` must:

```python
base = self.capture_snapshot(workpiece_id)
prepared = self.library.prepare_append(
    base.record, front_images, back_images,
    self.classifier.build_template_cache,
    operation_id=resolved_operation_id,
    progress_callback=progress_callback,
)
effective = self._prepare_effective_staged_cache(prepared)
self.classifier.save_template_cache(prepared.staged_record, effective)
with self._lock:
    if self._snapshots[workpiece_id].record.revision != prepared.base_revision:
        raise StaleWorkpieceRevisionError(
            f"Workpiece revision changed: expected {prepared.base_revision}, "
            f"current {self._snapshots[workpiece_id].record.revision}"
        )
    record, retired = self.library.commit_prepared(prepared)
    self._activate(record, effective)
self.library.remove_retired(retired)
```

On every exception, abort an unconsumed prepared update outside the catalog lock.

- [ ] **Step 8: Run catalog, classifier, geometry, and template-evolution regressions**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_catalog.py tests/test_orientation_classifier.py tests/test_geometry_mask_profiles.py tests/test_template_evolution.py -q -p no:cacheprovider
```

Expected: all tests pass; no fusion-score fixture changes are accepted.

### Task 5: Reconcile confirmed-template jobs after restart

**Files:**
- Modify: `src/template_evolution.py`
- Modify: `tests/test_template_evolution.py`

**Interfaces:**
- Consumes: manifest `last_template_update`
- Produces: job ID as append `operation_id`
- Produces: corrupt-job quarantine in `.evolution`

- [ ] **Step 1: Write a failing committed-but-building recovery test**

Fixture a manifest whose `last_template_update.operation_id` equals a persisted building job and whose digests match. Construct a restarted `TemplateEvolution`, run the next job, and assert state is completed and template counts do not increase.

- [ ] **Step 2: Write a failing unrelated-revision recovery test**

When the active revision changed but `last_template_update.operation_id` does not match, assert the restarted job becomes stale/failed with an explicit revision message and never calls append.

- [ ] **Step 3: Write a failing malformed-job quarantine test**

Write invalid JSON to `.evolution/jobs.json`. Construction must move it to a uniquely named `jobs.corrupt-*.json`, start with an empty job set, and leave the catalog usable.

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_template_evolution.py -k "restart or corrupt or reconcile" -q -p no:cacheprovider
```

- [ ] **Step 5: Implement reconciliation before base-revision rejection**

Pass `job["job_id"]` to catalog append. Before rebuilding a recovered job, compare manifest operation ID, target revision, and the set of stored item digests. Mark matching jobs completed and clean staged confirmation files. Do not treat a different revision as successful.

- [ ] **Step 6: Quarantine malformed persistence without failing model startup**

Catch JSON/schema errors in `_load`, atomically rename the invalid jobs file inside `.evolution`, log the quarantine path, and initialize empty operation/job maps. Do not catch filesystem permission errors that prevent safe persistence.

- [ ] **Step 7: Run template evolution and TCP regressions**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_template_evolution.py tests/test_orientation_tcp_service.py -q -p no:cacheprovider
```

Expected: confirmation coalescing, duplicate rejection, review, progress, retry, and protocol tests pass.

### Task 6: Remove startup lock inversion and bound handshakes

**Files:**
- Modify: `src/geometry_mask_profiles.py`
- Modify: `src/template_evolution.py`
- Modify: `src/workpiece_catalog.py`
- Modify: `src/orientation_tcp_service.py`
- Modify: `tests/test_geometry_mask_profiles.py`
- Modify: `tests/test_workpiece_catalog.py`
- Modify: `tests/test_orientation_tcp_service.py`

**Interfaces:**
- Produces: idempotent `start()` methods on both background worker owners
- Produces: `OrientationTcpServer(dispatcher, *, host="127.0.0.1", port=37651, handshake_timeout_seconds=5.0)`
- Lock rule: catalog code never calls geometry-profile code while holding the catalog lock

- [ ] **Step 1: Write a failing worker-start ordering test**

Instrument fake workers and catalog recovery. Assert recovery finishes and the runtime ready snapshot is installed before either worker's `start()` is invoked.

- [ ] **Step 2: Write a failing catalog/geometry lock regression**

Block `rebuild_active_cache` in a fake profile object during append or recovery. From a second thread, capture the active catalog snapshot and assert it completes while the profile call is blocked.

- [ ] **Step 3: Write a failing silent-handshake timeout test**

Start the TCP server with a 0.1-second handshake timeout, connect a raw socket without sending a hello line, wait for release, and prove a normal client can subsequently complete hello instead of receiving `SERVER_BUSY` forever.

- [ ] **Step 4: Run the focused tests and verify RED**

Run the three exact new tests. Expected: workers start in constructors, one catalog path holds its lock around profile work, and the claimed handshake has no timeout.

- [ ] **Step 5: Add explicit worker lifecycle**

Construct `GeometryMaskProfiles` and `TemplateEvolution` with `start_worker=False`, add idempotent `start()`, publish the recovered runtime snapshot, then start workers outside runtime/catalog locks. Preserve existing `shutdown()` behavior.

- [ ] **Step 6: Move all catalog-to-profile calls outside the catalog lock**

For recover and restore, build base snapshots first, release the catalog lock, prepare effective geometry/legacy caches, then perform a short revision-checked cache swap. Profile-to-catalog calls may remain only when no catalog path holds a profile lock in the opposite order. Pass records explicitly when available so profile helpers do not reacquire the catalog unnecessarily.

- [ ] **Step 7: Bound only the handshake phase**

Set the accepted socket timeout before reading the first hello. On successful ready handshake, restore blocking mode with `sock.settimeout(None)`. On timeout, release `_handshake_in_progress` and close the socket. Keep the existing one-active-client behavior.

- [ ] **Step 8: Run service concurrency and recovery regressions**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_catalog.py tests/test_geometry_mask_profiles.py tests/test_template_evolution.py tests/test_orientation_tcp_service.py -q -p no:cacheprovider
```

### Task 7: Separate Qt command failures from transport failures

**Files:**
- Modify: `qt_app/backendclient.h`
- Modify: `qt_app/backendclient.cpp`
- Modify: `qt_app/backendprocessmanager.h`
- Modify: `qt_app/backendprocessmanager.cpp`
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp`
- Modify: `qt_app/tests/test_backendclient.cpp`
- Modify: `qt_app/tests/test_backendprocessmanager.cpp`
- Modify: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Produces: `commandFailed(command, code, message)`
- Produces: `transportFailed(code, message)`
- Removes production dependence on generic `requestFailed(code, message)`

- [ ] **Step 1: Write failing BackendClient signal tests**

- A matching error response for `publish_geometry_mask_profile` emits one `commandFailed` with that command, emits no `transportFailed`, keeps the socket open, and ends in `Ready`.
- A malformed response, handshake timeout, and socket error emit `transportFailed` and leave `Ready`.
- Local `BUSY` and `NOT_READY` rejections include the attempted command in `commandFailed`.

- [ ] **Step 2: Write a failing manager isolation test**

After the manager is ready, have the service return a domain error. Assert `backendUnavailable` remains zero and no process launch/restart occurs.

- [ ] **Step 3: Write failing MainWindow preservation tests**

- Populate geometry draft state, emit `commandFailed("publish_geometry_mask_profile", "GEOMETRY_VALIDATION_FAILED", "validation rejected")`, and assert the dialog is enabled with its draft retained.
- Add completed batch rows, emit a predict command failure, and assert completed rows remain.
- Emit a transport failure during a mutation and assert the UI reports an unknown outcome without silently resending.

- [ ] **Step 4: Build the three Qt test targets and verify RED**

From a VS 2019 x64 developer command shell, regenerate and build `test_backendclient.pro`, `test_backendprocessmanager.pro`, and `test_mainwindow.pro`, then run each executable with `-platform offscreen -o -,txt`. Expected: new signal/state assertions fail before implementation.

- [ ] **Step 5: Implement BackendClient failure channels**

Business error handling captures `pending_->command`, clears the request, emits `commandFailed`, and sets `Ready` without aborting the socket. Protocol, connection, and timeout failures call a transport-only helper that aborts when necessary and emits `transportFailed` exactly once.

- [ ] **Step 6: Restrict BackendProcessManager to lifecycle failures**

Connect the manager only to handshake/transport signals. Preserve special loading retry behavior. A domain command response must be invisible to the manager.

- [ ] **Step 7: Make MainWindow command-specific and state-preserving**

Change `onClientRequestFailed` to accept the command argument and branch on that immutable value. Release only the affected busy indicator. Do not call `clearInspectionState` or `clearBatchState` for a command failure. On transport failure, stop in-flight timers, retain selections/results/drafts, mark a mutation outcome unknown, and require operator refresh before retry.

- [ ] **Step 8: Rebuild and run Qt tests GREEN**

Expected: all BackendClient, BackendProcessManager, and MainWindow tests pass under Qt 5.14.2 offscreen.

### Task 8: Implement ownership-aware asynchronous restart

**Files:**
- Modify: `qt_app/backendprocessmanager.h`
- Modify: `qt_app/backendprocessmanager.cpp`
- Modify: `qt_app/tests/test_backendprocessmanager.cpp`

**Interfaces:**
- Produces: internal lifecycle states for reconnect, graceful stop, terminate wait, kill wait, and relaunch
- Preserves: public `start()`, `restart()`, `shutdownOwnedService()`, and ownership signals

- [ ] **Step 1: Make the fake launcher support delayed exit**

Add a mode in which `terminate()` records the call but does not emit `finished` until the test explicitly calls `finish(exitCode)`.

- [ ] **Step 2: Write failing owned-restart ordering tests**

- `restart()` on an owned ready service sends shutdown and does not call `start()` again before process `finished`.
- If graceful shutdown does not exit by the first deadline, `terminate()` is called once.
- If terminate does not exit by the second deadline, `kill()` is called once.
- After `finished`, exactly one replacement launch occurs.

- [ ] **Step 3: Write failing external reconnect test**

For an externally owned ready service, restart disconnects/reconnects, never invokes terminate/kill/start, and emits ready after the next handshake.

- [ ] **Step 4: Build and run the manager test to verify RED**

Expected: current restart either reconnects an owned ready service without restarting it or starts before process exit.

- [ ] **Step 5: Implement the minimal asynchronous state machine**

Use one single-shot stop-escalation timer and explicit states. Process `finished` is the only normal transition to relaunch. The first timer expiry calls terminate and rearms; the second calls kill. External reconnect bypasses all process actions. Ignore duplicate restart clicks while a stop/relaunch transition is active.

- [ ] **Step 6: Run BackendProcessManager and full Qt regressions**

Build and run all Qt test targets used by `workpiece_orientation.pro`. Expected: no duplicate launches, no external termination, and no regression in loading-handshake startup.

### Task 9: Full verification and evidence report

**Files:**
- Create: `docs/verification/runtime-snapshot-stability-results.md`

**Interfaces:**
- Consumes all preceding tasks
- Produces reproducible test, timing, compatibility, and changed-file evidence

- [ ] **Step 1: Run the complete Python project suite**

Run from the repository root in the `shitu` environment with cache provider disabled and a verified writable basetemp. Record exact pass/skip/fail counts and elapsed time. Do not substitute older verification documents for the current run.

- [ ] **Step 2: Rebuild and run current Qt 5.14.2 tests**

Use:

```text
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat
E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe
nmake
```

Regenerate the changed test Makefiles before building. Run current binaries with the Qt 5.14.2 `bin` directory first on the process PATH and `-platform offscreen -o -,txt`.

- [ ] **Step 3: Run real-model compatibility smoke checks**

Load existing M1, M2, and M7 libraries. Verify old manifests recover, unequal template counts remain unchanged, and one prediction per available orientation completes. Do not mutate production runtime libraries; use a copied test library.

- [ ] **Step 4: Measure foreground/background behavior**

For 1+1, 5+10, 10+15, and 35+35 template sets, record:

- idle prediction elapsed time;
- prediction elapsed time while append is building;
- time from online request arrival to service;
- total background append time;
- active revision before and after commit.

The pass condition is behavioral rather than a fabricated hardware-independent millisecond limit: foreground work must complete before the blocked/background test is released, and in real-model runs it may wait only for the current bounded model step, not the complete rebuild.

- [ ] **Step 5: Audit the final diff**

Run `git diff --check`, list every modified file, and inspect that model constants and `inference_general.yaml` have no diff. Confirm no unrelated dirty files were staged or overwritten.

- [ ] **Step 6: Write the verification report**

Record exact commands, environment, current commit plus dirty-diff note, test results, measured timings, known limitations, and the deferred follow-up changes from the design. Do not claim geometry accuracy improvement from this stability change.
