# Workpiece Lifecycle, Confirmed Ingestion, and Interference Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add recoverable workpiece deletion, explicit asynchronous confirmed-template ingestion, and correspondence-based template-side interference masks without changing existing recognition models or thresholds.

**Architecture:** Keep `WorkpieceLibrary` as the durable image/manifest store and add a catalog façade that owns lifecycle and cache publication. Add a small evolution service with a durable job ledger, a single worker, immutable revision checks, and atomic cache swaps. Expose the domain operations through additive TCP commands and focused Qt controls; local mask filtering remains at the ALIKED/LightGlue template-feature seam.

**Tech Stack:** Python 3, `threading`, JSON manifests, OpenCV/Pillow decoding already used by the repository, pytest, Qt 5.14.2/QTest, existing JSON-lines TCP protocol.

## Global Constraints

- Do not change PP-ShiTuV2, ALIKED, LightGlue weights, model input size, or existing fusion thresholds.
- Existing protocol commands and version-1 workpiece manifests remain compatible.
- Every mutating request is idempotent by caller-supplied `operation_id`.
- Prediction uses the last stable immutable cache while a background job builds a candidate.
- A mask is template-side local-feature filtering only; absence of a correspondence is never treated as confirmed absence.
- Use TDD: each production behavior is preceded by a failing test and verified independently.

### Task 1: Workpiece catalog lifecycle

**Files:**
- Create: `src/workpiece_catalog.py`
- Modify: `src/workpiece_library.py`
- Modify: `src/orientation_classifier.py`
- Test: `tests/test_workpiece_catalog.py`

- [ ] Write tests for recycle, restore, purge, case-insensitive restore conflict, revision guard, legacy manifest recovery, and stale-worker cache publication.
- [ ] Run the new tests and verify they fail because the catalog interface is absent.
- [ ] Implement a locked catalog façade with `snapshot`, `recycle`, `list_recycled`, `restore`, `purge`, and compare-and-commit methods. Move active directories atomically into a dedicated recycle directory, keep IDs and revisions, and remove/publish classifier caches only inside the façade.
- [ ] Add additive revision/state metadata while retaining version-1 manifest loading.
- [ ] Run catalog tests and existing library/classifier tests.

### Task 2: Durable template evolution jobs

**Files:**
- Create: `src/template_evolution.py`
- Modify: `src/workpiece_library.py`
- Modify: `src/orientation_classifier.py`
- Test: `tests/test_template_evolution.py`

- [ ] Write tests for explicit front/back confirmation, exact duplicate rejection, operation-id replay, three-second coalescing with an injected clock, restart recovery, queued/building/cancel/retry/review transitions, stale revision rejection, and old-cache preservation after build failure.
- [ ] Run the tests to observe the expected missing-service failures.
- [ ] Implement durable staged image copies, compact job records, a single worker, immutable job snapshots, and an append/commit path that preserves the workpiece ID/name and atomically swaps the candidate cache.
- [ ] Keep prediction callable during builds and persist building jobs as queued on restart.
- [ ] Run evolution tests plus existing orientation/library/TCP tests.

### Task 3: Interference groups and local propagation

**Files:**
- Create: `src/interference_masks.py`
- Modify: `src/aliked_lightglue_matcher.py`
- Modify: `src/soft_center_matcher.py`
- Modify: `src/template_evolution.py`
- Test: `tests/test_interference_masks.py`

- [ ] Write deterministic tests for native-coordinate rectangle round-trip, two-source geometric consensus, disagreement/review state, safety rejection, rejected-proposal replay prevention, and filtered template keypoints.
- [ ] Run the tests and verify the missing mask seams fail.
- [ ] Implement version-2 annotation metadata, trusted groups, correspondence projection through the existing matcher adapter, high-confidence activation, low-confidence review, and template-side feature filtering. Preserve raw features for future propagation and keep query/global features unchanged.
- [ ] Run mask tests and the full Python suite available in the current environment.

### Task 4: TCP adapters and compatibility

**Files:**
- Modify: `src/orientation_tcp_service.py`
- Test: `tests/test_orientation_tcp_service.py`
- Test: `tests/test_orientation_service_integration.py`

- [ ] Write dispatcher tests for lifecycle, confirmation, job listing/actions, annotation get/save, idempotency, stable errors, and legacy command compatibility.
- [ ] Implement additive dispatch commands and progress/job snapshots while retaining the existing loading handshake and one-client behavior.
- [ ] Run TCP and integration tests, excluding only tests that require unavailable third-party model assets.

### Task 5: Qt controls

**Files:**
- Create: `qt_app/annotationeditor.h`, `qt_app/annotationeditor.cpp`
- Modify: `qt_app/backendclient.h`, `qt_app/backendclient.cpp`
- Modify: `qt_app/mainwindow.h`, `qt_app/mainwindow.cpp`, `qt_app/mainwindow.ui`
- Test: `qt_app/tests/test_backendclient.cpp`, `qt_app/tests/test_mainwindow.cpp`

- [ ] Write QTest cases for result-context invalidation, delete confirmation, front/back/no-add actions for single and batch results, job polling/actions, annotation coordinate mapping, provenance/review display, and disabled controls while a request is pending.
- [ ] Run the tests to confirm the new controls/signals are absent.
- [ ] Add additive BackendClient methods and focused annotation/job/delete widgets; bind controls to immutable result context and display progress, elapsed time, actual template counts, and review diagnostics.
- [ ] Run the Qt build and all existing Qt tests in offscreen mode.

### Task 6: Verification and documentation

**Files:**
- Modify: `docs/verification/workpiece-orientation-desktop-ui-checklist.md`
- Create: `docs/verification/workpiece-lifecycle-learning-mask-results.md`

- [ ] Run `git diff --check`, Python compilation, Python tests, and Qt tests.
- [ ] Record unavailable model-dependent tests explicitly and document cache/job timing observations.
- [ ] Review the diff for accidental changes to model configuration or fusion thresholds.

