# Geometry Fusion Regression Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make geometry publication blockers represent true paired regressions, keep synthetic geometry repairs out of ALIKED/LightGlue descriptors, and show actionable blocker evidence in Qt.

**Architecture:** Extend the existing leave-one-out validator to predict each held-out sample against both the raw and candidate caches. Keep PP-ShiTu on geometry-processed images, but extract local features from original images and filter them with fitted ignore masks. Preserve blocker diagnostics in a dedicated Qt state so per-template fit preview cannot erase them.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PaddleClas PP-ShiTuV2, ALIKED/LightGlue, pytest, Qt 5.14.2, Qt Test.

## Global Constraints

- Do not modify PP-ShiTu, ALIKED, or LightGlue model files.
- Do not change `GLOBAL_MARGIN_THRESHOLD`, `LOCAL_MIN_SCORE`, `LOCAL_MIN_MARGIN`, or `LOCAL_OVERRIDE_MARGIN`.
- Extract query local features from the original image exactly once per prediction.
- Preserve existing job JSON and old workpiece-library compatibility.
- Do not retrain a model.

---

### Task 1: Paired leave-one-out regression semantics

**Files:**
- Modify: `tests/test_orientation_classifier.py`
- Modify: `src/orientation_classifier.py:602`

**Interfaces:**
- Consumes: `TemplateCache.raw_global_vectors`, `TemplateCache.raw_local_features`, and `_cache_without_template`.
- Produces: `leave_one_out_report(record, candidate_cache)` with transition counters and branch evidence while retaining `correct_to_wrong` and `changed_predictions`.

- [ ] **Step 1: Write a failing test for a pre-existing baseline error**

Create a deterministic classifier double where both raw and candidate predictions are wrong for one held-out sample. Assert `correct_to_wrong == 0`, `wrong_to_wrong == 1`, and no changed prediction is emitted.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_orientation_classifier.py -k "paired_leave_one_out" -q`

Expected: FAIL because the current validator counts every candidate error as `correct_to_wrong`.

- [ ] **Step 3: Write a failing test for a real correct-to-wrong transition**

Provide literal raw and candidate results for the same held-out sample. Assert that the report contains `baseline_predicted`, `candidate_predicted`, branch predictions, margins, and `cause == "geometry_local_override"`.

- [ ] **Step 4: Implement paired prediction minimally**

For each held-out candidate cache, build the corresponding raw held-out cache, run `_predict_baseline` before `_predict_geometry`, classify the transition, and populate evidence fields. Keep exclusion and incomplete-report behavior unchanged.

- [ ] **Step 5: Run focused and classifier tests GREEN**

Run: `python -m pytest tests/test_orientation_classifier.py -q`

Expected: all tests pass.

### Task 2: Raw-image local feature masking

**Files:**
- Modify: `tests/test_orientation_classifier.py`
- Modify: `src/orientation_classifier.py`

**Interfaces:**
- Consumes: raw template local-feature caches, fitted template/query ignore masks, and original query images.
- Produces: geometry candidate caches and query local features derived from original images with all ignored regions filtered out.

- [ ] **Step 1: Write a failing template-feature test**

Build a template with raw local keypoints inside and outside a fitted ignored region. Assert `prepare_geometry_cache` filters `raw_local_features` with the complete ignore mask and does not use features re-extracted from the repaired image.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_orientation_classifier.py -k "geometry_local_features" -q`

Expected: FAIL because the current implementation extracts template local descriptors from the repaired image and only filters outside-mode masks.

- [ ] **Step 3: Write a failing query-feature test**

Assert `_predict_geometry` extracts local features once from the original query image, filters the same feature dictionary independently for front and back, and never extracts local features from processed query variants.

- [ ] **Step 4: Implement the minimal local processing fix**

Make `geometry_feature_mask` return the complete fitted ignore mask. In candidate-cache creation, filter each raw template feature dictionary. In query prediction, extract raw local features once and filter them with each direction's fitted mask. Keep global embedding processing and `_fuse_scores` unchanged.

- [ ] **Step 5: Run classifier and profile tests GREEN**

Run: `python -m pytest tests/test_orientation_classifier.py tests/test_geometry_mask_profiles.py -q`

Expected: all tests pass.

### Task 3: Backend blocker payload and compatibility

**Files:**
- Modify: `tests/test_geometry_mask_profiles.py`
- Modify: `src/geometry_mask_profiles.py:1000`

**Interfaces:**
- Consumes: enriched `regression.changed_predictions` from Task 1.
- Produces: `geometry_fusion_regression` blocker with changed template IDs and evidence; accepts legacy reports using `geometry_regression` fallback data.

- [ ] **Step 1: Write a failing blocker-payload test**

Assert that a real paired regression produces a blocker whose `templates` contains `front:22.png` and whose evidence is preserved. Assert that an absolute candidate error with `correct_to_wrong == 0` does not block.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_geometry_mask_profiles.py -k "fusion_regression" -q`

Expected: FAIL because the current blocker lacks template IDs and the new code.

- [ ] **Step 3: Implement compatible blocker construction**

Build `templates` from `changed_predictions[].template_id`, retain `correct_to_wrong`, and copy the evidence. Keep publication behavior for legacy stored jobs unchanged.

- [ ] **Step 4: Run profile tests GREEN**

Run: `python -m pytest tests/test_geometry_mask_profiles.py -q`

Expected: all tests pass.

### Task 4: Qt blocker explanation and diagnostic persistence

**Files:**
- Modify: `qt_app/tests/test_geometrymaskmanager.cpp`
- Modify: `qt_app/geometrymaskmanager.h`
- Modify: `qt_app/geometrymaskmanager.cpp:736`

**Interfaces:**
- Consumes: `blocking_issues[].templates` and enriched `changed_predictions`.
- Produces: persistent blocker summary text and regression-specific row tooltip/status styling.

- [ ] **Step 1: Write a failing Qt test**

Feed a completed validation job containing a `geometry_fusion_regression` blocker for `front:22.png`. Assert the matching row is highlighted, its tooltip explains global/local conflict, and diagnostics still contain the blocker after selecting the row.

- [ ] **Step 2: Build and verify RED**

Run the existing Qt test build for `test_geometrymaskmanager`, then execute the test binary.

Expected: FAIL because `changed_predictions` are not added to blocked template IDs and template preview replaces the diagnostic text.

- [ ] **Step 3: Implement minimal Qt state separation**

Store the validation blocker summary separately from the current template fit text. Compose both sections when the selected template changes. Add regression template IDs to row highlighting and use a fusion-specific tooltip.

- [ ] **Step 4: Build and run Qt tests GREEN**

Run the same Qt test target.

Expected: all tests pass.

### Task 5: Full verification and real-data acceptance

**Files:**
- Create: `docs/verification/geometry-fusion-regression-fix-results.md`

**Interfaces:**
- Consumes: completed backend and Qt changes.
- Produces: reproducible test and 28+28 leave-one-out evidence.

- [ ] **Step 1: Run the Python regression suite**

Run: `python -m pytest tests/test_orientation_classifier.py tests/test_geometry_mask_profiles.py tests/test_orientation_tcp_service.py tests/test_workpiece_catalog.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run the existing Qt geometry manager tests**

Expected: all tests pass under Qt 5.14.2.

- [ ] **Step 3: Run the real 28+28 paired leave-one-out check**

Use workpiece `31f082d1a04e486b9345846f4d585033`, draft revision 11, and the existing model environment. Record baseline accuracy, candidate accuracy, true correct-to-wrong count, and changed template evidence.

Expected: baseline 56/56, candidate 56/56, `correct_to_wrong == 0`.

- [ ] **Step 4: Document results**

Record exact commands, elapsed time, test totals, and whether the current rule can be published after revalidation.
