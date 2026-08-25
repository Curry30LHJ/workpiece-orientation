# Signed Geometry Boundary Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ] syntax) for tracking.

**Goal:** Make geometry-rule boundary offsets signed, preserve old profile behavior, and prevent front/back preview overlays from sharing stale state.

**Architecture:** GeometryCalibrator fits raw contours and derives an effective boundary only for masks and previews. GeometryMaskProfiles normalizes rule-level semantics and migrates unmarked legacy values. The existing TCP command carries the signed value, and the Qt dialog renders effective_shape while retaining raw diagnostics.

**Tech Stack:** Python 3.10, OpenCV, NumPy, pytest, JSON-over-TCP, Qt 5.14.2 Widgets, C++17, QtTest, Windows PowerShell.

## Global Constraints

- Work in E:/Project/wang/pp_813 and preserve unrelated dirty files and generated outputs.
- margin_ratio is signed in [-0.94, 0.94]: positive expands the fitted boundary, negative contracts it, and zero is unchanged.
- Scaling changes only radii, axes, or rectangle half-sizes; center and rotation remain unchanged.
- inside masks the effective boundary interior; outside masks its complement.
- Candidate ranking uses fixed internal search tolerances, never margin_ratio.
- Persist raw geometry and margin_semantics = "signed_boundary_v2"; compute effective geometry at runtime.
- Unmarked legacy inside value m maps to +m; unmarked legacy outside value m maps to -m. Missing legacy value uses 0.02. New signed rules use 0.0.
- Unknown markers and negative unmarked legacy values are rejected.
- Front/back state is independent. needs_reseed blocks use of old geometry but allows a new seed preview.
- Keep schema version 1 readable and do not modify PP-ShiTuV2, ALIKED, LightGlue, feature dimensions, model weights, fusion thresholds, or training.

## File Responsibility Map

- src/geometry_calibration.py: signed validation, raw/effective shapes, mask generation, and fixed-tolerance fitting.
- src/geometry_mask_profiles.py: marker normalization, legacy migration, and persistence.
- src/orientation_tcp_service.py: signed preview request range checking.
- qt_app/geometrymaskmanager.cpp: signed control, new-rule defaults, and preview routing.
- qt_app/geometryrulecanvas.cpp: effective overlay selection and stale-overlay clearing.
- src/orientation_classifier.py: refuse to build an active geometry mask for a direction containing needs_reseed and fall back to raw templates.
- tests/test_geometry_calibration.py, tests/test_geometry_mask_profiles.py, tests/test_orientation_tcp_service.py: Python regression coverage.
- qt_app/tests/test_geometrymaskmanager.cpp, qt_app/tests/test_geometryrulecanvas.cpp: Qt regression coverage.
- docs/verification/signed-boundary-offset-results.md: final evidence record.

---

### Task 1: Signed calibration and raw/effective diagnostics

**Files:**
- Modify: src/geometry_calibration.py
- Test: tests/test_geometry_calibration.py

**Interfaces:**
- Add _offset_shape(shape, margin_ratio) -> dict[str, Any].
- Keep GeometryCalibrator.fit_reference and fit signatures; return raw fitted_shape, effective_shape, and margin_ratio.
- Keep profile_patch.geometry as raw object-relative geometry.

- [ ] **Step 1: Write failing mask monotonicity tests**

Import _shape_mask and add:

~~~python
def test_signed_inside_and_outside_masks_use_one_boundary_formula():
    shape = {"shape": "ellipse", "cx": 80.0, "cy": 60.0,
             "rx": 40.0, "ry": 30.0, "angle_deg": 17.0}
    inside = [_shape_mask(shape, (140, 180), "inside", value)
              for value in (-0.20, 0.0, 0.20)]
    outside = [_shape_mask(shape, (140, 180), "outside", value)
               for value in (-0.20, 0.0, 0.20)]

    assert np.count_nonzero(inside[0]) < np.count_nonzero(inside[1]) < np.count_nonzero(inside[2])
    assert np.count_nonzero(outside[0]) > np.count_nonzero(outside[1]) > np.count_nonzero(outside[2])
~~~

- [ ] **Step 2: Write failing raw/effective and candidate-invariance tests**

Use _synthetic_ellipse and add:

~~~python
def test_fit_keeps_raw_shape_and_scales_only_effective_dimensions():
    profile = _ellipse_profile(mode="inside")
    profile["rules"][0]["margin_ratio"] = -0.10
    rule = GeometryCalibrator().fit(_synthetic_ellipse(), profile)["rules"][0]

    assert rule["effective_shape"]["rx"] < rule["fitted_shape"]["rx"]
    assert rule["effective_shape"]["ry"] < rule["fitted_shape"]["ry"]
    assert rule["effective_shape"]["cx"] == pytest.approx(rule["fitted_shape"]["cx"])
    assert rule["effective_shape"]["cy"] == pytest.approx(rule["fitted_shape"]["cy"])
    assert rule["effective_shape"]["angle_deg"] == pytest.approx(rule["fitted_shape"]["angle_deg"])


def test_fit_reference_candidates_do_not_change_with_margin_sign():
    image = _synthetic_ellipse(center=(168, 104), axes=(78, 57), angle=21.0)
    seed = {"shape": "circle", "cx": 168.0, "cy": 104.0, "r": 32.0}
    negative = GeometryCalibrator().fit_reference(image, seed, mode="inside", margin_ratio=-0.02)
    positive = GeometryCalibrator().fit_reference(image, seed, mode="inside", margin_ratio=0.02)

    assert negative["anchor_fit"]["candidates"] == positive["anchor_fit"]["candidates"]
    assert negative["rule_fit"]["candidates"] == positive["rule_fit"]["candidates"]
    assert negative["rule_fit"]["selected_candidate_index"] == positive["rule_fit"]["selected_candidate_index"]
~~~

- [ ] **Step 3: Write failing signed-range tests**

Accept -0.94, 0, and 0.94; reject -0.95 and 0.95 with the exact message margin_ratio must be in [-0.94, 0.94]. Assert fit_reference returns top-level margin_ratio and rule_fit.effective_shape.

- [ ] **Step 4: Run the red tests**

~~~powershell
E:/python/anaconda3/envs/shitu/python.exe -m pytest tests/test_geometry_calibration.py -k "signed or effective_shape or margin_sign" -q --basetemp=pytest-signed-calibration-red
~~~

Expected: failure because negative values are rejected, outside uses the old inverse factor, and effective fields do not exist.

- [ ] **Step 5: Implement the signed helper and mask usage**

In src/geometry_calibration.py add:

~~~python
def _offset_shape(shape: Mapping[str, Any], margin_ratio: float) -> dict[str, Any]:
    margin = _finite(margin_ratio, "margin_ratio")
    if margin < -0.94 or margin > 0.94:
        raise GeometryCalibrationError("margin_ratio must be in [-0.94, 0.94]")
    result = dict(shape)
    factor = 1.0 + margin
    if shape["shape"] == "rotated_rectangle":
        result["half_width"] = max(1.0, float(shape["half_width"]) * factor)
        result["half_height"] = max(1.0, float(shape["half_height"]) * factor)
    else:
        result["rx"] = max(1.0, float(shape["rx"]) * factor)
        result["ry"] = max(1.0, float(shape["ry"]) * factor)
    return result
~~~

Make _shape_mask draw _offset_shape for both modes and invert only after drawing for outside. Change all calibration range checks to [-0.94, 0.94]. Keep fixed ANCHOR_SEARCH_MARGIN_RATIO and RULE_SEARCH_BAND_RATIO as candidate tolerances.

- [ ] **Step 6: Add diagnostics and run green**

In _fit_rule return fitted_shape, effective_shape, margin_ratio, and ignore_mask. In fit_reference add rule_fit.effective_shape, top-level margin_ratio, and profile_patch.margin_semantics = "signed_boundary_v2"; keep profile_patch.geometry raw.

~~~powershell
E:/python/anaconda3/envs/shitu/python.exe -m pytest tests/test_geometry_calibration.py -q --basetemp=pytest-signed-calibration-green
~~~

Commit only the Task 1 files with message feat: apply signed geometry boundary offsets.

### Task 2: Profile marker and legacy migration

**Files:**
- Modify: src/geometry_mask_profiles.py
- Test: tests/test_geometry_mask_profiles.py

**Interfaces:**
- normalize_geometry_profile returns canonical signed rules with margin_semantics.
- GeometryMaskProfiles.preview_rule remains stateless and passes raw/effective preview fields through.

- [ ] **Step 1: Write failing migration tests**

Add tests that assert:
  - signed_boundary_v2 with -0.02 stays -0.02;
  - unmarked inside 0.02 becomes +0.02;
  - unmarked outside 0.02 becomes -0.02;
  - unmarked missing margin becomes +0.02 for inside;
  - marked missing margin becomes 0.0;
  - unknown marker and unmarked negative value raise InvalidGeometryProfileError;
  - save_draft writes margin_semantics to the returned draft and profile.json.

- [ ] **Step 2: Run red migration tests**

~~~powershell
E:/python/anaconda3/envs/shitu/python.exe -m pytest tests/test_geometry_mask_profiles.py -k "signed or legacy_margin or margin_semantics" -q --basetemp=pytest-signed-profiles-red
~~~

Expected: failure because the existing canonicalizer rejects negative margins and drops the marker.

- [ ] **Step 3: Implement the canonical migration branch**

In _canonical_rule use:

~~~python
SIGNED_MARGIN_SEMANTICS = "signed_boundary_v2"
marker = value.get("margin_semantics")
if marker is None:
    raw = _finite(value.get("margin_ratio", 0.02), f"{field}.margin_ratio")
    if raw < 0 or raw >= 0.95:
        raise InvalidGeometryProfileError(f"{field}.margin_ratio must be in [0, 0.95) for legacy rules")
    margin_ratio = raw if mode == "inside" else -raw
elif marker == SIGNED_MARGIN_SEMANTICS:
    margin_ratio = _finite(value.get("margin_ratio", 0.0), f"{field}.margin_ratio")
    if margin_ratio < -0.94 or margin_ratio > 0.94:
        raise InvalidGeometryProfileError(f"{field}.margin_ratio must be in [-0.94, 0.94]")
else:
    raise InvalidGeometryProfileError(f"{field}.margin_semantics is unsupported")
~~~

Return margin_semantics with every canonical rule. Do not bump PROFILE_SCHEMA_VERSION. Existing anchor, seed, review, enabled, and revision logic remains unchanged.

- [ ] **Step 4: Verify source-preserving migration and preview**

Add a test that loads an old unmarked profile, confirms snapshot returns the migrated outside value -0.02, and confirms the source profile bytes are unchanged until save_draft. Ensure preview_rule does not mutate draft or active profile and preserves calibrator raw/effective fields.

- [ ] **Step 5: Run green and commit**

~~~powershell
E:/python/anaconda3/envs/shitu/python.exe -m pytest tests/test_geometry_mask_profiles.py -q --basetemp=pytest-signed-profiles-green
~~~

Expected: all profile tests pass, including stale revisions, draft idempotency, review gates, and corruption isolation. Commit with message feat: migrate geometry margins to signed semantics.

### Task 3: TCP signed preview contract

**Files:**
- Modify: src/orientation_tcp_service.py
- Test: tests/test_orientation_tcp_service.py

- [ ] **Step 1: Add red request tests**

Extend the existing preview dispatch test with margin_ratio=-0.02 and assert direction, template_id, signed margin_ratio, and rule_fit.effective_shape. Add -0.95 and 0.95 to invalid requests.

- [ ] **Step 2: Run red TCP tests**

~~~powershell
E:/python/anaconda3/envs/shitu/python.exe -m pytest tests/test_orientation_tcp_service.py -k "preview_geometry_mask_rule" -q --basetemp=pytest-signed-tcp-red
~~~

Expected: valid negative request is rejected by the old [0, 0.95) guard.

- [ ] **Step 3: Change only the dispatcher range guard**

Use:

~~~python
if not math.isfinite(margin_value) or margin_value < -0.94 or margin_value > 0.94:
    return self._error(request_id, "INVALID_REQUEST", "margin_ratio must be in [-0.94, 0.94]")
~~~

Keep command names, request ids, stale-revision handling, and response envelopes unchanged. A low-confidence preview returns its explicit status; it never reuses a prior template overlay.

- [ ] **Step 4: Run green and commit**

~~~powershell
E:/python/anaconda3/envs/shitu/python.exe -m pytest tests/test_orientation_tcp_service.py -q --basetemp=pytest-signed-tcp-green
~~~

Expected: all TCP tests pass. Commit with message feat: accept signed geometry preview offsets.

### Task 4: Qt signed control and effective overlay

**Files:**
- Modify: qt_app/geometrymaskmanager.cpp
- Modify: qt_app/geometryrulecanvas.cpp
- Test: qt_app/tests/test_geometrymaskmanager.cpp
- Test: qt_app/tests/test_geometryrulecanvas.cpp

- [ ] **Step 1: Add red Qt tests**

Add a test that finds a widget named marginSpinBox, expects minimum -94, maximum 94, and accepts -2, 0, and 2. Add a test that clicks addRuleButton and expects margin_ratio 0.0 and margin_semantics signed_boundary_v2. Add a canvas test with distinct fitted_shape and effective_shape radii and expect fitShape() to report the effective radius.

- [ ] **Step 2: Run red Qt tests in an isolated build directory**

~~~powershell
$build = 'E:/Project/wang/pp_813/qt_app/build-test-signed-margin-red'
New-Item -ItemType Directory -Force -Path $build | Out-Null
Set-Location -LiteralPath $build
& 'E:/QT/5.14/5.14.2/msvc2017_64/bin/qmake.exe' '../tests/test_geometrymaskmanager.pro' CONFIG+=release
& 'E:/QT/Tools/QtCreator/bin/jom.exe'
& './release/test_geometrymaskmanager.exe' -platform offscreen
~~~

Expected: failure because the current range is 0..94, new rules default to 0.02, and the canvas uses fitted_shape only.

- [ ] **Step 3: Implement the signed widget**

Set marginSpinBox objectName, range -94..94, default 0, suffix %, and tooltip “负值向内收缩，正值向外扩张；仅影响最终忽略边界。” Rename the form label to 边界偏移. Make addRule write margin_ratio 0.0 and marker signed_boundary_v2. Make setCurrentRuleFromEditor write both fields and make loadCurrentRuleIntoEditor use signed fallback 0.0.

- [ ] **Step 4: Implement effective overlay selection**

In GeometryRuleCanvas::setFitOverlay choose effective_shape, then fitted_shape, then the legacy shape object. Keep setImage and mouse-press stale-overlay clearing. Pass the complete rule_fit object from setRulePreview so diagnostics still show both raw and effective shapes. Keep direction/template guards and candidate clearing.

- [ ] **Step 5: Run green Qt tests and commit**

Build and run both test projects in qt_app/build-test-signed-margin-green and qt_app/build-test-signed-margin-canvas-green using qmake, jom, and -platform offscreen. Expected: all listed Qt tests pass. Commit the four Task 4 files with message feat: show signed geometry offsets in Qt editor.

### Task 5: Compatibility and final verification

**Files:**
- Test: tests/test_geometry_calibration.py
- Test: tests/test_geometry_mask_profiles.py
- Test: tests/test_orientation_tcp_service.py
- Modify: src/orientation_classifier.py
- Test: tests/test_orientation_classifier.py
- Test: tests/test_workpiece_catalog.py
- Modify: docs/verification/signed-boundary-offset-results.md

- [ ] **Step 1: Add compatibility regressions**

Add an old unmarked outside 0.02 fixture and assert canonical -0.02. Add front +0.02/back -0.02 rules and assert normalization and effective shapes remain independent. In OrientationClassifier.prepare_geometry_cache, detect any needs_reseed rule for a direction before fitting; return that direction's raw global/local templates unchanged and report status needs_reseed, so stale geometry cannot create an active mask. Add a test asserting raw fallback until a successful new preview patch changes the rule state to ready.

Use this direction-level guard before the existing per-template fitting loop:

~~~python
if any(rule.get("editor_state") == "needs_reseed"
       for rule in direction.get("rules", [])):
    candidate_globals[label] = raw_global[label]
    candidate_locals[label] = raw_local[label]
    geometry_indices[label] = list(range(len(paths)))
    report[label] = [
        {"index": index, "template_id": f"{label}:{Path(path).name}",
         "status": "needs_reseed", "review_state": "included"}
        for index, path in enumerate(paths)
    ]
    if progress_callback is not None:
        for index in range(len(paths)):
            progress_callback(label, index + 1, len(paths))
    continue
~~~

- [ ] **Step 2: Run focused Python regression**

~~~powershell
E:/python/anaconda3/envs/shitu/python.exe -m pytest tests/test_geometry_calibration.py tests/test_geometry_mask_profiles.py tests/test_orientation_tcp_service.py tests/test_orientation_classifier.py tests/test_workpiece_catalog.py -q --basetemp=pytest-signed-focused
~~~

Expected: all focused tests pass; only expectations that explicitly describe the old outside-positive interpretation may change, and they must retain a legacy migration assertion.

- [ ] **Step 3: Build the main Qt release without overwriting a running executable**

Run scripts/build_qt5.ps1 in the normal release directory only when it is not locked; otherwise use a separate build-release-signed directory. Expected output is a successful Qt 5.14.2 release executable.

- [ ] **Step 4: Run full Python suite and whitespace check**

~~~powershell
E:/python/anaconda3/envs/shitu/python.exe -m pytest tests -q --basetemp=pytest-signed-full
git diff --check
~~~

Expected: full Python suite passes and no whitespace errors occur in task files.

- [ ] **Step 5: Write and commit verification evidence**

Create docs/verification/signed-boundary-offset-results.md listing implementation commits, exact commands/results, signed mask monotonicity, old-profile migration, front/back isolation, Qt offscreen results, and confirmation that PP-ShiTuV2/ALIKED/LightGlue/fusion thresholds were untouched. Commit with message docs: record signed geometry offset verification.

## Plan Self-Review

- Spec coverage: Task 1 covers signed math and candidate invariance; Task 2 covers migration/defaults; Task 3 covers TCP; Task 4 covers Qt input and effective rendering; Task 5 covers front/back isolation, needs_reseed, old libraries, and full regression.
- Placeholder scan: no unfinished-task wording or unspecified implementation step is present.
- Type consistency: _offset_shape returns a JSON-compatible shape consumed by masks, fit diagnostics, TCP responses, and Qt overlays; margin_semantics is always signed_boundary_v2 after normalization.
- Safety: no model, matcher, threshold, or training code is touched; unrelated dirty worktree changes are preserved.
