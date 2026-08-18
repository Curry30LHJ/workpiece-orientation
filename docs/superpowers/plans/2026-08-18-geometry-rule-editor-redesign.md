# Geometry Rule Editor Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inaccurate bounding-box drawing workflow with shape-specific gestures, stateless automatic reference fitting, per-template validation, and safe draft/publish behavior while preserving the existing recognition models and thresholds.

**Architecture:** Keep `GeometryCalibrator` as the single Python geometry engine and add one read-only preview path that converts a native-pixel seed into the existing object-relative profile format. Keep profile persistence and publication in `GeometryMaskProfiles`, expose preview through the existing TCP dispatcher, and let the Qt dialog own only local editing/history/view state. Published prediction remains all-or-none per query and atomically swaps template caches.

**Tech Stack:** Python 3.10, OpenCV, NumPy, pytest, PaddlePaddle/PP-ShiTuV2, ALIKED/LightGlue, Qt 5.14.2 Widgets, C++17, QtTest, JSON-over-TCP, Windows PowerShell.

## Global Constraints

- Work only in `E:\Project\wang\pp_813` on the current feature branch; preserve unrelated dirty-worktree changes and generated files.
- The three supported tools are circle, ellipse, and rotated rectangle; none is selected by default.
- Do not add freehand curves or polygons.
- Store production rules in object-relative coordinates; native image pixels are used only for the reference preview request and UI display.
- Do not modify PP-ShiTuV2, ALIKED, LightGlue, feature dimensions, model weights, fusion thresholds, or the training workflow.
- Positive template counts remain arbitrary and unequal; validation and prediction must not assume 5+5.
- If either orientation cannot fit every enabled rule for a query, both orientations use the unmasked baseline for that query and return `needs_review: true`.
- Old 5+5 libraries and old active interference caches remain usable until a new geometry revision is published.
- New fields are additive and absence-tolerant; existing geometry profile schema version 1 files must load unchanged.
- Use test-driven changes and commit only the files named by each task.

---

## File Responsibility Map

- `src/geometry_calibration.py`: rank shape candidates, infer an outer anchor, convert pixel seeds to object-relative geometry, and return serializable diagnostics.
- `src/geometry_mask_profiles.py`: validate additive editor fields, locate reference templates, execute stateless preview, enforce review/exclusion publication gates, and restore legacy cache state on rollback.
- `src/workpiece_catalog.py`: restore active legacy annotation caches when no geometry revision is active and provide an atomic legacy rollback boundary.
- `src/orientation_tcp_service.py`: validate and dispatch the new read-only `preview_geometry_mask_rule` request.
- `src/orientation_classifier.py`: exclude explicitly excluded templates from a geometry candidate cache and preserve all-or-none query fallback.
- `qt_app/geometryrulecanvas.h/.cpp`: own shape gestures, handles, coordinate transforms, zoom/pan, overlays, and gesture completion.
- `qt_app/geometrymaskmanager.h/.cpp`: own reference selection, draft history, numeric editing, preview candidates, template review state, validation table, and unsaved-change handling.
- `qt_app/mainwindow.h/.cpp`: bridge dialog preview requests and TCP responses without changing the backend client's single-request contract.
- Python and Qt test files: pin every compatibility, interaction, and publication rule before implementation.

### Task 1: Ranked Automatic Reference Fitting

**Files:**
- Modify: `src/geometry_calibration.py`
- Test: `tests/test_geometry_calibration.py`

**Interfaces:**
- Consumes: native BGR image, native-pixel `seed_shape`, `mode`, `margin_ratio`, and optional candidate indices.
- Produces: `GeometryCalibrator.fit_reference(image, seed_shape, *, mode, margin_ratio, anchor_candidate_index=None, rule_candidate_index=None) -> dict[str, Any]`.
- Preserves: `GeometryCalibrator.fit(image, direction_profile) -> dict[str, Any]`.

- [ ] **Step 1: Add failing tests for automatic anchor and ranked rule candidates**

```python
def test_fit_reference_returns_object_geometry_and_ranked_candidates():
    image = _synthetic_ellipse(center=(168, 104), axes=(78, 57), angle=21.0)
    seed = {"shape": "circle", "cx": 168.0, "cy": 104.0, "r": 32.0,
            "angle_deg": 0.0}

    result = GeometryCalibrator().fit_reference(
        image, seed, mode="inside", margin_ratio=0.02
    )

    assert result["status"] == "active"
    assert result["anchor_fit"]["selected_candidate_index"] == 0
    assert 1 <= len(result["anchor_fit"]["candidates"]) <= 3
    assert 1 <= len(result["rule_fit"]["candidates"]) <= 3
    assert result["profile_patch"]["anchor"]["mode"] == "auto"
    assert result["profile_patch"]["geometry"]["r"] > 0
    assert result["fit_duration_ms"] >= 0


def test_fit_reference_honors_selected_candidate_without_mutating_seed():
    image = _synthetic_ellipse()
    seed = {"shape": "circle", "cx": 155.0, "cy": 92.0, "r": 31.0}
    first = GeometryCalibrator().fit_reference(image, seed, mode="inside", margin_ratio=0.02)
    if len(first["rule_fit"]["candidates"]) < 2:
        pytest.skip("synthetic image exposes one valid rule candidate")
    second = GeometryCalibrator().fit_reference(
        image, seed, mode="inside", margin_ratio=0.02, rule_candidate_index=1
    )
    assert second["rule_fit"]["selected_candidate_index"] == 1
    assert seed == {"shape": "circle", "cx": 155.0, "cy": 92.0, "r": 31.0}
```

- [ ] **Step 2: Run the new tests and verify the missing API failure**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_geometry_calibration.py -k "fit_reference" -q --basetemp=pytest-plan-red-reference
```

Expected: FAIL with `AttributeError: 'GeometryCalibrator' object has no attribute 'fit_reference'`.

- [ ] **Step 3: Refactor candidate generation to return a stable ranked list**

Add an internal helper with a deterministic score order and a three-candidate cap:

```python
def _rank_shape_candidates(
    image: np.ndarray,
    expected: Mapping[str, Any],
    image_shape: tuple[int, int],
    margin_ratio: float,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for contour in _contours(image):
        candidate = _candidate_for_shape(contour, str(expected["shape"]))
        if candidate is None:
            continue
        score, support, visible, residual = _candidate_quality(
            candidate, expected, image_shape, margin_ratio
        )
        if support < MIN_EDGE_SUPPORT or visible < MIN_VISIBLE_RATIO:
            continue
        if residual > MAX_FIT_RESIDUAL_RATIO:
            continue
        value = dict(candidate)
        value.update(score=float(score), edge_support=float(support),
                     visible_ratio=float(visible), fit_residual=float(residual))
        ranked.append((float(score), value))
    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked[:limit]]
```

Change `_fit_shape` to return the first ranked candidate so existing `fit()` behavior remains stable.

- [ ] **Step 4: Add pixel-seed normalization and automatic outer-anchor selection**

Implement the public method and keep NumPy contour arrays out of the response:

```python
def fit_reference(self, image, seed_shape, *, mode, margin_ratio,
                  anchor_candidate_index=None, rule_candidate_index=None):
    started = time.perf_counter()
    normalized_seed = _native_seed_shape(seed_shape, image.shape[:2])
    anchor_candidates = _rank_outer_anchor_candidates(image, normalized_seed, limit=3)
    anchor = _select_candidate(anchor_candidates, anchor_candidate_index, "anchor")
    rule_candidates = _rank_shape_candidates(
        image, normalized_seed, image.shape[:2], RULE_SEARCH_BAND_RATIO, limit=3
    )
    fitted_rule = _select_candidate(rule_candidates, rule_candidate_index, "rule")
    geometry = _object_relative_geometry(normalized_seed, anchor)
    return {
        "status": "active",
        "anchor_fit": _candidate_report(anchor_candidates, anchor_candidate_index),
        "rule_fit": _candidate_report(rule_candidates, rule_candidate_index),
        "profile_patch": {
            "anchor": _normalized_anchor(anchor, image.shape[:2], mode="auto"),
            "geometry": geometry,
            "seed_geometry": deepcopy(geometry),
        },
        "fit_duration_ms": (time.perf_counter() - started) * 1000.0,
    }
```

`_rank_outer_anchor_candidates` must score centered, large, closed contours across ellipse/circle and rotated-rectangle families; ties are resolved by score and then contour area. Invalid candidate indices raise `GeometryCalibrationError` with `anchor_candidate_index out of range` or `rule_candidate_index out of range`.

- [ ] **Step 5: Extend normal `fit()` reports without changing masks**

Return `fitted_shape`, the public candidate list, `selected_candidate_index`, and `fit_duration_ms` for each fitted rule. Keep `ignore_mask` and existing scalar diagnostics unchanged so classifier consumers remain compatible:

```python
public.update({
    "fitted_shape": self._public_shape(fitted),
    "candidates": [_public_candidate(item) for item in candidates],
    "selected_candidate_index": selected_index,
    "fit_duration_ms": (time.perf_counter() - started) * 1000.0,
})
```

- [ ] **Step 6: Run calibration tests**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_geometry_calibration.py -q --basetemp=pytest-plan-green-reference
```

Expected: all tests in `tests/test_geometry_calibration.py` PASS.

- [ ] **Step 7: Commit the calculation boundary**

```powershell
git add -- src\geometry_calibration.py tests\test_geometry_calibration.py
git commit -m "feat: add ranked geometry reference fitting"
```

### Task 2: Additive Profile Metadata and Stateless Preview

**Files:**
- Modify: `src/geometry_mask_profiles.py`
- Test: `tests/test_geometry_mask_profiles.py`

**Interfaces:**
- Consumes: Task 1 `GeometryCalibrator.fit_reference`.
- Produces: `GeometryMaskProfiles.preview_rule(workpiece_id, *, expected_library_revision, direction, template_id, seed_shape, mode, margin_ratio, anchor_candidate_index=None, rule_candidate_index=None) -> dict[str, Any]`.
- Produces additive profile fields: `reference_template`, `anchor.mode`, `rule.seed_geometry`, `rule.editor_state`, and `template_reviews`.

- [ ] **Step 1: Add failing normalization tests for editor metadata**

```python
def test_profile_preserves_reference_seed_and_template_reviews():
    profile = circle_profile()
    front = profile["directions"]["front"]
    front["reference_template"] = {
        "template_id": "front:00.png", "direction": "front",
        "width": 512, "height": 512,
    }
    front["anchor"]["mode"] = "auto"
    front["rules"][0]["seed_geometry"] = dict(front["rules"][0]["geometry"])
    front["rules"][0]["editor_state"] = "ready"
    front["template_reviews"] = {
        "front:03.png": {"state": "excluded", "reason": "边缘遮挡严重"}
    }

    normalized = normalize_geometry_profile(profile)

    assert normalized["directions"]["front"]["reference_template"]["width"] == 512
    assert normalized["directions"]["front"]["anchor"]["mode"] == "auto"
    assert normalized["directions"]["front"]["rules"][0]["seed_geometry"]
    assert normalized["directions"]["front"]["rules"][0]["editor_state"] == "ready"
    assert normalized["directions"]["front"]["template_reviews"]["front:03.png"]["state"] == "excluded"
```

Add parameterized rejections for an empty exclusion reason, a cross-direction template id, an unsupported review state, an unsupported `editor_state`, and `editor_state == "needs_reseed"` with `enabled == true`.

- [ ] **Step 2: Add failing stateless preview tests**

Add `from unittest.mock import Mock` to `tests/test_geometry_mask_profiles.py`, then add:

```python
def test_preview_rule_checks_revision_and_does_not_mutate_profile(tmp_path: Path):
    catalog, _, record = make_geometry_catalog(tmp_path)
    calibrator = Mock()
    calibrator.fit_reference.return_value = {
        "status": "active",
        "anchor_fit": {"candidates": [], "selected_candidate_index": 0},
        "rule_fit": {"candidates": [], "selected_candidate_index": 0},
        "profile_patch": {
            "anchor": {"shape": "ellipse", "mode": "auto",
                       "coarse": {"cx": 0.5, "cy": 0.5, "rx": 0.4,
                                  "ry": 0.4, "angle_deg": 0.0}},
            "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5},
            "seed_geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5},
        },
        "fit_duration_ms": 1.0,
    }
    profiles = GeometryMaskProfiles(
        catalog, calibrator=calibrator, start_worker=False,
        storage_dir=tmp_path / "jobs",
    )
    before = profiles.snapshot(record.id)

    preview = profiles.preview_rule(
        record.id,
        expected_library_revision=record.revision,
        direction="front",
        template_id="front:00.png",
        seed_shape={"shape": "circle", "cx": 128.0, "cy": 128.0, "r": 50.0},
        mode="inside",
        margin_ratio=0.02,
    )

    assert preview["template_id"] == "front:00.png"
    assert preview["profile_patch"]["reference_template"]["direction"] == "front"
    assert profiles.snapshot(record.id) == before
```

Also assert stale revision, missing template, wrong direction, unreadable image, and out-of-range candidate selection produce stable domain errors.

- [ ] **Step 3: Run the new profile tests and verify failure**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_geometry_mask_profiles.py -k "reference_seed or preview_rule or template_reviews" -q --basetemp=pytest-plan-red-profile-preview
```

Expected: FAIL because the new fields are dropped and `preview_rule` is missing.

- [ ] **Step 4: Canonicalize additive fields without bumping schema version**

Implement strict helpers:

```python
SUPPORTED_REVIEW_STATES = {"included", "review", "excluded"}

def _canonical_template_reviews(value: Any, direction: str, field: str) -> dict[str, dict[str, str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidGeometryProfileError(f"{field} must be an object")
    result = {}
    for template_id, review in value.items():
        if not str(template_id).startswith(f"{direction}:"):
            raise InvalidGeometryProfileError(f"{field} contains a template from another direction")
        state = _non_empty_text(review.get("state"), f"{field}.{template_id}.state")
        reason = str(review.get("reason", "")).strip()
        if state not in SUPPORTED_REVIEW_STATES:
            raise InvalidGeometryProfileError(f"unsupported review state: {state}")
        if state != "included" and not reason:
            raise InvalidGeometryProfileError("review and excluded templates require a reason")
        result[str(template_id)] = {"state": state, "reason": reason}
    return result
```

Preserve absent fields with current defaults: manual anchor semantics, `seed_geometry == geometry`, `editor_state == "ready"`, no reference template, and every template included.

- [ ] **Step 5: Implement reference-template lookup and preview**

Build the stable template id using the same `front:<filename>` / `back:<filename>` convention used by `snapshot()`. Read only a path belonging to the requested record and direction. Merge Task 1 output with:

```python
preview["workpiece_id"] = workpiece_id
preview["direction"] = direction
preview["template_id"] = template_id
preview["profile_patch"]["reference_template"] = {
    "template_id": template_id,
    "direction": direction,
    "width": int(image.shape[1]),
    "height": int(image.shape[0]),
}
```

Do not write `profile.json`, revision files, job files, manifests, or runtime caches.

- [ ] **Step 6: Run profile tests**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_geometry_mask_profiles.py -q --basetemp=pytest-plan-green-profile-preview
```

Expected: all tests in `tests/test_geometry_mask_profiles.py` PASS.

- [ ] **Step 7: Commit metadata and preview orchestration**

```powershell
git add -- src\geometry_mask_profiles.py tests\test_geometry_mask_profiles.py
git commit -m "feat: preview geometry rules on reference templates"
```

### Task 3: Template Review, Exclusion, and Validation Reports

**Files:**
- Modify: `src/orientation_classifier.py`
- Modify: `src/geometry_mask_profiles.py`
- Test: `tests/test_orientation_classifier.py`
- Test: `tests/test_geometry_mask_profiles.py`

**Interfaces:**
- Consumes: canonical `template_reviews` from Task 2 and candidate-rich fit reports from Task 1.
- Produces: validation job `blocking_issues`, exclusion warnings, per-template candidate diagnostics, and geometry candidate caches containing only included templates.

- [ ] **Step 1: Add failing classifier tests for explicit exclusion**

```python
def test_geometry_cache_excludes_template_only_from_candidate_side(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(2)]
    back = [write_marker(tmp_path / "back-0.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    profile = geometry_profile()
    profile["directions"]["front"]["template_reviews"] = {
        "front:front-1.png": {"state": "excluded", "reason": "轮廓被遮挡"}
    }

    candidate, report = classifier.prepare_geometry_cache(
        "m7", record, profile, FakeGeometryCalibrator(), None
    )

    assert candidate.global_vectors["front"].shape[0] == len(record.front_images) - 1
    assert candidate.raw_global_vectors["front"].shape[0] == len(record.front_images)
    assert report["front"][1]["review_state"] == "excluded"
```

Add a failure case where all front templates are excluded.

- [ ] **Step 2: Add failing publication-gate tests**

```python
def test_review_blocks_publish_but_exclusion_requires_override(tmp_path: Path):
    profiles, record = _completed_profile_job(tmp_path, review_state="review")
    job = profiles.get_job("job-1")
    assert job["blocking_issues"][0]["code"] == "template_needs_review"
    with pytest.raises(GeometryProfilePublishError, match="blocking issues"):
        profiles.publish(record.id, "job-1", expected_library_revision=record.revision,
                         expected_draft_revision=1, operation_id="publish-review",
                         override_reason="已检查")
```

For `excluded`, assert `blocking_issues` is empty, `warnings` contains `template_excluded`, and an empty override reason is rejected.

- [ ] **Step 3: Run targeted tests and verify failure**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py tests\test_geometry_mask_profiles.py -k "excludes_template or review_blocks_publish or exclusion_requires_override" -q --basetemp=pytest-plan-red-template-review
```

Expected: FAIL because review state is not consumed.

- [ ] **Step 4: Filter candidate cache inputs by stable template id**

Inside `prepare_geometry_cache`, enumerate stored images with their stable ids. For `excluded`, keep the image in raw vectors/features and validation report but omit its masked vector/features from the active candidate side. For `review`, run fitting and set `review_state` while leaving the candidate data present. Raise `GeometryValidationError("front has no included templates")` or the back equivalent if a side becomes empty.

```python
review = direction.get("template_reviews", {}).get(template_id, {})
state = review.get("state", "included")
if state == "excluded":
    report[label].append({"template_id": template_id, "review_state": state,
                          "review_reason": review["reason"], "status": "excluded"})
    continue
embedding, features, fit_report = self._prepare_geometry_template(image, direction, calibrator)
candidate_embeddings[label].append(embedding)
candidate_features[label].append(features)
fit_report.update(template_id=template_id, review_state=state,
                  review_reason=review.get("reason", ""))
report[label].append(fit_report)
```

- [ ] **Step 5: Separate blocking issues from overridable warnings**

Build job fields with deterministic codes:

```python
job["blocking_issues"] = [
    {"code": "template_needs_review", "orientation": label,
     "template_id": template_id, "reason": reason}
    for ... if state == "review"
]
job["warnings"].extend(
    {"code": "template_excluded", "orientation": label,
     "template_id": template_id, "reason": reason}
    for ... if state == "excluded"
)
```

`publish()` rejects non-empty `blocking_issues` unconditionally. Existing warnings/regressions continue to require a non-empty override reason.

- [ ] **Step 6: Include candidate diagnostics in persisted reports**

Strip NumPy masks and arrays, retain `candidates`, `selected_candidate_index`, `fit_duration_ms`, `edge_support`, `visible_ratio`, and `fit_residual`. Keep the existing `progress.completed/total/phase` fields.

```python
def _serializable_geometry_fit(fit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in fit.items()
        if key not in {"ignore_mask", "contour"}
    }
```

- [ ] **Step 7: Run classifier and profile suites**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py tests\test_geometry_mask_profiles.py -q --basetemp=pytest-plan-green-template-review
```

Expected: both test files PASS.

- [ ] **Step 8: Commit template review behavior**

```powershell
git add -- src\orientation_classifier.py src\geometry_mask_profiles.py tests\test_orientation_classifier.py tests\test_geometry_mask_profiles.py
git commit -m "feat: review and exclude geometry templates safely"
```

### Task 4: Legacy Cache Continuity and Rollback

**Files:**
- Modify: `src/workpiece_catalog.py`
- Modify: `src/geometry_mask_profiles.py`
- Test: `tests/test_workpiece_catalog.py`
- Test: `tests/test_geometry_mask_profiles.py`
- Test: `tests/test_orientation_classifier.py`

**Interfaces:**
- Consumes: existing `active_interference_groups`, base cache raw features, and geometry revision pointers.
- Produces: `WorkpieceCatalog.restore_legacy_annotation_cache(workpiece_id, *, expected_revision, operation_id) -> WorkpieceRecord` and revision payload `previous_source: "legacy" | "geometry" | None`.

- [ ] **Step 1: Add failing recovery coverage for active legacy groups**

```python
def active_legacy_group():
    return {
        "group_id": "legacy", "name": "旧标注", "enabled": True,
        "propagation": {"state": "active"},
        "annotations": [{
            "orientation": "front", "index": 0, "status": "active",
            "regions": [{"x": 0, "y": 0, "width": 2, "height": 2}],
        }],
    }


def test_recover_applies_active_legacy_groups_when_geometry_manager_exists(tmp_path: Path):
    catalog, _, record = create_catalog(tmp_path)
    group = active_legacy_group()
    catalog.commit_annotation_document(
        record.id, [group], expected_revision=record.revision,
        operation_id="legacy-active", active_groups=[group],
    )
    restarted_classifier = LegacyMaskFakeClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), restarted_classifier)
    profiles = GeometryMaskProfiles(catalog, start_worker=False,
                                    storage_dir=tmp_path / "geometry-jobs")
    catalog.set_geometry_profiles(profiles)

    catalog.recover()

    cache = restarted_classifier.get_template_cache(record.id)
    assert cache.ignored_regions
    assert cache.geometry_profile is None
```

Define this fake in `tests/test_workpiece_catalog.py` so the test observes whether catalog recovery invoked the legacy path:

```python
class LegacyMaskFakeClassifier(FakeClassifier):
    def prepare_template_masks(self, workpiece_id, ignored_regions):
        base = self.caches[workpiece_id]
        candidate = TemplateCache(
            global_vectors=base.global_vectors,
            local_features=base.local_features,
            raw_global_vectors=base.raw_global_vectors or base.global_vectors,
            raw_local_features=base.raw_local_features or base.local_features,
            ignored_regions=ignored_regions,
        )
        return candidate, {}
```

Add this helper to `tests/test_geometry_mask_profiles.py` because test modules do not import each other:

```python
def active_legacy_group():
    return {
        "group_id": "legacy", "name": "旧标注", "enabled": True,
        "propagation": {"state": "active"},
        "annotations": [{
            "orientation": "front", "index": 0, "status": "active",
            "regions": [{"x": 0, "y": 0, "width": 2, "height": 2}],
        }],
    }
```

This test must reproduce the current branch behavior where `geometry_profiles is not None` skips active legacy masks.

- [ ] **Step 2: Add failing first-publish rollback coverage**

Extend `GeometryFakeClassifier` in `tests/test_geometry_mask_profiles.py` with:

```python
def prepare_template_masks(self, workpiece_id, ignored_regions):
    base = self.caches[workpiece_id]
    candidate = TemplateCache(
        global_vectors=base.global_vectors,
        local_features=base.local_features,
        raw_global_vectors=base.raw_global_vectors or base.global_vectors,
        raw_local_features=base.raw_local_features or base.local_features,
        ignored_regions=ignored_regions,
    )
    return candidate, {}
```

Then add:

```python
def test_first_geometry_publish_can_rollback_to_legacy_cache(tmp_path: Path):
    catalog, classifier, record = make_geometry_catalog(tmp_path)
    group = active_legacy_group()
    catalog.commit_annotation_document(
        record.id, [group], expected_revision=record.revision,
        operation_id="legacy-active", active_groups=[group],
    )
    record = catalog.get(record.id)
    profiles = GeometryMaskProfiles(catalog, calibrator=object(), start_worker=False,
                                    storage_dir=tmp_path / "jobs")
    draft = profiles.save_draft(
        record.id, circle_profile(), expected_library_revision=record.revision,
        expected_draft_revision=0, operation_id="draft-1",
    )
    job = profiles.start_validation(
        record.id, expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"], operation_id="validate-1",
    )
    profiles.run_next(force=True)
    published = profiles.publish(record.id, job["job_id"],
                                 expected_library_revision=record.revision,
                                 expected_draft_revision=draft["draft_revision"],
                                 operation_id="publish-geometry")
    assert published["active_revision"] == 1

    rolled = profiles.rollback(record.id,
                               expected_library_revision=record.revision + 1,
                               operation_id="rollback-legacy")
    assert rolled["active_revision"] is None
    assert classifier.get_template_cache(record.id).ignored_regions
```

- [ ] **Step 3: Pin all-or-none query fallback**

Add a classifier test whose front fit is active and back fit is low confidence. Assert global/local embedding extraction uses the raw image once, no masked side scores are fused, `geometry_mask.status == "low_confidence"`, and `needs_review is True`.

```python
def test_geometry_prediction_falls_back_both_sides_when_back_fit_fails(classifier, tmp_path):
    front = [write_marker(tmp_path / "front.png", 1)]
    back = [write_marker(tmp_path / "back.png", 2)]
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    classifier.geometry_calibrator = FakeGeometryCalibrator()
    candidate, _ = classifier.prepare_geometry_cache(
        "m7", SimpleNamespace(front_images=tuple(front), back_images=tuple(back)),
        geometry_profile(fail_marker=3), FakeGeometryCalibrator(),
    )
    classifier.set_template_cache("m7", candidate)
    result = classifier.predict("m7", write_marker(tmp_path / "query.png", 3))
    assert result["geometry_mask"]["status"] == "low_confidence"
    assert result["needs_review"] is True
    assert classifier.global_predictor.markers[-1] == 3
    assert classifier.extractor.markers[-1] == 3
```

- [ ] **Step 4: Run targeted tests and verify failures**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_workpiece_catalog.py tests\test_geometry_mask_profiles.py tests\test_orientation_classifier.py -k "legacy or all_or_none" -q --basetemp=pytest-plan-red-legacy
```

Expected: legacy recovery/rollback tests FAIL; all-or-none test may already PASS and becomes a regression guard.

- [ ] **Step 5: Centralize legacy cache preparation in the catalog**

Add a private helper that reads `get_annotation_document(record.id)["active_groups"]`, builds the active mask map, and calls `prepare_template_masks`. Use it from `recover()`, template append recovery, and `restore_legacy_annotation_cache`. When an active geometry revision exists, geometry still takes precedence.

```python
def _prepare_legacy_annotation_cache(self, record):
    document = self.library.get_annotation_document(record.id)
    groups = document["active_groups"]
    if not groups:
        return None
    masks = build_active_mask_map(len(record.front_images), len(record.back_images), groups)
    candidate, _ = self.classifier.prepare_template_masks(record.id, masks)
    return candidate
```

- [ ] **Step 6: Record and restore the predecessor source**

On first geometry publication, set revision payload `previous_source` to `"legacy"` when active legacy groups exist and no geometry revision is active; otherwise use `"geometry"` or `None`. In `rollback()`, if `previous_active_revision is None` and the current revision payload says `previous_source == "legacy"`, clear geometry pointers and atomically swap the catalog cache back to the prepared legacy candidate.

Do not delete `interference_groups` or `active_interference_groups` from the manifest.

```python
if previous_revision is None and payload.get("previous_source") == "legacy":
    record = self.catalog.restore_legacy_annotation_cache(
        workpiece_id,
        expected_revision=expected_library_revision,
        operation_id=operation_id,
    )
    document.update(active_revision=None, previous_active_revision=None, active=None)
    _atomic_write_json(profile_path, document)
    return self._snapshot_for_record(record)
```

- [ ] **Step 7: Run compatibility suites**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_workpiece_catalog.py tests\test_geometry_mask_profiles.py tests\test_orientation_classifier.py -q --basetemp=pytest-plan-green-legacy
```

Expected: all three test files PASS.

- [ ] **Step 8: Commit legacy continuity**

```powershell
git add -- src\workpiece_catalog.py src\geometry_mask_profiles.py tests\test_workpiece_catalog.py tests\test_geometry_mask_profiles.py tests\test_orientation_classifier.py
git commit -m "fix: preserve legacy masks across geometry publication"
```

### Task 5: Read-Only TCP Preview Command

**Files:**
- Modify: `src/orientation_tcp_service.py`
- Test: `tests/test_orientation_tcp_service.py`

**Interfaces:**
- Consumes: Task 2 `GeometryMaskProfiles.preview_rule`.
- Produces: JSON command `preview_geometry_mask_rule` with response key `preview`.

- [ ] **Step 1: Add a failing dispatcher contract test**

```python
def test_preview_geometry_mask_rule_dispatches_without_operation_id(runtime):
    request = {
        "request_id": "preview-1",
        "command": "preview_geometry_mask_rule",
        "workpiece_id": "m7",
        "base_library_revision": 4,
        "direction": "front",
        "template_id": "front:00.png",
        "seed_shape": {"shape": "circle", "cx": 120.0, "cy": 130.0, "r": 44.0},
        "mode": "inside",
        "margin_ratio": 0.02,
    }

    response = OrientationCommandDispatcher(runtime).dispatch(request)

    assert response["ok"] is True
    assert response["preview"]["template_id"] == "front:00.png"
    runtime.geometry_profiles.preview_rule.assert_called_once()
```

Add invalid-request cases for missing revision, invalid direction, empty template id, invalid shape object, invalid mode, invalid margin, and non-integer candidate index.

- [ ] **Step 2: Run the TCP test and verify unsupported-command failure**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py -k "preview_geometry_mask_rule" -q --basetemp=pytest-plan-red-preview-tcp
```

Expected: FAIL with the service's unknown-command response.

- [ ] **Step 3: Add strict request validation and dispatch**

Add `preview_geometry_mask_rule` to the geometry command set before the operation-id branch. Call:

```python
preview = profiles.preview_rule(
    workpiece_id,
    expected_library_revision=base_library_revision,
    direction=direction,
    template_id=template_id,
    seed_shape=seed_shape,
    mode=mode,
    margin_ratio=margin_ratio,
    anchor_candidate_index=anchor_candidate_index,
    rule_candidate_index=rule_candidate_index,
)
return self._response(request_id, ok=True, preview=preview)
```

The command must not require or accept an `operation_id` for state mutation.

- [ ] **Step 4: Run the complete TCP service suite**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py -q --basetemp=pytest-plan-green-preview-tcp
```

Expected: all tests in `tests/test_orientation_tcp_service.py` PASS.

- [ ] **Step 5: Commit the protocol seam**

```powershell
git add -- src\orientation_tcp_service.py tests\test_orientation_tcp_service.py
git commit -m "feat: expose geometry rule preview command"
```

### Task 6: Shape-Specific Qt Canvas Gestures

**Files:**
- Modify: `qt_app/geometryrulecanvas.h`
- Modify: `qt_app/geometryrulecanvas.cpp`
- Modify: `qt_app/tests/test_geometryrulecanvas.cpp`

**Interfaces:**
- Produces: `Tool::None`, `resetView()`, `cancelGesture()`, `setNumericShape(const QJsonObject &)`, and `shapeCommitted(const QJsonObject &)`.
- Preserves: `setImage`, `setTool`, `setCoarseShape`, `setFitOverlay`, `coarseShape`, and `shapeChanged`.

- [ ] **Step 1: Replace old circle expectations with center-radius tests**

```cpp
void TestGeometryRuleCanvas::circleUsesPressPointAsCenter() {
    GeometryRuleCanvas canvas;
    canvas.resize(600, 400);
    canvas.setImage(QImage(300, 300, QImage::Format_RGB32));
    canvas.setTool(GeometryRuleCanvas::Circle);
    QSignalSpy committed(&canvas, &GeometryRuleCanvas::shapeCommitted);
    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(300, 200));
    QTest::mouseRelease(&canvas, Qt::LeftButton, {}, QPoint(400, 200));
    const QJsonObject shape = canvas.coarseShape();
    QVERIFY(qAbs(shape.value("cx").toDouble() - 150.0) < 1.0);
    QVERIFY(qAbs(shape.value("cy").toDouble() - 150.0) < 1.0);
    QVERIFY(qAbs(shape.value("r").toDouble() - 75.0) < 1.0);
    QCOMPARE(committed.count(), 1);
}
```

Add tests for no drawing under `Tool::None`, ellipse center-to-corner axes, rotated-rectangle center-to-corner dimensions, and `Esc` cancelling an incomplete gesture.

- [ ] **Step 2: Add zoom/pan/native-coordinate regression tests**

Draw the same native point before and after wheel zoom, middle-button pan, and `R`. Assert `imagePoint` behavior through the resulting shape JSON, not private implementation details.

```cpp
void TestGeometryRuleCanvas::resetViewRestoresNativeDrawingCoordinates() {
    GeometryRuleCanvas canvas;
    canvas.resize(400, 400);
    canvas.setImage(QImage(200, 200, QImage::Format_RGB32));
    canvas.setTool(GeometryRuleCanvas::Circle);
    QWheelEvent zoomEvent(QPointF(200, 200), QPointF(200, 200), QPoint(), QPoint(0, 120),
                          Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(&canvas, &zoomEvent);
    QTest::keyClick(&canvas, Qt::Key_R);
    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(200, 200));
    QTest::mouseRelease(&canvas, Qt::LeftButton, {}, QPoint(300, 200));
    QVERIFY(qAbs(canvas.coarseShape().value("cx").toDouble() - 100.0) < 1.0);
}
```

- [ ] **Step 3: Run the Qt canvas test and verify old behavior fails**

Build and run:

```powershell
Set-Location E:\Project\wang\pp_813\qt_app\tests
qmake test_geometryrulecanvas.pro -o Makefile.geometryrulecanvas -spec win32-msvc
nmake /F Makefile.geometryrulecanvas.Release
.\release\test_geometryrulecanvas.exe -platform offscreen
```

Expected: the center-radius and `Tool::None` tests FAIL.

- [ ] **Step 4: Add explicit gesture and handle states**

Use internal states with one responsibility:

```cpp
enum class Interaction {
    Idle, Drawing, MovingCenter, ResizingPrimary,
    ResizingSecondary, Rotating, Panning
};
```

Set the initial tool to `None`. Circle press fixes center and drag length sets radius. Ellipse and rectangle press fix center and drag deltas set both axes. After completion, hit testing selects center/axis/rotation handles. Emit `shapeChanged` during preview and exactly one `shapeCommitted` on release.

- [ ] **Step 5: Add a reversible view transform**

Store `zoom_` in `[0.25, 8.0]` and `pan_` in widget pixels. `imageTarget()` incorporates both; `imagePoint()` applies the inverse transform and clamps to the native image bounds. `resetView()` restores fit-to-widget; `R` calls it. Wheel zoom anchors the native point under the cursor.

```cpp
QPointF GeometryRuleCanvas::imagePoint(const QPointF &widgetPoint) const {
    const QRectF target = imageTarget();
    const qreal scale = target.width() / qreal(image_.width());
    return QPointF(qBound(0.0, (widgetPoint.x() - target.left()) / scale,
                           qreal(image_.width())),
                   qBound(0.0, (widgetPoint.y() - target.top()) / scale,
                           qreal(image_.height())));
}
```

- [ ] **Step 6: Draw handles and layered overlays**

Keep cyan dashed coarse shape and green solid fit shape. Draw center, axis, and rotation handles only for the selected coarse shape. Draw the orange mask overlay below both outlines. `setNumericShape` validates positive dimensions and updates without synthesizing mouse events.

```cpp
painter.drawImage(imageTarget(), image_);
drawIgnoreOverlay(&painter, fitShape_, ignoreMode_, marginRatio_);
drawShape(&painter, fitShape_, QPen(QColor(35, 165, 75), 2.0));
drawShape(&painter, coarseShape_, QPen(QColor(0, 170, 200), 2.0, Qt::DashLine));
drawHandles(&painter, coarseShape_);
```

- [ ] **Step 7: Run Qt canvas tests**

Run:

```powershell
Set-Location E:\Project\wang\pp_813\qt_app\tests
qmake test_geometryrulecanvas.pro -o Makefile.geometryrulecanvas -spec win32-msvc
nmake /F Makefile.geometryrulecanvas.Release
.\release\test_geometryrulecanvas.exe -platform offscreen
```

Expected: `test_geometryrulecanvas.exe` exits with code 0.

- [ ] **Step 8: Commit the canvas interaction**

```powershell
git add -- qt_app\geometryrulecanvas.h qt_app\geometryrulecanvas.cpp qt_app\tests\test_geometryrulecanvas.cpp
git commit -m "feat: add precise geometry canvas gestures"
```

### Task 7: Qt Draft, Preview, Candidate, and Review Management

**Files:**
- Modify: `qt_app/geometrymaskmanager.h`
- Modify: `qt_app/geometrymaskmanager.cpp`
- Modify: `qt_app/tests/test_geometrymaskmanager.cpp`

**Interfaces:**
- Consumes: Task 6 `shapeCommitted` and Task 5 preview response shape.
- Produces: `previewRequested(const QJsonObject &request)`, `setRulePreview(const QJsonObject &preview)`, `setPreviewBusy(bool busy)`, local undo/redo stacks, dirty state, template review controls, and a per-template validation table.

- [ ] **Step 1: Add failing preview-request and no-default-tool tests**

```cpp
void TestGeometryMaskManager::committedSeedRequestsStatelessPreview() {
    GeometryMaskManagerDialog dialog;
    QJsonObject snapshot = profileSnapshot(4, 0, 0);
    snapshot.insert("templates", QJsonArray{QJsonObject{{"template_id", "front:front.png"},
        {"direction", "front"}, {"path", QStringLiteral("C:/fixture/front.png")}}});
    dialog.setSnapshot(snapshot);
    QSignalSpy preview(&dialog, &GeometryMaskManagerDialog::previewRequested);
    auto *shape = dialog.findChild<QComboBox *>(QStringLiteral("shapeCombo"));
    shape->setCurrentIndex(shape->findData(QStringLiteral("circle")));
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QMetaObject::invokeMethod(canvas, "shapeCommitted", Qt::DirectConnection,
                              Q_ARG(QJsonObject, QJsonObject{{"shape", "circle"},
                                  {"cx", 40.0}, {"cy", 30.0}, {"r", 20.0}}));
    QCOMPARE(preview.count(), 1);
    QCOMPARE(preview.takeFirst().at(0).toJsonObject().value("template_id").toString(),
             QStringLiteral("front:front.png"));
}
```

Assert the shape combo starts on a placeholder with empty data and the canvas tool is `None`.

- [ ] **Step 2: Add failing draft-history and template-review tests**

Cover: shape commit marks dirty; undo restores the prior JSON; redo reapplies it; `setSnapshot` clears dirty/history; reload restores the saved draft; active-version mode is read-only; copying active to draft marks dirty; copying a rule to the other direction creates a disabled `needs_reseed` rule with a new id; `review`/`excluded` require a reason; saving emits the complete draft; excluded rows remain visible in the validation table.

```cpp
void TestGeometryMaskManager::copiedRuleRequiresTargetReseed() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(snapshotWithReadyFrontRule());
    dialog.findChild<QPushButton *>(QStringLiteral("copyRuleButton"))->click();
    const QJsonArray backRules = dialog.draft().value("directions").toObject()
        .value("back").toObject().value("rules").toArray();
    QCOMPARE(backRules.size(), 1);
    QCOMPARE(backRules.at(0).toObject().value("editor_state").toString(),
             QStringLiteral("needs_reseed"));
    QVERIFY(!backRules.at(0).toObject().value("enabled").toBool());
}
```

Define `snapshotWithReadyFrontRule()` next to the existing `profileSnapshot()` helper:

```cpp
QJsonObject snapshotWithReadyFrontRule() {
    QJsonObject snapshot = profileSnapshot(4, 1, 0);
    QJsonObject draft = snapshot.value("draft").toObject();
    QJsonObject directions = draft.value("directions").toObject();
    QJsonObject front = directions.value("front").toObject();
    front.insert("anchor", QJsonObject{{"shape", "ellipse"}, {"mode", "auto"},
        {"coarse", QJsonObject{{"cx", 0.5}, {"cy", 0.5}, {"rx", 0.4},
                               {"ry", 0.4}, {"angle_deg", 0.0}}}});
    front.insert("rules", QJsonArray{QJsonObject{{"rule_id", "front-rule"},
        {"name", "中心反光"}, {"shape", "circle"},
        {"geometry", QJsonObject{{"cx", 0.0}, {"cy", 0.0}, {"r", 0.5}}},
        {"seed_geometry", QJsonObject{{"cx", 0.0}, {"cy", 0.0}, {"r", 0.5}}},
        {"mode", "inside"}, {"margin_ratio", 0.02}, {"enabled", true},
        {"editor_state", "ready"}}});
    directions.insert("front", front);
    draft.insert("directions", directions);
    snapshot.insert("draft", draft);
    return snapshot;
}
```

- [ ] **Step 3: Add failing preview-overlay and candidate tests**

Call `setRulePreview` with two anchor/rule candidates. Assert candidate combos show both, the returned `profile_patch.anchor`, `reference_template`, `geometry`, and `seed_geometry` enter the current draft, and selecting candidate 1 emits a new preview request with `rule_candidate_index: 1`.

```cpp
QJsonObject preview = oneRulePreviewWithTwoCandidates();
dialog.setRulePreview(preview);
auto *ruleCandidates = dialog.findChild<QComboBox *>(QStringLiteral("ruleCandidateCombo"));
QCOMPARE(ruleCandidates->count(), 2);
QSignalSpy previewSpy(&dialog, &GeometryMaskManagerDialog::previewRequested);
ruleCandidates->setCurrentIndex(1);
QCOMPARE(previewSpy.count(), 1);
QCOMPARE(previewSpy.takeFirst().at(0).toJsonObject()
             .value("rule_candidate_index").toInt(), 1);
```

Define the helper in `test_geometrymaskmanager.cpp`:

```cpp
QJsonObject oneRulePreviewWithTwoCandidates() {
    const QJsonObject first{{"shape", "circle"}, {"cx", 40.0}, {"cy", 30.0}, {"r", 18.0}};
    const QJsonObject second{{"shape", "circle"}, {"cx", 41.0}, {"cy", 30.0}, {"r", 20.0}};
    return QJsonObject{{"direction", "front"}, {"template_id", "front:front.png"},
        {"anchor_fit", QJsonObject{{"candidates", QJsonArray{first}},
                                    {"selected_candidate_index", 0}}},
        {"rule_fit", QJsonObject{{"candidates", QJsonArray{first, second}},
                                  {"selected_candidate_index", 0},
                                  {"fitted_shape", first}}},
        {"profile_patch", QJsonObject{
            {"reference_template", QJsonObject{{"template_id", "front:front.png"},
                {"direction", "front"}, {"width", 80}, {"height", 60}}},
            {"anchor", QJsonObject{{"shape", "ellipse"}, {"mode", "auto"},
                {"coarse", QJsonObject{{"cx", 0.5}, {"cy", 0.5},
                    {"rx", 0.4}, {"ry", 0.4}, {"angle_deg", 0.0}}}}},
            {"geometry", QJsonObject{{"cx", 0.0}, {"cy", 0.0}, {"r", 0.5}}},
            {"seed_geometry", QJsonObject{{"cx", 0.0}, {"cy", 0.0}, {"r", 0.5}}}}}};
}
```

- [ ] **Step 4: Run manager tests and verify failures**

Build and run:

```powershell
Set-Location E:\Project\wang\pp_813\qt_app\tests
qmake test_geometrymaskmanager.pro -o Makefile.geometrymaskmanager -spec win32-msvc
nmake /F Makefile.geometrymaskmanager.Release
.\release\test_geometrymaskmanager.exe -platform offscreen
```

Expected: new signal, history, candidate, and review tests FAIL.

- [ ] **Step 5: Rebuild the editor controls without changing dialog ownership**

Add object-named controls:

```text
shapeCombo, centerXSpin, centerYSpin, radiusSpin,
axisXSpin, axisYSpin, rotationSpin,
anchorCandidateCombo, ruleCandidateCombo,
undoButton, redoButton, resetRuleButton,
reloadDraftButton, copyRuleButton, enabledCheckBox,
versionCombo, copyActiveToDraftButton,
manualAnchorButton, dirtyLabel, validationTable,
reviewStateCombo, reviewReasonEdit
```

Keep the generic `inside/outside` combo. Remove `setAnchorButton` from the normal workflow; show `manualAnchorButton` only after preview reports automatic-anchor failure. `setPreviewBusy(true)` disables only preview-dependent controls and leaves close/cancel available; it must not clear the canvas or draft.

- [ ] **Step 6: Implement local draft snapshots and dirty-state rules**

Use bounded JSON snapshot histories rather than command subclasses:

```cpp
void GeometryMaskManagerDialog::applyDraftMutation(const QJsonObject &next) {
    if (next == draft_) return;
    undoHistory_.append(draft_);
    if (undoHistory_.size() > 100) undoHistory_.removeFirst();
    redoHistory_.clear();
    draft_ = next;
    dirty_ = true;
    refreshEditor();
}
```

Every user mutation passes through this method. `setSnapshot` resets the baseline. Saving does not clear dirty until the backend returns a new snapshot. Override `closeEvent` to ask before discarding a dirty draft.

`reloadDraftButton` restores `snapshot_["draft"]` after discard confirmation. `versionCombo == "active"` renders `snapshot_["active"]` as a read-only layer and disables mutating controls. `copyActiveToDraftButton` copies the active profile through `applyDraftMutation` and returns to draft mode; it never edits `snapshot_["active"]` in place.

- [ ] **Step 7: Implement preview request/response flow**

On `shapeCommitted`, emit one request containing `base_library_revision`, direction, stable template id, native-pixel seed, mode, margin, and current candidate indices. `setRulePreview` applies only a response matching the current direction/template/rule, updates the draft profile patch, fills candidate combos, calls `canvas_->setFitOverlay`, and renders diagnostics.

When automatic anchor fitting fails and the operator clicks `manualAnchorButton`, enter a one-shot `manualAnchorCapture_` state. The next committed shape is converted with the existing native-pixel-to-normalized anchor conversion, stored as `anchor.mode = "manual"`, and is not written into the rule geometry. The following committed shape resumes normal rule editing. Add a Qt test that verifies this two-gesture fallback and that the fallback button remains hidden after a successful automatic preview.

```cpp
QJsonObject request{
    {QStringLiteral("base_library_revision"), snapshot_.value("library_revision")},
    {QStringLiteral("direction"), direction()},
    {QStringLiteral("template_id"), templateCombo_->currentData(Qt::UserRole + 1)},
    {QStringLiteral("seed_shape"), shape},
    {QStringLiteral("mode"), modeCombo_->currentData()},
    {QStringLiteral("margin_ratio"), marginSpin_->value() / 100.0},
};
emit previewRequested(request);
```

`copyRuleButton` copies only name, shape, mode, margin, and provisional geometry to the other direction with a new UUID, `enabled = false`, and `editor_state = "needs_reseed"`. The copied rule is shown as “待重画”; the first successful target-side preview replaces its geometry and sets `editor_state = "ready"`. The operator must then explicitly enable it. Add a Qt test proving the copied rule cannot become active before target-side preview.

- [ ] **Step 8: Render validation rows and review state**

Populate one table row per template with direction, template id, status, selected candidate, confidence metrics, coverage, fit time, review state, and reason. `review` blocks publish. `excluded` enables publish only when the backend reports no blocking issue and the global override reason is non-empty.

```cpp
const QStringList columns{QStringLiteral("方向"), QStringLiteral("模板"),
    QStringLiteral("状态"), QStringLiteral("候选"), QStringLiteral("边缘支持"),
    QStringLiteral("可见比例"), QStringLiteral("残差"), QStringLiteral("覆盖率"),
    QStringLiteral("耗时(ms)"), QStringLiteral("复核状态"), QStringLiteral("原因")};
validationTable_->setColumnCount(columns.size());
validationTable_->setHorizontalHeaderLabels(columns);
```

- [ ] **Step 9: Run manager tests**

Run:

```powershell
Set-Location E:\Project\wang\pp_813\qt_app\tests
qmake test_geometrymaskmanager.pro -o Makefile.geometrymaskmanager -spec win32-msvc
nmake /F Makefile.geometrymaskmanager.Release
.\release\test_geometrymaskmanager.exe -platform offscreen
```

Expected: `test_geometrymaskmanager.exe` exits with code 0.

- [ ] **Step 10: Commit the dialog behavior**

```powershell
git add -- qt_app\geometrymaskmanager.h qt_app\geometrymaskmanager.cpp qt_app\tests\test_geometrymaskmanager.cpp
git commit -m "feat: manage geometry rule previews and drafts"
```

### Task 8: Main Window Preview Transport and Error Routing

**Files:**
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp`
- Modify: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Consumes: Task 7 `previewRequested` and Task 5 `preview_geometry_mask_rule`.
- Produces: `previewGeometryRule(const QJsonObject &request)` slot and response/error routing to the open geometry dialog.

- [ ] **Step 1: Add a failing request-wiring test**

```cpp
class GeometryPreviewServer : public QObject {
    Q_OBJECT
public:
    explicit GeometryPreviewServer(const QString &templatePath) : templatePath_(templatePath) {
        connect(&server_, &QTcpServer::newConnection, this, [this]() {
            socket_ = server_.nextPendingConnection();
            connect(socket_, &QTcpSocket::readyRead, this, &GeometryPreviewServer::readRequests);
        });
    }
    bool listen() { return server_.listen(QHostAddress::LocalHost, 0); }
    quint16 port() const { return server_.serverPort(); }
    QList<QJsonObject> requests() const { return requests_; }
    void setPreviewError(bool value) { previewError_ = value; }

private slots:
    void readRequests() {
        buffer_ += socket_->readAll();
        while (buffer_.contains('\n')) {
            const int newline = buffer_.indexOf('\n');
            const QJsonObject request = QJsonDocument::fromJson(buffer_.left(newline)).object();
            buffer_.remove(0, newline + 1);
            requests_.append(request);
            reply(request);
        }
    }
private:
    void reply(const QJsonObject &request);
    QTcpServer server_;
    QTcpSocket *socket_ = nullptr;
    QByteArray buffer_;
    QList<QJsonObject> requests_;
    QString templatePath_;
    bool previewError_ = false;
};

void TestMainWindow::geometryPreviewAddsWorkpieceAndUsesReadOnlyCommand() {
    QTemporaryDir dir;
    const QString templatePath = writeImages(dir, QStringLiteral("front"), 1).first();
    GeometryPreviewServer server(templatePath);
    QVERIFY(server.listen());
    BackendClient client;
    MainWindow window(&client, nullptr);
    client.connectToService(QHostAddress::LocalHost, server.port(), 500);
    QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
    auto *workpieces = window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
    QTRY_VERIFY_WITH_TIMEOUT(workpieces != nullptr && workpieces->count() == 1, 1000);
    QMetaObject::invokeMethod(&window, "openGeometryMaskManager", Qt::DirectConnection);
    auto *dialog = window.findChild<GeometryMaskManagerDialog *>();
    QVERIFY(dialog != nullptr);
    emit dialog->previewRequested(QJsonObject{{"base_library_revision", 4},
        {"direction", "front"}, {"template_id", "front:00.png"},
        {"seed_shape", QJsonObject{{"shape", "circle"}, {"cx", 10.0},
                                   {"cy", 10.0}, {"r", 5.0}}},
        {"mode", "inside"}, {"margin_ratio", 0.02}});
    QTRY_VERIFY_WITH_TIMEOUT(!server.requests().isEmpty(), 1000);
    const QJsonObject request = server.requests().last();
    QCOMPARE(request.value("command").toString(), QStringLiteral("preview_geometry_mask_rule"));
    QCOMPARE(request.value("workpiece_id").toString(), QStringLiteral("m7"));
    QVERIFY(!request.contains("operation_id"));
}
```

Implement `GeometryPreviewServer::reply` in the test for `hello`, `list_workpieces`, `get_geometry_mask_profile`, and `preview_geometry_mask_rule`. Add `#include <QComboBox>` to the test file.

```cpp
void GeometryPreviewServer::reply(const QJsonObject &request) {
    const QString command = request.value("command").toString();
    const QString requestId = request.value("request_id").toString();
    QJsonObject response{{"version", 1}, {"request_id", requestId}, {"ok", true}};
    if (command == QStringLiteral("hello")) {
        response.insert("service", "workpiece-orientation");
        response.insert("ready", true);
    } else if (command == QStringLiteral("list_workpieces")) {
        response.insert("workpieces", QJsonArray{QJsonObject{{"id", "m7"}, {"name", "M7"}}});
    } else if (command == QStringLiteral("get_geometry_mask_profile")) {
        response.insert("profile", QJsonObject{{"workpiece_id", "m7"},
            {"library_revision", 4}, {"draft_revision", 0},
            {"active_revision", QJsonValue()},
            {"draft", QJsonObject{{"schema_version", 1}, {"directions", QJsonObject{
                {"front", QJsonObject{{"anchor", QJsonValue()}, {"rules", QJsonArray()}}},
                {"back", QJsonObject{{"anchor", QJsonValue()}, {"rules", QJsonArray()}}}}}}},
            {"templates", QJsonArray{QJsonObject{{"template_id", "front:00.png"},
                {"direction", "front"}, {"path", templatePath_}}}}});
    } else if (command == QStringLiteral("preview_geometry_mask_rule") && previewError_) {
        response.insert("ok", false);
        response.insert("error", QJsonObject{{"code", "STALE_REVISION"},
                                              {"message", "工件库已变化"}});
    } else if (command == QStringLiteral("preview_geometry_mask_rule")) {
        response.insert("preview", oneCandidatePreview());
    }
    socket_->write(QJsonDocument(response).toJson(QJsonDocument::Compact) + "\n");
    socket_->flush();
}
```

Define the preview helper in the same test file:

```cpp
QJsonObject oneCandidatePreview() {
    const QJsonObject anchor{{"shape", "ellipse"}, {"cx", 40.0}, {"cy", 30.0},
                             {"rx", 28.0}, {"ry", 24.0}, {"angle_deg", 0.0}};
    const QJsonObject fitted{{"shape", "circle"}, {"cx", 40.0}, {"cy", 30.0},
                             {"r", 18.0}, {"angle_deg", 0.0}};
    return QJsonObject{{"workpiece_id", "m7"}, {"direction", "front"},
        {"template_id", "front:00.png"},
        {"anchor_fit", QJsonObject{{"candidates", QJsonArray{anchor}},
                                    {"selected_candidate_index", 0},
                                    {"fitted_shape", anchor}}},
        {"rule_fit", QJsonObject{{"candidates", QJsonArray{fitted}},
                                  {"selected_candidate_index", 0},
                                  {"fitted_shape", fitted}}},
        {"profile_patch", QJsonObject{
            {"reference_template", QJsonObject{{"template_id", "front:00.png"},
                {"direction", "front"}, {"width", 80}, {"height", 60}}},
            {"anchor", QJsonObject{{"shape", "ellipse"}, {"mode", "auto"},
                {"coarse", QJsonObject{{"cx", 0.5}, {"cy", 0.5}, {"rx", 0.35},
                                       {"ry", 0.4}, {"angle_deg", 0.0}}}}},
            {"geometry", QJsonObject{{"cx", 0.0}, {"cy", 0.0}, {"r", 0.6}}},
            {"seed_geometry", QJsonObject{{"cx", 0.0}, {"cy", 0.0}, {"r", 0.6}}}}}};
}
```

- [ ] **Step 2: Add failing response and failure-routing tests**

Use `GeometryPreviewServer` to send a successful preview and assert the dialog draft changes. Set `previewError_ = true`, send another preview, and assert the dialog remains open, its draft remains dirty, and the error text is visible.

```cpp
QTRY_VERIFY_WITH_TIMEOUT(dialog->draft().value("directions").toObject()
    .value("front").toObject().value("anchor").isObject(), 1000);
server.setPreviewError(true);
emit dialog->previewRequested(secondPreviewRequest());
QTRY_VERIFY_WITH_TIMEOUT(dialog->findChild<QLabel *>(QStringLiteral("statusLabel"))
    ->text().contains(QStringLiteral("工件库已变化")), 1000);
QVERIFY(dialog->isVisible());
QVERIFY(dialog->findChild<QLabel *>(QStringLiteral("dirtyLabel"))->text().contains(
    QStringLiteral("未保存")));
```

Define the request helper in the test file:

```cpp
QJsonObject secondPreviewRequest() {
    return QJsonObject{{"base_library_revision", 4}, {"direction", "front"},
        {"template_id", "front:00.png"},
        {"seed_shape", QJsonObject{{"shape", "circle"}, {"cx", 10.0},
                                   {"cy", 10.0}, {"r", 5.0}}},
        {"mode", "inside"}, {"margin_ratio", 0.02},
        {"rule_candidate_index", 0}};
}
```

- [ ] **Step 3: Run the MainWindow Qt test and verify failure**

Build and run:

```powershell
Set-Location E:\Project\wang\pp_813\qt_app\tests
qmake test_mainwindow.pro -o Makefile.mainwindow -spec win32-msvc
nmake /F Makefile.mainwindow.Release
.\release\test_mainwindow.exe -platform offscreen
```

Expected: preview wiring tests FAIL.

- [ ] **Step 4: Connect the preview request and send one backend command**

Add the slot declaration, connect it in `openGeometryMaskManager`, merge `geometryWorkpieceId_`, set `pendingCommand_`, and call `BackendClient::sendRequest`. Do not add a second socket or parallel request queue.

```cpp
void MainWindow::previewGeometryRule(const QJsonObject &request) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_)
        return;
    QJsonObject fields = request;
    fields.insert(QStringLiteral("workpiece_id"), geometryWorkpieceId_);
    pendingCommand_ = QStringLiteral("preview_geometry_mask_rule");
    geometryMaskManagerDialog_->setPreviewBusy(true);
    client_->sendRequest(pendingCommand_, fields);
}
```

- [ ] **Step 5: Route success and preserve draft on errors**

Handle `preview_geometry_mask_rule` before save/validate branches:

```cpp
if (command == QStringLiteral("preview_geometry_mask_rule")) {
    pendingCommand_.clear();
    if (geometryMaskManagerDialog_ != nullptr) {
        geometryMaskManagerDialog_->setRulePreview(
            response.value(QStringLiteral("preview")).toObject());
        geometryMaskManagerDialog_->setPreviewBusy(false);
    }
    return;
}
```

Extend geometry error-prefix routing to include preview and call `setOperationError` without replacing the dialog snapshot.

- [ ] **Step 6: Run MainWindow and backend-client tests**

Run:

```powershell
.\release\test_mainwindow.exe -platform offscreen
.\release\test_backendclient.exe -platform offscreen
```

Expected: both executables exit with code 0.

- [ ] **Step 7: Commit Qt transport integration**

```powershell
git add -- qt_app\mainwindow.h qt_app\mainwindow.cpp qt_app\tests\test_mainwindow.cpp
git commit -m "feat: connect geometry preview to desktop backend"
```

### Task 9: Full Verification, Performance Report, and Operator Acceptance

**Files:**
- Create: `docs/verification/geometry-rule-editor-redesign-results.md`
- Modify: `tests/test_geometry_calibration.py`
- Modify: `tests/test_geometry_mask_profiles.py`

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: reproducible test/build/performance evidence and an operator-facing acceptance checklist.

- [ ] **Step 1: Run focused Python suites**

```powershell
Set-Location E:\Project\wang\pp_813
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_geometry_calibration.py tests\test_geometry_mask_profiles.py tests\test_orientation_classifier.py tests\test_workpiece_catalog.py tests\test_orientation_tcp_service.py -q --basetemp=pytest-geometry-editor-focused
```

Expected: all focused tests PASS.

- [ ] **Step 2: Run the full Python suite**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q --basetemp=pytest-geometry-editor-full
```

Expected: all non-environment-skipped tests PASS; record the exact pass/skip count and elapsed time.

- [ ] **Step 3: Build the Qt 5.14.2 Release application**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_qt5.ps1
```

Expected: `qt_app\build-release\release\workpiece_orientation.exe` is produced without compiler/linker errors.

- [ ] **Step 4: Run all Qt tests offscreen**

Run the existing release test executables, including:

```powershell
qt_app\tests\release\test_geometryrulecanvas.exe -platform offscreen
qt_app\tests\release\test_geometrymaskmanager.exe -platform offscreen
qt_app\tests\release\test_mainwindow.exe -platform offscreen
qt_app\tests\release\test_backendclient.exe -platform offscreen
qt_app\tests\release\test_backendprocessmanager.exe -platform offscreen
qt_app\tests\release\test_annotationmanager.exe -platform offscreen
```

Expected: every process exits with code 0.

- [ ] **Step 5: Measure preview and validation scaling**

Add `import time` to `tests/test_geometry_mask_profiles.py`, then add deterministic pytest scaling coverage using the existing `write_image`, `fake_builder`, and `GeometryFakeClassifier` test helpers:

```python
@pytest.mark.parametrize("front_count,back_count", [(1, 1), (5, 10), (10, 15), (35, 35)])
def test_geometry_validation_processes_every_selected_template(
    tmp_path: Path, front_count: int, back_count: int
):
    library = WorkpieceLibrary(tmp_path / "library")
    classifier = GeometryFakeClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    front = [write_image(tmp_path / f"front-{index}.png", 10 + index)
             for index in range(front_count)]
    back = [write_image(tmp_path / f"back-{index}.png", 100 + index)
            for index in range(back_count)]
    record, _ = catalog.register("scale", front, back, False)
    profiles = GeometryMaskProfiles(
        catalog, calibrator=object(), start_worker=False,
        storage_dir=tmp_path / "jobs",
    )
    started = time.perf_counter()
    profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id=f"scale-{front_count}-{back_count}",
    )
    completed = profiles.run_next(force=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert completed["state"] == "completed"
    assert completed["progress"]["completed"] == front_count + back_count
    assert len(completed["report"]["front"]) == front_count
    assert len(completed["report"]["back"]) == back_count
    print({"front": front_count, "back": back_count, "elapsed_ms": elapsed_ms})
```

Add `import time` and `from statistics import median` to `tests/test_geometry_calibration.py`, then add this 20-run timing test without a machine-specific latency assertion:

```python
def test_fit_reference_reports_repeatable_geometry_timing():
    calibrator = GeometryCalibrator()
    image = _synthetic_ellipse()
    seed = {"shape": "circle", "cx": 155.0, "cy": 92.0, "r": 31.0}
    samples_ms = []
    for _ in range(20):
        started = time.perf_counter()
        result = calibrator.fit_reference(image, seed, mode="inside", margin_ratio=0.02)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        assert result["status"] == "active"
    print({"preview_median_ms": median(samples_ms)})
```

Run both timing tests with `pytest -s` and record:

- median `preview_geometry_mask_rule` fitting time for one 512-pixel template;
- total validation wall time and per-template mean;
- confirmation that runtime grows with the number of selected templates and no selected template is silently omitted.

The report must distinguish geometry-only preview time from PP-ShiTuV2/ALIKED cache-building time.

- [ ] **Step 6: Run the real backend smoke test**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\smoke_orientation_service.ps1 -StartupTimeoutSeconds 600
```

Expected: hello, library listing, and shutdown succeed. Record model-load duration separately from preview fitting duration.

- [ ] **Step 7: Perform the 35+35 operator checklist**

In the Qt application:

1. Open a workpiece containing 35 front and 35 back templates.
2. Select one clean reference template per direction.
3. Draw a center reflection circle using center-to-radius.
4. Verify the cyan seed, green fitted circle, orange ignored interior, candidate selector, and confidence diagnostics.
5. Switch to at least three other templates and verify their independently fitted overlays.
6. Add an outside rule and verify corner intrusions are orange while the workpiece interior remains unmasked.
7. Mark one template `review`, confirm publish is blocked, then mark it `excluded` with a reason and confirm the override reason is required.
8. Cancel one validation and verify the active revision is unchanged.
9. Publish, predict one image, then roll back and verify the pre-publication cache is restored.

- [ ] **Step 8: Write the verification report**

Create `docs/verification/geometry-rule-editor-redesign-results.md` with commit ids, exact commands, pass/skip counts, Qt exit codes, preview/validation timing table, smoke-test result, operator checklist outcome, and any environment-only skips. Do not claim a manual check passed unless it was performed.

- [ ] **Step 9: Run repository hygiene checks**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors. Confirm only task files are staged; leave all unrelated pre-existing modifications untouched.

- [ ] **Step 10: Commit verification evidence**

```powershell
git add -- tests\test_geometry_calibration.py tests\test_geometry_mask_profiles.py docs\verification\geometry-rule-editor-redesign-results.md
git commit -m "test: verify geometry rule editor redesign"
```

## Completion Criteria

- A circle uses center-to-radius and no drawing tool is selected by default.
- Ellipse and rotated rectangle have shape-specific center-based gestures, editable handles, numeric fields, zoom/pan, and reset.
- One committed gesture causes one read-only preview request; mouse movement never floods the backend.
- Preview converts native-pixel seeds to object-relative geometry before draft persistence.
- Every template has visible fit status, candidates, diagnostics, and overlay; review/exclusion states are explicit and audited.
- Validation is cancellable, progress is template-granular, and publication is blocked or requires an override exactly as specified.
- Query prediction is all-or-none across front/back geometry masks.
- Old libraries retain their active prediction cache until geometry publication and can be restored on first rollback.
- Existing model weights, algorithms, and fusion thresholds are unchanged.
- Full Python tests, Qt Release build, Qt offscreen tests, smoke test, and the 35+35 operator checklist have recorded results.
