# Lightweight Geometry Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a geometry-aware PP-ShiTu + per-workpiece Ridge online path that does not call ALIKED, LightGlue, or ORB and reaches warm single-image `P95 <= 25 ms` without adding errors on the fixed M1/M2/M7 acceptance set.

**Architecture:** Compile the published front/back geometry profile once, reuse the proven geometry candidate fitter on a shared 256-pixel query context, and replace Telea repair with a deterministic feathered neutral fill. Build one immutable fast cache per workpiece containing a three-embedding Ridge head; route predictions explicitly through `legacy` or `fast_geometry`, with background cache rebuilds and atomic publication.

**Tech Stack:** Python 3.10, NumPy 1.24, OpenCV 4.6, PaddlePaddle 3.2.2 / PP-ShiTuV2, pytest, Qt 5.14.2 Widgets/Network, MSVC 2019, PowerShell benchmarking.

## Global Constraints

- Online SLA: warmed backend single-image `P95 <= 25 ms` on the current NVIDIA GeForce RTX 4060 Ti.
- Timing includes image read, preprocessing, lightweight geometry, mask construction, PP-ShiTu, Ridge decision, and response construction; it excludes startup, model load, library build, and Qt rendering.
- Every readable image with enabled rules must attempt lightweight geometry; front and back remain independently calibrated.
- The `fast_geometry` online path must not call ALIKED, LightGlue, ORB, or any local-feature matcher.
- Do not retrain or modify PP-ShiTu weights or input definition.
- Preserve circles, ellipses, rotated rectangles, inside/outside semantics, signed margins, geometry publication, and rollback.
- Preserve arbitrary positive and unequal front/back template counts; never truncate selected templates.
- A low-confidence geometry fit returns a Ridge direction plus `needs_review=true`; invalid/missing fast cache or invalid PP-ShiTu features return a stable error, not a guessed label.
- Keep `legacy` explicit and usable. Do not switch the shipped default to `fast_geometry` until accuracy, review-rate, and latency gates all pass.
- Use `E:\python\anaconda3\envs\shitu\python.exe` for production-model Python tests and Qt 5.14.2/MSVC for desktop tests.

---

## File Map

### New Python units

- `src/fast_ridge.py`: deterministic weighted Ridge fitting, regularization selection, confidence threshold, and prediction.
- `src/fast_geometry.py`: profile compilation, one shared 256-pixel geometry context, candidate-window filtering, full-resolution masks, and fast fill.
- `src/fast_orientation.py`: immutable fast cache, template/augmentation feature build, online three-image PP-ShiTu batch, and result contract.
- `src/fast_cache_jobs.py`: single-worker background rebuild state machine with revision-checked publication.
- `scripts/benchmark_fast_geometry_inference.py`: reproducible legacy/fast accuracy and latency gate.
- `tests/test_fast_ridge.py`, `tests/test_fast_geometry.py`, `tests/test_fast_orientation.py`, `tests/test_fast_cache_jobs.py`, `tests/test_benchmark_fast_geometry_inference.py`: focused tests for the new units.

### Existing Python units to modify

- `src/orientation_classifier.py`: inference-mode loading, `TemplateCache.fast_runtime`, fast-cache persistence, build integration, and prediction routing.
- `src/workpiece_catalog.py`: cache status, background recovery, lifecycle invalidation/rebuild, and revision-safe publication.
- `src/orientation_tcp_service.py`: `--inference-mode`, stable fast-path errors, progress/status fields, and shutdown.
- `src/geometry_mask_profiles.py`: pass the active profile into fast-cache builds and invalidate/rebuild on publish or rollback without changing rule semantics.
- `src/workpiece_library.py`: accept structured feature-build progress while retaining the existing three-argument callback adapter.
- `tests/test_orientation_classifier.py`, `tests/test_workpiece_catalog.py`, `tests/test_orientation_tcp_service.py`, `tests/test_orientation_service_environment.py`, `tests/test_geometry_mask_profiles.py`, `tests/test_workpiece_library.py`: integration and compatibility coverage.

### Qt files to modify

- `qt_app/appconfig.h`, `qt_app/appconfig.cpp`, `qt_app/app_config.json.example`: validated `inference_mode` configuration.
- `qt_app/backendprocessmanager.cpp`: forward `--inference-mode`.
- `qt_app/inspectionpage.cpp`: fast-engine evidence, review reason, and stage timing display without local-match wording.
- `qt_app/workpiecelibrarypage.cpp`: fast-cache build phases and status.
- `qt_app/tests/test_appconfig.cpp`, `qt_app/tests/test_backendprocessmanager.cpp`, `qt_app/tests/test_inspectionpage.cpp`, `qt_app/tests/test_workpiecelibrarypage.cpp`, `qt_app/tests/test_mainwindow.cpp`: desktop regressions.

### Verification artifact

- `docs/verification/lightweight-geometry-fast-path-results.md`: final tests, accuracy table, cache/build cost, and P50/P95/P99/max.

---

### Task 1: Deterministic Per-Workpiece Ridge Head

**Files:**
- Create: `src/fast_ridge.py`
- Create: `tests/test_fast_ridge.py`

**Interfaces:**
- Produces: `RidgeHead`, `RidgeDecision`, `pack_embeddings`, `fit_ridge_head`, and `predict_ridge`.
- Consumed later by: `src/fast_orientation.py`.

- [ ] **Step 1: Write failing tests for packing, unequal classes, determinism, and review thresholds**

```python
import numpy as np
import pytest

from src.fast_ridge import fit_ridge_head, pack_embeddings, predict_ridge


def test_pack_embeddings_l2_normalizes_each_slot_in_fixed_order():
    packed = pack_embeddings([
        np.array([3.0, 4.0], np.float32),
        np.array([0.0, 2.0], np.float32),
        np.array([5.0, 0.0], np.float32),
    ])
    np.testing.assert_allclose(
        packed,
        np.array([0.6, 0.8, 0.0, 1.0, 1.0, 0.0], np.float32),
        atol=1e-6,
    )


def test_ridge_accepts_unequal_counts_and_separates_both_classes():
    front = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], np.float32)
    back = np.array([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]], np.float32)
    head = fit_ridge_head(front, back)
    assert all(predict_ridge(head, row).label == "front" for row in front)
    assert all(predict_ridge(head, row).label == "back" for row in back)
    assert head.training_summary["class_counts"] == {"front": 3, "back": 5}


@pytest.mark.parametrize("front_count,back_count", [(1, 1), (5, 10), (10, 15), (35, 35)])
def test_ridge_supports_required_template_count_matrix(front_count, back_count):
    front = np.tile(np.array([[1.0, 0.0]], np.float32), (front_count, 1))
    back = np.tile(np.array([[0.0, 1.0]], np.float32), (back_count, 1))
    head = fit_ridge_head(front, back)
    assert head.training_summary["class_counts"] == {
        "front": front_count,
        "back": back_count,
    }


def test_one_plus_one_builds_but_marks_small_margin_for_review():
    head = fit_ridge_head(
        np.array([[1.0, 0.0]], np.float32),
        np.array([[0.0, 1.0]], np.float32),
    )
    decision = predict_ridge(head, np.array([0.51, 0.49], np.float32))
    assert decision.needs_review is True
    assert head.training_summary["validation_status"] == "few_shot_unverified"


def test_fit_is_bitwise_deterministic():
    front = np.eye(4, dtype=np.float32)[:2]
    back = -np.eye(4, dtype=np.float32)[:3]
    first = fit_ridge_head(front, back)
    second = fit_ridge_head(front, back)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.bias == second.bias
    assert first.review_threshold == second.review_threshold


def test_non_finite_features_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        fit_ridge_head(
            np.array([[np.nan, 0.0]], np.float32),
            np.array([[0.0, 1.0]], np.float32),
        )


def test_leave_one_source_out_excludes_rotations_from_the_held_out_source():
    front = np.array([[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]], np.float32)
    back = np.array([[0.0, 1.0], [0.01, 0.99], [0.2, 0.8]], np.float32)
    head = fit_ridge_head(
        front,
        back,
        front_source_ids=[0, 0, 1],
        back_source_ids=[0, 0, 1],
        front_original_rows=[True, False, True],
        back_original_rows=[True, False, True],
    )
    assert head.training_summary["validation_folds"] == 4
    assert head.training_summary["validation_grouping"] == "leave_one_source_out"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_fast_ridge.py -q -p no:cacheprovider
```

Expected: collection fails with `ModuleNotFoundError: No module named 'src.fast_ridge'`.

- [ ] **Step 3: Implement the Ridge types and fixed numerical contract**

Create the following public dataclasses exactly as shown:

```python
RIDGE_REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

@dataclass(frozen=True)
class RidgeHead:
    weights: np.ndarray
    bias: float
    regularization: float
    review_threshold: float
    feature_dim: int
    training_summary: dict[str, Any]

@dataclass(frozen=True)
class RidgeDecision:
    label: str
    margin: float
    needs_review: bool

```

Add these public functions with the stated signatures:

- `pack_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray`
- `fit_ridge_head(front_features: np.ndarray, back_features: np.ndarray, *, front_source_ids: Sequence[int] | None = None, back_source_ids: Sequence[int] | None = None, front_original_rows: Sequence[bool] | None = None, back_original_rows: Sequence[bool] | None = None, regularization_grid: Sequence[float] = RIDGE_REGULARIZATION_GRID) -> RidgeHead`
- `predict_ridge(head: RidgeHead, feature: np.ndarray) -> RidgeDecision`

Implementation rules:

1. Convert inputs to finite `float64` for solving and return `float32` weights.
2. Assign `front=+1.0`, `back=-1.0` and weights `0.5 / class_count` so unequal template counts do not bias the head.
3. Default every row to a unique source ID and `original=True` when source/original metadata is omitted. For each regularization candidate, perform deterministic leave-one-source-out. Remove the held-out original and every augmentation derived from that source, train on all remaining rows, and score only the held-out original row. This prevents a rotated copy of the validation image leaking into its training fold. Skip folds that contain only one class; if every fold is skipped, select `1.0` and set `few_shot_unverified`.
4. Rank candidates by `(classification_errors, review_count, -minimum_correct_margin, regularization)`.
5. Refit on all supplied samples using weighted centered dual Ridge:

```python
mean_x = np.sum(sample_weight[:, None] * x, axis=0)
mean_y = float(np.sum(sample_weight * y))
xw = (x - mean_x) * np.sqrt(sample_weight[:, None])
yw = (y - mean_y) * np.sqrt(sample_weight)
alpha = np.linalg.solve(xw @ xw.T + regularization * np.eye(len(x)), yw)
weights = xw.T @ alpha
bias = mean_y - float(mean_x @ weights)
```

6. Normalize weights and bias by `max(norm(weights), 1e-12)` so margin is a feature-space distance.
7. For valid leave-one-source-out folds, set `review_threshold` to half the smallest absolute correctly classified fold margin, clamped to `[0.01, 0.25]`. For 1+1, use one quarter of the smaller training margin, clamped to the same interval.
8. Store `validation_status="validated"` only when every usable leave-one-source-out original is classified correctly; otherwise store `cross_validation_failed`, or `few_shot_unverified` when no two-class fold exists. `predict_ridge` returns `front` for non-negative margin, `back` otherwise, and reviews when `abs(margin) < review_threshold`.

- [ ] **Step 4: Run Ridge tests and the existing baseline math tests**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_fast_ridge.py tests/test_shitu_baseline.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/fast_ridge.py tests/test_fast_ridge.py
git commit -m "feat: add deterministic fast ridge head"
```

---

### Task 2: Shared Lightweight Geometry and Fast Masking

**Files:**
- Create: `src/fast_geometry.py`
- Create: `tests/test_fast_geometry.py`
- Modify: `src/geometry_calibration.py:279-375` only to expose thin public wrappers around the existing coarse-shape, rule-projection, candidate-window, and shape-mask semantics when needed.
- Modify: `tests/test_geometry_calibration.py`

**Interfaces:**
- Consumes: canonical profiles from `materialize_runtime_profile` and the existing `GeometryCalibrator` quality gates.
- Produces: `CompiledFastGeometry`, `FastGeometryVariants`, `compile_fast_geometry`, and `FastGeometryProcessor.build_variants`.
- Consumed later by: `FastOrientationEngine`.

- [ ] **Step 1: Write failing tests for no-op profiles, one shared context, masks, and safe fallback**

```python
def test_no_rules_produce_three_raw_slots_without_review():
    processor, calibrator = processor_with_fake_calibrator()
    image = marker_image(17)
    variants = processor.build_variants(image, compile_fast_geometry(None))
    assert len(variants.images) == 3
    assert all(np.array_equal(item, image) for item in variants.images)
    assert variants.status == "not_configured"
    assert variants.needs_review is False
    assert calibrator.prepare_calls == 0


def test_front_and_back_share_one_resized_context():
    processor, calibrator = processor_with_fake_calibrator()
    variants = processor.build_variants(marker_image(17, size=512), configured_profile())
    assert calibrator.prepare_calls == 1
    assert calibrator.fit_labels == ["front", "back"]
    assert calibrator.context_shapes == [(256, 256), (256, 256)]
    assert variants.timings_ms.keys() >= {"geometry_context", "geometry_fit", "mask_build"}


def test_inside_and_outside_masks_use_fast_fill_not_telea(monkeypatch):
    monkeypatch.setattr(cv2, "inpaint", lambda *args, **kwargs: pytest.fail("Telea called"))
    processor, _ = processor_with_fake_calibrator(mask_center=True)
    variants = processor.build_variants(marker_image(17), configured_profile())
    assert variants.images[1][32, 32].tolist() == [7, 7, 7]


def test_one_failed_direction_uses_raw_slot_and_requests_review():
    processor, _ = processor_with_fake_calibrator(fail_side="back")
    image = marker_image(17)
    variants = processor.build_variants(image, configured_profile())
    assert np.array_equal(variants.images[2], image)
    assert variants.needs_review is True
    assert variants.review_reasons == ("FAST_GEOMETRY_LOW_CONFIDENCE",)
```

Also add a parity test in `tests/test_geometry_calibration.py` asserting that the new public shape/margin wrappers produce the same masks as the current private implementation for circle, ellipse, rotated rectangle, inside, outside, and margins `-0.04`, `0.0`, `0.02`.

Add two more fast-geometry cases: `test_compilation_preserves_circle_ellipse_rotated_rectangle_and_signed_margin` compares every canonical field of a mixed profile after compilation; `test_multiple_rules_union_their_ignore_masks` uses two disjoint fake rule masks and asserts both regions, and no pixels outside either region, are filled.

Add `test_candidate_filter_keeps_distinct_inner_and_outer_boundaries`: render two concentric high-contrast boundaries, compile one expected-inner and one expected-outer rule, and assert the selected fitted radii remain within 5% of their respective ground-truth radii after the 256-pixel resize. This guards the candidate-window optimization from collapsing both rules onto the same edge.

- [ ] **Step 2: Run the focused geometry tests and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_fast_geometry.py tests/test_geometry_calibration.py -q -p no:cacheprovider
```

Expected: `src.fast_geometry` is missing and the new public wrappers are undefined.

- [ ] **Step 3: Implement compiled geometry and the shared 256-pixel path**

Create this public contract:

```python
FAST_GEOMETRY_FORMAT_VERSION = 1
FAST_GEOMETRY_MAX_SIDE = 256

@dataclass(frozen=True)
class CompiledFastGeometry:
    format_version: int
    profile_revision: int | None
    directions: dict[str, dict[str, Any] | None]

@dataclass(frozen=True)
class FastGeometryVariants:
    images: tuple[np.ndarray, np.ndarray, np.ndarray]
    status: str
    needs_review: bool
    review_reasons: Sequence[str]
    directions: dict[str, dict[str, Any]]
    timings_ms: dict[str, float]
```

Add `compile_fast_geometry(profile: Mapping[str, Any] | None) -> CompiledFastGeometry` and a `FastGeometryProcessor` with constructor `(calibrator: Any, *, max_side: int = FAST_GEOMETRY_MAX_SIDE)` plus `build_variants(image: np.ndarray, compiled: CompiledFastGeometry) -> FastGeometryVariants`.

The processor must:

1. Resize once with `INTER_AREA`, preserving aspect ratio and never upscaling.
2. Call `calibrator.prepare_context(resized)` once.
3. Pre-filter the shared contour tuple against each compiled direction's normalized center/scale window before calling the proven `GeometryCalibrator.fit`; do not compute a second Canny image.
4. Resize each successful `ignore_mask` to source size with `INTER_NEAREST`.
5. Reject `ignored_ratio >= 0.55` with `FAST_GEOMETRY_LOW_CONFIDENCE`.
6. Apply the direction's stored `fill_bgr`; feather only the mask edge with a Gaussian sigma of `max(0.5, min(height, width) * 0.005)` and never call `cv2.inpaint`.
7. Return slots in exactly `(raw, front_masked, back_masked)` order.
8. Use a raw copy for a failed direction, preserve the other successful direction, and set review rather than aborting the entire prediction.
9. Strip NumPy masks from serialized direction diagnostics.

- [ ] **Step 4: Run focused and full geometry regressions**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_fast_geometry.py tests/test_geometry_calibration.py tests/test_geometry_mask_profiles.py -q -p no:cacheprovider
```

Expected: all selected tests pass and existing profile semantics remain unchanged.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/fast_geometry.py src/geometry_calibration.py tests/test_fast_geometry.py tests/test_geometry_calibration.py
git commit -m "feat: add shared lightweight geometry processor"
```

---

### Task 3: Fast Cache Builder and Online Engine

**Files:**
- Create: `src/fast_orientation.py`
- Create: `tests/test_fast_orientation.py`

**Interfaces:**
- Consumes: `FastGeometryProcessor`, `CompiledFastGeometry`, `RidgeHead`, and an injected `embed_batch(images) -> list[np.ndarray]` callback.
- Produces: `FastRuntimeCache`, `FastOrientationEngine.build_cache`, and `FastOrientationEngine.predict`.
- Consumed later by: `OrientationClassifier`.

- [ ] **Step 1: Write failing tests for all-template participation, augmentation, batch shape, and result fields**

```python
def test_build_uses_every_unequal_template_and_keeps_actual_counts(tmp_path):
    front = write_markers(tmp_path, "front", 5, base=10)
    back = write_markers(tmp_path, "back", 12, base=100)
    engine, embedder = fake_engine()
    cache = engine.build_cache(
        front, back,
        geometry_profile=None,
        library_revision=4,
        model_fingerprint="model-a",
    )
    assert cache.template_counts == {"front": 5, "back": 12}
    assert cache.training_summary["original_samples"] == 17
    assert set(embedder.seen_paths) >= {str(path) for path in front + back}


def test_few_shot_rotation_augments_only_small_side(tmp_path):
    front = write_markers(tmp_path, "front", 5, base=10)
    back = write_markers(tmp_path, "back", 20, base=100)
    engine, _ = fake_engine()
    cache = engine.build_cache(front, back, geometry_profile=None,
                               library_revision=1, model_fingerprint="m")
    assert cache.training_summary["augmented_samples"] == {"front": 55, "back": 0}
    assert cache.training_summary["rotation_degrees"] == list(range(30, 360, 30))


def test_predict_calls_one_three_image_batch_and_never_local_models(tmp_path):
    engine, embedder = fake_engine()
    cache = separable_cache()
    result = engine.predict(marker_image(10), cache)
    assert embedder.batch_sizes == [3]
    assert result["inference_engine"] == "fast_geometry"
    assert result["decision_source"] == "fast_ridge"
    assert result["label"] in {"front", "back"}
    assert "local_prediction" not in result
    assert result["timings_ms"].keys() >= {
        "geometry_context", "geometry_fit", "mask_build",
        "global_batch", "linear_head", "total",
    }


def test_geometry_failure_returns_direction_and_review_reason():
    engine, _ = fake_engine(fail_geometry=True)
    result = engine.predict(marker_image(10), separable_cache())
    assert result["label"] in {"front", "back"}
    assert result["needs_review"] is True
    assert "FAST_GEOMETRY_LOW_CONFIDENCE" in result["review_reason_codes"]


def test_template_and_query_use_the_same_three_slot_preprocessing(tmp_path):
    path = write_marker(tmp_path / "front.png", 10)
    engine, embedder = fake_engine(record_batches=True)
    engine.build_cache(
        [path],
        [write_marker(tmp_path / "back.png", 200)],
        geometry_profile=configured_profile(),
        library_revision=1,
        model_fingerprint="m",
    )
    template_slots = tuple(image.copy() for image in embedder.batches[0][:3])
    query_slots = engine.geometry.build_variants(
        read_marker(path), compile_fast_geometry(configured_profile())
    ).images
    assert all(np.array_equal(left, right) for left, right in zip(template_slots, query_slots))
```

- [ ] **Step 2: Run the fast-engine test and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_fast_orientation.py -q -p no:cacheprovider
```

Expected: `ModuleNotFoundError: No module named 'src.fast_orientation'`.

- [ ] **Step 3: Implement immutable cache and engine**

Use these exact public fields:

```python
FAST_RUNTIME_FORMAT_VERSION = 1
FAST_FEATURE_LAYOUT = ("raw", "front_masked", "back_masked")
FAST_ROTATION_DEGREES = tuple(range(30, 360, 30))
FAST_AUGMENT_BELOW_PER_SIDE = 20
FAST_BUILD_IMAGE_BATCH = 32

@dataclass(frozen=True)
class FastRuntimeCache:
    format_version: int
    cache_revision: str
    library_revision: int
    geometry_profile_revision: int | None
    model_fingerprint: str
    template_signature: dict[str, Any]
    template_counts: dict[str, int]
    feature_layout: tuple[str, str, str]
    compiled_geometry: CompiledFastGeometry
    ridge_head: RidgeHead
    training_summary: dict[str, Any]

```

Add `FastOrientationEngine` with these exact call contracts:

- Constructor: `(embed_batch: Callable[[Sequence[np.ndarray]], list[np.ndarray]], geometry: FastGeometryProcessor, *, image_reader: Callable[[Path], np.ndarray])`
- `build_cache(front_paths: Sequence[Path], back_paths: Sequence[Path], *, geometry_profile: Mapping[str, Any] | None, library_revision: int, model_fingerprint: str, progress_callback: Callable[[dict[str, Any]], None] | None = None) -> FastRuntimeCache`
- `predict(image: np.ndarray, cache: FastRuntimeCache) -> dict[str, object]`

Build rules:

- Hash every decoded original template to create the signature; reject unreadable, duplicate, non-finite, or empty sides.
- Generate each original template's three variants first. For a side with fewer than 20 originals, rotate that template's raw/front-masked/back-masked variants with one identical affine transform around the fitted target center at the 11 fixed angles; when no rule supplies a center, use the decoded image center. Do not refit a rotated image and do not rotate the three slots independently.
- Flatten variant images into chunks of at most 32 images per PP-ShiTu call, then restore sample boundaries and call `pack_embeddings`.
- Give every original and all rotations derived from it the same source ID, mark only the unrotated row as original, and pass those arrays into `fit_ridge_head`. Regularization and review threshold therefore use leak-free leave-one-source-out folds while the final Ridge refit uses original plus augmented samples with class-balanced weights.
- Emit progress objects with phases `fast_originals`, `fast_augmentation`, `fast_ridge`, actual completed/total counts, and an explicit `unit` (`templates`, `augmented_samples`, or `ridge_head`) so the service never presents augmentation images as user templates.
- Include geometry review counts, augmentation counts, selected regularization, review threshold, and total build milliseconds in `training_summary`.

Predict rules:

- Validate format, model fingerprint, feature layout, and finite head values before inference.
- Call geometry, one three-image embedding batch, pack, and `predict_ridge`.
- Set `needs_review` if geometry is low-confidence, Ridge margin is low, or the head's `validation_status` is not `validated`; return `FAST_GEOMETRY_LOW_CONFIDENCE`, `FAST_CLASSIFIER_LOW_MARGIN`, and/or `FAST_MODE_NOT_VALIDATED` plus a Chinese `review_reason`. Do not invoke a fallback model for any of these statuses.
- Return `fast_cache_revision`, `geometry_status`, direction reports, `decision_margin`, `elapsed_ms`, and the required `timings_ms` fields. Compute `cache_revision` deterministically from the validated library/profile/model/template/runtime signature, not from wall-clock time.

- [ ] **Step 4: Run the new engine tests**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_fast_ridge.py tests/test_fast_geometry.py tests/test_fast_orientation.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/fast_orientation.py tests/test_fast_orientation.py
git commit -m "feat: build and run fast orientation caches"
```

---

### Task 4: Classifier Modes, Persistence, and Prediction Routing

**Files:**
- Modify: `src/orientation_classifier.py:32-43,70-83,164-275,288-386,407-561,1236-1400`
- Modify: `tests/test_orientation_classifier.py`

**Interfaces:**
- Consumes: `FastOrientationEngine` and `FastRuntimeCache`.
- Produces: `INFERENCE_MODES`, `DEFAULT_INFERENCE_MODE`, validated classifier `inference_mode`, `.fast_runtime_cache.pkl` persistence, and mode-aware `predict_with_cache`.
- Consumed later by: service and catalog.

- [ ] **Step 1: Add failing classifier tests**

Add these tests with explicit setup and assertions:

- `test_fast_prediction_uses_fast_runtime_without_extracting_local`: construct the existing fake classifier with counting extractor/matcher objects, set `inference_mode="fast_geometry"`, attach a fake fast engine returning `front`, and replace the built cache's `fast_runtime` with a valid fake. Predict a temporary query and assert label `front`, `library_revision == 9`, extractor calls `== 0`, matcher calls `== 0`, and `inference_engine == "fast_geometry"`.
- `test_fast_mode_rejects_missing_runtime_cache`: build a valid base cache without `fast_runtime`, predict in fast mode, and assert `OrientationClassifierError` contains `FAST_CACHE_NOT_READY` before either local fake is called.
- `test_fast_cache_round_trip_binds_library_geometry_and_model_revisions`: save a cache with library revision `7`, geometry revision `3`, and model fingerprint `model-a`; load it through a matching record/profile/model and assert all Ridge fields and template counts survive exactly.
- `test_fast_cache_revision_mismatch_is_ignored_without_deleting_base_cache`: save matching base and fast files, then load using library revision `8`; assert base cache loads, `fast_runtime is None`, and both cache files still exist.
- `test_attached_fast_cache_revision_mismatch_is_never_used`: attach a cache for library revision `7`, call prediction with revision `8`, and assert `OrientationClassifierError` contains `FAST_CACHE_REVISION_MISMATCH` before PP-ShiTu or local fakes are called.
- `test_old_v2_template_cache_loads_with_fast_runtime_none`: serialize the existing version-2 fixture, load it through the new classifier, and assert original global/local arrays remain equal and `fast_runtime is None`.
- `test_fast_load_does_not_construct_aliked_or_lightglue`: monkeypatch `src.aliked_lightglue_matcher.build_models` to raise `AssertionError("local stack loaded")`, load the real fake Paddle predictor with `inference_mode="fast_geometry"`, and assert loading succeeds with `extractor is None` and `matcher is None`.

- [ ] **Step 2: Run classifier tests and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py -k "fast_ or old_v2" -q -p no:cacheprovider
```

Expected: missing inference-mode/fast-cache attributes and tests fail.

- [ ] **Step 3: Implement mode-aware loading and `TemplateCache.fast_runtime`**

Add:

```python
INFERENCE_MODES = ("legacy", "fast_geometry", "compare")
DEFAULT_INFERENCE_MODE = "legacy"
FAST_RUNTIME_CACHE_FILE_NAME = ".fast_runtime_cache.pkl"

@dataclass(frozen=True)
class TemplateCache:
    # existing fields remain in the same order
    fast_runtime: FastRuntimeCache | None = None
```

`OrientationClassifier.__init__` stores validated `inference_mode` and `model_fingerprint`. `load` imports/builds the Torch local stack only for `legacy` and `compare`; fast mode constructs the Paddle predictor and `FastOrientationEngine` without importing model weights from ALIKED/LightGlue.

Mode behavior:

- `legacy`: existing build, geometry, local matching, and fusion remain byte-for-byte behavior compatible.
- `fast_geometry`: template build may store empty local placeholders, builds/loads `fast_runtime`, and `predict_with_cache` calls only `fast_engine.predict`.
- `compare`: loads both stacks; the online command still returns legacy, while the benchmark script explicitly calls both paths to prevent accidental production double inference.

For path inputs, start the fast total timer before `read_color_image`, record `timings_ms.decode`, pass the decoded array into the fast engine, and overwrite `timings_ms.total`/`elapsed_ms` with the full classifier-call duration. Array inputs record `decode=0.0`. Add a focused test using a delayed fake reader and assert decode time is nonzero and total is at least decode plus the engine's reported stages.

- [ ] **Step 4: Implement atomic separate fast-cache persistence**

Add these methods:

- `save_fast_runtime_cache(self, record: Any, cache: FastRuntimeCache) -> None`
- `load_fast_runtime_cache(self, record: Any) -> FastRuntimeCache | None`
- `build_fast_runtime_cache(self, record: Any, geometry_profile: Mapping[str, Any] | None, progress_callback: Callable[[dict[str, Any]], None] | None = None) -> FastRuntimeCache`

The pickle payload must contain a plain signature plus `FastRuntimeCache`. Build it inside a workpiece-local UUID temporary directory, `flush` and `fsync` the file, then use `os.replace` to atomically switch `.fast_runtime_cache.pkl`; remove the temporary directory afterward. A failed or stale build must never reach `os.replace`, so the last valid cache remains intact. Validate `library_revision`, geometry revision, model fingerprint, template content signature, feature layout, dimensions, and finite weights on load. Invalid fast cache returns `None` with a warning and leaves `.template_cache.pkl` untouched.

`save_template_cache` must serialize `dataclasses.replace(cache, fast_runtime=None)` to `.template_cache.pkl`, then persist `fast_runtime` only in `.fast_runtime_cache.pkl` when present; never duplicate the fast payload inside the base pickle. `load_template_cache` normalizes old pickles into a new `TemplateCache` and attaches a separately validated fast cache.

- [ ] **Step 5: Attach fast caches during base build and geometry candidate build**

- New workpiece/no profile: build a no-op fast cache.
- Active candidate profile: `prepare_geometry_cache` attaches a fast cache built from that exact profile and profile revision.
- In `fast_geometry`, skip candidate ALIKED/LightGlue extraction; in `compare`, retain it for the legacy side.
- Preserve existing progress callback compatibility.

- [ ] **Step 6: Run classifier, geometry, and persistence regressions**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py tests/test_geometry_mask_profiles.py tests/test_workpiece_library.py -q -p no:cacheprovider
```

Expected: all selected tests pass, including old cache recovery.

- [ ] **Step 7: Commit Task 4**

```powershell
git add src/orientation_classifier.py tests/test_orientation_classifier.py tests/test_geometry_mask_profiles.py tests/test_workpiece_library.py
git commit -m "feat: route classifier through fast geometry mode"
```

---

### Task 5: Revision-Safe Background Fast-Cache Jobs

**Files:**
- Create: `src/fast_cache_jobs.py`
- Create: `tests/test_fast_cache_jobs.py`
- Modify: `src/workpiece_catalog.py:37-120,552-605,641-748`
- Modify: `tests/test_workpiece_catalog.py`
- Modify: `src/geometry_mask_profiles.py:655-705,1290-1520`
- Modify: `tests/test_geometry_mask_profiles.py`

**Interfaces:**
- Consumes: classifier `build_fast_runtime_cache`, `save_fast_runtime_cache`, and immutable catalog snapshots.
- Produces: `FastCacheJobManager.schedule`, `snapshot`, `shutdown`; catalog `fast_cache_status(workpiece_id)`.
- Consumed later by: service summaries and Qt.

- [ ] **Step 1: Write failing job-state and stale-publication tests**

Use `threading.Event` barriers so the tests are deterministic:

- `test_job_transitions_queued_running_ready_and_publishes_once`: block the builder after it starts, assert `queued` then `running`, release it, wait for `ready`, and assert the publication callback received the cache exactly once.
- `test_duplicate_schedule_for_same_revision_returns_same_job`: schedule the same `(workpiece_id, library_revision, geometry_profile_revision)` twice while blocked; assert one builder call, equal job identity/status, and one publication.
- `test_stale_result_is_discarded_when_library_revision_changes`: have `publish` return `False` after the captured revision changes; assert final state `stale`, no active cache replacement, and no persisted fast-cache file.
- `test_failure_preserves_previous_ready_cache_and_exposes_error`: begin with a ready fast cache, raise `RuntimeError("build failed")` in the replacement build, and assert the old cache remains active while job state is `failed` with the message.
- `test_shutdown_waits_for_current_step_without_starting_more_jobs`: queue two workpieces, start the first, request shutdown from another thread, release the first, and assert the first finishes, the second builder is never entered, and shutdown returns.

Add catalog tests asserting:

- `recover()` returns before a blocked fast builder is released when a valid base cache exists.
- Prediction before completion returns `FAST_CACHE_NOT_READY` in fast mode.
- Completion swaps only `fast_runtime` on the matching snapshot revision.
- append, geometry publish, rollback, recycle, restore, and purge invalidate or reschedule exactly the affected workpiece.
- deletion removes `.fast_runtime_cache.pkl` with the workpiece directory; stale workers cannot recreate it.

- [ ] **Step 2: Run job/catalog tests and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_fast_cache_jobs.py tests/test_workpiece_catalog.py -k "fast_cache or background" -q -p no:cacheprovider
```

Expected: missing manager/status methods.

- [ ] **Step 3: Implement the single-worker manager**

```python
@dataclass(frozen=True)
class FastCacheJobSnapshot:
    workpiece_id: str
    library_revision: int
    geometry_profile_revision: int | None
    state: str  # queued, running, ready, failed, stale
    completed: int
    total: int
    elapsed_ms: float
    error: str | None

```

Add `FastCacheJobManager` with:

- `schedule(*, workpiece_id: str, library_revision: int, geometry_profile_revision: int | None, build: Callable[[Callable[[dict[str, Any]], None]], FastRuntimeCache], publish: Callable[[FastRuntimeCache], bool]) -> FastCacheJobSnapshot`
- `snapshot(workpiece_id: str) -> FastCacheJobSnapshot | None`
- `shutdown() -> None`

Use one `ThreadPoolExecutor(max_workers=1)`, deduplicate by `(workpiece_id, library_revision, geometry_profile_revision)`, and make `publish` return `False` when the captured catalog revision is stale.

- [ ] **Step 4: Integrate catalog lifecycle without holding the catalog lock during model work**

Add `fast_jobs` to `WorkpieceCatalog.__init__`. Build callbacks capture immutable record/profile values, release `_lock`, run model work, then reacquire `_lock` only to compare revisions and replace the snapshot with `dataclasses.replace(current.cache, fast_runtime=result)`.

`list_workpiece_summaries()` adds:

```python
"fast_cache": {
    "state": "ready" | "queued" | "running" | "failed" | "not_ready",
    "completed": int,
    "total": int,
    "elapsed_ms": float,
    "error": str | None,
}
```

Existing legacy recovery remains available. In fast mode, a valid base cache activates immediately and a missing/invalid fast cache is scheduled in the background.

- [ ] **Step 5: Make geometry publication and rollback build the matching fast revision before pointer swap**

`GeometryMaskProfiles.publish` and rollback must require a fast candidate whose profile revision matches the proposed pointer. Publication remains atomic: if fast build fails, active profile/cache stay unchanged. Background recovery applies only after a previously published revision is loaded at startup.

- [ ] **Step 6: Run lifecycle regressions**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_fast_cache_jobs.py tests/test_workpiece_catalog.py tests/test_geometry_mask_profiles.py tests/test_template_evolution.py -q -p no:cacheprovider
```

Expected: all selected tests pass with no catalog-lock regression.

- [ ] **Step 7: Commit Task 5**

```powershell
git add src/fast_cache_jobs.py src/workpiece_catalog.py src/geometry_mask_profiles.py tests/test_fast_cache_jobs.py tests/test_workpiece_catalog.py tests/test_geometry_mask_profiles.py
git commit -m "feat: rebuild fast caches in the background"
```

---

### Task 6: Structured Build Progress and Service Protocol

**Files:**
- Modify: `src/workpiece_library.py:70-78,319-430`
- Modify: `tests/test_workpiece_library.py`
- Modify: `src/orientation_tcp_service.py:20-45,255-700,925-990`
- Modify: `tests/test_orientation_tcp_service.py`
- Modify: `tests/test_orientation_service_environment.py`

**Interfaces:**
- Consumes: inference modes, fast cache status, and structured build progress.
- Produces: CLI `--inference-mode`, stable fast-path protocol errors, and fast cache fields in register/list responses.
- Consumed later by: Qt.

- [ ] **Step 1: Write failing protocol and progress tests**

Add these exact cases:

- `test_parser_accepts_legacy_fast_geometry_and_compare`: parameterize the three accepted values and assert `args.inference_mode` matches each value.
- `test_parser_rejects_unknown_inference_mode`: parse `--inference-mode automatic` and assert `SystemExit` code `2`.
- `test_fast_cache_not_ready_maps_to_stable_error_code`: make catalog prediction raise the classifier's cache-not-ready exception and assert the service response is `ok=false`, `error_code="FAST_CACHE_NOT_READY"`, and contains no label.
- `test_fast_feature_invalid_maps_to_stable_error_code`: return a non-finite fast embedding and assert `ok=false`, `error_code="FAST_FEATURE_INVALID"`, and contains no label.
- `test_fast_cache_build_failure_maps_to_stable_error_code`: expose a failed fast-cache job with no usable current cache and assert `ok=false`, `error_code="FAST_CACHE_BUILD_FAILED"`, and the operator message includes the job error.
- `test_unvalidated_fast_head_remains_a_prediction_with_review`: return a direction with `FAST_MODE_NOT_VALIDATED`; assert `ok=true`, the label remains present, and `needs_review=true`.
- `test_register_forwards_fast_original_augmentation_and_ridge_progress`: feed three structured callbacks and assert the service forwards the same phase names and monotonic completed/total values without rewriting augmentation totals.
- `test_list_workpieces_contains_fast_cache_status`: return a running job snapshot and assert the JSON summary preserves state, counts, elapsed time, and nullable error.
- `test_shutdown_stops_fast_cache_jobs`: invoke service shutdown and assert the fake job manager's shutdown method is called once before the listener close method.

Add a workpiece-library test that passes both legacy `callback(label, completed, total)` and new `callback(event_dict)` builder events and asserts the outward event is always a dictionary with `phase`, `completed`, and `total`.

- [ ] **Step 2: Run protocol tests and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_library.py tests/test_orientation_tcp_service.py tests/test_orientation_service_environment.py -k "fast or progress or parser" -q -p no:cacheprovider
```

Expected: new option/error/status fields are absent.

- [ ] **Step 3: Add inference mode and stable errors**

The parser must expose:

```python
parser.add_argument(
    "--inference-mode",
    choices=("legacy", "fast_geometry", "compare"),
    default="legacy",
)
```

Pass the value into `_load_runtime` and `OrientationClassifier.load`. Add service mappings:

```python
FAST_CACHE_NOT_READY
FAST_CACHE_REVISION_MISMATCH
FAST_CACHE_BUILD_FAILED
FAST_GEOMETRY_LOW_CONFIDENCE
FAST_CLASSIFIER_LOW_MARGIN
FAST_FEATURE_INVALID
FAST_MODE_NOT_VALIDATED
```

Missing, mismatched, or failed cache state and invalid features produce error responses. Geometry low confidence, Ridge low margin, and unvalidated per-workpiece head remain `ok=true` predictions with review fields.

- [ ] **Step 4: Return actual build/status information**

- `register` response adds `fast_cache_state`, `fast_cache_revision`, `training_summary`, and actual template counts.
- `list_workpieces` summaries include the catalog `fast_cache` object.
- The progress adapter maps fast phases without pretending augmentation count equals original template count.
- `request_shutdown` shuts down fast jobs before closing the listener.

- [ ] **Step 5: Run all service/library tests**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_library.py tests/test_workpiece_catalog.py tests/test_orientation_tcp_service.py tests/test_orientation_service_environment.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 6**

```powershell
git add src/workpiece_library.py src/orientation_tcp_service.py tests/test_workpiece_library.py tests/test_orientation_tcp_service.py tests/test_orientation_service_environment.py
git commit -m "feat: expose fast inference protocol state"
```

---

### Task 7: Qt Configuration and Fast-Path Evidence

**Files:**
- Modify: `qt_app/appconfig.h`
- Modify: `qt_app/appconfig.cpp`
- Modify: `qt_app/app_config.json.example`
- Modify: `qt_app/backendprocessmanager.cpp`
- Modify: `qt_app/inspectionpage.cpp`
- Modify: `qt_app/workpiecelibrarypage.cpp`
- Modify: `qt_app/tests/test_appconfig.cpp`
- Modify: `qt_app/tests/test_backendprocessmanager.cpp`
- Modify: `qt_app/tests/test_inspectionpage.cpp`
- Modify: `qt_app/tests/test_workpiecelibrarypage.cpp`
- Modify: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Consumes: service `inference_engine`, `geometry_status`, `decision_margin`, `review_reason`, `timings_ms`, and `fast_cache` status.
- Produces: validated desktop config, backend argument, and operator-readable fast evidence/status.

- [ ] **Step 1: Write failing Qt tests**

Required assertions:

```cpp
void acceptsInferenceModes_data() {
    QTest::addColumn<QString>("mode");
    QTest::newRow("legacy") << QStringLiteral("legacy");
    QTest::newRow("fast") << QStringLiteral("fast_geometry");
    QTest::newRow("compare") << QStringLiteral("compare");
}

void fastPredictionShowsRidgeAndGeometryWithoutLocalEvidence() {
    InspectionPage page;
    page.handleBackendResponse(QStringLiteral("predict"), QJsonObject{
        {"label", "front"}, {"inference_engine", "fast_geometry"},
        {"decision_source", "fast_ridge"}, {"decision_margin", 0.183},
        {"geometry_status", "active"}, {"needs_review", false},
        {"timings_ms", QJsonObject{{"geometry_fit", 3.1}, {"global_batch", 7.2}, {"total", 14.8}}}
    });
    const QString evidence = page.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))->toPlainText();
    QVERIFY(evidence.contains(QStringLiteral("快速判别")));
    QVERIFY(evidence.contains(QStringLiteral("几何规则：已应用")));
    QVERIFY(!evidence.contains(QStringLiteral("局部匹配")));
}
```

Also test:

- backend arguments contain `--inference-mode fast_geometry`;
- missing config defaults to `legacy` until final acceptance switches the example;
- invalid mode is rejected;
- low geometry/Ridge response displays the provided Chinese review reason;
- raw evidence displays engine, decision margin, fast cache revision, and stage timings;
- library page maps `fast_originals`, `fast_augmentation`, and `fast_ridge` to Chinese text and displays ready/running/failed cache states.

- [ ] **Step 2: Run focused Qt tests and verify RED**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_qt5_tests.ps1 -Targets test_appconfig,test_backendprocessmanager,test_inspectionpage,test_workpiecelibrarypage
```

Expected: assertions for inference mode and fast evidence fail.

- [ ] **Step 3: Implement configuration and argument forwarding**

Add `QString inferenceMode = QStringLiteral("legacy");` to `AppConfig`. Accept only `legacy`, `fast_geometry`, and `compare`. Append `--inference-mode` and the configured value in `BackendProcessManager::backendArguments()`.

- [ ] **Step 4: Render fast evidence without inventing local evidence**

In `InspectionPage::evidenceSummary` branch on `inference_engine`:

```cpp
if (engine == QStringLiteral("fast_geometry")) {
    lines << QStringLiteral("快速判别：%1（采用此结果）").arg(orientationText(record.label));
    lines << QStringLiteral("几何规则：%1").arg(geometryDescription(response));
} else {
    // retain the existing global/local legacy wording
}
```

`rawEvidence` prints decision margin and each timing key but no fake `local_scores`. `geometryDescription` reads the new top-level `geometry_status` first, then falls back to legacy `geometry_mask.status`.

- [ ] **Step 5: Render cache phases/status in the existing library page**

Extend `phaseText` with:

```cpp
fast_originals -> 提取快速特征
fast_augmentation -> 生成旋转增强
fast_ridge -> 构建快速判别器
```

Use the existing task status and details areas; do not add a new page or redesign unrelated controls.

- [ ] **Step 6: Run Qt focused and main-window tests**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_qt5_tests.ps1 -Targets test_appconfig,test_backendprocessmanager,test_inspectionpage,test_workpiecelibrarypage,test_mainwindow
```

Expected: all selected Qt tests pass.

- [ ] **Step 7: Commit Task 7**

```powershell
git add qt_app/appconfig.h qt_app/appconfig.cpp qt_app/app_config.json.example qt_app/backendprocessmanager.cpp qt_app/inspectionpage.cpp qt_app/workpiecelibrarypage.cpp qt_app/tests
git commit -m "feat: show fast geometry inference state in Qt"
```

---

### Task 8: Reproducible Compare and Performance Gate

**Files:**
- Create: `scripts/benchmark_fast_geometry_inference.py`
- Create: `tests/test_benchmark_fast_geometry_inference.py`
- Reuse: `scripts/benchmark_geometry_rule_inference.py` dataset/cache helpers where their inputs are identical.

**Interfaces:**
- Consumes: explicit legacy and fast classifier instances, the deterministic M1/M2/M7 `CaseSpec` selection from `benchmark_adaptive_local_search.py`, and stage timings.
- Produces: one JSON report with per-image outputs, aggregate accuracy/review rate, and latency percentiles.

- [ ] **Step 1: Write failing benchmark-statistics tests**

```python
def test_percentiles_use_all_samples_without_trimming():
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 100.0])
    assert summary["samples"] == 5
    assert summary["max_ms"] == 100.0
    assert summary["p95_ms"] == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))


```

Also add:

- `test_gate_rejects_added_error_high_review_or_slow_p95`: construct four summaries (passing, one added error, review rate `0.051`, P95 `25.001`) and assert only the first passes while each failure reports its exact gate name.
- `test_report_keeps_every_per_image_legacy_and_fast_result`: benchmark three fake queries and assert the JSON has three unique accuracy rows, each with both engine outputs and every measured repeat; assert no row is silently dropped.
- `test_warmup_samples_are_not_in_measured_distribution`: use a fake clock/engine returning five warmup sentinels followed by measured values, then assert `samples` and percentiles contain only measured values.
- `test_minimum_samples_cycles_queries_deterministically`: provide three query identities with `minimum_measured_samples=8`, assert the measured order is `0,1,2,0,1,2,0,1`, and assert accuracy still contains three unique rows rather than eight.
- `test_added_error_is_pairwise_legacy_correct_fast_wrong`: provide one legacy-wrong/fast-correct row, one both-wrong row, and one legacy-correct/fast-wrong row; assert `added_errors == 1` and the report names only the final image.
- `test_case_specs_are_exactly_m1_m2_m7_and_disjoint`: inject temporary M1 persisted templates plus deterministic M2/M7 folders into the reused selection helper, assert case names are exactly `M1`, `M2`, `M7`, and assert template/query SHA-256 intersection is empty for every case.
- `test_enabled_rule_effect_gate_requires_a_real_mask_or_embedding_change`: feed one enabled inside rule with identical raw/masked fake embeddings and assert the rule-effect gate fails; then change the masked embedding and assert it passes with the affected direction and query identity recorded.

- [ ] **Step 2: Run benchmark unit tests and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_benchmark_fast_geometry_inference.py -q -p no:cacheprovider
```

Expected: benchmark module/functions are missing.

- [ ] **Step 3: Implement the benchmark CLI and non-negotiable gates**

Required arguments:

```text
--project-root
--model-dir
--library-dir
--m1-workpiece-id
--warmup (default 50)
--repeats (default 1)
--minimum-measured-samples (default 1000)
--output
--max-p95-ms (default 25.0)
--max-added-errors (default 0)
--max-review-rate (default 0.05)
```

Call the existing deterministic `_build_case_specs(project_root, library_dir, m1_workpiece_id)` selection path so M1 uses its persisted 28+28 library while M2/M7 use the established seeded in-memory 5+5 split. Persist the returned selection inventory, including absolute image path, SHA-256, expected orientation, and dataset, into the output report. Assert the selected case set is exactly `M1`, `M2`, and `M7`; fail before inference if any case is absent, any path is unreadable, or any template/query content overlaps.

Build reproducible overlapping report groups from measured metadata rather than filename guesses: inside/outside interference groups come from the published rule names and directions actually applied; rotation bins come from fitted angle; offset bins from normalized fitted-center displacement; crop-size bins from decoded width/height. Preserve the raw values in every query row so group membership is auditable.

Follow the existing benchmark's parent/child process pattern: one `legacy` child writes accuracy rows and exits; a separate `fast_geometry` child loads no local stack, performs its warmups, writes accuracy plus measured latency rows, and exits; the parent joins rows by immutable query identity and evaluates gates. This keeps Torch/LightGlue GPU allocation out of the fast latency process. Cycle the fixed query order until `minimum_measured_samples` is reached, but count accuracy once per unique query. For every enabled rule direction, require at least one successful query to produce a non-empty mask and a raw-versus-masked embedding distance greater than `1e-6`; report the affected identities. Record every result and timing without trimming outliers. Exit non-zero when any gate fails.

- [ ] **Step 4: Run benchmark unit tests**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_benchmark_fast_geometry_inference.py tests/test_benchmark_adaptive_local_search.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 8**

```powershell
git add scripts/benchmark_fast_geometry_inference.py tests/test_benchmark_fast_geometry_inference.py
git commit -m "test: add fast geometry acceptance benchmark"
```

---

### Task 9: Production Accuracy and 1000-Sample Latency Checkpoint

**Files:**
- Modify only if evidence requires it: `src/fast_geometry.py`, `src/fast_ridge.py`, `src/fast_orientation.py`, and their focused tests.
- Create: `runtime_reports/fast-geometry-acceptance.json` as a local non-committed artifact.

**Interfaces:**
- Consumes: completed benchmark script and the existing fixed M1/M2/M7 held-out manifest.
- Produces: evidence deciding whether implementation may proceed to default activation.

- [ ] **Step 1: Materialize and audit the fixed M1/M2/M7 selection**

Use the exact M1/M2/M7 `CaseSpec` selection from `benchmark_adaptive_local_search.py`; assert path and content fingerprints do not overlap and save the selection inventory inside the result JSON. Run every unique selected non-template image once for accuracy, report the exact count rather than hard-coding a historical count, and cycle those queries deterministically until at least 1000 latency timings are collected. Repetition is allowed for latency distribution only; accuracy counts each unique image once.

- [ ] **Step 2: Run the production-model benchmark**

```powershell
E:\python\anaconda3\envs\shitu\python.exe scripts\benchmark_fast_geometry_inference.py `
  --project-root E:\Project\wang\pp_813 `
  --model-dir E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer `
  --library-dir E:\Project\wang\pp_813\runtime_library `
  --m1-workpiece-id 31f082d1a04e486b9345846f4d585033 `
  --warmup 50 `
  --minimum-measured-samples 1000 `
  --max-p95-ms 25 `
  --max-added-errors 0 `
  --max-review-rate 0.05 `
  --output runtime_reports\fast-geometry-acceptance.json
```

The report must contain the persisted M1 artifact fingerprint and the deterministic in-memory M2/M7 template/query inventories emitted by the existing benchmark helper; the script must fail if any of these three definitions is absent.

- [ ] **Step 3: Inspect stage budgets and apply only evidence-backed tuning**

Permitted tuning in this task is limited to:

- 256-pixel candidate-window width/scale constants in `fast_geometry.py`;
- feather sigma that preserves mask parity and avoids hard edges;
- PP-ShiTu build batch size, not online batch size 3;
- Ridge regularization grid/review threshold only when cross-validation evidence improves without adding errors.

For every tuning change, first add a focused failing regression using an affected real image copied only into the repository's established test-fixture location, then make the smallest constant/logic change and rerun Tasks 1--3 tests. Do not change PP-ShiTu, add a local model, remove slow samples, or relax gates.

- [ ] **Step 4: Apply the checkpoint decision**

- If `added_errors == 0`, `review_rate <= 0.05`, `p95_ms <= 25.0`, and every enabled rule direction passes the real-effect gate, continue to Task 10.
- If any condition fails after permitted tuning, stop implementation, leave the default as `legacy`, and write the failing stage/data evidence in the verification report. Do not claim completion.

- [ ] **Step 5: Commit evidence-backed tuning, if any**

```powershell
git add src/fast_geometry.py src/fast_ridge.py src/fast_orientation.py tests/test_fast_geometry.py tests/test_fast_ridge.py tests/test_fast_orientation.py
git commit -m "perf: tune fast geometry from production evidence"
```

Skip this commit only when no tracked source or test changed.

---

### Task 10: Full Regression, Default Activation, and Verification Report

**Files:**
- Modify: `qt_app/app_config.json.example` only if Task 9 gates passed.
- Create: `docs/verification/lightweight-geometry-fast-path-results.md`
- Modify: any test inventory documentation that explicitly enumerates test files.

**Interfaces:**
- Consumes: Task 9 acceptance JSON and all task-level tests.
- Produces: release-ready default selection, reproducible verification report, and clean branch.

- [ ] **Step 1: Run the complete Python project test suite**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q -p no:cacheprovider
```

Expected: all project tests pass; production-only tests may skip only when their documented environment flag is absent.

- [ ] **Step 2: Run all Qt tests**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_qt5_tests.ps1
```

Expected: every listed Qt target passes with zero failures.

- [ ] **Step 3: Build the Qt 5.14.2 release application**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_qt5.ps1
```

Expected: `qt_app\build-release\release\workpiece_orientation.exe` exists.

- [ ] **Step 4: Switch the example default only after the checkpoint passes**

If and only if Task 9 passed all three gates, set:

```json
"inference_mode": "fast_geometry"
```

in `qt_app/app_config.json.example`. Do not overwrite the user's local `qt_app/app_config.json`; report the one-line setting they must choose after acceptance.

- [ ] **Step 5: Write the final verification report from actual outputs**

The report must include:

- branch and commit;
- hardware/runtime/model fingerprint;
- exact unique query counts and template/query overlap audit;
- legacy versus fast accuracy per M1/M2/M7 and interference group;
- changed predictions with absolute image paths;
- review count/rate and reason counts;
- mean/P50/P95/P99/max for total and every stage;
- per-direction geometry real-effect rows with mask size and raw-versus-masked embedding distance;
- 1+1, 5+10, 10+15, and 35+35 build time, augmentation count, and cache size;
- proof that fast online call counters for ALIKED, LightGlue, and ORB are zero;
- all Python/Qt/release build results;
- remaining limitations and rollback instruction.

- [ ] **Step 6: Verify documentation and working tree**

```powershell
rg -n "TBD|TODO|FIXME|待定" docs/verification/lightweight-geometry-fast-path-results.md
git diff --check
git status --short
```

Expected: placeholder search returns no matches, `git diff --check` is clean, and only intended report/default files remain before commit.

- [ ] **Step 7: Commit Task 10**

```powershell
git add qt_app/app_config.json.example docs/verification/lightweight-geometry-fast-path-results.md
git commit -m "docs: verify lightweight geometry fast path"
```

If the default was not activated, omit `qt_app/app_config.json.example` from `git add` and state the failed gate in the commit report.

---

## Final Success Checklist

- [ ] Every enabled rule is attempted on every readable fast-mode input.
- [ ] Front/back geometry remain independently fitted from one shared context.
- [ ] Query PP-ShiTu is one batch of exactly raw/front/back images.
- [ ] Fast online call counters show zero ALIKED, LightGlue, ORB, or local matcher calls.
- [ ] Low geometry/Ridge confidence returns a label plus review; invalid cache/features return stable errors.
- [ ] 1+1 and unequal counts build without truncation; warnings remain visible.
- [ ] Old 5+5 libraries restore and rebuild fast caches without blocking an otherwise valid base snapshot.
- [ ] Publish, rollback, append, recycle, restore, purge, and stale job behavior are revision-safe.
- [ ] Fixed acceptance set adds zero errors and review rate is at most 5%.
- [ ] Every enabled rule direction has measured non-empty-mask and embedding-change evidence.
- [ ] Warm backend single-image total P95 is at most 25 ms over at least 1000 measured calls.
- [ ] Full Python, full Qt, and Qt release build pass.
- [ ] Default remains legacy unless every gate passes.
