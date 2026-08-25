# Geometry Fusion Regression Fix Verification

Date: 2026-08-24

## Root Cause

The inside-ignore rule was applied to local matching by inpainting the ignored region and re-extracting ALIKED descriptors from the synthetic image. On `front:22.png`, this changed the local prediction from `front` to `back`, and the local margin exceeded the existing override threshold. PP-ShiTu's geometry-masked global branch remained correctly classified as `front`.

The leave-one-out validator also counted every candidate error as `correct_to_wrong` without first checking the raw baseline result.

## Implemented Behavior

- PP-ShiTu continues to use direction-specific geometry-processed images.
- ALIKED query features are extracted once from the original image.
- Template local features reuse the raw cache.
- Both inside and outside ignored regions remove local keypoints.
- Paired leave-one-out validation records real baseline-to-candidate transitions.
- Fusion regressions include the affected template and branch evidence.
- Qt keeps publication blockers visible while the operator inspects fitted boundaries.
- Existing numerical fusion thresholds and model files are unchanged.

## Automated Tests

- `tests/test_orientation_classifier.py`: 29 passed.
- `tests/test_geometry_mask_profiles.py`: 45 passed.
- Related Python regression set (`geometry_calibration`, `orientation_classifier`, `geometry_mask_profiles`, `orientation_tcp_service`, `workpiece_catalog`): 177 passed.
- Full Python test suite: 254 passed, 3 skipped.
- Qt 5.14.2 `test_geometrymaskmanager`: 28 passed.
- Full Qt 5.14.2 desktop application build: succeeded.

## Real Workpiece Acceptance

Workpiece: `31f082d1a04e486b9345846f4d585033`

Templates: 28 front + 28 back

Draft revision: 11

Paired leave-one-out result:

- evaluated: 56
- skipped: 0
- correct to correct: 56
- correct to wrong: 0
- wrong to correct: 0
- wrong to wrong: 0
- changed predictions: none

`front:22.png` no longer produces a regression.

## Timing

- Geometry candidate-cache preparation: 2,352.2 ms for 56 templates.
- Paired leave-one-out validation: 98,954.3 ms for 56 templates.

Candidate preparation is faster because local template descriptors are reused from the raw cache rather than re-extracted from repaired images. Paired validation performs both raw and candidate predictions for every held-out sample, so it is intentionally more expensive than the previous candidate-only check. Production prediction does not add an extra model call; query-side ALIKED extraction is reduced from two passes to one pass.

## Operator Action

The persisted old validation job still contains the previous blocker. Restart the backend so it loads the updated Python code, reopen geometry-rule management, save the current draft if necessary, and run validation again. The application does not rewrite historical jobs silently.
