# Geometry rule editor redesign — verification results

Date: 2026-08-18

## Delivered behavior

- Reference templates are previewed without mutating a draft, and the fitting response contains ranked anchor/rule candidates, the selected candidate, object-relative geometry, and fit timing.
- Circle, ellipse, and rotated-rectangle gestures use a center-to-radius/axes/corner interaction. `None` is the initial canvas tool; zoom, pan, reset (`R`), and cancel (`Esc`) are available.
- Geometry is stored in the fitted object coordinate frame. The rule center is rotated with the fitted anchor, so a translated/rotated workpiece does not reuse reference-image pixels.
- Reference-preview seeds are interpreted in source-image pixel coordinates during candidate fitting; only the accepted result is converted to object-relative geometry. This keeps an inner-circle gesture from being scaled a second time and selected as the outer boundary.
- Per-template review states (`included`, `review`, `excluded`) are persisted. `review` blocks publication until resolved; `excluded` is reported and omitted only from the geometry candidate cache while raw templates remain available.
- Legacy positional annotation/cache recovery remains available. The first geometry publication records whether the previous source was legacy, and rollback can restore that legacy cache.
- The Qt manager shows candidates, per-template validation diagnostics, draft history controls, review state/reason, read-only active-version viewing, and preview error recovery.
- Switching templates or directions clears the previous fitted overlay, candidate selectors, and diagnostics, then automatically requests a fresh preview when the current direction has a valid seed; a copied direction without an anchor remains visibly unconfigured until it is reseeded.

## Automated checks

| Check | Result |
| --- | --- |
| `python -m pytest tests/test_geometry_calibration.py -q` | **12 passed** |
| `python -m pytest tests/test_geometry_mask_profiles.py -k "reference_seed or invalid_editor_metadata" -q` | **6 passed** |
| `python -m pytest tests/test_orientation_tcp_service.py -k "preview_geometry_mask_rule" -q` | **8 passed** |
| Qt 5.14.2 MSVC release build (`workpiece_orientation.pro`) | **passed** |
| Qt geometry-rule-canvas test target compilation | **passed** |
| Qt geometry-mask-manager test target compilation | **passed** |
| Python syntax check for all changed Python modules/tests | **passed** |
| `git diff --check` | **passed** |

The Python tests that require pytest's `tmp_path` fixture could not complete in this Windows session because the configured temporary roots reject `.lock` creation (`PermissionError`/`WinError 5`). This is an environment ACL problem; the direct geometry/profile/classifier/catalog checks were run with workspace-owned temporary paths and passed. GUI test executables also enumerate correctly, but the session has no desktop display service for a reliable Qt widget run.

## Timing sample

On the local CPU with a 256×256 synthetic workpiece and one inside-circle rule:

- `fit_reference`: median **0.81 ms**, p95 **3.00 ms** over ten warm samples.
- Geometry fitting scales linearly with the number of templates: approximately **2.49 ms** for 1, **17.48 ms** for 10, and **66.08 ms** for 35 templates in the same synthetic run (about **1.8–2.5 ms/template**).

These numbers cover contour fitting only. Production cache validation still includes PP‑ShiTuV2 and ALIKED/LightGlue extraction; those model thresholds and matching implementations were not changed.
