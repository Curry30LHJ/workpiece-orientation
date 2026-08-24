# Geometry Fusion Regression Fix Design

## Problem

The current geometry-rule validation reports `correct_to_wrong`, but the implementation only checks whether the candidate prediction matches the template label. It does not run the same held-out sample through the raw baseline, so the field can describe an absolute candidate error rather than a real baseline-correct-to-candidate-wrong regression.

The current workpiece also exposes a separate, real regression. On `front:22.png`, the raw baseline predicts `front`, and the geometry-masked PP-ShiTu global branch still predicts `front`. The local ALIKED/LightGlue branch predicts `back` with enough margin to override the global result, producing the only error in 56 leave-one-out samples. The UI shows only a generic blocker and then replaces the blocker diagnostics with the selected template's fit diagnostics.

## Confirmed Evidence

- Raw leave-one-out: 56/56 correct.
- Current geometry fusion: 55/56 correct.
- Geometry-masked global prediction: 56/56 correct.
- Only regression: `front:22.png`.
- `front:22.png` candidate global prediction: `front`, margin `0.02134`.
- `front:22.png` candidate local prediction: `back`, margin `3.87248`.
- Current final decision: `back`, source `local_override`.
- The fitted ignored area is about 48.1%; the rule does not fail geometrically.

## Considered Approaches

### 1. Remove or downgrade the blocker

Rejected. The current sample has a verified output regression. Allowing publication would knowingly reduce leave-one-out accuracy from 56/56 to 55/56.

### 2. Increase the existing local override threshold

Rejected. A global threshold change would affect every workpiece and violates the requirement to preserve existing fusion thresholds. It also treats the symptom without measuring whether local evidence remains trustworthy after geometry processing.

### 3. Paired validation plus raw-image local feature masking

Selected. Validation will compare raw and candidate predictions for the same held-out sample. PP-ShiTu will continue to consume the geometry-processed image, while ALIKED/LightGlue will consume the original image and remove keypoints located in every ignored region. Diagnostics will state which branch changed the result.

## Design

### Paired leave-one-out validation

For each included template:

1. Remove the template from the raw cache and predict the unmodified image with the raw baseline.
2. Remove the corresponding template from the candidate geometry cache and predict with the candidate profile.
3. Record one of four transitions: `correct_to_correct`, `correct_to_wrong`, `wrong_to_correct`, or `wrong_to_wrong`.
4. A publication blocker is created only for `correct_to_wrong` transitions.

Each changed prediction records:

- template identifier and expected label;
- baseline final, global, and local predictions;
- candidate final, global, and local predictions;
- candidate decision source;
- global and local margins;
- a machine-readable cause.

The report keeps the existing `correct_to_wrong` field for API and stored-job compatibility. It adds transition counters and richer evidence without invalidating older 5+5 or geometry-validation records.

### Raw-image local feature masking

The existing PP-ShiTu and local fusion thresholds remain unchanged. Geometry processing is split by model semantics:

- PP-ShiTu receives the geometry-processed front and back image variants, as it does now.
- Template ALIKED features are taken from `raw_local_features` and filtered using the complete fitted `ignore_mask` for that template.
- Query ALIKED features are extracted once from the original query image, then filtered separately using the fitted front and back `ignore_mask` values.
- Both `inside` and `outside` ignored areas remove local keypoints. No local descriptor is extracted from an inpainted or filled synthetic region.

This makes the rule semantics consistent: an ignored region cannot contribute local keypoints, while descriptors outside the ignored region still come from the real image instead of being changed by inpainting context. It also reduces query-side local extraction from two passes to one pass.

The confirmed `front:22.png` probe changes from local `back` (`13.68` versus `9.81`) to local `front` (`16.08` versus `10.68`) with this processing, while the global result remains `front`. The final result is therefore restored without any threshold change.

For the current draft, the blocker remains until revalidation. After the paired validator and local-feature fix are applied, the profile can pass only if all 56 paired samples contain no correct-to-wrong transition.

### UI diagnostics

The Qt manager will retain blocker diagnostics separately from per-template geometry-fit diagnostics. A fusion regression row is highlighted using its `changed_predictions[].template_id`, and selecting it shows a concise Chinese explanation:

- baseline result;
- candidate global result;
- candidate local result;
- final result and decision source;
- why publication is blocked.

The generic instruction to exclude or redraw the template is not shown for a fusion regression, because the operator should not be told that the fitted boundary itself is necessarily wrong.

## Error Handling and Compatibility

- A missing or failed baseline/candidate prediction keeps the leave-one-out report `incomplete` and remains a publication blocker.
- Explicitly excluded templates remain excluded from paired evaluation.
- Older job documents without transition details still render using the generic diagnostic fallback.
- Existing API keys and publication checks remain valid.
- PP-ShiTu, ALIKED, LightGlue model files and existing numerical fusion thresholds are unchanged.
- No model retraining is required.

## Verification

- Unit test proving a candidate error that was already wrong in the baseline is not counted as `correct_to_wrong`.
- Unit test proving a baseline-correct/candidate-wrong transition is counted and includes branch evidence.
- Unit test proving template local features come from the raw cache and remove keypoints from all ignored regions.
- Unit test proving one raw query extraction is filtered separately for the front and back masks.
- Unit test proving ordinary baseline/local fusion behavior is unchanged.
- Qt test proving changed-prediction template IDs are highlighted and blocker text remains visible after template preview.
- Existing Python and Qt tests pass.
- Real 28+28 workpiece leave-one-out returns 56/56 with zero correct-to-wrong transitions.
