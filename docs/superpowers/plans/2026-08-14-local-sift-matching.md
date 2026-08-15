# Local SIFT Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic OpenCV SIFT + RANSAC front/back template-matching benchmark and compare it with the existing PP-ShiTuV2 5+5 baseline.

**Architecture:** `src/local_sift_matcher.py` owns center-ROI feature extraction and pair scoring. A separate evaluator uses the existing deterministic split utility, scores every held-out sample against both labels, applies a reject rule, and saves report rows plus diagnostic match images.

**Tech Stack:** Python 3.10, OpenCV 4.6 SIFT, NumPy, unittest.

## Global Constraints

- Use exactly five templates per label and seed `20260813`.
- Do not download a model or train a classifier.
- Score a full center ROI only; do not derive any ROI from test labels.
- Preserve `src/shitu_baseline.py` behavior.
- Treat an insufficient local match or a near-tie as `uncertain`, never a forced side decision.

---

### Task 1: Local SIFT pair score

**Files:**
- Create: `src/local_sift_matcher.py`
- Modify: `tests/test_shitu_baseline.py`

**Interfaces:**
- Produces: `extract_center_roi(image: np.ndarray, ratio: float = 0.8) -> np.ndarray`
- Produces: `score_pair(query: np.ndarray, template: np.ndarray, ratio: float = 0.8) -> dict[str, float]`
- `score_pair` returns `{"good_matches": float, "inliers": float, "coverage": float, "score": float}` where `score = inliers + coverage`.

- [ ] **Step 1: Write failing tests**

```python
from src.local_sift_matcher import extract_center_roi, score_pair

def test_extract_center_roi_discards_equal_border():
    image = np.zeros((100, 200), dtype=np.uint8)
    assert extract_center_roi(image, ratio=0.8).shape == (80, 160)

def test_score_pair_has_geometric_inliers_for_translated_pattern():
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(image, "M1", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, -5]]), (256, 256))
    result = score_pair(image, shifted)
    assert result["inliers"] >= 4
    assert result["score"] > 4
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m unittest tests.test_shitu_baseline -v`

Expected: FAIL because `src.local_sift_matcher` does not exist.

- [ ] **Step 3: Implement the minimum matcher**

```python
def score_pair(query, template, ratio=0.8):
    sift = cv2.SIFT_create()
    query_keypoints, query_descriptors = sift.detectAndCompute(extract_center_roi(query, ratio), None)
    template_keypoints, template_descriptors = sift.detectAndCompute(extract_center_roi(template, ratio), None)
    # Return zeros when either descriptor set is absent.
    # Apply BFMatcher knnMatch(k=2), Lowe ratio 0.75, then cv2.findHomography(..., cv2.RANSAC, 5.0).
    # coverage is the fraction of the query ROI bounding-box area covered by RANSAC inlier points.
```

- [ ] **Step 4: Run the tests to verify pass**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m unittest tests.test_shitu_baseline -v`

Expected: PASS.

### Task 2: Deterministic benchmark and diagnostics

**Files:**
- Create: `src/evaluate_local_sift.py`
- Create at runtime: `reports/local_sift/<dataset>/report.json`
- Create at runtime: `reports/local_sift/<dataset>/matches/*.png`

**Interfaces:**
- Consumes: `split_labels(data_dir, template_count=5, seed=20260813)` from `src.shitu_baseline`.
- Consumes: `score_pair(query, template) -> dict[str, float]` from `src.local_sift_matcher`.
- Produces: one JSON report per dataset with templates, rows, accuracy, forced-decision accuracy, rejects, and confusion counts.

- [ ] **Step 1: Write failing evaluator test**

```python
from src.evaluate_local_sift import decide_label

def test_decide_label_rejects_tie_and_low_score():
    assert decide_label({"0": 3.0, "1": 2.8}, min_score=4.0, min_margin=0.5)[0] == "uncertain"
    assert decide_label({"0": 8.0, "1": 7.7}, min_score=4.0, min_margin=0.5)[0] == "uncertain"
    assert decide_label({"0": 8.0, "1": 2.0}, min_score=4.0, min_margin=0.5)[0] == "0"
```

- [ ] **Step 2: Run the test to verify failure**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m unittest tests.test_shitu_baseline -v`

Expected: FAIL because `src.evaluate_local_sift` does not exist.

- [ ] **Step 3: Implement evaluator and match visualization**

```python
def decide_label(scores, min_score, min_margin):
    ordered = sorted(scores, key=scores.get, reverse=True)
    margin = scores[ordered[0]] - scores[ordered[1]]
    if scores[ordered[0]] < min_score or margin < min_margin:
        return "uncertain", margin
    return ordered[0], margin
```

For every held-out image, compare it against every template in both labels, retain the highest template score per label, record its source filename, and write a diagnostic `cv2.drawMatches` image for every M1 forced error or rejection.

- [ ] **Step 4: Run unit tests and the full three-dataset benchmark**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m unittest tests.test_shitu_baseline -v`

Run: `E:\python\anaconda3\envs\shitu\python.exe src\evaluate_local_sift.py data\1_M1 data\1_M2 data\1_M7`

Expected: all tests pass; each dataset has one report and M1 contains diagnostic images.

- [ ] **Step 5: Compare results**

Compare M1 forced-decision accuracy and error count with PP-ShiTuV2 baseline `155/177 (87.57%)`; inspect diagnostic matches to determine whether retained points lie on the workpiece rather than background.
