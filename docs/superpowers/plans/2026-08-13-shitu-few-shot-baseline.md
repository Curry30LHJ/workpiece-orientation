# PP-ShiTuV2 Few-Shot Baseline Implementation Plan

**Goal:** Test 5-image-per-side registration against held-out workpiece images.

**Architecture:** The evaluator deterministically selects five templates from each label folder, extracts PP-ShiTuV2 embeddings, and classifies every remaining image by maximum cosine similarity. It writes CSV and JSON reports without altering source images.

**Tech Stack:** Python, PaddlePaddle GPU, PaddleClas, NumPy, OpenCV, pytest.

### Task 1: Template split and similarity classifier

**Files:** `src/shitu_baseline.py`, `tests/test_shitu_baseline.py`

1. Write tests for five-template deterministic label splits and maximum-cosine classification.
2. Verify the tests fail because the module does not exist.
3. Implement `split_labels()` and `classify_embedding()`.
4. Verify the tests pass.

### Task 2: GPU evaluator and reports

**Files:** `src/evaluate_shitu_baseline.py`, `src/shitu_baseline.py`, `tests/test_shitu_baseline.py`

1. Write a failing test for report accuracy, confusion counts, and low-confidence counts.
2. Implement `build_report()` plus CLI evaluation using the local PP-ShiTuV2 model.
3. Write CSV/JSON below `reports/<part-id>/`; do not write to `data/`.
4. Run `data/1_M7` with seed `20260813`, five templates per side, and margin threshold `0.05`.
