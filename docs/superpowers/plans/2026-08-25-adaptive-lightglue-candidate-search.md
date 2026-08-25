# Adaptive LightGlue Candidate Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 PP-ShiTuV2、ALIKED、LightGlue、模型权重、模板缓存格式和现有融合阈值的前提下，以保守的 Top-5 → Top-10 → 全量候选搜索降低 LightGlue 调用次数，并保留一键切回旧全量行为的进程级安全开关。

**Architecture:** PP-ShiTu 全局向量继续产生原有正反面分数，同时在每个方向内部为局部模板排序。低全局间隔查询按 Top-5、Top-10、全量逐级补算 LightGlue，且仅当现有 `_fuse_scores` 返回 `needs_review=false` 时提前结束；高全局间隔和 `exhaustive` 模式始终覆盖全部模板。候选搜索集中在 `OrientationClassifier` 内部，TCP 与 Qt 只负责把固定生命周期的 `local_search_mode` 传入后端；独立验收工具在两个独立 Python 进程中比较自适应与全量结果。

**Tech Stack:** Python 3.10、NumPy、OpenCV、PaddlePaddle/PaddleClas PP-ShiTuV2、PyTorch、ALIKED、LightGlue、pytest、JSON-lines TCP、Qt 5.14.2 Widgets/Test、C++17、qmake、MSVC 2019、Windows PowerShell。

## Global Constraints

- 保持 `GLOBAL_MARGIN_THRESHOLD = 0.05`、`LOCAL_MIN_SCORE = 4.0`、`LOCAL_MIN_MARGIN = 0.5`、`LOCAL_OVERRIDE_MARGIN = 3.0` 原值不变。
- 不修改 PP-ShiTuV2、ALIKED、LightGlue 的模型、权重、关键点上限、输入尺度或融合公式；不需要重新训练。
- 模板缓存格式与 `TEMPLATE_CACHE_FORMAT_VERSION = 2` 保持不变，现有工件库无需重建。
- 正反模板数量可以不同，Top-5/Top-10 只是上限；任何候选阶段均不得静默截断全局向量或局部模板。
- 全局向量数与局部模板数不一致、某一方向没有局部模板时，必须抛出明确的 `OrientationClassifierError`。
- `adaptive` 与 `exhaustive` 模式在一次后端进程生命周期内固定；修改 Qt 配置后通过现有“重启后端”生效。
- 正式默认值只有在 M1、M2、M7 固定验收集上逐图 `label` 和 `needs_review` 完全一致时才保留 `adaptive`；任一不一致则交付默认值改为 `exhaustive`。
- 性能毫秒数不写成单元测试硬断言；真实报告记录 P50/P95、阶段分布、平均 LightGlue 调用数和逐图差异。
- 当前工作树已有大量未提交修改。每个提交前先查看目标文件差异，使用 `git add -p -- <files>` 只暂存本任务新增片段，再执行 `git diff --cached --check` 和 `git diff --cached`；不得整文件暂存已有无关改动。

---

## File Map

- `src/orientation_classifier.py`：定义搜索模式、候选对齐校验、增量 LightGlue 调度、内部验收追踪，并把搜索接入 baseline、几何路径和几何失败回退。
- `tests/test_orientation_classifier.py`：覆盖排序、阶段扩展、不重复匹配、数量不等、损坏缓存、全量兼容、诊断字段和分段耗时。
- `src/orientation_tcp_service.py`：解析 `--local-search-mode`，在异步模型加载时把模式传入分类器。
- `tests/test_orientation_tcp_service.py`：验证缺省值、合法/非法参数和运行时加载透传。
- `qt_app/appconfig.h`、`qt_app/appconfig.cpp`：读取可选 `local_search_mode`，合法值仅为 `adaptive`、`exhaustive`。
- `qt_app/app_config.json.example`：展示安全开关配置。
- `qt_app/backendprocessmanager.cpp`：启动后端时传递 `--local-search-mode`。
- `qt_app/tests/test_appconfig.cpp`、`qt_app/tests/test_backendprocessmanager.cpp`：验证 Qt 缺省值、非法值拒绝和启动参数。
- `scripts/benchmark_adaptive_local_search.py`：在独立子进程运行两个模式，构建固定 M1/M2/M7 验收集，输出逐图 JSON 与 Markdown 门禁报告。
- `tests/test_benchmark_adaptive_local_search.py`：只测试报告比较、门禁和 Markdown 生成，不加载真实模型。
- `docs/verification/adaptive-lightglue-candidate-search-results.md`：由验收脚本用真实运行值生成，记录门禁结论和性能。

---

### Task 1: Implement the incremental local-search core

**Files:**
- Modify: `src/orientation_classifier.py:32-192,830-878`
- Modify: `tests/test_orientation_classifier.py:1-110,619-780`

**Interfaces:**
- Produces: `LOCAL_SEARCH_MODES = ("adaptive", "exhaustive")` and `DEFAULT_LOCAL_SEARCH_MODE = "adaptive"`.
- Produces: `LocalSearchResult(scores: dict[str, float], diagnostics: dict[str, object], matching_ms: float, trace: dict[str, object])`.
- Produces: `OrientationClassifier(global_predictor, extractor, matcher, device, *, local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE)` and `OrientationClassifier.load(project_root, model_dir, *, local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE)`.
- Produces: `_search_local(*, query_features_by_label: Mapping[str, dict[str, Any]], global_vectors: Mapping[str, np.ndarray], query_embeddings: Mapping[str, np.ndarray], local_features: Mapping[str, Sequence[dict[str, Any]]], image_shape: tuple[int, int], global_scores: dict[str, float]) -> LocalSearchResult`.
- Preserves: `_fuse_scores(global_scores, local_scores, started, *, geometry_mask=None)` remains the only authority for label, override and `needs_review` semantics.

- [ ] **Step 1: Add failing candidate-order and Top-5 stop tests**

Add the following helpers and tests to `tests/test_orientation_classifier.py`:

```python
def make_search_inputs(front_count, back_count, score_for):
    global_vectors = {}
    local_features = {}
    for label, count in (("front", front_count), ("back", back_count)):
        similarities = np.linspace(1.0, 0.1, count, dtype=np.float32)
        global_vectors[label] = similarities.reshape(-1, 1)
        local_features[label] = [
            {"label": label, "index": index, "score": float(score_for(label, index))}
            for index in range(count)
        ]
    return global_vectors, local_features


def run_local_search(classifier, global_vectors, local_features, global_scores):
    return classifier._search_local(
        query_features_by_label={
            "front": {"label": "front"},
            "back": {"label": "back"},
        },
        global_vectors=global_vectors,
        query_embeddings={
            "front": np.asarray([1.0], dtype=np.float32),
            "back": np.asarray([1.0], dtype=np.float32),
        },
        local_features=local_features,
        image_shape=(8, 8),
        global_scores=global_scores,
    )


def test_adaptive_local_search_stops_at_top5_without_reordering_features(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13, lambda label, index: 8.0 if label == "front" else 5.0
    )

    def score_pair(query, template, image_shape, matcher):
        calls.append((template["label"], template["index"]))
        return {"score": template["score"]}

    classifier._score_feature_pair = score_pair
    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.scores == {"front": 8.0, "back": 5.0}
    assert result.diagnostics["stage"] == "top5"
    assert result.diagnostics["matched_counts"] == {"front": 5, "back": 5}
    assert result.diagnostics["available_counts"] == {"front": 12, "back": 13}
    assert result.diagnostics["expanded_because"] is None
    assert result.diagnostics["exhaustive"] is False
    assert calls == [
        (label, index)
        for label in ("front", "back")
        for index in range(5)
    ]
    assert result.trace["ranked_indices"]["front"] == list(range(12))
```

- [ ] **Step 2: Run the focused test and confirm the missing interface fails**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py -k "adaptive_local_search_stops_at_top5" -q -p no:cacheprovider --basetemp=pytest-adaptive-search-core-red
```

Expected: FAIL because `_search_local` and `LocalSearchResult` do not exist yet.

- [ ] **Step 3: Add mode validation and result types**

Add near the existing classifier constants and dataclasses:

```python
LOCAL_SEARCH_MODES = ("adaptive", "exhaustive")
DEFAULT_LOCAL_SEARCH_MODE = "adaptive"
LOCAL_SEARCH_STAGE_LIMITS = (("top5", 5), ("top10", 10))


@dataclass(frozen=True)
class LocalSearchResult:
    scores: dict[str, float]
    diagnostics: dict[str, object]
    matching_ms: float
    trace: dict[str, object]


def _validate_local_search_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in LOCAL_SEARCH_MODES:
        raise ValueError(
            "local_search_mode must be 'adaptive' or 'exhaustive'"
        )
    return mode
```

Extend the constructor and loader without changing model construction:

```python
def __init__(
    self,
    global_predictor,
    extractor,
    matcher,
    device,
    *,
    extract_features_fn=None,
    score_feature_pair_fn=None,
    geometry_calibrator=None,
    model_gate=None,
    local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
) -> None:
    self.local_search_mode = _validate_local_search_mode(local_search_mode)
```

```python
@classmethod
def load(
    cls,
    project_root: Path,
    model_dir: Path,
    *,
    local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
) -> "OrientationClassifier":
    return cls(
        global_predictor,
        extractor,
        matcher,
        device,
        geometry_calibrator=GeometryCalibrator(),
        local_search_mode=local_search_mode,
    )
```

Retain all existing constructor assignments and the existing Torch-before-Paddle model loading order around these additions.

- [ ] **Step 4: Implement stable ranking, alignment validation and incremental scoring**

Replace the all-candidate `_score_local` implementation with focused helpers. Use stable sorting so equal PP-ShiTu similarities preserve cache order, move each direction’s query tensors once, and move each selected template only when first scored:

```python
@staticmethod
def _local_expansion_reason(
    global_scores: dict[str, float], local_scores: dict[str, float]
) -> str:
    local_prediction, local_margin = _decide_local_label(local_scores)
    if local_prediction == "uncertain":
        if max(local_scores.values()) < LOCAL_MIN_SCORE:
            return "local_uncertain"
        return "local_margin_low"
    global_prediction = max(global_scores, key=global_scores.get)
    if local_prediction != global_prediction:
        return "local_conflict"
    return "local_uncertain"


def _rank_local_candidates(
    self,
    global_vectors: Mapping[str, np.ndarray],
    query_embeddings: Mapping[str, np.ndarray],
    local_features: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[str, list[int]]:
    rankings = {}
    for label in ("front", "back"):
        vectors = np.asarray(global_vectors.get(label), dtype=np.float32)
        candidates = local_features.get(label)
        if vectors.ndim != 2 or candidates is None:
            raise OrientationClassifierError(
                f"template cache alignment error for {label}: missing vectors or local features"
            )
        if vectors.shape[0] == 0 or len(candidates) == 0:
            raise OrientationClassifierError(
                f"template cache alignment error for {label}: at least one local template is required"
            )
        if vectors.shape[0] != len(candidates):
            raise OrientationClassifierError(
                f"template cache alignment error for {label}: "
                f"{vectors.shape[0]} global vectors != {len(candidates)} local templates"
            )
        query = np.asarray(query_embeddings[label], dtype=np.float32)
        similarities = np.asarray(vectors @ query, dtype=np.float32).reshape(-1)
        rankings[label] = np.argsort(-similarities, kind="stable").tolist()
    return rankings
```

Implement `_search_local` with these invariants:

1. `score_cache[label]` is keyed by original template index.
2. `_score_to(limit)` scores only ranked indices not already present.
3. `matching_ms` accumulates only `_score_feature_pair` calls, not ALIKED extraction.
4. Each stage’s score is the maximum of all candidates scored so far for that direction.
5. Intermediate stop decisions call the unchanged `_fuse_scores(global_scores, scores, time.perf_counter())` and inspect only `needs_review`.
6. `trace` records full ranked indices plus each evaluated stage’s scores and fusion evidence, but only `diagnostics` is later copied into the service response.
7. `exhaustive` mode and adaptive queries whose global margin exceeds `GLOBAL_MARGIN_THRESHOLD` call `_score_to(available_count)` directly.
8. If a Top-5 or Top-10 limit already covers every available template, return that stage with `exhaustive=true` instead of invoking a duplicate stage.

Use this exact public diagnostic shape:

```python
diagnostics = {
    "stage": stage,
    "mode": self.local_search_mode,
    "matched_counts": {
        label: len(score_cache[label]) for label in ("front", "back")
    },
    "available_counts": available_counts,
    "expanded_because": expanded_because,
    "exhaustive": all(
        len(score_cache[label]) == available_counts[label]
        for label in ("front", "back")
    ),
    "global_margin_gate": global_margin_gate,
}
```

For direct full search use `expanded_because="exhaustive_mode"` or `"preserve_review_semantics"`; for a Top-5 stop use `None`; for later stages use the reason from the stage that triggered expansion.

Set `global_margin_gate="adaptive_low_margin"` only for adaptive queries whose global margin is at most `0.05`; set it to `"preserve_review_semantics"` for high-margin adaptive queries and all exhaustive queries. No third gate value is permitted.

- [ ] **Step 5: Add failing expansion, conflict, count and corruption tests**

Add parametrized tests that assert calls, stage, trace and errors:

```python
def test_adaptive_local_search_expands_to_top10_without_rematching(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13,
        lambda label, index: 8.0 if label == "front" and index == 5 else 5.0,
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.diagnostics["stage"] == "top10"
    assert result.diagnostics["matched_counts"] == {"front": 10, "back": 10}
    assert len(calls) == 20
    assert len(set(calls)) == 20
    assert [item["stage"] for item in result.trace["stages"]] == ["top5", "top10"]


def test_adaptive_local_search_reaches_full_once_when_still_uncertain(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13, lambda label, index: 5.0
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.diagnostics["stage"] == "full"
    assert result.diagnostics["matched_counts"] == {"front": 12, "back": 13}
    assert result.diagnostics["exhaustive"] is True
    assert len(calls) == 25
    assert len(set(calls)) == 25


@pytest.mark.parametrize(
    ("front_count", "back_count", "expected_stage", "expected_calls"),
    [(1, 1, "top5", 2), (5, 10, "top5", 10), (10, 15, "top5", 10)],
)
def test_adaptive_local_search_supports_unequal_counts(
    classifier, front_count, back_count, expected_stage, expected_calls
):
    calls = []
    global_vectors, local_features = make_search_inputs(
        front_count, back_count,
        lambda label, index: 8.0 if label == "front" else 5.0,
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )
    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )
    assert result.diagnostics["stage"] == expected_stage
    assert len(calls) == expected_calls


def test_local_search_rejects_misaligned_cache(classifier):
    global_vectors, local_features = make_search_inputs(
        5, 5, lambda label, index: 5.0
    )
    local_features["back"].pop()
    with pytest.raises(OrientationClassifierError, match="alignment error for back"):
        run_local_search(
            classifier, global_vectors, local_features,
            {"front": 0.51, "back": 0.50},
        )
```

Add two more focused cases in the same test group:

- global margin `0.90 - 0.20` forces `stage="full"`, `global_margin_gate="preserve_review_semantics"`, and every candidate is called once;
- a classifier constructed with `local_search_mode="exhaustive"` forces `stage="full"`, returns the same all-template maximum scores and `_fuse_scores` fields as the pre-change path, and reports the exact total template count.
- constructing a classifier with `local_search_mode="fast"` raises `ValueError` containing `local_search_mode`.

- [ ] **Step 6: Run the complete core test group**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py -k "local_search" -q -p no:cacheprovider --basetemp=pytest-adaptive-search-core-green
```

Expected: all candidate-search tests pass; no template index appears twice in any call log.

- [ ] **Step 7: Review and commit only the core hunks**

Run:

```powershell
git diff -- src/orientation_classifier.py tests/test_orientation_classifier.py
git add -p -- src/orientation_classifier.py tests/test_orientation_classifier.py
git diff --cached --check
git diff --cached
git commit -m "feat: add adaptive LightGlue candidate search core"
```

Expected: the staged diff contains only constants, mode validation, local-search helpers and their tests.

---

### Task 2: Wire candidate search into baseline and geometry inference

**Files:**
- Modify: `src/orientation_classifier.py:880-1147`
- Modify: `tests/test_orientation_classifier.py:237-325,471-550,619-790`

**Interfaces:**
- Consumes: `_search_local -> LocalSearchResult` from Task 1.
- Produces: every prediction response contains `local_search` with the approved public diagnostic fields.
- Produces: geometry `timings_ms` contains both `local_features` and `local_matching`; the former excludes LightGlue.
- Preserves: raw baseline fallback, geometry-active path and ordinary baseline all call the same candidate-search implementation.

- [ ] **Step 1: Add failing baseline integration tests**

Add a small cache with 12+13 templates whose top five local scores agree with the low-margin global prediction:

```python
def test_baseline_prediction_exposes_adaptive_search_diagnostics(classifier, tmp_path):
    calls = []
    cache = TemplateCache(
        global_vectors={
            "front": np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (12, 1)),
            "back": np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (13, 1)),
        },
        local_features={
            "front": [{"label": "front", "index": index, "score": 5.0} for index in range(12)],
            "back": [{"label": "back", "index": index, "score": 8.0} for index in range(13)],
        },
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = classifier.predict_with_cache(
        cache, write_marker(tmp_path / "adaptive-query.png", 3)
    )

    assert result["label"] == "back"
    assert result["needs_review"] is False
    assert result["local_search"]["stage"] == "top5"
    assert result["local_search"]["matched_counts"] == {"front": 5, "back": 5}
    assert len(calls) == 10
```

Add a companion test constructed with `local_search_mode="exhaustive"` and assert all 25 candidates are called, `stage="full"`, and the final `label`, `needs_review`, `global_scores`, `local_scores`, `global_margin`, `local_margin`, and `decision_source` equal an explicitly computed `_fuse_scores` result using the all-template maxima.

- [ ] **Step 2: Add failing geometry timing and fallback assertions**

Update `test_geometry_prediction_batches_front_and_back_global_embeddings` so the exact timing key set becomes:

```python
assert set(result["geometry_mask"]["timings_ms"]) == {
    "fit_context",
    "fit_directions",
    "mask_build",
    "global_batch",
    "local_features",
    "local_matching",
    "fusion",
}
assert result["local_search"]["mode"] == "adaptive"
```

In the existing one-direction geometry-fit failure test, assert that the raw baseline fallback also includes `result["local_search"]` and that its `available_counts` match `raw_local_features`, proving no second search implementation exists.

- [ ] **Step 3: Run integration tests and confirm diagnostics are absent**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py -k "adaptive_search_diagnostics or geometry_prediction_batches or one_direction_fit_failure" -q -p no:cacheprovider --basetemp=pytest-adaptive-search-wiring-red
```

Expected: FAIL on missing `local_search` and `local_matching`.

- [ ] **Step 4: Integrate the search into `_predict_baseline`**

Refactor `_predict_baseline` in this order:

```python
query_embedding = self._global_embedding(image)
_, global_scores, _ = classify_embedding(query_embedding, raw_globals)

local_started = time.perf_counter()
with self._inference_lock:
    extracted = self._extract_features(
        image, self.extractor, self.device, roi_ratio=ROI_RATIO
    )
if timings is not None:
    timings["local_features"] += (time.perf_counter() - local_started) * 1000.0

search = self._search_local(
    query_features_by_label={"front": extracted, "back": extracted},
    global_vectors=raw_globals,
    query_embeddings={"front": query_embedding, "back": query_embedding},
    local_features=raw_locals,
    image_shape=image.shape[:2],
    global_scores=global_scores,
)
if timings is not None:
    timings["local_matching"] += search.matching_ms

result = self._fuse_scores(
    global_scores, search.scores, started, geometry_mask=geometry_mask
)
result["local_search"] = search.diagnostics
```

Keep the existing global and fusion timers. Remove the old `_score_local` all-template calls after both inference paths use `_search_local`.

- [ ] **Step 5: Integrate the same search into `_predict_geometry`**

After the existing batched PP-ShiTu call and one-time ALIKED extraction/filtering, call `_search_local` with per-direction query embeddings and per-direction filtered local features:

```python
search = self._search_local(
    query_features_by_label=query_features,
    global_vectors=cache.global_vectors,
    query_embeddings=query_embeddings,
    local_features=cache.local_features,
    image_shape=image.shape[:2],
    global_scores=global_scores,
)
timings["local_matching"] = search.matching_ms
result = self._fuse_scores(
    global_scores, search.scores, started, geometry_mask=geometry_mask
)
result["local_search"] = search.diagnostics
```

Add `"local_matching": 0.0` to `_geometry_timings()`. Do not include candidate ranking or intermediate `trace` in the TCP result.

- [ ] **Step 6: Run classifier regression tests**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py -q -p no:cacheprovider --basetemp=pytest-adaptive-search-classifier-green
```

Expected: all tests pass. Update the old `test_prediction_scores_every_local_template` to construct an `exhaustive` classifier, so it remains an explicit backward-compatibility assertion rather than contradicting adaptive mode.

- [ ] **Step 7: Review and commit only inference-wiring hunks**

Run:

```powershell
git diff -- src/orientation_classifier.py tests/test_orientation_classifier.py
git add -p -- src/orientation_classifier.py tests/test_orientation_classifier.py
git diff --cached --check
git diff --cached
git commit -m "perf: use adaptive candidates in orientation inference"
```

Expected: no model constants, weights, cache serialization or geometry fitting code changed.

---

### Task 3: Thread the process-level mode through the Python service

**Files:**
- Modify: `src/orientation_tcp_service.py:908-960`
- Modify: `tests/test_orientation_tcp_service.py:850-930`

**Interfaces:**
- Consumes: `LOCAL_SEARCH_MODES`, `DEFAULT_LOCAL_SEARCH_MODE`, and `OrientationClassifier.load(project_root, model_dir, local_search_mode=mode)` from Task 1.
- Produces: `_build_argument_parser() -> argparse.ArgumentParser`.
- Produces: `_load_runtime(runtime, project_root, model_dir, library_dir, local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE) -> None`.
- Produces: CLI option `--local-search-mode {adaptive,exhaustive}`.

- [ ] **Step 1: Add failing parser and loader tests**

Add to `tests/test_orientation_tcp_service.py`:

```python
def test_local_search_mode_argument_defaults_to_adaptive(tmp_path):
    parser = service_module._build_argument_parser()
    args = parser.parse_args([
        "--project-root", str(tmp_path),
        "--model-dir", str(tmp_path / "models"),
        "--library-dir", str(tmp_path / "library"),
    ])
    assert args.local_search_mode == "adaptive"


def test_local_search_mode_argument_rejects_unknown_value(tmp_path):
    parser = service_module._build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--project-root", str(tmp_path),
            "--model-dir", str(tmp_path / "models"),
            "--library-dir", str(tmp_path / "library"),
            "--local-search-mode", "fast",
        ])
```

Extend the existing runtime-loader test so `LoadedClassifier.load` receives a keyword-only mode and records it:

```python
@classmethod
def load(cls, project_root, model_dir, *, local_search_mode):
    events.append(f"model-loaded:{local_search_mode}")
    return cls()
```

Invoke `_load_runtime(runtime, tmp_path, tmp_path / "models", tmp_path / "library", local_search_mode="exhaustive")` and expect `model-loaded:exhaustive` as the first event.

- [ ] **Step 2: Run the service tests and confirm parser/keyword failures**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py -k "local_search_mode or runtime_loader" -q -p no:cacheprovider --basetemp=pytest-adaptive-search-service-red
```

Expected: FAIL because `_build_argument_parser` is absent and `_load_runtime` does not accept the mode.

- [ ] **Step 3: Implement parser construction and loader propagation**

Import the shared constants from `src.orientation_classifier` and extract parser construction from `main`:

```python
def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Workpiece orientation loopback service"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37651)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument(
        "--local-search-mode",
        choices=LOCAL_SEARCH_MODES,
        default=DEFAULT_LOCAL_SEARCH_MODE,
    )
    return parser
```

Change runtime loading to:

```python
def _load_runtime(
    runtime: ServiceRuntime,
    project_root: Path,
    model_dir: Path,
    library_dir: Path,
    local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
) -> None:
    classifier = OrientationClassifier.load(
        project_root,
        model_dir,
        local_search_mode=local_search_mode,
    )
```

Preserve the existing try/except, catalog recovery and worker startup order. In `main`, parse with `_build_argument_parser()` and append `args.local_search_mode` to the loader thread arguments.

- [ ] **Step 4: Run service regression tests**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py -q -p no:cacheprovider --basetemp=pytest-adaptive-search-service-green
```

Expected: all tests pass, including early-listen loading behavior.

- [ ] **Step 5: Review and commit only service-mode hunks**

Run:

```powershell
git diff -- src/orientation_tcp_service.py tests/test_orientation_tcp_service.py
git add -p -- src/orientation_tcp_service.py tests/test_orientation_tcp_service.py
git diff --cached --check
git diff --cached
git commit -m "feat: expose local search mode in backend service"
```

---

### Task 4: Add the Qt configuration safety switch

**Files:**
- Modify: `qt_app/appconfig.h:8-19`
- Modify: `qt_app/appconfig.cpp:1-95`
- Modify: `qt_app/app_config.json.example`
- Modify: `qt_app/backendprocessmanager.cpp:236-246`
- Modify: `qt_app/tests/test_appconfig.cpp:25-90`
- Modify: `qt_app/tests/test_backendprocessmanager.cpp:125-190`

**Interfaces:**
- Produces: `AppConfig::localSearchMode`, defaulting to `QStringLiteral("adaptive")` subject to Task 6’s acceptance gate.
- Produces: optional JSON field `local_search_mode` accepting only `adaptive` or `exhaustive`.
- Produces: backend argument pair `--local-search-mode`, `config_.localSearchMode`.

- [ ] **Step 1: Add failing AppConfig tests**

Extend `acceptsChinesePathsAndDefaults`:

```cpp
QCOMPARE(config->localSearchMode, QStringLiteral("adaptive"));
```

Add data-driven legal-mode and illegal-mode tests:

```cpp
void acceptsLocalSearchModes_data() {
    QTest::addColumn<QString>("mode");
    QTest::newRow("adaptive") << QStringLiteral("adaptive");
    QTest::newRow("exhaustive") << QStringLiteral("exhaustive");
}

void acceptsLocalSearchModes() {
    QFETCH(QString, mode);
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    QJsonObject object = validConfig(temporary);
    object[QStringLiteral("local_search_mode")] = mode;
    QString error;
    const auto config = AppConfig::load(writeConfig(temporary, object), &error);
    QVERIFY2(config.has_value(), qPrintable(error));
    QCOMPARE(config->localSearchMode, mode);
}

void rejectsUnknownLocalSearchMode() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    QJsonObject object = validConfig(temporary);
    object[QStringLiteral("local_search_mode")] = QStringLiteral("fast");
    QString error;
    const auto config = AppConfig::load(writeConfig(temporary, object), &error);
    QVERIFY(!config.has_value());
    QVERIFY(error.contains(QStringLiteral("local_search_mode")));
}
```

- [ ] **Step 2: Add a failing backend argument assertion**

In `launchesPythonOnlyAfterInitialConnectionFailure`, create the manager from a mutable config with `localSearchMode = "exhaustive"`, then assert:

```cpp
const int modeIndex = launcher.lastArguments.indexOf(
    QStringLiteral("--local-search-mode")
);
QVERIFY(modeIndex >= 0);
QCOMPARE(
    launcher.lastArguments.value(modeIndex + 1),
    QStringLiteral("exhaustive")
);
```

In the existing owned-process restart test, set `localSearchMode` to `exhaustive`, complete the graceful restart, and repeat the same assertion against the second launch’s `lastArguments`. This proves restart does not silently revert the process-level mode.

- [ ] **Step 3: Build the two Qt tests and confirm failures**

From a VS 2019 x64 developer shell, run:

```powershell
cd E:\Project\wang\pp_813\qt_app\tests
E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe test_appconfig.pro -o Makefile.adaptive-appconfig CONFIG+=release
E:\QT\5.14\Tools\QtCreator\bin\jom.exe -f Makefile.adaptive-appconfig
.\release\test_appconfig.exe -platform offscreen -o -,txt
E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe test_backendprocessmanager.pro -o Makefile.adaptive-backend CONFIG+=release
E:\QT\5.14\Tools\QtCreator\bin\jom.exe -f Makefile.adaptive-backend
.\release\test_backendprocessmanager.exe -platform offscreen -o -,txt
```

Expected: compilation or assertions fail because `localSearchMode` and the CLI pair are absent.

- [ ] **Step 4: Implement optional JSON parsing and launch propagation**

Add to `AppConfig`:

```cpp
QString localSearchMode = QStringLiteral("adaptive");
```

After required string fields are read in `AppConfig::load`, parse the optional key:

```cpp
const QJsonValue searchModeValue = object.value(
    QStringLiteral("local_search_mode")
);
if (!searchModeValue.isUndefined()) {
    if (!searchModeValue.isString()) {
        setError(error, QStringLiteral(
            "local_search_mode must be adaptive or exhaustive"
        ));
        return std::nullopt;
    }
    const QString mode = searchModeValue.toString().trimmed().toLower();
    if (mode != QStringLiteral("adaptive")
        && mode != QStringLiteral("exhaustive")) {
        setError(error, QStringLiteral(
            "local_search_mode must be adaptive or exhaustive"
        ));
        return std::nullopt;
    }
    config.localSearchMode = mode;
}
```

Append this pair in `BackendProcessManager::backendArguments()`:

```cpp
QStringLiteral("--local-search-mode"), config_.localSearchMode,
```

Add this line to `qt_app/app_config.json.example`:

```json
"local_search_mode": "adaptive",
```

Keep the runtime `qt_app/app_config.json` untouched so the user’s local paths are not overwritten; absent keys use the compatible default.

- [ ] **Step 5: Rebuild Qt tests and the release application**

Repeat the two test commands from Step 3, then run from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1
```

Expected: both QtTest executables report zero failures and the Qt 5.14.2 release application builds successfully.

- [ ] **Step 6: Review and commit only Qt safety-switch hunks**

Run:

```powershell
git diff -- qt_app/appconfig.h qt_app/appconfig.cpp qt_app/app_config.json.example qt_app/backendprocessmanager.cpp qt_app/tests/test_appconfig.cpp qt_app/tests/test_backendprocessmanager.cpp
git add -p -- qt_app/appconfig.h qt_app/appconfig.cpp qt_app/app_config.json.example qt_app/backendprocessmanager.cpp qt_app/tests/test_appconfig.cpp qt_app/tests/test_backendprocessmanager.cpp
git diff --cached --check
git diff --cached
git commit -m "feat: configure adaptive or exhaustive local search"
```

---

### Task 5: Build a reproducible two-process acceptance benchmark

**Files:**
- Create: `scripts/benchmark_adaptive_local_search.py`
- Create: `tests/test_benchmark_adaptive_local_search.py`

**Interfaces:**
- Produces: `_compare_worker_payloads(exhaustive: dict, adaptive: dict) -> dict`.
- Produces: `_render_markdown(report: dict) -> str`.
- Produces: parent CLI that launches one `exhaustive` worker and one `adaptive` worker in separate Python processes and writes both JSON and Markdown.
- Consumes: `LocalSearchResult.trace` through a temporary wrapper around `_search_local`; trace is stored only in benchmark rows and never exposed by the TCP API.

- [ ] **Step 1: Write failing report-comparison tests**

Create `tests/test_benchmark_adaptive_local_search.py` with synthetic worker payloads:

```python
from scripts.benchmark_adaptive_local_search import (
    _compare_worker_payloads,
    _render_markdown,
)


def row(path, label, needs_review, stage, calls):
    return {
        "case": "M1",
        "image_path": path,
        "actual": "front",
        "label": label,
        "needs_review": needs_review,
        "elapsed_ms": 400.0,
        "local_search": {
            "stage": stage,
            "matched_counts": {"front": calls // 2, "back": calls // 2},
        },
        "evidence": {
            "global_scores": {"front": 0.51, "back": 0.50},
            "local_scores": {"front": 8.0, "back": 5.0},
            "trace": {"ranked_indices": {"front": [0], "back": [0]}},
        },
    }


def test_compare_requires_per_image_label_and_review_identity():
    exhaustive = {"mode": "exhaustive", "rows": [row("a.png", "front", True, "full", 56)]}
    adaptive = {"mode": "adaptive", "rows": [row("a.png", "front", False, "top5", 10)]}
    report = _compare_worker_payloads(exhaustive, adaptive)
    assert report["gate_passed"] is False
    assert report["label_mismatches"] == []
    assert [item["image_path"] for item in report["review_mismatches"]] == ["a.png"]
    assert report["review_mismatches"][0]["adaptive"]["evidence"]["trace"]


def test_markdown_reports_stage_counts_latency_and_gate():
    rows = [row("a.png", "front", False, "top5", 10)]
    report = _compare_worker_payloads(
        {"mode": "exhaustive", "rows": rows},
        {"mode": "adaptive", "rows": rows},
    )
    markdown = _render_markdown(report)
    assert "逐图标签一致：通过" in markdown
    assert "逐图复检状态一致：通过" in markdown
    assert "P50" in markdown
    assert "top5" in markdown
```

- [ ] **Step 2: Run tests and confirm the benchmark module is missing**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_benchmark_adaptive_local_search.py -q -p no:cacheprovider --basetemp=pytest-adaptive-benchmark-red
```

Expected: FAIL because the benchmark module does not exist.

- [ ] **Step 3: Implement deterministic case selection**

In `scripts/benchmark_adaptive_local_search.py`:

- map dataset label `0` to `front` and `1` to `back`;
- for M1, recover workpiece `31f082d1a04e486b9345846f4d585033`, assert actual cache counts are 28+28, hash every current template image with SHA-256, exclude matching dataset images, and take the first 20 sorted remaining images per class;
- for M2 and M7, call `split_labels(dataset_dir, template_count=20, seed=20260825)` and take the first 20 held-out images per class;
- build M2/M7 caches in memory only; do not register or modify `runtime_library`;
- verify every case has exactly 40 distinct held-out query hashes and no held-out hash occurs in its template set.

Use explicit row keys:

```python
row = {
    "case": case_name,
    "image_path": str(query_path),
    "actual": actual,
    "label": result["label"],
    "needs_review": bool(result["needs_review"]),
    "elapsed_ms": float(result["elapsed_ms"]),
    "local_search": result["local_search"],
    "evidence": {
        "global_scores": result["global_scores"],
        "global_margin": result["global_margin"],
        "local_scores": result["local_scores"],
        "local_margin": result["local_margin"],
        "decision_source": result["decision_source"],
        "trace": captured_trace,
    },
}
```

Capture `captured_trace` by temporarily wrapping the classifier instance’s `_search_local`: call the original bound method, append `LocalSearchResult.trace`, return the result unchanged, and restore the original method in `finally`. Assert exactly one trace is captured for each prediction.

- [ ] **Step 4: Implement isolated workers and comparison gate**

The parent invocation accepts:

```text
--project-root
--model-dir
--library-dir
--m1-workpiece-id
--warmup
--output-json
--output-markdown
```

Default `--m1-workpiece-id` to `31f082d1a04e486b9345846f4d585033` and `--warmup` to `5`. The parent launches the same script twice with an internal `--worker-mode` argument, first `exhaustive`, then `adaptive`. Build and execute the child command exactly as follows, so each process receives the same paths and writes a separate worker result:

```python
worker_command = [
    sys.executable,
    str(Path(__file__).resolve()),
    "--worker-mode", worker_mode,
    "--project-root", str(args.project_root),
    "--model-dir", str(args.model_dir),
    "--library-dir", str(args.library_dir),
    "--m1-workpiece-id", args.m1_workpiece_id,
    "--warmup", str(args.warmup),
    "--worker-output", str(worker_output),
]
subprocess.run(worker_command, check=True)
```

Each child loads its classifier with `OrientationClassifier.load(args.project_root, args.model_dir, local_search_mode=worker_mode)`, so mode never changes during that process.

`_compare_worker_payloads` must:

1. key rows by `(case, image_path, actual)` and reject missing/duplicate keys;
2. list every label mismatch and every `needs_review` mismatch with complete evidence from both modes;
3. compute per-case/per-mode P50, P95 and mean elapsed time with NumPy;
4. compute `top5`/`top10`/`full` counts and average `front + back` matched count;
5. set `gate_passed = not label_mismatches and not review_mismatches`;
6. retain correctness versus `actual` as descriptive data, never as a replacement for pairwise equivalence.

The parent report must also record `platform.platform()`, `sys.version`, GPU name when Paddle exposes it, and the installed Paddle, PyTorch and NumPy versions. `_render_markdown` renders these values before the per-dataset table so Task 6's report is self-contained.

Always write JSON and Markdown before exiting. Exit code is `0` when `gate_passed=true` and `2` when false.

- [ ] **Step 5: Run benchmark utility unit tests**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_benchmark_adaptive_local_search.py -q -p no:cacheprovider --basetemp=pytest-adaptive-benchmark-green
```

Expected: all pure comparison and Markdown tests pass without importing Paddle or Torch models.

- [ ] **Step 6: Review and commit benchmark tooling**

Run:

```powershell
git add -N -- scripts/benchmark_adaptive_local_search.py tests/test_benchmark_adaptive_local_search.py
git diff -- scripts/benchmark_adaptive_local_search.py tests/test_benchmark_adaptive_local_search.py
git add -p -- scripts/benchmark_adaptive_local_search.py tests/test_benchmark_adaptive_local_search.py
git diff --cached --check
git diff --cached
git commit -m "test: add adaptive LightGlue acceptance benchmark"
```

---

### Task 6: Run the release gate, choose the safe default, and complete regression

**Files:**
- Create: `docs/verification/adaptive-lightglue-candidate-search-results.md` through the benchmark command
- Conditionally modify on gate failure only: `src/orientation_classifier.py`, `qt_app/appconfig.h`, `qt_app/app_config.json.example`, related default-value tests

**Interfaces:**
- Consumes: the two-process benchmark from Task 5.
- Produces: a committed, machine-generated verification report with exact M1/M2/M7 results.
- Enforces: adaptive remains the default only when every held-out image matches exhaustive on both final label and `needs_review`.

- [ ] **Step 1: Run focused Python regression before the model benchmark**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py tests\test_orientation_tcp_service.py tests\test_benchmark_adaptive_local_search.py -q -p no:cacheprovider --basetemp=pytest-adaptive-focused
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the real M1/M2/M7 two-process gate**

Run from `E:\Project\wang\pp_813`:

```powershell
E:\python\anaconda3\envs\shitu\python.exe scripts\benchmark_adaptive_local_search.py --project-root E:\Project\wang\pp_813 --model-dir E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer --library-dir E:\Project\wang\pp_813\runtime_library --m1-workpiece-id 31f082d1a04e486b9345846f4d585033 --warmup 5 --output-json E:\Project\wang\pp_813\runtime_reports\adaptive-lightglue-candidate-search.json --output-markdown E:\Project\wang\pp_813\docs\verification\adaptive-lightglue-candidate-search-results.md
```

Expected on the validated design dataset: exit code `0`, M1/M2/M7 each contain 40 rows, and both mismatch lists are empty. The expected M1 order of magnitude is 370–400 ms average with an observed offline stage distribution near 29 Top-5, 6 Top-10 and 5 full; these figures are reported, not asserted.

- [ ] **Step 3: Apply the deterministic default-mode gate**

If Step 2 exits `0`, keep these exact defaults:

```python
DEFAULT_LOCAL_SEARCH_MODE = "adaptive"
```

```cpp
QString localSearchMode = QStringLiteral("adaptive");
```

```json
"local_search_mode": "adaptive"
```

If Step 2 exits `2`, preserve the adaptive implementation but change all three defaults and their default-value assertions to `exhaustive`:

```python
DEFAULT_LOCAL_SEARCH_MODE = "exhaustive"
```

```cpp
QString localSearchMode = QStringLiteral("exhaustive");
```

```json
"local_search_mode": "exhaustive"
```

In either branch, the alternate explicit mode remains accepted. Do not weaken the gate to aggregate accuracy.

- [ ] **Step 4: Run the complete project Python suite**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=pytest-adaptive-full
```

Expected: all project tests pass; integration tests that require an explicitly enabled real-service fixture may remain skipped according to their existing markers.

- [ ] **Step 5: Rebuild and run Qt safety-switch tests after the gate decision**

Repeat Task 4’s two QtTest builds and executions, then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1
```

Expected: AppConfig and BackendProcessManager tests pass with the selected default, and the Qt 5.14.2 release build succeeds.

- [ ] **Step 6: Inspect the generated verification report**

Confirm `docs/verification/adaptive-lightglue-candidate-search-results.md` contains:

- hardware and Paddle/PyTorch/NumPy versions;
- M1 28+28 and M2/M7 20+20 template counts;
- 40 held-out images per dataset with template-hash exclusion;
- per-mode P50/P95/mean, stage counts and average matched count;
- exact label/review mismatch lists with traces when non-empty;
- the selected default and an explicit `通过` or `阻断` release conclusion.

- [ ] **Step 7: Review and commit only verification/default hunks**

Run:

```powershell
git add -N -- docs/verification/adaptive-lightglue-candidate-search-results.md
git diff -- docs/verification/adaptive-lightglue-candidate-search-results.md src/orientation_classifier.py qt_app/appconfig.h qt_app/app_config.json.example tests/test_orientation_classifier.py qt_app/tests/test_appconfig.cpp
git add -p -- docs/verification/adaptive-lightglue-candidate-search-results.md src/orientation_classifier.py qt_app/appconfig.h qt_app/app_config.json.example tests/test_orientation_classifier.py qt_app/tests/test_appconfig.cpp
git diff --cached --check
git diff --cached
git commit -m "docs: record adaptive LightGlue release gate"
```

Expected: generated real values are present; no unrelated dirty-worktree changes are staged.

---

## Final Acceptance Checklist

- [ ] `adaptive` low-margin queries stop only when the unchanged fusion result says `needs_review=false`.
- [ ] high-global-margin queries and `exhaustive` mode match every template exactly once.
- [ ] Top-10/full expansion never repeats Top-5 work.
- [ ] 1+1、5+10、10+15、28+28 and unequal counts work without fixed-count assumptions.
- [ ] damaged cache alignment fails explicitly rather than truncating.
- [ ] baseline, geometry-active and raw fallback paths all return the same `local_search` schema.
- [ ] `local_features` excludes LightGlue and `local_matching` reports it separately.
- [ ] Qt config accepts only the two modes and forwards the selected mode to the backend.
- [ ] existing cache format, model settings and four fusion thresholds are unchanged.
- [ ] M1/M2/M7 per-image label and review gates determine the delivered default.
- [ ] complete Python tests, focused Qt tests and Qt 5.14.2 release build pass.
