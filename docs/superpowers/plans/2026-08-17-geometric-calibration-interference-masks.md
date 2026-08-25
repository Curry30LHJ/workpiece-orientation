# Geometric Calibration Interference Masks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用对象坐标下的圆、椭圆和旋转矩形边界，在每张模板与待测图上重新拟合中心反光和外部侵入屏蔽区，并以可验证、可发布、可回退的方式同时接入 PP-ShiTuV2 与 ALIKED/LightGlue。

**Architecture:** 新增纯计算深模块 `GeometryCalibrator`，隐藏 OpenCV 边界候选、对象坐标转换、置信度和遮罩合成；新增状态深模块 `GeometryMaskProfiles`，隐藏草稿、后台验证、不可变版本、审计和原子发布。`OrientationClassifier` 只消费已发布 profile，`WorkpieceCatalog` 保持持久化指针与运行时缓存的一致切换，Qt 通过短 TCP 请求编辑、轮询验证并显式发布。

**Tech Stack:** Python 3.10、OpenCV、NumPy、PaddlePaddle/PP-ShiTuV2、PyTorch/ALIKED/LightGlue、JSON-lines TCP V1、Qt 5.14.2 Widgets/Network/TestLib、pytest。

## Global Constraints

- 不重新训练或微调 PP-ShiTuV2、ALIKED、LightGlue。
- 不修改 `GLOBAL_MARGIN_THRESHOLD=0.05`、`LOCAL_MIN_SCORE=4.0`、`LOCAL_MIN_MARGIN=0.5`、`LOCAL_OVERRIDE_MARGIN=3.0`。
- V1 只支持圆、椭圆、可旋转矩形；不支持自由多边形和抽象干扰语义学习。
- 正反面 profile 独立；任一启用规则低置信度时整次预测退回无遮罩基线并设置复核状态。
- 内部忽略默认外扩 2%，外部忽略默认内缩 2%；有效区域低于 60% 或关键点保留率低于 30% 只警告，不静默阻止显式发布。
- 所有草稿、验证、发布和回退操作绑定工件修订、草稿修订及 `operation_id`；发布过程不得暴露部分状态。
- 旧 5+5 及任意模板数量工件库继续恢复；旧 `interference_groups` 保留归档但不再参与预测。
- 现有工作树包含用户改动；每个提交只暂存本任务列出的文件。

---

### Task 1: 对象坐标几何标定模块

**Files:**
- Create: `src/geometry_calibration.py`
- Create: `tests/test_geometry_calibration.py`

**Interfaces:**
- Consumes: BGR `numpy.ndarray`；一个方向的 JSON profile，包含 `anchor` 与 `rules`。
- Produces: `GeometryCalibrator.fit(image, direction_profile) -> dict`、`apply_ignore_mask(image, mask, fill_bgr) -> np.ndarray`、`filter_features_by_mask(features, mask) -> dict`。

- [ ] **Step 1: 写合成图红灯测试，固定对象坐标和置信度契约**

```python
def test_inner_circle_tracks_shift_scale_and_rotation_in_anchor_coordinates():
    image = synthetic_ellipse(center=(155, 92), axes=(72, 54), angle=24,
                              inner_circle=(0.0, 0.0, 0.56))
    result = GeometryCalibrator().fit(image, circle_profile(mode="inside"))
    assert result["status"] == "active"
    assert abs(result["anchor"]["cx"] - 155) <= 3
    assert result["rules"][0]["edge_support"] >= 0.35
    assert result["ignore_mask"][92, 155] == 255
    assert result["ignore_mask"][20, 20] == 0

def test_outer_rule_ignores_intrusion_but_keeps_shrunken_object_edge():
    image = synthetic_circle_with_corner_intrusions()
    result = GeometryCalibrator().fit(image, outer_profile(mode="outside", margin_ratio=0.02))
    assert result["status"] == "active"
    assert result["ignore_mask"][5, 5] == 255
    assert result["ignore_mask"][128, 128] == 0

def test_anchor_failure_returns_no_partial_mask():
    result = GeometryCalibrator().fit(np.full((256, 256, 3), 127, np.uint8), circle_profile())
    assert result["status"] == "low_confidence"
    assert "ignore_mask" not in result
```

- [ ] **Step 2: 运行测试并确认因模块不存在失败**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_calibration.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.geometry_calibration'`.

- [ ] **Step 3: 实现最小、集中式几何接口和固定初始阈值**

JSON 形状格式固定如下，禁止调用方发明额外坐标语义：

```json
{
  "directions": {
    "front": {
      "anchor": {"shape": "ellipse", "coarse": {"cx": 0.5, "cy": 0.5, "rx": 0.4, "ry": 0.35, "angle_deg": 0.0}},
      "rules": [
        {"rule_id": "inner-glare", "name": "内腔反光", "shape": "circle",
         "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.56},
         "mode": "inside", "margin_ratio": 0.02, "enabled": true}
      ]
    },
    "back": {"anchor": null, "rules": []}
  }
}
```

anchor 的 `coarse` 使用整图 0–1 坐标，只作为大范围初值；子规则 `geometry` 使用对象坐标：基准中心为 `(0,0)`，沿基准主轴/次轴分别除以对应半轴，角度相对基准角。旋转矩形使用 `cx/cy/half_width/half_height/angle_deg`，椭圆使用 `cx/cy/rx/ry/angle_deg`。

```python
ANCHOR_SEARCH_MARGIN_RATIO = 0.35
RULE_SEARCH_BAND_RATIO = 0.08
MIN_EDGE_SUPPORT = 0.35
MIN_VISIBLE_RATIO = 0.45
MAX_FIT_RESIDUAL_RATIO = 0.03

class GeometryCalibrator:
    def fit(self, image: np.ndarray, direction_profile: Mapping[str, Any]) -> dict[str, Any]:
        anchor = self._fit_anchor(image, direction_profile["anchor"])
        if anchor["status"] != "active":
            return {"status": "low_confidence", "anchor": anchor, "rules": []}
        fitted_rules = [self._fit_rule(image, anchor, rule)
                        for rule in direction_profile.get("rules", []) if rule.get("enabled", True)]
        if any(item["status"] != "active" for item in fitted_rules):
            return {"status": "low_confidence", "anchor": anchor, "rules": fitted_rules}
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for item in fitted_rules:
            mask = cv2.bitwise_or(mask, item["ignore_mask"])
        return {"status": "active", "anchor": anchor, "rules": fitted_rules,
                "ignore_mask": mask, "ignored_ratio": float(np.count_nonzero(mask) / mask.size)}
```

实现细节限定在私有方法中：圆用局部轮廓点最小二乘/RANSAC 圆拟合，椭圆用 `cv2.fitEllipse`，矩形用轮廓与 `cv2.minAreaRect`；候选只来自基准或对象坐标转换后的搜索带。

- [ ] **Step 4: 写椭圆、旋转矩形、边距方向、并集与特征过滤红灯测试**

测试分别断言拟合形状类型、角度误差不超过 3°、2% 边距方向、并集像素以及遮罩内关键点被移除且对齐 descriptor 维度保持一致。

- [ ] **Step 5: 实现遮罩应用、对象坐标转换和特征过滤**

`apply_ignore_mask` 复制输入图并只在 `mask == 255` 处写入固定 BGR 填充值；`filter_features_by_mask` 根据关键点原图坐标生成保留索引，并同步过滤 `keypoints/scores/descriptors`。私有对象坐标转换以基准中心为原点、主轴为单位尺度、基准角为旋转零点。

- [ ] **Step 6: 运行几何模块完整测试**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_calibration.py -q`

Expected: PASS，覆盖 `inside/outside`、2% 边距、中心偏移、裁剪尺寸变化、低可见弧和不可读输入对应的数据校验错误。

- [ ] **Step 7: 提交纯计算模块**

```powershell
git add src/geometry_calibration.py tests/test_geometry_calibration.py
git commit -m "feat: add object-relative geometry calibration"
```

---

### Task 2: 分类器双方向遮罩与可回退缓存

**Files:**
- Modify: `src/orientation_classifier.py`
- Modify: `tests/test_orientation_classifier.py`

**Interfaces:**
- Consumes: Task 1 的 `GeometryCalibrator`、活动 profile、已有 `TemplateCache`。
- Produces: `prepare_geometry_cache(workpiece_id, record, profile, calibrator, progress_callback=None)`；保持 `predict(workpiece_id, image_path)` 调用形式不变。

- [ ] **Step 1: 写红灯测试，证明全局和局部均使用遮罩且低置信度退回基线**

```python
def test_active_profile_builds_directional_query_features_without_changing_thresholds(
        classifier, record, query_path, fake_calibrator, fake_predictor):
    candidate, report = classifier.prepare_geometry_cache("part", record, active_profile, fake_calibrator)
    classifier.set_template_cache("part", candidate)
    result = classifier.predict("part", query_path)
    assert fake_predictor.images_seen[-2:] == [front_masked_rgb, back_masked_rgb]
    assert result["geometry_mask"]["status"] == "active"
    assert result["geometry_mask"]["profile_revision"] == 3

def test_one_direction_fit_failure_uses_raw_global_and_local_features(
        classifier, record, query_path, fake_calibrator, fake_predictor):
    result = classifier.predict("part", query_path)
    assert result["geometry_mask"]["status"] == "low_confidence"
    assert result["geometry_mask"]["needs_review"] is True
    assert result["needs_review"] is True
    assert np.array_equal(fake_predictor.images_seen[-1], raw_query_rgb)
```

- [ ] **Step 2: 运行目标测试并确认缺少几何缓存接口**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py -k "geometry or mask" -q`

Expected: FAIL because `prepare_geometry_cache` and `geometry_mask` response are absent.

- [ ] **Step 3: 扩展缓存格式并保留无掩码基础特征**

```python
TEMPLATE_CACHE_FORMAT_VERSION = 2

@dataclass(frozen=True)
class TemplateCache:
    global_vectors: dict[str, np.ndarray]
    local_features: dict[str, list[dict[str, Any]]]
    raw_global_vectors: dict[str, np.ndarray] | None = None
    raw_local_features: dict[str, list[dict[str, Any]]] | None = None
    geometry_profile: dict[str, Any] | None = None
    geometry_profile_revision: int | None = None
    geometry_template_report: dict[str, Any] | None = None
    ignored_regions: dict[str, list[dict[str, float]]] | None = None
```

`build_template_cache` 同时设置 `raw_global_vectors=global_vectors`；格式版本升级使旧 pickle 安全失效并由现有恢复流程重建，不影响旧模板图片。

- [ ] **Step 4: 实现候选缓存和双方向查询路径**

```python
def prepare_geometry_cache(self, workpiece_id, record, profile, calibrator, progress_callback=None):
    base = self._template_caches[workpiece_id]
    raw_globals = base.raw_global_vectors or base.global_vectors
    raw_locals = base.raw_local_features or base.local_features
    # 每张模板重新拟合、填充、提取；仅返回候选，不在此方法发布。
    return candidate_cache, validation_rows

def _predict_with_geometry(self, image, cache):
    fitted = {label: self.geometry_calibrator.fit(image, cache.geometry_profile[label])
              for label in ("front", "back")}
    if any(item["status"] != "active" for item in fitted.values()):
        return self._predict_from_features(image, cache.raw_global_vectors, cache.raw_local_features), fitted
    # front/back 各生成一份 embedding 和 local features，只与同方向模板比较。
```

`OrientationClassifier.__init__` 增加可选 `geometry_calibrator` 注入点，生产 `load()` 注入 `GeometryCalibrator()`，单元测试注入确定性 fake；profile 本身随活动 `TemplateCache` 保存，不从全局单例读取。

所有 Paddle/Torch 推理入口使用分类器私有 `threading.RLock`，锁粒度为一次图像特征提取或一次 LightGlue 配对，不能包住整批验证。

- [ ] **Step 5: 运行分类器完整回归**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py tests/test_global_local_fusion.py tests/test_shitu_baseline.py -q`

Expected: PASS；无 profile 时返回结果与改动前字段和分数一致。

- [ ] **Step 6: 提交分类器接入**

```powershell
git add src/orientation_classifier.py tests/test_orientation_classifier.py
git commit -m "feat: apply geometry masks to global and local inference"
```

---

### Task 3: 几何档案持久化与陈旧修订保护

**Files:**
- Create: `src/geometry_mask_profiles.py`
- Create: `tests/test_geometry_mask_profiles.py`
- Modify: `src/workpiece_library.py`
- Modify: `tests/test_workpiece_library.py`

**Interfaces:**
- Consumes: `WorkpieceLibrary.get(workpiece_id)` 和工件目录。
- Produces: `GeometryMaskProfiles.snapshot(workpiece_id)`、`save_draft(workpiece_id, draft, expected_library_revision, expected_draft_revision, operation_id)`；为后续验证和发布提供不可变档案格式。

- [ ] **Step 1: 写旧库、草稿修订、幂等和文件布局红灯测试**

```python
def test_old_library_has_empty_geometry_profile_without_manifest_migration(library, record):
    profiles = GeometryMaskProfiles(catalog, calibrator, start_worker=False)
    snapshot = profiles.snapshot(record.id)
    assert snapshot["library_revision"] == record.revision
    assert snapshot["draft_revision"] == 0
    assert snapshot["active_revision"] is None

def test_save_draft_requires_both_revisions_and_is_idempotent(profiles, record, draft):
    first = profiles.save_draft(record.id, draft,
        expected_library_revision=1, expected_draft_revision=0, operation_id="op-1")
    second = profiles.save_draft(record.id, draft,
        expected_library_revision=1, expected_draft_revision=0, operation_id="op-1")
    assert first == second
    assert first["draft_revision"] == 1
```

- [ ] **Step 2: 运行测试确认模块不存在**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py tests/test_workpiece_library.py -k "geometry_profile" -q`

Expected: FAIL with missing `GeometryMaskProfiles`.

- [ ] **Step 3: 实现档案规范化、原子草稿写入和校验**

```python
class GeometryMaskProfiles:
    def snapshot(self, workpiece_id: str) -> dict[str, Any]:
        record = self.catalog.get(workpiece_id)
        document = self._load_document(record)
        return self._snapshot(record, document)

    def save_draft(self, workpiece_id: str, draft: dict[str, Any], *,
                   expected_library_revision: int, expected_draft_revision: int,
                   operation_id: str) -> dict[str, Any]:
        return self._idempotent(operation_id, lambda: self._save_draft_once(
            workpiece_id, draft, expected_library_revision, expected_draft_revision
        ))
```

目录固定为：

```text
<workpiece>/geometry_masks/profile.json
<workpiece>/geometry_masks/revisions/<revision>.json
<workpiece>/geometry_masks/previews/<job_id>/<template_id>.png
```

写入使用同目录临时文件、`flush`、`os.fsync`、`os.replace`；规则校验拒绝重复 `rule_id`、未知形状、非法对象坐标、空名称、非法 mode 和非有限边距。

- [ ] **Step 4: 在 `WorkpieceLibrary` 中增加最小 manifest 指针提交接口**

```python
def replace_geometry_profile_pointers(self, workpiece_id: str, *,
                                      expected_revision: int,
                                      active_revision: int | None,
                                      previous_active_revision: int | None) -> WorkpieceRecord:
    """Atomically update only geometry profile pointers and workpiece revision."""
```

该接口不读取验证报告、不构建缓存；它只拥有 manifest 和 `_records` 的一致更新。

- [ ] **Step 5: 运行持久化与旧库回归**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py tests/test_workpiece_library.py -q`

Expected: PASS，包括旧 schema、任意模板数量、损坏 profile 隔离和陈旧修订不改文件。

- [ ] **Step 6: 提交档案持久化**

```powershell
git add src/geometry_mask_profiles.py src/workpiece_library.py tests/test_geometry_mask_profiles.py tests/test_workpiece_library.py
git commit -m "feat: persist versioned geometry mask drafts"
```

---

### Task 4: 后台验证、留一回归与原子发布

**Files:**
- Modify: `src/geometry_mask_profiles.py`
- Modify: `src/workpiece_catalog.py`
- Modify: `tests/test_geometry_mask_profiles.py`
- Modify: `tests/test_workpiece_catalog.py`

**Interfaces:**
- Consumes: Task 2 的候选缓存，Task 3 的草稿档案和 manifest 指针接口。
- Produces: `start_validation/get_job/action/publish/rollback/shutdown`；catalog 的单次原子缓存发布。

- [ ] **Step 1: 写后台任务与在线预测并存红灯测试**

```python
def test_validation_returns_job_before_worker_finishes_and_exposes_progress(
        profiles, record, catalog, query_path):
    job = profiles.start_validation(record.id, expected_library_revision=1,
                                    expected_draft_revision=1, operation_id="validate-1")
    assert job["state"] in {"queued", "running"}
    assert job["job_id"]
    assert catalog.predict(record.id, query_path)["label"] in {"front", "back"}

def test_restart_marks_running_job_interrupted_but_keeps_draft(
        catalog, calibrator, running_job_storage, job_id, record):
    restarted = GeometryMaskProfiles(catalog, calibrator, start_worker=False)
    assert restarted.get_job(job_id)["state"] == "interrupted"
    assert restarted.snapshot(record.id)["draft_revision"] == 1
```

- [ ] **Step 2: 写发布失败和回退原子性红灯测试**

```python
def test_publish_swaps_manifest_and_cache_only_after_candidate_is_complete(
        profiles, record, classifier, completed_job_id):
    old_cache = classifier.get_template_cache(record.id)
    classifier.fail_candidate_build = True
    with pytest.raises(GeometryProfilePublishError):
        profiles.publish(record.id, job_id, expected_library_revision=1,
                         expected_draft_revision=1, operation_id="publish-1")
    assert classifier.get_template_cache(record.id) is old_cache
    assert profiles.snapshot(record.id)["active_revision"] is None
```

- [ ] **Step 3: 运行验证任务测试并确认接口缺失**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py tests/test_workpiece_catalog.py -k "validation or publish or rollback" -q`

Expected: FAIL because job and atomic profile publication interfaces are absent.

- [ ] **Step 4: 实现任务状态机和验证报告**

```python
def start_validation(self, workpiece_id, *, expected_library_revision,
                     expected_draft_revision, operation_id) -> dict:
    return self._enqueue_validation(workpiece_id, expected_library_revision,
                                    expected_draft_revision, operation_id)
def get_job(self, job_id: str) -> dict:
    with self._condition:
        return deepcopy(self._jobs[job_id])
def action(self, job_id: str, action: str) -> dict:
    return self._apply_job_action(job_id, action)
def publish(self, workpiece_id, job_id, *, expected_library_revision,
            expected_draft_revision, operation_id, override_reason="") -> dict:
    return self._publish_validated_job(workpiece_id, job_id,
        expected_library_revision, expected_draft_revision, operation_id, override_reason)
def rollback(self, workpiece_id, *, expected_library_revision,
             operation_id) -> dict:
    return self._rollback_once(workpiece_id, expected_library_revision, operation_id)
def shutdown(self) -> None:
    with self._condition:
        self._stop = True
        self._condition.notify_all()
    if self._worker is not None:
        self._worker.join(timeout=2)
```

状态仅允许：`queued -> running -> completed|failed|cancelled`，重启恢复把 `running` 改成 `interrupted`。报告逐模板保存 `fit_status/confidence/edge_support/visible_ratio/fit_residual/ignored_ratio/remaining_ratio/baseline_label/candidate_label`。

- [ ] **Step 5: 实现留一验证而不重复提取模板特征**

候选构建时每张模板只提取一次遮罩后的全局与局部特征；留一阶段把当前模板当查询，从对应缓存视图排除同一 `template_id` 后评分。报告将 `correct -> wrong` 计为回归；存在回归或 60%/30% 警告时，`publish` 要求非空 `override_reason`。

- [ ] **Step 6: 在 catalog 写锁中实现 manifest 指针与缓存引用交换**

```python
def publish_geometry_profile(self, workpiece_id: str, candidate_cache: TemplateCache, *,
                             profile_revision: int, previous_profile_revision: int | None,
                             expected_revision: int, operation_id: str) -> WorkpieceRecord:
    # 候选文件已落盘；持锁更新 manifest，随后只做 dict 引用交换。
```

`WorkpieceCatalog.predict` 与发布继续使用同一 `RLock`；持久化成功后调用 `set_template_cache`，该调用不得执行 I/O 或模型推理。

- [ ] **Step 7: 运行状态、原子性和 catalog 回归**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py tests/test_workpiece_catalog.py -q`

Expected: PASS；取消只在模板边界生效，陈旧验证不可发布，一键回退恢复上一个 revision 的缓存和指针。

- [ ] **Step 8: 提交验证与发布模块**

```powershell
git add src/geometry_mask_profiles.py src/workpiece_catalog.py tests/test_geometry_mask_profiles.py tests/test_workpiece_catalog.py
git commit -m "feat: validate and publish geometry mask profiles"
```

---

### Task 5: 恢复、旧位置标注归档与确认入库守卫

**Files:**
- Modify: `src/workpiece_catalog.py`
- Modify: `src/template_evolution.py`
- Modify: `tests/test_workpiece_catalog.py`
- Modify: `tests/test_template_evolution.py`

**Interfaces:**
- Consumes: 活动 profile、`GeometryMaskProfiles.validate_new_template(workpiece_id, orientation, image_path)`。
- Produces: 重启恢复活动几何缓存；确认图片低置信度时保持 `needs_review`。

- [ ] **Step 1: 写恢复和旧标注不生效红灯测试**

```python
def test_recover_rebuilds_active_geometry_cache_and_archives_legacy_masks(
        catalog, classifier, record):
    recovered = catalog.recover()
    cache = classifier.get_template_cache(record.id)
    assert cache.geometry_profile_revision == 2
    assert cache.ignored_regions in ({}, None)
    assert catalog.get_annotation_snapshot(record.id)["legacy_archived"] is True
```

- [ ] **Step 2: 写异步确认入库守卫红灯测试**

```python
def test_confirmed_template_waits_for_review_when_active_geometry_cannot_fit(
        evolution, catalog, record, glare_image, original_front_count):
    job = evolution.submit_confirmation(record.id, "front", glare_image, operation_id="add-1")
    completed = evolution.run_next(force=True)
    assert completed["state"] == "needs_review"
    assert completed["review_reason"] == "geometry_mask_low_confidence"
    assert len(catalog.get(record.id).front_images) == original_front_count
```

- [ ] **Step 3: 运行恢复与确认守卫测试并确认仍使用旧位置标注**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_catalog.py tests/test_template_evolution.py -k "geometry or legacy" -q`

Expected: FAIL because recovery still reapplies `active_interference_groups` and evolution still calls `_requires_mask_review`.

- [ ] **Step 4: 恢复路径只加载活动几何 profile**

`WorkpieceCatalog.recover` 在基础缓存恢复后读取 manifest 的 `geometry_mask_active_revision`，通过 profile manager 重建/加载对应候选缓存；删除现有把 `active_interference_groups` 重新应用到预测缓存的分支。旧查询接口增加 `legacy_archived: true`，旧保存响应增加 `archived: true`，且不调用 `prepare_template_masks/set_template_masks`。

- [ ] **Step 5: 用几何校验替换 `_requires_mask_review` 的位置递推依赖**

```python
def _requires_geometry_review(self, job: dict, record) -> dict | None:
    for item in job["items"]:
        result = self.geometry_profiles.validate_new_template(
            job["workpiece_id"], item["orientation"], Path(item["path"])
        )
        if result["status"] != "active":
            return result
    return None
```

`TemplateEvolution.__init__` 增加必需依赖 `geometry_profiles: GeometryMaskProfiles`；测试和 `ServiceRuntime.set_ready` 均显式传入，避免 worker 自行创建第二套档案状态。

审核通过后 `append_templates` 必须按当前活动 profile 重建候选并原子发布，不能把旧缓存中的矩形 `ignored_regions` 按索引复制到新模板。

- [ ] **Step 6: 运行恢复与模板演进回归**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_catalog.py tests/test_template_evolution.py tests/test_interference_masks.py -q`

Expected: PASS；旧位置工具函数仍可供归档读取测试，但没有生产预测调用点。

- [ ] **Step 7: 提交生命周期接入**

```powershell
git add src/workpiece_catalog.py src/template_evolution.py tests/test_workpiece_catalog.py tests/test_template_evolution.py
git commit -m "feat: guard template evolution with geometry profiles"
```

---

### Task 6: TCP 后台任务协议与错误映射

**Files:**
- Modify: `src/orientation_tcp_service.py`
- Modify: `tests/test_orientation_tcp_service.py`
- Modify: `tests/test_orientation_service_integration.py`

**Interfaces:**
- Consumes: `GeometryMaskProfiles` 的公开接口。
- Produces: 七个增量 JSON-lines 命令及 `predict.geometry_mask` 可选对象。

- [ ] **Step 1: 写命令契约红灯测试**

```python
def test_geometry_validation_returns_job_without_streaming_long_work(client):
    response = request(client, "validate_geometry_mask_draft", {
        "workpiece_id": "part", "base_library_revision": 4,
        "base_draft_revision": 2, "operation_id": "v-1"})
    assert response["ok"] is True
    assert response["job"]["state"] in {"queued", "running"}

def test_geometry_profile_rejects_stale_draft_with_stable_code(client):
    response = request(client, "publish_geometry_mask_profile", stale_fields)
    assert response["error"]["code"] == "STALE_GEOMETRY_PROFILE"
```

- [ ] **Step 2: 运行协议测试并确认命令未知**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_tcp_service.py -k "geometry_mask" -q`

Expected: FAIL because the dispatcher has no geometry commands.

- [ ] **Step 3: 扩展 `ServiceRuntime` 生命周期**

`ServiceRuntime.set_ready` 构造一个 `GeometryMaskProfiles` 并注入 `TemplateEvolution`；runtime snapshot 增加 `geometry_profiles`；服务器 shutdown 依次停止 profile worker 与 evolution worker，重复 shutdown 安全。

- [ ] **Step 4: 实现命令与稳定错误码**

命令为：`get_geometry_mask_profile`、`save_geometry_mask_draft`、`validate_geometry_mask_draft`、`get_geometry_mask_validation_job`、`geometry_mask_validation_job_action`、`publish_geometry_mask_profile`、`rollback_geometry_mask_profile`。错误至少映射为 `INVALID_GEOMETRY_PROFILE`、`STALE_GEOMETRY_PROFILE`、`GEOMETRY_VALIDATION_NOT_READY`、`GEOMETRY_VALIDATION_FAILED`、`GEOMETRY_OVERRIDE_REQUIRED`。

- [ ] **Step 5: 保持协议 V1 和消息大小保护并运行集成回归**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_tcp_service.py tests/test_orientation_service_integration.py -q`

Expected: PASS；现有 hello/list/register/predict/annotation 命令字段不变，未知 `geometry_mask` 响应字段可被旧客户端忽略。

- [ ] **Step 6: 提交 TCP 接入**

```powershell
git add src/orientation_tcp_service.py tests/test_orientation_tcp_service.py tests/test_orientation_service_integration.py
git commit -m "feat: expose geometry mask profile commands"
```

---

### Task 7: Qt 几何形状画布

**Files:**
- Create: `qt_app/geometryrulecanvas.h`
- Create: `qt_app/geometryrulecanvas.cpp`
- Create: `qt_app/tests/test_geometryrulecanvas.cpp`
- Create: `qt_app/tests/test_geometryrulecanvas.pro`
- Modify: `qt_app/workpiece_orientation.pro`

**Interfaces:**
- Consumes: 原图 `QImage`、形状类型、原图坐标粗画/拟合/遮罩叠加。
- Produces: `GeometryRuleCanvas`，可画圆、椭圆、旋转矩形并输出原图坐标 JSON。

- [ ] **Step 1: 写 letterbox 坐标、形状和旋转红灯测试**

```cpp
void TestGeometryRuleCanvas::draggedCircleUsesNativeImageCoordinates() {
    GeometryRuleCanvas canvas;
    canvas.resize(600, 400);
    canvas.setImage(QImage(300, 300, QImage::Format_RGB32));
    canvas.setTool(GeometryRuleCanvas::Circle);
    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(200, 100));
    QTest::mouseRelease(&canvas, Qt::LeftButton, {}, QPoint(400, 300));
    const QJsonObject shape = canvas.coarseShape();
    QCOMPARE(shape.value("shape").toString(), QStringLiteral("circle"));
    QVERIFY(qAbs(shape.value("cx").toDouble() - 150.0) < 1.0);
}
```

- [ ] **Step 2: 构建测试确认类不存在**

Run: `E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe qt_app\tests\test_geometryrulecanvas.pro -o qt_app\tests\Makefile.geometryrulecanvas`

Expected: compile setup fails because `geometryrulecanvas.h` is absent.

- [ ] **Step 3: 实现画布接口、渲染层和原图坐标变换**

```cpp
class GeometryRuleCanvas : public QWidget {
    Q_OBJECT
public:
    enum Tool { Circle, Ellipse, RotatedRectangle };
    void setImage(const QImage &image);
    void setTool(Tool tool);
    void setCoarseShape(const QJsonObject &shape);
    void setFitOverlay(const QJsonObject &fit, const QImage &maskOverlay);
    QJsonObject coarseShape() const;
    void setRotationDegrees(qreal degrees);
signals:
    void shapeChanged(const QJsonObject &shape);
};
```

粗画为青色虚线、拟合边界为绿色实线、最终忽略层为半透明橙色；旋转矩形角度由数值框调整，避免 V1 引入复杂旋转手柄。

- [ ] **Step 4: 编译并运行 Qt 画布测试**

Run: `cmd /c "call C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat && E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe qt_app\tests\test_geometryrulecanvas.pro -o qt_app\tests\Makefile.geometryrulecanvas && nmake /f qt_app\tests\Makefile.geometryrulecanvas /NOLOGO && qt_app\tests\release\test_geometryrulecanvas.exe"`

Expected: PASS for all three shapes, resize/letterbox mapping, overlay and rotation.

- [ ] **Step 5: 提交画布**

```powershell
git add qt_app/geometryrulecanvas.h qt_app/geometryrulecanvas.cpp qt_app/tests/test_geometryrulecanvas.cpp qt_app/tests/test_geometryrulecanvas.pro qt_app/workpiece_orientation.pro
git commit -m "feat: add geometry rule drawing canvas"
```

---

### Task 8: Qt 规则管理、验证轮询与显式发布

**Files:**
- Create: `qt_app/geometrymaskmanager.h`
- Create: `qt_app/geometrymaskmanager.cpp`
- Create: `qt_app/tests/test_geometrymaskmanager.cpp`
- Create: `qt_app/tests/test_geometrymaskmanager.pro`
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp`
- Modify: `qt_app/mainwindow.ui`
- Modify: `qt_app/workpiece_orientation.pro`
- Modify: `qt_app/tests/test_mainwindow.cpp`
- Modify: `qt_app/tests/test_mainwindow.pro`

**Interfaces:**
- Consumes: Task 6 响应快照和 Task 7 画布。
- Produces: 草稿编辑、验证任务轮询、汇总发布、回退和预测复核状态 UI。

- [ ] **Step 1: 写对话框状态机红灯测试**

```cpp
void TestGeometryMaskManager::publishDisabledUntilCompletedValidationMatchesDraft() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 1));
    QVERIFY(!dialog.findChild<QPushButton *>("publishButton")->isEnabled());
    dialog.setValidationJob(completedJob(4, 2));
    QVERIFY(dialog.findChild<QPushButton *>("publishButton")->isEnabled());
}

void TestGeometryMaskManager::warningPublishRequiresOverrideReason() {
    GeometryMaskManagerDialog dialog;
    dialog.setValidationJob(jobWithRegression());
    QSignalSpy spy(&dialog, &GeometryMaskManagerDialog::publishRequested);
    dialog.findChild<QPushButton *>("publishButton")->click();
    QCOMPARE(spy.count(), 0);
}
```

- [ ] **Step 2: 构建管理器测试并确认新类不存在**

Run: `E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe qt_app\tests\test_geometrymaskmanager.pro -o qt_app\tests\Makefile.geometrymaskmanager`

Expected: compile setup fails because `geometrymaskmanager.h` is absent.

- [ ] **Step 3: 实现三栏管理器与完整草稿编辑**

```cpp
signals:
    void snapshotRequested(const QString &workpieceId);
    void saveDraftRequested(const QJsonObject &draft, int libraryRevision,
                            int draftRevision);
    void validateRequested(int libraryRevision, int draftRevision);
    void validationJobRequested(const QString &jobId);
    void validationJobActionRequested(const QString &jobId, const QString &action);
    void publishRequested(const QString &jobId, int libraryRevision,
                          int draftRevision, const QString &overrideReason);
    void rollbackRequested(int libraryRevision);
```

左栏管理方向、基准和规则；中栏显示模板/任务状态；右栏显示画布、拟合诊断和遮罩预览。启停、删除、重命名均修改本地完整草稿，点击保存后统一发送 `save_geometry_mask_draft`。

- [ ] **Step 4: MainWindow 接入短请求轮询，不修改 BackendClient 单请求约束**

验证启动响应返回 `job_id` 后，用 `QTimer` 在客户端 `Ready` 时发送 `get_geometry_mask_validation_job`；收到 `queued/running` 继续轮询，`completed/failed/cancelled/interrupted` 停止。关闭管理器只停止 Qt 轮询，不隐式取消后端任务。

- [ ] **Step 5: 替换主界面入口和预测状态呈现**

按钮文字改为“管理几何干扰规则”。`predict.geometry_mask.status == low_confidence|unavailable|unreadable` 时结果区显示“遮罩未启用，需复核”，保留原预测标签和原有证据；`active` 时显示 profile revision 和覆盖率；`not_configured` 不增加警告。

- [ ] **Step 6: 归档旧位置管理器而不删除用户数据文件**

从 `workpiece_orientation.pro` 和 MainWindow 生产入口移除 `annotationcanvas/annotationeditor/annotationmanager`，源文件保留用于读取旧审计数据；首次发现旧 `interference_groups` 时显示一次迁移提示，不提供“重新递推”动作。

- [ ] **Step 7: 构建并运行 Qt 管理器和主窗口测试**

Run: `cmd /c "call C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat && E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe qt_app\tests\test_geometrymaskmanager.pro -o qt_app\tests\Makefile.geometrymaskmanager && nmake /f qt_app\tests\Makefile.geometrymaskmanager /NOLOGO && qt_app\tests\release\test_geometrymaskmanager.exe"`

Run: `cmd /c "call C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat && E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe qt_app\tests\test_mainwindow.pro -o qt_app\tests\Makefile.mainwindow && nmake /f qt_app\tests\Makefile.mainwindow /NOLOGO && qt_app\tests\release\test_mainwindow.exe"`

Expected: PASS；无活动规则、验证运行、警告覆盖发布、回退、旧归档提示和低置信度预测均有覆盖。

- [ ] **Step 8: 提交 Qt 工作流**

```powershell
git add qt_app/geometrymaskmanager.h qt_app/geometrymaskmanager.cpp qt_app/tests/test_geometrymaskmanager.cpp qt_app/tests/test_geometrymaskmanager.pro qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/workpiece_orientation.pro qt_app/tests/test_mainwindow.cpp qt_app/tests/test_mainwindow.pro
git commit -m "feat: manage and publish geometry mask profiles"
```

---

### Task 9: 全链路回归、性能测量与验收记录

**Files:**
- Create: `docs/verification/geometric-calibration-interference-masks-results.md`
- Modify: `scripts/smoke_orientation_service.ps1`

**Interfaces:**
- Consumes: 所有前序任务。
- Produces: 可重复的后端/Qt/真实模板验证记录，不修改识别阈值。

- [ ] **Step 1: 扩展 smoke 脚本验证 profile 生命周期**

脚本按顺序执行 hello、list、get profile、保存草稿、启动验证、轮询完成、显式发布、predict、回退，并验证每个 response 的 `request_id`、`ok`、revision 与 `geometry_mask.status`。

- [ ] **Step 2: 运行全部 Python 测试**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q`

Expected: 全部 PASS；若机器相关环境测试按既有 marker 跳过，在验收记录中列出具体测试名和原因。

- [ ] **Step 3: 构建 Qt Release 并运行所有 Qt 测试程序**

Run: `powershell -ExecutionPolicy Bypass -File scripts\build_qt5.ps1`

Expected: `qt_app\build-release\release\workpiece_orientation.exe` 构建成功。

逐个运行：`test_appconfig.exe`、`test_backendclient.exe`、`test_backendprocessmanager.exe`、`test_annotationmanager.exe`、`test_geometryrulecanvas.exe`、`test_geometrymaskmanager.exe`、`test_mainwindow.exe`；旧 annotation 测试继续证明归档查看代码未损坏。

- [ ] **Step 4: 用现有工件库记录正确性与耗时**

对至少一个 35+35 工件分别记录：无规则基线、中心 `inside`、外边界 `outside`、二者并集的建档验证总耗时；对同一批测试图记录预测 p50/p95、遮罩拟合失败数、正反正确/错误/不确定数。验证期间并行发送至少 10 次 predict，记录是否完成与延迟，确认后台任务没有占住 TCP 请求。

- [ ] **Step 5: 写验收记录并明确性能影响**

`docs/verification/geometric-calibration-interference-masks-results.md` 必须包含：提交号、环境、模板数量、测试命令和结果、Qt 构建路径、基线/遮罩准确率、验证总耗时、预测 p50/p95、缓存文件大小、低置信度样本清单，以及未通过项的具体原因。

- [ ] **Step 6: 运行 diff 和文档自检**

Run: `git diff --check`

Expected: `git diff --check` 无错误，验收记录列出的命令、结果和失败原因均为实际值。

- [ ] **Step 7: 提交验收材料**

```powershell
git add scripts/smoke_orientation_service.ps1 docs/verification/geometric-calibration-interference-masks-results.md
git commit -m "test: verify geometric interference masking"
```

---

## Plan Self-Review Matrix

| Spec requirement | Implementation task |
|---|---|
| 对象坐标、基准边界、圆/椭圆/旋转矩形、内外与并集 | Task 1 |
| PP-ShiTuV2 与 ALIKED/LightGlue 同时屏蔽、双方向查询、低置信度基线 | Task 2 |
| 草稿、版本、审计、旧库和陈旧修订 | Task 3 |
| 后台验证、留一回归、60%/30% 警告、显式发布与回退 | Task 4 |
| 恢复、旧位置标注归档、异步确认入库 | Task 5 |
| 增量 TCP V1 命令与稳定错误码 | Task 6 |
| 圆/椭圆/旋转矩形 Qt 画框与叠加预览 | Task 7 |
| Qt 规则管理、任务轮询、发布汇总与预测提示 | Task 8 |
| 全量回归、真实 35+35 工件与性能报告 | Task 9 |
