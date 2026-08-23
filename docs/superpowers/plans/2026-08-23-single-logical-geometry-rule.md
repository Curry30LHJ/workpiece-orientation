# 单一逻辑几何规则与双方向标定 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把几何干扰规则改为一份用户可见的逻辑规则和正反两套独立标定，确保保存的是程序拟合边界、未知方向推理使用正确的双分支，并修复 Qt 规则清单、覆盖层、验证诊断和发布流程。

**Architecture:** 新增独立 schema v2 模块，保存顶层逻辑规则及按方向索引的 calibration；运行时把 v2 物化为现有标定器使用的单方向 profile，旧 v1 active 在 v2 成功发布前保持原样。查询图先复用一次边缘/轮廓上下文，再生成正反处理图；PP-ShiTu 以单个双图 batch 提取全局特征，ALIKED/LightGlue 保持现有语义和阈值。Qt 始终展示顶层规则清单，方向和模板只改变标定上下文，所有覆盖层按 `rule_id + direction + template_id + revision` 精确匹配。

**Tech Stack:** Python 3.10、NumPy、OpenCV、pytest、PaddlePaddle/PaddleClas、ALIKED/LightGlue、JSON-lines TCP；Qt 5.14.2 Widgets/Test、C++17、qmake、MSVC 2019。

## Global Constraints

- 不修改 PP-ShiTu、ALIKED、LightGlue 的实现、权重、输入尺寸或现有融合阈值。
- `mode` 继续使用 `inside` / `outside`；`margin_semantics` 继续使用 `signed_boundary_v2`。
- `margin_ratio > 0` 表示边界向外扩张，`margin_ratio < 0` 表示向内收缩。
- 手绘 `seed_geometry` 只用于搜索和编辑；运行时只允许使用程序选中的 `geometry`。
- 每条启用规则必须同时具有 `front` 和 `back` 的 `ready` calibration 才能发布。
- 任一方向拟合失败、缓存修订不一致或有效面积不安全时，整次几何增强回退原始基线并标记 `needs_review`。
- 旧 v1 active 在 v2 草稿成功发布前必须继续工作；迁移不得静默合并、删除或覆盖旧规则。
- 正反处理图必须通过一次 PP-ShiTu 双图 GPU batch 获取 embedding；ALIKED/LightGlue 在接口不支持安全 batch 时保持现有逐方向调用。
- 所有变更先写失败测试、确认 RED，再做最小实现并确认 GREEN。
- 当前工作树已有大量修改。每次只暂存任务列出的路径，并在提交前运行 `git diff --cached --name-only`；若暂存内容包含无法确认归属的旧改动，则保留工作树、不创建该任务提交。

---

## File Structure

- Create: `src/geometry_profile_schema.py` — v1/v2 规范化、迁移、冲突处置和运行时物化的唯一责任模块。
- Modify: `src/geometry_calibration.py` — 保存 selected fitted geometry，并提供可复用的查询拟合上下文。
- Modify: `src/geometry_mask_profiles.py` — 文档持久化、草稿/active 兼容、验证、发布和缓存修订编排。
- Modify: `src/orientation_classifier.py` — v2 运行时 profile、双方向查询和 PP-ShiTu batch。
- Modify: `src/orientation_tcp_service.py` — v2 请求校验、迁移冲突处置命令及稳定错误码。
- Modify: `qt_app/geometrymaskmanager.h`
- Modify: `qt_app/geometrymaskmanager.cpp` — 单一规则清单、双方向 calibration 编辑、精确覆盖层和简化发布。
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp` — 迁移处置请求、统一发布编排和错误提示。
- Modify: `tests/test_geometry_mask_profiles.py`
- Modify: `tests/test_geometry_calibration.py`
- Modify: `tests/test_orientation_classifier.py`
- Modify: `tests/test_orientation_tcp_service.py`
- Modify: `tests/test_orientation_service_integration.py`
- Modify: `qt_app/tests/test_geometrymaskmanager.cpp`
- Modify: `qt_app/tests/test_mainwindow.cpp`
- Create: `scripts/benchmark_geometry_rule_inference.py` — 固定预热、重复次数和分段耗时输出的可复现实测入口。
- Create: `docs/verification/single-logical-geometry-rule-results.md` — 测试、迁移、准确率和 P50/P95 结果。

---

### Task 1: 建立 schema v2、v1 迁移和运行时物化

**Files:**
- Create: `src/geometry_profile_schema.py`
- Modify: `src/geometry_mask_profiles.py:20-330`
- Test: `tests/test_geometry_mask_profiles.py`

**Interfaces:**
- Produces: `normalize_profile_v2(profile: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `migrate_profile_v1(profile: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `materialize_direction_profile(profile: Mapping[str, Any], direction: str) -> dict[str, Any]`
- Produces: `materialize_runtime_profile(profile: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `DuplicateLogicalRuleError(InvalidGeometryProfileError)`，重复顶层 ID 或未处置同义重复规则使用该异常。
- Preserves: `geometry_mask_profiles.normalize_geometry_profile()` 作为兼容导出，但返回 canonical v2。
- Test helpers: 在 `tests/test_geometry_mask_profiles.py` 增加 `v2_profile_fixture()`、`legacy_pair_fixture()` 和 `legacy_front_zero_back_four_fixture()`，全部返回纯 JSON-compatible dict。

- [ ] **Step 1: 写 schema v2 和唯一配对迁移的失败测试**

```python
def test_v2_uses_one_logical_rule_with_two_direction_calibrations():
    profile = normalize_profile_v2(v2_profile_fixture())
    assert [rule["rule_id"] for rule in profile["rules"]] == ["glare"]
    assert profile["directions"]["front"]["calibrations"]["glare"]["geometry"]["r"] == 0.70
    assert profile["directions"]["back"]["calibrations"]["glare"]["geometry"]["r"] == 0.64


def test_unique_legacy_front_back_rules_are_paired_without_copying_geometry():
    migrated = migrate_profile_v1(legacy_pair_fixture())
    assert len(migrated["rules"]) == 1
    rule_id = migrated["rules"][0]["rule_id"]
    assert migrated["directions"]["front"]["calibrations"][rule_id]["geometry"]["r"] == 0.70
    assert migrated["directions"]["back"]["calibrations"][rule_id]["geometry"]["r"] == 0.64
    assert migrated["migration"]["conflicts"] == []
```

- [ ] **Step 2: 写重复、歧义和缺失方向不得静默处理的失败测试**

```python
def test_legacy_duplicates_are_preserved_and_reported_as_conflicts():
    migrated = migrate_profile_v1(legacy_front_zero_back_four_fixture())
    assert len(migrated["rules"]) == 4
    source_ids = {
        item["source_rule_id"]
        for conflict in migrated["migration"]["conflicts"]
        for item in conflict["sources"]
    }
    assert source_ids == {"back-glare-a", "back-glare-b", "back-intrusion-a", "back-intrusion-b"}
    assert all(
        rule["rule_id"] in migrated["directions"]["back"]["calibrations"]
        for rule in migrated["rules"]
    )
```

- [ ] **Step 3: 运行 RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py -k "v2_uses_one or legacy_front_back or legacy_duplicates" -q -p no:cacheprovider --basetemp=pytest-v2-schema-red
```

Expected: FAIL，原因是 `geometry_profile_schema` 尚不存在，当前 profile 仍按方向保存 `rules`。

- [ ] **Step 4: 实现 canonical v2 和迁移算法**

`geometry_profile_schema.py` 固定公开常量和结构：

```python
PROFILE_SCHEMA_VERSION = 2
LEGACY_PROFILE_SCHEMA_VERSION = 1
SIGNED_MARGIN_SEMANTICS = "signed_boundary_v2"
DIRECTIONS = ("front", "back")


def materialize_direction_profile(profile: Mapping[str, Any], direction: str) -> dict[str, Any]:
    normalized = normalize_profile_v2(profile)
    side = normalized["directions"][direction]
    rules = []
    for logical in normalized["rules"]:
        calibration = side["calibrations"].get(logical["rule_id"])
        if calibration is None or calibration["state"] != "ready":
            continue
        rules.append({
            **logical,
            "geometry": deepcopy(calibration["geometry"]),
            "seed_geometry": deepcopy(calibration.get("seed_geometry", calibration["geometry"])),
            "editor_state": "ready",
        })
    return {
        "anchor": deepcopy(side.get("anchor")),
        "rules": rules,
        "template_reviews": deepcopy(side.get("template_reviews", {})),
        "fill_bgr": deepcopy(side.get("fill_bgr")),
    }
```

迁移顺序必须是：同一且唯一 `rule_id` → 唯一语义签名 → 每条未配对旧规则各自保留为逻辑规则并记录 conflict。语义签名只使用规范化 `name/shape/mode/margin_ratio/enabled`，不比较 geometry。

- [ ] **Step 5: 增加约束校验**

拒绝孤儿 calibration、重复顶层 `rule_id`、未知方向、无效 shape/mode、越界 margin、`ready` 却缺少 geometry，以及 calibration 引用不存在规则。缺失另一方向 calibration 可保存草稿，但不能发布。

- [ ] **Step 6: 运行 GREEN 与现有 profile 测试**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py -q -p no:cacheprovider --basetemp=pytest-v2-schema-green
```

- [ ] **Step 7: 提交任务**

```powershell
git add -- src/geometry_profile_schema.py src/geometry_mask_profiles.py tests/test_geometry_mask_profiles.py
git diff --cached --name-only
git commit -m "feat: add v2 logical geometry profile schema"
```

---

### Task 2: 兼容保存 v2 草稿、保留 v1 active 并处理迁移冲突

**Files:**
- Modify: `src/geometry_profile_schema.py`
- Modify: `src/geometry_mask_profiles.py:407-594,1061-1318`
- Test: `tests/test_geometry_mask_profiles.py`

**Interfaces:**
- Consumes: Task 1 的 v2 canonicalizer 和 materializer。
- Produces: `resolve_migration_conflict(profile, conflict_id, resolution) -> dict[str, Any]`
- Produces: `GeometryMaskProfiles.resolve_migration(workpiece_id: str, conflict_id: str, resolution: Mapping[str, Any], *, expected_library_revision: int, expected_draft_revision: int, operation_id: str) -> dict[str, Any]`
- Produces: `GeometryProfileMigrationConflictError(GeometryProfilePublishError)`，供发布门禁和 TCP 稳定错误码使用。
- Preserves: revision 文件中 profile 自带的 `schema_version`，允许 v2 document 暂时包含 v1 active 和 v2 draft。
- Test helpers: 增加 `conflicted_v2_fixture()`，并复用本文件现有 fake library/catalog 创建 `profile_store` fixture；`save_conflicted_v2_draft()` 和 `complete_validation_job()` 只封装公开 API，不直接写内部成员。

- [ ] **Step 1: 写 v1 active 不被迁移草稿改写的失败测试**

```python
def test_saving_migrated_v2_draft_keeps_legacy_active_bytes_unchanged(profile_store):
    before = deepcopy(profile_store.snapshot("m7")["active"])
    draft = migrate_profile_v1(profile_store.snapshot("m7")["draft"])
    saved = profile_store.save_draft(
        "m7", draft,
        expected_library_revision=1,
        expected_draft_revision=0,
        operation_id="save-v2",
    )
    assert saved["draft"]["schema_version"] == 2
    assert saved["active"] == before
    assert saved["active"]["schema_version"] == 1
```

- [ ] **Step 2: 写冲突处置和发布门禁失败测试**

```python
def test_unresolved_migration_conflict_blocks_publish(profile_store):
    saved = save_conflicted_v2_draft(profile_store)
    job = complete_validation_job(profile_store, saved)
    with pytest.raises(GeometryProfilePublishError, match="MIGRATION_CONFLICT"):
        profile_store.publish(
            "m7", job["job_id"],
            expected_library_revision=1,
            expected_draft_revision=saved["draft_revision"],
            operation_id="publish-conflicted",
        )


def test_keep_only_resolution_removes_explicit_duplicates_but_preserves_survivor():
    resolved = resolve_migration_conflict(
        conflicted_v2_fixture(), "duplicate-back-glare",
        {"action": "keep_only", "survivor_rule_id": "back-glare-a"},
    )
    assert [rule["rule_id"] for rule in resolved["rules"]] == ["back-glare-a", "back-intrusion-a", "back-intrusion-b"]
    assert "back-glare-a" in resolved["directions"]["back"]["calibrations"]
    assert "back-glare-b" not in resolved["directions"]["back"]["calibrations"]
```

- [ ] **Step 3: 运行 RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py -k "legacy_active_bytes or migration_conflict or keep_only_resolution" -q -p no:cacheprovider --basetemp=pytest-v2-persistence-red
```

- [ ] **Step 4: 实现双 schema document 加载和原子保存**

`_load_document()` 对旧 document 只在内存中生成 v2 draft；`active` 和旧 revision 文件保持原始 v1。第一次保存 v2 草稿时写出顶层 document v2，但把原 `active` 对象逐字段保留。所有 active 缓存入口调用：

```python
runtime_profile = materialize_runtime_profile(stored_profile)
candidate, report = prepare(
    workpiece_id,
    record,
    runtime_profile,
    self.calibrator,
    progress_callback,
    base_cache=base_cache,
)
```

- [ ] **Step 5: 实现明确冲突处置**

只接受以下 resolution：

```python
SUPPORTED_MIGRATION_ACTIONS = {"keep_only", "pair", "keep_separate"}
```

- `keep_only`：调用者必须给出 conflict 中的 survivor，明确删除同一方向其余重复项；
- `pair`：必须给出一个 front 和一个 back source，把两个 calibration 绑定到 survivor；
- `keep_separate`：保留所有逻辑规则，仅把歧义冲突标记为人工确认；缺失方向仍由发布门禁阻止，直到用户完成标定。

每次处置使用 library/draft revision 和 operation_id，写入新 draft revision；不得修改 active。
已处理 conflict 从 `migration.conflicts` 移除，并把 `{conflict_id, resolution, resolved_at}` 追加到 `migration.resolutions`，保证处置历史可审计。

- [ ] **Step 6: 实现发布与回滚兼容**

发布 v2 时把 canonical v2 写入新 revision，再物化并构建缓存，成功后原子切 active。回滚可指回 v1 或 v2 revision，缓存构建统一通过 `materialize_runtime_profile()`。

- [ ] **Step 7: 运行 GREEN**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py tests/test_workpiece_catalog.py -k "geometry or migration or publish or rollback" -q -p no:cacheprovider --basetemp=pytest-v2-persistence-green
```

- [ ] **Step 8: 提交任务**

```powershell
git add -- src/geometry_profile_schema.py src/geometry_mask_profiles.py tests/test_geometry_mask_profiles.py tests/test_workpiece_catalog.py
git diff --cached --name-only
git commit -m "feat: preserve legacy geometry profiles during v2 migration"
```

---

### Task 3: 保存 selected fitted geometry，seed 只作编辑溯源

**Files:**
- Modify: `src/geometry_calibration.py:501-593`
- Modify: `src/geometry_mask_profiles.py:622-689`
- Test: `tests/test_geometry_calibration.py`
- Test: `tests/test_geometry_mask_profiles.py`

**Interfaces:**
- Consumes: `GeometryCalibrator._relative_geometry(pixel_shape, anchor)`。
- Produces: `GeometryCalibrator.fit_reference(image, seed_shape, *, mode, margin_ratio, anchor_candidate_index, rule_candidate_index)["profile_patch"]["geometry"]` 来自 selected rule candidate。
- Produces: 同一返回值中的 `profile_patch.seed_geometry` 来自原始手绘 seed。
- Adds: `GeometryMaskProfiles.preview_rule(workpiece_id: str, *, expected_library_revision: int, rule_id: str, direction: str, template_id: str, seed_shape: Mapping[str, Any], mode: str, margin_ratio: float, anchor_candidate_index: int | None, rule_candidate_index: int | None) -> dict[str, Any]`。
- Produces: `GeometryContextMismatchError(GeometryValidationError)`，用于拒绝过期的规则、方向、模板或 revision 上下文。
- Test helpers: 在校准测试中增加 `concentric_circle_image(outer_radius: int, inner_radius: int) -> np.ndarray`；profile preview 测试复用 Task 2 的 `profile_store` fixture。

- [ ] **Step 1: 写拟合值与 seed 值不同的确定性失败测试**

```python
def test_fit_reference_persists_selected_fitted_geometry_not_seed():
    image = concentric_circle_image(outer_radius=90, inner_radius=58)
    seed = {"shape": "circle", "cx": 128.0, "cy": 128.0, "r": 82.0}
    result = GeometryCalibrator().fit_reference(image, seed, mode="inside")
    fitted = result["rule_fit"]["fitted_shape"]
    anchor = result["anchor_fit"]["fitted_shape"]
    expected_fitted = GeometryCalibrator._relative_geometry(fitted, anchor)
    expected_seed = GeometryCalibrator._relative_geometry(seed, anchor)
    assert result["profile_patch"]["geometry"]["r"] == pytest.approx(expected_fitted["r"])
    assert result["profile_patch"]["seed_geometry"]["r"] == pytest.approx(expected_seed["r"])
    assert result["profile_patch"]["geometry"]["r"] != pytest.approx(
        result["profile_patch"]["seed_geometry"]["r"]
    )
```

- [ ] **Step 2: 写 preview 上下文失败测试**

```python
def test_preview_echoes_exact_rule_direction_template_and_revision(profile_store):
    preview = profile_store.preview_rule(
        "m7",
        expected_library_revision=1,
        rule_id="glare",
        direction="back",
        template_id="back:25.png",
        seed_shape={"shape": "circle", "cx": 180.0, "cy": 180.0, "r": 150.0},
        mode="inside",
        margin_ratio=-0.04,
    )
    assert preview["rule_id"] == "glare"
    assert preview["direction"] == "back"
    assert preview["template_id"] == "back:25.png"
    assert preview["base_library_revision"] == 1
```

- [ ] **Step 3: 运行 RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_calibration.py tests/test_geometry_mask_profiles.py -k "persists_selected or preview_echoes_exact" -q -p no:cacheprovider --basetemp=pytest-fitted-save-red
```

Expected: 第一项显示当前 `profile_patch.geometry` 仍等于 seed 相对值；第二项缺少 rule_id/revision。

- [ ] **Step 4: 最小修复 profile patch**

```python
profile_anchor = self._normalized_anchor(anchor, image_shape)
fitted_geometry = self._relative_geometry(rule, anchor)
seed_geometry = self._relative_geometry(seed, anchor)
profile_patch = {
    "anchor": {"mode": "auto", **profile_anchor},
    "geometry": fitted_geometry,
    "seed_geometry": seed_geometry,
    "mode": mode,
    "margin_ratio": margin,
    "margin_semantics": "signed_boundary_v2",
}
```

不要改变 `effective_shape` 的边距计算；profile geometry 保存原始拟合边界，运行时继续应用 `margin_ratio`。

- [ ] **Step 5: 让 preview 严格携带上下文**

`preview_rule()` 强制非空 `rule_id`，响应增加 `base_library_revision`。低置信度响应仍返回上下文，但不得包含 `profile_patch`。

- [ ] **Step 6: 运行 GREEN**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_calibration.py tests/test_geometry_mask_profiles.py -q -p no:cacheprovider --basetemp=pytest-fitted-save-green
```

- [ ] **Step 7: 提交任务**

```powershell
git add -- src/geometry_calibration.py src/geometry_mask_profiles.py tests/test_geometry_calibration.py tests/test_geometry_mask_profiles.py
git diff --cached --name-only
git commit -m "fix: persist selected fitted geometry boundary"
```

---

### Task 4: v2 验证报告、缓存物化和双方向发布门禁

**Files:**
- Modify: `src/geometry_mask_profiles.py:537-620,722-1189`
- Modify: `src/orientation_classifier.py:358-523`
- Test: `tests/test_geometry_mask_profiles.py`
- Test: `tests/test_orientation_classifier.py`

**Interfaces:**
- Consumes: `materialize_runtime_profile()` 和 `materialize_direction_profile()`。
- Produces: validation row 顶层仅保存模板总体字段，`row["rules"]` 保存按 logical `rule_id` 的诊断。
- Produces: publish blocking codes `MISSING_DIRECTION_CALIBRATION`、`FITTED_GEOMETRY_MISSING`、`MIGRATION_CONFLICT`、`PROFILE_CACHE_REVISION_MISMATCH`。
- Produces: `MissingDirectionCalibrationError(GeometryProfilePublishError)`、`FittedGeometryMissingError(GeometryProfilePublishError)` 和 `GeometryCacheRevisionMismatchError(GeometryProfilePublishError)`。
- Test helpers: 增加 `v2_ready_profile()`、`v2_profile_missing_back()`、`v2_profile_missing_fitted_geometry()` 和 `validate_profile()`；record 使用 5 张 front、12 张 back 的现有 fake path 结构。

- [ ] **Step 1: 写方向数量不等且使用同一逻辑规则的缓存失败测试**

```python
def test_v2_cache_materializes_same_rule_for_unequal_direction_counts(classifier, record):
    profile = v2_ready_profile(front_r=0.70, back_r=0.64)
    cache, report = classifier.prepare_geometry_cache("m7", record, profile)
    assert len(cache.global_vectors["front"]) == 5
    assert len(cache.global_vectors["back"]) == 12
    assert {item["rules"][0]["rule_id"] for item in report["front"]} == {"glare"}
    assert {item["rules"][0]["rule_id"] for item in report["back"]} == {"glare"}
```

- [ ] **Step 2: 写缺失方向与修订不一致发布失败测试**

```python
@pytest.mark.parametrize(
    "profile,code",
    [
        (v2_profile_missing_back(), "MISSING_DIRECTION_CALIBRATION"),
        (v2_profile_missing_fitted_geometry(), "FITTED_GEOMETRY_MISSING"),
    ],
)
def test_enabled_v2_rule_publish_requires_both_ready_calibrations(profile_store, profile, code):
    job = validate_profile(profile_store, profile)
    assert any(item["code"] == code for item in job["blocking_issues"])
```

- [ ] **Step 3: 运行 RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py tests/test_orientation_classifier.py -k "v2_cache_materializes or both_ready_calibrations or revision_mismatch" -q -p no:cacheprovider --basetemp=pytest-v2-cache-red
```

- [ ] **Step 4: 在所有缓存入口统一物化**

`prepare_geometry_cache()` 的公开入口可以接受 v1 或 v2，但函数开始处立即执行：

```python
runtime_profile = materialize_runtime_profile(profile)
candidate_profile = deepcopy(runtime_profile)
```

TemplateCache 另存原始 canonical active profile 和 `geometry_profile_revision`，不能把物化结构写回 profile.json。

- [ ] **Step 5: 修正 validation report 结构**

每行使用：

```python
row = {
    "index": index,
    "template_id": template_id,
    "direction": label,
    "status": fit.get("status", "low_confidence"),
    "review_state": review_state,
    "review_reason": review_reason,
    "rules": serializable_fit.get("rules", []),
}
```

候选、support、residual 不复制到 row 顶层。

- [ ] **Step 6: 增加发布硬门禁**

遍历所有 `enabled` logical rule，检查 front/back calibration、state、geometry 和 migration conflict。构建缓存后比较 candidate cache revision 与 draft revision；不一致时拒绝 active 指针切换。

- [ ] **Step 7: 运行 GREEN**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_mask_profiles.py tests/test_orientation_classifier.py tests/test_workpiece_catalog.py -q -p no:cacheprovider --basetemp=pytest-v2-cache-green
```

- [ ] **Step 8: 提交任务**

```powershell
git add -- src/geometry_mask_profiles.py src/orientation_classifier.py tests/test_geometry_mask_profiles.py tests/test_orientation_classifier.py tests/test_workpiece_catalog.py
git diff --cached --name-only
git commit -m "feat: validate and cache bidirectional geometry calibrations"
```

---

### Task 5: 复用查询拟合上下文并批量提取正反 PP-ShiTu 特征

**Files:**
- Modify: `src/geometry_calibration.py:469-850`
- Modify: `src/orientation_classifier.py:193-203,825-966`
- Test: `tests/test_geometry_calibration.py`
- Test: `tests/test_orientation_classifier.py`

**Interfaces:**
- Produces: `GeometryCalibrator.prepare_context(image: np.ndarray) -> GeometryFitContext`
- Changes: `GeometryCalibrator.fit(image, direction_profile, *, context=None) -> dict[str, Any]`
- Produces: `OrientationClassifier._global_embeddings(images: Sequence[np.ndarray]) -> list[np.ndarray]`
- Produces: `OrientationClassifier._prepare_geometry_queries(image: np.ndarray, profile: Mapping[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]`
- Preserves: `_global_embedding(image)` 作为单图包装器。
- Test helpers: 扩展现有 `geometry_profile()` helper 生成 v2 ready profile，并增加 `geometry_cache`、`unsafe_geometry_cache` 与 `query_path` fixtures。

- [ ] **Step 1: 写轮廓上下文只计算一次的失败测试**

```python
def test_two_direction_fits_reuse_one_contour_context(monkeypatch):
    calls = 0
    original = geometry_calibration._contours

    def counted(image):
        nonlocal calls
        calls += 1
        return original(image)

    monkeypatch.setattr(geometry_calibration, "_contours", counted)
    calibrator = GeometryCalibrator()
    image = _synthetic_ellipse()
    context = calibrator.prepare_context(image)
    calibrator.fit(image, front_runtime_profile(), context=context)
    calibrator.fit(image, back_runtime_profile(), context=context)
    assert calls == 1
```

- [ ] **Step 2: 写 PP-ShiTu 双图一次调用的失败测试**

```python
def test_geometry_prediction_batches_front_and_back_global_embeddings(classifier, geometry_cache, query_path):
    result = classifier.predict_with_cache(geometry_cache, query_path)
    assert result["geometry_mask"]["status"] == "active"
    assert classifier.global_predictor.calls == 1
    assert classifier.global_predictor.batch_sizes == [2]
    assert classifier.global_predictor.markers == [[3, 4]]
```

更新 FakeGlobalPredictor，使 `predict(images)` 为每个输入返回对应向量，并记录完整 batch 顺序；不得只读取 `images[0]`。

- [ ] **Step 3: 写任一方向失败整体回退测试**

```python
def test_one_direction_fit_failure_uses_single_raw_baseline(classifier, unsafe_geometry_cache, query_path):
    result = classifier.predict_with_cache(unsafe_geometry_cache, query_path)
    assert result["geometry_mask"]["status"] == "low_confidence"
    assert result["geometry_mask"]["needs_review"] is True
    assert result["geometry_mask"]["fallback"] == "raw_baseline"
```

- [ ] **Step 4: 运行 RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_calibration.py tests/test_orientation_classifier.py -k "reuse_one_contour or batches_front_and_back or one_direction_fit_failure" -q -p no:cacheprovider --basetemp=pytest-v2-query-red
```

- [ ] **Step 5: 实现只读拟合上下文**

```python
@dataclass(frozen=True)
class GeometryFitContext:
    image_shape: tuple[int, int]
    contours: Sequence[np.ndarray]


def prepare_context(self, image: np.ndarray) -> GeometryFitContext:
    return GeometryFitContext(
        image_shape=tuple(image.shape[:2]),
        contours=tuple(_contours(image)),
    )
```

`fit()`、`_fit_shape()` 和 `_fit_rule()` 使用 context.contours；两个方向仍分别选择各自 anchor，不能强行共享正反 calibration 的具体边界。

- [ ] **Step 6: 实现全局 embedding batch**

```python
def _global_embeddings(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
    rgb_images = [image[:, :, ::-1] for image in images]
    with self._inference_lock:
        embeddings = self.global_predictor.predict(rgb_images)
    if len(embeddings) != len(images):
        raise OrientationClassifierError("global predictor returned unexpected batch size")
    return [np.asarray(item, dtype=np.float32) for item in embeddings]


def _global_embedding(self, image: np.ndarray) -> np.ndarray:
    return self._global_embeddings([image])[0]
```

`_predict_geometry()` 先执行 `materialize_runtime_profile(cache.geometry_profile)`，再由 `_prepare_geometry_queries()` 完成两边 fit、遮罩和安全检查，最后一次调用 `_global_embeddings([masked_front, masked_back])`。局部特征仍按方向提取，不修改 LightGlue 匹配路径。任何安全回退都在 `geometry_mask.reason_code` 写入 `GEOMETRY_FALLBACK_TO_BASELINE`，具体失败原因另存为 `fallback_reason`。

- [ ] **Step 7: 添加分段耗时诊断**

结果 `geometry_mask.timings_ms` 固定包含 `fit_context`、`fit_directions`、`mask_build`、`global_batch`、`local_features`、`fusion`。计时只用于诊断，不参与分类决策。

- [ ] **Step 8: 运行 GREEN**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_calibration.py tests/test_orientation_classifier.py -q -p no:cacheprovider --basetemp=pytest-v2-query-green
```

- [ ] **Step 9: 提交任务**

```powershell
git add -- src/geometry_calibration.py src/orientation_classifier.py tests/test_geometry_calibration.py tests/test_orientation_classifier.py
git diff --cached --name-only
git commit -m "perf: batch bidirectional geometry query embeddings"
```

---

### Task 6: 扩展 TCP 契约和稳定错误码

**Files:**
- Modify: `src/orientation_tcp_service.py:419-630`
- Test: `tests/test_orientation_tcp_service.py`
- Test: `tests/test_orientation_service_integration.py`

**Interfaces:**
- Changes: `preview_geometry_mask_rule` 强制 `rule_id`。
- Adds command: `resolve_geometry_mask_migration`。
- Preserves commands: get/save/validate/publish/rollback 的名称和 revision/operation_id 幂等语义。
- Test helpers: 扩展现有 fake geometry profiles，增加 `last_resolution` 和与生产签名一致的 `resolve_migration()`；dispatcher fixture 继续使用当前 ServiceRuntime 注入方式。

- [ ] **Step 1: 写 preview 精确上下文和错误码失败测试**

```python
def test_preview_geometry_rule_requires_rule_id(dispatcher):
    response = dispatcher.dispatch({
        "id": "preview-1",
        "command": "preview_geometry_mask_rule",
        "workpiece_id": "m7",
        "base_library_revision": 1,
        "direction": "front",
        "template_id": "front:00.png",
        "seed_shape": {"shape": "circle", "cx": 100.0, "cy": 100.0, "r": 60.0},
        "mode": "inside",
    })
    assert response["ok"] is False
    assert response["error"]["code"] == "NO_SELECTED_RULE"
```

- [ ] **Step 2: 写冲突处置命令和发布错误映射失败测试**

```python
def test_resolve_geometry_migration_forwards_revisioned_resolution(dispatcher, geometry_profiles):
    response = dispatcher.dispatch({
        "id": "resolve-1",
        "command": "resolve_geometry_mask_migration",
        "workpiece_id": "m7",
        "base_library_revision": 1,
        "base_draft_revision": 3,
        "operation_id": "resolve-op-1",
        "conflict_id": "duplicate-back-glare",
        "resolution": {"action": "keep_only", "survivor_rule_id": "back-glare-a"},
    })
    assert response["ok"] is True
    assert geometry_profiles.last_resolution["conflict_id"] == "duplicate-back-glare"
```

- [ ] **Step 3: 运行 RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_tcp_service.py tests/test_orientation_service_integration.py -k "requires_rule_id or resolve_geometry_migration or geometry_error_code" -q -p no:cacheprovider --basetemp=pytest-v2-tcp-red
```

- [ ] **Step 4: 实现请求校验和命令路由**

`preview_geometry_mask_rule` 在读取 seed 前校验非空 rule_id，并传入 `profiles.preview_rule()`。`resolve_geometry_mask_migration` 必须校验两个 revision、operation_id、conflict_id 和 resolution object，再调用 `profiles.resolve_migration()`。

- [ ] **Step 5: 映射稳定错误码**

后端不得依靠字符串包含关系区分以下情况：

```python
GEOMETRY_ERROR_CODES = {
    MissingDirectionCalibrationError: "MISSING_DIRECTION_CALIBRATION",
    FittedGeometryMissingError: "FITTED_GEOMETRY_MISSING",
    GeometryProfileMigrationConflictError: "MIGRATION_CONFLICT",
    DuplicateLogicalRuleError: "DUPLICATE_LOGICAL_RULE",
    GeometryContextMismatchError: "GEOMETRY_CONTEXT_MISMATCH",
    GeometryCacheRevisionMismatchError: "PROFILE_CACHE_REVISION_MISMATCH",
}
```

保留现有 `INVALID_GEOMETRY_PROFILE`、`STALE_GEOMETRY_PROFILE` 和 validation job 错误兼容。

- [ ] **Step 6: 运行 GREEN**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_tcp_service.py tests/test_orientation_service_integration.py -q -p no:cacheprovider --basetemp=pytest-v2-tcp-green
```

- [ ] **Step 7: 提交任务**

```powershell
git add -- src/orientation_tcp_service.py tests/test_orientation_tcp_service.py tests/test_orientation_service_integration.py
git diff --cached --name-only
git commit -m "feat: expose v2 geometry profile protocol"
```

---

### Task 7: Qt 使用单一逻辑规则清单和双方向 calibration 编辑

**Files:**
- Modify: `qt_app/geometrymaskmanager.h`
- Modify: `qt_app/geometrymaskmanager.cpp:35-65,150-250,637-881,1158-1185`
- Test: `qt_app/tests/test_geometrymaskmanager.cpp`

**Interfaces:**
- Consumes: snapshot `draft.rules[]` 和 `draft.directions[side].calibrations[rule_id]`。
- Removes: `copyRuleToOtherDirection()`、`copyRuleButton_` 和“复制到另一方向”。
- Produces: `logicalRules()`, `currentCalibration()`, `setCurrentCalibration()` Qt 私有 helpers。
- Test helpers: 用 `configuredV2ProfileSnapshot(frontPath, backPath)` 替换旧 v1 fixture，并为每个 QListWidgetItem 写入 `rule_id` 的 `Qt::UserRole`。

- [ ] **Step 1: 把 Qt fixture 改成 v2 并写规则清单稳定性失败测试**

```cpp
void TestGeometryMaskManager::ruleInventoryDoesNotChangeAcrossValidationDirections() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    auto *rules = dialog.findChild<QListWidget *>(QStringLiteral("ruleList"));
    QCOMPARE(rules->count(), 2);
    const QString firstId = rules->item(0)->data(Qt::UserRole).toString();

    QVERIFY(QMetaObject::invokeMethod(&dialog, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 55)));
    QCOMPARE(rules->count(), 2);
    QCOMPARE(rules->item(0)->data(Qt::UserRole).toString(), firstId);
}
```

- [ ] **Step 2: 写新增、删除和公共字段只操作一次的失败测试**

```cpp
void TestGeometryMaskManager::deleteLogicalRuleRemovesBothCalibrations() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    dialog.findChild<QListWidget *>(QStringLiteral("ruleList"))->setCurrentRow(0);
    dialog.findChild<QPushButton *>(QStringLiteral("deleteRuleButton"))->click();
    const QJsonObject draft = dialog.draft();
    QCOMPARE(draft.value("rules").toArray().size(), 1);
    QVERIFY(!draft.value("directions").toObject().value("front").toObject()
                 .value("calibrations").toObject().contains("glare"));
    QVERIFY(!draft.value("directions").toObject().value("back").toObject()
                 .value("calibrations").toObject().contains("glare"));
}
```

- [ ] **Step 3: 运行 RED**

Build and run:

```powershell
Set-Location E:\Project\wang\pp_813\qt_app\tests
& 'E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe' test_geometrymaskmanager.pro CONFIG+=release -o Makefile.geometrymaskmanager
nmake /f Makefile.geometrymaskmanager /NOLOGO
& '.\release\test_geometrymaskmanager.exe' -platform offscreen
```

Expected: fixture/清单测试失败，因为 `refreshRuleList()` 仍读取当前方向的 `rules`。

- [ ] **Step 4: 改为顶层逻辑规则身份**

规则行的 `Qt::UserRole` 保存稳定 `rule_id`，切方向后按 ID 恢复选择，不能只恢复 row index。显示文本包含两个 calibration 状态：

```cpp
const QString text = QStringLiteral("%1 [%2]  正面:%3  反面:%4")
    .arg(rule.value("name").toString(), enabledText,
         calibrationState(QStringLiteral("front"), ruleId),
         calibrationState(QStringLiteral("back"), ruleId));
```

- [ ] **Step 5: 改写编辑和删除语义**

名称、shape、mode、margin、enabled 更新 `draft.rules`；geometry、seed、reference_template、state 更新当前方向 calibration。新增规则只创建一个 logical rule；删除规则同时删除两边 calibration 和关联 rule review。

- [ ] **Step 6: 删除复制操作**

删除按钮、连接、slot、成员和旧复制测试；用“同一规则切换到反面后可直接手绘并生成 back calibration”测试替代。

- [ ] **Step 7: 运行 GREEN**

重复 Step 3，Expected: `test_geometrymaskmanager.exe` 全部通过。

- [ ] **Step 8: 提交任务**

```powershell
git add -- qt_app/geometrymaskmanager.h qt_app/geometrymaskmanager.cpp qt_app/tests/test_geometrymaskmanager.cpp
git diff --cached --name-only
git commit -m "feat: edit one logical geometry rule across directions"
```

---

### Task 8: 修复 Qt 覆盖层状态机和嵌套验证诊断

**Files:**
- Modify: `qt_app/geometrymaskmanager.h`
- Modify: `qt_app/geometrymaskmanager.cpp:520-760,940-1045`
- Modify: `qt_app/geometryrulecanvas.cpp`
- Test: `qt_app/tests/test_geometrymaskmanager.cpp`
- Test: `qt_app/tests/test_geometryrulecanvas.cpp`

**Interfaces:**
- Consumes: preview 精确上下文字段和 validation `row.rules[]`。
- Produces: `currentEditContextKey() -> QString`。
- Preserves: `GeometryRuleCanvas::setCoarseShape()` 和 `setFitOverlay()`；空对象明确清除对应层。
- Test helpers: 增加 `validationWithRuleBoundary()`、`exactPreview()`、`savedV2SnapshotWithGeometry()`、`validationWithNestedMetrics()`、`selectLogicalRule()`、`selectTemplate()` 和 `selectValidationRow()`，返回值或操作目标都显式包含 rule/direction/template/revision。

- [ ] **Step 1: 写未选规则不得回退第一条边界的失败测试**

```cpp
void TestGeometryMaskManager::noSelectedRuleClearsEveryRuleOverlay() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    dialog.setValidationJob(validationWithRuleBoundary(QStringLiteral("glare")));
    auto *rules = dialog.findChild<QListWidget *>(QStringLiteral("ruleList"));
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    rules->clearSelection();
    rules->setCurrentRow(-1);
    QVERIFY(QMetaObject::invokeMethod(&dialog, "refreshTemplatePreview", Qt::DirectConnection));
    QVERIFY(canvas->coarseShape().isEmpty());
    QVERIFY(canvas->fitShape().isEmpty());
}
```

- [ ] **Step 2: 写保存后隐藏 seed、切模板不重放 seed 的失败测试**

```cpp
void TestGeometryMaskManager::savedCalibrationShowsFittedBoundaryOnlyForExactContext() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    dialog.setRulePreview(exactPreview("glare", "front", "front:00.png", 7, fittedCircle(61.0)));
    dialog.setSnapshot(savedV2SnapshotWithGeometry("glare", "front", 61.0, 82.0, 8));
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(canvas->coarseShape().isEmpty());
    QCOMPARE(canvas->fitShape().value("r").toDouble(), 61.0);
    selectTemplate(dialog, QStringLiteral("front:01.png"));
    QVERIFY(canvas->coarseShape().isEmpty());
}
```

- [ ] **Step 3: 写嵌套规则诊断失败测试**

```cpp
void TestGeometryMaskManager::validationDiagnosticsComeFromSelectedNestedRule() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    dialog.setValidationJob(validationWithNestedMetrics("glare", 2, 0.81, 0.013));
    selectLogicalRule(dialog, QStringLiteral("glare"));
    selectValidationRow(dialog, 0);
    const QString text = dialog.findChild<QTextEdit *>(QStringLiteral("geometryDiagnostics"))->toPlainText();
    QVERIFY(text.contains(QStringLiteral("候选: 2")));
    QVERIFY(text.contains(QStringLiteral("边缘支持: 0.810")));
    QVERIFY(text.contains(QStringLiteral("残差: 0.013")));
}
```

- [ ] **Step 4: 运行 RED**

重复 Task 7 的 qmake/nmake/QtTest 命令。Expected: 当前空 rule_id 仍选择第一条；dirty guide 会跨模板显示；表格仍读 row 顶层指标。

- [ ] **Step 5: 实现精确上下文状态机**

`currentEditContextKey()` 使用：

```cpp
return QStringLiteral("%1|%2|%3|%4")
    .arg(currentRuleId(), direction(), currentTemplateId())
    .arg(snapshot_.value(QStringLiteral("draft_revision")).toInt());
```

只有 manual-edit key 与当前 key 相同才显示 coarse seed；收到拟合预览或保存 snapshot 后清空 manual-edit key。`applyFittedBoundaryForCurrentTemplate()` 在 ruleId 为空时先清空 fit overlay 并返回 false，匹配 validation/preview 时要求四项上下文全部相等。

- [ ] **Step 6: 简化模板表和规则诊断**

模板表只保留方向、模板、总体状态和待处理规则数。当前规则的 candidate/support/residual 从 `row.rules` 中按 `rule_id` 查找后显示到 diagnostics；不存在匹配项时显示“当前规则在此模板无拟合报告”，不填 `-1/0`。

- [ ] **Step 7: 运行 GREEN**

构建并运行 `test_geometryrulecanvas.exe` 和 `test_geometrymaskmanager.exe -platform offscreen`，Expected: 全部通过。

- [ ] **Step 8: 提交任务**

```powershell
git add -- qt_app/geometrymaskmanager.h qt_app/geometrymaskmanager.cpp qt_app/geometryrulecanvas.cpp qt_app/tests/test_geometrymaskmanager.cpp qt_app/tests/test_geometryrulecanvas.cpp
git diff --cached --name-only
git commit -m "fix: bind geometry overlays to exact editor context"
```

---

### Task 9: 简化发布流程并提供迁移冲突处置入口

**Files:**
- Modify: `qt_app/geometrymaskmanager.h`
- Modify: `qt_app/geometrymaskmanager.cpp:175-235,1048-1188`
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp:420-610,870-975,1120-1140`
- Test: `qt_app/tests/test_geometrymaskmanager.cpp`
- Test: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Adds signal: `migrationResolutionRequested(conflictId, resolution, libraryRevision, draftRevision)`。
- Adds MainWindow request: `resolve_geometry_mask_migration`。
- Keeps primary action: `publishWorkflowRequested(const QJsonObject &draft, int libraryRevision, int draftRevision, const QString &overrideReason)`。
- Removes standalone visible “发布规则”主入口；底层 publish command 仍由 workflow 调用。
- Adds private value type: `struct MissingCalibration { bool valid = false; QString ruleId; QString direction; };` 和 `firstMissingEnabledCalibration() const`。
- Test helpers: 增加 `v2SnapshotMissingBackCalibration()`、`v2SnapshotWithDuplicateConflict()`、`selectMigrationConflict()` 和 `chooseMigrationSurvivor()`；这些 helper 只通过控件 objectName 操作界面。

- [ ] **Step 1: 写缺少反面 calibration 自动定位的失败测试**

```cpp
void TestGeometryMaskManager::publishWorkflowNavigatesToMissingDirectionCalibration() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(v2SnapshotMissingBackCalibration());
    dialog.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"))->click();
    auto *direction = dialog.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    QCOMPARE(direction->currentData().toString(), QStringLiteral("back"));
    QVERIFY(dialog.findChild<QLabel *>(QStringLiteral("geometryStatusLabel"))->text()
                .contains(QStringLiteral("请先完成反面标定")));
}
```

- [ ] **Step 2: 写迁移冲突明确处置的失败测试**

```cpp
void TestGeometryMaskManager::migrationConflictKeepOnlyEmitsExplicitResolution() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(v2SnapshotWithDuplicateConflict());
    QSignalSpy spy(&dialog, &GeometryMaskManagerDialog::migrationResolutionRequested);
    selectMigrationConflict(dialog, QStringLiteral("duplicate-back-glare"));
    chooseMigrationSurvivor(dialog, QStringLiteral("back-glare-a"));
    dialog.findChild<QPushButton *>(QStringLiteral("resolveMigrationButton"))->click();
    QCOMPARE(spy.count(), 1);
    const QJsonObject resolution = spy.takeFirst().at(1).toJsonObject();
    QCOMPARE(resolution.value("action").toString(), QStringLiteral("keep_only"));
    QCOMPARE(resolution.value("survivor_rule_id").toString(), QStringLiteral("back-glare-a"));
}
```

- [ ] **Step 3: 写主流程 save → validate → publish 顺序测试**

在 `test_mainwindow.cpp` 用 fake BackendClient 响应依次断言：点击一次后先发送 save；收到新 draft revision 后发送 validate；job completed 且无 blocking 后发送 publish。任何阶段错误停止链路并保留当前 active。

- [ ] **Step 4: 运行 RED**

使用相应 `.pro` 分别构建运行 `test_geometrymaskmanager.exe` 与 `test_mainwindow.exe -platform offscreen`。Expected: 缺失 calibration 不定位；无冲突处置 signal；workflow 尚未覆盖新错误码。

- [ ] **Step 5: 实现单一主发布按钮**

界面只把“保存、验证并发布”作为主操作。点击时先本地检查：

```cpp
if (const MissingCalibration missing = firstMissingEnabledCalibration(); missing.valid) {
    selectRuleById(missing.ruleId);
    selectDirection(missing.direction);
    showActionMessage(missing.direction == QStringLiteral("front")
        ? QStringLiteral("请先完成正面标定")
        : QStringLiteral("请先完成反面标定"));
    return;
}
emit publishWorkflowRequested(draft_, libraryRevision(), draftRevision(), overrideReason());
```

保留高级验证/回滚入口；隐藏 standalone publish button，避免用户重复执行保存。

- [ ] **Step 6: 实现冲突面板和后端请求**

冲突面板显示 source 方向、旧规则名、ID 和 calibration 状态；只提供 spec 支持的 `keep_only/pair/keep_separate`。MainWindow 为每次请求生成 operation_id，收到成功 snapshot 后刷新规则和冲突面板。

- [ ] **Step 7: 映射可操作错误提示**

`MISSING_DIRECTION_CALIBRATION` 定位缺失方向；`MIGRATION_CONFLICT` 打开冲突面板；`GEOMETRY_CONTEXT_MISMATCH` 丢弃过期 preview 并重新请求；`PROFILE_CACHE_REVISION_MISMATCH` 提示重新验证；其他错误保留原 code/message，不再只显示 `Internal server error`。

- [ ] **Step 8: 运行 GREEN**

构建并运行两个 Qt 测试。Expected: 全部通过，并确认 Qt 5.14.2 编译无新增 API 兼容问题。

- [ ] **Step 9: 提交任务**

```powershell
git add -- qt_app/geometrymaskmanager.h qt_app/geometrymaskmanager.cpp qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/tests/test_geometrymaskmanager.cpp qt_app/tests/test_mainwindow.cpp
git diff --cached --name-only
git commit -m "feat: streamline geometry rule validation and publish"
```

---

### Task 10: 端到端兼容、真实数据回归和性能报告

**Files:**
- Create: `scripts/benchmark_geometry_rule_inference.py`
- Modify: `tests/test_orientation_service_integration.py`
- Create: `docs/verification/single-logical-geometry-rule-results.md`

**Interfaces:**
- Consumes: 完整 v2 profile、双图 batch、TCP/Qt workflow。
- Produces benchmark JSON fields: `mode`, `samples`, `warmup`, `p50_ms`, `p95_ms`, `mean_ms`, `timings_ms`。
- Test helpers: `service_fixture` 使用临时旧 v1 revision 文件和 fake classifier，公开 `predict_with_legacy_active()`、`create_v2_draft_without_publishing()`、`active_schema_version()`、`complete_both_calibrations()`、`validate_and_publish()`、`rollback()`，每个 helper 只调用公开 service/profile API。

- [ ] **Step 1: 写旧 v1 恢复、v2 发布和回滚集成测试**

```python
def test_legacy_active_survives_v2_draft_publish_and_rollback(service_fixture):
    assert service_fixture.predict_with_legacy_active()["prediction"] in {"front", "back", "uncertain"}
    draft = service_fixture.create_v2_draft_without_publishing()
    assert service_fixture.active_schema_version() == 1
    service_fixture.complete_both_calibrations(draft)
    service_fixture.validate_and_publish()
    assert service_fixture.active_schema_version() == 2
    service_fixture.rollback()
    assert service_fixture.active_schema_version() == 1
```

- [ ] **Step 2: 运行 Python 相关全量测试**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_calibration.py tests/test_geometry_mask_profiles.py tests/test_orientation_classifier.py tests/test_orientation_tcp_service.py tests/test_orientation_service_integration.py tests/test_workpiece_catalog.py tests/test_template_evolution.py -q -p no:cacheprovider --basetemp=pytest-v2-focused
```

Expected: 全部通过；若 Windows 临时目录 ACL 失败，改用新的明确可写 `--basetemp`，不得把基础设施错误改成 skip。

- [ ] **Step 3: 运行 Qt 全量相关测试**

使用 Qt 5.14.2/MSVC release 构建并运行：

```text
test_geometryrulecanvas.exe -platform offscreen
test_geometrymaskmanager.exe -platform offscreen
test_backendclient.exe -platform offscreen
test_mainwindow.exe -platform offscreen
```

Expected: 全部返回 0。

- [ ] **Step 4: 实现可复现 benchmark 脚本**

命令接口固定为：

```powershell
E:\python\anaconda3\envs\shitu\python.exe scripts/benchmark_geometry_rule_inference.py --workpiece-id 31f082d1a04e486b9345846f4d585033 --query-dir data/benchmark_queries --warmup 10 --repeats 30 --output runtime_reports/geometry-v2-benchmark.json
```

脚本对每张图先预热，再分别运行 `baseline`、`v1_active`、`v2_serial_reference`、`v2_batch`；使用 `numpy.percentile(samples, 50/95)`，并汇总 `geometry_mask.timings_ms`。串行 reference 仅在脚本内通过显式测试 seam 调用，不能成为生产默认路径。

- [ ] **Step 5: 在同一工控机和同一数据集实测**

记录硬件、Paddle/PyTorch/CUDA 版本、模板数量、查询数量、预热次数。确认生产 v2 每个查询的 PP-ShiTu predictor 调用记录为一个 batch size 2；不把机器相关毫秒数写成 pytest 硬断言。

- [ ] **Step 6: 回归当前真实缺陷数据**

用当前运行库迁移结果确认：

- 原始 `front=0/back=4` 的四条来源均可见；
- 重复冲突未被静默合并；
- 点击验证表正面首行和反面末行时 logical rule ID 集合相同；
- seed 半径约 `1.344673`、fitted 半径约 `0.949347` 的用例保存 fitted 值；
- 未选规则、缺少报告或 context 不匹配时画布无边界。

- [ ] **Step 7: 写验证报告**

`docs/verification/single-logical-geometry-rule-results.md` 必须列出：修改文件、Python/Qt 测试命令及数量、v1/v2 迁移结果、真实数据准确率、基线/v1/v2 串行/v2 batch 的 P50/P95、各分段耗时、已知环境限制，以及模型/阈值未变化的 diff 证据。

- [ ] **Step 8: 最终质量检查**

```powershell
git diff --check
rg -n "GLOBAL_MARGIN_THRESHOLD|LOCAL_MIN_SCORE|LOCAL_MIN_MARGIN|LOCAL_OVERRIDE_MARGIN" src/orientation_classifier.py
git status --short
```

确认没有未解释的阈值变化、调试输出、临时 profile、生成 Makefile 或 benchmark 大文件进入提交。

- [ ] **Step 9: 提交交付材料**

```powershell
git add -- scripts/benchmark_geometry_rule_inference.py tests/test_orientation_service_integration.py docs/verification/single-logical-geometry-rule-results.md
git diff --cached --name-only
git commit -m "test: verify v2 geometry rules and inference latency"
```

---

## Completion Gate

- [ ] 顶层逻辑规则清单在正反模板间身份和数量不变。
- [ ] 每条启用规则有独立的 front/back ready calibration。
- [ ] 保存后的 runtime geometry 等于 selected fitted candidate，不等于粗略 seed。
- [ ] 旧 v1 active 在 v2 发布前继续预测，迁移冲突无静默数据损失。
- [ ] 未选规则或上下文不匹配时画布不显示边界；保存后只显示 fitted boundary。
- [ ] 验证模板总体状态与当前规则诊断分层显示。
- [ ] 同一查询不预判方向，正反处理图使用一次 PP-ShiTu batch size 2。
- [ ] 任一方向失败整体回退 raw baseline 并标记 `needs_review`。
- [ ] 单一“保存、验证并发布”主流程可完成无冲突发布，并能定位缺失 calibration。
- [ ] Python、Qt 相关测试全部通过，真实 P50/P95 和准确率写入验证报告。
- [ ] PP-ShiTu、ALIKED、LightGlue、模型权重和融合阈值没有变化。
