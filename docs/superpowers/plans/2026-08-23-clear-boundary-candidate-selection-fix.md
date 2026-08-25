# 清晰圆形边界候选误判修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复清晰内边界已经被 OpenCV 检出、却因圆形角度惩罚和过早截断候选而返回 `low_confidence` 的问题。

**Architecture:** 保留现有边缘检测、质量阈值和拓扑约束。候选先完成质量排序，再让规则拓扑筛选看到全部合格候选，最后才截取最多三个供界面选择；圆和近圆不使用无物理意义的拟合角度参与评分，圆/椭圆使用准确的轴对齐外接范围进行拓扑判断。

**Tech Stack:** Python 3、NumPy、OpenCV、pytest；不修改 PP-ShiTuV2、ALIKED、LightGlue、融合阈值或模型权重。

## Global Constraints

- 只修改 `src/geometry_calibration.py`、`tests/test_geometry_calibration.py` 和本验证记录。
- 不调整 `MIN_EDGE_SUPPORT`、`MIN_VISIBLE_RATIO`、`MAX_FIT_RESIDUAL_RATIO`、搜索带宽或拓扑容差。
- `fit_reference()` 对外仍最多返回三个 anchor 候选和三个 rule 候选。
- 圆形旋转无意义；明显非圆的椭圆和旋转矩形仍保留旋转评分。
- 拟合失败不得生成 `profile_patch` 或部分遮罩。

---

### Task 1: 建立确定性回归测试

**Files:**
- Modify: `tests/test_geometry_calibration.py`

**Interfaces:**
- Consumes: `_shape_extents(shape) -> tuple[float, float]`、`_shape_error(candidate, expected, image_shape, margin_ratio) -> float`、`GeometryCalibrator._rank_candidates(...) -> list[dict]`、`_filter_rule_candidates(...) -> list[dict]`
- Produces: 三个无需模型权重、可稳定复现本次误判的单元测试。

- [x] **Step 1: 增加圆形角度和外接范围测试**

```python
def test_circle_geometry_is_rotation_invariant_for_scoring_and_extents():
    expected = {"shape": "circle", "cx": 180.0, "cy": 180.0,
                "rx": 142.0, "ry": 142.0, "angle_deg": -7.7}
    rotated = {**expected, "angle_deg": 89.0}
    assert _shape_extents(rotated) == pytest.approx((142.0, 142.0))
    assert _shape_error(rotated, expected, (360, 360), 0.16) == pytest.approx(0.0)
```

- [x] **Step 2: 增加准确椭圆外接范围测试**

```python
def test_rotated_ellipse_uses_exact_axis_aligned_extents():
    shape = {"shape": "ellipse", "cx": 0.0, "cy": 0.0,
             "rx": 20.0, "ry": 10.0, "angle_deg": 45.0}
    expected = np.sqrt(20.0 ** 2 * 0.5 + 10.0 ** 2 * 0.5)
    assert _shape_extents(shape) == pytest.approx((expected, expected))
```

- [x] **Step 3: 增加“合法候选位于相似度前三之后”的测试**

构造半径为 `150、149、151、140` 的同心圆轮廓，以半径 150 为手绘期望和 anchor。前三个相似候选都因与 anchor 重合而不满足 `inside`，半径 140 的候选必须仍可进入拓扑筛选并被保留。

- [x] **Step 4: 运行 RED**

Run: `python -m pytest -q tests/test_geometry_calibration.py -k "rotation_invariant or exact_axis_aligned_extents or beyond_similarity_top_three"`

Expected: 现有实现至少因圆形外接范围、角度评分和 `_rank_candidates()` 只返回前三个而失败。

### Task 2: 最小修复候选评分与筛选顺序

**Files:**
- Modify: `src/geometry_calibration.py`
- Test: `tests/test_geometry_calibration.py`

**Interfaces:**
- Consumes: Task 1 的三个失败测试。
- Produces: 准确 `_shape_extents()`、圆/近圆旋转无关评分、规则拓扑先于候选截断的 `fit_reference()` 与 `_fit_rule()`。

- [x] **Step 1: 修正形状外接范围**

```python
if shape["shape"] != "rotated_rectangle":
    return (
        max(1.0, sqrt((sx * cos_angle) ** 2 + (sy * sin_angle) ** 2)),
        max(1.0, sqrt((sx * sin_angle) ** 2 + (sy * cos_angle) ** 2)),
    )
```

圆是该公式 `sx == sy` 的自然特例，因此其外接范围不随拟合角度变化；旋转矩形继续使用当前线性组合公式。

- [x] **Step 2: 让圆和近圆忽略旋转差异**

增加小型 helper 判定旋转是否有意义：`circle` 永远无意义，椭圆长短轴差不超过 5% 时无意义；只有期望和候选都具有稳定方向时才计算 `angle_error`。圆形候选对外角度归一为 `0.0`。

- [x] **Step 3: 把候选截断移到拓扑筛选之后**

`_rank_candidates()` 返回全部通过质量门的排序结果；anchor 路径立即取 `[:3]`；rule 路径先调用 `_filter_rule_candidates()`，再取 `[:3]`。`_fit_shape()` 仍只使用第一个，`_fit_rule()` 先拓扑筛选再使用第一个。

- [x] **Step 4: 增加失败原因统计**

当规则无候选时，`rule_fit` 写入 `reason_code`：质量候选为零时为 `boundary_candidate_not_found`，质量候选存在但均不满足拓扑时为 `topology_constraint_failed`，并写入 `quality_candidate_count` 与 `topology_valid_candidate_count`。

- [x] **Step 5: 运行 GREEN 和完整校准测试**

Run: `python -m pytest -q tests/test_geometry_calibration.py`

Expected: 全部通过，已有候选列表长度契约仍为 1 至 3。

### Task 3: 真实样本回归与交付核对

**Files:**
- Create: `docs/verification/clear-boundary-candidate-selection-fix-results.md`
- Verify: `runtime_library/31f082d1a04e486b9345846f4d585033/1/01.png`

**Interfaces:**
- Consumes: 修复后的 `GeometryCalibrator.fit_reference()`。
- Produces: 可审计的真实样本状态、所选边界、候选数和测试结果。

- [x] **Step 1: 用截图对应参数回归**

读取 `back:01.png`，使用 `circle(cx=173, cy=177, r=142, angle=-7.7)`、`mode=inside`、`margin_ratio=-0.04` 调用 `fit_reference()`。

- [x] **Step 2: 验证成功条件**

结果必须为 `active`、包含 `profile_patch`，所选规则必须满足 `topology_valid=true` 和 `relation_to_anchor=contained`，拟合边界不得选择物体外轮廓。

- [x] **Step 3: 运行相关回归套件**

Run: `python -m pytest -q tests/test_geometry_calibration.py tests/test_geometry_mask_profiles.py`

- [x] **Step 4: 检查工作区差异**

Run: `git diff --check -- src/geometry_calibration.py tests/test_geometry_calibration.py docs/verification/clear-boundary-candidate-selection-fix-results.md`

确认没有改动模型、阈值、Qt 协议和用户其他文件。当前工作区已有大量未提交修改，因此本补丁不单独执行 git commit，避免把同文件中此前用户工作错误地混入提交。
