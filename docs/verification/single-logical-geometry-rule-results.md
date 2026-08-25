# 单一逻辑几何规则与双方向标定验证结果

日期：2026-08-23
分支：`feature/20260815/workpiece-orientation-desktop-ui`

## 结论

本轮已完成“一条用户可见逻辑规则、正反面分别标定”的 schema、持久化、验证发布、推理、TCP 和 Qt 工作流改造。旧 v1 active 在 v2 发布前保持可用；运行时只使用程序选中的拟合边界；未知方向查询会生成正反两张处理图，并通过一次 PP-ShiTu batch size 2 提取全局特征。

自动化验证共通过：

- Python：252/252 个唯一用例通过。其中 249 个常规用例通过，3 个生产模型集成用例在显式启用后通过。
- Qt 5.14.2/MSVC：62/62 个用例通过。
- 生产数据烟雾回归：M1、M2、M7 各取 5+5 模板，并各测试 1 张留出正面和 1 张留出反面，共 6/6 判断正确。该结果只代表固定留出样本，不等同于整个数据集准确率。

## 修改范围

核心实现：

- `src/geometry_profile_schema.py`：v2 canonical schema、v1 安全迁移、冲突处置、运行时物化。
- `src/geometry_calibration.py`：保存 selected fitted geometry，seed 仅作搜索和编辑溯源；复用轮廓上下文。
- `src/geometry_mask_profiles.py`：v1 active/v2 draft 共存、验证、发布、回滚和缓存修订门禁。
- `src/orientation_classifier.py`：按方向物化规则、双方向查询、PP-ShiTu 双图 batch、整体回退和分段计时。
- `src/orientation_tcp_service.py`：v2 预览上下文、迁移冲突处置命令、稳定错误码。
- `qt_app/geometrymaskmanager.h/.cpp`：稳定逻辑规则清单、双方向 calibration、精确覆盖层、迁移处置和简化发布。
- `qt_app/mainwindow.h/.cpp`：保存→验证→发布编排、迁移请求和可操作错误提示。
- `scripts/benchmark_geometry_rule_inference.py`：基线、v1、v2 串行参考和 v2 batch 的可复现实测入口。

测试覆盖：

- `tests/test_geometry_calibration.py`
- `tests/test_geometry_mask_profiles.py`
- `tests/test_orientation_classifier.py`
- `tests/test_orientation_tcp_service.py`
- `tests/test_orientation_service_integration.py`
- `qt_app/tests/test_geometrymaskmanager.cpp`
- `qt_app/tests/test_geometryrulecanvas.cpp`
- `qt_app/tests/test_backendclient.cpp`
- `qt_app/tests/test_mainwindow.cpp`

## Python 测试

几何、服务和目录相关测试：

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_geometry_calibration.py tests/test_geometry_mask_profiles.py tests/test_orientation_classifier.py tests/test_orientation_tcp_service.py tests/test_orientation_service_integration.py tests/test_workpiece_catalog.py tests/test_template_evolution.py -q -p no:cacheprovider
```

结果：197 个用例中 194 通过、3 个生产集成用例按预期跳过；0 failure，0 error，13.081 s。

其余 Python 测试模块：55/55 通过，0 failure，0 error，6.926 s。两组覆盖 `tests/` 下全部 18 个测试文件，合计 249 个常规用例通过。

生产模型集成测试：

```powershell
$env:WORKPIECE_ORIENTATION_RUN_INTEGRATION='1'
$env:WORKPIECE_ORIENTATION_PROJECT_ROOT='E:\Project\wang\pp_813'
$env:WORKPIECE_ORIENTATION_MODEL_DIR='E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer'
$env:WORKPIECE_ORIENTATION_PYTHON='E:\python\anaconda3\envs\shitu\python.exe'
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_service_integration.py -q -p no:cacheprovider
```

结果：3/3 通过，0 skip，119.124 s。每个测试内部验证一张留出正面和一张留出反面，因此共有 6/6 个查询标签正确。

关键回归用例：

- `test_saving_v2_draft_keeps_legacy_active_unchanged`
- `test_publish_and_rollback_swap_manifest_revision_and_runtime_cache`
- `test_fit_reference_persists_selected_fitted_geometry_not_seed`
- `test_two_direction_fits_reuse_one_contour_context`
- `test_geometry_prediction_batches_front_and_back_global_embeddings`

## Qt 测试

Qt 5.14.2、MSVC Release、`-platform offscreen`：

| 测试程序 | 结果 |
|---|---:|
| `test_backendclient.exe` | 14/14 |
| `test_geometryrulecanvas.exe` | 11/11 |
| `test_geometrymaskmanager.exe` | 24/24 |
| `test_mainwindow.exe` | 13/13 |
| 合计 | 62/62 |

覆盖的关键交互包括：规则清单不随方向/模板改变、按精确上下文显示拟合边界、迁移冲突处置、缺失方向标定定位、以及单按钮保存→验证→发布。

## 真实工件库迁移审计

只读审计对象：`31f082d1a04e486b9345846f4d585033`。

- 模板：正面 28 张，反面 28 张。
- 当前文档：schema v1，library revision 1，draft revision 7，`active_revision=null`。
- 旧草稿：正面 0 条规则，反面 4 条规则。
- 内存迁移后：4 条逻辑规则、正面 0 个 calibration、反面 4 个 calibration、2 个显式冲突。
- 四个旧 `source_rule_id` 全部保留，没有静默合并或删除。
- 当前旧草稿的 `geometry` 与 `seed_geometry` 相同，这是旧数据在修复前保存造成的；新预览与保存路径已有确定性测试保证保存 selected fitted geometry。

因此，这个真实工件库目前还不能发布完整 v2：需要先处理 2 个迁移冲突，并补齐 4 条规则的正面 calibration。旧数据没有被自动改写。

## 准确率与性能

### 已取得的真实结果

- M1/M2/M7 生产模型烟雾回归：6/6 查询正确。
- 单 batch 行为：`test_geometry_prediction_batches_front_and_back_global_embeddings` 验证正反处理图只调用一次 PP-ShiTu，batch size 为 2。

### P50/P95 状态

真实工件库尚无已发布 active geometry profile（`active_revision=null`），且 v2 正面 calibration 不完整。为了避免伪造或用不合法的“复制反面到正面”代替标定，本次没有填写 baseline/v1/v2 的虚构毫秒数。

完成正反标定并发布后，可在同一工控机直接运行：

```powershell
E:\python\anaconda3\envs\shitu\python.exe scripts\benchmark_geometry_rule_inference.py `
  --workpiece-id 31f082d1a04e486b9345846f4d585033 `
  --query-dir data\benchmark_queries `
  --warmup 10 `
  --repeats 30 `
  --output runtime_reports\geometry-v2-benchmark.json
```

脚本会输出每种模式的 `samples`、`warmup`、`repeats`、`p50_ms`、`p95_ms`、`mean_ms`、`timings_ms`、predictor batch sizes 和标签分布。`v2_serial_reference` 仅用于基准比较，生产路径仍是 `v2_batch`。

## 环境与不变项

- Windows 10 19045
- Python 3.10.20
- NumPy 1.24.4
- PaddlePaddle 3.2.2，设备 `gpu:0`
- PyTorch 2.7.1+cu118，CUDA 11.8
- GPU：NVIDIA GeForce RTX 4060 Ti
- Qt 5.14.2 / MSVC

`third_party` 无工作树改动，模型和权重文件未变化。`src/orientation_classifier.py` 的现有融合阈值仍为：

- `GLOBAL_MARGIN_THRESHOLD = 0.05`
- `LOCAL_MIN_SCORE = 4.0`
- `LOCAL_MIN_MARGIN = 0.5`
- `LOCAL_OVERRIDE_MARGIN = 3.0`

PP-ShiTu、ALIKED、LightGlue 的实现、权重、输入尺寸和融合阈值均未修改，也不需要重新训练模型。

## 已知限制与下一步

当前唯一未完成的真实性能证据是完整 v2 active 下的 P50/P95。其前置条件不是代码修改，而是对真实工件库完成迁移冲突处置、正面独立标定、验证和发布；之后运行上述脚本即可得到可对比、可复现的结果。
