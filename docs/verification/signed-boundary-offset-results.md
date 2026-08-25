# Signed boundary offset verification

日期：2026-08-18

## 已实现

- `margin_ratio` 采用统一的带符号边界偏移：负值向内收缩，零保持拟合边界，正值向外扩张。
- `inside` 和 `outside` 使用同一套几何偏移公式；`outside` 只在绘制完成后取补集。
- 旧规则缺少 `margin_semantics` 时保持兼容：`inside: +raw`、`outside: -raw`，缺少旧字段时仍使用 `0.02` 默认值。
- 新规则保存 `margin_semantics: "signed_boundary_v2"`，默认偏移为 `0.0`。
- Qt 控件改为 `边界偏移`，范围 `-94%..+94%`，并显示负/零/正的含义；预览画布优先显示 `effective_shape`。
- 任一方向含 `needs_reseed` 规则时，模板缓存和查询都回退到该工件的原始特征，不使用旧几何遮罩，并返回 `needs_reseed/needs_review`。

## Python 验证

使用环境：`E:\python\anaconda3\envs\shitu\python.exe`。

| 命令 | 结果 |
| --- | --- |
| `pytest tests/test_geometry_calibration.py -q -p no:cacheprovider` | 21 passed |
| `pytest tests/test_geometry_mask_profiles.py -k "signed_rule or legacy_margin or missing_legacy or margin_semantics or front_and_back_signed_offsets" -q -p no:cacheprovider` | 8 passed |
| `pytest tests/test_orientation_tcp_service.py -k "preview_geometry_mask_rule" -q -p no:cacheprovider` | 10 passed |
| `pytest tests/test_orientation_classifier.py -k "needs_reseed_direction" -q -p no:cacheprovider` | 1 passed |
| `py_compile`（四个生产模块及对应测试模块） | 通过 |

完整 TCP 回归在当前机器上得到 `37 passed, 1 error`；唯一错误是 pytest 的 `tmp_path` 无法创建锁文件（`PermissionError: ...pytest-of-Administrator...\.lock`），不是断言失败。包含临时目录的完整几何配置回归同样只因该 Windows 临时目录 ACL 失败；不涉及本次代码断言。

## Qt 验证

Qt 5.14.2/MSVC 2019 Release 构建通过：

- `qt_app/build-test-signed-margin-red`：`test_geometrymaskmanager.exe` 构建成功；负/零/正控件测试和新规则签名测试均返回 0。
- `qt_app/build-test-signed-canvas-green`：`effectiveShapeIsPreferredOverRawShape` 返回 0。
- `qt_app/build-release-signed`：主程序 `release/workpiece_orientation.exe` 构建成功。

已有的画布拖拽回归中，两个旧测试在当前工作树上仍报告半径/尺寸断言失败；该问题与本次签名偏移逻辑无关，新增的有效边界测试通过。

## 性能影响

边界偏移只在拟合结果上做常数时间的尺寸缩放；候选搜索、PP-ShiTu、ALIKED、LightGlue 和融合阈值未改变。因此建库/预测耗时不会随正负号产生可测的模型级增量，仍主要由图像读取、特征提取和匹配决定。`needs_reseed` 回退路径反而跳过该方向几何拟合，使用原始缓存。
