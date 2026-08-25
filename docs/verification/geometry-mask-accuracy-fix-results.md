# 几何干扰规则识别率修复验证记录

日期：2026-08-19  
分支：feature/20260815/workpiece-orientation-desktop-ui

## 修复范围

本轮修复了三类导致识别率下降的问题：

1. 几何候选现在按规则方向和物体锚点做拓扑约束。inside 只能接受锚点内部边界，outside 只能接受包围锚点的边界；没有合法候选时返回 topology_constraint_failed，不生成部分 ignore_mask。
2. inside 区域不再用白色/中性色整块填充，也不再删除内部全部局部关键点。inside 使用 OpenCV Telea inpaint 保留边缘纹理，outside 仍按原有常量填充并过滤外部局部关键点。模板和查询使用同一处理路径。
3. 留一验证重新读取每张模板并重新提取全局/局部特征，记录 skipped、fit_failures 和 changed_predictions。严重忽略比例、关键点保留率不足、验证不完整或真实回归会阻塞发布。含不安全模板的几何缓存在运行时自动回退 raw baseline 并标记 needs_review；查询拟合忽略比例达到 0.55 也会回退。

未修改 PP-ShiTuV2、ALIKED、LightGlue、模型权重、输入尺寸和现有融合阈值。旧版无几何 profile 的模板缓存仍使用 raw baseline。

## 修改文件

生产代码：

- src/geometry_calibration.py
- src/orientation_classifier.py
- src/geometry_mask_profiles.py

测试：

- tests/test_geometry_calibration.py
- tests/test_orientation_classifier.py
- tests/test_geometry_mask_profiles.py
- tests/test_workpiece_catalog.py

计划：

- docs/superpowers/plans/2026-08-19-geometry-mask-accuracy-fix.md

## 测试结果

相关跨模块回归：

- 命令：python -m pytest -q tests/test_geometry_calibration.py tests/test_orientation_classifier.py tests/test_geometry_mask_profiles.py tests/test_orientation_tcp_service.py tests/test_workpiece_catalog.py
- 结果：127 passed

项目 tests 全量（使用 shitu 环境，包含 LightGlue/Paddle）：

- 命令：E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q --maxfail=1
- 结果：195 passed, 3 skipped
- 3 个 skipped 是已有真实服务/模型集成测试，没有新增失败。

默认 Python 运行仓库根目录全量 pytest 时，收集阶段因环境没有安装 LightGlue 失败；使用 shitu 环境运行根目录全量时，第三方 PaddleClas 自带 test_hubserving.py 在收集阶段触发既有 TypeError。项目自身 tests 已用正确环境完整通过，这两项未修改。

## 真实 35+35 A/B

对象：runtime_library/616532fcc692442a88e1898f57908f9c，正反各 35 张。  
留出集：data/1_M1/0 和 data/1_M1/1，各 20 张，共 40 张；排除了与模板内容相同的图片哈希。模型、融合阈值和查询顺序相同。

第一轮（只启用几何候选，不做严重退化回退）：

- raw baseline：40/40，100%
- geometry：38/40，95%
- 2 个样本发生 front/back 翻转，且查询几何状态为 active
- 原因定位：两张翻转查询的拟合忽略比例分别为 0.5702 和 0.5741，已超过发布安全阈值 0.55；此前已发布旧 profile 没有经过新的硬门禁。

加入安全回退后的最终 A/B：

- raw baseline：40/40，100%
- geometry cache：40/40，100%
- changed_predictions：0
- 几何候选缓存构建：约 6638 ms
- baseline 40 张查询总耗时：约 112345 ms，p50 832.7 ms，p95 915.3 ms
- geometry 40 张查询总耗时：约 33011 ms，p50 819.8 ms，p95 869.1 ms
- 当前 profile 模板拟合：front 29/35 active、6 个低置信度；back 15/35 active、20 个低置信度。由于存在不安全模板，最终运行时按设计回退 raw baseline，因此没有把不可靠的混合几何缓存用于预测。

这次 A/B 的结论是：旧 profile 在未验证时确实会降低正确率；修复后的验证门禁和运行时回退消除了这两次翻转，并保留了低置信度提示。若现场重新标定后所有模板通过拓扑、面积、关键点和真实留一验证，才会发布并启用几何缓存；否则系统安全地使用 raw baseline。

## 交付注意事项

- 重新验证/发布规则时，必须等待 regression.status=completed 且 skipped=0。
- blocking_issues 为空才允许发布；warnings 仍可由操作员填写覆盖原因处理。
- 当前 35+35 profile 应先重新绘制/调整内部边界并重新验证，不建议直接沿用旧 active revision。

