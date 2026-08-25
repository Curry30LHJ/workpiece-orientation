# 几何干扰规则识别率修复实施计划

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 修复几何干扰规则引入后识别率下降的问题：内部规则必须拟合到锚点内部边界，内部区域不能用白色背景抹掉有效特征，验证必须真实执行留一法并在严重退化或回归时阻止发布；保持 PP-ShiTuV2、ALIKED、LightGlue、融合阈值、旧版缓存和未启用几何规则的行为不变。

**Architecture:** 几何校准增加规则相对于物体锚点的拓扑约束和诊断字段；图像处理把内部干扰改为边缘保持修复、外部侵入继续使用常量填充；分类器统一使用同一几何处理路径，并对外部区域才删除局部关键点；验证重新提取每个留一查询的全局/局部特征；发布把严重退化、验证不完整和真实回归升级为阻塞问题。

**Tech Stack:** Python 3、NumPy、OpenCV、pytest、现有 Paddle/ALIKED/LightGlue 推理代码；不改模型权重和现有融合常量。

## Global Constraints

- 只修改本计划列出的 Python 源码、Python 测试和验证文档；保留工作区中其他未相关改动。
- 不修改 PP-ShiTuV2、ALIKED、LightGlue 的实现、权重、输入尺寸和现有融合阈值。
- 旧版没有几何配置的缓存继续按原始特征预测；旧版 5+5 工件库和缺少新增诊断字段的 profile 必须可恢复。
- 几何拟合失败、修复失败、留一跳过不得产生可发布的部分 mask/cache。
- 每个任务先新增复现测试并运行 RED，再写最小实现并运行 GREEN。

---

## Task 1 — 拓扑候选回归测试

Files: E:/Project/wang/pp_813/tests/test_geometry_calibration.py

- 增加同心外圆/内圆合成图，测试 test_inside_rule_rejects_outer_contour_candidate：fit 为 active，规则 relation_to_anchor 为 contained，拟合尺寸小于锚点（2% 容差），ignored_ratio < 0.55；当前实现应先失败。
- 增加 test_inside_rule_without_nested_boundary_returns_low_confidence_without_mask：只有外轮廓时为 low_confidence，reason_code 为 topology_constraint_failed，顶层没有 ignore_mask。
- 增加 test_outside_rule_candidate_encloses_anchor：外部候选 relation_to_anchor 为 encloses 且 topology_valid 为 true。
- 运行 RED：Set-Location E:\Project\wang\pp_813；python -m pytest -q tests/test_geometry_calibration.py -k "topology or nested_boundary or outside_rule_candidate"。

## Task 2 — 实现拓扑筛选与诊断

Files: E:/Project/wang/pp_813/src/geometry_calibration.py

- 增加 TOPOLOGY_TOLERANCE_RATIO = 0.02。
- 实现 _shape_relation(candidate, anchor, mode, tolerance)，在候选旋转坐标系比较中心距离和半轴/半宽高；inside 只接受候选严格位于 anchor 内，outside 只接受候选包围 anchor；支持 circle、ellipse、rotated_rectangle，返回合法性、contained/encloses/invalid、尺寸比例。
- 实现 _filter_rule_candidates，并在 _fit_rule、fit_reference 的 rule 候选路径使用；显式 rule_candidate_index 仍针对筛选后的候选。
- 无合法候选返回 low_confidence、topology_constraint_failed，不生成 mask；成功诊断增加 relation_to_anchor、scale_ratio、topology_valid；same_as_anchor 外部路径标记 encloses。
- 运行 GREEN：python -m pytest -q tests/test_geometry_calibration.py -k "topology or nested_boundary or outside_rule_candidate"，再运行该文件全量测试。

## Task 3 — 内部修复/外部过滤测试

Files: E:/Project/wang/pp_813/tests/test_geometry_calibration.py

- 增加 apply_geometry_fit(image, fit, fill_bgr) 测试：inside 区域不能整块变为白色/固定中性色，outside 仍使用指定 fill，未忽略像素不变。
- 增加 geometry_feature_mask(fit) 测试：inside 返回全零，outside 返回外部 mask，多规则按 OR 合并。
- 覆盖 inside keypoint 保留和 outside keypoint 过滤，保留旧 apply_ignore_mask/filter_features_by_mask 兼容测试。
- 运行 RED：python -m pytest -q tests/test_geometry_calibration.py -k "geometry_fit or feature_mask or inpaint"。

## Task 4 — 实现统一几何图像处理

Files: E:/Project/wang/pp_813/src/geometry_calibration.py

- 实现 apply_geometry_fit(image, fit, fill_bgr) 和 geometry_feature_mask(fit)。
- outside mask 先用 BGR 常量填充；inside mask 与 outside 差集使用 cv2.inpaint(..., cv2.INPAINT_TELEA) 小半径修复。OpenCV 失败抛 GeometryCalibrationError，让上层回退 raw cache，禁止静默白色填充。
- geometry_feature_mask 只返回 outside 规则 mask；inside 用于图像修复，不删除内部关键点；保留 apply_ignore_mask 原语义。
- 运行上述 RED 测试至 GREEN，再运行 test_geometry_calibration.py 全量。

## Task 5 — 分类器统一处理与真实留一法

Files: E:/Project/wang/pp_813/tests/test_orientation_classifier.py；E:/Project/wang/pp_813/src/orientation_classifier.py

先测试：
- 将旧测试改名为 test_leave_one_out_reextracts_query_features，期望 2+2 fixture 中 global predictor 和 local extractor 各增加 4 次，evaluated=4、correct_to_wrong=0。
- 增加 inside 几何缓存测试，确认 feature_mask_mode 为 none 或 outside_only 且内部关键点不被全部删除；增加 outside 只过滤外部关键点测试。
- RED：python -m pytest -q tests/test_orientation_classifier.py -k "leave_one_out_reextracts or inside or outside"。

实现：
- prepare_geometry_cache 和 _predict_geometry 统一调用 apply_geometry_fit；局部特征只用 geometry_feature_mask 过滤；报告增加 feature_mask_mode、keypoints_before/after、remaining_ratio。
- 保留拟合失败、needs_reseed、旧 profile 的 raw fallback。
- 重写 leave_one_out_report：每个原始模板重新读取、按同一 profile 处理、重新提取 global/local；用移除当前模板的临时 cache 评分，记录 evaluated、skipped、fit_failures、changed_predictions、correct_to_wrong。skipped > 0 时 status=incomplete；异常/跳过不伪装完成。
- 增加 _cache_without_template，仅复制和移除当前候选，不改变正式 cache。
- GREEN：先运行该筛选测试，再运行 test_orientation_classifier.py 全量。

## Task 6 — 验证发布门禁测试

Files: E:/Project/wang/pp_813/tests/test_geometry_mask_profiles.py

- 增加严重 ignored_ratio >= 0.55 的 geometry_mask_too_large、remaining_ratio <= 0.20 的 keypoint_retention_critical 测试，override 也不能 publish。
- 增加 leave_one_out status=incomplete 且 skipped>0 的 leave_one_out_incomplete 阻塞测试。
- 增加 correct_to_wrong>0 的 geometry_regression 阻塞测试，override 不能绕过。
- 保留 warning-only（>0.40、<0.30）可用 override 发布的旧契约。
- RED：python -m pytest -q tests/test_geometry_mask_profiles.py -k "critical or incomplete or regression"。

## Task 7 — 实现验证阻塞门禁

Files: E:/Project/wang/pp_813/src/geometry_mask_profiles.py

- 增加 _validation_blocking_issues(report, regression)，保留模板 review 阻塞，追加面积/关键点严重退化、验证非 completed、skipped>0、真实回归。
- 调整 _run_validation 顺序：candidate/report 后先执行 leave_one_out，再统一计算 warnings 和 blocking_issues；异常记录 failed、fit_failures 并阻塞。
- 保留 warnings、blocking_issues、regression 三个独立字段；warning-only 仍可用 override_reason，blocking 和真实回归一律拒绝；旧 job 缺字段按兼容默认值。
- GREEN：python -m pytest -q tests/test_geometry_mask_profiles.py -k "critical or incomplete or regression"，再运行该文件全量。

## Task 8 — 端到端回归与 A/B

Files: E:/Project/wang/pp_813/tests/test_orientation_tcp_service.py（仅必要时）；E:/Project/wang/pp_813/tests/test_workpiece_catalog.py（仅必要时）；新增 E:/Project/wang/pp_813/docs/verification/geometry-mask-accuracy-fix-results.md

- 运行相关套件：python -m pytest -q tests/test_geometry_calibration.py tests/test_orientation_classifier.py tests/test_geometry_mask_profiles.py tests/test_orientation_tcp_service.py tests/test_workpiece_catalog.py；环境允许时再运行 python -m pytest -q。
- 用旧版无 geometry profile 的 5+5 fixture 恢复并预测，确认 raw baseline；确认不等模板数量不依赖固定数量。
- 对 runtime_library 当前 35+35 工件执行同一 40 张留出集的 raw/修复后 geometry A/B，比较准确率、changed_predictions、每张 query 延迟、缓存构建延迟；此前 7 个由 geometry 引入的翻转应消除，未达标则逐条报告。
- 记录拟合成功率、阻塞/跳过数、关键点保留率和耗时；清理临时脚本/调试输出，不纳入无关文件。
- 最终运行 git -C E:\Project\wang\pp_813 diff --check 和 git -C E:\Project\wang\pp_813 status --short。

## Task 9 — 交付检查

- [ ] 新增测试均先 RED 后 GREEN。
- [ ] 旧 profile/cache 恢复通过，模型与融合阈值未改。
- [ ] 拟合/inpaint/留一失败不会生成可发布半成品。
- [ ] warnings、blocking_issues、regression 可独立审计，Qt 旧响应仍可解析。
- [ ] 验证文档报告修改文件、测试结果、A/B 准确率和建库/预测耗时变化。
- [ ] 只提交本修复相关文件，不处理已有无关 dirty 文件。

