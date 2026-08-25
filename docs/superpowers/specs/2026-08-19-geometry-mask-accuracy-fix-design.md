# 几何干扰规则识别率修复设计

## 1. 目标与范围

修复几何干扰规则发布后识别率下降的问题。当前问题表现为内圆规则偶尔拟合到外圆、忽略区域覆盖大半工件、内部区域被白色填充，以及验证报告显示 `correct_to_wrong=0` 但真实查询仍发生方向翻转。

本次变更覆盖四个边界：

1. 几何候选拟合必须尊重 `inside/outside` 与 anchor 的拓扑关系；
2. 内部和外部忽略区域使用不同的图像处理策略，避免制造白色人工边界；
3. 验证必须从原始查询图片重新提取特征，并与线上预测使用同一处理链路；
4. 严重遮罩或验证不完整时不可通过发布硬门。

不修改 PP-ShiTuV2、ALIKED、LightGlue、ROI 比例、特征融合公式、现有融合阈值和模型权重，不需要重新训练。旧版 5+5 工件库、无几何规则工件库和旧位置遮罩缓存继续可恢复。

## 2. 已确认语义

规则含义保持不变：

- `inside`：忽略拟合边界内部；
- `outside`：忽略拟合边界外部；
- `margin_ratio`：有符号边界偏移，`effective = fitted * (1 + margin_ratio)`；
- 正面、反面独立拟合和独立缓存，不互相复制像素边界。

当前规则未满足可靠拟合时不得静默使用错误边界。分类器应保留原始缓存并将结果标记为 `needs_review`，验证/发布层应能看到明确原因。

## 3. 几何拟合修复

### 3.1 候选关系约束

在 `GeometryCalibrator` 内增加候选与 anchor 的关系判定，圆、椭圆和旋转矩形均使用自身轴向尺寸和中心距离进行检查：

- `inside` 候选：候选边界必须完全位于 anchor 的内侧，允许最多 2% 的数值误差，但不能与 anchor 重合或超出 anchor；
- `outside` 候选：候选边界必须包住 anchor，允许最多 2% 的数值误差；规则形状与 anchor 尺寸在当前容差内时继续直接复用 anchor；
- 关系检查在候选排序前执行，默认候选索引只在合法候选集合内生效；
- 合法候选为空时返回 `low_confidence`、`reason_code=topology_constraint_failed`，不生成部分掩膜。

候选报告增加可诊断字段：`relation_to_anchor`、`scale_ratio` 和 `topology_valid`。这些字段只用于报告和界面诊断，不改变现有协议中原有字段的含义。

### 3.2 Anchor 稳定性

Anchor 仍由当前图片拟合，不采用上一张图片的像素坐标。排序继续使用参考模板的对象相对期望，但增加：

- 候选中心必须位于期望中心搜索范围内；
- 候选尺寸偏离参考尺寸时增加明确的尺寸惩罚；
- 接触图像边界、残差超限或不满足最小边缘支持的候选不得成为 anchor；
- 不通过 anchor 关系约束的规则不得借用 anchor 作为规则结果。

这一步只消除内外轮廓混淆，不引入新的分割模型。

### 3.3 失败回退

`fit()` 只要有启用规则无法满足拓扑约束或拟合置信度不足，就返回无 `ignore_mask` 的 `low_confidence` 结果。`OrientationClassifier` 遇到该结果时沿用原始模板向量/局部特征完成预测，同时设置 `geometry_mask.status`、`reason_code` 和 `needs_review=true`。

这样，规则失效的代价是回退到已验证的基线，而不是用错误边界破坏特征。

## 4. 遮罩图像与局部特征处理

### 4.1 分离内部和外部规则

保留 `ignore_mask` 作为局部特征空间的诊断掩膜，同时在拟合结果中保留每条规则的原始掩膜和模式。分类器新增一个内部 helper，根据规则模式生成用于全局 embedding 和局部特征提取的处理图像：

- `inside`：对规则内部使用 OpenCV 边缘保持的修复/平滑（默认 `cv2.inpaint`，固定半径），不使用背景白色常量填充；
- `outside`：使用当前方向的背景填充颜色，外部侵入区域不参与特征；
- 多规则按稳定顺序合成，`outside` 区域不会被内部修复覆盖；
- 模板构建和查询预测调用同一 helper，保证处理一致。

现有 `apply_ignore_mask(image, mask, fill_bgr)` 保留兼容，用于旧位置遮罩和外部常量填充；几何规则不再直接对所有规则调用该函数。

### 4.2 局部特征保留

对于 `inside` 修复后的图像，不再按整块内部掩膜删除所有局部关键点；关键点从修复图像直接提取，以保留稳定边界和纹理支持。对于 `outside`，继续按外部掩膜过滤关键点。报告同时记录：

- `keypoints_before`；
- `keypoints_after`；
- `remaining_ratio`；
- `feature_mask_mode`。

这不是放宽规则，而是把“忽略反光变化”从硬删除关键点改成先消除局部亮度差异，再让 ALIKED/LightGlue 判断稳定结构。

### 4.3 填充颜色兼容

旧 profile 中已有 `fill_bgr` 时继续读取；新几何 profile 可选写入 `fill_strategy`，缺失时按规则模式推导。旧 profile 的 `fill_bgr=[255,255,255]` 不会再用于 `inside` 修复，因此不需要迁移旧模板图片。

## 5. 真实验证与发布安全

### 5.1 留一验证

重写 `leave_one_out_report()` 的查询侧：

1. 读取当前模板原图；
2. 按候选 cache 的 geometry profile 对 front/back 分别拟合查询边界；
3. 使用与 `_predict_geometry()` 相同的内部修复、外部填充、特征提取和局部过滤；
4. 从同方向候选集合中排除当前模板；
5. 执行现有 `_fuse_scores()`，记录方向、分数、边距和几何状态。

报告新增 `fit_failures`、`changed_predictions` 和 `skipped`。任何查询无法按候选 profile 完整验证时，报告状态为 `incomplete`，不能伪装成无误判的 `completed`。

留一验证必须重新调用 `_global_embedding()` 和 `_extract_local()`；不允许直接把缓存模板向量当作查询向量。

### 5.2 警告与硬门

保留当前性能提醒阈值：

- `ignored_ratio > 0.40`：警告；
- `remaining_ratio < 0.30`：警告。

新增不可覆盖的严重问题：

- `ignored_ratio >= 0.55`：`geometry_mask_too_large` blocking issue；
- `remaining_ratio <= 0.20`：`keypoint_retention_critical` blocking issue；
- 留一验证状态不是 `completed` 或 `skipped > 0`：`leave_one_out_incomplete` blocking issue；
- 验证中出现真实方向翻转：沿用回归错误计数并要求重新调整规则，不得以普通说明文字绕过。

已有警告仍可在操作员明确填写原因后覆盖；上述 blocking issue 不允许覆盖。发布前必须完成当前候选 cache 与 profile revision 的一致性检查。

## 6. 测试设计

### Python 单元测试

新增或修改以下 seam：

1. 合成同心外圆/内圆图片：`inside` 规则不得选外圆，选中结果的尺寸必须严格小于 anchor，外圆误选场景返回绿色测试；
2. `outside` 规则必须包住 anchor，内部候选被排除；
3. 关系候选为空时无 `ignore_mask` 并返回 `topology_constraint_failed`；
4. 内部规则使用修复策略而不是白色常量，外部规则仍使用背景填充；
5. 内部规则保留从修复图像提取的关键点，外部规则继续过滤外部关键点；
6. 留一验证重新调用全局和局部特征提取，报告包含实际 `evaluated`/`skipped`；
7. 严重忽略比例、严重关键点损失和不完整验证进入 blocking issues；
8. 旧 profile 缺少新字段仍按兼容路径恢复。

所有测试使用确定性合成图片和现有 fake predictor/matcher，不加载 GPU 模型。

### 受控 A/B 验收

使用当前 35+35 模板和 `data/1_M1` 中未进入模板库的留出图片，分别运行原始 cache 和几何 cache：

- 几何 cache 不得比当前原始基线产生新的系统性方向翻转；
- 任何翻转都必须在报告中有对应的拟合/遮罩诊断；
- 当前已复现的 40 张子集作为回归对照，目标是消除 7 个规则引入的方向翻转。

## 7. 不在本次范围内

- 不重新训练 PP-ShiTuV2、ALIKED 或 LightGlue；
- 不改变 `GLOBAL_MARGIN_THRESHOLD`、`LOCAL_MIN_SCORE`、`LOCAL_OVERRIDE_MARGIN` 等现有融合阈值；
- 不新增分割模型；
- 不改变模板数量、工件删除、确认入库和 Qt 批量检测已有协议；
- 不删除旧版缓存或历史 geometry revision。

