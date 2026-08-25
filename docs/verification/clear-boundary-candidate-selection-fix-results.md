# 清晰边界候选误判修复验证结果

## 问题结论

`back:01.png` 的内边界已经被边缘检测正确提取。失败发生在后续候选决策：

1. `_rank_candidates()` 在拓扑筛选前只保留相似度前三名，前三名均为外圈附近的重复边缘；真正的内部边界位于其后并被提前丢弃；
2. 圆和近圆的 `fitEllipse` 角度不稳定，但旧评分把该角度当作有效差异；
3. 旧的圆/椭圆外接范围使用旋转矩形公式，圆在 45° 时会被错误放大到真实半径的约 1.414 倍。

## 修复内容

- 圆和椭圆使用准确的旋转椭圆轴对齐外接范围；
- 长短轴差不超过 5% 的圆/近圆不参与角度差评分，近圆候选角度稳定为 0°；
- 质量排序保留全部合格候选，规则先做 inside/outside 拓扑筛选，再最多返回三个候选给界面；
- 失败诊断区分 `boundary_candidate_not_found` 与 `topology_constraint_failed`，并返回质量候选数和拓扑合法候选数；
- 未修改边缘支持、可见比例、残差、搜索带宽和拓扑容差。

## 真实模板回归

输入：

- 图片：`runtime_library/31f082d1a04e486b9345846f4d585033/1/01.png`（界面中的 `back:01.png`）
- 手绘种子：圆心 `(173, 177)`、半径 `142`、角度 `-7.7°`
- 规则：`inside`，边界偏移 `-4%`

结果：

- 状态：`active`（修复前为 `low_confidence`）；
- anchor：中心 `(179.55, 181.23)`，半轴 `(169.26, 168.89)`；
- 内边界：中心 `(177.71, 181.64)`，半轴 `(146.61, 146.51)`；
- 拓扑：`contained`，`topology_valid=true`；
- 返回 rule 候选：3 个；
- 成功生成 `profile_patch`。

30 次同进程回归的热运行中位拟合耗时为 `2.159 ms`，P95 为 `3.024 ms`。首次调用出现一次 OpenCV 冷启动开销，不属于候选筛选本身。候选排序原本已经遍历全部轮廓，本修复只让拓扑检查继续查看其余合格候选，因此未增加模型推理或特征提取开销。

## 自动化测试

- `tests/test_geometry_calibration.py`：`32 passed`；
- `tests/test_geometry_mask_profiles.py`、`tests/test_orientation_classifier.py`、`tests/test_orientation_tcp_service.py`：合计 `96 passed`；
- 总计：`128 passed`。

下游套件首次在受限测试环境中因 pytest 临时目录 ACL 失败，改用仓库内独立临时目录后全部通过；该临时目录已删除。
