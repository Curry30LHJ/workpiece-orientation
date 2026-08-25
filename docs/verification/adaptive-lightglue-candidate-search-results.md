# Adaptive LightGlue Candidate Search 验收结果

- 逐图标签一致：通过
- 逐图复检状态一致：通过
- 两进程输入指纹一致且完整：通过
- M1/M2/M7 发布门禁：通过
- 正式默认模式：`adaptive`

## 环境与可复现信息

| 项目 | 值 |
| --- | --- |
| 时间戳（UTC） | 2026-08-25T07:43:47.828731+00:00 |
| 平台 | Windows-10-10.0.19045-SP0 |
| Python | 3.10.20 | packaged by conda-forge | (main, Jun 11 2026, 03:28:51) [MSC v.1944 64 bit (AMD64)] |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| GPU 驱动 | 560.94 |
| Paddle / CUDA | 3.2.2 / 11.8 |
| Torch / CUDA | 2.7.1+cu118 / 11.8 |
| NumPy | 1.24.4 |
| Git 提交 | `1b241925df4875ddc613c321e3dc5a46dc020563` |
| 工作树 | dirty |
| tracked binary diff SHA-256 | `7e5d30a2886feb1a8cc1e2cb21e014a88ca320d575469f49e30546b33f5f6796` |
| 模型 ID | `general_PPLCNetV2_base_pretrained_v1.0_infer` |
| M1 工件库 ID | `31f082d1a04e486b9345846f4d585033` |
| 每数据集 warmup | 5 |

### 输入指纹摘要

| 模式 | overall SHA-256 | project sources SHA-256 | model SHA-256 |
| --- | --- | --- | --- |
| exhaustive | `ae59205f57f71d8ad4d5808a6c668b3eaebcec8edd16a52aaaf4619c8c1a6143` | `d97aaed124e37b0a1a44a5400f708a975d217ba48e067f36cacb14b928fc0c71` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |
| adaptive | `ae59205f57f71d8ad4d5808a6c668b3eaebcec8edd16a52aaaf4619c8c1a6143` | `d97aaed124e37b0a1a44a5400f708a975d217ba48e067f36cacb14b928fc0c71` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |

### 隔离进程

| 模式 | PID | 端口 | 传输 | 总耗时（ms） |
| --- | ---: | --- | --- | ---: |
| exhaustive | 79680 | N/A | direct isolated subprocess | 195906.93 |
| adaptive | 43700 | N/A | direct isolated subprocess | 77592.36 |

## 固定验收集

| 数据集 | 原始图片（正/反） | 查询（正/反） | 模板（正/反） | 模板/查询重叠 | 缓存 ID |
| --- | --- | --- | --- | ---: | --- |
| M1 | 95/92 | 20/20 | 28/28 | 0 | `31f082d1a04e486b9345846f4d585033` |
| M2 | 72/58 | 20/20 | 20/20 | 0 | `in-memory:M2:seed-20260825` |
| M7 | 40/45 | 20/20 | 20/20 | 0 | `in-memory:M7:seed-20260825` |

## 每数据集结果

| 数据集 | 模式 | 图片 | 准确率 | 复检 | wall mean/P50/P95（ms） | local_matching mean/P50/P95（ms） | top5/top10/full | 平均匹配数 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| M1 | exhaustive | 40 | 1.0000 | 2 | 1567.10/1573.18/1743.32 | 1405.53/1410.65/1576.97 | 0/0/40 | 56.00 |
| M1 | adaptive | 40 | 1.0000 | 2 | 328.24/234.27/836.64 | 220.80/128.71/724.99 | 29/6/5 | 17.25 |
| M2 | exhaustive | 40 | 1.0000 | 1 | 531.93/515.39/634.78 | 481.57/467.70/582.69 | 0/0/40 | 40.00 |
| M2 | adaptive | 40 | 1.0000 | 1 | 484.49/480.19/562.94 | 436.14/433.53/512.06 | 0/0/40 | 40.00 |
| M7 | exhaustive | 40 | 1.0000 | 0 | 530.44/529.67/611.24 | 480.14/481.79/558.32 | 0/0/40 | 40.00 |
| M7 | adaptive | 40 | 1.0000 | 0 | 560.41/532.84/728.82 | 509.68/483.96/673.82 | 0/0/40 | 40.00 |

## 总体性能

- 查询 wall time 加速比（exhaustive/adaptive）：1.915x
- 含模型加载、建库与 warmup 的进程总时长加速比：2.525x
- 标签不一致：0 张
- 复检状态不一致：0 张
