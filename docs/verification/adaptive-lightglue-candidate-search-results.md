# Adaptive LightGlue Candidate Search 验收结果

- 逐图标签一致：通过
- 逐图复检状态一致：通过
- 两进程输入指纹一致且完整：通过
- M1/M2/M7 发布门禁：通过
- 正式默认模式：`adaptive`

## 环境与可复现信息

| 项目 | 值 |
| --- | --- |
| 时间戳（UTC） | 2026-08-25T06:36:55.109006+00:00 |
| 平台 | Windows-10-10.0.19045-SP0 |
| Python | 3.10.20 | packaged by conda-forge | (main, Jun 11 2026, 03:28:51) [MSC v.1944 64 bit (AMD64)] |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| GPU 驱动 | 560.94 |
| Paddle / CUDA | 3.2.2 / 11.8 |
| Torch / CUDA | 2.7.1+cu118 / 11.8 |
| NumPy | 1.24.4 |
| Git 提交 | `d3fe9fb20c8562d879b5b7d19a1ab7408ee95b24` |
| 工作树 | dirty |
| tracked binary diff SHA-256 | `7e5d30a2886feb1a8cc1e2cb21e014a88ca320d575469f49e30546b33f5f6796` |
| 模型 ID | `general_PPLCNetV2_base_pretrained_v1.0_infer` |
| M1 工件库 ID | `31f082d1a04e486b9345846f4d585033` |
| 每数据集 warmup | 5 |

### 输入指纹摘要

| 模式 | overall SHA-256 | project sources SHA-256 | model SHA-256 |
| --- | --- | --- | --- |
| exhaustive | `b7644694f99b1739180830a1ab9e18aa391018f9b0608e68c0113b1d3af83d5d` | `f4e3ab32df2b9d7e16d260376b7acd70d90212faf4dabc3f854ff709e229dc59` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |
| adaptive | `b7644694f99b1739180830a1ab9e18aa391018f9b0608e68c0113b1d3af83d5d` | `f4e3ab32df2b9d7e16d260376b7acd70d90212faf4dabc3f854ff709e229dc59` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |

### 隔离进程

| 模式 | PID | 端口 | 传输 | 总耗时（ms） |
| --- | ---: | --- | --- | ---: |
| exhaustive | 17172 | N/A | direct isolated subprocess | 96319.81 |
| adaptive | 77828 | N/A | direct isolated subprocess | 75572.54 |

## 固定验收集

| 数据集 | 原始图片（正/反） | 查询（正/反） | 模板（正/反） | 模板/查询重叠 | 缓存 ID |
| --- | --- | --- | --- | ---: | --- |
| M1 | 95/92 | 20/20 | 28/28 | 0 | `31f082d1a04e486b9345846f4d585033` |
| M2 | 72/58 | 20/20 | 20/20 | 0 | `in-memory:M2:seed-20260825` |
| M7 | 40/45 | 20/20 | 20/20 | 0 | `in-memory:M7:seed-20260825` |

## 每数据集结果

| 数据集 | 模式 | 图片 | 准确率 | 复检 | wall mean/P50/P95（ms） | local_matching mean/P50/P95（ms） | top5/top10/full | 平均匹配数 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| M1 | exhaustive | 40 | 1.0000 | 2 | 823.87/819.05/905.60 | 708.55/705.15/791.48 | 0/0/40 | 56.00 |
| M1 | adaptive | 40 | 1.0000 | 2 | 324.74/241.42/835.67 | 217.49/132.43/722.99 | 29/6/5 | 17.25 |
| M2 | exhaustive | 40 | 1.0000 | 1 | 459.75/457.30/537.44 | 412.39/411.12/488.57 | 0/0/40 | 40.00 |
| M2 | adaptive | 40 | 1.0000 | 1 | 468.04/476.55/541.63 | 420.26/429.48/491.33 | 0/0/40 | 40.00 |
| M7 | exhaustive | 40 | 1.0000 | 0 | 490.01/488.51/559.30 | 442.33/442.06/509.65 | 0/0/40 | 40.00 |
| M7 | adaptive | 40 | 1.0000 | 0 | 504.78/505.03/608.15 | 457.15/458.41/559.10 | 0/0/40 | 40.00 |

## 总体性能

- 查询 wall time 加速比（exhaustive/adaptive）：1.367x
- 含模型加载、建库与 warmup 的进程总时长加速比：1.275x
- 标签不一致：0 张
- 复检状态不一致：0 张
