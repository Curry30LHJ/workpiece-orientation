# Adaptive LightGlue Candidate Search 验收结果

- 逐图标签一致：通过
- 逐图复检状态一致：通过
- 两进程输入指纹一致且完整：通过
- M1/M2/M7 发布门禁：通过
- 正式默认模式：`adaptive`

## 环境与可复现信息

| 项目 | 值 |
| --- | --- |
| 时间戳（UTC） | 2026-08-25T07:16:03.454344+00:00 |
| 平台 | Windows-10-10.0.19045-SP0 |
| Python | 3.10.20 | packaged by conda-forge | (main, Jun 11 2026, 03:28:51) [MSC v.1944 64 bit (AMD64)] |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| GPU 驱动 | 560.94 |
| Paddle / CUDA | 3.2.2 / 11.8 |
| Torch / CUDA | 2.7.1+cu118 / 11.8 |
| NumPy | 1.24.4 |
| Git 提交 | `1a5f5ba661576e3fafe767c180705a524be03894` |
| 工作树 | dirty |
| tracked binary diff SHA-256 | `7e5d30a2886feb1a8cc1e2cb21e014a88ca320d575469f49e30546b33f5f6796` |
| 模型 ID | `general_PPLCNetV2_base_pretrained_v1.0_infer` |
| M1 工件库 ID | `31f082d1a04e486b9345846f4d585033` |
| 每数据集 warmup | 5 |

### 输入指纹摘要

| 模式 | overall SHA-256 | project sources SHA-256 | model SHA-256 |
| --- | --- | --- | --- |
| exhaustive | `8364fe3fa5d1d5cb5a11fa1f42643225c13a82b702bf5dbcd5e1083ff23e3edf` | `d97aaed124e37b0a1a44a5400f708a975d217ba48e067f36cacb14b928fc0c71` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |
| adaptive | `8364fe3fa5d1d5cb5a11fa1f42643225c13a82b702bf5dbcd5e1083ff23e3edf` | `d97aaed124e37b0a1a44a5400f708a975d217ba48e067f36cacb14b928fc0c71` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |

### 隔离进程

| 模式 | PID | 端口 | 传输 | 总耗时（ms） |
| --- | ---: | --- | --- | ---: |
| exhaustive | 78904 | N/A | direct isolated subprocess | 98169.90 |
| adaptive | 22404 | N/A | direct isolated subprocess | 74870.54 |

## 固定验收集

| 数据集 | 原始图片（正/反） | 查询（正/反） | 模板（正/反） | 模板/查询重叠 | 缓存 ID |
| --- | --- | --- | --- | ---: | --- |
| M1 | 95/92 | 20/20 | 28/28 | 0 | `31f082d1a04e486b9345846f4d585033` |
| M2 | 72/58 | 20/20 | 20/20 | 0 | `in-memory:M2:seed-20260825` |
| M7 | 40/45 | 20/20 | 20/20 | 0 | `in-memory:M7:seed-20260825` |

## 每数据集结果

| 数据集 | 模式 | 图片 | 准确率 | 复检 | wall mean/P50/P95（ms） | local_matching mean/P50/P95（ms） | top5/top10/full | 平均匹配数 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| M1 | exhaustive | 40 | 1.0000 | 2 | 835.08/828.16/910.49 | 719.35/713.47/788.41 | 0/0/40 | 56.00 |
| M1 | adaptive | 40 | 1.0000 | 2 | 321.81/238.25/797.60 | 215.58/131.59/685.22 | 29/6/5 | 17.25 |
| M2 | exhaustive | 40 | 1.0000 | 1 | 471.06/475.12/533.72 | 423.37/429.33/483.76 | 0/0/40 | 40.00 |
| M2 | adaptive | 40 | 1.0000 | 1 | 471.56/473.10/564.65 | 424.07/425.73/516.67 | 0/0/40 | 40.00 |
| M7 | exhaustive | 40 | 1.0000 | 0 | 511.40/508.03/573.54 | 463.14/461.74/524.87 | 0/0/40 | 40.00 |
| M7 | adaptive | 40 | 1.0000 | 0 | 500.61/503.78/554.74 | 452.77/456.65/506.54 | 0/0/40 | 40.00 |

## 总体性能

- 查询 wall time 加速比（exhaustive/adaptive）：1.405x
- 含模型加载、建库与 warmup 的进程总时长加速比：1.311x
- 标签不一致：0 张
- 复检状态不一致：0 张
