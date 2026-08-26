# Adaptive LightGlue Candidate Search 验收结果

- 逐图标签一致：通过
- 逐图复检状态一致：通过
- 两进程输入指纹一致且完整：通过
- M1/M2/M7 发布门禁：通过
- 正式默认模式：`adaptive`

## 环境与可复现信息

| 项目 | 值 |
| --- | --- |
| 时间戳（UTC） | 2026-08-26T12:54:26.433698+00:00 |
| 平台 | Windows-10-10.0.19045-SP0 |
| Python | 3.10.20 | packaged by conda-forge | (main, Jun 11 2026, 03:28:51) [MSC v.1944 64 bit (AMD64)] |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| GPU 驱动 | 560.94 |
| Paddle / CUDA | 3.2.2 / 11.8 |
| Torch / CUDA | 2.7.1+cu118 / 11.8 |
| NumPy | 1.24.4 |
| Git 提交 | `219b4152c4812589f2d26e7c4361260e1325ce2f` |
| 工作树 | dirty |
| tracked binary diff SHA-256 | `bd314ddd66ee6253bdbf0f1318f38134a815dd058fd8d5f17cb93a4482518e0f` |
| 模型 ID | `general_PPLCNetV2_base_pretrained_v1.0_infer` |
| M1 工件库 ID | `31f082d1a04e486b9345846f4d585033` |
| 每数据集 warmup | 1 |

### 输入指纹摘要

| 模式 | overall SHA-256 | project sources SHA-256 | model SHA-256 |
| --- | --- | --- | --- |
| exhaustive | `7bb66782bbea4d01e4ea7403d26c9bb078862da913937e31f0081d6c4c788015` | `f50523621aa141a6c1281ca0120a7840f34156298a48e901332ee1da941091a1` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |
| adaptive | `7bb66782bbea4d01e4ea7403d26c9bb078862da913937e31f0081d6c4c788015` | `f50523621aa141a6c1281ca0120a7840f34156298a48e901332ee1da941091a1` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |

### 隔离进程

| 模式 | PID | 端口 | 传输 | 总耗时（ms） |
| --- | ---: | --- | --- | ---: |
| exhaustive | 51652 | N/A | direct isolated subprocess | 123284.43 |
| adaptive | 59256 | N/A | direct isolated subprocess | 71582.01 |

## 固定验收集

| 数据集 | 原始图片（正/反） | 查询（正/反） | 模板（正/反） | 模板/查询重叠 | 缓存 ID |
| --- | --- | --- | --- | ---: | --- |
| M1 | 95/92 | 20/20 | 28/28 | 0 | `31f082d1a04e486b9345846f4d585033` |
| M2 | 72/58 | 20/20 | 20/20 | 0 | `in-memory:M2:seed-20260825` |
| M7 | 40/45 | 20/20 | 20/20 | 0 | `in-memory:M7:seed-20260825` |

## 每数据集结果

| 数据集 | 模式 | 图片 | 准确率 | 复检 | wall mean/P50/P95（ms） | local_matching mean/P50/P95（ms） | top5/top10/full | 平均匹配数 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| M1 | exhaustive | 40 | 1.0000 | 2 | 853.75/847.96/940.95 | 733.74/731.05/819.02 | 0/0/40 | 56.00 |
| M1 | adaptive | 40 | 1.0000 | 2 | 332.03/239.99/859.90 | 224.09/134.90/743.30 | 29/6/5 | 17.25 |
| M2 | exhaustive | 40 | 1.0000 | 1 | 478.17/476.33/542.54 | 428.59/428.98/493.40 | 0/0/40 | 40.00 |
| M2 | adaptive | 40 | 1.0000 | 1 | 488.73/485.36/587.09 | 438.60/436.11/538.62 | 0/0/40 | 40.00 |
| M7 | exhaustive | 40 | 1.0000 | 0 | 514.12/504.59/578.01 | 462.72/452.64/524.81 | 0/0/40 | 40.00 |
| M7 | adaptive | 40 | 1.0000 | 0 | 517.25/519.90/565.77 | 466.67/468.79/513.26 | 0/0/40 | 40.00 |

## 总体性能

- 查询 wall time 加速比（exhaustive/adaptive）：1.380x
- 含模型加载、建库与 warmup 的进程总时长加速比：1.722x
- 标签不一致：0 张
- 复检状态不一致：0 张
