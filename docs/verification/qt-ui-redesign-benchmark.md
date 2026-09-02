# Adaptive LightGlue Candidate Search 验收结果

- 逐图标签一致：通过
- 逐图复检状态一致：通过
- 两进程输入指纹一致且完整：通过
- M1/M2/M7 发布门禁：通过
- 正式默认模式：`adaptive`

## 精确复现命令

```powershell
E:\python\anaconda3\envs\shitu\python.exe E:\Project\wang\pp_813\scripts\benchmark_adaptive_local_search.py --project-root E:\Project\wang\pp_813 --model-dir E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer --library-dir E:\Project\wang\pp_813\runtime_library --m1-workpiece-id 31f082d1a04e486b9345846f4d585033 --warmup 1 --output-json E:\Project\wang\pp_813\docs\verification\qt-ui-redesign-benchmark.json --output-markdown E:\Project\wang\pp_813\docs\verification\qt-ui-redesign-benchmark.md
```

## 环境与可复现信息

| 项目 | 值 |
| --- | --- |
| 时间戳（UTC） | 2026-08-26T14:07:36.801842+00:00 |
| 平台 | Windows-10-10.0.19045-SP0 |
| Python | 3.10.20 | packaged by conda-forge | (main, Jun 11 2026, 03:28:51) [MSC v.1944 64 bit (AMD64)] |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| GPU 驱动 | 560.94 |
| Paddle / CUDA | 3.2.2 / 11.8 |
| Torch / CUDA | 2.7.1+cu118 / 11.8 |
| NumPy | 1.24.4 |
| Git 提交 | `ba09e4e7a85bf6d45ddb4f9d35877b01fab2c850` |
| 工作树 | clean |
| tracked binary diff SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| 模型 ID | `general_PPLCNetV2_base_pretrained_v1.0_infer` |
| M1 工件库 ID | `31f082d1a04e486b9345846f4d585033` |
| 每数据集 warmup | 1 |

### 输入指纹摘要

| 模式 | overall SHA-256 | project sources SHA-256 | model SHA-256 |
| --- | --- | --- | --- |
| exhaustive | `82a4fc90b3d6c057657c8c5aaac36d677e4595408196790ef58c516a2eb2a46d` | `c7d61cb93bdbbf8a553ef68016f20c8930a036b7c22d7a61af95ab595f43275a` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |
| adaptive | `82a4fc90b3d6c057657c8c5aaac36d677e4595408196790ef58c516a2eb2a46d` | `c7d61cb93bdbbf8a553ef68016f20c8930a036b7c22d7a61af95ab595f43275a` | `7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4` |

### 隔离进程

| 模式 | PID | 端口 | 传输 | 总耗时（ms） |
| --- | ---: | --- | --- | ---: |
| exhaustive | 66484 | N/A | direct isolated subprocess | 84179.82 |
| adaptive | 18012 | N/A | direct isolated subprocess | 67900.13 |

## 固定验收集

| 数据集 | 原始图片（正/反） | 查询（正/反） | 模板（正/反） | 模板/查询重叠 | 缓存 ID |
| --- | --- | --- | --- | ---: | --- |
| M1 | 95/92 | 20/20 | 28/28 | 0 | `31f082d1a04e486b9345846f4d585033` |
| M2 | 72/58 | 20/20 | 20/20 | 0 | `in-memory:M2:seed-20260825` |
| M7 | 40/45 | 20/20 | 20/20 | 0 | `in-memory:M7:seed-20260825` |

## 每数据集结果

| 数据集 | 模式 | 图片 | 准确率 | 复检 | wall mean/P50/P95（ms） | local_matching mean/P50/P95（ms） | top5/top10/full | 平均匹配数 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| M1 | exhaustive | 40 | 1.0000 | 2 | 776.12/769.88/839.25 | 665.34/660.24/724.33 | 0/0/40 | 56.00 |
| M1 | adaptive | 40 | 1.0000 | 2 | 319.93/229.95/843.16 | 216.14/127.27/725.62 | 29/6/5 | 17.25 |
| M2 | exhaustive | 40 | 1.0000 | 1 | 440.24/450.44/500.88 | 395.36/408.20/455.77 | 0/0/40 | 40.00 |
| M2 | adaptive | 40 | 1.0000 | 1 | 461.76/458.85/533.33 | 417.06/415.72/489.99 | 0/0/40 | 40.00 |
| M7 | exhaustive | 40 | 1.0000 | 0 | 470.92/470.60/558.19 | 425.68/422.84/510.78 | 0/0/40 | 40.00 |
| M7 | adaptive | 40 | 1.0000 | 0 | 498.93/500.52/560.13 | 452.51/454.04/512.25 | 0/0/40 | 40.00 |

## 总体性能

- 查询 wall time 加速比（exhaustive/adaptive）：1.318x
- 含模型加载、建库与 warmup 的进程总时长加速比：1.240x
- 标签不一致：0 张
- 复检状态不一致：0 张
