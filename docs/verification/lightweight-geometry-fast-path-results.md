# 轻量几何快速路径验证结果

## 结论

`fast_geometry` 在固定的 M1/M2/M7 验收集上通过正式门槛：legacy 与 fast 均为 120/120，新增错误 0，复检率 0%，1,000 次未裁剪计时的总耗时 P95 为 23.599665 ms。P95 相对 25 ms 门槛保留 1.400335 ms 余量。全量 Python、全部 Qt 自动化测试和 Qt 5.14.2 Release 构建均通过，因此示例配置已切换为 `fast_geometry`。

本报告不包含人工 UI 验收结论。原始验收 JSON 仍是本地、未跟踪的证据文件，不进入 Git 提交。

## 版本、硬件与指纹

- 分支：`feature/20260827/lightweight-geometry-fast-path`
- 报告及默认值修改前的源提交：`5bd9f374cff73160dcb01fe26abb8ced29fb0c67`
- 验收时间：`2026-08-28T08:06:57.938002+08:00`
- 操作系统：Windows 10 `10.0.19045`
- CPU：13th Gen Intel Core i7-13700KF，16 核 / 24 逻辑处理器
- 内存：34,163,982,336 bytes（约 31.8 GiB）
- GPU：NVIDIA GeForce RTX 4060 Ti；驱动 `560.94`
- Python：3.10.20；NumPy 1.24.4
- PaddlePaddle：3.2.2；Paddle CUDA Runtime 11.8
- Torch：2.7.1+cu118；本次 fast 子进程没有加载可用 Torch CUDA 运行时
- PP-ShiTu 模型目录：`E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer`
- 模型文件集合 SHA-256：`7cb248bbe84b4eadbcbe2bdf92a38c1985bd1bec81d6a8945df84263bd6cc8d4`
- `OrientationClassifier` 模型指纹：`1fab156fb025705a836ad6c28590fc32e1141ae5c530282369412a3078300b33`
- 被验收源文件集合 SHA-256：`63f7c1cb4def4dc5f855589db89fe84d32e6941a66c00238abe00183001d86b2`
- 输入、模型和源文件总指纹：`4fc94b7c3fc2d22f300328270e15c82cf1904bee5987c6fc0640a255c7b5599a`

M1 持久化工件 ID 为 `31f082d1a04e486b9345846f4d585033`，关键工件库指纹如下：

| 项目 | 修订 | SHA-256 |
|---|---:|---|
| `manifest.json` | 库修订 2 | `357cca653f985fead461ab1730d9a76695bd456a27ceb765ecd42a4e6471cca4` |
| `.template_cache.pkl` | 库修订 2 | `09c7e1631632093977fe93859f6b03e93470d53879b8477ef467e687284eb373` |
| `geometry_masks/revisions/13.json` | 几何修订 13 | `d86c6e285cd079d061c8b14d8ecbffd6e383d4a759cbb4ba8dceb4e99ccdf924` |

## 正式验收命令与原始证据

正式命令为：

```powershell
E:\python\anaconda3\envs\shitu\python.exe scripts\benchmark_fast_geometry_inference.py `
  --project-root E:\Project\wang\pp_813 `
  --model-dir E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer `
  --library-dir E:\Project\wang\pp_813\runtime_library `
  --m1-workpiece-id 31f082d1a04e486b9345846f4d585033 `
  --warmup 50 `
  --minimum-measured-samples 1000 `
  --max-p95-ms 25 `
  --max-added-errors 0 `
  --max-review-rate 0.05 `
  --output runtime_reports\fast-geometry-acceptance.json
```

- 原始证据绝对路径：`E:\Project\wang\pp_813\runtime_reports\fast-geometry-acceptance.json`
- 原始证据 SHA-256：`211c38482184b3640cd16a12bf93d37ae45ac97ca8dae56907fae17dff8599df`
- 结果：`passed=true`
- 原始 JSON 是有意保留的本地未跟踪文件，不属于本次提交。

组合 JSON 没有顶层 `warmup_predictions` 字段；其 `settings.warmup=50`，且基准脚本在测量循环前执行 50 次 warmup，因此 warmup 没有进入下述 1,000 个延迟样本。

## 数据选择与 SHA 隔离审计

| 工件 | 数据集正/反总数 | 模板正/反 | 查询正/反 | 唯一模板 SHA | 唯一查询 SHA | 模板/查询 SHA 重叠 | 选择指纹 |
|---|---:|---:|---:|---:|---:|---:|---|
| M1 | 95 / 92 | 28 / 28 | 20 / 20 | 56 | 40 | 0 | `b20de5ad71a7700699de2f615a9edbace11f884708271e67341b742dcbc8e17b` |
| M2 | 72 / 58 | 20 / 20 | 20 / 20 | 40 | 40 | 0 | `a7ec2d5ee19f002a0d2e5b982cab09b0275550d158db00187ae8856fe9eef847` |
| M7 | 40 / 45 | 20 / 20 | 20 / 20 | 40 | 40 | 0 | `9d2905f7f0d46d4894868c0ce8a6e27f19e143d1417a485fa159d250ddd1650a` |
| 全局 | — | 136 | 120 | 136 | 120 | 0 | — |

M2/M7 使用现有确定性选择辅助函数实际产生的 20+20 模板；旧计划中“5+5”的一句描述已经过时。所有 120 张唯一查询只参与一次准确率统计；延迟测量按固定顺序循环查询，直到达到 1,000 次。

## 准确率、变化与复检

| 范围 | 查询数 | Legacy 正确 | Fast 正确 | 预测变化 | 新增错误 |
|---|---:|---:|---:|---:|---:|
| M1 | 40 | 40/40（100%） | 40/40（100%） | 0 | 0 |
| M2 | 40 | 40/40（100%） | 40/40（100%） | 0 | 0 |
| M7 | 40 | 40/40（100%） | 40/40（100%） | 0 | 0 |
| 总计 | 120 | 120/120（100%） | 120/120（100%） | 0 | 0 |

- 变化预测的绝对图片路径：无；原始数组为空。
- 新增错误的绝对图片路径：无；原始数组为空。
- 需要复检：0/120，复检率 0%；复检原因计数为空，因此也没有复检图片路径。

可审计分组结果如下；同一查询可同时属于多个分组：

| 分组 | 查询数 | Legacy | Fast | 变化 | 新增错误 |
|---|---:|---:|---:|---:|---:|
| `interference:front:inside:中心反光` | 40 | 40/40 | 40/40 | 0 | 0 |
| `interference:back:inside:中心反光` | 40 | 40/40 | 40/40 | 0 | 0 |
| `crop:<=256` | 80 | 80/80 | 80/80 | 0 | 0 |
| `crop:257-512` | 40 | 40/40 | 40/40 | 0 | 0 |
| `offset:0-0.05` | 40 | 40/40 | 40/40 | 0 | 0 |
| `rotation:0-15` | 40 | 40/40 | 40/40 | 0 | 0 |

## 延迟

所有统计使用完整的 1,000 个测量样本，不裁剪慢样本，单位均为毫秒。

| 阶段 | 样本 | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| decode | 1000 | 0.670919 | 0.462550 | 1.190020 | 1.433094 | 1.779600 |
| geometry_context | 1000 | 0.385968 | 0.000000 | 1.203985 | 1.710524 | 3.400000 |
| geometry_fit | 1000 | 1.016599 | 0.000000 | 3.222055 | 4.015773 | 5.539600 |
| mask_build | 1000 | 3.543913 | 0.000000 | 10.410690 | 13.370183 | 24.566200 |
| global_batch | 1000 | 4.993400 | 4.706400 | 6.735755 | 8.925023 | 10.747700 |
| linear_head | 1000 | 0.017961 | 0.016500 | 0.029010 | 0.044406 | 0.116800 |
| **total** | **1000** | **11.712680** | **6.081450** | **23.599665** | **29.471453** | **36.300200** |

总耗时 P95 比 25 ms 门槛低 1.400335 ms，满足约定的 `P95 <= 25 ms`。P99 29.471453 ms 和 Max 36.300200 ms 均超过 25 ms；本项目 SLA 是 P95，因此不能把 P99/Max 描述为达标。

## 几何规则实际生效证据

固定集合中只有 M1 存在已发布且启用的几何规则。两侧规则均产生非空 mask，且 raw 与 masked 的嵌入距离大于 `1e-6`：

| 工件/方向 | 状态 | Mask pixels | Mask ratio | 嵌入距离 | 查询身份 | 证据图片绝对路径 |
|---|---|---:|---:|---:|---|---|
| M1/front | active | 62,087 | 0.4790663580 | 0.9991443274 | `2be63e8e0b24e3eedbe8a3b01d4667f9c2180abe8010cb33b06d82088988f930` | `E:\Project\wang\pp_813\data\1_M1\0\2260_1411_0_10_2026_05_25_07_33_26_7933.png` |
| M1/back | active | 62,632 | 0.4832716049 | 1.0097709542 | `2be63e8e0b24e3eedbe8a3b01d4667f9c2180abe8010cb33b06d82088988f930` | `E:\Project\wang\pp_813\data\1_M1\0\2260_1411_0_10_2026_05_25_07_33_26_7933.png` |

不能据此宣称 M2/M7 的几何规则鲁棒性已经验证，因为固定集合中的 M2/M7 没有已发布启用规则。

## 本地特征隔离证据

正式基准在两个独立子进程中运行：legacy PID `24664`，fast PID `28676`。fast 子进程记录的在线调用计数为：

| ALIKED | LightGlue | ORB |
|---:|---:|---:|
| 0 | 0 | 0 |

这些计数由 fast worker 守卫结构化记录在 `workers.fast_geometry.local_feature_call_counts`，并不位于 `gates.results` 中；不能因为门槛表没有这些字段而忽略该证据。

## 真实 Fast Cache 建库矩阵

四组数据均在同一个已加载的生产 PP-ShiTu 模型/GPU 实例上，使用 `data\1_M1\0` 和 `data\1_M1\1` 中按文件名确定性选择、内容互异的图片，以及 M1 几何配置修订 13。正式计时之前执行了一次单图模型 warmup。

计时范围严格为 `FastOrientationEngine.build_cache`：包含图片解码、模板几何、已配置旋转、PP-ShiTu 特征批次和 Ridge 拟合；排除进程/模型启动、Qt、工件库复制和磁盘持久化。模型加载 3,163.043 ms 和单图 warmup 192.073 ms 均被排除。缓存大小通过最高 pickle protocol 在内存中序列化得到，未写入任何工件库。因此下表不是端到端 Qt 建库耗时。

| 正+反 | 实际源图正/反 | 增强正/反/总 | 训练行正/反/总 | build_ms | 独立 wall_ms | 验证状态 | 正则化 | 复检阈值 | 几何复检正/反 | Cache bytes |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1+1 | 1 / 1 | 11 / 11 / 22 | 12 / 12 / 24 | 202.732 | 205.216 | few_shot_unverified | 1.0 | 0.027022133 | 0 / 0 | 8,520 |
| 5+10 | 5 / 10 | 55 / 110 / 165 | 60 / 120 / 180 | 2,420.843 | 2,433.926 | validated | 0.001 | 0.039204065 | 0 / 0 | 10,535 |
| 10+15 | 10 / 15 | 110 / 165 / 275 | 120 / 180 / 300 | 5,258.186 | 5,278.076 | validated | 0.001 | 0.039111430 | 0 / 0 | 12,093 |
| 35+35 | 35 / 35 | 0 / 0 / 0 | 35 / 35 / 70 | 2,785.459 | 2,793.538 | validated | 0.0001 | 0.027791023 | 0 / 0 | 19,220 |

内容去重审计：

| 正+反 | 已选/唯一 | 重复 | 正反重叠 | 正面集合 SHA-256 | 反面集合 SHA-256 | 合并集合 SHA-256 |
|---|---:|---:|---:|---|---|---|
| 1+1 | 2 / 2 | 0 | 0 | `11e1907acb986d000066873bd415aeeeae031532ecfa50222d5ed6f84e4418c1` | `91a817bfac3b59963d521a169829c5aa78266503304e9c2ce4e8074647207b94` | `6ec7b220fef7613b920997f38d053cde35169d12a03d593f526839458df3bd59` |
| 5+10 | 15 / 15 | 0 | 0 | `b5dc6fa966dfaf06fa1ed961ccd31cd18193f4f9b0a50e4af370069b78db8b61` | `a15140f636be3a14110af8f57397bf6f523f618bdd249c8cf19bf94cd034a5f3` | `b7eea5485cca76a4936f9a55bef142de84eb343ced129d6c4b4d727193dbdc78` |
| 10+15 | 25 / 25 | 0 | 0 | `9edf36f87a3231ab8d8f9a9442074c99ccc7057445509b25868d3e67911039e7` | `0a10d9dcd506e4f76ba85f9e9548539a8bd7edfbed471f1c05ac8672a2e1faba` | `9c92a9595ac92806a8ae7a46333b1aa9032f54a15faa4354d7baff984f4621a2` |
| 35+35 | 70 / 70 | 0 | 0 | `34fe48464997f297faf709bdb1870f70b0eab3ca43e0ee2757541fa0e6c57fb3` | `ae2f69d7cf6ac81b6ddde0b93ed7be020d4769727c9fa0a19f866c9d8693d200` | `c88d039493f5af36b7de3d28d2aa9f7cb3a2883c171143f46f9fbc8ada20c2dc` |

少于每侧 20 张时会生成旋转增强，因此 10+15 的训练行数和建库时间反而高于 35+35。模板数量主要影响离线缓存构建；缓存建成后，在线查询固定为 raw/front-masked/back-masked 三路嵌入和一个固定维度 Ridge 头，不随模板张数线性增加。

## 自动化回归与 Release

### Python

```text
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q -p no:cacheprovider
589 passed, 3 skipped in 34.20s
```

三个 skip 是 `tests/test_orientation_service_integration.py` 的 M1/M2/M7 参数用例；按文档要求，未设置 `WORKPIECE_ORIENTATION_RUN_INTEGRATION=1` 时跳过生产服务集成测试。正式生产模型验收由上面的独立 benchmark 完成。

### Qt 全目标

`powershell -ExecutionPolicy Bypass -File scripts\run_qt5_tests.ps1` 返回 0。逐目标 QTest 结果如下：

| 目标 | Passed | Failed | Skipped | QTest 时间 |
|---|---:|---:|---:|---:|
| test_appconfig | 12 | 0 | 0 | 34 ms |
| test_backendclient | 14 | 0 | 0 | 2,295 ms |
| test_backendprocessmanager | 11 | 0 | 0 | 7,333 ms |
| test_annotationmanager | 10 | 0 | 0 | 79 ms |
| test_appfoundation | 16 | 0 | 0 | 21 ms |
| test_inspectionpage | 39 | 0 | 0 | 294 ms |
| test_workpiecelibrarypage | 40 | 0 | 0 | 504 ms |
| test_geometryrulecanvas | 18 | 0 | 0 | 12 ms |
| test_geometryrulespage | 78 | 0 | 0 | 1,120 ms |
| test_mainwindow | 116 | 0 | 0 | 21,464 ms |
| **总计** | **354** | **0** | **0** | **33,156 ms** |

示例默认值修改后的聚焦最终树复验：test_appconfig 12、test_backendprocessmanager 11、test_inspectionpage 39，共 62 passed、0 failed、0 skipped。

### Qt 5.14.2 Release

- 命令：`powershell -ExecutionPolicy Bypass -File scripts\build_qt5.ps1`
- 结果：成功
- 可执行文件：`E:\Project\wang\pp_813\qt_app\build-release\release\workpiece_orientation.exe`
- 大小：776,192 bytes
- 文件时间：`2026-08-28 04:16:52 +08:00`（本次增量构建没有需要重新链接的 C++ 源，因此保留已有链接时间）
- 修改示例默认值后已再次运行 Release 构建。

## 默认启用、操作与回滚

`qt_app/app_config.json.example` 现在包含：

```json
"inference_mode": "fast_geometry"
```

用户现有的 `qt_app/app_config.json` 没有被覆盖。构建脚本会优先复制该本地文件，因此已有部署若仍为 `legacy` 或未写该字段，操作员需要在本地配置中明确设置上述一行，并重启后端后才会使用快速路径。

回滚方式：把本地配置改为 `"inference_mode": "legacy"`，然后重启后端。legacy 路径继续保留且未被删除。

## 剩余限制

- 延迟结论只适用于上述硬件、模型、代码指纹和固定选择；更换 GPU、驱动、Paddle 或输入分布后必须复测。
- P99 和 Max 仍超过 25 ms，虽然不违反当前 P95 SLA，但代表仍有尾延迟优化空间。
- 1+1 可以正常建库，但验证状态为 `few_shot_unverified`；它不是“充分验证”的模板规模。
- M2/M7 在固定集合上验证了 fast 分类准确率，但没有已发布几何规则，不能外推为其几何干扰规则已经验证。
- 本报告只记录自动化 Qt 测试和构建，不声称已完成人工界面验收。
