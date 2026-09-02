# PP-ShiTuV2 原生 C++ 推理评估报告

日期：2026-09-01<br>
分支：`feature/20260901/ppshitu-cpp-inference`<br>
评估代码提交：`e0383ede781b1682c216535b3e2a58aec4a81329`

## 结论

**DO NOT INTEGRATE**

原生 C++ 评估程序已经完成，且在 M1、M2、M7 数据上通过了特征数值一致性和最终决策回放；当前开发机上批量特征提取的 P95 也明显低于现有 Python 特征路径。根据已确认的门禁，仍不能把它接入生产 Qt/后端：

1. 尚未在甲方的 i5-8250U 工控机上完成真实的完整后端 P95 测量，因此不能宣称 `P95 ≤ 100 ms` 或 `P95 ≤ 25 ms`。
2. 原生程序目前是隔离评估工具，不是生产后端替换；生产路径仍使用现有 Python 实现。

这不是正确性失败，而是目标硬件和生产集成证据尚未具备。当前分支保留为可复现实验分支，后续若要集成，应先在目标机完成同一套全后端验收。

## 评估范围与隔离边界

- 只新增 PP-ShiTuV2 全局特征的原生 C++ 推理评估器；没有修改 Qt 页面、Python 生产预测、PP-ShiTu/ALIKED/LightGlue、几何规则、Ridge 参数、融合阈值或工件库格式。
- 原生评估器使用官方 Paddle Inference CPU AVX/MKL、OpenCV 4.6、MSVC 2019 x64；不在构建阶段联网下载依赖。
- 输入顺序、BGR→RGB、224×224 双线性缩放、归一化、NCHW 排布和 L2 归一化与当前 PaddleClas 路径保持一致。
- 原生程序支持 Unicode 路径、真实 batch、独立 worker predictor、结构化错误码、阶段耗时和输入/模型指纹。
- `scripts/benchmark_batch_inference.py` 仅增加了慢速 CPU 后端启动时的握手超时保护，不改变热推理逻辑。

## 依赖与产物指纹

完整依赖、下载地址、构建版本和 SHA-256 已锁定在 [`native/ppshitu_rec_benchmark/dependencies.json`](../../native/ppshitu_rec_benchmark/dependencies.json)。关键身份如下：

| 项目 | 值 |
| --- | --- |
| Paddle Inference | 3.0.0，CPU AVX/MKL，提交 `6ed5dd3` |
| OpenCV | 4.6.0，Windows x64/vc15 |
| 编译器 | MSVC 19.29，Visual Studio 2019，Release x64 |
| 原生可执行文件 | `ppshitu_rec_benchmark.exe`，165,376 bytes |
| 原生可执行文件 SHA-256 | `BECF4443DAE60B574035EA86903B5F8C45B045C4D4F59EBD2FC2B8B9E6493F90` |
| 原生模型指纹 | `b5c295881006bdc7438a86a358f52289f510fe70c9f2429fc23b6d7a4427ec4a` |

原生运行目录只放置 `inference.pdmodel` 和 `inference.pdiparams`。源模型目录中的 `.pdiparams.info` 不参与推理；两份实际模型文件的大小和 SHA-256 已写入依赖锁。原生目录与 Python 源目录的“目录级”指纹因此不同，但模型文件和输出结果已通过逐行比较验证。

## 正确性结果

### 全量 embedding 一致性

使用当前提交重新构建的原生可执行文件，对三个数据集的全部 PNG 做了同序输入比较。门禁为最小余弦相似度 ≥ 0.9999、最大绝对误差 ≤ 0.001，512 维输出逐行对齐。

| 数据集 | 图片数 | 最小余弦相似度 | 最大绝对误差 | 结果 |
| --- | ---: | ---: | ---: | --- |
| M1 | 187 | 0.9999999999977921 | 3.5390258e-07 | 通过 |
| M2 | 130 | 0.9999999999963569 | 3.9488077e-07 | 通过 |
| M7 | 85 | 0.9999999999944955 | 5.4389238e-07 | 通过 |

原始报告和 `.f32` 文件位于（Git 忽略目录）：

`release_staging/cpp-inference-evaluation/dataset-parity-final/`

### 最终决策回放

为每个数据集选择 40 张查询，使用现有 Python `FastGeometryProcessor` 生成 171 个唯一槽位，再将这些槽位按同一顺序交给原生特征提取器，最后用现有快速决策逻辑回放。比较字段包括最终标签、是否需要复检、全局标签、几何是否应用、正反分数和 margin。

| 数据集 | 查询数 | 标签一致 | 复检结论一致 | 几何状态一致 | 差异数 |
| --- | ---: | --- | --- | --- | ---: |
| M1 | 40 | 是 | 是 | 是 | 0 |
| M2 | 40 | 是 | 是 | 是 | 0 |
| M7 | 40 | 是 | 是 | 是 | 0 |

回放总结果为 `overall_passed=true`，原始结果在：

`release_staging/cpp-inference-evaluation/decision-replay-final-all2/result.json`

## 性能结果

### 五张图片原生特征提取扫频

每项均为 50 次预热、200 次测量，输入固定为同一组 M1 五张图片；时间单位为毫秒。`P95` 是完整的 decode + preprocess + inference + normalize。

| 配置 | P50 | P95 | P99 | 最大 | inference P95 | 峰值工作集 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 worker / batch 1 | 101.326 | 118.205 | 127.001 | 132.941 | 109.886 | 169 MB |
| 1 worker / batch 2 | 94.863 | 111.849 | 120.770 | 129.314 | 102.159 | 210 MB |
| 1 worker / batch 4 | 94.647 | 112.896 | 123.955 | 130.366 | 104.078 | 245 MB |
| 1 worker / batch 5 | 89.640 | 106.691 | 113.447 | 119.594 | 98.968 | 240 MB |
| 2 workers / batch 2 | 89.237 | 108.406 | 129.499 | 171.499 | 161.640 | 304 MB |
| 4 workers / batch 1 | 71.655 | 93.346 | 99.124 | 104.362 | 252.453 | 434 MB |

单张图片 1000 次测量（用于观察单请求而非批量吞吐）：

| CPU 线程 | P50 | P95 | P99 | 最大 | inference P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.089 | 25.164 | 28.681 | 31.732 | 23.269 |
| 2 | 20.097 | 25.448 | 29.033 | 40.392 | 23.799 |
| 4 | 20.097 | 25.272 | 28.271 | 31.773 | 23.340 |

当前开发机上，五张图片的 batch-5 原生特征 P95 为 **106.691 ms**；四 worker/batch-1 的吞吐型 P95 为 **93.346 ms**，但峰值内存和 inference 尾延迟更高。不能把这些数字直接当作 i5-8250U 结果。

原始扫频报告：

`release_staging/cpp-inference-evaluation/native-sweep-final/`

### 与现有 Python 路径的对照与投影

同一开发机、同一五张图片的 Python 特征 P95 为 `318.0389500397723 ms`。现有 CPU 包的完整 `predict_batch` 基线（1 worker、1 thread、batch 5）P95 为 `324.49995001079515 ms`。按照规格中唯一允许的投影公式：

```text
projected_pipeline_p95 = python_pipeline_p95
                       - python_feature_p95
                       + native_feature_p95
                     = 324.49995001079515
                       - 318.0389500397723
                       + 106.691055
                     = 113.15205497102284 ms
```

由此得到：

- 原生特征边界相对 Python 特征 P95 改善约 **66.45%**；
- 投影完整流水线相对现有包 P95 改善约 **65.13%**；
- 两项均超过“至少 15%”的开发机评估门槛。

投影不是生产验收：它没有测量原生代码接入 Qt/后端后的协议、队列、缓存和并发开销，也没有在 i5-8250U 上执行。因此不能据此宣布甲方目标达标。

## 门禁判定

| 门禁 | 结果 | 说明 |
| --- | --- | --- |
| 三数据集 embedding 数值一致性 | 通过 | 全部行满足余弦和绝对误差阈值 |
| M1/M2/M7 最终标签、复检、几何状态一致性 | 通过 | 各 40 张，0 差异 |
| 原生特征 P95 至少改善 15% | 通过（开发机） | batch-5 相对改善 66.45% |
| 投影完整流水线 P95 至少改善 15% | 通过（投影） | 相对改善 65.13% |
| i5-8250U 完整后端 P95 验收 | **未测量** | 规格要求真实目标机数据，不能用投影替代 |
| 生产 Qt/后端原生替换 | **未实施** | 本分支按规格保持生产路径不变 |

因此最终结论必须是 `DO NOT INTEGRATE`。

## 测试与构建记录

- 原生契约测试 + Python 比较器定向测试：`27 passed`。
- 强制原生真实集成测试：`8 passed, 19 deselected`；确认不是全部跳过。
- Qt 5.14.2 测试脚本：11 个测试目标均编译并执行，脚本退出码 `0`。
- 全量 Python 测试最终重跑：`845 passed, 11 skipped, 1 warning`（278.80 s）。首轮曾出现一次 `test_fast_sidecar_hashing_never_blocks_catalog_snapshot_or_prediction` 的 1 秒时序断言波动；该测试单独重跑、所在文件重跑以及第二次全量重跑均通过，未修改生产代码来掩盖该时序差异。
- `scripts/compare_cpp_python_inference.py` 已通过语法检查；依赖锁 JSON 可解析。
- 原生 Release 构建成功，嵌入提交指纹与本报告提交一致。

用于复现原生集成测试的环境变量：

```powershell
$env:WORKPIECE_CPP_BENCHMARK_EXE = "E:\Project\wang\ai_区分正反\.native-build\ppshitu-cpp\Release\ppshitu_rec_benchmark.exe"
$env:WORKPIECE_CPP_MODEL_DIR = "E:\workpiece_native_runtime\models\shitu_rec"
E:\python\anaconda3\envs\shitu\python.exe -m pytest `
  tests\test_cpp_inference_benchmark_contract.py `
  tests\test_compare_cpp_python_inference.py -q -p no:cacheprovider -m integration
```

## 后续集成前置条件

1. 在甲方 i5-8250U 工控机上，使用同一输入协议测量冷启动、单张和五张批量的完整后端 P95/P99，并记录内存峰值。
2. 将原生 predictor 接入一个隔离的后端适配层，先做协议/取消/异常/并发回归，不直接替换现有生产路径。
3. 在目标机重新执行三数据集决策回放和 Qt 冒烟，再由独立报告把结论从 `DO NOT INTEGRATE` 改为 `INTEGRATE`（只有所有门禁均有证据时才允许）。
