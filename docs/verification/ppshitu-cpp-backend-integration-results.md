# PP-ShiTuV2 原生 C++ 后端接入验证报告

日期：2026-09-01
分支：`feature/20260901/ppshitu-cpp-inference`
接入提交：`571747d2706351657b2c87bf33c6dfead961b3dc`

## 结论

**已完成可回退的 opt-in 接入，默认仍使用 Python 后端。**

原生 C++ PP-ShiTuV2 特征服务已经接入 Python 分类器、TCP 后端和 Qt 配置链路。将 `pp_backend` 设为 `native_cpp` 后，后端会启动一个常驻的 `ppshitu_rec_service.exe`，模型只加载一次，并通过二进制管道处理批量 RGB 图像；几何规则、缓存身份、工件库格式、融合阈值和 Qt TCP 请求格式均保持不变。未显式选择 `native_cpp` 时，现有 Python 路径完全不变。

本报告中的性能数字来自当前开发机，不是甲方 i5-8250U 的验收结果。目标机仍需现场测量后，才能确认甲方的吞吐/延迟指标。

## 接入边界与配置

支持条件：

- `compute_device=cpu`
- `inference_mode=fast_geometry`
- `pp_backend=native_cpp`
- 必须提供 `native_pp_executable`，且可访问同目录的 Paddle Inference/OpenCV/MKL DLL

示例配置：

```json
{
  "compute_device": "cpu",
  "inference_mode": "fast_geometry",
  "pp_backend": "native_cpp",
  "native_pp_executable": "E:/Project/wang/pp_813/.native-build/ppshitu-cpp-task4/Release/ppshitu_rec_service.exe",
  "model_sha256": "1fab156fb025705a836ad6c28590fc32e1141ae5c530282369412a3078300b33"
}
```

`native_cpp` 不支持 GPU、`legacy` 或 `compare` 模式；配置不满足时后端会报告明确错误，不会静默退回 Python。Qt 会把该配置转发给 TCP 服务，并校验服务 HELLO 中的后端身份、模型指纹和 512 维特征维度。

示例中的可执行文件路径只是当前开发机路径；交付时请改为目标机上的绝对路径，或改为相对于 `app_config.json` 所在目录的有效路径。

源模型目录级 SHA-256 仍是缓存身份，必须按 `model_dir` 实际内容填写。当前仓库目录（包含 `.pdiparams.info`）的指纹是 `1fab156fb025705a836ad6c28590fc32e1141ae5c530282369412a3078300b33`；契约测试使用的 ASCII 两文件副本 `E:/workpiece_native_runtime/models/shitu_rec` 指纹是 `b5c295881006bdc7438a86a358f52289f510fe70c9f2429fc23b6d7a4427ec4a`，两者不能混填。

可用下面的命令计算任意模型目录的值：

```powershell
E:\python\anaconda3\envs\shitu\python.exe -c "from pathlib import Path; from src.model_fingerprint import model_directory_sha256; print(model_directory_sha256(Path(r'E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer')))"
```

因此切换到原生后不需要重建已有 5+5 或不等数量模板库；恢复和预测继续读取原有缓存。`.pdiparams.info` 不参与推理，但会参与目录级指纹计算；原生服务的 ASCII staging 副本只用于 C++ 加载和完整性诊断，不改变缓存键。

## 实现内容

| 层 | 主要变更 |
| --- | --- |
| C++ 协议/服务 | `native/ppshitu_rec_benchmark` 增加固定小端二进制协议、共享 RGB 预处理、常驻 predictor 服务、结构化错误和阶段耗时；benchmark 与 service 共用同一核心。 |
| Python 客户端 | `src/native_pp_protocol.py`、`src/native_pp_client.py` 实现握手、批量顺序保持、超时、子进程退出检测、stderr 诊断和清理。 |
| 分类器 | `src/orientation_classifier.py` 增加显式 backend 选择；批量 worker 使用独立 native client；关闭、异常和维度变化均显式失败。 |
| TCP 后端 | `src/orientation_tcp_service.py` 增加 native 参数校验、HELLO 元数据和启动错误码，并在关闭时回收 C++ 子进程。 |
| Qt | `qt_app/appconfig.*`、`backendprocessmanager.*`、`appheader.*`、`mainwindow.cpp` 增加配置转发、身份校验和诊断显示；旧 Python HELLO 仍兼容。 |
| 回归测试 | 新增 `tests/test_native_backend_regression.py`，覆盖 1+1、5+10、10+15、35+35、旧缓存恢复、无静默回退和子进程清理。 |

未修改 PP-ShiTu 权重、ALIKED、LightGlue、几何规则、Ridge 参数、现有融合阈值或工件库/模板缓存格式。

## 正确性与兼容性证据

| 验证项 | 结果 |
| --- | --- |
| native backend 回归（不依赖真实 Paddle） | `6 passed` |
| native 服务真实进程契约（HELLO、两图有序 RESULT、错误帧） | `3 passed` |
| C++ 协议/预处理/benchmark 契约 | `11 passed, 3 deselected` |
| Python 定向集成组合（native classifier、TCP、fast path） | `302 passed` |
| Qt 5.14.2 全量脚本 | 11 个目标编译并执行，脚本退出码 `0` |
| 旧缓存和不等模板数 | 回归测试通过；缓存身份未改变 |

native 服务真实测试使用：

```text
服务：.native-build/ppshitu-cpp-task4/Release/ppshitu_rec_service.exe
模型：E:/workpiece_native_runtime/models/shitu_rec
特征维度：512
服务版本：ppshitu-native-cpp/1
```

完整 Python 回归首次运行得到 `877 passed, 19 skipped, 2 failed`；失败均为既有 `test_fast_cache_jobs.py` 的 1 秒时序断言及其后 Windows 临时目录占用连带失败。该文件单独重跑为 `35 passed`，时序用例连续三次均通过，未修改生产代码掩盖该波动。

## 性能测量

### 测试环境

- Windows 10 build 19045
- 当前开发机：Intel Core i7-13700KF（16 物理核 / 24 逻辑核）
- 模型：512 维 PP-ShiTuV2，输入为同一组 M1 图像
- 原生 benchmark：50 次预热、200 次测量（95 图吞吐另用 20 次预热、50 次测量）

### 原生特征路径（开发机）

| 配置 | 完整特征提取 P50 (ms) | P95 (ms) | P99 (ms) | 峰值内存 |
| --- | ---: | ---: | ---: | ---: |
| 1 worker / batch 1 / 1 thread | 67.74 | 84.75 | 120.45 | 168 MiB |
| 1 worker / batch 5 / 1 thread | 42.06 | 51.52 | 56.37 | 234 MiB |
| 1 worker / batch 5 / 2 threads | 38.31 | 48.37 | 50.77 | 235 MiB |
| 2 workers / batch 2 / 2 threads | 53.24 | 67.94 | 71.79 | 356 MiB |
| 4 workers / batch 1 / 1 thread | 68.98 | 82.65 | 93.85 | 442 MiB |

常驻 Python 客户端对 5 张图的端到端往返（含 IPC、服务预处理、推理和返回）为：P50 `58.60 ms`、P95 `66.97 ms`、P99 `72.20 ms`。同一开发机、同一输入的现有 Python 特征路径为 P50 `506.32 ms`、P95 `544.22 ms`，端到端总耗时 P50 `515.41 ms`、P95 `553.96 ms`。这说明当前接入主要消除了 Python predictor 的重复调度/解释器开销；并行 worker 并非默认开启，需按目标机内存和实测结果选择。

95 张图的原生批量总耗时（不是单图 P95）为：batch 32 单 worker P50 `882.49 ms`、P95 `956.75 ms`；4 worker/batch 1 P50 `714.47 ms`、P95 `843.39 ms`。报告文件保存在 Git 忽略目录：

`release_staging/cpp-backend-evaluation/current/`

上述数据只用于开发机趋势判断，不能直接承诺 i5-8250U 的 `P95 ≤ 25 ms` 或“五图 <100 ms”。i5-8250U 需要在现场用同一模型、同一 DLL、同一输入集和同一测量脚本复测；同时应记录冷启动、单图、5 图批量、峰值内存和长时间稳定性。

## 产物与依赖指纹

完整依赖记录见 [`native/ppshitu_rec_benchmark/dependencies.json`](../../native/ppshitu_rec_benchmark/dependencies.json)。当前 Release 产物：

| 产物 | 大小 | SHA-256 |
| --- | ---: | --- |
| `ppshitu_rec_service.exe` | 132,096 bytes | `22b7c3b577b83b8ab9da1edcfd09da2aa44e598fdf70276775f6f3721fc8befd` |
| `ppshitu_rec_benchmark.exe` | 166,400 bytes | `5d1f7b67073050c7e6de5ed8633cb524b4103e05759b983be5d15b7578aebfdf` |
| `paddle_inference.dll` | 121,158,656 bytes | `4b1c820e7f46eb02771c020c471ae84164bf316e7b60bee7a3fbfa4a63771d4b` |
| `opencv_world460.dll` | 64,350,720 bytes | `012b3b79f380a5d813b78d836bf9f378b6f79f514b413996a7c2af8b800cb122` |
| `mkldnn.dll` | 47,322,112 bytes | `a25df93583067d58b84f7b7a4d645d1c6881770d01879129455ef893e4c2673d` |
| `mklml.dll` | 92,649,344 bytes | `e2a7fd93e1534568626cffe22676b15164a5a2b3a62f1919a55993478b45811e` |

模型两文件的大小和 SHA-256 已在依赖清单中锁定。上述依赖下载归档的来源/校验值沿用此前已记录的锁文件，当前工作区不重复下载归档。

## 回退与上线建议

1. 先在目标机把 `pp_backend` 保持为 `python` 完成基线记录。
2. 复制 native service、依赖 DLL 和 ASCII 模型副本，修改为 `native_cpp`，重启后确认 Qt 详情中的 backend、模型指纹和特征维度。
3. 用少量 1+1、5+10 和旧 5+5 工件库做结果对照，再进行 95 图吞吐测试。
4. 若 native 服务启动、握手、维度或性能不满足要求，将 `pp_backend` 改回 `python` 并重启；不需要删除缓存或重建工件库。native 模式不会自动降级，便于发现部署问题。

当前建议结论为 **在目标机验收前以 opt-in 方式试运行，不替换默认 Python 后端**。只有目标机完成同协议、同输入集的 P95/P99 和稳定性记录后，才决定是否把 native_cpp 设为交付配置。
