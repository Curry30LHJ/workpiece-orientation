# 离线便携版 GPU/CPU 验收报告

验证日期：2026-08-29。分支：`feature/20260828/offline-portable-packages`。

本报告只记录实际生成的 ZIP、实际启动/协议冒烟和 1,000 次请求基准，不把源代码测试或模拟数据当作生产性能。机器可读明细见 [offline-portable-gpu-cpu-results.json](./offline-portable-gpu-cpu-results.json)。

## 可交付文件

| 版本 | 文件 | 大小 | SHA-256 | 内容审计 |
|---|---|---:|---|---|
| GPU | `release_artifacts/WorkpieceOrientation-GPU-x64-1.0.0.zip` | 2,537,515,338 bytes | `02d7e7fe1e958a50e8c6ceb7184559aa0893d8cdcff5655a59e9d1c850be1817` | 通过 |
| CPU | `release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip` | 254,500,024 bytes | `4044820dcabbd1fcc8c517eb7d80b6b6b6a64478b4210640be0536052d51f2db` | 通过 |

两个包均为 Windows x64 免安装、离线 onedir 包，启动入口为包内 `WorkpieceOrientation.exe`。包内只保留空的数据目录骨架：首次运行时工件库、缓存、日志和几何规则由用户创建；当前开发机已有工件库和规则没有被打入 ZIP。解压后审计通过，未发现绝对路径、ZIP 路径穿越、符号链接、CPU 包 CUDA 运行库或应用 Python 源码。

包内运行时版本：Python 3.10、Paddle 3.2.2、PaddleClas 2.6.0、PyInstaller 6.22.2、Qt 5.14.2。两个包都使用 `fast_geometry` 和 `packaged_executable`，启动看门狗为 600 秒、单请求超时为 120 秒；600 秒是冻结 Paddle 首次加载的上限，不代表正常每次启动都要等待十分钟。

## 端到端冒烟

在每个包的副本中执行了加载轮询、5+10 模板建库、正反各一次预测、回收、恢复和优雅退出：

| 版本 | 模板数 | 正面结果 | 反面结果 | 建库响应耗时 | 进程退出 | 临时数据清理 |
|---|---:|---|---|---:|---:|---|
| GPU | 5+10 | front | back | 5,542.21 ms | 0 | 是 |
| CPU | 5+10 | front | back | 36,139.12 ms | 0 | 是 |

冒烟报告分别位于 `release_staging/final-1.0.0-qtfix2/reports/gpu-smoke.json` 和 `cpu-smoke.json`。两版都确认 `template_counts` 为 `{"front": 5, "back": 10}`，没有静默截断。

另外，在本轮重建后的两个 staging 包中直接启动了 Qt 主程序的隐藏打包冒烟入口：`WorkpieceOrientation.exe --package-smoke-test`。GPU 版退出码为 0、耗时 9,856 ms；CPU 版退出码为 0、耗时 5,983 ms（开发机当前系统/模型缓存状态）。该入口现在会保持 Qt 事件循环，直到收到后端就绪结果，不再出现“立即返回 0、实际没有启动后端”的假阳性。验证使用默认 Windows 平台；`qoffscreen.dll` 不属于交付包，也不是甲方用户的启动方式。

## 1,000 次请求基准

基准使用同一份外部 M1/M2/M7 验收集：50 次 warm-up、1,000 次计时请求、120 个唯一留出查询；验收图片没有进入 ZIP。

| 版本 | Ready 启动 | 后端平均 | 后端 P50 | 后端 P95 | 后端 P99 | 后端最大 | 准确率 | 复核率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU | 76,462.16 ms | 11.164 ms | 10.993 ms | **14.530 ms** | 17.155 ms | 21.823 ms | 120/120、100% | 0% |
| CPU | 15,297.45 ms | 176.567 ms | 171.401 ms | 222.440 ms | 258.947 ms | 304.655 ms | 120/120、100% | 0% |

GPU 的正式门槛是后端完整耗时 P95 ≤ 25 ms，本次 14.530 ms 通过；往返 P95 为 15.599 ms。CPU 版不套用 25 ms 门槛，但功能和标签与 GPU 版逐查询一致（标签差异 0、复核差异 0）。两版基准结束时进程退出码均为 0，未触发强制终止。

## 测试结果

- Python 完整套件：`726 passed, 5 skipped, 0 failed`。
- Qt 5.14.2 离屏测试：11/11 目标通过，包括 `test_mainwindow` 和 `test_startupsmokecontroller`。
- 打包/冒烟相关聚焦测试：`61 passed, 0 failed`。
- 打包后 Qt 主程序冒烟：GPU/CPU 2/2 通过，退出码均为 0。
- Qt 测试和 Python 测试均在 ASCII 临时路径下运行；当前执行器直接把中文工作路径传给原生工具时会转码，这是测试环境限制，不是应用路径兼容性结论。

## 模板数量对耗时的含义

已有 fast-cache 实验（详见 [轻量几何快速路径结果](./lightweight-geometry-fast-path-results.md)）显示，模板数量主要影响离线特征缓存构建：1+1、5+10、10+15、35+35 的特征构建耗时分别为 202.732、2,420.843、5,258.186、2,785.459 ms。少于每侧 20 张时会触发旋转增强，所以耗时不是简单单调函数；所有模板仍会被保留和参与建库。缓存完成后，`fast_geometry` 在线查询使用固定维度的三路特征和 Ridge 头，本次 5+10 基准的 GPU P95 为 14.530 ms，CPU P95 为 222.440 ms。

## 尚未完成的验证

本机没有干净的 Windows 10/11 虚拟机，因此尚未声称“无 Python/Conda/Qt/CUDA Toolkit 的干净机”验收。交付前建议在甲方目标机断网执行一次：校验 SHA-256、解压到含中文和空格且路径超过 180 字符的目录、打开 Qt、建库/检测/退出，并确认 NVIDIA 驱动与 GPU 版 Paddle/CUDA 运行库兼容。CPU 包可直接在无 NVIDIA GPU 的纯 CPU 机器上使用，但从本次实测看不满足 25 ms 延迟目标。

本轮包内 Qt 冒烟修复对应源码提交 `d3d1fa0834f11964919124359921c2ebc5db27cc`；该提交也写入两个包的 `version.json`。此前针对 `-platform offscreen` 的探测会因交付包未携带 Qt 的 `qoffscreen.dll` 而失败，这不是默认 Windows 启动路径上的故障，已改用实际交付启动方式复验。

设计与实施依据：[离线便携包设计](../superpowers/specs/2026-08-28-offline-portable-gpu-cpu-packaging-design.md) 和 [实施计划](../superpowers/plans/2026-08-28-offline-portable-gpu-cpu-packages.md)。
