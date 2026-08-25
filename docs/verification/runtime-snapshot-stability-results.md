# 运行时快照稳定性实施与验证记录

日期：2026-08-22  
分支：`feature/20260815/workpiece-orientation-desktop-ui`  
基线提交：`03b3642`  
工作区状态：存在本轮开始前已经存在的未提交修改；本轮未执行提交、暂存、重置或覆盖无关修改。

## 结论

本轮实现已完成并通过项目回归、Qt 5.14.2 回归、真实 PP-ShiTuV2 服务冒烟以及 M1/M2/M7 真实数据集集成验证。

核心行为已经改变为：在线识别始终读取一个不可变的已发布工件快照；异步模板追加、几何缓存准备和恢复工作在快照外执行；全部准备成功且基线修订仍匹配时，才原子切换到新修订。后台构建期间在线识别继续使用旧修订，不再等待整个建库过程。

Qt 端同时将“命令业务失败”和“后端传输失败”拆开处理。业务错误不会再把后端误判为不可用；真正断线时保留当前图片、结果、批量列表和对话框草稿，并明确提示请求结果未知。自管后端重启改为异步退出、terminate、kill、单次重启状态机，不再阻塞界面线程。

## 实施范围

生产代码：

- `src/model_execution_gate.py`
- `src/orientation_classifier.py`
- `src/workpiece_catalog.py`
- `src/workpiece_library.py`
- `src/geometry_mask_profiles.py`
- `src/template_evolution.py`
- `src/orientation_tcp_service.py`
- `qt_app/backendclient.h`
- `qt_app/backendclient.cpp`
- `qt_app/backendprocessmanager.h`
- `qt_app/backendprocessmanager.cpp`
- `qt_app/mainwindow.h`
- `qt_app/mainwindow.cpp`

新增或调整的验证：

- `tests/test_model_execution_gate.py`
- `tests/test_orientation_classifier.py`
- `tests/test_workpiece_catalog.py`
- `tests/test_workpiece_library.py`
- `tests/test_geometry_mask_profiles.py`
- `tests/test_template_evolution.py`
- `tests/test_orientation_tcp_service.py`
- `tests/test_orientation_service_integration.py`
- `qt_app/tests/test_backendclient.cpp`
- `qt_app/tests/test_backendprocessmanager.cpp`
- `qt_app/tests/test_mainwindow.cpp`
- `qt_app/tests/test_geometryrulecanvas.cpp`

设计与计划：

- `docs/superpowers/specs/2026-08-22-runtime-snapshot-stability-design.md`
- `docs/superpowers/plans/2026-08-22-runtime-snapshot-stability.md`

## 环境

- Windows，Asia/Shanghai
- Python 3.10.20，64 位
- `paddlepaddle-gpu 3.2.2`
- `opencv-python 4.6.0.66`
- Paddle 设备检查：`gpu:0`
- Qt 5.14.2，64 位 MSVC 套件
- Release 程序：`qt_app/build-release/release/workpiece_orientation.exe`
- Release 程序大小：474112 bytes；生成时间：2026-08-22 20:34:56

## Python 项目回归

命令：

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=pytest-local-runtime-final-green
```

结果：

- 219 passed
- 3 skipped
- 0 failed
- 40.45 s

3 个默认跳过项是需要显式启用生产模型的 M1/M2/M7 集成测试；它们已在下述真实模型验证中单独启用并通过。

从仓库根目录不限定 `tests/` 执行 pytest 会额外收集 `third_party/PaddleClas` 自带的 Paddle Serving 示例，因其独立运行依赖和包路径而产生 3 个收集错误。正式项目测试范围是本仓库的 `tests/`，该范围已经全部通过。

## Qt 5.14.2 回归与 Release 构建

在 VS2019/Qt 5.14.2 命令环境中，对各测试 `.pro` 重新运行 qmake 和 nmake，再以 `-platform offscreen -o -,txt` 执行测试程序。结果：

| 测试目标 | 通过 | 失败 |
|---|---:|---:|
| AppConfig | 5 | 0 |
| BackendClient | 14 | 0 |
| BackendProcessManager | 11 | 0 |
| GeometryRuleCanvas | 11 | 0 |
| GeometryMaskManager | 12 | 0 |
| AnnotationManager | 10 | 0 |
| MainWindow | 12 | 0 |
| 合计 | 75 | 0 |

Release 构建命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_qt5.ps1
```

构建成功，产物为 `qt_app/build-release/release/workpiece_orientation.exe`。

测试日志中仍有两个非阻塞环境警告：测试环境字体目录不存在，以及个别测试用空路径触发 `QFSFileEngine` 警告；均未造成测试失败，也与本轮状态机修改无关。

## 真实模型与兼容性验证

### 服务冒烟

命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_orientation_service.ps1 -StartupTimeoutSeconds 600
```

观察结果：

- 服务能够先监听并返回 `loading`；
- 模型及现有工件库恢复后返回 `ready`；
- `list_workpieces` 成功；
- `shutdown` 成功；
- 退出后端口 37651 无残留监听。

当前包含较大工件库的冷启动在本机约需 6–7 分钟。新握手逻辑会持续识别 `loading` 状态，不再把正常冷启动误判成连接失败；本轮没有修改模型加载内容，因此不声称缩短了模型本身的加载耗时。

### M1/M2/M7 集成

命令等价于：

```powershell
$env:WORKPIECE_ORIENTATION_RUN_INTEGRATION = "1"
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_service_integration.py -q -p no:cacheprovider
```

结果：`3 passed in 154.06s`。

每个数据集使用正面 5 张、反面 5 张建立隔离的临时工件库，再分别预测一张未入库的正面和反面图片；M1、M2、M7 六次方向判断全部正确。测试客户端现在会消费建库进度事件直到收到最终响应，且三个数据集共用一次服务启动，避免把正常进度事件或重复冷启动误报成超时。生产运行时工件库未被修改。

旧版 5+5 库恢复、不等数量模板恢复与预测、准备失败回滚、修订冲突、重复 operation ID 幂等、损坏任务隔离等行为均由项目回归覆盖。

## 前台/后台行为与规模计时

下表使用确定性的假模型和真实 `WorkpieceLibrary` 文件事务测量，目的是隔离并验证快照与锁行为；数值不是 PP-ShiTuV2 的硬件性能基准。在线计时从请求进入 catalog 调度点到结果返回，因此这里的“服务到结果”不包含 TCP 编解码。

命令：

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_workpiece_catalog.py -k "prediction_stays_available_during_sized_background_append" -q -s -p no:cacheprovider --basetemp=pytest-local-runtime-timing-final
```

| 模板规模 | 空闲预测 | 后台追加期间预测 | 调度到结果 | 后台追加总耗时 | 修订 |
|---|---:|---:|---:|---:|---:|
| 1+1 | 0.015 ms | 0.039 ms | 0.039 ms | 24.597 ms | 1 → 2 |
| 5+10 | 0.028 ms | 0.032 ms | 0.032 ms | 93.448 ms | 1 → 2 |
| 10+15 | 0.010 ms | 0.030 ms | 0.030 ms | 159.691 ms | 1 → 2 |
| 35+35 | 0.011 ms | 0.019 ms | 0.019 ms | 268.203 ms | 1 → 2 |

四组测试都在后台构建被人为阻塞、尚未允许提交时完成了在线预测，预测返回修订 1；释放后台构建后才切换到修订 2。测试通过条件是这个先后关系，而不是某个与硬件相关的毫秒阈值。

真实模型下，在线请求不能中断已经开始的一次 GPU 操作；它最多等待当前有界模型步骤完成，然后优先于下一项后台模板步骤执行。模板越多，后台总建库时间仍近似随模板数量增长，但不再把整个增长时间转化为在线不可用时间。

## 差异与约束审计

- `git diff --check` 退出成功；仅报告现有 LF/CRLF 转换提醒，没有空白错误。
- `third_party/PaddleClas/deploy/configs/inference_general.yaml` 相对 HEAD 无差异。
- 现有融合常量保持为：`GLOBAL_MARGIN_THRESHOLD=0.05`、`LOCAL_MIN_SCORE=4.0`、`LOCAL_MIN_MARGIN=0.5`、`LOCAL_OVERRIDE_MARGIN=3.0`。
- 未修改 PP-ShiTu、ALIKED、LightGlue、模型权重、图像预处理和融合阈值。
- 未执行 git add、commit、reset、checkout 或清理用户已有未提交文件。

## 已知边界

- 快照切换解决的是“长建库阻塞在线识别”和“半成品修订可见”问题，不提升几何拟合或分类精度，不能据此声称准确率提高。
- 当前模型执行仍需要串行访问 GPU；在线请求可以越过尚未开始的后台步骤，但不能抢占已经运行的单个步骤。
- 真正传输中断后，客户端无法确定服务端是否已经完成有副作用的命令，所以界面保留现场并显示“结果未知”，应刷新工件库或按 operation ID 安全重试。
- 工作区包含本轮之前的多项未提交修改，源代码未单独提交，以免把用户的既有改动混入一个不可审查的提交。
