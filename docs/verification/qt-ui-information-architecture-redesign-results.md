# Qt UI 信息架构重构验证报告

## 交付范围

- 基线：`219b4152c4812589f2d26e7c4361260e1325ce2f`
- 交付范围：`219b4152..HEAD`
- 生产代码与 benchmark 对应提交：`ba09e4e7a85bf6d45ddb4f9d35877b01fab2c850`（benchmark 启动时工作树为 clean）
- 最终证据提交：`HEAD`（本报告所在提交；只增加/更新验证报告、benchmark 结果和固定截图，不再修改生产代码）
- Qt：5.14.2 Widgets / MSVC x64
- Python：3.10.20（`shitu` 环境）
- 后端识别模型、PP-ShiTu、ALIKED、LightGlue、融合阈值及协议均未修改。

本任务完成了统一主题、高 DPI、语义图标和状态、三页最终布局、检测模式切换、路径缩略与完整 tooltip、键盘焦点顺序、关闭保护、旧标注 UI 入口清理以及最终验证证据。为修正最终截图暴露的真实布局问题，生产文件范围必须额外包含 `inspectionpage.cpp/.h`、`workpiecelibrarypage.cpp/.h` 和 `inspectionimageview.cpp/.h`；相关测试 `.pro` 也必须链接生产主题与资源。这些修改只作用于 UI/视图状态，不改变识别语义。

## 修改文件

### 生产代码与资源

- `qt_app/main.cpp`
- `qt_app/apptheme.cpp`
- `qt_app/appheader.cpp`
- `qt_app/taskstatuswidget.cpp`, `qt_app/taskstatuswidget.h`
- `qt_app/mainwindow.cpp`, `qt_app/mainwindow.h`, `qt_app/mainwindow.ui`
- `qt_app/inspectionpage.cpp`, `qt_app/inspectionpage.h`, `qt_app/inspectionpage.ui`
- `qt_app/inspectionimageview.cpp`, `qt_app/inspectionimageview.h`
- `qt_app/workpiecelibrarypage.cpp`, `qt_app/workpiecelibrarypage.h`, `qt_app/workpiecelibrarypage.ui`
- `qt_app/geometryrulespage.cpp`
- `qt_app/resources.qrc`, `qt_app/resources/theme.qss`
- `qt_app/resources/icons/nav-inspection.svg`
- `qt_app/resources/icons/nav-library.svg`
- `qt_app/resources/icons/nav-geometry.svg`
- `qt_app/resources/icons/status-success.svg`
- `qt_app/resources/icons/status-warning.svg`
- `qt_app/resources/icons/status-error.svg`

### 测试与证据

- `qt_app/tests/test_appfoundation.cpp`
- `qt_app/tests/test_inspectionpage.cpp`, `qt_app/tests/test_inspectionpage.pro`
- `qt_app/tests/test_workpiecelibrarypage.cpp`, `qt_app/tests/test_workpiecelibrarypage.pro`
- `qt_app/tests/test_geometryrulespage.cpp`, `qt_app/tests/test_geometryrulespage.pro`
- `qt_app/tests/test_mainwindow.cpp`, `qt_app/tests/test_mainwindow.pro`
- `scripts/benchmark_adaptive_local_search.py`
- `tests/test_benchmark_adaptive_local_search.py`
- `docs/verification/qt-ui-redesign-benchmark.json`
- `docs/verification/qt-ui-redesign-benchmark.md`
- `docs/verification/screenshots/qt-ui-inspection.png`
- `docs/verification/screenshots/qt-ui-library.png`
- `docs/verification/screenshots/qt-ui-geometry.png`

保留了 `annotationmanager` 的源文件、协议、模型和独立测试；只删除 MainWindow/工件库中的过时静态入口与旧对话框适配路径。

## TDD 结果

RED 阶段覆盖了：缺失语义图标、空警告文本、不可见模式切换、运行批量时仍可切换、长路径丢失、主按钮重复、焦点顺序、35+35 模板布局、几何页 splitter/高级区、旧标注入口、窗口最小尺寸、关闭确认以及 dirty-save-active 连续保护。最终截图又复现了模式栏占用过多纵向空间和图片首次布局后仅约 25×25 像素；独立复审进一步复现了反面预测后键盘焦点仍按正面顺序、DPI 截图误用固定逻辑尺寸，以及 125%/150% 下三个真实水平滚动问题。

GREEN 阶段增加了条件驱动的回归断言，并修复上述行为：预测按钮后的 Tab 顺序随正反面视觉顺序同步；截图客户区按 DPR 反算；每张图均断言物理 1920×1080、主操作完整位于窗口内且页面无水平滚动条；benchmark 报告写入精确可复制命令。主窗口全套测试发现的两处既有异步测试竞态也改为等待真实页面完成条件，不使用固定延时掩盖结果。

## Qt 测试

命令：`powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1`

脚本逐目标执行，10 个目标全部退出码 0。各可执行文件结果如下：

| 测试目标 | 通过 | 失败 |
| --- | ---: | ---: |
| `test_appconfig` | 8 | 0 |
| `test_backendclient` | 14 | 0 |
| `test_backendprocessmanager` | 11 | 0 |
| `test_annotationmanager` | 10 | 0 |
| `test_appfoundation` | 15 | 0 |
| `test_inspectionpage` | 31 | 0 |
| `test_workpiecelibrarypage` | 24 | 0 |
| `test_geometryrulecanvas` | 18 | 0 |
| `test_geometryrulespage` | 78 | 0 |
| `test_mainwindow` | 110 | 0 |
| **合计** | **319** | **0** |

`test_mainwindow` 在最终修正后额外连续运行两轮，均为 `110 passed, 0 failed`。

## Python 回归

命令：

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -m "not integration" -q -p no:cacheprovider --basetemp="$env:TEMP\pytest-ui-redesign"
```

结果：`375 passed, 3 deselected in 14.78s`。临时目录位于系统 `%TEMP%`，仓库内未生成 pytest 缓存或 basetemp。

## Release 构建

命令：`powershell -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1`

结果：成功。可执行文件：`E:\Project\wang\pp_813\qt_app\build-release\release\workpiece_orientation.exe`。

## 固定截图与 DPI 矩阵

截图由 `test_mainwindow::captureFixedUiEvidenceWhenRequested` 通过 `QT_UI_CAPTURE_DIR` 可复现生成。测试显式应用生产 `AppTheme`，使用 Windows 平台插件、无边框客户区，并固定目标物理屏为 1920×1080；逻辑客户区由 `devicePixelRatio` 反算。三档均直接断言 `QImage::size()` 为 1920×1080，同时断言当前页主操作完整可见且所有可见滚动区域没有水平滚动条。

- [检测工作台](screenshots/qt-ui-inspection.png) — 1920×1080
- [工件库](screenshots/qt-ui-library.png) — 1920×1080
- [几何规则](screenshots/qt-ui-geometry.png) — 1920×1080

| `QT_SCALE_FACTOR` | 进程 | 每页物理 PNG | 逻辑客户区 | 视觉结果 |
| ---: | --- | --- | --- | --- |
| 1.00 | 独立 Windows 进程 | 1920×1080 | 1920×1080 | 通过；正式三张截图入库 |
| 1.25 | 独立 Windows 进程 | 1920×1080 | 1536×864 | 通过；三页主操作完整可见，无水平滚动或裁切 |
| 1.50 | 独立 Windows 进程 | 1920×1080 | 1280×720 | 通过；三页主操作完整可见，无水平滚动或裁切 |

人工查看三档截图确认：检测模式栏保持紧凑，主图首次显示和 resize 后合理填充视口；35+35 摘要不挤坏工件库；几何页中心画布获得主要空间，高级区域仍可滚动；生产 primary 蓝色、面板、间距和焦点样式已生效。

## 工作流验收

| 场景 | 证据 | 结果 |
| --- | --- | --- |
| 启动为单张模式、单/批量真实切换、批量运行时锁定 | `test_inspectionpage` 模式/批量用例；检测页截图 | 通过 |
| 单张/批量复核、选中批量结果后回显图片与证据 | `test_mainwindow` batch/review 用例 | 通过 |
| 1+12、35+35、超过 30 张模板及完整路径 tooltip | `test_workpiecelibrarypage`、`test_mainwindow` 模板用例；工件库截图 | 通过 |
| 建库/演进阶段和失败状态保留 | `test_mainwindow` registration/evolution 用例 | 通过 |
| 几何正反面、拟合边界、验证/发布/回滚及高级区 | `test_geometryrulespage`、`test_mainwindow` geometry workflow 用例；几何页截图 | 通过 |
| 断线/重连/loading/timeout/restart | BackendClient/ProcessManager/MainWindow 用例 | 通过 |
| 长中文路径、不可读图片、缩略文字保留完整 tooltip | Inspection/Library/MainWindow 用例 | 通过 |
| 键盘主流程与清晰焦点；正反预测后焦点顺序与视觉顺序一致 | AppFoundation/Inspection/Library 焦点顺序用例 | 通过 |
| dirty 草稿后保存仍继续检查真实活动任务；取消关闭保留状态 | `test_mainwindow` 四个关闭保护用例 | 通过 |

## 推理性能与语义门禁

精确可复制命令及完整结果见 [qt-ui-redesign-benchmark.md](qt-ui-redesign-benchmark.md)，机器可读结果见 [qt-ui-redesign-benchmark.json](qt-ui-redesign-benchmark.json)。报告记录的代码提交为 `ba09e4e7a85bf6d45ddb4f9d35877b01fab2c850`，工作树为 clean；因此结果可直接映射到本次生产代码。

- M1/M2/M7 共 120 个查询，标签不一致 0，复检状态不一致 0。
- 发布门禁：通过；正式默认模式仍为 `adaptive`。
- 查询 wall time 自适应相对穷举加速 1.318×；含加载、建库和 warmup 的总进程时长加速 1.240×（84.18s → 67.90s）。

## 已知限制

- 125%/150% 截图仅作为系统临时目录中的矩阵证据，不提交额外 PNG，避免仓库截图污染。
- 截图使用确定性的合成工件和测试服务器状态，验证 UI 布局与状态呈现；真实相机/现场光照不属于本 UI 任务的视觉验收范围。
- Qt `offscreen` 测试会打印平台插件不支持 `propagateSizeHints()` 以及字体目录提示；Windows 原生截图、Release 构建与测试结果均正常。
