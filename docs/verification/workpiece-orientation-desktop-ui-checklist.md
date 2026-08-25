# 工件正反面检测桌面端验收清单

## 构建环境

- Windows 10/11 x64。
- Qt 5.14.2 MSVC2017 64-bit。
- Visual Studio 2019 x64 Native Tools。
- `shitu` Conda 环境可导入 Paddle、PyTorch、ALIKED/LightGlue。
- 将 `qt_app/app_config.json.example` 复制为本机未提交的 `qt_app/app_config.json`，填写本机路径；构建脚本会自动将它复制到 Qt 可执行文件同目录。运行时仍要求 `app_config.json` 与 exe 同目录，且所有路径使用绝对路径或由项目根目录解析。

构建命令：

```powershell
.\scripts\build_qt5.ps1
```

## 自动化验收

- Python 非集成测试：以本次运行结果为准，覆盖 1+1、5+10、10+15、超过 30 张、无效/重复图片和旧库恢复。
- Qt AppConfig、BackendClient、BackendProcessManager、MainWindow：以本次 Qt 5.14.2 offscreen 测试结果为准。
- 真实服务集成：M1、M2、M7 各 `1 passed`（固定 seed=20260813，中文路径建库/预测）。
- `openspec validate workpiece-orientation-desktop-ui --strict` 通过。

本次持久化缓存实测（M7，35+35 模板，RTX 级 GPU）：首次无缓存启动约 412 秒；生成约 4.7 MB `.template_cache.pkl` 后再次启动约 216 秒，减少约 48%。模型初始化仍占主要时间，缓存主要消除了模板特征重复提取。

真实模型闭环需要显式执行：

```powershell
$env:WORKPIECE_ORIENTATION_RUN_INTEGRATION = '1'
python -m pytest tests/test_orientation_service_integration.py -m integration -v
```

## 手工验收

1. 删除或暂时移走 `app_config.json`，启动程序，确认显示“后端：配置错误”，程序不崩溃。
2. 先启动 Python 服务，再启动 Qt，确认 Qt 复用已有服务，不结束外部 Python 进程。
3. 不启动服务直接启动 Qt，确认 Qt 只在首次连接失败后启动本会话服务，并显示“后端：已连接”。
4. 分别多选任意数量的正面和反面图片（例如 1+1、5+12、超过 30 张）；界面显示实际数量，少量模板仅提示建议补充，空列表、无效图片或重复图片才阻止建库请求。
5. 使用已有工件名建库，确认先发送不覆盖请求；收到覆盖提示后选择“否”不覆盖，选择“是”才发送 `replace=true`。
6. 建库成功后刷新列表，选择新工件，选择一张待测图片，确认预览和检测按钮状态正确。
7. 检测结果确认同时显示正/反标签、全局/局部原始得分、间隔、决策来源和耗时；不显示百分比置信度。
8. `needs_review=true` 时确认出现“建议人工复检”，后端断线或退出时确认结果和旧预览被清空，并出现“重启后端”。
9. 点击“重启后端”只重启本会话拥有的进程；外部服务只重新连接，不被 Qt 终止。
10. 退出 Qt 后确认本会话启动的服务收到 `shutdown` 并退出，外部服务仍在运行。

## 失败排查

- Qt 显示“配置错误”：检查 `qt_app/app_config.json` 是否已从示例复制并填写正确；重新构建后确认 `app_config.json` 与 exe 同目录。
- Qt 长时间显示未连接：首次加载 PaddleClas 与 ALIKED/LightGlue 可能需要数分钟；确认配置中的 `startup_timeout_ms` 为 600000。端口会在加载开始时立即可连接，界面应显示“后端：模型加载中”并等待握手变为就绪，不应重复启动多个 Python 进程。
- 后端重复启动仍需重建模板特征时，检查每个活动工件目录是否存在 `.template_cache.pkl`；缓存命中会跳过模板特征提取，缓存缺失、损坏或模板内容变化会自动完整重建。
- Qt 无法连接：检查 `host` 必须为 `127.0.0.1`、端口是否被占用、`app_config.json` 是否与 exe 同目录。
- 后端启动失败：直接运行 `src/orientation_tcp_service.py` 查看 stderr；不要按名称结束所有 Python 进程。
- PyTorch `WinError 127`：检查 `shitu` 环境中的 PyTorch/CUDA DLL 版本匹配和 VC++ 运行库；先通过 `python -c "import torch; print(torch.cuda.is_available())"` 再运行真实集成测试。
