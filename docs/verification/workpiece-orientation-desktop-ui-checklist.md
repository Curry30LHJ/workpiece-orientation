# 工件正反面检测桌面端验收清单

## 构建环境

- Windows 10/11 x64。
- Qt 5.14.2 MSVC2017 64-bit。
- Visual Studio 2019 x64 Native Tools。
- `shitu` Conda 环境可导入 Paddle、PyTorch、ALIKED/LightGlue。
- `app_config.json` 放在 Qt 可执行文件同目录，且所有路径使用绝对路径或由项目根目录解析。

构建命令：

```powershell
.\scripts\build_qt5.ps1
```

## 自动化验收

- Python 非集成测试：`45 passed, 3 deselected`。
- Qt AppConfig：5/5。
- Qt BackendClient：8/8。
- Qt BackendProcessManager：7/7。
- Qt MainWindow：7/7。
- `openspec validate workpiece-orientation-desktop-ui --strict` 通过。

真实模型闭环需要显式执行：

```powershell
$env:WORKPIECE_ORIENTATION_RUN_INTEGRATION = '1'
python -m pytest tests/test_orientation_service_integration.py -m integration -v
```

## 手工验收

1. 删除或暂时移走 `app_config.json`，启动程序，确认显示“后端：配置错误”，程序不崩溃。
2. 先启动 Python 服务，再启动 Qt，确认 Qt 复用已有服务，不结束外部 Python 进程。
3. 不启动服务直接启动 Qt，确认 Qt 只在首次连接失败后启动本会话服务，并显示“后端：已连接”。
4. 选择正面 5 张、反面 5 张图片；少于 5 张、重复路径、空名称都必须在本地阻止建库请求。
5. 使用已有工件名建库，确认先发送不覆盖请求；收到覆盖提示后选择“否”不覆盖，选择“是”才发送 `replace=true`。
6. 建库成功后刷新列表，选择新工件，选择一张待测图片，确认预览和检测按钮状态正确。
7. 检测结果确认同时显示正/反标签、全局/局部原始得分、间隔、决策来源和耗时；不显示百分比置信度。
8. `needs_review=true` 时确认出现“建议人工复检”，后端断线或退出时确认结果和旧预览被清空，并出现“重启后端”。
9. 点击“重启后端”只重启本会话拥有的进程；外部服务只重新连接，不被 Qt 终止。
10. 退出 Qt 后确认本会话启动的服务收到 `shutdown` 并退出，外部服务仍在运行。

## 失败排查

- Qt 无法连接：检查 `host` 必须为 `127.0.0.1`、端口是否被占用、`app_config.json` 是否与 exe 同目录。
- 后端启动失败：直接运行 `src/orientation_tcp_service.py` 查看 stderr；不要按名称结束所有 Python 进程。
- PyTorch `WinError 127`：检查 `shitu` 环境中的 PyTorch/CUDA DLL 版本匹配和 VC++ 运行库；先通过 `python -c "import torch; print(torch.cuda.is_available())"` 再运行真实集成测试。
