# 递推失败容错验证记录

日期：2026-08-17

## 验证范围

- 单个模板对的局部特征投影异常只影响对应目标，后续模板继续递推。
- 连续未知投影异常达到错误预算时，中止候选草稿且不改变当前识别使用的标注修订。
- CUDA/设备级局部模型异常通过 TCP 返回稳定的 `MODEL_ERROR`，不向客户端暴露堆栈。
- Qt 干扰标注管理器显示持久化错误，并将 `projection_failed` 等原因码转换为中文说明。
- 旧有递推进度事件、模板数量和预测协议字段保持不变。

## 自动化结果

| 检查项 | 命令/目标 | 结果 |
| --- | --- | --- |
| 模板递推与错误预算 | `tests/test_template_evolution.py` | 17 passed |
| TCP 服务、进度和稳定错误码 | `tests/test_orientation_tcp_service.py` | 26 passed |
| 局部模型异常分类 | `tests/test_orientation_classifier.py` 新增 2 个测试 | 2 passed |
| LightGlue 推理上下文 | `test_projection_runs_lightglue_inside_torch_inference_context` | passed |
| Python 语法 | `py_compile`（三个服务模块） | passed |
| Qt 干扰标注管理器 | `test_annotationmanager.exe` | 10 passed |
| Qt 主窗口 | `test_mainwindow.exe` | 12 passed |
| Qt Release 应用 | `qt_app/build-release/Makefile.Release` | build passed |

Python 两个核心文件合计 43 个测试通过。Qt 测试在 `QT_QPA_PLATFORM=offscreen` 下运行；环境会提示 Qt 安装包缺少字体目录，但不影响测试结果。

## 行为结论

单个投影失败现在会生成 `projection_failed` 待复核目标，并携带失败来源、尝试来源数、成功投影数和失败计数；它不会被当作“确认不存在”。连续异常触发预算后，后端返回 `MODEL_ERROR`，活动标注修订和预测缓存保持不变。诊断堆栈写入工件库目录的 `diagnostics/orientation-service.log`，并按 10 MiB、3 个备份滚动。

## 性能影响

本次修改只增加每个模板对的异常捕获、少量诊断字段和日志写入，不改变 PP-ShiTuV2、ALIKED、LightGlue、关键点数量、输入尺寸或融合阈值。正常递推的模型计算量不变；只有异常路径会产生日志 I/O，连续异常达到预算后会提前结束，避免继续无效计算。
