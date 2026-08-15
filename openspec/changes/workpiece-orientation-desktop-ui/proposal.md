## Why

当前正反面识别流程只能通过命令行运行，现场操作人员无法直接完成新工件的少样本建库和单张检测。需要提供一个与既有 Python 推理环境协同工作的 Qt 5.14.2 桌面界面，使该流程可以在工控机上被直接使用。

## What Changes

- 新增基于 qmake 的 Qt 5.14.2 C++ 单窗口应用，用于工件正反面建库和单张检测。
- 新增 Python 单图推理入口，复用现有 PP-ShiTuV2 全局特征、ALIKED/LightGlue 最多 512 关键点及软中心融合策略。
- Qt 使用 `QTcpSocket` 连接仅监听 `127.0.0.1` 的常驻 Python 推理服务；没有可用服务时再通过 QProcess 按需启动服务。双方使用带版本、请求标识的 UTF-8 JSON 行协议。
- 支持查询已建库工件、事务式建立或明确覆盖 5+5 模板库，并缓存模板全局向量与局部特征。
- 在界面中显示待测图、正面/反面结果、全局和局部得分、间隔、决策来源、耗时及复检提示；不把原始得分表述为概率置信度。

## Capabilities

### New Capabilities

- `desktop-orientation-inspection`: Qt 桌面应用支持已建库工件列表、本地图片事务式建库和单张正反面检测。
- `python-inference-bridge`: Python 后端通过稳定的 JSON 协议提供工件查询、建库和单图推理接口。

### Modified Capabilities

无。

## Impact

- 新增 Qt 5.14.2 C++ qmake 工程、配置文件及界面资源。
- 新增或调整 `src/` 下的 Python 推理封装和本机 TCP 服务入口。
- 运行时依赖既有 `shitu` Conda 环境、PaddlePaddle、PyTorch、ALIKED 和 LightGlue 权重。
- 默认开发环境使用 `E:/QT/5.14/5.14.2/msvc2017_64/bin/qmake.exe`、`E:/python/anaconda3/envs/shitu/python.exe` 和回环地址 `127.0.0.1:37651`；实际部署路径与端口由应用配置文件指定。
- 首版只支持本地文件选择；不接入工业相机、批处理、数据库、用户权限或目标分割。
