# Brainstorm Summary

- Change: workpiece-orientation-desktop-ui
- Date: 2026-08-15
- Status: confirmed

## Confirmed Technical Approach

- 用户选择独立本地推理服务方案，Qt 不再把推理后端作为仅服务于自身生命周期的标准输入/输出子进程。
- 服务仅监听 `127.0.0.1`，Python 使用标准库 socket，Qt 使用 `QTcpSocket`；不向局域网开放。
- 后端使用版本 1 的 UTF-8 JSON 行协议，保持模型与每个工件的 5+5 模板特征缓存常驻。
- Qt 启动时先连接已有服务；连接失败才通过 QProcess 按需启动 Python 服务并轮询连接。
- Qt 记录服务是否由本次会话启动：退出时只关闭自身启动的服务，外部已有服务仅断开连接。
- 模板库使用安全内部标识、`manifest.json` 和带备份回滚的事务式目录交换。
- 推理使用完整图、ALIKED 关键点上限 512、软中心局部匹配和 PP-ShiTuV2 全局融合，返回原始证据与复检标志。
- 服务连接异常后不自动循环重启；Qt 清空旧结果并由操作员点击“重启后端”，该操作先重连已有服务，再按需启动新服务。

## Confirmed Component Boundaries

- Python: `OrientationClassifier`、`WorkpieceLibrary`、`TcpJsonServer` 三个独立职责模块。
- Qt: `BackendClient`（QTcpSocket）、`BackendProcessManager`（发现/按需启动）、`MainWindow`、配置加载器四个职责模块。

## Key Trade-offs and Risks

- 独立本地服务可复用模型并解耦生命周期，但需要处理固定回环端口、协议握手、启动竞争、断线重连和单客户端约束。
- 内存缓存提高检测速度，但服务启动和新建库会产生一次性特征提取等待。
- 当前阈值来自探索性实验，因此冲突或证据不足必须提示复检。

## Testing Strategy

- Python 单元测试覆盖融合边界、事务回滚、缓存、TCP JSON 分包、握手和错误隔离。
- Qt Test 使用假 TCP 服务覆盖 JSON 分包、请求关联、已有服务复用、按需启动、配置错误、忙状态与断线。
- 使用实际模型执行中文路径协议冒烟测试，并用 M1/M2/M7 固定样本核对服务化与实验入口标签一致性。

## Spec Patches

- 把 proposal 中残留的“原子建库”统一改为“事务式建库”。
- 将 QProcess 标准输入/输出协议改为仅监听 `127.0.0.1`、默认端口 37651 的 TCP JSON 行协议。
- 补充 Qt 发现已有服务、按需启动、协议握手、启动所有权、单客户端、断线清理和用户显式重启场景。
