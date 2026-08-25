## ADDED Requirements

### Requirement: Python 后端提供仅限回环地址的 TCP JSON 服务
Python 后端 SHALL 使用标准库 socket，仅绑定配置的 `127.0.0.1` 地址和端口，并在模型加载线程运行期间先开始监听。服务 SHALL 按行接收和返回 UTF-8 JSON；每个请求 SHALL 包含 `version=1`、唯一 `request_id` 和 `command`，响应 SHALL 回传相同请求标识。日志不得写入 TCP JSON 响应。

#### Scenario: 模型加载期间服务可握手
- **WHEN** 客户端在模型、模板扫描或缓存恢复完成前连接并发送 `hello`
- **THEN** 后端返回相同请求标识、`service=workpiece-orientation`、`ready=false`、`status=loading` 和“模型加载中”，并继续接受后续 `hello` 或 `shutdown`

#### Scenario: 服务握手成功
- **WHEN** 客户端连接后发送版本 1 的 `hello` 请求
- **THEN** 后端返回相同请求标识、`service=workpiece-orientation`、`version=1` 和 `ready=true`

#### Scenario: 处理合法预测请求
- **WHEN** 后端收到包含已建库工件名称和有效图片路径的 `predict` JSON 请求
- **THEN** 后端返回一行 `ok=true` 的 JSON 响应，其中包含相同请求标识、最终标签、全局与局部得分、间隔、决策来源、复检标志和耗时

#### Scenario: 处理无效 JSON 请求
- **WHEN** 后端收到不能解析或缺少命令字段的请求行
- **THEN** 后端返回一行包含稳定错误码和错误消息的 `ok=false` JSON 响应，并继续处理后续请求

#### Scenario: 请求协议版本不兼容
- **WHEN** 请求的 `version` 不是 1
- **THEN** 后端返回 `UNSUPPORTED_PROTOCOL_VERSION` 错误，并且不执行请求命令

#### Scenario: 拒绝非回环监听地址
- **WHEN** 服务启动参数中的监听地址不是 `127.0.0.1`
- **THEN** 服务启动失败并报告 `INVALID_BIND_ADDRESS`

### Requirement: Python 后端只允许一个活跃客户端
Python 后端 SHALL 同一时间只允许一个完成握手的客户端处理业务请求，并串行执行该客户端的请求。

#### Scenario: 第二个客户端连接
- **WHEN** 已有一个活跃客户端时另一个客户端连接
- **THEN** 服务向新连接返回 `SERVER_BUSY` 并关闭该连接，不影响当前客户端

### Requirement: Python 后端列出可用工件
Python 后端 SHALL 实现 `list_workpieces` 命令，并只返回目录完整、正反均至少有 1 张有效模板的工件；正反模板数量可以不同。

#### Scenario: 服务重启后恢复工件列表
- **WHEN** 后端启动时模板目录中存在完整工件库
- **THEN** `list_workpieces` 返回这些工件，且后端已为其重建模板特征缓存

### Requirement: Python 后端事务式持久化和校验模板
Python 后端 SHALL 在收到 `register` 请求时验证工件显示名称及两组模板。系统 SHALL 拒绝空名称、路径分隔符、控制字符、任一面为空、路径或解码内容重复、以及无法读取的图片；不得静默截断用户选择。系统使用安全内部标识保存工件，通过 `manifest.json` 记录显示名称和正反实际模板数量，固定 `0=front/正面`、`1=back/反面`。

#### Scenario: 拒绝无效模板集
- **WHEN** `register` 请求中任一面模板为空、存在重复图片或包含不可读取图片
- **THEN** 后端返回 `ok=false`，且不得创建不完整的持久化模板库

#### Scenario: 默认拒绝同名工件
- **WHEN** `register` 请求的工件名称已经存在且 `replace=false`
- **THEN** 后端返回 `WORKPIECE_EXISTS`，并保持现有模板和缓存不变

#### Scenario: 事务式覆盖同名工件
- **WHEN** 同名工件请求包含 `replace=true`，且新的正反模板列表均非空、全部有效并成功提取特征
- **THEN** 后端先把旧库改名为备份，再把已完成验证和特征提取的临时库改名为正式库，成功后刷新缓存并删除备份

#### Scenario: 覆盖建库中途失败
- **WHEN** 新模板复制、读取或特征提取任一步骤失败
- **THEN** 后端删除临时建库结果并保留旧模板库和旧缓存

### Requirement: Python 后端缓存模板特征
Python 后端 SHALL 在建库成功或服务启动恢复工件时，按请求的正反模板数量计算并缓存全部模板的 PP-ShiTuV2 全局向量及 ALIKED 局部特征。单张预测 SHALL 复用缓存，不得重复提取模板特征。

#### Scenario: 连续预测同一工件
- **WHEN** 后端对同一已建库工件连续处理多张待测图片
- **THEN** 每次预测只提取待测图片特征并复用该工件模板缓存

### Requirement: Python 后端使用固定的 512 关键点融合策略
Python 后端 SHALL 对有效预测请求使用完整输入图、最多 512 个 ALIKED 关键点、软中心局部匹配和 PP-ShiTuV2 全局特征。运行参数 SHALL 固定为 `GLOBAL_MARGIN_THRESHOLD=0.05`、`LOCAL_MIN_SCORE=4.0`、`LOCAL_MIN_MARGIN=0.5` 和 `LOCAL_OVERRIDE_MARGIN=3.0`。

#### Scenario: 对已建库工件预测
- **WHEN** 后端收到已完成建库工件的有效 `predict` 请求
- **THEN** 后端基于该工件正反模板返回 `front` 或 `back` 标签，以及全局得分、全局间隔、局部得分、局部间隔、局部预测、决策来源和复检标志

#### Scenario: 局部证据覆盖低间隔全局结果
- **WHEN** 全局间隔不大于 0.05，局部预测有效且局部间隔不小于 3.0
- **THEN** 最终标签采用局部预测，并返回 `decision_source=local_override`

#### Scenario: 证据不足或冲突时提示复检
- **WHEN** 低全局间隔下局部最高分低于 4.0 或局部间隔低于 0.5，或者全局预测与有效局部预测不一致
- **THEN** 后端返回 `needs_review=true`，且不得把原始得分声明为概率置信度

### Requirement: Python 后端串行处理请求并支持关闭
Python 后端 SHALL 一次处理一个业务请求，并实现 `shutdown` 命令。单个请求失败不得导致服务退出；客户端正常断开后服务 SHALL 返回等待下一个客户端。收到关闭请求后 SHALL 返回响应并正常结束服务进程。

#### Scenario: 正常关闭服务
- **WHEN** 后端收到有效的 `shutdown` 请求
- **THEN** 后端返回对应请求标识的成功响应并结束请求循环
