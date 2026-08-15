# Comet Design Handoff

- Change: workpiece-orientation-desktop-ui
- Phase: design
- Mode: compact
- Context hash: 6936d7e562bde4ac9168c51c46fd2ae13028fae91ec4e2a705e9608db9b140ff

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/workpiece-orientation-desktop-ui/proposal.md

- Source: openspec/changes/workpiece-orientation-desktop-ui/proposal.md
- Lines: 1-30
- SHA256: 7b099fc06e0012a7576900a7f99c888f5a55bdf02eea35ba0431925a16aaac08

```md
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
```

## openspec/changes/workpiece-orientation-desktop-ui/design.md

- Source: openspec/changes/workpiece-orientation-desktop-ui/design.md
- Lines: 1-82
- SHA256: 4678b8a1587837d11b91aec02d8fc7d3e45e3470a42463cbf98eb1ebf76aff76

[TRUNCATED]

```md
## Context

现有项目已在 M1、M2、M7 数据上评估“PP-ShiTuV2 全局特征 + ALIKED/LightGlue 关键点上限 512 + 软中心融合”，但入口均为面向数据集的 Python 命令行脚本。当前阈值来自探索性实验，报告明确要求在独立注册集上继续验证，因此桌面应用必须展示原始决策证据和复检状态，不能把得分包装成概率。现场环境使用 Qt 5.14.2，AI 依赖位于 `shitu` Conda 环境；将模型迁入 C++ 会重复处理 Paddle、PyTorch 和模型权重部署问题。

## Goals / Non-Goals

**Goals:**

- 提供 Qt 5.14.2 Widgets 单窗口应用，完成工件列表刷新、5+5 模板建库和单张检测。
- 让 Python 进程常驻加载模型，并缓存每个工件的模板全局向量和局部特征。
- 使用可版本化、可关联请求、支持中文路径的 JSON 行协议。
- 复用完整图、关键点上限 512 和软中心融合策略，并展示全局/局部证据、决策来源和复检提示。

**Non-Goals:**

- 不直接连接工业相机，不支持连续取流或批量检测。
- 不实现目标分割、多工件检测、数据库、账户、网络服务或模板删除功能。
- 不把模型原始得分校准为概率，也不将 AI 模型重写为 C++ 推理实现。

## Decisions

### 使用 qmake 构建 Qt Widgets C++ 前端

Qt 工程使用 qmake `.pro` 文件和 Qt Widgets，默认开发套件为已安装的 `E:/QT/5.14/5.14.2/msvc2017_64/bin/qmake.exe`。工程不链接第三方 C++ AI 库，因此部署到其他工控机时也可使用兼容的 Qt 5.14.2 MSVC 套件重新构建。

应用从可执行文件旁的 `app_config.json` 读取 `python_executable`、`backend_script`、`project_root`、`model_dir`、`library_dir`、`host` 和 `port`。`host` 只允许 `127.0.0.1`，默认端口为 37651。路径不存在、端口非法或配置字段缺失时，不启动后端并在界面显示配置错误。默认开发配置使用 `E:/python/anaconda3/envs/shitu/python.exe`，部署时不把绝对开发路径编译进程序。

### 使用本机 TCP 服务并由 Qt 按需启动

Qt 启动后先使用 `QTcpSocket` 连接配置中的 `127.0.0.1:37651`。连接成功后发送 `hello` 完成协议与服务身份握手；只有握手成功才认为后端就绪。连接失败时，Qt 使用 QProcess 启动配置中的 Python 服务，参数包含 `-u`、服务脚本、项目/模型/模板目录、主机和端口，然后在最长 120 秒内轮询连接。Python 完成模型加载和模板缓存恢复后才开始监听端口。

Qt 记录服务所有权：连接启动前已存在的服务视为外部服务，本次会话启动成功的 QProcess 视为自有服务。程序关闭时只向自有服务发送 `shutdown`；外部服务仅断开 socket。连接异常后不自动循环重启，界面清空旧结果并提供“重启后端”按钮；按钮先重连已有服务，自有进程无响应时可终止后重新启动，外部进程则不被 Qt 终止。

选择回环 TCP 而不是标准输入/输出，是为了让模型服务能够跨 Qt 会话复用并与界面生命周期解耦；选择 `127.0.0.1` 而不是局域网地址，是为了保持首版的本机边界。服务只允许一个活跃客户端，额外连接收到 `SERVER_BUSY` 后关闭。

### 使用版本化 UTF-8 JSON 行协议

所有 TCP 请求均包含 `version=1`、唯一 `request_id` 和 `command`，命令包括 `hello`、`list_workpieces`、`register`、`predict`、`shutdown`。每个响应必须回传相同 `request_id`。`hello` 成功响应包含 `service=workpiece-orientation`、协议版本和 `ready=true`；端口被其他程序占用或握手内容不匹配时，Qt 报告端口冲突且不启动第二个服务。错误响应使用稳定错误码和中文消息：

```json
{"version":1,"request_id":"12","ok":false,"error":{"code":"IMAGE_UNREADABLE","message":"无法读取图片"}}
```

Qt 和 Python 均按换行符累积 socket 缓冲区，能够处理半行和一次到达多行的情况。双方固定 UTF-8；Python 使用 `json.dumps(..., ensure_ascii=False)`，Qt 使用 `QJsonDocument`。Python 日志输出到标准错误或日志文件，不写入 TCP 响应。客户端一次只允许一个在途业务请求，响应请求标识不匹配时进入协议错误状态。

### 事务式持久化模板并缓存特征

Qt 提交工件显示名称、正面和反面各 5 张互不重复的绝对图片路径。后端拒绝空名称、路径分隔符、控制字符及同组重复图片，固定 `0=front/正面`、`1=back/反面`。目录使用后端生成的安全内部标识，不直接把显示名称拼入路径；每个工件目录保存 `manifest.json`，记录内部标识、显示名称和两类固定映射。

`register` 默认 `replace=false`，同名工件返回 `WORKPIECE_EXISTS`；用户在 Qt 明确确认覆盖后才发送 `replace=true`。后端先在同一父目录的临时目录中完成图片复制、清单写入、可读性校验和模板特征提取。覆盖时将旧目录改名为唯一备份目录，再将临时目录改名为正式目录；任一步骤失败都恢复旧目录，成功后才删除备份。服务串行处理请求，因此交换期间没有其他预测读取该目录。后端启动时扫描模板库清单并重建内存缓存，建库成功后只刷新该工件缓存。`list_workpieces` 返回可用工件列表，Qt 不从文件系统自行推断。

### 返回可解释的融合证据而不是概率置信度

首版固定 `ROI_RATIO=1.0`、`MAX_NUM_KEYPOINTS=512`、`GLOBAL_MARGIN_THRESHOLD=0.05`、`LOCAL_MIN_SCORE=4.0`、`LOCAL_MIN_MARGIN=0.5` 和 `LOCAL_OVERRIDE_MARGIN=3.0`。`MAX_NUM_KEYPOINTS` 是上限，实际特征数可少于 512。1024 上限没有提高当前数据的最终准确率，并可能增加遮挡区域误匹配，因此不暴露为界面选项。

后端返回 `global_scores`、`global_margin`、`local_scores`、`local_margin`、`local_prediction`、`decision_source`、最终 `label` 和 `needs_review`。局部最高分低于 4.0 或局部间隔低于 0.5 时，`local_prediction=uncertain`，这一定义为“局部证据不足”。全局间隔不大于 0.05，且局部预测有效、局部间隔不小于 3.0 时，最终标签采用局部结果并标记 `decision_source=local_override`；其他情况采用全局结果。以下任一条件成立时 `needs_review=true`：全局间隔不大于 0.05 且 `local_prediction=uncertain`；或者全局与有效局部预测不一致。全局与局部在低全局间隔下给出同一有效标签时允许 `needs_review=false`。

这些阈值属于当前实验配置而非概率校准结论；后续现场验证可以修改后端常量，但不改变 JSON 协议。

## Risks / Trade-offs

- [Python 环境、脚本或权重路径不存在] → 启动前校验配置；Qt 显示具体字段错误并保持操作按钮禁用。
- [模型首次加载慢] → 常驻后端只加载一次；Qt 使用连接轮询与 `hello` 握手明确就绪边界。
- [端口被其他程序占用] → 连接后校验服务身份和协议版本；握手失败时报告端口冲突，不盲目启动第二个服务。
- [多个 Qt 客户端竞争] → 服务只保留一个活跃客户端，其他连接返回 `SERVER_BUSY`。
- [中文路径或 JSON 拆包失败] → 固定 UTF-8、按换行缓存解析，并用中文目录测试。
- [覆盖建库中途失败] → 同父目录临时建库并通过旧库备份执行事务式交换，失败时恢复旧库。
- [阈值尚未独立校准] → 展示原始得分和决策来源，对证据冲突或不足的结果标记人工复检。
- [后端或连接异常] → Qt 清空旧结果、禁用操作并显示后端不可用；由用户显式重连或重启，自有与外部服务按所有权分别处理。

## Migration Plan

1. 保留现有命令行评估脚本和模型目录不变。
2. 新增可测试的模板库、特征缓存、单图融合推理和 JSON 服务入口。
3. 新增独立 qmake Qt 工程、TCP 服务客户端/进程管理器与 `app_config.json.example`。
4. 使用 M1、M2、M7 的固定样本完成后端回归、回环 TCP 和中文路径协议冒烟验证。
5. 使用 Qt 5.14.2 MSVC2017 64 位套件编译，并验证已有服务复用、按需启动、建库、覆盖、重启恢复、检测和错误状态。
6. 回滚时停用 Qt 程序即可；原有 Python 评估命令不受影响。

## Open Questions
```

Full source: openspec/changes/workpiece-orientation-desktop-ui/design.md

## openspec/changes/workpiece-orientation-desktop-ui/tasks.md

- Source: openspec/changes/workpiece-orientation-desktop-ui/tasks.md
- Lines: 1-24
- SHA256: c989575614355b2877bf1b043c3b4540835396268a396e82779efdee4cd837f9

```md
## 1. Python 推理后端

- [ ] 1.1 提取可复用的模型加载器和关键点上限 512 的软中心融合单图分类器，返回全局/局部证据、决策来源和复检标志。
- [ ] 1.2 实现安全工件标识、清单文件、5+5 模板校验、同名覆盖确认、带备份回滚的事务式目录替换和 `list_workpieces`。
- [ ] 1.3 实现模板全局向量与局部特征缓存，并支持建库刷新和服务重启恢复。
- [ ] 1.4 实现仅监听 `127.0.0.1` 的 TCP 服务、`hello`/`shutdown`、版本化 UTF-8 JSON 行协议、请求标识、单客户端和稳定错误码。
- [ ] 1.5 为融合边界、模板事务更新、缓存、TCP JSON 分包、握手、客户端竞争和错误响应添加不依赖真实权重的自动化测试。

## 2. Qt 5.14.2 应用

- [ ] 2.1 创建 qmake Qt Widgets 工程、包含 TCP 主机/端口的 `app_config.json.example`、主窗口布局及后端状态显示。
- [ ] 2.2 实现 `BackendClient` 的 QTcpSocket 连接、握手、UTF-8 JSON 行缓冲、请求关联、忙状态和断线处理。
- [ ] 2.3 实现 `BackendProcessManager` 的已有服务发现、QProcess 按需启动、服务所有权、120 秒启动超时、显式重启和所有权关闭。
- [ ] 2.4 实现工件列表刷新、名称输入、正反面各 5 张模板选择、同名覆盖确认和建库请求。
- [ ] 2.5 实现待测图片选择、原图预览、检测请求，以及标签、原始证据、决策来源、耗时和复检提示展示。
- [ ] 2.6 使用假 TCP 服务和 Qt Test 覆盖半行/多行 JSON、请求标识不匹配、服务复用、按需启动、`SERVER_BUSY`、断线、配置错误和按钮状态。

## 3. 集成与交付验证

- [ ] 3.1 使用实际 `shitu` 环境运行 Python 自动化测试，并执行包含中文路径的回环 TCP JSON 协议冒烟验证。
- [ ] 3.2 使用 M1、M2、M7 固定样本验证服务化推理与现有 512 实验入口的标签一致性。
- [ ] 3.3 使用 Qt 5.14.2 MSVC2017 64 位 qmake 编译 Qt 工程并运行 Qt Test。
- [ ] 3.4 手动验证已有服务复用、按需启动、建库、同名覆盖、显式重启、外部服务所有权、单张检测、复检提示、端口冲突及后端异常退出。
- [ ] 3.5 编写 Windows 工控机配置、构建、运行和故障排查说明。
```

## openspec/changes/workpiece-orientation-desktop-ui/specs/desktop-orientation-inspection/spec.md

- Source: openspec/changes/workpiece-orientation-desktop-ui/specs/desktop-orientation-inspection/spec.md
- Lines: 1-75
- SHA256: 59a2c0c25a73a87b50de43f651c56aebce1162b4867a00098438746d03720ebc

```md
## ADDED Requirements

### Requirement: Qt 桌面界面发现或按需启动本地服务
系统 SHALL 从配置文件读取 Python 服务路径和 `127.0.0.1` 回环端口，启动时先尝试连接已有服务。连接失败时，系统 SHALL 通过 QProcess 按需启动 Python 服务，并在 TCP 连接及版本 1 `hello` 握手成功前禁用建库和检测。配置、启动、模型加载、端口或握手失败时，系统 SHALL 显示可理解的错误状态。

#### Scenario: 复用已有服务
- **WHEN** Qt 启动时配置端口上已有服务，且 `hello` 响应的服务身份和协议版本正确
- **THEN** Qt 复用该服务并标记为外部服务，不启动新的 Python 进程

#### Scenario: 按需启动服务
- **WHEN** Qt 启动时无法连接配置端口
- **THEN** Qt 启动配置的 Python 服务，在最长 120 秒内轮询连接，并在 `hello` 成功后开放工件列表刷新和建库操作

#### Scenario: 后端启动失败
- **WHEN** 配置路径无效、进程无法启动、模型加载失败、连接超时或端口上服务握手不匹配
- **THEN** 界面保持相关操作禁用，显示错误原因，且不展示过期检测结果

#### Scenario: 服务正在被其他客户端使用
- **WHEN** 后端返回 `SERVER_BUSY`
- **THEN** 界面提示服务正在被其他客户端使用，并保持建库和检测禁用

### Requirement: Qt 桌面界面展示已建库工件
系统 SHALL 通过后端 `list_workpieces` 命令获取工件列表，并允许操作员选择一个工件进行检测。界面不得通过直接扫描模板目录推断可用工件。

#### Scenario: 后端返回工件列表
- **WHEN** 后端就绪或操作员点击刷新
- **THEN** 界面展示所有可用工件，并在列表为空时禁用检测操作

### Requirement: Qt 桌面界面提供事务式工件模板建库
系统 SHALL 允许操作员输入工件名称，并分别选择正面和反面模板图片。界面 SHALL 在提交前要求每一面恰好选择 5 张互不重复的 PNG、JPEG 或 BMP 图片，并将建库结果或错误显示给操作员。

#### Scenario: 成功建立新工件模板库
- **WHEN** 操作员输入未使用的工件名称，并选择正面和反面各 5 张有效图片后提交
- **THEN** 系统将模板建库成功并提示该工件可用于检测

#### Scenario: 模板数量不符合要求
- **WHEN** 操作员提交的正面或反面图片数量不是 5 张
- **THEN** 系统不发送建库请求并显示具体的数量错误

#### Scenario: 同名工件默认不覆盖
- **WHEN** 操作员提交的工件名称已经存在
- **THEN** 界面先提示同名冲突，只有操作员明确确认覆盖后才发送 `replace=true` 的建库请求

### Requirement: Qt 桌面界面执行单张检测
系统 SHALL 允许操作员选择一个已建库工件和一张本地待测图片，并展示原图、正面或反面结果、全局与局部类别得分、两个间隔、决策来源和耗时。系统不得把这些原始得分显示为概率置信度。

#### Scenario: 成功检测本地图片
- **WHEN** 操作员选择已建库工件和有效待测图片并发起检测
- **THEN** 系统展示待测原图及后端返回的检测结果

#### Scenario: 检测请求正在执行
- **WHEN** 一个建库或检测请求尚未收到响应
- **THEN** 界面禁用重复提交并显示处理中状态，同时保持窗口事件循环可响应

### Requirement: Qt 桌面界面提示需要复检的结果
系统 SHALL 在后端响应的 `needs_review` 为真时，将检测结果明确标记为“建议人工复检”，并同时展示造成该结论的全局间隔、局部间隔和决策来源。

#### Scenario: 证据不足或冲突的检测
- **WHEN** 后端返回 `needs_review=true`
- **THEN** 界面在结果区域显示人工复检提示

### Requirement: Qt 桌面界面处理后端不可用
系统 SHALL 在 Python 后端未就绪或异常退出时禁用依赖后端的操作，并向操作员显示可理解的错误状态。

#### Scenario: 后端启动失败
- **WHEN** Qt 程序无法启动 Python 后端或后端在运行时退出
- **THEN** 界面显示后端不可用原因，且不展示过期的检测结果

#### Scenario: 用户显式重启后端
- **WHEN** 连接异常后操作员点击“重启后端”
- **THEN** Qt 先尝试重连已有服务；仍不可用时只终止本次 Qt 启动且无响应的服务进程，再按需启动新服务

#### Scenario: 程序正常关闭
- **WHEN** 操作员关闭 Qt 程序
- **THEN** Qt 只向本次会话启动的服务发送 `shutdown` 并等待退出；对于启动前已存在的外部服务，Qt 只断开 TCP 连接
```

## openspec/changes/workpiece-orientation-desktop-ui/specs/python-inference-bridge/spec.md

- Source: openspec/changes/workpiece-orientation-desktop-ui/specs/python-inference-bridge/spec.md
- Lines: 1-86
- SHA256: a4a595ebd7a9d1ff058b44ab711684d6a2f432f17cb18ba28ad5254b8db54897

[TRUNCATED]

```md
## ADDED Requirements

### Requirement: Python 后端提供仅限回环地址的 TCP JSON 服务
Python 后端 SHALL 使用标准库 socket，仅绑定配置的 `127.0.0.1` 地址和端口，并在模型加载、模板扫描及缓存恢复完成后开始监听。服务 SHALL 按行接收和返回 UTF-8 JSON；每个请求 SHALL 包含 `version=1`、唯一 `request_id` 和 `command`，响应 SHALL 回传相同请求标识。日志不得写入 TCP JSON 响应。

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
Python 后端 SHALL 实现 `list_workpieces` 命令，并只返回目录完整、正反各有 5 张有效模板的工件。

#### Scenario: 服务重启后恢复工件列表
- **WHEN** 后端启动时模板目录中存在完整工件库
- **THEN** `list_workpieces` 返回这些工件，且后端已为其重建模板特征缓存

### Requirement: Python 后端事务式持久化和校验模板
Python 后端 SHALL 在收到 `register` 请求时验证工件显示名称及两组模板。系统 SHALL 拒绝空名称、路径分隔符、控制字符、同组重复文件及任一面不是 5 张有效图片的请求，并使用安全内部标识保存工件，通过 `manifest.json` 记录显示名称，固定 `0=front/正面`、`1=back/反面`。

#### Scenario: 拒绝无效模板集
- **WHEN** `register` 请求中任一面模板不足、超过或包含不可读取图片
- **THEN** 后端返回 `ok=false`，且不得创建不完整的持久化模板库

#### Scenario: 默认拒绝同名工件
- **WHEN** `register` 请求的工件名称已经存在且 `replace=false`
- **THEN** 后端返回 `WORKPIECE_EXISTS`，并保持现有模板和缓存不变

#### Scenario: 事务式覆盖同名工件
- **WHEN** 同名工件请求包含 `replace=true`，且新的 5+5 模板全部有效并成功提取特征
- **THEN** 后端先把旧库改名为备份，再把已完成验证和特征提取的临时库改名为正式库，成功后刷新缓存并删除备份

#### Scenario: 覆盖建库中途失败
- **WHEN** 新模板复制、读取或特征提取任一步骤失败
- **THEN** 后端删除临时建库结果并保留旧模板库和旧缓存

### Requirement: Python 后端缓存模板特征
Python 后端 SHALL 在建库成功或服务启动恢复工件时计算并缓存 10 张模板的 PP-ShiTuV2 全局向量及 ALIKED 局部特征。单张预测 SHALL 复用缓存，不得重复提取模板特征。

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

```

Full source: openspec/changes/workpiece-orientation-desktop-ui/specs/python-inference-bridge/spec.md

