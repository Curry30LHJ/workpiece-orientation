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

Qt 启动后先使用 `QTcpSocket` 连接配置中的 `127.0.0.1:37651`。连接成功后发送 `hello` 完成协议与服务身份握手；只有 `ready=true` 才认为后端就绪。连接失败时，Qt 使用 QProcess 启动配置中的 Python 服务，参数包含 `-u`、服务脚本、项目/模型/模板目录、主机和端口，然后在 `startup_timeout_ms` 配置的时限内轮询连接。由于 PaddleClas 与 ALIKED/LightGlue 首次加载可能超过 5 分钟，示例配置将该值设为 600000 毫秒。Python 先绑定端口，再在后台线程加载模型和恢复模板缓存；加载期间 `hello` 返回 `ready=false/status=loading`，Qt 显示“后端：模型加载中”并只重试握手，不重复启动 Python 进程。

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

无阻塞实现的开放问题。阈值独立校准和工业相机接入作为后续独立变更处理。
