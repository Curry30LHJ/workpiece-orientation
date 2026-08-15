---
comet_change: workpiece-orientation-desktop-ui
role: technical-design
canonical_spec: openspec
---

# 工件正反面 Qt 桌面应用技术设计

## 1. 设计目标

本变更把现有面向数据集的 Python 评估代码整理为可复用的少样本正反面推理服务，并提供 Qt 5.14.2 C++ Widgets 界面。Qt 只负责本地操作流程、图片展示和服务生命周期协调；Python 保留 PP-ShiTuV2、ALIKED、LightGlue 和 CUDA 推理。

首版只处理操作员已经裁剪好的单工件图片。目标工件位于画面中心附近，允许旋转、偏移和其他工件的部分遮挡；不做背景分割、工业相机取流或自动工件类型识别。操作员在检测前明确选择已建库工件。

## 2. 总体架构

```text
Qt MainWindow
 ├─ AppConfig
 ├─ BackendProcessManager
 │   ├─ 发现已有服务
 │   └─ QProcess 按需启动/关闭自有服务
 └─ BackendClient（QTcpSocket）
          │ UTF-8 JSON Lines / 127.0.0.1:37651
          ▼
Python TcpJsonServer
 ├─ WorkpieceLibrary
 │   ├─ manifest 与模板事务
 │   └─ 模板特征缓存
 └─ OrientationClassifier
     ├─ PP-ShiTuV2 全局特征
     └─ ALIKED + LightGlue 软中心局部匹配
```

默认配置使用 `127.0.0.1:37651`。服务拒绝绑定非回环地址，不向局域网开放。Qt 启动时优先复用已有服务；只有连接失败时才启动 Python。该边界既允许模型跨 Qt 会话常驻，也保留双击 Qt 程序即可使用的操作体验。

## 3. 代码边界

### 3.1 Python

计划新增以下模块：

- `src/orientation_classifier.py`
  - 构造 Paddle 全局特征模型和 ALIKED/LightGlue 512 上限模型。
  - 提取查询图全局向量和局部特征。
  - 使用一个工件的 5+5 模板缓存完成融合决策。
  - 返回纯 Python 字典，不处理文件持久化、socket 或 JSON。
- `src/workpiece_library.py`
  - 校验工件名称和 5+5 图片集合。
  - 维护安全内部标识、`manifest.json`、模板目录和事务式覆盖。
  - 构建、恢复及查询模板缓存。
  - 不处理 Qt 或 TCP。
- `src/orientation_tcp_service.py`
  - 解析命令行配置，加载模型和模板库后监听回环端口。
  - 完成 JSON 行拆包、协议校验、错误映射、单客户端管理和命令分发。
  - 不包含分类公式或目录交换细节。

现有 `src/evaluate_global_local_fusion.py` 继续作为实验入口。生产分类器提取其中已经验证的最小逻辑，但不调用 `evaluate_dataset()`，也不使用随机模板划分或报告写入流程。

### 3.2 Qt

计划在 `qt_app/` 创建：

- `workpiece_orientation.pro`：Qt 5.14.2 Widgets、Network、Test 构建配置。
- `appconfig.h/.cpp`：读取并校验可执行文件旁的 JSON 配置。
- `backendclient.h/.cpp`：管理 `QTcpSocket`、JSON 行缓冲、请求标识、超时和响应信号。
- `backendprocessmanager.h/.cpp`：发现已有服务、按需启动 QProcess、记录所有权并执行显式重启/关闭。
- `mainwindow.h/.cpp/.ui`：建库与检测交互，不直接操作 socket 或 Python 进程。
- `main.cpp`：应用入口。
- `tests/`：使用 Qt Test 和假 TCP 服务验证客户端及窗口状态。

`MainWindow` 只订阅高层信号，例如 `backendReady()`、`requestFailed()`、`predictionReady()`；socket 分包和进程所有权不得进入窗口代码。

## 4. 配置

可执行文件旁提供 `app_config.json.example`：

```json
{
  "python_executable": "E:/python/anaconda3/envs/shitu/python.exe",
  "backend_script": "E:/Project/wang/pp_813/src/orientation_tcp_service.py",
  "project_root": "E:/Project/wang/pp_813",
  "model_dir": "E:/Project/wang/pp_813/third_party/models/shiru_rec/general_PPLCNetV2_base_pretrained_v1.0_infer",
  "library_dir": "E:/Project/wang/pp_813/runtime_library",
  "host": "127.0.0.1",
  "port": 37651,
  "startup_timeout_ms": 120000,
  "request_timeout_ms": 120000
}
```

Qt 对必填字段、端口范围和路径存在性做第一层校验；Python 对模型目录、模板目录写权限和监听地址做最终校验。配置只允许 `host=127.0.0.1`。示例配置可以包含开发机路径，真实 `app_config.json` 不提交硬编码部署假设。

Python 启动参数由 Qt 逐项传给 QProcess，不通过 shell 拼接，避免空格、中文和特殊字符造成命令解释问题。

## 5. 服务发现与生命周期

### 5.1 启动

1. Qt 加载配置并进入 `Connecting`。
2. `BackendClient` 尝试连接配置端口。
3. 连接成功后发送 `hello`；服务身份和协议版本正确才进入 `Ready`。
4. 初次连接失败时，`BackendProcessManager` 启动 Python 服务并设置 `ownedByThisSession=true`。
5. Python 加载模型、扫描模板库、恢复缓存，然后绑定端口。
6. Qt 在 120 秒内定期重连；握手成功后刷新工件列表。
7. 超时、QProcess 提前退出或握手错误进入 `Error`，建库和检测保持禁用。

若端口上存在非本服务程序，TCP 连接可能成功但 `hello` 失败。此时 Qt 报告端口冲突，不启动第二个服务。

### 5.2 单客户端

`TcpJsonServer` 的接收循环持续接受连接，并用活跃客户端锁保证只有一个完成握手的客户端。额外连接收到 `SERVER_BUSY` 后关闭。第一客户端断开后释放锁并允许新客户端连接。分类与建库请求在活跃客户端线程内串行执行，不并发访问 GPU 或模板事务。

### 5.3 断线、重启和关闭

连接异常时 Qt 立即：

- 清空上一张检测结果；
- 禁用建库和检测；
- 显示断线原因和“重启后端”按钮；
- 不自动循环重启。

用户点击重启后，Qt 先尝试重新连接。如果仍失败且服务为本会话启动，Qt 可终止无响应的自有 QProcess 后重新启动；外部服务不得被 Qt 终止，只提示用户处理外部进程。

Qt 退出时，若 `ownedByThisSession=true`，先发送 `shutdown` 并等待服务正常退出，超时后终止自有 QProcess；如果连接的是外部服务，只关闭 socket。

## 6. TCP JSON 行协议

### 6.1 帧与通用字段

- 每条消息是一个 UTF-8 JSON 对象，以单个 `\n` 结束。
- 接收端累积字节直到换行，必须支持半行和一次收到多行。
- 单行上限为 1 MiB，超出返回 `MESSAGE_TOO_LARGE` 并关闭连接。
- 每个请求包含 `version=1`、字符串 `request_id` 和 `command`。
- 每个响应回传相同 `request_id` 和布尔值 `ok`。
- Qt 同一时刻只允许一个在途业务请求；不匹配的请求标识视为协议错误。

错误格式：

```json
{
  "version": 1,
  "request_id": "12",
  "ok": false,
  "error": {
    "code": "IMAGE_UNREADABLE",
    "message": "无法读取图片"
  }
}
```

### 6.2 命令

握手：

```json
{"version":1,"request_id":"1","command":"hello"}
{"version":1,"request_id":"1","ok":true,"service":"workpiece-orientation","ready":true}
```

列出工件：

```json
{"version":1,"request_id":"2","command":"list_workpieces"}
{"version":1,"request_id":"2","ok":true,"workpieces":[{"id":"...","name":"M7"}]}
```

建库请求：

```json
{
  "version":1,
  "request_id":"3",
  "command":"register",
  "name":"M7",
  "replace":false,
  "front_images":["...5 paths..."],
  "back_images":["...5 paths..."]
}
```

检测请求：

```json
{
  "version":1,
  "request_id":"4",
  "command":"predict",
  "workpiece_id":"...",
  "image_path":"E:/检测图片/sample.png"
}
```

成功检测响应包含：

```json
{
  "version":1,
  "request_id":"4",
  "ok":true,
  "label":"front",
  "global_prediction":"front",
  "global_scores":{"front":0.91,"back":0.88},
  "global_margin":0.03,
  "local_prediction":"front",
  "local_scores":{"front":12.4,"back":5.1},
  "local_margin":7.3,
  "decision_source":"local_override",
  "needs_review":false,
  "elapsed_ms":248.5
}
```

`shutdown` 仅由确认拥有服务进程的 Qt 会话发送。服务先返回成功响应，再停止监听并退出。

### 6.3 错误码

首版稳定错误码包括：`INVALID_REQUEST`、`UNSUPPORTED_PROTOCOL_VERSION`、`MESSAGE_TOO_LARGE`、`SERVER_BUSY`、`INVALID_BIND_ADDRESS`、`PORT_IN_USE`、`CONFIG_INVALID`、`WORKPIECE_NOT_FOUND`、`WORKPIECE_EXISTS`、`INVALID_WORKPIECE_NAME`、`INVALID_TEMPLATE_SET`、`IMAGE_UNREADABLE`、`MODEL_ERROR` 和 `INTERNAL_ERROR`。

## 7. 模板库与缓存

### 7.1 目录结构

```text
runtime_library/
  <uuid>/
    manifest.json
    0/                 # front / 正面，5 张
    1/                 # back / 反面，5 张
```

`manifest.json` 保存 schema 版本、内部 UUID、显示名称、创建/更新时间和固定类别映射。显示名称用于 UI，UUID 用于请求和目录，不把用户文本拼入路径。同名检查使用去除首尾空白后的 Unicode 名称；Windows 下按大小写不敏感方式比较。

### 7.2 事务式注册

1. 在 `library_dir` 同级创建唯一临时目录。
2. 校验两组各 5 张、组内无重复、格式支持且 OpenCV 可读取。
3. 复制图片并写入清单。
4. 提取 10 张模板的全局向量和局部特征。
5. 新建时把临时目录改名为正式 UUID 目录。
6. 覆盖时先把旧目录改名为唯一备份，再把临时目录改名为正式目录。
7. 正式目录和缓存更新成功后删除备份；失败则恢复旧目录和旧缓存。

服务启动时清理无正式清单的临时目录；如果发现正式目录缺失但存在可验证的备份，则恢复备份并记录日志。

### 7.3 特征缓存

- 全局模板向量保存为 NumPy 数组。
- ALIKED 模板特征以 CPU tensor 保存，避免工件数量增加时长期占用 GPU 显存。
- 当前选中工件的 10 份局部特征在预测时搬到 GPU；切换工件时释放上一工件的 GPU 模板引用。
- 模型始终常驻 GPU。
- 服务启动时扫描并恢复所有完整工件的 CPU 缓存；损坏工件不进入 `list_workpieces`，错误写入日志但不阻止其他工件可用。

## 8. 推理与复检规则

固定参数：

```text
ROI_RATIO=1.0
MAX_NUM_KEYPOINTS=512       # 上限，不保证实际提取到 512
GLOBAL_MARGIN_THRESHOLD=0.05
LOCAL_MIN_SCORE=4.0
LOCAL_MIN_MARGIN=0.5
LOCAL_OVERRIDE_MARGIN=3.0
```

每张查询图只提取一次全局向量和局部特征。每类全局得分取 5 个模板中的最大余弦相似度；每类局部得分取 5 个模板中最大的软中心 RANSAC 得分。

局部最高分低于 4.0 或局部间隔低于 0.5 时，局部预测为 `uncertain`。全局间隔不大于 0.05，局部预测有效且局部间隔不小于 3.0 时采用局部标签，否则采用全局标签。

以下情况返回 `needs_review=true`：

- 全局间隔不大于 0.05 且局部预测为 `uncertain`；
- 全局预测与有效局部预测不一致。

原始得分和间隔不是校准概率，Qt 不显示百分比置信度。

## 9. Qt 界面状态

后端状态机：

```text
Disconnected → Connecting → StartingService → Handshaking → Ready
      ▲              │              │               │          │
      └──────────────┴──────────────┴───────────────┴── Error ◄┘
```

`Ready` 下可以进入 `Busy` 执行建库或检测，请求结束后回到 `Ready`。所有非 `Ready` 状态均禁用提交按钮。主窗口至少包含：后端状态与重启按钮、工件选择/刷新区、5+5 建库区、待测图片预览、检测按钮、标签结果、证据明细、耗时和复检提示。

## 10. 测试策略

### 10.1 Python 单元测试

- 使用假全局/局部模型验证融合边界和复检规则。
- 使用临时目录验证名称校验、5+5 校验、同名拒绝、事务覆盖失败回滚、清单恢复和缓存刷新。
- 使用回环临时端口验证 `hello`、请求标识、半包/多包、非法 UTF-8/JSON、超长消息、单客户端和断线后重新接受客户端。
- 注入会抛异常的分类器，验证单请求失败不会结束服务。

### 10.2 Qt Test

- 假 TCP 服务分块发送和连续发送多条响应，验证客户端缓冲。
- 验证请求标识不匹配、握手服务身份不符、`SERVER_BUSY` 和超时。
- 验证已有服务不启动 QProcess，连接失败才启动，且关闭时只结束自有服务。
- 验证断线清空结果、按钮禁用和用户显式重启。
- 验证配置缺失、非回环 host、非法端口和中文路径。

### 10.3 集成验证

- 在 `shitu` 环境启动真实 TCP 服务，执行中文路径建库与单图检测。
- 用固定 5+5 模板和固定样本核对 M1、M2、M7 服务化结果与现有 512 实验入口标签一致。
- 使用 Qt 5.14.2 MSVC2017 64 位 qmake 编译应用并完成完整操作冒烟测试。

## 11. 交付边界

交付内容包括 Python 服务、Qt 源码与 qmake 工程、示例配置、自动化测试和 Windows 使用说明。模型权重继续使用当前项目目录，不复制或转换。工业相机、远程访问、多客户端并发、模板删除和阈值独立校准属于后续变更。
