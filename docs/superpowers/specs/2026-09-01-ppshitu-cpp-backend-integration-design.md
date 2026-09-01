# PP-ShiTuV2 原生 C++ 特征服务接入设计

日期：2026-09-01  
分支：`feature/20260901/ppshitu-cpp-inference`  
基线：现有 Python TCP 后端、`native/ppshitu_rec_benchmark` 基准程序

## 1. 背景

当前产品后端已经在 `fast_geometry` 路径中使用 Python Paddle `RecPredictor` 提取 PP-ShiTuV2 全局特征。几何规则、模板缓存、Ridge 分类、工件库恢复和 Qt TCP 协议均依赖该特征接口。此前新增的原生 C++ 程序只用于离线基准：它可以加载相同的 `general_PPLCNetV2_base_pretrained_v1.0_infer` 模型并输出 512 维归一化特征，但不监听请求、不了解工件库，也没有接入后端。因此，当前需要一个可回滚、可测量的“原生特征服务”接入，用于在目标 CPU 上验证吞吐和端到端效果。

本设计只替换全局特征提取实现，不把整个后端重写为 C++。这样可以隔离语言/运行时的收益，同时保留已经验证的几何规则和业务语义。

## 2. 目标与非目标

### 2.1 目标

1. 在 Windows x64 上提供长生命周期的 C++ PP-ShiTu 特征服务，并由 Python 后端按需调用。
2. 单图和批量请求都复用已启动的模型进程，避免每张图重复加载模型。
3. 保证 C++ 与现有 Python 路径的预处理、特征维度、归一化、输出顺序和最终分类语义一致。
4. 批量池中的每个 worker 使用独立 C++ 服务进程，避免并发调用同一个 Paddle predictor。
5. 提供显式的 `python`/`native_cpp` 选择、稳定错误码、超时和进程退出处理；原生模式失败时不得静默回退。
6. 保持现有工件库、模板缓存、几何规则、Ridge 参数、融合阈值、Qt TCP API 和旧库恢复兼容。
7. 记录传输、C++ 预处理、模型执行和后处理耗时，使“是否变快”可以被复现实验验证。

### 2.2 非目标

- 不修改 PP-ShiTu、ALIKED、LightGlue 或现有融合阈值。
- 不重新训练、量化或更换模型。
- 不在本次接入中改变几何规则的含义、模板缓存格式或标签决策。
- 不把原生服务强制设为默认；默认仍为 Python，便于现有部署回滚。
- 不把 C++ 服务改造成完整工件库/TCP 业务后端。
- 不承诺未经 i5-8250U 实测的 P95 目标；开发机数据只能作为诊断依据。

## 3. 采用的架构

```text
Qt (现有 TCP 协议)
        |
Python orientation_tcp_service
        |
OrientationClassifier / FastOrientationEngine
        |  (RGB uint8 batch)
        +--> Python RecPredictor       [pp_backend=python]
        |
        +--> NativePPClient ----------> ppshitu_rec_service.exe
                                      (一个长期存活的 predictor)
```

- `OrientationClassifier` 保留现有公共预测接口；调用方不感知特征来自 Python 还是 C++。
- `pp_backend=python` 完全走当前实现。
- `pp_backend=native_cpp` 只在 `compute_device=cpu` 且 `inference_mode=fast_geometry` 下启用。其余组合在启动时明确拒绝，并返回可读错误，不做隐式降级。
- C++ 服务由 Python 子进程管理。标准输入/输出承载协议，标准错误承载日志，避免日志污染二进制数据流。单图预测复用一个客户端；批量池为每个 slot 建立一个客户端和一个服务进程。
- C++ 服务只接收已经完成业务裁剪的 RGB 图像，不自行定位工件，不访问工件库和几何规则。

## 4. C++ 服务程序

### 4.1 命令行

在现有 `native/ppshitu_rec_benchmark` 目录新增服务目标（可与基准目标共享预处理和特征提取代码）：

```text
ppshitu_rec_service.exe --serve \
  --model-dir <model directory> \
  --threads <positive integer, default 1>
```

启动时必须检查模型目录中的 `inference.pdmodel`/`inference.pdiparams`（以及新旧 Paddle 命名兼容项）、CPU 指令集和输出维度；检查失败时写入错误并以非零码退出。服务进程不得在请求期间重新加载模型。

### 4.2 二进制帧协议（v1）

使用 little-endian、无文本日志混入的 framed stream。所有整数为无符号定长整数，浮点为 IEEE-754 little-endian `float32`。每个帧均包含完整长度，服务端对长度设上限；上限必须覆盖当前最大配置批次（建议默认至少 256 MiB，并允许通过启动参数显式调小/调大），超限或截断立即返回错误并关闭连接。Windows 管道以二进制模式打开，HELLO 和每个 RESULT 写完后必须显式 flush。

帧头：

```text
magic[4]      = "PPSH"
version[u16]  = 1
kind[u16]     = HELLO(1) | PREDICT(2) | RESULT(3) | ERROR(4) | CLOSE(5)
payload_len[u32]
request_id[u64]
```

HELLO（服务启动后主动发送一次）至少包含：协议版本、模型 SHA-256、特征维度、最大批量、线程数和服务版本。Python 收到并校验后才把后端状态置为 ready。

PREDICT payload：

```text
count[u32]
repeat count times:
  width[u32], height[u32], channels[u16] (=3), data_len[u32]
  data[data_len]  # 连续 RGB uint8，行优先
```

RESULT payload：

```text
count[u32], dimension[u32]
preprocess_ms[f32], inference_ms[f32], postprocess_ms[f32]
embedding[count * dimension]  # 行优先、每行 L2 归一化
```

结果顺序必须与请求顺序相同；`count=0`、尺寸为零、通道数不是 3、数据长度不匹配、非有限 embedding 都是错误。ERROR payload 包含稳定错误码、短消息和可选诊断字段；错误帧只对应当前 `request_id`，协议错误或服务退出则客户端将连接标记为不可复用。

建议错误码：`PROTOCOL_VERSION_MISMATCH`、`FRAME_TOO_LARGE`、`INVALID_IMAGE`、`MODEL_LOAD_FAILED`、`CPU_FEATURE_UNSUPPORTED`、`NATIVE_PP_INFERENCE_FAILED`、`NATIVE_PP_SERVICE_EXITED`、`NATIVE_PP_TIMEOUT`、`NATIVE_PP_DIMENSION_MISMATCH`。

### 4.3 服务循环和并发

- 服务主线程顺序读取帧、校验并调度；单个 predictor 只在其拥有的执行线程中调用。
- 一个服务实例内部默认串行处理请求；批量并发通过多个服务实例实现。若以后启用服务内部线程，必须重新做 embedding 一致性和内存峰值验证。
- CLOSE 或 EOF 触发清理并返回；父进程终止时 Python 先发送 CLOSE，再在有限等待后强制回收子进程。
- stdout 只能写协议帧；启动诊断和 Paddle 日志全部写 stderr，并由 Python 截取最近一段用于错误提示。服务端不得把异常文本直接写入 stdout；未捕获异常也必须转换为 ERROR 帧后再退出。

## 5. Python 后端接入

### 5.1 配置和参数

为 `orientation_tcp_service.py` 增加：

```text
--pp-backend python|native_cpp       # 默认 python
--native-pp-executable <path>       # native_cpp 必填
```

`OrientationClassifier.load` 增加同名配置项。原生模式仅创建 `NativePPClient`，不创建 Python Paddle predictor；Python 模式的导入和初始化路径保持不变。启动 hello 元数据增加：`pp_backend`、`native_service_version`、`native_model_sha256`、`feature_dim` 和能力列表，供 Qt“已连接”状态和诊断页显示。

### 5.2 客户端边界

新增窄接口（例如 `src/native_pp_client.py`）：

- `start()`：启动服务、读取 HELLO、校验协议/模型/维度并设置 ready。
- `predict(rgb_images)`：接受 RGB `uint8` 的单图或 batch，发送一个 PREDICT 帧，返回按输入顺序排列的 `float32` embedding 和分阶段耗时。
- `close()`：幂等发送 CLOSE、关闭管道并回收子进程。

客户端必须使用单写者/单读者锁，给每个请求设置有限超时；超时、EOF、坏帧、非零退出或维度变化转换为上述稳定错误码，并保留 stderr 尾部。请求失败后不得继续复用已损坏的进程。

`OrientationClassifier._global_embeddings` 继续负责 BGR→RGB 的现有语义，随后把 RGB 数组交给所选 backend。C++ 客户端不再二次颜色转换；输出仍按当前 Python 路径执行相同的 L2 归一化约定（若 C++ 已归一化，Python 仅做有限值/范数校验，避免重复改变数值）。几何规则、Ridge 和结果对象不变。

### 5.3 批量池

`prepare_batch_pool` 的 session factory 根据 backend 创建独立 `_BatchSession`：

- Python backend：保持现有独立 RecPredictor 逻辑。
- native backend：每个 slot 启动一个 `NativePPClient`/服务进程；slot 关闭时回收对应进程。

批量任务仍由现有池负责分片、并发和按索引合并；不得在共享 predictor 上加“看似安全”的线程锁来代替隔离。`close_batch_pool` 和后端统一关闭路径必须保证所有 native 子进程最终退出，重复关闭不报错。

### 5.4 失败语义和回滚

- `native_cpp` 启动失败、模型指纹/维度不符、服务崩溃或超时：后端返回明确错误，任务状态为失败/不可用；绝不悄悄切换到 Python，以免性能和诊断结果被掩盖。
- 运维可把 `pp_backend` 改回 `python` 并重启后端，无需重建工件库或模板缓存。
- 原有模型源目录指纹仍用于库和缓存兼容；native 模型 SHA 只作为运行时诊断字段。不得因 C++ staging 目录不同而使旧 5+5 或任意数量模板库失效。

## 6. Qt 配置与界面

在 `qt_app/appconfig.*` 增加可选字段：

```json
{
  "pp_backend": "python",
  "native_pp_executable": ".../ppshitu_rec_service.exe"
}
```

- 缺省或未知字段按配置错误处理；缺省 backend 使用 `python`。
- `BackendProcessManager` 仅在选择 `native_cpp` 时转发两个参数，并在可执行文件不存在时在启动前给出路径错误。
- Qt TCP 协议不变；连接状态显示“后端已连接”必须以 Python 后端收到 C++ HELLO 并完成模型校验为准，而不是以子进程已创建为准。
- 详情页显示 backend、特征维度、模型指纹和最近一次 native 错误，但不要求显示局部匹配点。

## 7. 兼容性与安全边界

1. 现有 Python 默认路径、旧工件库、模板缓存、几何规则草稿/发布版本和 Qt 请求字段必须继续工作。
2. 只允许本机父子进程通信；不开放额外 TCP 端口，不接受来自网络的任意命令。
3. 帧长度、图像尺寸、批量数量和单请求等待时间均有上限；错误消息不得回显任意大块图像数据。
4. 模型路径和可执行文件路径必须经过现有配置校验并记录规范化绝对路径；禁止从请求内容改变模型路径。
5. 服务退出、父进程退出和 Qt 重启都必须清理子进程，避免残留占用模型文件或端口。

## 8. 测试与验收标准

### 8.1 先行的契约测试

- HELLO、PREDICT、RESULT、ERROR、CLOSE 的帧编解码、版本和长度校验。
- 空 batch、坏尺寸、截断帧、超时、服务提前退出和维度不符均得到稳定错误码。
- 多请求和多 worker 的输入/输出顺序、request_id 对应关系和进程清理。

### 8.2 后端回归

- `pp_backend=python` 的现有测试全部通过。
- native 与 Python 对同一固定图片的 embedding 余弦相似度 ≥ 0.9999、最大绝对误差 ≤ 0.001、维度和有限值完全一致。
- 在 M1、M2、M7 及当前另外两个数据集上，最终正反面标签、复核标志、几何规则应用状态和模板数量结果与 Python 基线一致；1+1、5+10、10+15、超过 30 张模板库均可恢复和预测。
- 重启/关闭后无 native 服务残留；切回 Python 不需要重建库。

### 8.3 性能验收

分别在开发机和 i5-8250U 上测量单图、5 图和 95 图批次，至少报告 P50/P95/P99：

1. C++ 预处理；
2. 模型执行；
3. Python↔C++ 传输；
4. Python 后处理；
5. 完整 `predict_batch` wall time。

模型加载和首次启动单独报告，不混入热推理。并行配置至少覆盖 1、2、4 个 worker；同时记录 CPU 利用率和峰值内存。结论必须区分“模型计算变快”和“批量吞吐提高”，不得用平均值替代 P95，也不得以提高复核率或关闭几何规则换取速度。只有目标机真实 P95 达标后，才能对甲方宣称达标。

## 9. 实施顺序与回滚点

1. 先实现协议编解码和不依赖 Paddle 的契约测试。
2. 将现有 C++ 预处理/特征提取抽成可复用库，增加 `--serve` 目标和启动握手。
3. 实现 Python `NativePPClient`、单图接入和错误映射；运行 embedding/标签回归。
4. 接入批量 session factory 和完整关闭路径；运行并发、重启和泄漏测试。
5. 增加 Qt 配置转发和诊断显示；验证默认 Python 与显式 native 两条路径。
6. 在两类硬件上采集性能报告，再决定是否将 native 作为交付包的可选后端。

每一步都保留可运行的 Python 默认路径。若任何正确性或稳定性门禁失败，只需将配置设为 `pp_backend=python` 并重启；不修改或迁移现有工件库、模板缓存和几何规则。

## 10. 待实施文件范围

预计新增/修改（实施阶段再按实际代码结构调整）：

- `native/ppshitu_rec_benchmark/`：服务入口、共享协议、服务循环和构建目标；
- `src/native_pp_client.py`、`src/orientation_classifier.py`、`src/orientation_tcp_service.py`：后端选择、生命周期和批量池；
- `qt_app/appconfig.*`、`qt_app/backendprocessmanager.*`、示例配置：字段校验和参数转发；
- `tests/`：协议、生命周期、回归和性能测量测试；
- `docs/verification/`：目标硬件实测报告和依赖指纹清单。

任何不属于上述接入边界的算法、规则或 UI 行为变化不在本设计内。
