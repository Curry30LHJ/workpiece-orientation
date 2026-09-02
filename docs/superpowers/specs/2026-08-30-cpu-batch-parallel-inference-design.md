# CPU 批量并行推理与冻结包导入修复设计

日期：2026-08-30  
分支：`feature/20260830/cpu-inference-optimization`

## 1. 背景与目标

当前 CPU 版本有两个相互独立但会同时影响交付的问题：

1. 某些交付包以 PyInstaller 冻结后启动时找不到 `src.orientation_classifier`，导致后端在 Qt 中显示不可用。
2. Qt 批量检测逐张发送 `predict`，后端的全局推理锁又会把同一进程内的模型调用串行化，批量吞吐无法利用 i5-8250U 的多个物理核心。

本变更的目标是：

- 修复脚本运行、PyInstaller 冻结运行和从任意工作目录启动时的 `src.*` 导入问题；
- 增加向后兼容的 `predict_batch` 协议，使一批图片能够并行完成推理；
- 在 CPU 上提供可配置的独立推理会话池，默认优先测试 4 会话×1 线程，并在不适用时自动降级；
- 保持单张 `predict` 路径、PP-ShiTu 权重及预处理、几何规则、ALIKED/LightGlue、Ridge 和现有融合阈值不变；
- 以 i5-8250U 的“5 张不同图片、模型已预热、后端收到请求至返回结果”作为主要性能验收场景，记录 P50/P95，而不是未经实测承诺固定时延；
- 不因并行失败而静默丢图、重排结果或改变识别语义。

## 2. 约束与非目标

- 不重新训练、微调或替换 PP-ShiTu 模型。
- 本阶段不引入 ONNX Runtime、OpenVINO、INT8 或新特征模型；若并行后仍达不到目标，另立模型级优化变更。
- 不改变旧工件库、旧 5+5 模板缓存、正反面不等模板数量和几何规则数据格式。
- 不改变单张调用的返回字段和阈值；旧 Qt 或旧后端仍可使用逐张 `predict`。
- 不使用 8 个独立会话强行占满逻辑线程；4 个物理核心是默认并发上限，2×2 与 4×1 通过实机基准比较。
- 批量接口不得静默截断输入；空列表、非字符串路径和超过协议消息容量的请求必须显式返回错误。

## 3. 现状诊断

### 3.1 导入错误

`src/orientation_tcp_service.py` 以文件入口启动时会把仓库根目录加入 `sys.path`，然后使用 `from src...` 绝对导入。`src` 当前是隐式命名空间包，开发环境通常可以工作，但 PyInstaller 的静态模块图在不同构建入口、工作目录和收集配置下可能漏收子模块。现有冻结归档中虽然有正常包包含 `src.orientation_classifier`，但缺少构建时的显式契约，无法阻止交付了错误或过期的包。

### 3.2 批量瓶颈

- `OrientationCommandDispatcher` 只有单张 `predict` 命令。
- Qt `InspectionPage` 当前一次只发送一张图片，并等待该响应后再发送下一张。
- `OrientationClassifier._global_embeddings` 和 `_extract_local` 共用 `_inference_lock`，即使外围创建线程，也会在模型调用处排队。
- 模板快照是不可变的，适合被多个推理会话共享；模板全局特征不应在每张查询图上重算。

## 4. 方案比较

### 方案 A：单会话批量输入

把 5 张图作为一个 Paddle batch 交给现有 predictor。实现最简单、内存最少，但已有 CPU 基准显示 batch size 增大不能稳定降低单图耗时，不能解决当前锁和 Qt 逐张往返问题。作为没有并行池时的兼容回退。

### 方案 B：进程内独立推理会话池（采用）

为批量请求创建有限数量的独立 Paddle predictor，会话之间并行处理不同图片；模板快照和分类逻辑只读共享。CPU 默认先验证 4×1 和 2×2 两种配置，池不可用时退回单会话批量或逐张路径。该方案不改变模型和结果语义，改动集中在后端调度及 Qt 协议适配，最适合当前离线包。

### 方案 C：多个后端进程

每个进程拥有一个完整模型，隔离性最好，但启动时间、内存、端口管理、打包和 Qt 生命周期复杂；在确认方案 B 无法满足实机目标前不采用。

## 5. 总体架构

```text
Qt 选择 N 张图片
    -> hello 能力协商
    -> predict_batch(workpiece_id, image_paths)
    -> 后端校验请求并捕获一个不可变 catalog 快照
    -> 并行解码/预处理
    -> BatchInferencePool 分发到独立 Paddle 会话
    -> 复用同一 cache 的既有 FastOrientationEngine/分类逻辑
    -> 按输入索引还原结果
    -> 返回逐项结果 + 阶段耗时 + 降级信息
```

### 5.1 导入和冻结包边界

- 新增 `src/__init__.py`，把业务模块声明为显式包。
- 入口统一通过运行时根目录解析导入；冻结模式使用 PyInstaller 的运行时目录，不依赖当前工作目录。
- `deploy/orientation_backend.spec` 使用明确的 `src.*` hidden-import 列表（至少包含 `orientation_tcp_service`、`orientation_classifier`、`workpiece_catalog`、`workpiece_library`、`runtime_data`、`geometry_*`、`fast_*` 及其实际依赖）。
- 构建脚本在生成 exe 后检查归档中每个必需模块的完整名称，并从临时、非仓库工作目录启动一次 `hello`/自检；检查失败时构建失败。
- 服务启动日志记录包版本、构建指纹、入口路径和 `src` 模块来源，便于识别 Qt 指向了旧包。

### 5.2 推理会话池

新增职责单一的 `BatchInferencePool`（建议放在 `src/parallel_inference.py`）：

- `submit_many(items, predict_one) -> list[BatchItemResult]`：提交有序任务，返回顺序严格等于输入顺序；
- 每个 worker 持有独立、不可并发复用的 Paddle predictor/RecPredictor；优先使用当前 Paddle 版本支持的 `clone()`/`PredictorPool`，不能安全克隆时创建独立会话；
- worker 不直接调用主 classifier 的 `_global_embeddings`。每个会话绑定一个轻量 worker classifier/engine（共享只读的校准器、模板快照和配置，独立绑定全局 predictor；需要 legacy/compare 时同时独立绑定局部 extractor/matcher），从结构上绕开主 classifier 的 `_inference_lock`；无法建立独立绑定时必须降级，不能通过移除锁冒险并发调用同一 predictor；
- worker 使用 1 个数学线程，池大小由物理核心数和配置共同限制，默认最多 4；
- 池初始化和一次性预热在后端 ready 前完成，或在 `hello` 中明确报告 `batch_ready=false`，避免第一次批量请求把建池耗时伪装成推理耗时；
- 会话创建失败、内存不足或运行时异常时记录明确原因，关闭池并使用串行回退，不修改输入和输出；
- 使用有界队列，单批只创建有限任务，不创建无界线程；同一 catalog 快照可安全供多个 worker 读取；
- 同时到达单张和批量请求时由调度器保证 predictor 不被交叉使用：批量占用池，单张使用主会话；不允许通过增加线程造成 MKL/Paddle 过度并行。

主会话仍保留现有 CPU 线程配置（默认 4）用于单张请求。批量池默认 `workers=4, threads_per_worker=1`，同时支持 `workers=2, threads_per_worker=2` 作为配置/基准候选。GPU 不启用该 CPU 池，继续使用原有路径。

### 5.3 预测语义

`WorkpieceCatalog` 增加批量入口，逐项捕获同一快照和 `library_revision`，调用既有 `classifier.predict_with_cache`。批量并行只改变调用调度，不改变：

- 图像读取和几何处理的语义；
- 模板向量、局部特征、几何规则和忽略区域；
- 全局/局部融合顺序、阈值、候选选择和返回字段。

如果未来需要共享 query 解码缓存，必须以图像路径和文件修改信息为键，并在文件变化时失效；本变更先不引入持久化查询缓存，避免陈旧结果。

## 6. 协议设计

### 6.1 `hello` 能力

在现有响应中增加非破坏性字段：

```json
{
  "capabilities": {
    "predict_batch": true,
    "batch_ready": true,
    "batch_workers": 4,
    "batch_threads_per_worker": 1
  }
}
```

旧客户端忽略新增字段；池未就绪或降级时 `predict_batch` 可以保留为 `true`，但 `batch_ready=false`，Qt 必须显示状态并选择兼容路径。

### 6.2 `predict_batch` 请求与响应

请求：

```json
{
  "version": 1,
  "request_id": "...",
  "command": "predict_batch",
  "workpiece_id": "...",
  "image_paths": ["a.png", "b.png", "c.png"]
}
```

成功响应保持输入顺序：

```json
{
  "version": 1,
  "request_id": "...",
  "ok": true,
  "items": [
    {"index": 0, "image_path": "a.png", "ok": true, "prediction": {}},
    {"index": 1, "image_path": "b.png", "ok": false,
     "error": {"code": "IMAGE_UNREADABLE", "message": "..."}}
  ],
  "batch_timings_ms": {
    "decode": 0.0,
    "inference": 0.0,
    "postprocess": 0.0,
    "total": 0.0
  },
  "worker_count": 4,
  "fallback": null
}
```

至少一项成功时返回 `ok=true` 并逐项报告失败；空列表、工作件不存在或请求结构无效返回稳定的顶层错误。任何失败项都不能导致其他项被静默丢弃或重新排序。旧后端不认识该命令时，Qt 自动拆分为原有 `predict` 请求。

## 7. Qt 行为

- `BackendClient` 在握手后保存能力字段。
- 批量检测优先发送一个 `predict_batch`；收到响应后按 `index` 更新已有记录和结果表，点击任意图片仍显示对应原图和结果。
- 后端返回逐项错误时，Qt 将该项标为失败并继续显示其他结果；不会因一张坏图阻断整批。
- `batch_ready=false`、旧后端或协议错误时，界面明确显示“批量并行不可用，已切换兼容模式”，随后使用逐张请求。
- 批量按钮显示处理中、已完成数量、总耗时和实际 worker 数；不把 Qt 绘制耗时计入后端算法指标。

## 8. 错误处理与回退

1. 启动导入失败：记录完整模块名和包路径，运行时返回 `RUNTIME_SELF_CHECK_FAILED`，不显示模糊的“未连接”。
2. 池预热失败：服务仍可 ready，`hello.batch_ready=false`，单张和串行批量可用；日志包含异常类型、会话数和线程配置。
3. 单个图片读取/推理失败：只影响对应 `items[index]`，错误码沿用现有分类错误映射。
4. 批量请求取消或连接断开：停止向连接写入结果，释放未完成任务；不会修改工件库和模板缓存。
5. 线程/会话配置非法：回退到安全的 2×2，再失败回退单会话，并在响应 `fallback` 中说明。

## 9. 测试与验收

### 9.1 导入和打包回归

- 从仓库根目录、临时目录和冻结 exe 所在目录启动入口，均能完成 `hello`。
- PyInstaller 归档审计断言所有生产 `src.*` 模块存在；错误入口或缺失模块必须让构建测试失败。
- 交付包在没有仓库源码、没有当前工作目录依赖时可启动；日志包含实际 exe 路径和版本指纹。

### 9.2 后端单元/集成测试

- 空列表、非字符串路径、未知工件返回稳定错误。
- 5 个带序号的 fake 任务并发执行时，观测到最大活动 worker 大于 1，且返回顺序为 0..4。
- 一个任务失败不会丢失或重排其余任务。
- 池不可用时明确走串行回退；单张 `predict` 的现有测试全部继续通过。
- 同一 `WorkpieceCatalog` 快照在并发调用期间 revision 不变；旧 5+5 库、1+1、5+10、10+15 和超过 30 张模板均可恢复并预测。
- `hello` 能力协商覆盖新后端、旧后端和 `batch_ready=false` 三种情况。

### 9.3 Qt 测试

- 新后端批量响应按输入顺序更新记录和右侧结果；点击记录显示对应图片。
- 单项失败、整批回退和旧后端均有可见状态，批量按钮不会永久禁用。
- 现有逐张批量测试保持通过。

### 9.4 性能验收

在 i5-8250U、固定 CPU 电源策略、固定模型和 5 张不同图片上：

1. 模型和会话池预热 50 次；
2. 连续执行至少 200 个 5 图批次；
3. 分别记录后端端到端、纯模型、解码/预处理、后处理和序列化耗时；
4. 报告均值、P50、P95、P99、最大值、吞吐（图/秒）、worker/线程配置和 CPU 温度/降频情况（若可取得）。

主要性能目标是稳态 5 图批次 P95 ≤ 100 ms；这是目标机器上的实测门槛，不在开发机上预先承诺。若未达到，报告真实瓶颈并保留可复现基线，不以增加逻辑线程或未验证的模型转换宣称达标。单张性能和首次启动/预热时间单独报告。

## 10. 发布与回滚

- 先在当前特性分支提交设计、测试和实现，再构建 CPU 免安装包并执行冻结归档审计。
- 通过环境变量或配置关闭批量池即可恢复单会话路径；Qt 能力协商也提供自动回退，因此不需要回滚工件库。
- 若实机内存、稳定性或准确率不满足要求，保留 `predict_batch` 协议但默认关闭池，继续交付已有单张 CPU 版本；模型级优化另行立项。
