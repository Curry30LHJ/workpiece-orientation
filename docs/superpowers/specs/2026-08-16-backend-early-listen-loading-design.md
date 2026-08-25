# 后端提前监听与模型加载状态设计

## 目标

让 Python 后端在模型加载期间先监听 loopback TCP 端口，Qt 显示“模型加载中”，避免首次加载 PaddleClas、ALIKED/LightGlue 超过启动超时后被误判为连接失败。

## 约束

- 服务仍只绑定 `127.0.0.1`。
- 现有 `hello`、`list_workpieces`、`register`、`predict`、`shutdown` 命令保持兼容。
- 模型加载期间不接受建库和检测请求。
- Qt 不重复启动同一会话的 Python 服务。
- 外部已运行服务仍由外部进程负责，Qt 不主动终止。

## Python 服务架构

新增线程安全的运行时状态，状态为 `loading`、`ready` 或 `failed`，并持有完成加载后的分类器和工件库。主线程先创建 TCP listener，再启动后台加载线程；加载线程完成模型初始化和模板库恢复后原子地切换到 `ready`。

握手响应约定如下：

```json
{"version":1,"request_id":"...","ok":true,"service":"workpiece-orientation","ready":false,"status":"loading","message":"模型加载中"}
```

加载完成后继续返回现有的 `ready=true` 响应。加载失败时返回 `ok=false`、`ready=false` 和 `MODEL_LOAD_FAILED` 错误。除 `hello` 和 `shutdown` 外，`loading` 状态下的命令返回 `MODEL_LOADING`。

服务端在收到 `ready=false` 的握手后保持连接循环，允许客户端重新发送 `hello`；收到 `ready=true` 后才进入已认证业务会话。

## Qt 状态机

`BackendClient` 将 `ready=false/status=loading` 转换为 `MODEL_LOADING` 事件，不把它当作致命握手错误。`BackendProcessManager` 收到该事件时只安排 500 ms 后重试握手，不再次启动 Python 进程，并发出 `backendLoading` 信号。`MainWindow` 显示“后端：模型加载中”，禁用建库、刷新和检测按钮；握手成功后恢复现有流程。

模型加载失败仍进入“后端不可用”，并显示后端返回的错误详情。现有服务所有权和退出清理规则不变。

## 测试与验收

- Python TCP 测试验证：listener 在加载完成前可连接；加载状态握手；加载状态命令拒绝；加载完成后握手成功；加载失败错误；loading 期间 shutdown。
- Qt 测试验证：loading 握手不触发重复启动；按 500 ms 重试；ready 后只触发一次 `backendReady`；界面显示加载状态并禁用操作。
- 真实服务冒烟验证：`hello/list_workpieces/shutdown` 通过，且不再依赖 120 秒内完成模型初始化。

