---
change: workpiece-orientation-desktop-ui
design-doc: docs/superpowers/specs/2026-08-15-workpiece-orientation-desktop-ui-design.md
base-ref: 416f46fd10b053500a35abf3fff91459f66fdd2d
---

# 工件正反面 Qt 桌面应用实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 把现有 PP-ShiTuV2 + ALIKED/LightGlue 512 关键点实验代码封装成可持久化少样本模板的本地 TCP 推理服务，并交付可在 Qt 5.14.2/MSVC2017_64 上构建的 C++ Widgets 应用。

**架构：** Python 端由 `OrientationClassifier`、`WorkpieceLibrary` 和仅监听回环地址的 JSON Lines TCP 服务组成；Qt 端通过 `QTcpSocket` 复用或连接由 `QProcess` 按需启动的服务。工件建库采用正反面各 5 张图和事务式目录替换，预测返回原始全局/局部证据、决策来源与人工复检标志。

**技术栈：** Python 3、PaddlePaddle/PaddleClas、PyTorch、ALIKED、LightGlue、OpenCV、NumPy、pytest；Qt 5.14.2 Widgets/Network/Test、qmake、MSVC2019（兼容 msvc2017_64 Qt 包）、jom。

## 全局约束

- Qt 固定使用 `E:/QT/5.14/5.14.2/msvc2017_64`，不引入 Qt 6 API。
- TCP 只允许 `127.0.0.1`，默认端口 `37651`，协议版本固定为 `1`，每帧为单行 UTF-8 JSON，最大 1 MiB。
- 服务同一时刻只允许一个完成握手的客户端；额外客户端返回 `SERVER_BUSY`。
- 工件显示名不直接作为目录名；内部目录使用 UUID，固定 `0=front`、`1=back`，每面必须恰好 5 张可读且不重复的图片。
- 推理固定 `ROI_RATIO=1.0`、`MAX_NUM_KEYPOINTS=512`、`GLOBAL_MARGIN_THRESHOLD=0.05`、`LOCAL_MIN_SCORE=4.0`、`LOCAL_MIN_MARGIN=0.5`、`LOCAL_OVERRIDE_MARGIN=3.0`。
- 原始得分不是概率；Qt 不显示“百分比置信度”。
- Qt 先连接并握手已有服务，失败后才启动 Python；只关闭本次 Qt 会话拥有的 Python 进程，断线后不自动循环重启。
- 所有 Python 业务测试必须可注入假模型，不依赖真实权重；真实 GPU/权重只在集成验证使用。

---

### Task 1：可复用的融合分类器

**对应 OpenSpec：** 1.1

**文件：**
- Create: `src/orientation_classifier.py`
- Create: `tests/test_orientation_classifier.py`
- Reuse: `src/aliked_lightglue_matcher.py`
- Reuse: `src/soft_center_matcher.py`
- Reuse: `src/shitu_baseline.py`

**接口：**
- Produces: `TemplateCache`、`OrientationClassifier.build_template_cache()`、`set_template_cache()`、`predict()`。
- `predict(workpiece_id: str, image_path: Path) -> dict` 返回 `label/global_prediction/global_scores/global_margin/local_prediction/local_scores/local_margin/decision_source/needs_review/elapsed_ms`。

- [ ] **Step 1：先写融合边界和缓存复用失败测试**

```python
def test_low_global_margin_is_overridden_by_decisive_local_evidence(tmp_path):
    classifier, counters = make_fake_classifier(
        global_result=("back", {"front": .88, "back": .91}, .03),
        local_scores={"front": 12.4, "back": 5.1},
    )
    classifier.set_template_cache("m7", fake_cache())
    result = classifier.predict("m7", write_image(tmp_path / "测试.png"))
    assert result["label"] == "front"
    assert result["decision_source"] == "local_override"
    assert result["needs_review"] is True  # 有效局部预测与全局预测冲突
    assert counters["template_extract"] == 0

def test_weak_local_evidence_keeps_global_and_requests_review(tmp_path):
    classifier, _ = make_fake_classifier(
        global_result=("front", {"front": .90, "back": .88}, .02),
        local_scores={"front": 3.9, "back": 3.0},
    )
    classifier.set_template_cache("m7", fake_cache())
    result = classifier.predict("m7", write_image(tmp_path / "q.png"))
    assert result["label"] == "front"
    assert result["local_prediction"] == "uncertain"
    assert result["needs_review"] is True
```

- [ ] **Step 2：运行测试并确认因模块不存在而失败**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py -v`

Expected: FAIL，包含 `ModuleNotFoundError: src.orientation_classifier`。

- [ ] **Step 3：实现固定阈值、模型注入和 CPU/GPU 特征边界**

```python
@dataclass(frozen=True)
class TemplateCache:
    global_vectors: dict[str, np.ndarray]
    local_features: dict[str, list[dict[str, torch.Tensor]]]

class OrientationClassifier:
    def __init__(self, global_predictor, extractor, matcher, device):
        raise NotImplementedError
    @classmethod
    def load(cls, project_root: Path, model_dir: Path) -> "OrientationClassifier":
        raise NotImplementedError
    def build_template_cache(
        self, front_paths: Sequence[Path], back_paths: Sequence[Path]
    ) -> TemplateCache:
        raise NotImplementedError
    def set_template_cache(self, workpiece_id: str, cache: TemplateCache) -> None:
        raise NotImplementedError
    def remove_template_cache(self, workpiece_id: str) -> None:
        raise NotImplementedError
    def predict(self, workpiece_id: str, image_path: Path) -> dict[str, object]:
        raise NotImplementedError
```

实现要点必须精确为：ALIKED 以 `build_models(512)` 创建；整图以 `extract_features(image, self.extractor, self.device, roi_ratio=1.0)` 提取；局部模板保存为递归 `.detach().cpu()` 后的 tensor；预测时仅将当前工件 10 份局部特征送到 GPU；每类全局/局部得分取 5 个模板最大值；局部有效条件为最高分 `>=4.0` 且间隔 `>=0.5`；仅当全局间隔 `<=0.05` 且有效局部间隔 `>=3.0` 时局部覆盖。全局与有效局部冲突，或低全局间隔且局部不确定时，`needs_review=True`。

- [ ] **Step 4：补充正常全局决策、未知工件和不可读图片测试并运行**

```python
def test_confident_global_result_is_kept_even_when_local_differs(fake_classifier, query_path):
    result = fake_classifier(global_margin=.20, local_scores={"front": 4.0, "back": 9.0}).predict("m7", query_path)
    assert result["label"] == result["global_prediction"]
    assert result["decision_source"] == "global"

def test_unknown_workpiece_raises_workpiece_not_found(classifier, query_path):
    with pytest.raises(WorkpieceNotFoundError):
        classifier.predict("missing", query_path)

def test_unreadable_query_raises_image_unreadable(classifier, tmp_path):
    with pytest.raises(ImageUnreadableError):
        classifier.predict("m7", tmp_path / "missing.png")

def test_build_template_cache_extracts_each_of_ten_templates_once(classifier, template_paths, counters):
    cache = classifier.build_template_cache(template_paths[:5], template_paths[5:])
    assert counters == {"global": 10, "local": 10}
    assert cache.global_vectors["front"].shape[0] == 5
    assert len(cache.local_features["back"]) == 5
```

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py tests/test_global_local_fusion.py tests/test_soft_center_matcher.py -v`

Expected: PASS。

- [ ] **Step 5：提交分类器**

```powershell
git add src/orientation_classifier.py tests/test_orientation_classifier.py
git -c user.name=Codex -c user.email=codex@local commit -m "feat: add reusable orientation classifier"
```

---

### Task 2：事务式工件库与缓存恢复

**对应 OpenSpec：** 1.2、1.3

**文件：**
- Create: `src/workpiece_library.py`
- Create: `tests/test_workpiece_library.py`

**接口：**
- Consumes: `TemplateCache` 与回调 `build_cache(front_paths, back_paths)`。
- Produces: `WorkpieceRecord`、`WorkpieceLibrary.recover()`、`list_workpieces()`、`register()`。

- [ ] **Step 1：写名称、5+5、同名和回滚失败测试**

```python
def test_register_creates_uuid_manifest_and_fixed_label_folders(tmp_path):
    library = WorkpieceLibrary(tmp_path / "库")
    record, cache = library.register("M7", five_images(tmp_path, "正"),
                                     five_images(tmp_path, "反"), False, fake_builder)
    assert UUID(record.id)
    manifest = json.loads((tmp_path / "库" / record.id / "manifest.json").read_text("utf-8"))
    assert manifest["name"] == "M7"
    assert len(list((tmp_path / "库" / record.id / "0").iterdir())) == 5
    assert len(list((tmp_path / "库" / record.id / "1").iterdir())) == 5

def test_failed_replace_preserves_old_directory_and_cache(tmp_path):
    library = populated_library(tmp_path)
    old = library.get("M7")
    with pytest.raises(FeatureBuildError):
        library.register("M7", front_new, back_new, True, raising_builder)
    assert library.get("M7").id == old.id
    assert old.template_names == library.get("M7").template_names
```

- [ ] **Step 2：运行并确认失败**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_library.py -v`

Expected: FAIL，模块或接口尚不存在。

- [ ] **Step 3：实现记录、校验、事务提交与恢复**

```python
@dataclass(frozen=True)
class WorkpieceRecord:
    id: str
    name: str
    root: Path
    front_images: tuple[Path, ...]
    back_images: tuple[Path, ...]

class WorkpieceLibrary:
    def __init__(self, library_dir: Path):
        raise NotImplementedError
    def recover(self, build_cache: CacheBuilder) -> list[tuple[WorkpieceRecord, TemplateCache]]:
        raise NotImplementedError
    def list_workpieces(self) -> list[dict[str, str]]:
        raise NotImplementedError
    def get(self, workpiece_id: str) -> WorkpieceRecord:
        raise NotImplementedError
    def register(self, name, front_images, back_images, replace, build_cache):
        raise NotImplementedError
```

事务顺序固定为：规范化名称并做 Windows 大小写不敏感查重；验证每面恰好 5 个不同的 `.png/.jpg/.jpeg/.bmp` 且 OpenCV 可读；在 `library_dir` 内创建 `.staging-<uuid>`；复制为确定性文件名；写完整 `manifest.json`；对 staging 中的 10 张图构建缓存；覆盖时将旧目录改名 `.backup-<uuid>`；staging 改名为正式 UUID；更新内存索引后删除备份。任一步失败均删除 staging，并恢复旧目录和旧索引。`recover()` 清理无清单 staging；正式目录缺失且存在可验证 backup 时恢复 backup；损坏工件跳过并记录日志。

- [ ] **Step 4：补充恢复、大小写同名、重复路径、不可读图和损坏工件隔离测试**

```python
@pytest.mark.parametrize("name", ["", "  ", "a/b", "a\\b", "a\x00b"])
def test_invalid_names_are_rejected(name, tmp_path, ten_images):
    with pytest.raises(InvalidWorkpieceNameError):
        WorkpieceLibrary(tmp_path / "lib").register(name, ten_images[:5], ten_images[5:], False, fake_builder)

def test_windows_casefold_name_collision_requires_replace(populated_library, ten_images):
    with pytest.raises(WorkpieceExistsError):
        populated_library.register("m7", ten_images[:5], ten_images[5:], False, fake_builder)

def test_duplicate_template_path_is_rejected(tmp_path, ten_images):
    front = [ten_images[0]] * 5
    with pytest.raises(InvalidTemplateSetError):
        WorkpieceLibrary(tmp_path / "lib").register("M7", front, ten_images[5:], False, fake_builder)

def test_recover_skips_corrupt_record_but_loads_valid_record(library_with_valid_and_corrupt_records):
    recovered = library_with_valid_and_corrupt_records.recover(fake_builder)
    assert [record.name for record, _ in recovered] == ["M7"]

def test_recover_rebuilds_each_valid_cache_once(library_with_two_records, counting_builder):
    recovered = library_with_two_records.recover(counting_builder)
    assert len(recovered) == 2
    assert counting_builder.calls == 2
```

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_workpiece_library.py -v`

Expected: PASS。

- [ ] **Step 5：提交工件库**

```powershell
git add src/workpiece_library.py tests/test_workpiece_library.py
git -c user.name=Codex -c user.email=codex@local commit -m "feat: add transactional workpiece library"
```

---

### Task 3：版本化 TCP JSON 服务

**对应 OpenSpec：** 1.4、1.5

**文件：**
- Create: `src/orientation_tcp_service.py`
- Create: `tests/test_orientation_tcp_service.py`

**接口：**
- Consumes: `OrientationClassifier`、`WorkpieceLibrary`。
- Produces: `JsonLineConnection`、`OrientationCommandDispatcher`、`OrientationTcpServer` 和命令行入口。

- [ ] **Step 1：先写协议拆包、握手和稳定错误码测试**

```python
def test_partial_and_multiple_json_lines_are_decoded(socket_pair):
    socket_pair.client.sendall(b'{"version":1,"request_id":"1","command":"hel')
    socket_pair.client.sendall(b'lo"}\n{"version":1,"request_id":"2","command":"list_workpieces"}\n')
    assert socket_pair.server_reader.read_message()["request_id"] == "1"
    assert socket_pair.server_reader.read_message()["request_id"] == "2"

def test_bad_json_does_not_stop_following_request(client):
    client.send_raw(b'{bad}\n')
    assert client.read()["error"]["code"] == "INVALID_REQUEST"
    assert client.request("hello")["ok"] is True

def test_wrong_version_returns_unsupported_protocol_version(client):
    response = client.request("hello", version=2)
    assert response["error"]["code"] == "UNSUPPORTED_PROTOCOL_VERSION"

def test_second_client_receives_server_busy(active_client, server_address):
    second = TcpTestClient(server_address)
    assert second.request("hello")["error"]["code"] == "SERVER_BUSY"
```

- [ ] **Step 2：运行并确认失败**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_tcp_service.py -v`

Expected: FAIL，模块尚不存在。

- [ ] **Step 3：实现 JSON 行传输与命令分发**

```python
PROTOCOL_VERSION = 1
SERVICE_NAME = "workpiece-orientation"
MAX_MESSAGE_BYTES = 1024 * 1024

class JsonLineConnection:
    def read_message(self) -> dict[str, object] | None:
        raise NotImplementedError
    def send(self, payload: Mapping[str, object]) -> None:
        raise NotImplementedError

class OrientationCommandDispatcher:
    def dispatch(self, request: Mapping[str, object]) -> dict[str, object]:
        raise NotImplementedError

class OrientationTcpServer:
    def serve_forever(self) -> None:
        raise NotImplementedError
    def request_shutdown(self) -> None:
        raise NotImplementedError
```

`dispatch()` 必须先验证 `request_id`、`version`、`command`，再处理 `hello/list_workpieces/register/predict/shutdown`。所有响应回传相同 `request_id` 和 `version=1`。预期业务异常映射为规格中的稳定错误码；未知异常记录 traceback，仅向客户端返回 `INTERNAL_ERROR`。单个请求失败后继续读取下一行。

- [ ] **Step 4：实现单活跃客户端、关闭顺序与 CLI**

服务 accept 循环为每个连接启动守护线程，但用锁保证只有首个完成 `hello` 的连接成为活跃客户端；额外连接收到 `SERVER_BUSY` 后关闭。业务命令只在已握手活跃连接中串行执行。`shutdown` 先发送成功响应，再设置停止事件、关闭监听 socket 并退出。CLI 参数精确为：

```text
--host 127.0.0.1 --port 37651 --project-root PATH --model-dir PATH --library-dir PATH
```

非 `127.0.0.1` 在创建模型前即以 `INVALID_BIND_ADDRESS` 退出；端口占用报告 `PORT_IN_USE`。

- [ ] **Step 5：补充注册、预测、超长消息、非法 UTF-8、断线重连和请求失败存活测试**

```python
def test_register_and_predict_responses_preserve_request_id(client, valid_register_payload):
    response = client.request_json({**valid_register_payload, "request_id": "register-7"})
    assert response["request_id"] == "register-7" and response["ok"] is True
    response = client.request("predict", request_id="predict-8", workpiece_id=response["workpiece"]["id"], image_path=valid_register_payload["front_images"][0])
    assert response["request_id"] == "predict-8" and response["label"] in {"front", "back"}

def test_message_over_one_mib_returns_message_too_large_and_closes(client):
    client.send_raw(b"{" + b"x" * (1024 * 1024) + b"}\n")
    assert client.read()["error"]["code"] == "MESSAGE_TOO_LARGE"
    assert client.wait_closed()

def test_invalid_utf8_returns_invalid_request(client):
    client.send_raw(b"\xff\n")
    assert client.read()["error"]["code"] == "INVALID_REQUEST"

def test_first_disconnect_allows_a_new_client(active_client, server_address):
    active_client.close()
    replacement = TcpTestClient(server_address)
    assert replacement.request("hello")["ok"] is True

def test_classifier_exception_returns_model_error_and_server_survives(client, fake_classifier):
    fake_classifier.predict.side_effect = RuntimeError("gpu")
    assert client.request("predict", workpiece_id="m7", image_path="q.png")["error"]["code"] == "MODEL_ERROR"
    assert client.request("list_workpieces")["ok"] is True

def test_shutdown_response_arrives_before_server_exits(client, running_server):
    assert client.request("shutdown")["ok"] is True
    running_server.join(timeout=2)
    assert not running_server.is_alive()
```

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py tests/test_workpiece_library.py tests/test_orientation_tcp_service.py -v`

Expected: PASS。

- [ ] **Step 6：提交 Python 服务**

```powershell
git add src/orientation_tcp_service.py tests/test_orientation_tcp_service.py
git -c user.name=Codex -c user.email=codex@local commit -m "feat: expose orientation inference over loopback tcp"
```

---

### Task 4：Qt qmake 工程、配置与主窗口骨架

**对应 OpenSpec：** 2.1

**文件：**
- Create: `qt_app/workpiece_orientation.pro`
- Create: `qt_app/app_config.json.example`
- Create: `qt_app/appconfig.h`
- Create: `qt_app/appconfig.cpp`
- Create: `qt_app/main.cpp`
- Create: `qt_app/mainwindow.h`
- Create: `qt_app/mainwindow.cpp`
- Create: `qt_app/mainwindow.ui`
- Create: `qt_app/tests/test_appconfig.cpp`
- Create: `qt_app/tests/test_appconfig.pro`

**接口：**
- Produces: `AppConfig::load()` 与可启动的 Qt Widgets 主窗口。

- [ ] **Step 1：先写配置解析失败测试**

```cpp
void TestAppConfig::rejectsNonLoopbackHost() {
    const QString path = writeConfig(QJsonObject{{"host", "0.0.0.0"}, {"port", 37651}});
    QString error;
    const auto config = AppConfig::load(path, &error);
    QVERIFY(!config.has_value());
    QVERIFY(error.contains(QStringLiteral("127.0.0.1")));
}

void TestAppConfig::acceptsChinesePaths() { /* 写入完整合法 JSON，并逐字段 QCOMPARE */ }
```

- [ ] **Step 2：创建 qmake 配置并确认测试先失败**

应用 `.pro` 使用 `QT += widgets network`、`CONFIG += c++17`；测试 `.pro` 使用 `QT += testlib network`。运行：

```powershell
New-Item -ItemType Directory -Force qt_app\build-test-appconfig | Out-Null
Set-Location qt_app\build-test-appconfig
E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe ..\tests\test_appconfig.pro
E:\QT\5.14\Tools\QtCreator\bin\jom.exe
```

Expected: 编译失败，因为 `AppConfig` 尚未实现。

- [ ] **Step 3：实现配置对象和最小主窗口布局**

```cpp
struct AppConfig {
    QString pythonExecutable;
    QString backendScript;
    QString projectRoot;
    QString modelDir;
    QString libraryDir;
    QHostAddress host;
    quint16 port = 37651;
    int startupTimeoutMs = 120000;
    int requestTimeoutMs = 120000;
    static std::optional<AppConfig> load(const QString &path, QString *error);
};
```

校验所有必填路径非空且存在、host 仅 `127.0.0.1`、端口为 1..65535、超时为正数。`mainwindow.ui` 必须包含：后端状态、重启按钮、工件列表与刷新按钮、名称输入、正反面模板选择、建库按钮、待测图预览、检测按钮、结果/证据/耗时/复检区域；初始时建库和检测禁用。

- [ ] **Step 4：运行配置测试和应用编译**

Run: 上述 qmake/jom 命令，再对 `qt_app/workpiece_orientation.pro` 建独立 `build-release` 目录运行 qmake/jom。

Expected: `test_appconfig.exe -txt` PASS，应用链接成功。

- [ ] **Step 5：提交 Qt 骨架**

```powershell
git add qt_app
git -c user.name=Codex -c user.email=codex@local commit -m "feat: scaffold qt5 desktop application"
```

---

### Task 5：QTcpSocket 协议客户端

**对应 OpenSpec：** 2.2、2.6 的协议部分

**文件：**
- Create: `qt_app/backendclient.h`
- Create: `qt_app/backendclient.cpp`
- Create: `qt_app/tests/faketcpserver.h`
- Create: `qt_app/tests/faketcpserver.cpp`
- Create: `qt_app/tests/test_backendclient.cpp`
- Create: `qt_app/tests/test_backendclient.pro`
- Modify: `qt_app/workpiece_orientation.pro`

**接口：**
- Produces: `BackendClient::connectToService()`、`sendRequest()`、`disconnectFromService()` 和高层信号。

- [ ] **Step 1：写半行/多行、握手身份、请求关联和超时测试**

```cpp
void TestBackendClient::buffersPartialAndMultipleResponses();
void TestBackendClient::rejectsWrongServiceIdentity();
void TestBackendClient::rejectsMismatchedRequestId();
void TestBackendClient::emitsServerBusy();
void TestBackendClient::timesOutOneOutstandingRequest();
void TestBackendClient::disconnectClearsPendingRequest();
```

`FakeTcpServer` 使用 `QTcpServer`，允许测试逐字节发送或一次发送多行 JSON，并保留收到的请求供断言。

- [ ] **Step 2：运行并确认测试失败**

Run: 在独立构建目录对 `test_backendclient.pro` 执行 qmake、jom，再运行 `debug\test_backendclient.exe -txt`。

Expected: 编译失败，因为客户端类尚未实现。

- [ ] **Step 3：实现状态机和 JSON 行缓冲**

```cpp
class BackendClient : public QObject {
    Q_OBJECT
public:
    enum class State { Disconnected, Connecting, Handshaking, Ready, Busy, Error };
    void connectToService(const QHostAddress &host, quint16 port);
    QString sendRequest(const QString &command, const QJsonObject &fields = {});
    void disconnectFromService();
signals:
    void stateChanged(State state, const QString &detail);
    void handshakeSucceeded();
    void responseReceived(const QString &command, const QJsonObject &response);
    void requestFailed(const QString &code, const QString &message);
    void connectionLost(const QString &reason);
};
```

`readyRead` 将字节追加到 `QByteArray`，按 `\n` 循环切帧，保留最后半行；每行按 UTF-8 解析为对象。连接后自动发送 `hello`，只在 `service=workpiece-orientation`、`version=1`、`ready=true` 时进入 Ready。同一时刻仅允许一个业务请求；响应必须匹配 pending `request_id`。任何协议错、超时或断线清空 pending、进入 Error 并发出高层信号。

- [ ] **Step 4：运行全部 BackendClient Qt Test**

Expected: 所列测试 PASS，并在每个测试后断言没有悬挂 socket 或计时器。

- [ ] **Step 5：提交 TCP 客户端**

```powershell
git add qt_app/backendclient.* qt_app/tests/faketcpserver.* qt_app/tests/test_backendclient.* qt_app/tests/test_backendclient.pro qt_app/workpiece_orientation.pro
git -c user.name=Codex -c user.email=codex@local commit -m "feat: add qt tcp backend client"
```

---

### Task 6：后端进程发现、所有权与显式重启

**对应 OpenSpec：** 2.3、2.6 的生命周期部分

**文件：**
- Create: `qt_app/backendprocessmanager.h`
- Create: `qt_app/backendprocessmanager.cpp`
- Create: `qt_app/processlauncher.h`
- Create: `qt_app/processlauncher.cpp`
- Create: `qt_app/tests/test_backendprocessmanager.cpp`
- Create: `qt_app/tests/test_backendprocessmanager.pro`
- Modify: `qt_app/workpiece_orientation.pro`

**接口：**
- Consumes: `AppConfig`、`BackendClient`。
- Produces: `BackendProcessManager::start()`、`restart()`、`shutdownOwnedService()`、`ownedByThisSession()`。

- [ ] **Step 1：写服务复用、按需启动、所有权关闭和失败状态测试**

```cpp
void TestBackendProcessManager::reusesExistingServiceWithoutLaunchingProcess();
void TestBackendProcessManager::launchesPythonOnlyAfterInitialConnectionFailure();
void TestBackendProcessManager::startupTimesOutAfterConfiguredDeadline();
void TestBackendProcessManager::restartReconnectsBeforeReplacingOwnedProcess();
void TestBackendProcessManager::shutdownDoesNotTerminateExternalService();
void TestBackendProcessManager::shutdownRequestsExitForOwnedService();
```

通过 `IProcessLauncher` 假实现记录启动、终止和进程退出，不在单元测试中真正启动 Python。

- [ ] **Step 2：运行并确认测试失败**

Run: 对 `test_backendprocessmanager.pro` 执行 qmake/jom 和 `-txt` 测试。

Expected: 编译失败，管理器接口尚不存在。

- [ ] **Step 3：实现按需启动和所有权规则**

```cpp
class BackendProcessManager : public QObject {
    Q_OBJECT
public:
    void start();
    void restart();
    void shutdownOwnedService();
    bool ownedByThisSession() const;
signals:
    void backendReady();
    void backendUnavailable(const QString &reason);
};
```

`start()` 先连接并握手；只有连接失败才使用 `QProcess::setProgram()` 与 `setArguments()` 启动，不通过 shell 拼接。启动后按短间隔重连，最长 120 秒。若 TCP 可连但 hello 身份错误，报告端口冲突且不得启动第二服务。断线只报告不可用，不自动启动循环。`restart()` 先重连；仍失败且进程为本会话拥有时才终止无响应进程并重启；外部服务不得终止。退出时仅拥有服务发送 `shutdown`，超时后才结束自有 QProcess。

- [ ] **Step 4：运行管理器测试并验证所有权矩阵**

Expected: 六个场景 PASS；外部服务路径中的 launcher terminate/start 调用数均为 0。

- [ ] **Step 5：提交生命周期管理器**

```powershell
git add qt_app/backendprocessmanager.* qt_app/processlauncher.* qt_app/tests/test_backendprocessmanager.* qt_app/tests/test_backendprocessmanager.pro qt_app/workpiece_orientation.pro
git -c user.name=Codex -c user.email=codex@local commit -m "feat: manage local backend service lifecycle"
```

---

### Task 7：建库与单张检测界面

**对应 OpenSpec：** 2.4、2.5、2.6 的窗口状态部分

**文件：**
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp`
- Modify: `qt_app/mainwindow.ui`
- Create: `qt_app/tests/test_mainwindow.cpp`
- Create: `qt_app/tests/test_mainwindow.pro`

**接口：**
- Consumes: `BackendProcessManager` 的可用状态与 `BackendClient` 的响应信号。
- Produces: 完整建库、检测、复检提示和断线清理交互。

- [ ] **Step 1：先写模板数量、Busy、断线清理和结果展示测试**

```cpp
void TestMainWindow::requiresExactlyFiveUniqueImagesPerSide();
void TestMainWindow::confirmsBeforeSendingReplaceTrue();
void TestMainWindow::busyStateDisablesRegisterAndPredict();
void TestMainWindow::disconnectClearsPreviousPredictionAndEnablesRestart();
void TestMainWindow::predictionShowsRawEvidenceWithoutPercentConfidence();
void TestMainWindow::needsReviewShowsProminentChineseWarning();
```

使用注入的假 manager/client 驱动信号；通过 `findChild()` 按稳定 objectName 检查按钮、标签和文本。

- [ ] **Step 2：运行并确认窗口测试失败**

Run: 对 `test_mainwindow.pro` 执行 qmake/jom；设置 `QT_QPA_PLATFORM=offscreen` 后运行测试。

Expected: FAIL，所需控件行为尚未实现。

- [ ] **Step 3：实现建库交互**

正/反面各使用一次多选文件对话框，只接受 PNG/JPEG/BMP；界面显示 5 个文件名并在提交前再次校验绝对路径去重。工件名为空或数量错误时本地提示，不发送请求。先发送 `replace=false`；若后端返回 `WORKPIECE_EXISTS`，使用 `QMessageBox` 明确确认后重新发送 `replace=true`。成功后刷新 `list_workpieces` 并选择新工件。

- [ ] **Step 4：实现检测、证据与状态清理**

检测前必须已选工件且图片可由 `QImageReader` 读取；预览按控件尺寸等比缩放。响应中将 `front/back` 显示为“正面/反面”，分别展示全局正反得分、全局间隔、局部正反得分、局部间隔、决策来源和毫秒耗时；不得加 `%`。`needs_review=true` 时显示“建议人工复检”。所有非 Ready/Busy 状态禁用提交；断线、进程退出、协议错误时立即清空结果与旧预览状态，并显示“重启后端”按钮。

- [ ] **Step 5：运行窗口测试和完整应用构建**

Expected: Qt Test PASS；`workpiece_orientation.exe` 链接成功，启动时缺少真实 `app_config.json` 会显示配置错误且不崩溃。

- [ ] **Step 6：提交完整界面**

```powershell
git add qt_app/mainwindow.* qt_app/tests/test_mainwindow.* qt_app/tests/test_mainwindow.pro
git -c user.name=Codex -c user.email=codex@local commit -m "feat: add registration and inspection workflows"
```

---

### Task 8：Python 与 TCP 真实环境集成验证

**对应 OpenSpec：** 3.1、3.2

**文件：**
- Create: `tests/test_orientation_service_integration.py`
- Create: `scripts/smoke_orientation_service.ps1`
- Create: `reports/orientation_desktop_ui/.gitkeep`（若报告目录由测试运行时创建，则不提交运行结果）

**接口：**
- Consumes: 真实 `shitu` 环境、现有 M1/M2/M7 数据和当前模型目录。
- Produces: 可重复的中文路径回环冒烟命令和标签一致性证据。

- [ ] **Step 1：写标记为 integration 的真实服务测试**

```python
@pytest.mark.integration
@pytest.mark.parametrize("dataset_name", ["1_M1", "1_M2", "1_M7"])
def test_service_label_matches_existing_512_pipeline(dataset_name, fixed_registration_split):
    expected = fixed_registration_split.experiment_label(dataset_name)
    actual = fixed_registration_split.service_label(dataset_name)
    assert actual["label"] == expected
```

测试从每类固定取 5 张模板建立临时库，待测样本必须与模板排除；临时库和中文测试图片路径使用 pytest 临时目录，不污染正式 `runtime_library`。

- [ ] **Step 2：先运行快速 Python 单元测试**

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -m "not integration" -v`

Expected: PASS，且不加载真实 Paddle/ALIKED 权重。

- [ ] **Step 3：实现 PowerShell 冒烟脚本并运行真实集成测试**

脚本使用 `Start-Process -WindowStyle Hidden` 启动服务，等待 hello，依次执行中文路径建库、list、predict、shutdown；在 `finally` 中仅停止脚本自己启动且仍存活的进程。不得按名称结束所有 Python 进程。

Run: `E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_service_integration.py -m integration -v`

Expected: M1/M2/M7 服务标签与现有 512 入口一致，TCP 中文路径冒烟通过。

- [ ] **Step 4：提交集成测试与脚本**

```powershell
git add tests/test_orientation_service_integration.py scripts/smoke_orientation_service.ps1
git -c user.name=Codex -c user.email=codex@local commit -m "test: verify production orientation service"
```

---

### Task 9：Qt 5.14.2 构建、Qt Test 与手工验收

**对应 OpenSpec：** 3.3、3.4

**文件：**
- Create: `scripts/build_qt5.ps1`
- Create: `docs/verification/workpiece-orientation-desktop-ui-checklist.md`

- [ ] **Step 1：实现固定工具链构建脚本**

脚本校验以下路径并失败即退出：

```text
E:/QT/5.14/5.14.2/msvc2017_64/bin/qmake.exe
E:/QT/5.14/Tools/QtCreator/bin/jom.exe
C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat
```

在 `qt_app/build-release` 与每个 `qt_app/tests/build-*` 目录中调用 qmake/jom；然后设置 `QT_QPA_PLATFORM=offscreen` 运行全部 Qt Test。脚本不得删除项目根目录或任何数据目录，仅清理明确的 Qt build 子目录。

- [ ] **Step 2：运行 Qt 自动化构建**

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1`

Expected: 应用和所有 Qt Test 构建成功，Qt Test 全部 PASS。

- [ ] **Step 3：按所有权和异常矩阵执行手工验收并记录结果**

验收表逐项记录日期、命令/操作、预期、实际和证据路径，至少覆盖：已有服务复用、连接失败后按需启动、5+5 新建、同名覆盖取消/确认、单图正反结果、人工复检提示、断线清空旧结果、显式重启、外部服务不被关闭、自有服务退出、第二客户端 `SERVER_BUSY`、错误服务占用端口、后端异常退出。

- [ ] **Step 4：提交构建脚本和验收表**

```powershell
git add scripts/build_qt5.ps1 docs/verification/workpiece-orientation-desktop-ui-checklist.md
git -c user.name=Codex -c user.email=codex@local commit -m "test: add qt5 build and acceptance checks"
```

---

### Task 10：Windows 工控机交付说明与最终回归

**对应 OpenSpec：** 3.5

**文件：**
- Create: `docs/workpiece-orientation-desktop-ui-windows.md`
- Modify: `openspec/changes/workpiece-orientation-desktop-ui/tasks.md`

- [ ] **Step 1：编写部署和操作文档**

文档必须给出：`shitu` 环境校验；Paddle GPU、PyTorch CUDA、PaddleClas、LightGlue 依赖检查；模型目录结构；`app_config.json.example` 复制与字段解释；Python 服务单独启动；Qt 构建；Qt 应用启动；正反各 5 张建库；单图检测；复检含义；端口占用、`SERVER_BUSY`、模型加载失败、中文路径、Qt 平台插件和 GPU 显存故障排查。明确说明真实 `app_config.json`、数据集、权重、运行库和实验报告不提交 Git。

- [ ] **Step 2：运行完整回归**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -m "not integration" -v
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_service_integration.py -m integration -v
powershell -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1
openspec validate workpiece-orientation-desktop-ui --strict
```

Expected: Python 单元与集成测试、Qt 构建与 Qt Test、OpenSpec 严格校验全部 PASS。

- [ ] **Step 3：逐项勾选 OpenSpec tasks 并核对工作区**

仅在对应验收证据存在时将 `tasks.md` 的 1.1–3.5 改为 `[x]`。运行 `git status --short`，确认没有数据、模型、运行模板库、构建产物或真实配置被暂存。

- [ ] **Step 4：提交文档与任务状态**

```powershell
git add docs/workpiece-orientation-desktop-ui-windows.md openspec/changes/workpiece-orientation-desktop-ui/tasks.md
git -c user.name=Codex -c user.email=codex@local commit -m "docs: add windows deployment guide"
```

---

## 计划自查

- **规格覆盖：** Python 分类、5+5 事务模板库、缓存恢复、回环 TCP/单客户端/稳定错误码、Qt 服务复用与所有权、建库、检测、复检、断线重启、M1/M2/M7 一致性、Qt5 构建和 Windows 文档均有对应任务。
- **占位符扫描：** 无 `TBD`、`TODO`、`implement later` 或省略的测试体；`tuple[Path, ...]` 中的省略号仅是 Python 变长元组类型语法。算法、接口、命令和验收结果均已明确。
- **类型一致性：** `workpiece_id` 全链路使用字符串 UUID；协议请求/响应使用 JSON 对象；Qt 请求标识为 `QString`；Python 缓存接口在分类器与工件库之间统一为 `TemplateCache`。
