# 后端提前监听与模型加载状态实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with checkpoints.

**Goal:** 让 Python 服务在模型加载期间先监听端口，并让 Qt 显示“后端：模型加载中”且自动重试握手。

**Architecture:** 用线程安全运行时状态保存 `loading/ready/failed` 和分类器/工件库。TCP server 先绑定，再由后台线程加载模型；`hello` 在 loading 时返回非致命状态。Qt 将 `MODEL_LOADING` 作为可重试状态，不重复启动服务，ready 后沿用现有业务流程。

**Tech Stack:** Python 3.10、标准库 `threading/socket`、pytest、Qt 5.14.2/C++17、Qt Test、现有 PaddleClas + ALIKED/LightGlue。

## Global Constraints

- 服务只绑定 `127.0.0.1`。
- 模型加载期间不执行 `register` 和 `predict`。
- 外部已运行服务不由 Qt 终止。
- Qt 不因 `ready=false/status=loading` 重复启动 Python 进程。
- 真实配置的 `startup_timeout_ms` 继续保留为 600000 毫秒，作为慢机器兜底。

---

### Task 1: Python loading runtime and protocol regression

**Files:**
- Modify: `src/orientation_tcp_service.py`
- Test: `tests/test_orientation_tcp_service.py`

**Interfaces:**
- Add a thread-safe runtime object representing `loading`, `ready`, and `failed` states.
- `OrientationCommandDispatcher` accepts that runtime and returns `MODEL_LOADING` for business commands before readiness.
- `hello` returns `ok=true, ready=false, status=loading` while the runtime is loading and `ok=true, ready=true` when ready.

- [x] **Step 1: Write failing Python tests**

Add tests that create a delayed fake runtime, assert the listener accepts a connection before readiness, assert the first `hello` contains `ready=false/status=loading`, assert `list_workpieces` returns `MODEL_LOADING`, switch the runtime to ready and assert a later `hello` succeeds, and assert `shutdown` works while loading.

- [x] **Step 2: Run the focused tests and verify the expected failure**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_tcp_service.py -q
```

Expected: FAIL because the current dispatcher requires an already-created classifier/library and `hello` has no loading state.

- [x] **Step 3: Implement the minimal runtime state and server handshake changes**

Bind `OrientationTcpServer` before model construction. Make the dispatcher read the runtime state under a lock; keep `hello` and `shutdown` available while loading, reject business commands with `MODEL_LOADING`, and only enter the authenticated client loop when `hello.ready` is true. Start model loading in a daemon thread from `main()` and atomically publish the classifier/library or a startup error.

- [x] **Step 4: Run the focused Python tests and verify they pass**

Run the same pytest command; expected: all TCP service tests pass, including existing protocol and ownership tests.

### Task 2: Qt client and process-manager loading state

**Files:**
- Modify: `qt_app/backendclient.h`
- Modify: `qt_app/backendclient.cpp`
- Modify: `qt_app/backendprocessmanager.h`
- Modify: `qt_app/backendprocessmanager.cpp`
- Test: `qt_app/tests/test_backendclient.cpp`
- Test: `qt_app/tests/test_backendprocessmanager.cpp`

**Interfaces:**
- Add a `backendLoading(QString)` signal on `BackendProcessManager`.
- Treat a loading handshake as `MODEL_LOADING`, close/retry the current connection, and do not call `launchBackend()` again once `launchRequested_` is true.
- Preserve existing `HANDSHAKE_FAILED`, `SERVER_BUSY`, `TIMEOUT`, and external-service ownership behavior.

- [x] **Step 1: Write failing Qt tests**

Extend the fake TCP server to return a loading handshake first and a ready handshake on a later connection. Assert the client emits `MODEL_LOADING`; assert the process manager emits `backendLoading`, retries, starts the launcher only once, and eventually emits `backendReady`.

- [x] **Step 2: Run the focused Qt tests and verify the expected failure**

Run the existing Qt Test executable after rebuilding `qt_app/tests/test_backendclient.pro` and `test_backendprocessmanager.pro` with `QT_QPA_PLATFORM=offscreen`. Expected: FAIL because `ready=false` is currently treated as `HANDSHAKE_FAILED` and no loading signal exists.

- [x] **Step 3: Implement loading-aware handshake and retry**

When a valid service response has `ready=false` and `status=loading`, emit `requestFailed("MODEL_LOADING", message)`, abort only the current socket, and let the manager retry on its existing 500 ms retry timer. Add `backendLoading` and connect it to the main window later.

- [x] **Step 4: Run the focused Qt tests and verify they pass**

Run both rebuilt Qt Test executables; expected: all existing tests plus the loading-state cases pass.

### Task 3: Qt user-visible loading status

**Files:**
- Modify: `qt_app/mainwindow.cpp`
- Modify: `qt_app/mainwindow.h` only if a slot declaration is required
- Test: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Connect `BackendProcessManager::backendLoading` to a main-window slot/lambda.
- Display `后端：模型加载中` and a non-error explanatory message while loading; existing button-state rules keep registration and prediction disabled until ready.

- [x] **Step 1: Write the failing window test**

Emit the manager loading signal in the fake-client window fixture and assert the backend status label contains `模型加载中` and the registration/prediction buttons are disabled.

- [x] **Step 2: Run the focused window test and verify the expected failure**

Run the rebuilt `test_mainwindow.exe`; expected: FAIL because the main window has no loading-signal connection.

- [x] **Step 3: Implement the loading status connection**

Add the signal connection and keep error styling reserved for actual unavailable/configuration failures. Do not enable any operation until `onBackendReady` runs.

- [x] **Step 4: Run the focused window test and verify it passes**

Run `test_mainwindow.exe` with `QT_QPA_PLATFORM=offscreen`; expected: PASS.

### Task 4: Startup wiring, docs, and full verification

**Files:**
- Modify: `src/orientation_tcp_service.py` startup path if Task 1 leaves construction helpers
- Modify: `scripts/smoke_orientation_service.ps1`
- Modify: `docs/verification/workpiece-orientation-desktop-ui-checklist.md`
- Modify: `openspec/changes/workpiece-orientation-desktop-ui/design.md`
- Modify: `openspec/changes/workpiece-orientation-desktop-ui/specs/desktop-orientation-inspection/spec.md`

**Interfaces:**
- Real service startup must expose the port before model load and eventually pass the existing smoke sequence.
- Smoke script must retry loading handshakes for the configured 600-second cold-start budget.

- [x] **Step 1: Run the real service smoke test before changing its expectations**

Use the current smoke command and record the old behavior: it waits for model initialization before the first TCP connection.

- [x] **Step 2: Update the smoke probe for loading responses**

The probe must accept a connected socket whose first `hello` has `ready=false`, then send subsequent `hello` requests on that connection until `ready=true`, while preserving request-id and shutdown assertions.

- [x] **Step 3: Run the full verification set**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -m "not integration" -q
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_orientation_service.ps1 -StartupTimeoutSeconds 600
powershell -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1
openspec validate workpiece-orientation-desktop-ui --strict
```

Expected: Python tests, Qt tests/build, real hello/list/shutdown smoke, and strict OpenSpec validation all pass.

- [x] **Step 4: Clean up and review**

Check for temporary debug logs, verify no model/runtime artifacts are staged, run `git diff --check`, and record the measured loading behavior in the verification checklist.
