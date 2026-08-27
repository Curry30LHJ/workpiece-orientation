# Backend Status and Batch Layout Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the managed-backend lifecycle shown by the Qt shell and make the batch-inspection result pane usable from 1024×640 through 1920×1080 without changing backend or recognition behavior.

**Architecture:** Extend the shared backend presentation vocabulary, then make `MainWindow` the only component that reduces client/process events into that vocabulary. Keep transport retry behavior unchanged. Make `InspectionPage` responsive through splitter minimums, compact-mode presentation based on the actual result-pane width, and a table-first batch review layout while preserving record-ID-driven selection.

**Tech Stack:** C++17, Qt 5.14.2 Widgets/Network/Test, qmake, MSVC 2019, PowerShell, existing Python pytest suite.

## Global Constraints

- Keep Qt 5.14.2 compatibility; do not use APIs introduced after Qt 5.14.
- Do not change backend process arguments, the 500 ms retry interval, startup timeout, `BackendClient` protocol, or request serialization.
- Do not modify PP-ShiTu, ALIKED, LightGlue, geometry rules, fusion thresholds, prediction fields, template caches, or workpiece data.
- Preserve existing Qt object names, public signals, record IDs, batch filtering, confirmation, auto-advance, and failure-retention behavior.
- Interpret 440 px and 520 px as Qt logical pixels; Windows DPI scaling must not be applied a second time.
- Do not add third-party dependencies or retrain a model.
- Follow TDD for each behavior change and commit only the files owned by that task.

---

## File Structure

- Modify `qt_app/appheader.h` — add `Starting` and `Recovering` to the shared presentation state.
- Modify `qt_app/appheader.cpp` — render distinct startup/recovery copy and severity.
- Modify `qt_app/mainwindow.h` — own the stable backend presentation phase and readiness history.
- Modify `qt_app/mainwindow.cpp` — reduce client/process events through one presentation method and ignore transient managed-startup failures.
- Modify `qt_app/inspectionpage.ui` — table-first batch review layout, two-row filters, horizontal confirmation actions, and a 440 px result-pane minimum.
- Modify `qt_app/inspectionpage.h` — declare compact-layout and elided-target helpers.
- Modify `qt_app/inspectionpage.cpp` — react to actual result-pane width, hide/restore the elapsed column, and preserve full target text in a tooltip.
- Modify `qt_app/resources/theme.qss` — add the compact result-title rule.
- Modify `qt_app/tests/test_appfoundation.cpp` — verify the new shared lifecycle labels.
- Modify `qt_app/tests/test_mainwindow.cpp` — reproduce retry churn and recovery sequences.
- Modify `qt_app/tests/test_inspectionpage.cpp` — verify narrow/wide batch geometry, visible actions, columns, row selection, and optional screenshot capture.
- Create `docs/verification/backend-status-and-batch-layout-fix-results.md` — record exact final Qt/Python/build/visual evidence.
- Create `docs/verification/screenshots/qt-ui-inspection-batch-1085x752.png` — reproducible narrow batch layout evidence.

---

### Task 1: Add Stable Startup and Recovery Presentation States

**Files:**
- Modify: `qt_app/appheader.h:17-25`
- Modify: `qt_app/appheader.cpp:18-31,170-211`
- Test: `qt_app/tests/test_appfoundation.cpp`

**Interfaces:**
- Consumes: existing `BackendUiState`, `BackendStatusDetails`, and `AppHeader::setBackendDetails(const BackendStatusDetails &)`.
- Produces: `BackendUiState::Starting` and `BackendUiState::Recovering`; both are consumed by Task 2.

- [ ] **Step 1: Write the failing lifecycle-label test**

Add this slot to `TestAppFoundation`:

```cpp
void backendLifecycleLabelsDistinguishStartupLoadingAndRecovery() {
    AppHeader header;
    auto *label = header.findChild<QLabel *>(QStringLiteral("backendStatusLabel"));
    QVERIFY(label != nullptr);

    header.setBackendState(BackendUiState::Starting, QStringLiteral("正在启动服务"));
    QVERIFY(label->text().contains(QStringLiteral("正在启动")));
    QVERIFY(!label->text().contains(QStringLiteral("模型加载中")));

    header.setBackendState(BackendUiState::Loading, QStringLiteral("正在加载权重"));
    QVERIFY(label->text().contains(QStringLiteral("模型加载中")));

    header.setBackendState(BackendUiState::Recovering, QStringLiteral("连接暂时中断"));
    QVERIFY(label->text().contains(QStringLiteral("正在重连")));
    QCOMPARE(label->property("messageKind").toString(), QStringLiteral("warning"));
}
```

- [ ] **Step 2: Run the focused test target and verify RED**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_appfoundation
```

Expected: compilation fails because `BackendUiState::Starting` and `BackendUiState::Recovering` do not exist.

- [ ] **Step 3: Extend the shared enum and header rendering**

Replace the enum in `appheader.h` with:

```cpp
enum class BackendUiState {
    Disconnected,
    Starting,
    Loading,
    Ready,
    Busy,
    Recovering,
    Error
};
```

Extend `backendStateText` in `appheader.cpp` with these cases:

```cpp
case BackendUiState::Starting:
    return QStringLiteral("后端：正在启动");
case BackendUiState::Recovering:
    return QStringLiteral("后端：正在重连");
```

Use the following severity rules in `AppHeader::setBackendState`:

```cpp
QString messageKind = QStringLiteral("neutral");
if (state == BackendUiState::Ready) messageKind = QStringLiteral("success");
if (state == BackendUiState::Starting || state == BackendUiState::Loading
    || state == BackendUiState::Busy || state == BackendUiState::Recovering) {
    messageKind = QStringLiteral("warning");
}
if (state == BackendUiState::Error || state == BackendUiState::Disconnected) {
    messageKind = QStringLiteral("error");
}
```

Derive the summary detail in `AppHeader::setBackendDetails` with this exact precedence:

```cpp
QString summaryDetail;
if (details.state == BackendUiState::Error) {
    summaryDetail = details.recentError;
} else if (details.state == BackendUiState::Loading) {
    summaryDetail = details.modelDetail;
} else if (details.state == BackendUiState::Recovering) {
    summaryDetail = details.recentError.isEmpty()
        ? details.connectionDetail : details.recentError;
} else {
    summaryDetail = details.connectionDetail;
}
setBackendState(details.state, summaryDetail);
```

- [ ] **Step 4: Run the focused test target and verify GREEN**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_appfoundation
```

Expected: all `test_appfoundation` cases pass, including the new lifecycle-label case.

- [ ] **Step 5: Commit Task 1**

```powershell
git add qt_app/appheader.h qt_app/appheader.cpp qt_app/tests/test_appfoundation.cpp
git commit -m "feat: add stable backend lifecycle labels"
```

---

### Task 2: Aggregate Managed Backend Events in MainWindow

**Files:**
- Modify: `qt_app/mainwindow.h:78-85,87-205`
- Modify: `qt_app/mainwindow.cpp:112-116,849-876,1447-1503,1506-1625,1635-1680,2111-2114`
- Test: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Consumes: `BackendUiState::Starting` and `BackendUiState::Recovering` from Task 1; existing `BackendProcessManager` signals and `BackendClient::State`.
- Produces: `MainWindow::presentBackendState(const BackendStatusDetails &)`, `backendPresentationState_`, and `backendEverReady_`. Task 3 does not depend on these private members.

- [ ] **Step 1: Write the failing managed-startup churn test**

Add this slot to `TestMainWindow`:

```cpp
void managedStartupDoesNotExposeRetryChurn() {
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(37651), &client, &launcher);
    MainWindow window(&client, &manager);
    auto *status = window.findChild<QLabel *>(QStringLiteral("backendStatusLabel"));
    QVERIFY(status != nullptr);
    QVERIFY(status->text().contains(QStringLiteral("正在启动")));

    auto setClientState = [&window](BackendClient::State state, const QString &detail) {
        return QMetaObject::invokeMethod(
            &window, "onClientStateChanged", Qt::DirectConnection,
            Q_ARG(BackendClient::State, state), Q_ARG(QString, detail));
    };
    QVERIFY(setClientState(BackendClient::State::Connecting,
                           QStringLiteral("正在连接后端")));
    QVERIFY(setClientState(BackendClient::State::Error,
                           QStringLiteral("Connection refused")));
    QVERIFY(QMetaObject::invokeMethod(
        &window, "onClientTransportFailed", Qt::DirectConnection,
        Q_ARG(QString, QStringLiteral("CONNECTION_ERROR")),
        Q_ARG(QString, QStringLiteral("Connection refused"))));
    QVERIFY(setClientState(BackendClient::State::Disconnected,
                           QStringLiteral("后端连接已断开")));

    QVERIFY(status->text().contains(QStringLiteral("正在启动")));
    QVERIFY(!status->text().contains(QStringLiteral("未连接")));
    QVERIFY(!status->text().contains(QStringLiteral("不可用")));
}
```

- [ ] **Step 2: Write the failing loading, terminal-error, and reconnect tests**

Add these slots to `TestMainWindow`:

```cpp
void managedLoadingAndTerminalFailureHaveStablePrecedence() {
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(37652), &client, &launcher);
    MainWindow window(&client, &manager);
    auto *status = window.findChild<QLabel *>(QStringLiteral("backendStatusLabel"));
    QVERIFY(status != nullptr);

    emit manager.backendLoading(QStringLiteral("正在加载权重"));
    QVERIFY(status->text().contains(QStringLiteral("模型加载中")));
    QVERIFY(QMetaObject::invokeMethod(
        &window, "onClientStateChanged", Qt::DirectConnection,
        Q_ARG(BackendClient::State, BackendClient::State::Disconnected),
        Q_ARG(QString, QStringLiteral("后端连接已断开"))));
    QVERIFY(status->text().contains(QStringLiteral("模型加载中")));

    emit manager.backendUnavailable(QStringLiteral("后端启动超时"));
    QVERIFY(status->text().contains(QStringLiteral("不可用")));
    QVERIFY(status->toolTip().contains(QStringLiteral("后端启动超时")));
    QVERIFY(QMetaObject::invokeMethod(
        &window, "onClientStateChanged", Qt::DirectConnection,
        Q_ARG(BackendClient::State, BackendClient::State::Disconnected),
        Q_ARG(QString, QStringLiteral("后端连接已断开"))));
    QVERIFY(status->text().contains(QStringLiteral("不可用")));
}

void managedConnectionLossShowsOneRecoveryStateUntilReady() {
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(37653), &client, &launcher);
    MainWindow window(&client, &manager);
    auto *status = window.findChild<QLabel *>(QStringLiteral("backendStatusLabel"));
    QVERIFY(status != nullptr);

    QVERIFY(QMetaObject::invokeMethod(&window, "onBackendReady", Qt::DirectConnection));
    QVERIFY(status->text().contains(QStringLiteral("已连接")));
    QVERIFY(QMetaObject::invokeMethod(
        &window, "onClientTransportFailed", Qt::DirectConnection,
        Q_ARG(QString, QStringLiteral("CONNECTION_LOST")),
        Q_ARG(QString, QStringLiteral("连接已断开"))));
    QVERIFY(status->text().contains(QStringLiteral("正在重连")));

    for (BackendClient::State state : {BackendClient::State::Error,
                                       BackendClient::State::Disconnected,
                                       BackendClient::State::Connecting}) {
        QVERIFY(QMetaObject::invokeMethod(
            &window, "onClientStateChanged", Qt::DirectConnection,
            Q_ARG(BackendClient::State, state),
            Q_ARG(QString, QStringLiteral("自动重连"))));
        QVERIFY(status->text().contains(QStringLiteral("正在重连")));
    }

    QVERIFY(QMetaObject::invokeMethod(&window, "onBackendReady", Qt::DirectConnection));
    QVERIFY(status->text().contains(QStringLiteral("已连接")));
}

void unmanagedClientErrorRemainsDirectlyVisible() {
    BackendClient client;
    MainWindow window(&client, nullptr);
    auto *status = window.findChild<QLabel *>(QStringLiteral("backendStatusLabel"));
    QVERIFY(status != nullptr);

    QVERIFY(QMetaObject::invokeMethod(
        &window, "onClientStateChanged", Qt::DirectConnection,
        Q_ARG(BackendClient::State, BackendClient::State::Error),
        Q_ARG(QString, QStringLiteral("外部后端连接失败"))));
    QVERIFY(status->text().contains(QStringLiteral("不可用")));
    QVERIFY(status->toolTip().contains(QStringLiteral("外部后端连接失败")));
}
```

- [ ] **Step 3: Run the MainWindow target and verify RED**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_mainwindow
```

Expected: the new tests fail because the initial managed state is “未连接”, startup transport errors become unavailable/disconnected, and reconnect attempts overwrite one another.

- [ ] **Step 4: Add the presentation reducer state to MainWindow**

Add these declarations to the private section of `mainwindow.h`:

```cpp
void presentBackendState(const BackendStatusDetails &details);

BackendUiState backendPresentationState_ = BackendUiState::Disconnected;
bool backendEverReady_ = false;
QString backendRecoveryDetail_;
```

Add this implementation near the backend slots in `mainwindow.cpp`:

```cpp
void MainWindow::presentBackendState(const BackendStatusDetails &details) {
    backendPresentationState_ = details.state;
    appHeader_->setBackendDetails(details);
    QString pageDetail = details.connectionDetail;
    if (details.state == BackendUiState::Loading) pageDetail = details.modelDetail;
    if (details.state == BackendUiState::Busy) pageDetail = details.currentTask;
    if (details.state == BackendUiState::Error
        || details.state == BackendUiState::Recovering) {
        pageDetail = details.recentError.isEmpty()
            ? details.connectionDetail : details.recentError;
    }
    workpieceLibraryPage_->setBackendState(details.state, pageDetail);
}
```

Initialize the managed and unmanaged shells distinctly in `initializeUi`:

```cpp
BackendStatusDetails initialBackendDetails;
initialBackendDetails.state = manager_ == nullptr
    ? BackendUiState::Disconnected : BackendUiState::Starting;
initialBackendDetails.connectionDetail = manager_ == nullptr
    ? QStringLiteral("未连接") : QStringLiteral("正在启动后端");
initialBackendDetails.modelDetail = manager_ == nullptr
    ? QStringLiteral("未加载") : QStringLiteral("等待模型就绪");
initialBackendDetails.canRestart = manager_ != nullptr;
presentBackendState(initialBackendDetails);
```

In `setBackendError`, replace the direct header/library calls with:

```cpp
backendEverReady_ = false;
presentBackendState(details);
```

- [ ] **Step 5: Route explicit process lifecycle signals through the reducer**

In `onBackendReady`, set readiness history before presenting:

```cpp
backendEverReady_ = true;
BackendStatusDetails details;
details.state = BackendUiState::Ready;
details.connectionDetail = QStringLiteral("已连接");
details.modelDetail = QStringLiteral("已加载");
details.canRestart = true;
presentBackendState(details);
```

Keep the existing page availability, timer, refresh, and button logic around this block unchanged.

In `onBackendLoading`, retain the existing availability and message logic but replace direct state writes with:

```cpp
BackendStatusDetails details;
details.state = BackendUiState::Loading;
details.connectionDetail = QStringLiteral("已连接");
details.modelDetail = message.isEmpty()
    ? QStringLiteral("模型加载中") : message;
details.canRestart = manager_ != nullptr;
presentBackendState(details);
```

In `onBackendUnavailable`, keep all existing interruption cleanup and replace the direct header/library calls with:

```cpp
BackendStatusDetails details;
details.state = BackendUiState::Error;
details.connectionDetail = QStringLiteral("连接中断");
details.modelDetail = QStringLiteral("状态未知");
details.currentTask = interruptedTask;
details.recentError = reason;
details.canRestart = true;
presentBackendState(details);
```

- [ ] **Step 6: Replace transient client-to-UI mapping with stable precedence**

Replace `MainWindow::onClientStateChanged` with:

```cpp
void MainWindow::onClientStateChanged(BackendClient::State state,
                                      const QString &detail) {
    clientBusy_ = state == BackendClient::State::Busy;
    if (state != BackendClient::State::Ready
        && state != BackendClient::State::Busy) {
        backendReadyHandled_ = false;
    }

    if (state == BackendClient::State::Ready) {
        backendReady_ = true;
        backendEverReady_ = true;
        BackendStatusDetails details;
        details.state = BackendUiState::Ready;
        details.connectionDetail = QStringLiteral("已连接");
        details.modelDetail = QStringLiteral("已加载");
        details.canRestart = manager_ != nullptr;
        presentBackendState(details);
    } else if (state == BackendClient::State::Busy) {
        BackendStatusDetails details;
        details.state = BackendUiState::Busy;
        details.connectionDetail = QStringLiteral("已连接");
        details.modelDetail = QStringLiteral("已加载");
        details.currentTask = detail;
        details.canRestart = manager_ != nullptr;
        presentBackendState(details);
    } else {
        backendReady_ = false;
        if (manager_ != nullptr) {
            if (backendPresentationState_ != BackendUiState::Error) {
                if (!backendEverReady_) {
                    BackendStatusDetails details;
                    details.state = backendPresentationState_ == BackendUiState::Loading
                        ? BackendUiState::Loading : BackendUiState::Starting;
                    details.connectionDetail = QStringLiteral("正在启动后端");
                    details.modelDetail = details.state == BackendUiState::Loading
                        ? QStringLiteral("模型加载中") : QStringLiteral("等待模型就绪");
                    details.recentError = state == BackendClient::State::Error
                        ? detail : QString();
                    details.canRestart = true;
                    presentBackendState(details);
                } else if (state == BackendClient::State::Connecting
                           || state == BackendClient::State::Handshaking) {
                    BackendStatusDetails details;
                    details.state = BackendUiState::Recovering;
                    details.connectionDetail = QStringLiteral("正在重新连接");
                    details.modelDetail = QStringLiteral("等待后端恢复");
                    details.recentError = backendRecoveryDetail_;
                    details.canRestart = true;
                    presentBackendState(details);
                }
            }
        } else {
            BackendStatusDetails details;
            details.connectionDetail = detail;
            details.canRestart = state == BackendClient::State::Error;
            if (state == BackendClient::State::Connecting
                || state == BackendClient::State::Handshaking) {
                details.state = BackendUiState::Starting;
                details.modelDetail = QStringLiteral("等待后端就绪");
            } else if (state == BackendClient::State::Error) {
                details.state = BackendUiState::Error;
                details.modelDetail = QStringLiteral("状态未知");
                details.recentError = detail;
            } else {
                details.state = BackendUiState::Disconnected;
                details.modelDetail = QStringLiteral("未加载");
            }
            presentBackendState(details);
        }
    }

    updateButtonStates();
    if (state == BackendClient::State::Ready
        && dispatchStagedGeometryDraftSave()) return;
    if (state == BackendClient::State::Ready
        && dispatchGeometryWorkflowContinuation()) return;
    if (state == BackendClient::State::Ready
        && pendingNavigationKind_ != PendingNavigationKind::None) {
        tryStartPendingGeometrySave();
    }
    if (state == BackendClient::State::Ready && backendReadyHandled_) {
        dispatchQueuedCommand();
    }
}
```

- [ ] **Step 7: Make transport failure transient only while a manager owns recovery**

Replace `onClientTransportFailed` with:

```cpp
void MainWindow::onClientTransportFailed(const QString &code,
                                         const QString &message) {
    const bool recoverableTransport = code == QStringLiteral("CONNECTION_ERROR")
        || code == QStringLiteral("CONNECTION_LOST")
        || code == QStringLiteral("TIMEOUT")
        || code == QStringLiteral("MODEL_LOADING");
    if (manager_ != nullptr && !backendEverReady_ && recoverableTransport) {
        backendRecoveryDetail_ = message;
        return;
    }
    if (manager_ != nullptr && backendPresentationState_ == BackendUiState::Error) {
        backendRecoveryDetail_ = message;
        return;
    }
    if (manager_ != nullptr && backendPresentationState_ == BackendUiState::Recovering
        && recoverableTransport) {
        backendRecoveryDetail_ = message;
        return;
    }

    onBackendUnavailable(message);
    if (manager_ != nullptr && backendEverReady_ && recoverableTransport) {
        backendRecoveryDetail_ = message;
        BackendStatusDetails details;
        details.state = BackendUiState::Recovering;
        details.connectionDetail = QStringLiteral("正在重新连接");
        details.modelDetail = QStringLiteral("等待后端恢复");
        details.recentError = message;
        details.canRestart = true;
        presentBackendState(details);
    }
}
```

At the beginning of `onBackendReady`, add:

```cpp
backendRecoveryDetail_.clear();
```

At the beginning of `restartBackend`, present a stable restart phase before calling the manager:

```cpp
if (manager_ != nullptr) {
    BackendStatusDetails details;
    details.state = backendEverReady_
        ? BackendUiState::Recovering : BackendUiState::Starting;
    details.connectionDetail = backendEverReady_
        ? QStringLiteral("正在重新启动后端") : QStringLiteral("正在启动后端");
    details.modelDetail = QStringLiteral("等待模型就绪");
    details.recentError = backendRecoveryDetail_;
    details.canRestart = true;
    presentBackendState(details);
    manager_->restart();
}
```

- [ ] **Step 8: Run focused and neighboring Qt tests**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_appfoundation,test_backendclient,test_backendprocessmanager,test_workpiecelibrarypage,test_mainwindow
```

Expected: every target passes; the new startup/loading/recovery tests pass without changing `BackendProcessManager` retry tests.

- [ ] **Step 9: Commit Task 2**

```powershell
git add qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/tests/test_mainwindow.cpp
git commit -m "fix: stabilize managed backend status"
```

---

### Task 3: Make the Batch Result Pane Responsive

**Files:**
- Modify: `qt_app/inspectionpage.ui:55-102`
- Modify: `qt_app/inspectionpage.h:59-94,95-122`
- Modify: `qt_app/inspectionpage.cpp:5-143,717-782,871-980`
- Modify: `qt_app/resources/theme.qss:5-7`
- Test: `qt_app/tests/test_inspectionpage.cpp`
- Test: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Consumes: existing `InspectionRecord`, `InspectionPage::beginBatch`, `handleBackendResponse`, and record-ID-driven table selection.
- Produces: private `InspectionPage::updateResponsivePresentation()` and `InspectionPage::setCurrentResultTargetText(const QString &)`. No public API changes.

- [ ] **Step 1: Write the failing narrow batch layout test**

Add `#include <QGridLayout>`, `#include <QScrollBar>`, and `#include <QSplitter>` to `test_inspectionpage.cpp`, then add:

```cpp
void narrowBatchLayoutKeepsTableAndReviewActionsUsable() {
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    QStringList paths;
    for (int index = 0; index < 12; ++index) {
        paths.append(writeImage(
            directory,
            QStringLiteral("很长的批量检测文件名_%1_用于验证窄屏布局.png").arg(index)));
    }
    QVERIFY(!paths.contains(QString()));

    InspectionPage page;
    page.resize(1037, 620);
    page.setCurrentWorkpiece(QStringLiteral("m1"), QStringLiteral("M1"));
    page.setBackendAvailable(true, false, QString());
    page.beginBatch(paths, QStringLiteral("m1"));
    for (int index = 0; index < paths.size(); ++index) {
        page.handleBackendResponse(
            QStringLiteral("predict"),
            predictionResponse(index % 2 == 0
                                   ? QStringLiteral("front")
                                   : QStringLiteral("back"),
                               index % 3 == 0));
    }
    page.show();
    QCoreApplication::processEvents();

    auto *resultPane = page.findChild<QWidget *>(QStringLiteral("resultPane"));
    auto *table = page.findChild<QTableWidget *>(
        QStringLiteral("batchResultsTableWidget"));
    auto *detail = page.findChild<QWidget *>(QStringLiteral("resultDetailPane"));
    auto *filters = page.findChild<QGridLayout *>(QStringLiteral("batchFilterLayout"));
    auto *front = page.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
    auto *back = page.findChild<QPushButton *>(QStringLiteral("confirmBackButton"));
    auto *reject = page.findChild<QPushButton *>(QStringLiteral("rejectConfirmationButton"));
    QVERIFY(resultPane != nullptr);
    QVERIFY(table != nullptr);
    QVERIFY(detail != nullptr);
    QVERIFY(filters != nullptr);
    QVERIFY(front != nullptr);
    QVERIFY(back != nullptr);
    QVERIFY(reject != nullptr);

    QVERIFY(resultPane->width() >= 440);
    QVERIFY(table->height() >= 140);
    QCOMPARE(table->rowCount(), paths.size());
    QVERIFY(table->isColumnHidden(3));
    QVERIFY(!table->horizontalScrollBar()->isVisible());
    QVERIFY(table->mapTo(resultPane, QPoint()).y()
            < detail->mapTo(resultPane, QPoint()).y());
    QCOMPARE(front->mapTo(resultPane, QPoint()).y(),
             back->mapTo(resultPane, QPoint()).y());
    QCOMPARE(back->mapTo(resultPane, QPoint()).y(),
             reject->mapTo(resultPane, QPoint()).y());

    int row = -1;
    int column = -1;
    int rowSpan = 0;
    int columnSpan = 0;
    filters->getItemPosition(filters->indexOf(
        page.findChild<QPushButton *>(QStringLiteral("unprocessedBatchFilterButton"))),
        &row, &column, &rowSpan, &columnSpan);
    QCOMPARE(row, 1);
}
```

- [ ] **Step 2: Write the failing wide-layout and elided-target tests**

Add:

```cpp
void wideBatchLayoutRestoresElapsedColumn() {
    InspectionPage page;
    page.resize(1920, 900);
    page.setMode(InspectionMode::Batch);
    page.show();
    QCoreApplication::processEvents();

    auto *table = page.findChild<QTableWidget *>(
        QStringLiteral("batchResultsTableWidget"));
    auto *result = page.findChild<QLabel *>(QStringLiteral("resultLabel"));
    QVERIFY(table != nullptr);
    QVERIFY(result != nullptr);
    QVERIFY(!table->isColumnHidden(3));
    QVERIFY(!result->property("compact").toBool());
}

void narrowResultTargetIsElidedButPreservesFullTooltip() {
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString path = writeImage(
        directory,
        QStringLiteral("超长文件名_批量复核结果_需要完整保留在工具提示中.png"));
    QVERIFY(!path.isEmpty());

    InspectionPage page;
    page.resize(1037, 620);
    page.setCurrentWorkpiece(QStringLiteral("m1"), QStringLiteral("M1"));
    page.setBackendAvailable(true, false, QString());
    page.beginBatch({path}, QStringLiteral("m1"));
    page.handleBackendResponse(
        QStringLiteral("predict"), predictionResponse(QStringLiteral("front")));
    page.show();
    QCoreApplication::processEvents();

    auto *target = page.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"));
    QVERIFY(target != nullptr);
    QVERIFY(target->toolTip().contains(QFileInfo(path).fileName()));
    QVERIFY(target->text().size() < target->toolTip().size());
}

void scaled720pWindowKeepsBatchReviewActionsVisible() {
    MainWindow window;
    window.resize(1024, 640);
    auto *page = window.findChild<InspectionPage *>();
    QVERIFY(page != nullptr);
    page->setMode(InspectionMode::Batch);
    window.show();
    QCoreApplication::processEvents();

    auto *resultPane = page->findChild<QWidget *>(QStringLiteral("resultPane"));
    auto *table = page->findChild<QTableWidget *>(
        QStringLiteral("batchResultsTableWidget"));
    QVERIFY(resultPane != nullptr);
    QVERIFY(table != nullptr);
    QCOMPARE(window.size(), QSize(1024, 640));
    QVERIFY(resultPane->width() >= 440);
    QVERIFY(table->height() >= 90);
    QVERIFY(!table->horizontalScrollBar()->isVisible());

    for (const QString &name : {QStringLiteral("confirmFrontButton"),
                                QStringLiteral("confirmBackButton"),
                                QStringLiteral("rejectConfirmationButton")}) {
        auto *button = page->findChild<QPushButton *>(name);
        QVERIFY(button != nullptr);
        QVERIFY(button->isVisibleTo(&window));
        const QRect rect(button->mapTo(&window, QPoint()), button->size());
        QVERIFY2(window.rect().contains(rect), qPrintable(name));
    }
}
```

Place `scaled720pWindowKeepsBatchReviewActionsVisible` in `TestMainWindow`, not `TestInspectionPage`; all other tests in this step belong to `TestInspectionPage`.

- [ ] **Step 3: Run the inspection target and verify RED**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage,test_mainwindow
```

Expected: the new tests fail because the result pane has no 440 px minimum, filters/actions use one-row/vertical layouts, detail precedes the table, the elapsed column remains visible, the target label wraps, and the 1024×640 shell cannot guarantee the batch review controls stay in bounds.

- [ ] **Step 4: Restructure only the batch review portion of the UI file**

Apply these exact structural changes in `inspectionpage.ui`:

```xml
<widget class="QWidget" name="resultPane">
 <property name="minimumSize">
  <size><width>440</width><height>0</height></size>
 </property>
```

Replace `batchFilterLayout` with a `QGridLayout` whose positions are:

```xml
<layout class="QGridLayout" name="batchFilterLayout">
 <property name="horizontalSpacing"><number>8</number></property>
 <property name="verticalSpacing"><number>8</number></property>
 <item row="0" column="0"><widget class="QPushButton" name="allBatchFilterButton"><property name="text"><string>全部</string></property><property name="checkable"><bool>true</bool></property><property name="checked"><bool>true</bool></property></widget></item>
 <item row="0" column="1"><widget class="QPushButton" name="needsReviewBatchFilterButton"><property name="text"><string>需复检</string></property><property name="checkable"><bool>true</bool></property></widget></item>
 <item row="1" column="0"><widget class="QPushButton" name="unprocessedBatchFilterButton"><property name="text"><string>未处理</string></property><property name="checkable"><bool>true</bool></property></widget></item>
 <item row="1" column="1"><widget class="QPushButton" name="failedBatchFilterButton"><property name="text"><string>失败</string></property><property name="checkable"><bool>true</bool></property></widget></item>
</layout>
```

Inside `batchReviewSplitter`, put `batchResultsTableWidget` before `resultDetailPane`. Set:

```xml
<property name="childrenCollapsible"><bool>false</bool></property>
```

Give `batchResultsTableWidget` a minimum height of 140. Replace `confirmationActionsLayout` with `QHBoxLayout` while preserving the same three button object names and initial order. Move this action layout immediately after `reviewLabel`, before the target and evidence controls, so the primary review actions remain visible when vertical space is constrained. Set `currentResultTargetLabel.wordWrap` to `false`.

- [ ] **Step 5: Add compact presentation and deterministic table widths**

In `inspectionpage.h`, add:

```cpp
protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    void updateResponsivePresentation();
    void setCurrentResultTargetText(const QString &text);

    bool compactResultLayout_ = false;
```

In the constructor after `setupUi`, install the filter and configure the reordered splitter/table:

```cpp
ui->resultPane->installEventFilter(this);
ui->inspectionSplitter->setSizes({650, 350});
ui->batchReviewSplitter->setSizes({300, 240});
ui->batchReviewSplitter->setChildrenCollapsible(false);
ui->batchResultsTableWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(
    0, QHeaderView::Stretch);
for (int column = 1; column < 5; ++column) {
    ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(
        column, QHeaderView::Fixed);
}
ui->batchResultsTableWidget->setColumnWidth(1, 64);
ui->batchResultsTableWidget->setColumnWidth(2, 56);
ui->batchResultsTableWidget->setColumnWidth(3, 96);
ui->batchResultsTableWidget->setColumnWidth(4, 112);
```

Remove the old `ResizeToContents` loop. Add:

```cpp
bool InspectionPage::eventFilter(QObject *watched, QEvent *event) {
    if (watched == ui->resultPane && event->type() == QEvent::Resize) {
        updateResponsivePresentation();
    }
    return QWidget::eventFilter(watched, event);
}

void InspectionPage::updateResponsivePresentation() {
    const bool compact = ui->resultPane->width() < 520;
    if (compactResultLayout_ != compact
        || !ui->resultLabel->property("responsiveInitialized").toBool()) {
        compactResultLayout_ = compact;
        ui->resultLabel->setProperty("compact", compact);
        ui->resultLabel->setProperty("responsiveInitialized", true);
        ui->resultLabel->style()->unpolish(ui->resultLabel);
        ui->resultLabel->style()->polish(ui->resultLabel);
        ui->batchResultsTableWidget->setColumnHidden(3, compact);
    }
    const QString fullText = ui->currentResultTargetLabel
                                 ->property("fullText").toString();
    if (!fullText.isEmpty()) {
        ui->currentResultTargetLabel->setText(
            ui->currentResultTargetLabel->fontMetrics().elidedText(
                fullText, Qt::ElideMiddle,
                qMax(80, ui->currentResultTargetLabel->width() - 4)));
    }
}

void InspectionPage::setCurrentResultTargetText(const QString &text) {
    ui->currentResultTargetLabel->setProperty("fullText", text);
    ui->currentResultTargetLabel->setToolTip(text);
    if (text.isEmpty()) {
        ui->currentResultTargetLabel->clear();
        return;
    }
    ui->currentResultTargetLabel->setText(
        ui->currentResultTargetLabel->fontMetrics().elidedText(
            text, Qt::ElideMiddle,
            qMax(80, ui->currentResultTargetLabel->width() - 4)));
}
```

Add `class QEvent;` with the other forward declarations in `inspectionpage.h`, and add this include in `inspectionpage.cpp`:

```cpp
#include <QEvent>
```

Call `setCurrentResultTargetText(QString())` where `renderActiveState` currently clears the label. In `renderRecord`, replace the direct `setText` call with:

```cpp
setCurrentResultTargetText(
    QStringLiteral("当前：%1（%2）")
        .arg(QFileInfo(record.imagePath).fileName(), detail));
```

Call `updateResponsivePresentation()` once at the end of the constructor and once after mode visibility changes in `renderActiveState`.

- [ ] **Step 6: Add compact title styling**

Add this rule immediately after the existing `QLabel#resultLabel` rule in `theme.qss`:

```css
QLabel#resultLabel[compact="true"] { font-size: 30px; }
```

Do not change the default 40 px result title used by wide/single layouts.

- [ ] **Step 7: Preserve useful tooltips for fixed-width table cells**

In `rebuildBatchTable`, set a tooltip on every cell instead of only column 0:

```cpp
item->setToolTip(column == 0 ? record.imagePath : values.at(column));
```

This keeps fixed status/result cells inspectable without enabling a horizontal scrollbar.

- [ ] **Step 8: Run the inspection and shell tests and verify GREEN**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage,test_mainwindow
```

Expected: both targets pass; at 1037×620 the table is above the detail, at least 140 px high, the elapsed column is hidden, filters occupy two rows, actions occupy one row, and no horizontal scrollbar is visible. At 1920 px the elapsed column is restored.

- [ ] **Step 9: Commit Task 3**

```powershell
git add qt_app/inspectionpage.ui qt_app/inspectionpage.h qt_app/inspectionpage.cpp qt_app/resources/theme.qss qt_app/tests/test_inspectionpage.cpp qt_app/tests/test_mainwindow.cpp
git commit -m "fix: make batch review layout responsive"
```

---

### Task 4: Capture Narrow Batch Evidence and Run Full Verification

**Files:**
- Modify: `qt_app/tests/test_inspectionpage.cpp`
- Create: `docs/verification/screenshots/qt-ui-inspection-batch-1085x752.png`
- Create: `docs/verification/backend-status-and-batch-layout-fix-results.md`

**Interfaces:**
- Consumes: stable backend presentation from Task 2 and responsive batch layout from Task 3.
- Produces: reproducible screenshot evidence and a final verification record; no production API.

- [ ] **Step 1: Add opt-in screenshot capture to the existing inspection test target**

Add this slot to `TestInspectionPage`:

```cpp
void captureNarrowBatchEvidenceWhenRequested() {
    const QString captureDirectory = QString::fromLocal8Bit(
        qgetenv("QT_UI_CAPTURE_DIR"));
    if (captureDirectory.isEmpty()) return;
    QVERIFY(QDir().mkpath(captureDirectory));

    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    QStringList paths;
    for (int index = 0; index < 12; ++index) {
        paths.append(writeImage(
            directory,
            QStringLiteral("batch-review-%1.png").arg(index, 2, 10, QLatin1Char('0'))));
    }

    InspectionPage page;
    page.resize(1085, 752);
    page.setCurrentWorkpiece(QStringLiteral("m1"), QStringLiteral("圆形端盖 A"));
    page.setBackendAvailable(true, false, QString());
    page.beginBatch(paths, QStringLiteral("m1"));
    for (int index = 0; index < paths.size(); ++index) {
        page.handleBackendResponse(
            QStringLiteral("predict"),
            predictionResponse(index % 2 == 0
                                   ? QStringLiteral("front")
                                   : QStringLiteral("back"),
                               index == 1 || index == 4 || index == 8));
    }
    page.show();
    QCoreApplication::processEvents();

    const QImage image = page.grab().toImage();
    QVERIFY(!image.isNull());
    QCOMPARE(image.size(), QSize(1085, 752));
    QVERIFY(image.save(QDir(captureDirectory).filePath(
        QStringLiteral("qt-ui-inspection-batch-1085x752.png"))));
}
```

Add this include with the screenshot test:

```cpp
#include <QDir>
```

- [ ] **Step 2: Run the screenshot target on the Windows platform plugin**

First build and run the target once through the normal test runner:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage
```

Then run only the capture case with the Windows platform plugin and a forced scale factor of 1 so that logical and physical pixels are both 1085×752:

```powershell
$env:PATH = 'E:\QT\5.14\5.14.2\msvc2017_64\bin;' + $env:PATH
$env:QT_UI_CAPTURE_DIR = 'E:\Project\wang\pp_813\docs\verification\screenshots'
$env:QT_SCALE_FACTOR = '1'
& .\qt_app\tests\build-test_inspectionpage\release\test_inspectionpage.exe captureNarrowBatchEvidenceWhenRequested -platform windows -o -,txt
Remove-Item Env:QT_UI_CAPTURE_DIR
Remove-Item Env:QT_SCALE_FACTOR
```

Expected: `docs/verification/screenshots/qt-ui-inspection-batch-1085x752.png` exists at exactly 1085×752 and shows a multi-row table, selected-result detail, and one-row confirmation actions without overlap.

- [ ] **Step 3: Run the responsive cases at 100%, 125%, and 150% scaling**

Run the already-built focused executables in independent logical-scale passes:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
foreach ($factor in @('1', '1.25', '1.5')) {
    $env:QT_SCALE_FACTOR = $factor
    & .\qt_app\tests\build-test_inspectionpage\release\test_inspectionpage.exe narrowBatchLayoutKeepsTableAndReviewActionsUsable wideBatchLayoutRestoresElapsedColumn narrowResultTargetIsElidedButPreservesFullTooltip -platform offscreen -o -,txt
    if ($LASTEXITCODE -ne 0) { throw "Inspection DPI verification failed at scale $factor" }
    & .\qt_app\tests\build-test_mainwindow\release\test_mainwindow.exe scaled720pWindowKeepsBatchReviewActionsVisible -platform offscreen -o -,txt
    if ($LASTEXITCODE -ne 0) { throw "MainWindow DPI verification failed at scale $factor" }
}
Remove-Item Env:QT_QPA_PLATFORM
Remove-Item Env:QT_SCALE_FACTOR
```

Expected: all four focused cases pass at every scale; the 440/520 thresholds continue to behave as logical pixels.

- [ ] **Step 4: Run the complete Qt test matrix**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1
```

Expected: all ten targets exit with code 0. Based on the current 319-case baseline plus the ten planned cases, the expected total is 329 passed and 0 failed.

- [ ] **Step 5: Run the complete non-integration Python regression**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -m "not integration" -q -p no:cacheprovider --basetemp="$env:TEMP\pytest-backend-status-layout"
```

Expected: the existing baseline remains 375 passed and 3 deselected; no repository-local pytest cache or manual output directory is created.

- [ ] **Step 6: Build the Release executable**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1
```

Expected: exit code 0 and `qt_app/build-release/release/workpiece_orientation.exe` exists.

- [ ] **Step 7: Record the exact verification results**

Create `docs/verification/backend-status-and-batch-layout-fix-results.md` with this structure, using exact observed counts and elapsed times from Steps 3–5:

```markdown
# 后端状态稳定化与批量结果布局修复验证结果

## 修复结果

- 冷启动自动重连期间顶部状态保持“后端：正在启动”。
- 模型加载、连接成功、运行中断线恢复和终止性错误使用互不冲突的状态。
- 1085×752 批量页面结果表、详情和确认操作均完整可用。
- 后端启动、模型、识别算法、融合阈值和模板数据未修改。

## Qt 测试

- 命令：`powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1`
- 结果：329 passed，0 failed。

## Python 回归

- 命令：`E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -m "not integration" -q -p no:cacheprovider --basetemp="$env:TEMP\pytest-backend-status-layout"`
- 结果：375 passed，3 deselected。

## Release 构建

- 命令：`powershell -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1`
- 结果：成功。
- 产物：`E:\Project\wang\pp_813\qt_app\build-release\release\workpiece_orientation.exe`。

## 视觉证据

- [1085×752 批量检测](screenshots/qt-ui-inspection-batch-1085x752.png)
- 结果栏最小宽度：440 个逻辑像素。
- 紧凑模式阈值：520 个逻辑像素。
- 窄栏隐藏耗时列，宽栏自动恢复；耗时仍保留在原始证据中。
- 100%、125%、150% 三档缩放下的窄屏与宽屏布局测试均通过。
```

If an observed count differs because the repository test inventory changed after this plan was written, record the exact observed count and list the added or removed test names; do not force the document to claim 329.

- [ ] **Step 8: Check the final diff and commit verification evidence**

Run:

```powershell
git diff --check
git status --short
```

Confirm only the screenshot test, screenshot, and verification report remain uncommitted, then run:

```powershell
git add qt_app/tests/test_inspectionpage.cpp docs/verification/backend-status-and-batch-layout-fix-results.md docs/verification/screenshots/qt-ui-inspection-batch-1085x752.png
git commit -m "test: verify backend status and batch layout"
```

Expected: clean worktree and no generated pytest/manual/process files in the repository.

---

## Final Acceptance Checklist

- [ ] Cold startup never alternates visible “未连接” and “正在加载” states.
- [ ] `MODEL_LOADING`, `Ready`, `Busy`, reconnecting, and terminal failure remain distinguishable.
- [ ] Real connection loss still preserves uncertain in-flight work and disables unsafe actions.
- [ ] The manager/client retry and startup tests pass unchanged.
- [ ] At 1024×640 and 1085×752 logical size, the batch result pane is at least 440 px wide.
- [ ] The narrow result table has usable height, no horizontal scrollbar, and no elapsed column.
- [ ] At 1920×1080 the elapsed column is visible.
- [ ] Clicking a row still updates image, evidence, and confirmation target by record ID.
- [ ] Single-image inspection and keyboard focus order do not regress.
- [ ] Full Qt tests, Python tests, Release build, and visual evidence pass.
