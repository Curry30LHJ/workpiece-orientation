# Main Inspection Workflow Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every batch result selectable, reviewable, and safely confirmable through the main result controls while preserving single-image inspection behavior.

**Architecture:** `MainWindow` owns a typed record for every completed batch prediction and treats the table only as a view. A shared renderer displays single and batch responses, while explicit selection and confirmation identities prevent running requests, viewed results, and asynchronous confirmation replies from overwriting each other.

**Tech Stack:** C++17, Qt 5.14.2 Widgets/Network/TestLib, Qt Designer `.ui`, qmake, MSVC 2017 ABI with Visual Studio 2019 build tools.

## Global Constraints

- Modify only the Qt main inspection window and its tests; do not modify PP-ShiTuV2, ALIKED, LightGlue, model weights, recognition fusion thresholds, geometry-rule management, backend prediction protocol, or serial batch execution.
- Keep Qt 5.14.2 compatibility; do not use Qt 6-only APIs.
- Keep exactly five batch columns: file, result, review, elapsed milliseconds, processing status.
- Batch execution must finish successfully before any front/back/reject action is enabled.
- Selecting a visible image must never leave “开始检测” targeting a different hidden image.
- Preserve completed results across workpiece changes and backend disconnects; clear them when a new batch selection replaces them.
- The repository already has uncommitted changes in the same Qt files. Do not stage or commit those files if that would include pre-existing user changes; use test and diff checkpoints instead.

---

## File Structure

- Modify: `qt_app/mainwindow.h` — batch records, selection context, pending confirmation identity, helper declarations.
- Modify: `qt_app/mainwindow.cpp` — table projection, shared rendering, selection, confirmation transitions, safety gates, summaries.
- Modify: `qt_app/mainwindow.ui` — current-image and current-action labels.
- Modify: `qt_app/tests/test_mainwindow.cpp` — controllable batch/confirmation fixture and interaction regressions.
- Create: `docs/verification/main-inspection-workflow-polish-results.md` — actual test, build, and manual evidence.

---

### Task 1: Store Batch Results and Make Rows Selectable

**Files:**
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp`
- Modify: `qt_app/mainwindow.ui`
- Test: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Produces: `BatchResultState`, `BatchResult`, `batchResults_`, `selectedBatchResultIndex_`, `renderPredictionResult(...)`, `selectBatchResult(...)`, and `updateBatchRow(...)`.
- Consumes: existing `appendBatchResult(...)`, `orientationText(...)`, `formatScore(...)`, and `updatePreview()`.

- [ ] **Step 1: Make test responses distinguishable**

In `BatchPredictionServer`, store requests and vary scores by prediction index:

```cpp
QList<QJsonObject> requests() const { return requests_; }

// After parsing each valid request:
requests_.append(request);

// In the predict branch after incrementing predictionCount_:
const double frontScore = 0.95 - 0.10 * predictionCount_;
send({
    {"version", 1}, {"request_id", requestId}, {"ok", true},
    {"label", predictionCount_ % 2 == 0 ? "back" : "front"},
    {"global_scores", QJsonObject{{"front", frontScore}, {"back", 1.0 - frontScore}}},
    {"global_margin", qAbs(frontScore - (1.0 - frontScore))},
    {"local_prediction", "front"},
    {"local_scores", QJsonObject{{"front", 8.0 + predictionCount_}, {"back", 1.0}}},
    {"local_margin", 7.0 + predictionCount_}, {"decision_source", "global"},
    {"needs_review", false}, {"elapsed_ms", 12.5 + predictionCount_},
});
```

- [ ] **Step 2: Write the failing row-selection test**

```cpp
void selectingBatchRowShowsItsImageAndEvidence() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    const QStringList paths = writeImages(dir, QStringLiteral("select"), 3);

    client.connectToService(QHostAddress::LocalHost, server.port(), 500);
    QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
    emit client.responseReceived(QStringLiteral("list_workpieces"), QJsonObject{
        {"workpieces", QJsonArray{QJsonObject{{"id", "m1"}, {"name", "M1"}}}},
    });
    window.setBatchImagePaths(paths);
    QVERIFY(QMetaObject::invokeMethod(&window, "submitBatchPrediction", Qt::DirectConnection));

    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE_WITH_TIMEOUT(table->rowCount(), 3, 1500);
    QCOMPARE(table->columnCount(), 5);
    table->setCurrentCell(1, 0);

    QTRY_VERIFY(window.findChild<QLabel *>(QStringLiteral("currentImageLabel"))
                    ->text().contains(QStringLiteral("select-1.png")));
    QVERIFY(window.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))
                ->toPlainText().contains(QStringLiteral("0.75")));
    QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"))
                ->text().contains(QStringLiteral("select-1.png")));
}
```

- [ ] **Step 3: Build and verify RED**

Run from a VS 2019 x64 developer shell:

```bat
cd /d E:\Project\wang\pp_813\qt_app\tests
E:\QT\5.14\5.14.2\msvc2017_64\bin\qmake.exe test_mainwindow.pro -o Makefile.mainwindow -spec win32-msvc
nmake /f Makefile.mainwindow.Release /NOLOGO
release\test_mainwindow.exe selectingBatchRowShowsItsImageAndEvidence -platform offscreen
```

Expected: FAIL because the table has seven columns and selection does not render the stored response.

- [ ] **Step 4: Add typed state to `mainwindow.h`**

```cpp
enum class BatchResultState {
    Pending, Submitting, QueuedFront, QueuedBack, Rejected, SubmitFailed,
};

struct BatchResult {
    QString imagePath;
    QString workpieceId;
    QJsonObject response;
    QString label;
    bool needsReview = false;
    double elapsedMs = 0.0;
    BatchResultState state = BatchResultState::Pending;
    QString submitError;
};

enum class ResultContext { None, Single, Batch };

QVector<BatchResult> batchResults_;
int selectedBatchResultIndex_ = -1;
int pendingConfirmationBatchIndex_ = -1;
QString pendingConfirmationOrientation_;
ResultContext resultContext_ = ResultContext::None;
bool changingBatchSelection_ = false;
bool batchSelectionPinned_ = false;
bool batchCompletedSuccessfully_ = false;
```

Declare:

```cpp
void renderPredictionResult(const QString &imagePath, const QJsonObject &response,
                            const QString &sourceText);
void selectBatchResult(int index, bool userInitiated);
void updateBatchRow(int index);
QString batchResultStateText(BatchResultState state) const;
```

Add `#include <QVector>`.

- [ ] **Step 5: Simplify the table and add context labels**

In `mainwindow.ui`, add `currentImageLabel` below `imagePreviewLabel` and `currentResultTargetLabel` before the dynamic confirmation buttons. Initialize both with empty text.

In `initializeUi()`:

```cpp
ui->batchResultsTableWidget->setColumnCount(5);
ui->batchResultsTableWidget->setHorizontalHeaderLabels({
    QStringLiteral("文件"), QStringLiteral("结果"), QStringLiteral("复检"),
    QStringLiteral("耗时（毫秒）"), QStringLiteral("处理状态")});
ui->batchResultsTableWidget->setSelectionMode(QAbstractItemView::SingleSelection);
connect(ui->batchResultsTableWidget, &QTableWidget::currentCellChanged,
        this, [this](int row, int, int, int) {
            if (!changingBatchSelection_) selectBatchResult(row, true);
        });
```

- [ ] **Step 6: Store each response and project five text cells**

Replace per-row action buttons in `appendBatchResult()`:

```cpp
BatchResult result;
result.imagePath = batchImagePaths_.at(batchIndex_);
result.workpieceId = batchWorkpieceId_;
result.response = response;
result.label = response.value(QStringLiteral("label")).toString();
result.needsReview = response.value(QStringLiteral("needs_review")).toBool();
result.elapsedMs = response.value(QStringLiteral("elapsed_ms")).toDouble();
batchResults_.append(result);

const int row = batchResults_.size() - 1;
ui->batchResultsTableWidget->insertRow(row);
updateBatchRow(row);
if (selectedBatchResultIndex_ < 0) selectBatchResult(row, false);
```

`updateBatchRow()` writes all five cells, sets the filename tooltip to the absolute path, and applies a pale warning brush when `needsReview` is true. `clearBatchResults()` clears typed records, indices, table rows, labels, and Batch context.

- [ ] **Step 7: Extract shared rendering and implement row selection**

Move the existing `predict` evidence formatting into `renderPredictionResult()`. It assigns `inspectionImagePath_ = imagePath`, calls `updatePreview()`, fills both labels and the existing result/evidence widgets, then calls `updateButtonStates()`.

The non-batch `predict` response sets `resultContext_ = ResultContext::Single`, updates `lastPredictionWorkpieceId_`, `lastPredictionImagePath_`, and `lastPredictionResponse_`, then calls the shared renderer. The batch response branch must stop writing those single-result fields; it only appends its typed `BatchResult`.

```cpp
void MainWindow::selectBatchResult(int index, bool userInitiated) {
    if (index < 0 || index >= batchResults_.size()) return;
    selectedBatchResultIndex_ = index;
    resultContext_ = ResultContext::Batch;
    if (userInitiated) batchSelectionPinned_ = true;
    changingBatchSelection_ = true;
    ui->batchResultsTableWidget->setCurrentCell(index, 0);
    changingBatchSelection_ = false;
    const BatchResult &result = batchResults_.at(index);
    renderPredictionResult(result.imagePath, result.response, QStringLiteral("批量结果"));
}
```

- [ ] **Step 8: Run GREEN and the existing batch regression**

```bat
release\test_mainwindow.exe selectingBatchRowShowsItsImageAndEvidence batchPredictionSendsEachImageAndShowsSummary -platform offscreen
```

Expected: both PASS.

- [ ] **Step 9: Check the diff without staging mixed changes**

```bat
git diff --check -- qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/tests/test_mainwindow.cpp
git diff --stat -- qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/tests/test_mainwindow.cpp
```

Expected: no whitespace errors; do not create a mixed commit.

---

### Task 2: Preserve Manual Selection and Select the First Review Item

**Files:**
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp`
- Test: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Consumes: Task 1 `batchResults_`, `batchSelectionPinned_`, and `selectBatchResult(...)`.
- Produces: `preferredPendingBatchResult(int) const`, `updateBatchSummary()`, and deterministic running/completion selection.

- [ ] **Step 1: Add held responses and review rows to the fixture**

```cpp
void setReviewRows(const QSet<int> &rows) { reviewRows_ = rows; }
void setHoldPredictions(bool hold) { holdPredictions_ = hold; }
void replyNextPrediction();

struct PendingPrediction { QString requestId; int oneBasedIndex; };
QQueue<PendingPrediction> pendingPredictions_;
QSet<int> reviewRows_;
bool holdPredictions_ = false;
```

Add `#include <QQueue>` and `#include <QSet>` to the test file.

Move response construction into `sendPrediction(requestId, oneBasedIndex)`. Enqueue in the predict branch when held; otherwise reply immediately. Set `needs_review` from `reviewRows_.contains(oneBasedIndex - 1)`.

- [ ] **Step 2: Write three failing behavior tests**

`TestMainWindow` gets this private helper so every later batch test has a complete, named setup path:

```cpp
static void startBatch(BatchPredictionServer &server, BackendClient &client,
                       MainWindow &window, const QStringList &paths,
                       const QJsonArray &workpieces = QJsonArray{
                           QJsonObject{{"id", "m1"}, {"name", "M1"}}}) {
    client.connectToService(QHostAddress::LocalHost, server.port(), 500);
    QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
    emit client.responseReceived(QStringLiteral("list_workpieces"),
                                 QJsonObject{{"workpieces", workpieces}});
    window.setBatchImagePaths(paths);
    QVERIFY(QMetaObject::invokeMethod(&window, "submitBatchPrediction",
                                      Qt::DirectConnection));
}

void laterBatchResponsesDoNotReplaceManualSelection() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    server.setHoldPredictions(true);
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    const QStringList paths = writeImages(dir, QStringLiteral("held"), 4);
    startBatch(server, client, window, paths);

    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    server.replyNextPrediction();
    QTRY_COMPARE(table->rowCount(), 1);
    server.replyNextPrediction();
    QTRY_COMPARE(table->rowCount(), 2);
    table->setCurrentCell(0, 0);
    server.replyNextPrediction();
    QTRY_COMPARE(table->rowCount(), 3);

    QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentImageLabel"))
                ->text().contains(QStringLiteral("held-0.png")));
}

void batchCompletionSelectsFirstReviewResult() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    server.setReviewRows(QSet<int>{1, 2});
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    const QStringList paths = writeImages(dir, QStringLiteral("review"), 3);
    startBatch(server, client, window, paths);

    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE_WITH_TIMEOUT(table->rowCount(), 3, 1500);
    QTRY_COMPARE(table->currentRow(), 1);
    QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"))
                ->text().contains(QStringLiteral("review-1.png")));
}

void batchWithoutReviewSelectsFirstResult() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    const QStringList paths = writeImages(dir, QStringLiteral("normal"), 3);
    startBatch(server, client, window, paths);

    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE_WITH_TIMEOUT(table->rowCount(), 3, 1500);
    QTRY_COMPARE(table->currentRow(), 0);
}
```

- [ ] **Step 3: Run RED**

```bat
release\test_mainwindow.exe laterBatchResponsesDoNotReplaceManualSelection batchCompletionSelectsFirstReviewResult batchWithoutReviewSelectsFirstResult -platform offscreen
```

Expected: review-priority and selection-pinning assertions FAIL.

- [ ] **Step 4: Implement preferred pending selection**

```cpp
int MainWindow::preferredPendingBatchResult(int afterIndex) const {
    auto eligible = [this](int i) {
        const auto state = batchResults_.at(i).state;
        return state == BatchResultState::Pending || state == BatchResultState::SubmitFailed;
    };
    for (int pass = 0; pass < 2; ++pass) {
        for (int offset = 1; offset <= batchResults_.size(); ++offset) {
            const int index = (afterIndex + offset + batchResults_.size()) % batchResults_.size();
            if (eligible(index) && (pass == 1 || batchResults_.at(index).needsReview)) return index;
        }
    }
    return -1;
}
```

Reset `batchSelectionPinned_` at batch start. While appending, do not select new rows after a user selection. At successful completion select `preferredPendingBatchResult(-1)`.

- [ ] **Step 5: Implement running and final summaries**

During execution show `已完成 X/Y，当前文件：<name>`. After completion count total, front, back, uncertain, review, processed, and pending. Count queued front/back and rejected as processed; count pending and submit-failed as pending.

- [ ] **Step 6: Run GREEN**

Run `release\test_mainwindow.exe -platform offscreen`. Expected: all tests PASS.

- [ ] **Step 7: Check the task diff**

Run `git diff --check -- qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/tests/test_mainwindow.cpp`. Expected: no whitespace errors.

---

### Task 3: Unify Confirmation Controls and Auto-Advance

**Files:**
- Modify: `qt_app/mainwindow.h`
- Modify: `qt_app/mainwindow.cpp`
- Test: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Consumes: Task 2 pending-result selection and the existing `submit_confirmation` command.
- Produces: `currentBatchResultCanBeProcessed() const`, `submitCurrentConfirmation(...)`, batch-aware reject, and exact reply routing.

- [ ] **Step 1: Add confirmation replies to the fixture**

```cpp
int confirmationCount() const { return confirmationCount_; }
void failNextConfirmation() { failNextConfirmation_ = true; }

// In readRequests():
} else if (command == QStringLiteral("submit_confirmation")) {
    ++confirmationCount_;
    if (failNextConfirmation_) {
        failNextConfirmation_ = false;
        send({{"version", 1}, {"request_id", requestId}, {"ok", false},
              {"error", QJsonObject{{"code", "MODEL_ERROR"}, {"message", "queue failed"}}}});
    } else {
        send({{"version", 1}, {"request_id", requestId}, {"ok", true},
              {"job", QJsonObject{{"job_id", QStringLiteral("job-%1").arg(confirmationCount_)}}}});
    }
}
```

- [ ] **Step 2: Write failing confirmation tests**

Add:

```cpp
void confirmingBatchResultUpdatesStatusAndAdvances() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    server.setReviewRows(QSet<int>{0, 1});
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    startBatch(server, client, window, writeImages(dir, QStringLiteral("confirm"), 3));
    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE(table->rowCount(), 3);
    QTRY_COMPARE(table->currentRow(), 0);

    window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->click();
    QTRY_COMPARE(server.confirmationCount(), 1);
    QTRY_COMPARE(table->item(0, 4)->text(), QStringLiteral("正面已排队"));
    QTRY_COMPARE(table->currentRow(), 1);
}

void rejectingBatchResultIsLocalAndAdvances() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    server.setReviewRows(QSet<int>{0, 1});
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    startBatch(server, client, window, writeImages(dir, QStringLiteral("reject"), 3));
    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE(table->rowCount(), 3);
    const int before = server.confirmationCount();

    window.findChild<QPushButton *>(QStringLiteral("rejectConfirmationButton"))->click();
    QCOMPARE(server.confirmationCount(), before);
    QCOMPARE(table->item(0, 4)->text(), QStringLiteral("不入库"));
    QCOMPARE(table->currentRow(), 1);
}

void failedBatchConfirmationStaysSelectedAndCanRetry() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    server.setReviewRows(QSet<int>{0});
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    startBatch(server, client, window, writeImages(dir, QStringLiteral("failed"), 2));
    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE(table->rowCount(), 2);
    server.failNextConfirmation();

    window.findChild<QPushButton *>(QStringLiteral("confirmBackButton"))->click();
    QTRY_COMPARE(table->item(0, 4)->text(), QStringLiteral("提交失败"));
    QCOMPARE(table->currentRow(), 0);
    QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))
                ->text().contains(QStringLiteral("queue failed")));
    QTRY_VERIFY(window.findChild<QPushButton *>(QStringLiteral("confirmBackButton"))->isEnabled());
}
```

- [ ] **Step 3: Run RED**

Run the three named tests. Expected: FAIL because replies are not associated with batch rows.

- [ ] **Step 4: Route global buttons by active context**

```cpp
void MainWindow::submitCurrentConfirmation(const QString &orientation) {
    if (resultContext_ == ResultContext::Batch) {
        if (!currentBatchResultCanBeProcessed()) return;
        BatchResult &result = batchResults_[selectedBatchResultIndex_];
        result.state = BatchResultState::Submitting;
        result.submitError.clear();
        pendingConfirmationBatchIndex_ = selectedBatchResultIndex_;
        pendingConfirmationOrientation_ = orientation;
        updateBatchRow(selectedBatchResultIndex_);
        submitTemplateConfirmation(result.workpieceId, result.imagePath, orientation);
        return;
    }
    submitTemplateConfirmation(lastPredictionWorkpieceId_, lastPredictionImagePath_, orientation);
}
```

Change the helper signature to:

```cpp
void submitTemplateConfirmation(const QString &workpieceId, const QString &imagePath,
                                const QString &orientation);
```

- [ ] **Step 5: Route success and failure to the exact row**

On success, set `QueuedFront` or `QueuedBack`, clear the pending identity, update the row/summary, then select `preferredPendingBatchResult(completedIndex)`. On failure, set only that row to `SubmitFailed`, store the message, retain selection, clear pending identity, and re-run `updateButtonStates()`.

- [ ] **Step 6: Make reject batch-aware**

For Batch context, change only the selected processable record to `Rejected`, update its row/summary, then auto-advance. For Single context retain the existing last-result clearing behavior.

- [ ] **Step 7: Centralize confirmation eligibility**

`currentBatchResultCanBeProcessed()` requires: successful batch completion, Ready backend/client, matching workpiece ID, valid selected index, and state Pending or SubmitFailed. Use the same boolean for all three buttons. Set `currentResultTargetLabel` to the filename/state or an explicit reason such as `批量检测完成后可处理`, `结果来自其他工件`, or `后端不可用，结果已保留`.

- [ ] **Step 8: Run GREEN and diff checks**

Run the full main-window test executable, then:

```bat
git diff --check -- qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/tests/test_mainwindow.cpp
```

Expected: all tests PASS; no whitespace errors.

---

### Task 4: Enforce Workpiece, Disconnect, and Partial-Failure Safety

**Files:**
- Modify: `qt_app/mainwindow.cpp`
- Test: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Consumes: Task 3 confirmation gate and Task 1 stored records.
- Produces: preserved batch evidence across state changes and exact partial-failure feedback.

- [ ] **Step 1: Write failing safety tests**

First add a deterministic prediction failure to `BatchPredictionServer`:

```cpp
void failPredictionRow(int zeroBasedRow) { failedPredictionRow_ = zeroBasedRow; }
int failedPredictionRow_ = -1;

// In the predict branch, after assigning zeroBasedRow:
if (zeroBasedRow == failedPredictionRow_) {
    send({{"version", 1}, {"request_id", requestId}, {"ok", false},
          {"error", QJsonObject{{"code", "MODEL_ERROR"}, {"message", "predict failed"}}}});
    continue;
}
```

Add `#include <QComboBox>` to the test file for the workpiece-switch test.

Add the tests:

```cpp
void workpieceSwitchPreservesBatchButBlocksConfirmationUntilSwitchedBack() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    const QJsonArray workpieces{
        QJsonObject{{"id", "m1"}, {"name", "M1"}},
        QJsonObject{{"id", "m2"}, {"name", "M2"}}};
    startBatch(server, client, window, writeImages(dir, QStringLiteral("switch"), 2),
               workpieces);
    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    auto *combo = window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
    auto *front = window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
    QTRY_COMPARE(table->rowCount(), 2);
    QTRY_VERIFY(front->isEnabled());

    combo->setCurrentIndex(1);
    QCOMPARE(table->rowCount(), 2);
    QVERIFY(!front->isEnabled());
    QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"))
                ->text().contains(QStringLiteral("其他工件")));
    combo->setCurrentIndex(0);
    QTRY_VERIFY(front->isEnabled());
}

void disconnectPreservesBatchEvidenceAndDisablesConfirmation() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    startBatch(server, client, window, writeImages(dir, QStringLiteral("disconnect"), 2));
    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE(table->rowCount(), 2);
    table->setCurrentCell(1, 0);
    const QString evidence = window.findChild<QTextEdit *>(
        QStringLiteral("evidenceTextEdit"))->toPlainText();

    emit client.transportFailed(QStringLiteral("CONNECTION_LOST"), QStringLiteral("lost"));
    QCOMPARE(table->rowCount(), 2);
    QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentImageLabel"))
                ->text().contains(QStringLiteral("disconnect-1.png")));
    QCOMPARE(window.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))
                 ->toPlainText(), evidence);
    QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->isEnabled());
}

void replacingBatchSelectionClearsOldRowsAndContext() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    startBatch(server, client, window, writeImages(dir, QStringLiteral("old"), 2));
    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE(table->rowCount(), 2);

    window.setBatchImagePaths(writeImages(dir, QStringLiteral("new"), 3));
    QCOMPARE(table->rowCount(), 0);
    QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"))->text().isEmpty());
    QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->isEnabled());
}

void midBatchFailureKeepsCompletedRowsAndShowsExactProgress() {
    BatchPredictionServer server;
    QVERIFY(server.listen());
    server.failPredictionRow(2);
    BackendClient client;
    PassiveLauncher launcher;
    BackendProcessManager manager(configFor(server.port()), &client, &launcher);
    MainWindow window(&client, &manager);
    QTemporaryDir dir;
    startBatch(server, client, window, writeImages(dir, QStringLiteral("partial"), 4));
    auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
    QTRY_COMPARE_WITH_TIMEOUT(table->rowCount(), 2, 1500);
    const QString summary = window.findChild<QLabel *>(
        QStringLiteral("batchSummaryLabel"))->text();
    QTRY_VERIFY(summary.contains(QStringLiteral("已完成 2/4")));
    QVERIFY(summary.contains(QStringLiteral("partial-2.png")));
}
```

- [ ] **Step 2: Run RED**

Run those four named tests. Expected: at least workpiece gating and exact partial summary FAIL.

- [ ] **Step 3: Preserve batch context on workpiece changes**

In the combo-box handler, clear single-result state only when `resultContext_ != ResultContext::Batch`. Never clear typed batch records, selected row, preview, or evidence there. Always call `updateButtonStates()`.

- [ ] **Step 4: Preserve evidence on disconnect**

Keep the selected Batch result rendered in `onBackendUnavailable()`. If a confirmation was in flight, mark that row `SubmitFailed` with an outcome-unknown message and clear the pending identity. Do not resubmit automatically.

- [ ] **Step 5: Clear only when a new batch replaces the old batch**

`setBatchImagePaths()` and `clearBatchResults()` clear records, table, indices, pinned selection, pending batch confirmation, completion flag, and Batch context. They must not silently retain a confirmation target from the old batch.

- [ ] **Step 6: Show exact partial-failure feedback**

```cpp
ui->batchSummaryLabel->setText(
    QStringLiteral("批量检测未完成：已完成 %1/%2，失败文件：%3，原因：%4")
        .arg(batchResults_.size()).arg(batchImagePaths_.size())
        .arg(QFileInfo(batchImagePaths_.at(batchIndex_)).fileName()).arg(message));
```

Set `batchCompletedSuccessfully_ = false` on failure and true only in `finishBatchPrediction()`.

- [ ] **Step 7: Prevent stale preview pixels**

If `updatePreview()` cannot read the selected file, clear its pixmap before showing `图片无法读取：<文件名>`. Keep textual evidence visible.

- [ ] **Step 8: Run GREEN and diff checks**

Run the full main-window suite and `git diff --check` for the two touched files. Expected: all PASS; no whitespace errors.

---

### Task 5: Regression, Qt 5.14.2 Release Build, and Verification

**Files:**
- Verify: `qt_app/mainwindow.h`
- Verify: `qt_app/mainwindow.cpp`
- Verify: `qt_app/mainwindow.ui`
- Verify: `qt_app/tests/test_mainwindow.cpp`
- Create: `docs/verification/main-inspection-workflow-polish-results.md`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: auditable results and a Qt 5.14.2 Release executable.

- [ ] **Step 1: Run the complete main-window suite**

```bat
cd /d E:\Project\wang\pp_813\qt_app\tests
release\test_mainwindow.exe -platform offscreen -o -,txt
```

Expected: every existing and new test passes.

- [ ] **Step 2: Run the other existing Qt suites**

```bat
release\test_appconfig.exe -platform offscreen -o -,txt
release\test_backendclient.exe -platform offscreen -o -,txt
release\test_backendprocessmanager.exe -platform offscreen -o -,txt
release\test_annotationmanager.exe -platform offscreen -o -,txt
release\test_geometryrulecanvas.exe -platform offscreen -o -,txt
release\test_geometrymaskmanager.exe -platform offscreen -o -,txt
```

Regenerate a missing executable from its matching `.pro` with the same qmake/MSVC environment, then run it. Expected: every suite passes.

- [ ] **Step 3: Build the application**

```powershell
Set-Location -LiteralPath 'E:\Project\wang\pp_813'
.\scripts\build_qt5.ps1
```

Expected: `Qt build succeeded` and `qt_app\build-release\release\workpiece_orientation.exe` exists.

- [ ] **Step 4: Perform the manual workflow check**

Use a batch with front, back, uncertain, and review results. Verify running-row viewing, exact image/evidence linkage, disabled confirmation until completion, first-review selection, three processing actions, automatic advance, workpiece mismatch blocking, switch-back recovery, and disconnect preservation.

- [ ] **Step 5: Write the verification record**

Create the document with exact headings:

```markdown
# 主检测窗口交互完善验证结果
## 修改范围
## 批量结果联动
## 确认状态机
## 异常与安全门
## 自动化测试
## Qt 5.14.2 Release 构建
## 人工验收
## 已知限制
```

Record actual pass counts, executable path, build timestamp, and environmental limitations. Do not claim an unperformed manual check.

- [ ] **Step 6: Run final checks**

```bat
git diff --check -- qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/tests/test_mainwindow.cpp docs/verification/main-inspection-workflow-polish-results.md
git status --short
```

Expected: no whitespace errors and no model/backend/geometry-manager files newly changed by this implementation.

- [ ] **Step 7: Preserve the dirty-worktree boundary**

Do not create a mixed implementation commit while the same files contain pre-existing uncommitted work. Report the exact modified files and leave the verified implementation in the working tree. If an isolated implementation-only staging mechanism becomes available, commit only that delta as `feat: polish batch inspection workflow`.
