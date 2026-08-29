# Qt UI Information Architecture Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Qt 5.14.2 desktop application around a detection-first, single-window information architecture with a polished industrial visual system while preserving every existing recognition, template, geometry, and recovery contract.

**Architecture:** MainWindow becomes the persistent application shell and the single BackendClient request coordinator. InspectionPage, WorkpieceLibraryPage, and GeometryRulesPage own their page-local state and emit command intents; MainWindow serializes and routes those commands without changing the model protocol semantics. The migration is staged so every task ends with a buildable, testable application.

**Tech Stack:** C++17, Qt 5.14.2 Widgets/Network/TestLib, Qt Designer forms, qmake, MSVC 2017 ABI with Visual Studio 2019 build tools, Python 3.10, pytest, JSON-lines TCP protocol v1, existing PP-ShiTuV2/ALIKED/LightGlue backend.

## Global Constraints

- Keep Qt 5.14.2 compatibility; do not use Qt 6-only APIs and do not migrate to QML or Web UI.
- Use the 1920×1080 light industrial design approved in the specification.
- Do not modify PP-ShiTuV2, ALIKED, LightGlue, model weights, fusion thresholds, geometry fitting semantics, or training behavior.
- Do not add login, roles, camera capture protocols, backend request parallelism, QML, Web UI, or a fourth top-level page.
- Keep BackendClient single-request serialization, handshake, timeout, restart ownership, and connection recovery behavior.
- The only protocol extension allowed is additive read-only workpiece summary/detail data and additive evolution progress fields; protocol version remains 1 and existing fields retain their meaning.
- Raw global/local scores and margins are not calibrated probabilities; never render them as percentage confidence.
- Preserve arbitrary positive front/back template counts, unequal counts, all selected images, old 5+5 libraries, and atomic cache switching.
- Preserve geometry profile revisions, original-image coordinates, bidirectional calibration, validation, publish, migration, and rollback behavior.
- Preserve completed images, results, evidence, and drafts across page navigation and backend failure.
- Retain existing critical objectName values until their tests have migrated.
- Do not delete legacy annotation implementation files in this change; only remove obsolete forwarding controls created by the UI migration.
- Every task follows RED → GREEN → focused regression → commit.

---

## Scope Check

The specification contains three pages, but they are not independent deployable subsystems: all three share one MainWindow, one current detection workpiece, one BackendClient, one in-flight request, and cross-page long-running task state. Splitting them into unrelated plans would duplicate or contradict request ownership. This plan therefore uses four ordered workstreams inside one executable:

1. visual foundation and application shell;
2. inspection workflow;
3. workpiece library workflow and its additive read-only data;
4. geometry-rule page migration and final integration.

Each task leaves the application buildable and has a reviewer-sized acceptance boundary.

## File Structure

### New shared UI files

- Create: qt_app/apptheme.h, qt_app/apptheme.cpp — install the approved palette and semantic widget properties.
- Create: qt_app/resources/theme.qss — central Qt stylesheet; no per-page hard-coded colors.
- Create: qt_app/resources.qrc — package the stylesheet and the final monochrome SVG icon set.
- Create: qt_app/appheader.h, qt_app/appheader.cpp — page navigation, current detection workpiece, backend status, diagnostics, restart.
- Create: qt_app/taskstatuswidget.h, qt_app/taskstatuswidget.cpp — one persistent cross-page task/progress/status surface.
- Create: scripts/run_qt5_tests.ps1 — reproducible Qt 5.14.2/MSVC test builder and runner with one isolated build directory per target.
- Create: qt_app/tests/test_appfoundation.cpp, qt_app/tests/test_appfoundation.pro — fast theme/header/status tests.

### Inspection files

- Create: qt_app/inspectiontypes.h — stable result identities and inspection enums.
- Create: qt_app/inspectionimageview.h, qt_app/inspectionimageview.cpp — dropped-image acceptance, native image display, zoom, pan and reset for the inspection page.
- Create: qt_app/inspectionpage.h, qt_app/inspectionpage.cpp, qt_app/inspectionpage.ui — single/batch state, preview, result card, evidence, review and filters.
- Create: qt_app/tests/test_inspectionpage.cpp, qt_app/tests/test_inspectionpage.pro — page-local tests without a real socket.

### Workpiece library files

- Create: qt_app/workpiecelibrarypage.h, qt_app/workpiecelibrarypage.cpp, qt_app/workpiecelibrarypage.ui — browsing, editing, registration, details, deletion and evolution jobs.
- Create: qt_app/tests/test_workpiecelibrarypage.cpp, qt_app/tests/test_workpiecelibrarypage.pro — page-local validation and state tests.
- Modify: src/workpiece_library.py — per-template inventory metadata for new writes with read-only old-manifest fallback.
- Modify: src/workpiece_catalog.py — read-only workpiece summaries and details.
- Modify: src/template_evolution.py — persistent phase/completed/total progress.
- Modify: src/orientation_tcp_service.py — additive list fields and get_workpiece_details.
- Modify: tests/test_workpiece_library.py, tests/test_workpiece_catalog.py, tests/test_template_evolution.py, tests/test_orientation_tcp_service.py.

### Geometry files

- Rename: qt_app/geometrymaskmanager.h → qt_app/geometryrulespage.h.
- Rename: qt_app/geometrymaskmanager.cpp → qt_app/geometryrulespage.cpp.
- Rename: qt_app/tests/test_geometrymaskmanager.cpp → qt_app/tests/test_geometryrulespage.cpp.
- Rename: qt_app/tests/test_geometrymaskmanager.pro → qt_app/tests/test_geometryrulespage.pro.
- Modify: qt_app/geometryrulecanvas.h, qt_app/geometryrulecanvas.cpp — independent guide/fitted/effective/mask visibility.
- Modify: qt_app/tests/test_geometryrulecanvas.cpp.

### Application integration

- Modify: qt_app/main.cpp — high-DPI attributes and one-time theme installation.
- Modify: qt_app/mainwindow.h, qt_app/mainwindow.cpp, qt_app/mainwindow.ui — application shell, command ownership, response routing and navigation guard.
- Modify: qt_app/workpiece_orientation.pro, qt_app/tests/test_appfoundation.pro, qt_app/tests/test_inspectionpage.pro, qt_app/tests/test_workpiecelibrarypage.pro, qt_app/tests/test_geometryrulecanvas.pro, qt_app/tests/test_geometryrulespage.pro and qt_app/tests/test_mainwindow.pro.
- Modify: qt_app/tests/test_mainwindow.cpp — cross-page and backend integration regressions.
- Create: docs/verification/qt-ui-information-architecture-redesign-results.md — actual test, build, visual and timing evidence.

---

### Task 1: Establish the Theme and Shared Status Components

**Files:**
- Create: qt_app/apptheme.h
- Create: qt_app/apptheme.cpp
- Create: qt_app/resources/theme.qss
- Create: qt_app/resources.qrc
- Create: qt_app/taskstatuswidget.h
- Create: qt_app/taskstatuswidget.cpp
- Create: qt_app/tests/test_appfoundation.cpp
- Create: qt_app/tests/test_appfoundation.pro
- Create: scripts/run_qt5_tests.ps1
- Modify: qt_app/main.cpp
- Modify: qt_app/workpiece_orientation.pro

**Interfaces:**
- Produces: `QString AppTheme::styleSheet()`, `void AppTheme::apply(QApplication *)`, `TaskStatusWidget::setIdle()`, `TaskStatusWidget::setRunning(const QString &, const QString &, int, int, qint64)`, `TaskStatusWidget::setMessage(MessageKind, const QString &, const QString &)`, and `TaskStatusWidget::actionRequested()`.
- Consumes: QApplication and ordinary Qt dynamic properties only; no backend dependency.

- [ ] **Step 1: Write the failing foundation tests**

Create test_appfoundation.cpp with:

    #include <QtTest>
    #include <QApplication>
    #include <QLabel>
    #include <QProgressBar>
    #include "apptheme.h"
    #include "taskstatuswidget.h"

    class TestAppFoundation : public QObject {
        Q_OBJECT
    private slots:
        void themeContainsApprovedTokens() {
            const QString qss = AppTheme::styleSheet();
            QVERIFY(qss.contains(QStringLiteral("#F4F6F8"), Qt::CaseInsensitive));
            QVERIFY(qss.contains(QStringLiteral("#2563EB"), Qt::CaseInsensitive));
            QVERIFY(qss.contains(QStringLiteral("#15803D"), Qt::CaseInsensitive));
            QVERIFY(qss.contains(QStringLiteral("#C9362B"), Qt::CaseInsensitive));
        }

        void taskStatusKeepsActionableState() {
            TaskStatusWidget widget;
            widget.setRunning(QStringLiteral("建立工件库"),
                              QStringLiteral("提取特征"), 7, 20, 1250);
            QCOMPARE(widget.findChild<QProgressBar *>(
                         QStringLiteral("globalTaskProgressBar"))->value(), 7);
            QVERIFY(widget.findChild<QLabel *>(
                        QStringLiteral("globalTaskTitleLabel"))->text()
                        .contains(QStringLiteral("建立工件库")));
            widget.setMessage(TaskStatusWidget::MessageKind::Error,
                              QStringLiteral("后端断开"), QStringLiteral("重试"));
            QCOMPARE(widget.property("messageKind").toString(),
                     QStringLiteral("error"));
        }
    };

    QTEST_MAIN(TestAppFoundation)
    #include "test_appfoundation.moc"

- [ ] **Step 2: Add the reproducible Qt test runner**

Create `scripts/run_qt5_tests.ps1` with this complete content:

    param(
        [string[]]$Targets = @(
            'test_appconfig', 'test_backendclient', 'test_backendprocessmanager',
            'test_annotationmanager', 'test_appfoundation', 'test_inspectionpage',
            'test_workpiecelibrarypage', 'test_geometryrulecanvas',
            'test_geometryrulespage', 'test_mainwindow'
        ),
        [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
        [string]$QtBin = 'E:\QT\5.14\5.14.2\msvc2017_64\bin',
        [string]$Jom = 'E:\QT\5.14\Tools\QtCreator\bin\jom.exe',
        [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat'
    )

    $qmake = Join-Path $QtBin 'qmake.exe'
    foreach ($required in @($qmake, $Jom, $VcVars)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Required Qt test path does not exist: $required"
        }
    }

    foreach ($target in $Targets) {
        $project = Join-Path $ProjectRoot "qt_app\tests\$target.pro"
        $buildDir = Join-Path $ProjectRoot "qt_app\tests\build-$target"
        if (-not (Test-Path -LiteralPath $project)) {
            throw "Qt test project does not exist: $project"
        }
        New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
        Push-Location -LiteralPath $buildDir
        try {
            $configure = ('call "' + $VcVars + '" && "' + $qmake + '" "' + $project + '" CONFIG+=release')
            cmd.exe /d /s /c $configure
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

            $build = 'call "' + $VcVars + '" && "' + $Jom + '"'
            cmd.exe /d /s /c $build
            if ($LASTEXITCODE -ne 0) {
                $fallback = ('call "' + $VcVars + '" && nmake /f Makefile.Release /NOLOGO')
                cmd.exe /d /s /c $fallback
            }
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

            $executable = Join-Path $buildDir "release\$target.exe"
            if (-not (Test-Path -LiteralPath $executable)) {
                throw "Qt test executable does not exist: $executable"
            }
            $previousPlatform = $env:QT_QPA_PLATFORM
            try {
                $env:QT_QPA_PLATFORM = 'offscreen'
                & $executable '-platform' 'offscreen' '-o' '-,txt'
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            } finally {
                $env:QT_QPA_PLATFORM = $previousPlatform
            }
        } finally {
            Pop-Location
        }
    }

- [ ] **Step 3: Build and verify RED**

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_appfoundation

Expected: build fails because AppTheme and TaskStatusWidget do not exist.

- [ ] **Step 4: Implement the public theme and status interfaces**

Create apptheme.h:

    #pragma once
    #include <QString>
    class QApplication;
    namespace AppTheme {
    QString styleSheet();
    void apply(QApplication *application);
    }

Create taskstatuswidget.h:

    #pragma once
    #include <QWidget>

    class QLabel;
    class QProgressBar;
    class QPushButton;

    class TaskStatusWidget : public QWidget {
        Q_OBJECT
    public:
        enum class MessageKind { Neutral, Success, Warning, Error };
        Q_ENUM(MessageKind)

        explicit TaskStatusWidget(QWidget *parent = nullptr);
        void setIdle();
        void setRunning(const QString &title, const QString &phase,
                        int completed, int total, qint64 elapsedMs);
        void setMessage(MessageKind kind, const QString &text,
                        const QString &actionText = QString());
    signals:
        void actionRequested();
    private:
        QLabel *titleLabel_;
        QLabel *detailLabel_;
        QProgressBar *progressBar_;
        QPushButton *actionButton_;
    };

AppTheme::apply must set the loaded stylesheet exactly once. TaskStatusWidget sets dynamic property messageKind to neutral, success, warning or error and repolishes itself after property changes.

- [ ] **Step 5: Add the approved QSS and resource registration**

theme.qss must define:

    QMainWindow, QWidget[pageRoot="true"] { background: #F4F6F8; color: #17212B; font-family: "Microsoft YaHei"; font-size: 14px; }
    QLabel { background: transparent; color: #17212B; }
    QFrame[panel="true"] { background: #FFFFFF; border: 1px solid #D9E0E7; border-radius: 6px; }
    QPushButton[role="primary"] { background: #2563EB; color: #FFFFFF; border: 1px solid #2563EB; min-height: 34px; }
    QPushButton[role="danger"] { color: #C9362B; border: 1px solid #C9362B; background: #FFFFFF; min-height: 34px; }
    QWidget[messageKind="success"] { color: #15803D; }
    QWidget[messageKind="warning"] { color: #B7791F; }
    QWidget[messageKind="error"] { color: #C9362B; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #FFFFFF; border: 1px solid #C9D2DC; min-height: 32px; padding: 0 8px; }
    QTableView { background: #FFFFFF; alternate-background-color: #F7F9FB; selection-background-color: #DCE9FF; selection-color: #17212B; }

Register theme.qss under :/theme/theme.qss in resources.qrc. Add apptheme.cpp, taskstatuswidget.cpp and resources.qrc to workpiece_orientation.pro and test_appfoundation.pro. Call AppTheme::apply(&application) once after QApplication construction.

- [ ] **Step 6: Run GREEN**

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_appfoundation

Expected: all foundation tests pass.

- [ ] **Step 7: Run the existing MainWindow smoke test**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_mainwindow

Expected: all current tests pass because no layout behavior changed.

- [ ] **Step 8: Commit**

    git add qt_app/apptheme.h qt_app/apptheme.cpp qt_app/taskstatuswidget.h qt_app/taskstatuswidget.cpp qt_app/resources/theme.qss qt_app/resources.qrc qt_app/main.cpp qt_app/workpiece_orientation.pro qt_app/tests/test_appfoundation.cpp qt_app/tests/test_appfoundation.pro scripts/run_qt5_tests.ps1
    git commit -m "feat: add Qt application theme foundation"

---

### Task 2: Build the Persistent Application Shell and Navigation

**Files:**
- Create: qt_app/appheader.h
- Create: qt_app/appheader.cpp
- Modify: qt_app/mainwindow.h
- Modify: qt_app/mainwindow.cpp
- Modify: qt_app/mainwindow.ui
- Modify: qt_app/workpiece_orientation.pro
- Modify: qt_app/tests/test_appfoundation.cpp
- Modify: qt_app/tests/test_appfoundation.pro
- Modify: qt_app/tests/test_mainwindow.cpp
- Modify: qt_app/tests/test_mainwindow.pro

**Interfaces:**
- Consumes: AppTheme and TaskStatusWidget from Task 1.
- Produces: AppPage, BackendUiState, AppHeader, MainWindow::requestPage(AppPage), and a persistent QStackedWidget named mainPageStack.

- [ ] **Step 1: Write failing header and navigation tests**

Add to test_appfoundation.cpp:

Include `<QComboBox>`, `<QLabel>`, `<QSignalSpy>`, `<QStackedWidget>` and `appheader.h` in the respective test files before adding:

    void headerDoesNotChangeDetectionTargetWhileBrowsing() {
        AppHeader header;
        QSignalSpy spy(&header, &AppHeader::currentWorkpieceRequested);
        header.setWorkpieces({{QStringLiteral("m1"), QStringLiteral("M1")},
                              {QStringLiteral("m2"), QStringLiteral("M2")}},
                             QStringLiteral("m1"));
        header.setCurrentWorkpieceId(QStringLiteral("m1"));
        QCOMPARE(header.currentWorkpieceId(), QStringLiteral("m1"));
        QCOMPARE(spy.count(), 0);
    }

Add to test_mainwindow.cpp:

    void startsOnInspectionAndPreservesHeaderAcrossNavigation() {
        MainWindow window;
        auto *stack = window.findChild<QStackedWidget *>(
            QStringLiteral("mainPageStack"));
        QVERIFY(stack != nullptr);
        QCOMPARE(stack->currentIndex(), 0);
        QVERIFY(QMetaObject::invokeMethod(&window, "showWorkpieceLibrary",
                                          Qt::DirectConnection));
        QCOMPARE(stack->currentIndex(), 1);
        QVERIFY(window.findChild<QLabel *>(
                    QStringLiteral("backendStatusLabel")) != nullptr);
        QVERIFY(window.findChild<QComboBox *>(
                    QStringLiteral("workpieceComboBox")) != nullptr);
    }

- [ ] **Step 2: Build and verify RED**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_appfoundation,test_mainwindow

Expected: AppHeader and mainPageStack are missing.

- [ ] **Step 3: Implement AppHeader**

Create appheader.h:

    #pragma once
    #include <QWidget>
    #include <QList>
    #include <QPair>

    enum class AppPage { Inspection = 0, WorkpieceLibrary = 1, GeometryRules = 2 };
    Q_DECLARE_METATYPE(AppPage)

    enum class BackendUiState { Disconnected, Loading, Ready, Busy, Error };

    struct BackendStatusDetails {
        BackendUiState state = BackendUiState::Disconnected;
        QString connectionDetail;
        QString modelDetail;
        QString currentTask;
        QString recentError;
        bool canRestart = false;
    };

    class AppHeader : public QWidget {
        Q_OBJECT
    public:
        explicit AppHeader(QWidget *parent = nullptr);
        void setCurrentPage(AppPage page);
        void setWorkpieces(const QList<QPair<QString, QString>> &items,
                           const QString &currentWorkpieceId);
        void setCurrentWorkpieceId(const QString &workpieceId);
        QString currentWorkpieceId() const;
        void setBackendState(BackendUiState state, const QString &detail);
        void setBackendDetails(const BackendStatusDetails &details);
    signals:
        void pageRequested(AppPage page);
        void currentWorkpieceRequested(const QString &workpieceId);
        void backendDetailsRequested();
        void restartBackendRequested();
    };

Use three checkable navigation buttons, one QComboBox with objectName workpieceComboBox, QLabel backendStatusLabel, and separate details/restart buttons. Blocking signals in setWorkpieces and setCurrentWorkpieceId is mandatory. AppHeader owns a collapsed child QFrame named backendDetailsPanel. Clicking backendStatusLabel's adjacent details button toggles that panel; it renders connectionDetail, modelDetail, currentTask and recentError as labelled text, and shows restart only when canRestart is true. It is a child panel, never a new top-level window.

Delete the old `workpieceComboBox` from the migrated central content in this task so only AppHeader owns that objectName. Replace every old MainWindow lookup with `appHeader_->currentWorkpieceId()`. The later library browse control must be named `libraryWorkpieceList`, never `workpieceComboBox`.

- [ ] **Step 4: Replace MainWindow’s top-level form with a shell**

mainwindow.ui must contain only:

- a host layout for AppHeader;
- QStackedWidget mainPageStack;
- three persistent page containers in order `inspectionPageHost`, `libraryPageHost`, `geometryPageHost`, each with a zero-margin host layout named `<page>HostLayout`;
- a host layout for TaskStatusWidget.

Move the old central content into the first stacked container temporarily so current behavior remains reachable. Add one temporary QLabel reading `工件库页面将在 Task 7 接入` to stack index 1 and one reading `几何规则页面将在 Task 9 接入` to stack index 2; those labels are deleted in their named tasks. Do not duplicate the existing controls.

Add slots:

    void showInspection();
    void showWorkpieceLibrary();
    void showGeometryRules();

and:

    bool MainWindow::requestPage(AppPage page) {
        if (page == currentPage_) return true;
        mainPageStack_->setCurrentIndex(static_cast<int>(page));
        currentPage_ = page;
        appHeader_->setCurrentPage(page);
        return true;
    }

Declare the corresponding members explicitly in `mainwindow.h` and use `ui->mainPageStack` consistently rather than mixing generated and manual pointers:

    AppHeader *appHeader_ = nullptr;
    TaskStatusWidget *globalTaskStatus_ = nullptr;
    AppPage currentPage_ = AppPage::Inspection;

- [ ] **Step 5: Route backend and current-workpiece state through AppHeader**

Map BackendClient states to BackendUiState. AppHeader’s currentWorkpieceRequested becomes the only implicit change to the detection target. Merely navigating or selecting a future library detail row must not call it.

Add `backendDetailsPanelShowsRecoveryContext()` to `test_appfoundation.cpp`: populate all five `BackendStatusDetails` fields, click the details button, assert that the connection, model, current task and recent error strings are visible, and assert that the restart button is visible only when `canRestart` is true. This covers the specification's backend diagnostic entry without adding another window.

- [ ] **Step 6: Run GREEN and focused regressions**

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_appfoundation,test_mainwindow

Expected: new navigation tests and every existing test pass.

- [ ] **Step 7: Commit**

    git add qt_app/appheader.h qt_app/appheader.cpp qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/workpiece_orientation.pro qt_app/tests/test_appfoundation.cpp qt_app/tests/test_appfoundation.pro qt_app/tests/test_mainwindow.cpp qt_app/tests/test_mainwindow.pro
    git commit -m "feat: add persistent Qt application navigation"

---

### Task 3: Extract the Single-Image Inspection Page

**Files:**
- Create: qt_app/inspectiontypes.h
- Create: qt_app/inspectionimageview.h
- Create: qt_app/inspectionimageview.cpp
- Create: qt_app/inspectionpage.h
- Create: qt_app/inspectionpage.cpp
- Create: qt_app/inspectionpage.ui
- Create: qt_app/tests/test_inspectionpage.cpp
- Create: qt_app/tests/test_inspectionpage.pro
- Modify: qt_app/mainwindow.h
- Modify: qt_app/mainwindow.cpp
- Modify: qt_app/mainwindow.ui
- Modify: qt_app/workpiece_orientation.pro
- Modify: qt_app/tests/test_mainwindow.cpp
- Modify: qt_app/tests/test_mainwindow.pro

**Interfaces:**
- Consumes: `AppHeader::currentWorkpieceId()` and MainWindow's existing `BackendClient`.
- Produces: `InspectionRecord`, `InspectionImageView::setImagePath(const QString &)`, `InspectionImageView::resetView()`, `InspectionImageView::imageDropped(const QString &)`, `InspectionPage::commandRequested(const QString &, const QJsonObject &)`, `InspectionPage::handleBackendResponse(const QString &, const QJsonObject &)`, `InspectionPage::handleBackendFailure(const QString &, const QString &, const QString &)`, and page-local single/recent-result state.

- [ ] **Step 1: Write valid-image test helpers and failing single-page tests**

In `test_inspectionpage.cpp`, define these helpers before the test class so no test relies on workstation-specific paths:

    static QString writeImage(QTemporaryDir &directory, const QString &name) {
        const QString path = directory.filePath(name);
        QImage image(48, 48, QImage::Format_RGB32);
        image.fill(Qt::darkGray);
        return image.save(path) ? path : QString();
    }

    static InspectionRecord resultRecord(const QString &id,
                                         const QString &path) {
        InspectionRecord record;
        record.id = id;
        record.imagePath = path;
        record.workpieceId = QStringLiteral("m1");
        record.response = QJsonObject{
            {QStringLiteral("label"), QStringLiteral("front")},
            {QStringLiteral("global_margin"), 0.03},
            {QStringLiteral("local_margin"), 7.3},
            {QStringLiteral("needs_review"), false}
        };
        record.label = QStringLiteral("front");
        record.elapsedMs = 12.0;
        record.completedAt = QDateTime::currentDateTime();
        return record;
    }

Add exact tests `resultShowsDecisionStateWithoutPercentConfidence`, `backendFailurePreservesVisibleImageAndEvidence`, `rawEvidenceStartsCollapsed`, `selectingRecentRecordDoesNotResubmitPrediction`, `droppedImageSelectsButDoesNotPredict`, and `zoomAndResetDoNotChangeImagePath`. Each creates a `QTemporaryDir`, calls `writeImage`, and asserts the path is non-empty. The recent-record test calls `showSingleResult` for IDs `one` and `two`, attaches `QSignalSpy` to `commandRequested`, calls `selectRecentRecord("one")`, then asserts the first path/evidence are restored and the spy count remains zero.

- [ ] **Step 2: Build and verify RED**

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage

Expected: fail because the page and image-view classes do not exist.

- [ ] **Step 3: Define stable inspection types**

Create `inspectiontypes.h`:

    #pragma once
    #include <QDateTime>
    #include <QJsonObject>
    #include <QString>

    enum class InspectionMode { Single, Batch };
    enum class InspectionUiState {
        Idle, Running, Completed, NeedsReview, Failed, BackendUnavailable
    };
    enum class BatchFilter { All, NeedsReview, Unprocessed, Failed };
    enum class BatchDisposition {
        Pending, Submitting, PredictionFailed,
        QueuedFront, QueuedBack, Rejected, SubmitFailed
    };

    struct InspectionRecord {
        QString id;
        QString imagePath;
        QString workpieceId;
        QJsonObject response;
        QString label;
        bool needsReview = false;
        double elapsedMs = 0.0;
        QDateTime completedAt;
        BatchDisposition disposition = BatchDisposition::Pending;
        QString evolutionJobId;
        QString error;
    };

- [ ] **Step 4: Implement the inspection image view**

Create `inspectionimageview.h` with this public boundary:

    class InspectionImageView : public QGraphicsView {
        Q_OBJECT
    public:
        explicit InspectionImageView(QWidget *parent = nullptr);
        bool setImagePath(const QString &path);
        QString imagePath() const;
        qreal zoomFactor() const;
    public slots:
        void resetView();
    signals:
        void imageDropped(const QString &path);
    protected:
        void dragEnterEvent(QDragEnterEvent *event) override;
        void dropEvent(QDropEvent *event) override;
        void wheelEvent(QWheelEvent *event) override;
    };

Accept exactly one local file with suffix png, jpg, jpeg, bmp, tif or tiff that `QImageReader` can decode. A drop emits `imageDropped(path)`; InspectionPage selects it but does not predict until the predict button is pressed. Clamp wheel zoom to 0.1–8.0, use `ScrollHandDrag` for panning, and make reset fit the whole image without modifying the source path or pixels.

- [ ] **Step 5: Implement InspectionPage's single-image contract**

Expose:

    void setMode(InspectionMode mode);
    InspectionMode mode() const;
    InspectionUiState uiState() const;
    void setCurrentWorkpiece(const QString &id, const QString &name);
    void setBackendAvailable(bool available, bool busy, const QString &reason);
    void setSingleImagePath(const QString &path);
    QString singleImagePath() const;
    void showSingleResult(const InspectionRecord &record);
    void showSingleFailure(const QString &message);
    void selectRecentRecord(const QString &recordId);
    void handleBackendResponse(const QString &command, const QJsonObject &response);
    void handleBackendFailure(const QString &command, const QString &code,
                              const QString &message);

    signals:
        void commandRequested(const QString &command, const QJsonObject &fields);
        void confirmationRequested(const QString &recordId,
                                   const QString &workpieceId,
                                   const QString &imagePath,
                                   const QString &orientation);
        void rejectionRequested(const QString &recordId);

The `.ui` uses a 65/35 QSplitter, embeds `InspectionImageView`, places choose/drop/predict controls next to the image area, preserves currentImageLabel, resultLabel, reviewLabel, evidenceTextEdit, currentResultTargetLabel, confirmFrontButton, confirmBackButton and rejectConfirmationButton, and adds resultMessageLabel, rawEvidenceContainer, recentInspectionList and zoom/reset controls. Each recent item renders an image thumbnail, result, completion time, review marker and入库 state. `rejectConfirmationButton` means “不入库” and uses the secondary outlined role, not the destructive role. The visible evidence summary contains three or four plain-language lines for global feature, local match, geometry rule and review reason; raw scores/candidates/timing appear only in the collapsed detail container and never as calibrated percentages.

Keep the latest 50 completed or failed single records in memory. Each record stores completion time, review flag, result and asynchronous入库 state. Store each stable record ID in `Qt::UserRole`; selection restores image, result, evidence, review and confirmation target without emitting predict. Single and batch states are separate members selected by `InspectionMode`; switching mode must not overwrite either state. Map page states explicitly to idle/running/completed/needs-review/failed/backend-unavailable; entering a failure or disconnect state preserves the current image/result.

When prediction is front, order/label the three actions as `确认正面`, `修正为反面`, `不入库`; reverse the first two when prediction is back. Add `predictedOrientationOrdersConfirmCorrectRejectActions()` and assert both orientations. Confirmation emits the stable record ID and MainWindow maps the returned evolution job ID back to that record; display `等待写入`, `正在更新缓存`, `已参与预测`, or `写入失败` without blocking the next inspection.

- [ ] **Step 6: Add explicit MainWindow command ownership**

Declare:

    enum class CommandOwner { None, System, Inspection, Library, Geometry };
    void sendPageCommand(CommandOwner owner, const QString &command,
                         const QJsonObject &fields);
    void clearPendingCommand();

    CommandOwner pendingOwner_ = CommandOwner::None;
    QString pendingCommand_;

MainWindow sets pendingOwner_ and pendingCommand_ before `BackendClient::sendRequest`. On response/failure it copies and clears both before invoking the page handler, so a handler may emit the next serial command. Connect `InspectionPage::commandRequested` to `sendPageCommand(CommandOwner::Inspection, command, fields)`.

- [ ] **Step 7: Move single-image integration assertions**

Move only prediction-specific assertions under InspectionPage in `predictionShowsRawEvidenceWithoutPercentConfidence` and `transportFailurePreservesPreviousPredictionAndMarksOutcomeUnknown`. In `busyStateDisablesRegisterAndPredict` and `loadingStateShowsStatusAndKeepsActionsDisabled`, prediction assertions target InspectionPage while registration assertions remain in the temporary library page until Task 7. Add `returningToInspectionPreservesDisplayedImageAndResult` and `switchingModesDoesNotOverwriteEitherResult`.

- [ ] **Step 8: Run GREEN**

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage,test_mainwindow

Expected: all page tests and single-image integration tests pass.

- [ ] **Step 9: Commit**

    git add qt_app/inspectiontypes.h qt_app/inspectionimageview.h qt_app/inspectionimageview.cpp qt_app/inspectionpage.h qt_app/inspectionpage.cpp qt_app/inspectionpage.ui qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/workpiece_orientation.pro qt_app/tests/test_inspectionpage.cpp qt_app/tests/test_inspectionpage.pro qt_app/tests/test_mainwindow.cpp qt_app/tests/test_mainwindow.pro
    git commit -m "refactor: extract Qt inspection page"

---

### Task 4: Add Stable Batch Identity, Filters, and Safe Stop

**Files:**
- Modify: qt_app/inspectiontypes.h
- Modify: qt_app/inspectionpage.h
- Modify: qt_app/inspectionpage.cpp
- Modify: qt_app/inspectionpage.ui
- Modify: qt_app/tests/test_inspectionpage.cpp
- Modify: qt_app/tests/test_mainwindow.cpp

**Interfaces:**
- Consumes: `InspectionPage` and `CommandOwner::Inspection` from Task 3.
- Produces: `beginBatch(const QStringList &, const QString &)`, `setBatchFilter(BatchFilter)`, `selectedRecordId()`, `batchRecordIds()`, `requestBatchStop()`, completed/failed counters, stop-after-current semantics, and confirmation routing by stable record ID.

- [ ] **Step 1: Add the response helper and failing stable-identity tests**

In `test_inspectionpage.cpp`, include `<QTableWidget>` and add:

    static QJsonObject predictionResponse(const QString &label,
                                          bool needsReview = false) {
        return QJsonObject{
            {QStringLiteral("label"), label},
            {QStringLiteral("needs_review"), needsReview},
            {QStringLiteral("global_margin"), 0.04},
            {QStringLiteral("local_margin"), 6.0},
            {QStringLiteral("elapsed_ms"), 11.0}
        };
    }

Add `batchIdsAreNonEmptyAndUnique()`: create three real images with Task 3's `writeImage`, call `beginBatch`, then assert `batchRecordIds()` has size 3, contains no empty value, and `QSet<QString>(ids.begin(), ids.end()).size() == 3`.

Add `filteredSelectionStillTargetsTheSameRecord()`: create two real images, call `beginBatch`, capture the two IDs, feed a non-review response followed by a review response, set `BatchFilter::NeedsReview`, select visible row 0, and assert `selectedRecordId()` equals the second original ID.

Add `failedFilterIncludesPredictionAndConfirmationFailures()` and assert that one `PredictionFailed` record and one `SubmitFailed` record are visible while a successful record is hidden.

- [ ] **Step 2: Add the failing stop-after-current test**

Create three real images in a `QTemporaryDir`; attach `QSignalSpy` to `commandRequested`; call `beginBatch`; assert one predict was emitted; call `requestBatchStop`; feed the current predict response; assert no second predict was emitted, `completedBatchCount() == 1`, and the two unsent records remain `Pending` and visible under `Unprocessed`.

- [ ] **Step 3: Run RED**

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage

Expected: fail because the batch ID/filter/stop APIs are absent.

- [ ] **Step 4: Define and implement the batch API**

Expose:

    void beginBatch(const QStringList &paths, const QString &workpieceId);
    void setBatchFilter(BatchFilter filter);
    BatchFilter batchFilter() const;
    QString selectedRecordId() const;
    QStringList batchRecordIds() const;
    void requestBatchStop();
    int completedBatchCount() const;
    int failedBatchCount() const;
    void setRecordDisposition(const QString &recordId,
                              BatchDisposition disposition,
                              const QString &evolutionJobId = QString(),
                              const QString &error = QString());

`beginBatch` validates every path, creates exactly one `InspectionRecord` per input in input order, assigns `QUuid::createUuid().toString(QUuid::WithoutBraces)` once, and reuses that ID through submission, response, filtering and confirmation. Store the current request ID, selected ID, original order, filter, cursor, fixed batch workpiece ID, selection-pinned flag and stop flag inside InspectionPage. Do not derive an ID from a table row or path.

In batch mode, replace the recent strip with `batchResultsTableWidget` inside a vertical splitter so its height is adjustable while reusing the same image view/result/evidence card. The table columns are file, prediction, review, elapsed time and processing state; filter tabs and text counters remain above it.

Each table item stores the ID:

    item->setData(Qt::UserRole, record.id);

Every selection, response, confirmation, failure and auto-advance first resolves the record by ID. MainWindow remembers the pending confirmation record ID, sends a unique operation_id, then calls `setRecordDisposition` with the returned job ID; later evolution polling updates every record carrying that job ID. Coalesced jobs may update multiple records but never select by row number or path alone. A command-level predict failure marks that record `PredictionFailed`, retains its error and continues only if MainWindow reports the backend still Ready. A transport failure marks the current outcome unknown and stops submitting further images.

- [ ] **Step 5: Implement the four filters and counters**

Use:

    bool InspectionPage::matchesFilter(const InspectionRecord &record) const {
        switch (batchFilter_) {
        case BatchFilter::All:
            return true;
        case BatchFilter::NeedsReview:
            return record.needsReview;
        case BatchFilter::Unprocessed:
            return record.disposition == BatchDisposition::Pending
                || record.disposition == BatchDisposition::SubmitFailed;
        case BatchFilter::Failed:
            return record.disposition == BatchDisposition::PredictionFailed
                || record.disposition == BatchDisposition::SubmitFailed
                || !record.error.isEmpty();
        }
        return true;
    }

The summary shows processed, successful, needs-review and failed counts as text plus progress. Rebuilding the visible table preserves `selectedRecordId_` if still visible; otherwise select the first visible needs-review record, then first pending record. Prediction failures count as processed and failed, not successful.

- [ ] **Step 6: Implement stop-after-current**

`requestBatchStop()` sets `stopRequested_`. If predict is in flight, its response/failure returns normally; afterward finish as stopped and emit no next predict. If nothing is in flight, finish immediately. Completed/failed rows remain inspectable and confirmable, unsent rows remain Pending, and no cancel protocol command is added.

- [ ] **Step 7: Preserve integration safety gates**

Keep passing `selectingBatchRowShowsItsImageAndEvidence`, `laterBatchResponsesDoNotReplaceManualSelection`, `batchCompletionSelectsFirstReviewResult`, `confirmingBatchResultUpdatesStatusAndAdvances`, `failedBatchConfirmationStaysSelectedAndCanRetry`, `workpieceSwitchPreservesBatchButBlocksConfirmationUntilSwitchedBack`, `switchingWorkpieceDuringBatchDoesNotRetargetPendingRequests`, `disconnectPreservesBatchEvidenceAndDisablesConfirmation`, `disconnectDuringBatchConfirmationMarksOutcomeUnknown`, and `midBatchFailureKeepsCompletedRowsAndShowsExactProgress`. Update the last test to assert that the failed image remains as an explicit failed record.

- [ ] **Step 8: Run GREEN**

Run:

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage,test_mainwindow

Expected: all inspection and MainWindow regressions pass.

- [ ] **Step 9: Commit**

    git add qt_app/inspectiontypes.h qt_app/inspectionpage.h qt_app/inspectionpage.cpp qt_app/inspectionpage.ui qt_app/tests/test_inspectionpage.cpp qt_app/tests/test_mainwindow.cpp
    git commit -m "feat: add robust Qt batch review mode"

---

### Task 5: Add Read-Only Workpiece Summaries and Details

**Files:**
- Modify: src/workpiece_library.py
- Modify: src/workpiece_catalog.py
- Modify: src/orientation_tcp_service.py
- Modify: tests/test_workpiece_library.py
- Modify: tests/test_workpiece_catalog.py
- Modify: tests/test_orientation_tcp_service.py

**Interfaces:**
- Produces: `WorkpieceLibrary.get_workpiece_metadata(workpiece_id) -> dict[str, Any]`, `WorkpieceLibrary.get_template_inventory(workpiece_id) -> list[dict[str, Any]]`, `WorkpieceCatalog.list_workpiece_summaries() -> list[dict[str, Any]]`, `WorkpieceCatalog.get_workpiece_details(workpiece_id) -> dict[str, Any]`, additive `list_workpieces` fields, and the protocol-v1 `get_workpiece_details` command.
- Consumes: existing `WorkpieceRecord`, manifest, geometry snapshot and `read_color_image`.

- [ ] **Step 1: Add concrete catalog test helpers and failing summary tests**

In `tests/test_workpiece_catalog.py`, reuse its existing `image` and `FakeClassifier` and add:

    class SummaryGeometryProfiles:
        def snapshot(self, workpiece_id):
            return {
                "profile_status": "ok",
                "active": {"rules": [{"rule_id": "glare"},
                                      {"rule_id": "intrusion"}]},
            }

    def create_catalog_with_counts(tmp_path, front_count, back_count):
        classifier = FakeClassifier()
        library = WorkpieceLibrary(tmp_path / "summary-library")
        catalog = WorkpieceCatalog(library, classifier, SummaryGeometryProfiles())
        front = [image(tmp_path / f"summary-front-{i}.png", 10 + i)
                 for i in range(front_count)]
        back = [image(tmp_path / f"summary-back-{i}.png", 80 + i)
                for i in range(back_count)]
        record, _ = catalog.register("M-summary", front, back, False)
        return catalog, classifier, record

Add tests with exact assertions:

    def test_workpiece_summary_reports_unequal_counts_and_rules(tmp_path):
        catalog, _, record = create_catalog_with_counts(tmp_path, 1, 12)
        summary = catalog.list_workpiece_summaries()[0]
        assert summary["id"] == record.id
        assert summary["template_counts"] == {"front": 1, "back": 12}
        assert summary["geometry_rule_count"] == 2
        assert summary["geometry_status"] == "ok"
        assert summary["detectable"] is True

Also add `test_details_return_all_unequal_and_over_thirty_templates` for 31+1, checking all 32 unique template IDs and no truncation.

- [ ] **Step 2: Write old-manifest no-rewrite/no-rebuild tests**

Register a real 5+5 library, remove `template_inventory` and `created_at` from its manifest, write it back, then save the exact manifest bytes. Call `get_workpiece_details` and assert: counts are 5+5, all ten entries use `source == "initial_registration"`, all `added_at is None`, and manifest bytes are unchanged. Extend the existing cache-first recovery test with this legacy manifest and assert the cache loader is used and `build_calls == 0`; reading details must never rebuild or rewrite a legacy library.

- [ ] **Step 3: Write inventory-maintenance tests**

In `tests/test_workpiece_library.py`, assert that registration writes one inventory entry per copied image with `source=initial_registration`; `prepare_append(base_record, front_images, back_images, fake_builder, operation_id="confirmed-1", source="confirmed_inspection")` preserves all old entries and adds every new entry with that source and a non-empty UTC `added_at`; an aborted or failed prepared append leaves the active manifest byte-for-byte unchanged; replace registration builds a fresh inventory only for the replacement images.

- [ ] **Step 4: Write failing protocol tests with defined fixtures**

Extend `tests/test_orientation_tcp_service.py::FakeLibrary` with `get`, `get_workpiece_metadata`, and `get_template_inventory` for workpiece `m7`. Return a one-front/twelve-back record and thirteen inventory dictionaries whose preview paths and readable flags are deterministic. Add:

    def test_workpiece_list_and_details_are_additive(client):
        listing = client.request("list_workpieces")
        item = listing["workpieces"][0]
        assert item["id"] == "m7"
        assert item["name"] == "M7"
        assert item["template_counts"] == {"front": 1, "back": 12}
        details = client.request("get_workpiece_details", workpiece_id="m7")
        assert details["ok"] is True
        assert len(details["workpiece"]["templates"]) == 13

    def test_unknown_workpiece_details_return_stable_code(client):
        response = client.request("get_workpiece_details",
                                  workpiece_id="missing")
        assert response["error"]["code"] == "WORKPIECE_NOT_FOUND"

- [ ] **Step 5: Run RED**

    E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_workpiece_library.py tests\test_workpiece_catalog.py tests\test_orientation_tcp_service.py -q -p no:cacheprovider --basetemp="$env:TEMP\pytest-ui-workpiece-red"

Expected: metadata, summary/detail APIs and service command are missing.

- [ ] **Step 6: Implement manifest metadata without forcing migration**

Keep the current schema version compatible and write this additive field for new registration/replacement/append manifests:

    "template_inventory": [
      {
        "template_id": "front:00.png",
        "direction": "front",
        "filename": "00.png",
        "source": "initial_registration",
        "added_at": "2026-08-25T00:00:00+00:00"
      }
    ]

`register()` writes every copied template. Extend `WorkpieceLibrary.prepare_append(base_record, front_images, back_images, build_cache, *, operation_id, progress_callback=None, source="manual_append")` and `WorkpieceCatalog.append_templates(workpiece_id, front_images, back_images, *, operation_id=None, progress_callback=None, source="manual_append")`; the catalog forwards source to the staging call. Staging preserves the stored or derived old inventory and appends all new entries; `TemplateEvolution` passes `source="confirmed_inspection"` in Task 6. Missing inventory is derived from `record.front_images/back_images` with `source=initial_registration`; `added_at` is `created_at` or `None` when unavailable. `get_workpiece_metadata` returns `updated_at or created_at or None`. Neither read method writes a manifest or touches a cache.

- [ ] **Step 7: Implement catalog summaries and details**

Under the catalog lock, derive `detectable` from `record.id in self._snapshots`, not a constant. `geometry_rule_count` is the exact length of active `rules`, and `geometry_status` comes from `profile_status` or `not_configured`. `get_workpiece_details` returns the same summary plus every template's stable ID, direction, absolute preview path, source, added_at and `readable = read_color_image(path) is not None`. Keep `WorkpieceCatalog.list_workpieces()` as a compatibility alias that returns `list_workpiece_summaries()` so existing id/name consumers receive an additive object.

- [ ] **Step 8: Add the protocol-v1 detail command**

Change `list_workpieces` dispatch to `catalog.list_workpiece_summaries()`. Validate a non-empty string for `get_workpiece_details`, return `workpiece=catalog.get_workpiece_details(workpiece_id)`, and let existing KeyError/WorkpieceNotFoundError mapping produce `WORKPIECE_NOT_FOUND`. Protocol version remains 1 and id/name meanings do not change.

- [ ] **Step 9: Run GREEN**

Run the Step 5 command. Expected: all focused tests pass, exact id/name-only assertions against catalog/service are changed to subset assertions, while `WorkpieceLibrary.list_workpieces()` keeps its existing minimal contract.

- [ ] **Step 10: Commit**

    git add src/workpiece_library.py src/workpiece_catalog.py src/orientation_tcp_service.py tests/test_workpiece_library.py tests/test_workpiece_catalog.py tests/test_orientation_tcp_service.py
    git commit -m "feat: expose workpiece library summaries"

---

### Task 6: Expose Atomic Evolution Progress Phases

**Files:**
- Modify: src/template_evolution.py
- Modify: src/workpiece_catalog.py
- Modify: tests/test_template_evolution.py
- Modify: tests/test_workpiece_catalog.py

**Interfaces:**
- Consumes: `WorkpieceCatalog.append_templates(workpiece_id, front_images, back_images, *, operation_id=None, progress_callback=None, source="manual_append")` and the existing staging/atomic commit boundary.
- Produces: unchanged task `state` values (`queued`, `building`, `needs_review`, `completed`, `failed`, `cancelled`) plus additive `phase` values (`queued`, `validating`, `copying`, `features`, `committing`, `active`), `completed`, `total`, `progress`, and `recovery_detail`.

- [ ] **Step 1: Add blocking/failing classifier fixtures**

In `tests/test_template_evolution.py`, import `threading`, change `setup_catalog(tmp_path, front_count=1, classifier=None)` to use the supplied classifier or the existing FakeClassifier, and add:

    class BlockingProgressClassifier(FakeClassifier):
        def __init__(self):
            super().__init__()
            self.block_appends = False
            self.feature_started = threading.Event()
            self.release_feature = threading.Event()

        def build_template_cache(self, front, back, progress_callback=None):
            if self.block_appends:
                if progress_callback is not None:
                    progress_callback("front", 0, len(front))
                self.feature_started.set()
                assert self.release_feature.wait(2.0)
            return super().build_template_cache(front, back, progress_callback)

    class FailingProgressClassifier(FakeClassifier):
        def __init__(self):
            super().__init__()
            self.fail_appends = False

        def build_template_cache(self, front, back, progress_callback=None):
            if self.fail_appends:
                if progress_callback is not None:
                    progress_callback("front", 0, len(front))
                raise RuntimeError("synthetic feature failure")
            return super().build_template_cache(front, back, progress_callback)

- [ ] **Step 2: Write the failing intermediate-phase and atomicity tests**

For the blocking test, create the catalog before setting `block_appends=True`, submit one valid confirmed image, and run `evolution.run_next` in a `threading.Thread`. Wait for `feature_started`, then call the existing `get_job(job_id)` and assert `state == "building"`, `phase == "features"`, and the active snapshot revision is still the original revision. Release the event, join the thread, assert it ended, then assert `state == "completed"`, `phase == "active"`, and revision increased exactly once.

For the failure test, create the catalog before setting `fail_appends=True`, run synchronously, then assert `state == "failed"`, `phase == "features"`, a readable error, and the old active revision/cache remain unchanged.

- [ ] **Step 3: Write restart and action-phase tests**

Persist a job as `state="building", phase="features"`, construct a new TemplateEvolution on the same storage, and assert the same job ID returns as `state="queued", phase="queued"` with non-empty `recovery_detail`. For `retry`, `cancel`, and `resolve-review`, use the existing `action(job_id, action)` API and assert the returned `job_id` is unchanged; retry and resolve-review set phase queued, while cancel keeps its last phase and sets state cancelled.

- [ ] **Step 4: Run RED**

    E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_template_evolution.py tests\test_workpiece_catalog.py -q -p no:cacheprovider --basetemp="$env:TEMP\pytest-ui-evolution-red"

Expected: phase fields and progress persistence are absent.

- [ ] **Step 5: Persist progress under the existing job lock**

New jobs start with `phase="queued"`, `completed=0`, `total=len(items)`, `progress=0`. In `run_next`, set validating before geometry validation. Pass a callback into `catalog.append_templates`; for existing copying/features events, update phase/completed/total and compute integer progress as zero when total is zero, otherwise `min(99, completed * 100 // total)`. Persist under `_condition` only if the same job is still building. Pass `source="confirmed_inspection"` so Task 5 inventory identifies confirmed images.

- [ ] **Step 6: Add the committing event without moving the atomic boundary**

In `WorkpieceCatalog.append_templates`, emit `{"phase": "committing", "completed": total, "total": total}` immediately before `commit_prepared_append`, where total is the staged front+back template count. The callback never calls `_activate`; activation remains exclusively in `commit_prepared_append`. After successful return, TemplateEvolution sets `state="completed", phase="active", progress=100`; on exception it sets `state="failed"` and retains the last non-terminal phase and old snapshot.

- [ ] **Step 7: Preserve recovery and job actions**

When loading an interrupted building job, set state/phase queued and a Chinese-readable `recovery_detail` explaining that the previous cache build was interrupted and will retry. Retry and resolve-review clear error/recovery detail and set queued; cancel does not invent successful progress. All transitions retain the original job ID and continue to use the existing persisted document lock.

- [ ] **Step 8: Run GREEN**

Run the Step 4 command. Expected: all phase, restart, action and existing atomicity tests pass.

- [ ] **Step 9: Commit**

    git add src/template_evolution.py src/workpiece_catalog.py tests/test_template_evolution.py tests/test_workpiece_catalog.py
    git commit -m "feat: report template evolution phases"

---

### Task 7: Build and Integrate WorkpieceLibraryPage

**Files:**
- Create: qt_app/workpiecelibrarypage.h
- Create: qt_app/workpiecelibrarypage.cpp
- Create: qt_app/workpiecelibrarypage.ui
- Create: qt_app/tests/test_workpiecelibrarypage.cpp
- Create: qt_app/tests/test_workpiecelibrarypage.pro
- Modify: qt_app/mainwindow.h
- Modify: qt_app/mainwindow.cpp
- Modify: qt_app/mainwindow.ui
- Modify: qt_app/workpiece_orientation.pro
- Modify: qt_app/tests/test_mainwindow.cpp
- Modify: qt_app/tests/test_mainwindow.pro

**Interfaces:**
- Consumes: Task 5 summary/detail JSON, Task 6 evolution job JSON, `BackendUiState`, and Task 3's `commandRequested` routing convention.
- Produces: WorkpieceLibraryPage, explicit browse-vs-detection activation, complete registration/delete command intents, overwrite retry, page-local draft state and cross-page task updates.

- [ ] **Step 1: Create valid Qt image helpers and failing page tests**

In `test_workpiecelibrarypage.cpp`, use `QTemporaryDir` and a local `writeImages(directory, prefix, count, markerBase)` helper that writes distinct 32×32 PNG files with `QImage`. Create a second helper that copies one file to a new path to test same decoded content across directions. Add these exact tests:

    void unequalCountsAndWarningsDoNotBlockRegistration();
    void everySelectedImageIsEmittedInOriginalOrder();
    void emptyUnreadableAndDuplicateTemplatesBlockRegistration();
    void moreThanThirtyTemplatesOnlyWarns();
    void searchFiltersLibraryWithoutChangingDetectionTarget();
    void exactNameConfirmationIsRequiredForRecycle();
    void detailsRenderEveryTemplateWithoutTruncation();
    void evolutionRowsUseJobIdAndPreserveFailures();
    void backendUnavailableKeepsTheEditingDraft();
    void overwriteConfirmationResendsAllPathsWithReplaceTrue();

For the central emission test, spy on the real public signal, invoke the real private slot, and inspect JSON:

    QSignalSpy spy(&page, &WorkpieceLibraryPage::commandRequested);
    page.setTemplatePaths(frontPaths, backPaths);
    page.setWorkpieceName(QStringLiteral("M1"));
    QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                      Qt::DirectConnection));
    QCOMPARE(spy.count(), 1);
    QCOMPARE(spy.first().at(0).toString(), QStringLiteral("register"));
    const QJsonObject fields = spy.first().at(1).toJsonObject();
    QCOMPARE(fields.value("replace").toBool(), false);
    QCOMPARE(fields.value("progress_events").toBool(), true);
    QCOMPARE(fields.value("front_images").toArray(), QJsonArray::fromStringList(frontPaths));
    QCOMPARE(fields.value("back_images").toArray(), QJsonArray::fromStringList(backPaths));

- [ ] **Step 2: Build and verify RED**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_workpiecelibrarypage

Expected: fail because WorkpieceLibraryPage is missing.

- [ ] **Step 3: Implement the page contract**

Expose:

    void setWorkpieces(const QJsonArray &workpieces,
                       const QString &currentDetectionWorkpieceId);
    void setWorkpieceDetails(const QJsonObject &details);
    void setEvolutionJobs(const QJsonArray &jobs);
    void setBackendState(BackendUiState state, const QString &detail);
    void setRegistrationProgress(const QJsonObject &progress, qint64 elapsedMs);
    void setRegistrationResult(const QJsonObject &response);
    void handleBackendResponse(const QString &command,
                               const QJsonObject &response);
    void handleBackendFailure(const QString &command, const QString &code,
                              const QString &message);
    void setOperationError(const QString &code, const QString &message);
    void setTemplatePaths(const QStringList &front, const QStringList &back);
    void setWorkpieceName(const QString &name);
    void setReplaceConfirmationHandler(std::function<bool(const QString &)> handler);
    bool hasUnsavedChanges() const;
    void discardEditingDraft();
    QString browsedWorkpieceId() const;

    signals:
        void commandRequested(const QString &command, const QJsonObject &fields);
        void setCurrentWorkpieceRequested(const QString &workpieceId);
        void dirtyChanged(bool dirty);
        void taskStatusChanged(const QString &title, const QString &phase,
                               int completed, int total, qint64 elapsedMs);

Preserve the existing registration control objectNames. Use `librarySearchEdit` and `libraryWorkpieceList` for browsing; never reuse AppHeader's `workpieceComboBox`. The left list shows name, real front/back counts, rule count, detectable status and updated time. The right side shows every template thumbnail with direction/source/added time/readable state, the latest registration result, and evolution records keyed by job ID.

- [ ] **Step 4: Port validation and overwrite retry without changing semantics**

Move normalized paths, image readability/content duplicate validation, low-count warning, >30 warning, progress timer and actual completion counts from MainWindow. Empty direction, unreadable file, duplicate path and duplicate decoded content block submit; low count and >30 only warn. Keep every valid selected path in order in the JSON arrays.

On `WORKPIECE_EXISTS`, call the injected confirmation handler (or the production QMessageBox when no handler is installed); if accepted, re-emit the saved, unchanged registration fields with only `replace=true`. MainWindow's existing `setTemplatePaths`, `setWorkpieceName` and `setReplaceConfirmationHandler` temporarily forward to WorkpieceLibraryPage so old integration tests remain valid.

Use an inline `recycleNameConfirmationEdit` in the danger section. Enable recycle only when its text exactly equals the browsed display name. Then emit:

    emit commandRequested(QStringLiteral("recycle_workpiece"), {
        {QStringLiteral("workpiece_id"), browsedWorkpieceId_},
        {QStringLiteral("operation_id"),
         QUuid::createUuid().toString(QUuid::WithoutBraces)}
    });

- [ ] **Step 5: Route summaries, details and long-running tasks**

MainWindow converts each list summary to an id/name pair for `AppHeader::setWorkpieces` and passes the full QJsonArray only to WorkpieceLibraryPage. Route get_workpiece_details only to the library page; route register progress/result/failure to the page and TaskStatusWidget; route list_evolution_jobs to the page and global task status; refresh list/details after register or recycle success. A details failure updates only the library page and never replaces an inspection result.

Evolution polling stays in MainWindow, runs only when BackendClient is Ready and no request is pending, and continues while the page is hidden. A registered/evolution task shows queued, validating, copying, features, committing, active or failed with exact completed/total and elapsed values.

- [ ] **Step 6: Add browse/target and dirty-navigation integration tests**

Add `browsingLibraryDoesNotRetargetInspection`, `explicitSetCurrentUpdatesHeaderAndInspection`, `registrationContinuesWhileInspectionPageIsVisible`, `detailsFailureDoesNotOverwriteInspectionResult`, `reconnectPreservesLibraryDraftAndEvolutionRows`, and `dirtyLibraryDraftCanCancelNavigation`. Selecting a list row only requests details. Only the explicit action emits `setCurrentWorkpieceRequested`.

When leaving a dirty library page, show three choices: `保留并离开` keeps the draft and navigates, `放弃修改` calls `discardEditingDraft` and navigates, and `取消` leaves page and target unchanged. Tests operate the QMessageBox with a zero-delay QTimer and assert all three branches. This prompt does not submit registration silently.

Keep passing `acceptsUnequalTemplateCountsAndShowsLowCountWarning`, `allowsMoreThanThirtyTemplatesWithPerformanceWarning`, `registrationCompletionShowsActualTemplateCounts`, `confirmsBeforeSendingReplaceTrue`, `busyStateDisablesRegisterAndPredict`, and `loadingStateShowsStatusAndKeepsActionsDisabled` after moving their registration assertions to WorkpieceLibraryPage.

- [ ] **Step 7: Run GREEN**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_workpiecelibrarypage,test_mainwindow,test_backendclient

Expected: all three targets pass offscreen.

- [ ] **Step 8: Commit**

    git add qt_app/workpiecelibrarypage.h qt_app/workpiecelibrarypage.cpp qt_app/workpiecelibrarypage.ui qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/workpiece_orientation.pro qt_app/tests/test_workpiecelibrarypage.cpp qt_app/tests/test_workpiecelibrarypage.pro qt_app/tests/test_mainwindow.cpp qt_app/tests/test_mainwindow.pro
    git commit -m "refactor: add Qt workpiece library page"

---

### Task 8: Separate Geometry Canvas Overlay Layers

**Files:**
- Modify: qt_app/geometryrulecanvas.h
- Modify: qt_app/geometryrulecanvas.cpp
- Modify: qt_app/tests/test_geometryrulecanvas.cpp

**Interfaces:**
- Produces: independent guide/fitted/effective/mask visibility and exact fitted/effective getters.
- Consumes: existing setCoarseShape(), setFitOverlay(), shapeChanged(), shapeCommitted() and native-image coordinate conversion.

- [ ] **Step 1: Write failing layer-state tests**

Add:

    void hidingGuideDoesNotDeleteItsGeometry() {
        GeometryRuleCanvas canvas;
        const QJsonObject guide{{"shape", "circle"}, {"cx", 100.0},
                                {"cy", 100.0}, {"r", 60.0}};
        canvas.setCoarseShape(guide);
        canvas.setGuideVisible(false);
        QCOMPARE(canvas.coarseShape(), guide);
        QVERIFY(!canvas.guideVisible());
    }

    void fittedAndEffectiveBoundariesRemainDistinct() {
        GeometryRuleCanvas canvas;
        canvas.setFitOverlay({
            {"fitted_shape", QJsonObject{{"shape", "circle"}, {"r", 50.0}}},
            {"effective_shape", QJsonObject{{"shape", "circle"}, {"r", 48.0}}}
        });
        QCOMPARE(canvas.fittedShape().value("r").toDouble(), 50.0);
        QCOMPARE(canvas.effectiveShape().value("r").toDouble(), 48.0);
        canvas.setFittedBoundaryVisible(false);
        QVERIFY(!canvas.fittedBoundaryVisible());
        QCOMPARE(canvas.effectiveShape().value("r").toDouble(), 48.0);
    }

- [ ] **Step 2: Run RED**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_geometryrulecanvas

Expected: visibility APIs and separate shapes are missing.

- [ ] **Step 3: Add the minimal canvas API**

Add:

    void setGuideVisible(bool visible);
    void setFittedBoundaryVisible(bool visible);
    void setEffectiveBoundaryVisible(bool visible);
    void setMaskOverlayVisible(bool visible);
    bool guideVisible() const;
    bool fittedBoundaryVisible() const;
    bool effectiveBoundaryVisible() const;
    bool maskOverlayVisible() const;
    QJsonObject fittedShape() const;
    QJsonObject effectiveShape() const;

Keep fitShape() as the compatibility getter: effective shape first, fitted shape second. Visibility changes call update() only and never mutate shapes or coordinates.

- [ ] **Step 4: Paint fixed layer semantics**

- guide: cyan dashed;
- fitted boundary: green solid;
- effective boundary: amber dashed/solid contrast;
- ignored mask: translucent amber.

The paint order is image → mask → guide → fitted → effective → edit handles. No paint branch may call imagePoint() or modify zoom/pan.

- [ ] **Step 5: Run GREEN and all existing coordinate tests**

Run the Step 2 command. Expected: all old and new tests pass, including drag, resize, zoom, pan and effective-shape preference.

- [ ] **Step 6: Commit**

    git add qt_app/geometryrulecanvas.h qt_app/geometryrulecanvas.cpp qt_app/tests/test_geometryrulecanvas.cpp
    git commit -m "feat: separate geometry canvas overlay layers"

---

### Task 9: Migrate Geometry Management from Dialog to Embedded Page

**Files:**
- Rename: qt_app/geometrymaskmanager.h → qt_app/geometryrulespage.h
- Rename: qt_app/geometrymaskmanager.cpp → qt_app/geometryrulespage.cpp
- Rename: qt_app/tests/test_geometrymaskmanager.cpp → qt_app/tests/test_geometryrulespage.cpp
- Rename: qt_app/tests/test_geometrymaskmanager.pro → qt_app/tests/test_geometryrulespage.pro
- Modify: qt_app/mainwindow.h
- Modify: qt_app/mainwindow.cpp
- Modify: qt_app/mainwindow.ui
- Modify: qt_app/workpiece_orientation.pro
- Modify: qt_app/tests/test_mainwindow.cpp
- Modify: qt_app/tests/test_mainwindow.pro

**Interfaces:**
- Consumes: every current GeometryMaskManagerDialog setter/signal and Task 8 canvas.
- Produces: `GeometryRulesPage : QWidget`, `hasUnsavedChanges()`, `discardUnsavedChanges()`, `requestSaveDraft()`, `unsavedChangesChanged(bool)`, and one persistent embedded instance.

- [ ] **Step 1: Rename the test target and write valid embedding checks**

Use `git mv` for the four files, update only test includes/class names, then add:

    void pageIsAChildWidgetAndHasNoDialogCloseAction() {
        QWidget host;
        GeometryRulesPage page(&host);
        QVERIFY(qobject_cast<QDialog *>(&page) == nullptr);
        QVERIFY(!page.isWindow());
        const auto buttons = page.findChildren<QPushButton *>();
        for (QPushButton *button : buttons)
            QVERIFY2(button->text() != QStringLiteral("关闭"),
                     "embedded page must not retain the dialog close button");
    }

Add `geometryPageIsEmbeddedAndPersistent` to MainWindow tests: navigate to geometry, find objectName `geometryRulesPage`, assert its parent chain reaches `geometryPageHost` and it is not a window; navigate away/back and assert pointer identity is unchanged.

- [ ] **Step 2: Run RED**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_geometryrulespage,test_mainwindow

Expected: renamed page/test targets do not build until the class and qmake files migrate.

- [ ] **Step 3: Perform the mechanical class migration**

Rename GeometryMaskManagerDialog to GeometryRulesPage and change QDialog to QWidget. Remove `<QDialog>`, `QDialog::reject`, `cancelButton_`, the “关闭” button construction/connection and dialog-only size/window flags. Preserve these setters/getters exactly:

    void setSnapshot(const QJsonObject &snapshot);
    void setValidationJob(const QJsonObject &job);
    void setBusy(bool busy);
    void setPreviewBusy(bool busy);
    void setRulePreview(const QJsonObject &preview);
    void setOperationError(const QString &message);
    QJsonObject draft() const;
    QJsonObject snapshot() const;

Preserve every command signal exactly:

    void snapshotRequested(const QString &workpieceId);
    void saveDraftRequested(const QJsonObject &draft, int libraryRevision,
                            int draftRevision);
    void publishWorkflowRequested(const QJsonObject &draft, int libraryRevision,
                                  int draftRevision,
                                  const QString &overrideReason);
    void validateRequested(int libraryRevision, int draftRevision);
    void validationJobRequested(const QString &jobId);
    void validationJobActionRequested(const QString &jobId,
                                      const QString &action);
    void publishRequested(const QString &jobId, int libraryRevision,
                          int draftRevision, const QString &overrideReason);
    void rollbackRequested(int libraryRevision);
    void previewRequested(const QJsonObject &request);
    void migrationResolutionRequested(const QString &conflictId,
                                      const QJsonObject &resolution,
                                      int libraryRevision, int draftRevision);

- [ ] **Step 4: Update every qmake reference**

Set the renamed test project to:

    TARGET = test_geometryrulespage
    SOURCES += test_geometryrulespage.cpp ../geometryrulespage.cpp ../geometryrulecanvas.cpp
    HEADERS += ../geometryrulespage.h ../geometryrulecanvas.h

Replace geometrymaskmanager paths with geometryrulespage paths in `workpiece_orientation.pro` and `test_mainwindow.pro`, and update C++ includes in MainWindow/tests. Do not leave both old and new files in a target.

- [ ] **Step 5: Add one dirty-state path**

Expose:

    bool hasUnsavedChanges() const;
    void discardUnsavedChanges();
    void requestSaveDraft();

    signals:
        void unsavedChangesChanged(bool dirty);

Implement one private `setDirty(bool)` and route `setSnapshot`, every mutation, successful save refresh and `discardUnsavedChanges` through it. Emit only on an actual state change. Discard restores the last snapshot draft without a backend call; requestSaveDraft invokes the same save path as the page button.

- [ ] **Step 6: Embed the existing page without resetting work**

Create GeometryRulesPage once, set objectName `geometryRulesPage`, and add it to the existing `geometryPageHostLayout` at stack index 2; do not call `QStackedWidget::addWidget` and accidentally create index 3. Remove WA_DeleteOnClose, show, raise, activateWindow and dialog-pointer clearing.

Delete the old `openGeometryMaskManager` actions that clear `geometryValidationJobId_`, clear publish-workflow state, or stop validation polling. First entry for a workpiece requests a snapshot; ordinary round-trips for the same workpiece do not reload. A real detection-workpiece change requests the new snapshot after dirty-state handling. Page hiding does not stop `geometryPollTimer_`; disconnect retains the job ID, and reconnect resumes polling. Send validation progress to TaskStatusWidget so it remains visible outside the page.

Replace every “manager pointer exists” busy test with actual client busy, pending command, dirty or validation-job conditions; a permanent page pointer never blocks unrelated actions.

- [ ] **Step 7: Run GREEN with the full pre-layout geometry regression**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_geometryrulespage,test_geometryrulecanvas,test_mainwindow

Expected: all former manager behavior tests plus embedding/persistence tests pass before visual simplification.

The carried-forward page tests must still prove that circle, ellipse and rotated rectangle can all be saved, no shape is silently made the default main tool, signed margin semantics remain negative=inward and positive=outward for both inside/outside modes, and one logical rule exposes separate front/back calibration state.

- [ ] **Step 8: Commit**

    git add qt_app/geometrymaskmanager.h qt_app/geometrymaskmanager.cpp qt_app/geometryrulespage.h qt_app/geometryrulespage.cpp qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/workpiece_orientation.pro qt_app/tests/test_geometrymaskmanager.cpp qt_app/tests/test_geometrymaskmanager.pro qt_app/tests/test_geometryrulespage.cpp qt_app/tests/test_geometryrulespage.pro qt_app/tests/test_mainwindow.cpp qt_app/tests/test_mainwindow.pro
    git commit -m "refactor: embed geometry rules as an application page"

---

### Task 10: Simplify Geometry Validation and Publishing UX

**Files:**
- Modify: qt_app/geometryrulespage.h
- Modify: qt_app/geometryrulespage.cpp
- Modify: qt_app/tests/test_geometryrulespage.cpp
- Modify: qt_app/mainwindow.h
- Modify: qt_app/mainwindow.cpp
- Modify: qt_app/tests/test_mainwindow.cpp

**Interfaces:**
- Consumes: the existing `publishWorkflowRequested` serial chain, validation job JSON and Task 8 layer APIs.
- Produces: one primary save/validate/publish action, actionable issue rows, `setBackendAvailable(bool, const QString &)`, collapsed advanced controls, and asynchronous page/workpiece/window navigation protection.

- [ ] **Step 1: Write failing page workflow tests**

Add exact page tests:

    void normalStateExposesOnePrimaryWorkflowAction();
    void disabledWorkflowShowsTheExactReasonNearby();
    void backendUnavailableDisablesWorkflowWithoutClearingState();
    void advancedDiagnosticsAreCollapsedByDefault();
    void migrationControlsAppearOnlyWhenConflictExists();
    void savedDraftHidesGuideAndShowsFittedBoundary();
    void layerVisibilityChangesDoNotDirtyDraft();
    void issueRowStoresAndSelectsExactContext();
    void templateLevelIssueDoesNotInventRuleSelection();

The primary test finds `publishWorkflowButton`, asserts role primary, and counts visible buttons with role primary as exactly one. It also finds a non-empty `publishDisabledReasonLabel` whenever the action is disabled. The backend-unavailable test snapshots draft/canvas/table values, calls `setBackendAvailable(false, "后端断开")`, then verifies those values remain while the action disables and displays that reason.

- [ ] **Step 2: Write failing navigation and hidden-task tests**

Add MainWindow tests `dirtyGeometryPageCanCancelPageNavigation`, `dirtyGeometryPageCanCancelWorkpieceChange`, `dirtyGeometryPageCanCancelWindowClose`, `geometrySaveFailureCancelsPendingNavigation`, `geometrySaveSuccessCompletesPendingNavigation`, `validationContinuesWhileGeometryPageIsHidden`, and `reconnectResumesGeometryValidationPolling`. Use a zero-delay QTimer to choose each QMessageBox action and the existing fake backend to return save success/failure.

- [ ] **Step 3: Run RED**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_geometryrulespage,test_mainwindow

Expected: simplified workflow, context roles and asynchronous guards are absent.

- [ ] **Step 4: Reorganize the three columns without changing draft JSON**

Left: logical rule list and shared fields. Center: direction/template selector, canvas and permanent guide/fitted/effective/mask legend toggles. Right: validation summary, issue table, disabled-reason text and publishWorkflowButton.

Move candidates, raw diagnostics, manual anchor fallback, migration resolution, version copy and rollback into a checkable `advancedGeometryGroup` that starts collapsed. Keep migration controls hidden unless `migration.conflicts` is non-empty. Preserve existing objectNames for rule list, direction, templates, candidates, review fields, validation table and workflow button.

- [ ] **Step 5: Implement explicit validation summary and issue identity**

Map template states as follows: active → 通过; low_confidence → 低置信度; not_configured → 未配置; needs_reseed or failed → 拟合失败; excluded → 已排除 with its reason. Expand issue rows per rule problem where available. Store:

    enum ValidationIssueRole {
        DirectionRole = Qt::UserRole + 1,
        TemplateIdRole,
        RuleIdRole,
        ReasonCodeRole
    };

Clicking a row selects its stored direction, template and non-empty rule ID. A template-level issue with no rule ID selects only direction/template and never falls back to the first rule. `publishDisabledReasonLabel` shows one actionable sentence and the issue table shows human-readable reasons, not only raw JSON.

- [ ] **Step 6: Apply fixed overlay display semantics**

After a successful preview or saved snapshot, hide guide by default and show fitted boundary; show effective boundary/mask according to persistent display toggles. Retain guide geometry for an explicit “手绘引导” toggle. Display toggles call only Task 8 visibility APIs and never set draft dirty.

- [ ] **Step 7: Implement one backend-availability gate**

Add `setBackendAvailable(bool available, const QString &reason)` and `backendAvailable_`. Every workflow-button condition includes backendAvailable_. Disconnect calls it with false instead of only `setBusy(false)`; reconnect sets true. The page keeps draft, canvas, validation rows and error detail. TaskStatusWidget/AppHeader retain the restart action.

- [ ] **Step 8: Implement asynchronous geometry navigation protection**

Add to MainWindow:

    enum class PendingNavigationKind { None, Page, Workpiece, CloseWindow };
    PendingNavigationKind pendingNavigationKind_ = PendingNavigationKind::None;
    AppPage pendingNavigationPage_ = AppPage::Inspection;
    QString pendingNavigationWorkpieceId_;
    bool bypassCloseGuard_ = false;

For a dirty geometry page, page change, detection-workpiece change and closeEvent offer `保存并继续`, `放弃修改`, `取消`. Cancel changes nothing. Discard calls `discardUnsavedChanges` then continues. Save records exactly one pending target, calls `requestSaveDraft`, keeps the current page/workpiece/window open and disables geometry editing until the reply. Save success updates the snapshot/dirty state before completing the pending action; save failure clears the pending target, re-enables editing and stays in place with the error. While save is pending, ignore a second navigation request. For CloseWindow, ignore the original QCloseEvent and call close again with `bypassCloseGuard_` only after successful save.

- [ ] **Step 9: Preserve the serial publish chain and warning override**

Keep save_geometry_mask_draft → validate_geometry_mask_draft → poll get_geometry_mask_validation_job → publish_geometry_mask_profile. Blocking issues stop and populate the issue list. With no blocker and no regression/warning requiring approval, publish automatically. A warning/regression shows the existing inline override-reason field; the same primary workflow button continues only after a non-empty reason, preserving the current backend contract and existing warning tests. Do not restore separate save/validate/publish primary buttons. Hidden page state continues receiving polling updates.

- [ ] **Step 10: Run GREEN and protocol regression**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_geometryrulespage,test_geometryrulecanvas,test_mainwindow
    E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_geometry_calibration.py tests\test_geometry_mask_profiles.py tests\test_orientation_tcp_service.py -k geometry -q -p no:cacheprovider --basetemp="$env:TEMP\pytest-ui-geometry"

Expected: all pass; no Python geometry production file changes.

- [ ] **Step 11: Commit**

    git add qt_app/geometryrulespage.h qt_app/geometryrulespage.cpp qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/tests/test_geometryrulespage.cpp qt_app/tests/test_mainwindow.cpp
    git commit -m "feat: simplify geometry rule publishing workflow"

---

### Task 11: Finish Visual Polish, Remove Obsolete UI Paths, and Verify

**Files:**
- Modify: qt_app/main.cpp
- Modify: qt_app/apptheme.cpp
- Modify: qt_app/appheader.cpp
- Modify: qt_app/taskstatuswidget.cpp
- Modify: qt_app/resources/theme.qss
- Modify: qt_app/resources.qrc
- Create: qt_app/resources/icons/nav-inspection.svg
- Create: qt_app/resources/icons/nav-library.svg
- Create: qt_app/resources/icons/nav-geometry.svg
- Create: qt_app/resources/icons/status-success.svg
- Create: qt_app/resources/icons/status-warning.svg
- Create: qt_app/resources/icons/status-error.svg
- Modify: qt_app/mainwindow.h
- Modify: qt_app/mainwindow.cpp
- Modify: qt_app/mainwindow.ui
- Modify: qt_app/inspectionpage.ui
- Modify: qt_app/workpiecelibrarypage.ui
- Modify: qt_app/geometryrulespage.cpp
- Modify: qt_app/tests/test_appfoundation.cpp
- Modify: qt_app/tests/test_inspectionpage.cpp
- Modify: qt_app/tests/test_workpiecelibrarypage.cpp
- Modify: qt_app/tests/test_geometryrulespage.cpp
- Modify: qt_app/tests/test_mainwindow.cpp
- Create: docs/verification/qt-ui-information-architecture-redesign-results.md
- Create: docs/verification/qt-ui-redesign-benchmark.md
- Create when benchmark runs: docs/verification/qt-ui-redesign-benchmark.json
- Create: docs/verification/screenshots/qt-ui-inspection.png
- Create: docs/verification/screenshots/qt-ui-library.png
- Create: docs/verification/screenshots/qt-ui-geometry.png

**Interfaces:**
- Consumes: every page/shared component from Tasks 1–10 and `scripts/run_qt5_tests.ps1`.
- Produces: final 1920×1080 UI, Qt 5 high-DPI behavior, protected application close, clean shell and auditable verification evidence.

- [ ] **Step 1: Add failing final layout, semantics and close tests**

Add exact Qt tests for: minimumSize not exceeding 1280×720; primary result actions visible at 1280×720 and 1920×1080; elided long paths retaining full tooltip; every warning/error having non-empty text; exactly one visible primary-role button in each normal page state; tab order reaching nav/input/result/confirmation; library/geometry pages having MainWindow ancestry; `closeWithActiveTaskRequiresConfirmation`; and `cancelClosePreservesWindowAndTaskState`.

For active tasks, MainWindow's closeEvent first runs the dirty-geometry save guard from Task 10, then—if any register, batch, evolution build, validation or backend request is active—offers `退出应用` and `继续运行`. Cancel ignores the close event without mutating task state. Accepting exits without inventing a backend cancel protocol.

- [ ] **Step 2: Run RED**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1 -Targets test_appfoundation,test_inspectionpage,test_workpiecelibrarypage,test_geometryrulespage,test_mainwindow

Expected: final semantic/layout/close assertions fail before polish.

- [ ] **Step 3: Enable Qt 5 high-DPI behavior before QApplication**

Use this order in `main.cpp`:

    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    QApplication application(argc, argv);
    AppTheme::apply(&application);

- [ ] **Step 4: Add the exact icon assets and semantic roles**

Each icon is 24×24, fill none, stroke `#5F6B76`, stroke-width 1.8, linecap/linejoin round. Use this wrapper with the listed path data:

    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
         viewBox="0 0 24 24" fill="none" stroke="#5F6B76"
         stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="PATH_DATA"/>
    </svg>

Path data:

- nav-inspection: `M4 5h16v14H4z M8 9h8 M8 13h5`
- nav-library: `M4 6h16v12H4z M4 10h16 M8 6v12`
- nav-geometry: `M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18 M12 8v8 M8 12h8`
- status-success: `M5 12l4 4L19 6`
- status-warning: `M12 4l9 16H3z M12 9v5 M12 17h.01`
- status-error: `M6 6l12 12 M18 6L6 18`

Register all six in resources.qrc. Apply role=primary only to the current page's single main action; role=danger only to recycle/purge/destructive overwrite actions requiring confirmation. “不入库” remains secondary. Use panel=true for white surfaces. No gradient, emoji, character icon, shadow-heavy card or page-local color stylesheet.

- [ ] **Step 5: Apply the approved 8-pixel layout system**

Set root page property `pageRoot=true`, 24 px page margins, 16–24 px panel padding, 8/16/24 spacing increments, 4–6 px radii, result label 36–48 px and section heading 16–18 px. Use stretch factors and splitters rather than new fixed image sizes. At 1280×720, primary actions remain reachable without horizontal scrolling; at 1920×1080 the approved 65/35 detection split is the default.

- [ ] **Step 6: Remove only obsolete UI migration paths**

Delete dynamic creation of old library/confirmation/geometry forwarding buttons, openAnnotationEditor/openAnnotationManager forwarding, openGeometryMaskManager top-level behavior, obsolete dialog pointer/close state and inline red/amber styles replaced by semantic properties. Keep legacy annotation sources, protocols, model behavior and unrelated backend logic.

- [ ] **Step 7: Run GREEN across every Qt test with isolated builds**

    powershell -ExecutionPolicy Bypass -File .\scripts\run_qt5_tests.ps1

The runner must build and execute: test_appconfig, test_backendclient, test_backendprocessmanager, test_annotationmanager, test_appfoundation, test_inspectionpage, test_workpiecelibrarypage, test_geometryrulecanvas, test_geometryrulespage and test_mainwindow. Expected: zero failures/crashes.

- [ ] **Step 8: Run Python regression in the system temporary directory**

    E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -m "not integration" -q -p no:cacheprovider --basetemp="$env:TEMP\pytest-ui-redesign"

Expected: all non-integration tests pass and no pytest directory appears in the repository.

- [ ] **Step 9: Build the Qt 5.14.2 Release application**

    Set-Location -LiteralPath 'E:\Project\wang\pp_813'
    powershell -ExecutionPolicy Bypass -File .\scripts\build_qt5.ps1

Expected: `E:\Project\wang\pp_813\qt_app\build-release\release\workpiece_orientation.exe`.

- [ ] **Step 10: Perform visual/workflow acceptance and capture fixed evidence**

At 1920×1080 with Windows scaling 100%, 125% and 150%, verify startup single mode; navigation/state retention; single/batch review; 1+12 and >30 build/display; evolution phases; geometry front/back calibration/fitted boundary/blocker navigation/publish/rollback; disconnect/reconnect/loading/timeout/restart; long Chinese paths/unreadable images; keyboard-only main flow; and no clipped/overlapping primary controls.

Capture the three active pages at 1920×1080 to the three fixed PNG paths listed in Files. Do not put manual screenshots elsewhere in the repository.

- [ ] **Step 11: Verify inference behavior is unchanged**

Run:

    E:\python\anaconda3\envs\shitu\python.exe scripts\benchmark_adaptive_local_search.py --project-root E:\Project\wang\pp_813 --model-dir E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer --library-dir E:\Project\wang\pp_813\runtime_library --warmup 1 --output-json docs\verification\qt-ui-redesign-benchmark.json --output-markdown docs\verification\qt-ui-redesign-benchmark.md

Compare labels and P50/P95 against `docs/verification/adaptive-lightglue-candidate-search-results.json`. Expected: labels identical and no systematic backend timing increase. If model/library/dataset prerequisites are absent, still create `qt-ui-redesign-benchmark.md` containing the exact missing path/condition and the words `BLOCKED — not measured`; do not create a synthetic JSON and do not claim the gate passed.

- [ ] **Step 12: Write the verification report**

Record commit range, exact changed files, Qt result counts per executable, Python count, Release path, three linked screenshots and DPI matrix, manual single/batch/library/geometry results, benchmark comparison or blocked prerequisite, and known limitations. Every pass claim must have an attached command/result or screenshot.

- [ ] **Step 13: Final diff and artifact checks**

    git diff --check
    git status --short
    git diff --stat

Confirm no build tree, runtime library, pytest cache, model, user data or screenshot outside `docs/verification/screenshots` is staged.

- [ ] **Step 14: Commit exact final files**

    git add qt_app/main.cpp qt_app/apptheme.cpp qt_app/appheader.cpp qt_app/taskstatuswidget.cpp qt_app/resources/theme.qss qt_app/resources.qrc qt_app/resources/icons/nav-inspection.svg qt_app/resources/icons/nav-library.svg qt_app/resources/icons/nav-geometry.svg qt_app/resources/icons/status-success.svg qt_app/resources/icons/status-warning.svg qt_app/resources/icons/status-error.svg qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/mainwindow.ui qt_app/inspectionpage.ui qt_app/workpiecelibrarypage.ui qt_app/geometryrulespage.cpp qt_app/tests/test_appfoundation.cpp qt_app/tests/test_inspectionpage.cpp qt_app/tests/test_workpiecelibrarypage.cpp qt_app/tests/test_geometryrulespage.cpp qt_app/tests/test_mainwindow.cpp docs/verification/qt-ui-information-architecture-redesign-results.md docs/verification/qt-ui-redesign-benchmark.md docs/verification/screenshots/qt-ui-inspection.png docs/verification/screenshots/qt-ui-library.png docs/verification/screenshots/qt-ui-geometry.png
    if (Test-Path -LiteralPath 'docs/verification/qt-ui-redesign-benchmark.json') { git add docs/verification/qt-ui-redesign-benchmark.json }
    git commit -m "feat: complete Qt UI information architecture redesign"

---

## Completion Gate

Implementation is complete only when:

1. all three pages live in one MainWindow and start on single inspection;
2. current detection workpiece is explicit and library browsing cannot change it;
3. single/batch images, evidence and confirmations remain correctly paired by stable identity;
4. all selected templates participate and old libraries restore;
5. registration/evolution/validation tasks remain visible across navigation;
6. one logical geometry rule remains visible across front/back calibration;
7. saved geometry displays fitted/effective boundaries, not a stale manual guide;
8. the normal geometry path uses one primary save/validate/publish action;
9. backend failure preserves user-visible state and offers a recovery action;
10. page changes, detection-workpiece changes and window close protect unsaved geometry; active tasks require explicit close confirmation;
11. Qt/Python regressions, Release build and manual visual acceptance are documented;
12. models, thresholds and recognition semantics are unchanged.
