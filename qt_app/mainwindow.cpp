#include "mainwindow.h"

#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonObject>
#include <QMessageBox>
#include <QPushButton>
#include <QAbstractButton>
#include <QTableWidget>
#include <QTextEdit>
#include <QTimer>
#include <QUuid>

#include "backendclient.h"
#include "backendprocessmanager.h"
#include "annotationeditor.h"
#include "annotationmanager.h"
#include "geometrymaskmanager.h"
#include "inspectionpage.h"
#include "inspectiontypes.h"
#include "taskstatuswidget.h"
#include "workpiecelibrarypage.h"

namespace {
const QStringList kImageFilters = {QStringLiteral("PNG/JPEG/BMP (*.png *.jpg *.jpeg *.bmp)")};

QStringList dialogFilters() {
    return kImageFilters;
}

}

MainWindow::MainWindow(QWidget *parent)
    : MainWindow(nullptr, nullptr, parent) {}

MainWindow::MainWindow(BackendClient *client, BackendProcessManager *manager, QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), client_(client), manager_(manager) {
    ui->setupUi(this);
    initializeUi();
    connectBackendSignals();
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::initializeUi() {
    appHeader_ = new AppHeader(ui->centralwidget);
    ui->appHeaderHostLayout->addWidget(appHeader_);
    inspectionPage_ = new InspectionPage(ui->inspectionPageHost);
    ui->inspectionPageHostLayout->addWidget(inspectionPage_);
    workpieceLibraryPage_ = new WorkpieceLibraryPage(ui->libraryPageHost);
    ui->libraryPageHostLayout->addWidget(workpieceLibraryPage_);
    chooseImageButton_ = inspectionPage_->findChild<QPushButton *>(
        QStringLiteral("chooseImageButton"));
    predictButton_ = inspectionPage_->findChild<QPushButton *>(
        QStringLiteral("predictButton"));
    chooseBatchImagesButton_ = inspectionPage_->findChild<QPushButton *>(
        QStringLiteral("chooseBatchImagesButton"));
    batchPredictButton_ = inspectionPage_->findChild<QPushButton *>(
        QStringLiteral("batchPredictButton"));
    confirmFrontButton_ = inspectionPage_->findChild<QPushButton *>(
        QStringLiteral("confirmFrontButton"));
    confirmBackButton_ = inspectionPage_->findChild<QPushButton *>(
        QStringLiteral("confirmBackButton"));
    rejectConfirmationButton_ = inspectionPage_->findChild<QPushButton *>(
        QStringLiteral("rejectConfirmationButton"));
    currentImageLabel_ = inspectionPage_->findChild<QLabel *>(
        QStringLiteral("currentImageLabel"));
    resultLabel_ = inspectionPage_->findChild<QLabel *>(QStringLiteral("resultLabel"));
    reviewLabel_ = inspectionPage_->findChild<QLabel *>(QStringLiteral("reviewLabel"));
    batchSummaryLabel_ = inspectionPage_->findChild<QLabel *>(
        QStringLiteral("batchSummaryLabel"));
    currentResultTargetLabel_ = inspectionPage_->findChild<QLabel *>(
        QStringLiteral("currentResultTargetLabel"));
    evidenceTextEdit_ = inspectionPage_->findChild<QTextEdit *>(
        QStringLiteral("evidenceTextEdit"));
    batchResultsTableWidget_ = inspectionPage_->findChild<QTableWidget *>(
        QStringLiteral("batchResultsTableWidget"));
    globalTaskStatus_ = new TaskStatusWidget(ui->centralwidget);
    ui->taskStatusHostLayout->addWidget(globalTaskStatus_);
    ui->mainPageStack->setCurrentIndex(static_cast<int>(AppPage::Inspection));
    appHeader_->setCurrentPage(AppPage::Inspection);

    BackendStatusDetails initialBackendDetails;
    initialBackendDetails.connectionDetail = QStringLiteral("未连接");
    initialBackendDetails.modelDetail = QStringLiteral("未加载");
    initialBackendDetails.canRestart = manager_ != nullptr;
    appHeader_->setBackendDetails(initialBackendDetails);

    annotationEditorButton_ = workpieceLibraryPage_->findChild<QPushButton *>(
        QStringLiteral("annotationEditorButton"));
    predictButton_->setEnabled(false);
    chooseImageButton_->setEnabled(false);
    chooseBatchImagesButton_->setEnabled(false);
    batchPredictButton_->setEnabled(false);
    resultLabel_->setText(QStringLiteral("尚未检测"));
    reviewLabel_->clear();
    batchSummaryLabel_->clear();
    evidenceTextEdit_->clear();
    evolutionPollTimer_ = new QTimer(this);
    evolutionPollTimer_->setInterval(5000);
    connect(evolutionPollTimer_, &QTimer::timeout, this, &MainWindow::pollEvolutionJobs);
    geometryPollTimer_ = new QTimer(this);
    geometryPollTimer_->setInterval(1000);
    connect(geometryPollTimer_, &QTimer::timeout, this, &MainWindow::pollGeometryValidation);
    clearBatchResults();
    updateButtonStates();
    connect(chooseImageButton_, &QPushButton::clicked,
            this, &MainWindow::chooseInspectionImage);
    connect(chooseBatchImagesButton_, &QPushButton::clicked,
            this, &MainWindow::chooseBatchImages);
    connect(batchPredictButton_, &QPushButton::clicked,
            this, &MainWindow::submitBatchPrediction);
    connect(inspectionPage_, &InspectionPage::commandRequested,
            this, [this](const QString &command, const QJsonObject &fields) {
                sendPageCommand(CommandOwner::Inspection, command, fields);
            });
    connect(inspectionPage_, &InspectionPage::confirmationRequested,
            this, [this](const QString &recordId, const QString &workpieceId,
                         const QString &imagePath, const QString &orientation) {
                pendingConfirmationRecordId_ = recordId;
                pendingConfirmationOrientation_ = orientation;
                submitTemplateConfirmation(workpieceId, imagePath, orientation);
            });
    connect(inspectionPage_, &InspectionPage::batchFinished,
            this, [this](bool) {
                batchInFlight_ = false;
                resultContext_ = ResultContext::Batch;
                updateButtonStates();
            });
    connect(workpieceLibraryPage_, &WorkpieceLibraryPage::commandRequested,
            this, [this](const QString &command, const QJsonObject &fields) {
                if (command == QStringLiteral("list_workpieces")) {
                    refreshWorkpieces();
                    return;
                }
                sendPageCommand(CommandOwner::Library, command, fields);
            });
    connect(workpieceLibraryPage_, &WorkpieceLibraryPage::setCurrentWorkpieceRequested,
            this, [this](const QString &workpieceId) {
                appHeader_->setCurrentWorkpieceId(workpieceId);
                inspectionPage_->setCurrentWorkpiece(appHeader_->currentWorkpieceId(),
                                                     appHeader_->currentWorkpieceName());
                workpieceLibraryPage_->setWorkpieces(
                    workpieceSummaries_, appHeader_->currentWorkpieceId());
                if (resultContext_ != ResultContext::Batch) {
                    lastPredictionWorkpieceId_.clear();
                    lastPredictionImagePath_.clear();
                    lastPredictionResponse_ = QJsonObject();
                }
                updateButtonStates();
            });
    connect(workpieceLibraryPage_, &WorkpieceLibraryPage::taskStatusChanged,
            this, [this](const QString &title, const QString &phase,
                         int completed, int total, qint64 elapsedMs) {
                globalTaskStatus_->setRunning(title, phase, completed, total,
                                              qMax<qint64>(0, elapsedMs));
                if (phase == QStringLiteral("failed")) {
                    globalTaskStatus_->setMessage(
                        TaskStatusWidget::MessageKind::Error,
                        elapsedMs < 0 ? QStringLiteral("任务失败 · 耗时未知")
                                      : QStringLiteral("任务失败"));
                } else if (elapsedMs < 0) {
                    globalTaskStatus_->setMessage(
                        TaskStatusWidget::MessageKind::Neutral,
                        QStringLiteral("%1 · %2/%3 · 耗时未知")
                            .arg(phase).arg(completed).arg(total));
                }
            });
    connect(appHeader_, &AppHeader::pageRequested,
            this, [this](AppPage page) { requestPage(page); });
    connect(appHeader_, &AppHeader::restartBackendRequested,
            this, &MainWindow::restartBackend);
    connect(appHeader_, &AppHeader::currentWorkpieceRequested, this, [this](const QString &) {
        if (annotationManagerDialog_ != nullptr) {
            annotationManagerDialog_->close();
            annotationManagerDialog_.clear();
            annotationWorkpieceId_.clear();
        }
        inspectionPage_->setCurrentWorkpiece(appHeader_->currentWorkpieceId(),
                                             appHeader_->currentWorkpieceName());
        workpieceLibraryPage_->setWorkpieces(
            workpieceSummaries_, appHeader_->currentWorkpieceId());
        if (resultContext_ != ResultContext::Batch) {
            lastPredictionWorkpieceId_.clear();
            lastPredictionImagePath_.clear();
            lastPredictionResponse_ = QJsonObject();
        }
        updateButtonStates();
    });
    connect(annotationEditorButton_, &QPushButton::clicked,
            this, &MainWindow::openAnnotationManager);
    confirmFrontButton_->setEnabled(false);
    confirmBackButton_->setEnabled(false);
    rejectConfirmationButton_->setEnabled(false);
}

bool MainWindow::requestPage(AppPage page) {
    if (page == currentPage_) return true;
    if (currentPage_ == AppPage::WorkpieceLibrary
        && workpieceLibraryPage_ != nullptr
        && workpieceLibraryPage_->hasUnsavedChanges()) {
        QMessageBox prompt(this);
        prompt.setWindowTitle(QStringLiteral("未保存的工件草稿"));
        prompt.setText(QStringLiteral("工件库中有未保存的编辑草稿，如何处理？"));
        QAbstractButton *keep = prompt.addButton(
            QStringLiteral("保留并离开"), QMessageBox::AcceptRole);
        QAbstractButton *discard = prompt.addButton(
            QStringLiteral("放弃修改"), QMessageBox::DestructiveRole);
        QAbstractButton *cancel = prompt.addButton(
            QStringLiteral("取消"), QMessageBox::RejectRole);
        prompt.exec();
        if (prompt.clickedButton() == cancel || prompt.clickedButton() == nullptr) {
            appHeader_->setCurrentPage(currentPage_);
            return false;
        }
        if (prompt.clickedButton() == discard) {
            workpieceLibraryPage_->discardEditingDraft();
        }
        Q_UNUSED(keep)
    }
    ui->mainPageStack->setCurrentIndex(static_cast<int>(page));
    currentPage_ = page;
    appHeader_->setCurrentPage(page);
    return true;
}

void MainWindow::showInspection() {
    requestPage(AppPage::Inspection);
}

void MainWindow::showWorkpieceLibrary() {
    requestPage(AppPage::WorkpieceLibrary);
}

void MainWindow::showGeometryRules() {
    requestPage(AppPage::GeometryRules);
}

void MainWindow::connectBackendSignals() {
    if (client_ == nullptr) {
        return;
    }
    connect(client_, &BackendClient::handshakeSucceeded,
            this, &MainWindow::onBackendReady);
    connect(client_, &BackendClient::stateChanged,
            this, &MainWindow::onClientStateChanged);
    connect(client_, &BackendClient::progressReceived,
            this, &MainWindow::onClientProgress);
    connect(client_, &BackendClient::responseReceived,
            this, &MainWindow::onClientResponse);
    connect(client_, &BackendClient::commandFailed,
            this, &MainWindow::onClientCommandFailed);
    connect(client_, &BackendClient::transportFailed,
            this, &MainWindow::onClientTransportFailed);
    if (manager_ != nullptr) {
        connect(manager_, &BackendProcessManager::backendReady,
                this, &MainWindow::onBackendReady);
        connect(manager_, &BackendProcessManager::backendLoading,
                this, &MainWindow::onBackendLoading);
        connect(manager_, &BackendProcessManager::backendUnavailable,
                this, &MainWindow::onBackendUnavailable);
    }
}

void MainWindow::sendPageCommand(CommandOwner owner, const QString &command,
                                 const QJsonObject &fields) {
    if (client_ == nullptr || command.isEmpty()) return;
    if (clientBusy_ || !pendingCommand_.isEmpty()
        || client_->state() != BackendClient::State::Ready) {
        const bool replaceContinuation = owner == CommandOwner::Library
            && command == QStringLiteral("register")
            && fields.value(QStringLiteral("replace")).toBool();
        const bool latestBrowseIntent = owner == CommandOwner::Library
            && command == QStringLiteral("get_workpiece_details")
            && queuedOwner_ == CommandOwner::Library
            && queuedCommand_ == QStringLiteral("get_workpiece_details");
        if (queuedCommand_.isEmpty() || replaceContinuation || latestBrowseIntent) {
            queuedOwner_ = owner;
            queuedCommand_ = command;
            queuedFields_ = fields;
        } else if (owner == CommandOwner::Inspection
                   && command == QStringLiteral("predict")
                   && queuedOwner_ == CommandOwner::System
                   && (queuedCommand_ == QStringLiteral("list_evolution_jobs")
                       || queuedCommand_ == QStringLiteral("list_workpieces"))) {
            queuedOwner_ = owner;
            queuedCommand_ = command;
            queuedFields_ = fields;
        } else {
            showLibraryMessage(
                QStringLiteral("已有后续操作等待发送，请稍后重试：%1").arg(command), true);
        }
        return;
    }
    pendingOwner_ = owner;
    pendingCommand_ = command;
    pendingFields_ = fields;
    client_->sendRequest(command, fields);
}

void MainWindow::dispatchQueuedCommand() {
    if (queuedCommand_.isEmpty() || client_ == nullptr || clientBusy_
        || !pendingCommand_.isEmpty()
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    const CommandOwner owner = queuedOwner_;
    const QString command = queuedCommand_;
    const QJsonObject fields = queuedFields_;
    queuedOwner_ = CommandOwner::None;
    queuedCommand_.clear();
    queuedFields_ = QJsonObject();
    sendPageCommand(owner, command, fields);
}

void MainWindow::clearPendingCommand() {
    pendingOwner_ = CommandOwner::None;
    pendingCommand_.clear();
    pendingFields_ = QJsonObject();
}

void MainWindow::setTemplatePaths(const QStringList &frontPaths, const QStringList &backPaths) {
    workpieceLibraryPage_->setTemplatePaths(frontPaths, backPaths);
}

void MainWindow::setWorkpieceName(const QString &name) {
    workpieceLibraryPage_->setWorkpieceName(name);
}

void MainWindow::setInspectionImagePath(const QString &path) {
    inspectionImagePath_ = QFileInfo(path).absoluteFilePath();
    inspectionPage_->setMode(InspectionMode::Single);
    inspectionPage_->setSingleImagePath(inspectionImagePath_);
    resultContext_ = ResultContext::None;
    lastPredictionImagePath_.clear();
    lastPredictionWorkpieceId_.clear();
    lastPredictionResponse_ = QJsonObject();
    updateButtonStates();
}

void MainWindow::setBatchImagePaths(const QStringList &paths) {
    batchImagePaths_ = normalizedPaths(paths);
    batchInFlight_ = false;
    lastPredictionImagePath_.clear();
    lastPredictionWorkpieceId_.clear();
    lastPredictionResponse_ = QJsonObject();
    clearBatchResults();
    inspectionPage_->setMode(InspectionMode::Batch);
    chooseBatchImagesButton_->setText(batchImagePaths_.isEmpty()
                                             ? QStringLiteral("选择批量图片")
                                             : QStringLiteral("选择批量图片（%1）").arg(batchImagePaths_.size()));
    updateButtonStates();
}

void MainWindow::setReplaceConfirmationHandler(std::function<bool(const QString &)> handler) {
    workpieceLibraryPage_->setReplaceConfirmationHandler(std::move(handler));
}

void MainWindow::setBackendError(const QString &message) {
    backendReady_ = false;
    backendReadyHandled_ = false;
    clientBusy_ = false;
    queuedOwner_ = CommandOwner::None;
    queuedCommand_.clear();
    queuedFields_ = QJsonObject();
    BackendStatusDetails details;
    details.state = BackendUiState::Error;
    details.connectionDetail = QStringLiteral("配置错误");
    details.modelDetail = QStringLiteral("未加载");
    details.recentError = message;
    details.canRestart = false;
    appHeader_->setBackendDetails(details);
    workpieceLibraryPage_->setBackendState(BackendUiState::Error, message);
    showLibraryMessage(message, true);
    updateButtonStates();
}

void MainWindow::chooseInspectionImage() {
    const QString path = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择待测图片"), QString(), dialogFilters().constFirst());
    if (!path.isEmpty()) {
        setInspectionImagePath(path);
    }
}

void MainWindow::chooseBatchImages() {
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, QStringLiteral("选择批量待测图片"), QString(), dialogFilters().constFirst());
    if (!paths.isEmpty()) {
        setBatchImagePaths(paths);
    }
}

void MainWindow::refreshWorkpieces() {
    if (client_ == nullptr || !backendReady_) {
        return;
    }
    deferredUserWorkpieceRefresh_ = true;
    if (batchInFlight_ || clientBusy_ || !pendingCommand_.isEmpty()
        || !queuedCommand_.isEmpty()
        || client_->state() != BackendClient::State::Ready) {
        showLibraryMessage(QStringLiteral("当前操作完成后刷新工件列表…"));
        return;
    }
    dispatchDeferredUserRefresh();
}

void MainWindow::requestWorkpieceRefresh(bool preserveRegistrationSummary,
                                         CommandOwner owner) {
    if (client_ == nullptr || !backendReady_) {
        return;
    }
    sendPageCommand(owner, QStringLiteral("list_workpieces"), QJsonObject());
    if (!preserveRegistrationSummary) {
        showLibraryMessage(QStringLiteral("正在刷新工件列表…"));
    }
}

void MainWindow::dispatchDeferredUserRefresh() {
    if (!deferredUserWorkpieceRefresh_ || batchInFlight_ || client_ == nullptr
        || !backendReady_ || clientBusy_ || !pendingCommand_.isEmpty()
        || !queuedCommand_.isEmpty()
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    requestWorkpieceRefresh(false, CommandOwner::UserRefresh);
}

void MainWindow::submitRegistration() {
    QMetaObject::invokeMethod(workpieceLibraryPage_, "submitRegistration",
                              Qt::DirectConnection);
}

void MainWindow::submitPrediction() {
    if (predictButton_ != nullptr) predictButton_->click();
}

void MainWindow::submitBatchPrediction() {
    if (client_ == nullptr || !backendReady_ || clientBusy_ || batchInFlight_
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    const QString workpieceId = selectedWorkpieceId();
    if (workpieceId.isEmpty()) {
        batchSummaryLabel_->setText(QStringLiteral("请先选择工件"));
        return;
    }
    if (batchImagePaths_.isEmpty()) {
        batchSummaryLabel_->setText(QStringLiteral("请先选择批量图片"));
        return;
    }
    for (const QString &path : batchImagePaths_) {
        QString error;
        if (!validateImagePath(path, &error)) {
            batchSummaryLabel_->setText(QStringLiteral("批量检测无法开始：%1").arg(error));
            return;
        }
    }
    batchInFlight_ = true;
    resultContext_ = ResultContext::Batch;
    inspectionPage_->beginBatch(batchImagePaths_, workpieceId);
    updateButtonStates();
}

void MainWindow::confirmFrontTemplate() {
    if (confirmFrontButton_ != nullptr) confirmFrontButton_->click();
}

void MainWindow::confirmBackTemplate() {
    if (confirmBackButton_ != nullptr) confirmBackButton_->click();
}

void MainWindow::rejectTemplateConfirmation() {
    lastPredictionImagePath_.clear();
    lastPredictionWorkpieceId_.clear();
    lastPredictionResponse_ = QJsonObject();
    confirmFrontButton_->setEnabled(false);
    confirmBackButton_->setEnabled(false);
    rejectConfirmationButton_->setEnabled(false);
    reviewLabel_->setText(QStringLiteral("本次结果不入库"));
}

void MainWindow::openAnnotationEditor() {
    openAnnotationManager();
}

void MainWindow::openAnnotationManager() {
    openGeometryMaskManager();
}

void MainWindow::openGeometryMaskManager() {
    const QString workpieceId = workpieceLibraryPage_ != nullptr
        && !workpieceLibraryPage_->browsedWorkpieceId().isEmpty()
        ? workpieceLibraryPage_->browsedWorkpieceId() : selectedWorkpieceId();
    if (workpieceId.isEmpty() || client_ == nullptr || !backendReady_ || clientBusy_
        || client_->state() != BackendClient::State::Ready) {
        showLibraryMessage(QStringLiteral("请先选择已建立的工件库，且等待后端空闲"), true);
        return;
    }
    geometryWorkpieceId_ = workpieceId;
    geometryPublishAfterValidation_ = false;
    geometryPublishOverrideReason_.clear();
    geometryValidationJobId_.clear();
    if (geometryPollTimer_ != nullptr) geometryPollTimer_->stop();
    if (geometryMaskManagerDialog_ == nullptr) {
        geometryMaskManagerDialog_ = new GeometryMaskManagerDialog(this);
        geometryMaskManagerDialog_->setAttribute(Qt::WA_DeleteOnClose);
        connect(geometryMaskManagerDialog_, &GeometryMaskManagerDialog::saveDraftRequested,
                this, &MainWindow::saveGeometryDraft);
        connect(geometryMaskManagerDialog_, &GeometryMaskManagerDialog::validateRequested,
                this, &MainWindow::validateGeometryDraft);
        connect(geometryMaskManagerDialog_, &GeometryMaskManagerDialog::validationJobActionRequested,
                this, &MainWindow::geometryJobAction);
        connect(geometryMaskManagerDialog_, &GeometryMaskManagerDialog::publishRequested,
                this, &MainWindow::publishGeometryProfile);
        connect(geometryMaskManagerDialog_, &GeometryMaskManagerDialog::publishWorkflowRequested,
                this, &MainWindow::publishGeometryWorkflow);
        connect(geometryMaskManagerDialog_, &GeometryMaskManagerDialog::rollbackRequested,
                this, &MainWindow::rollbackGeometryProfile);
        connect(geometryMaskManagerDialog_, &GeometryMaskManagerDialog::previewRequested,
                this, &MainWindow::previewGeometryRule);
        connect(geometryMaskManagerDialog_, &GeometryMaskManagerDialog::migrationResolutionRequested,
                this, &MainWindow::resolveGeometryMigration);
    }
    geometryMaskManagerDialog_->setBusy(true);
    geometryMaskManagerDialog_->show();
    geometryMaskManagerDialog_->raise();
    geometryMaskManagerDialog_->activateWindow();
    requestGeometryProfile(workpieceId);
    showLibraryMessage(QStringLiteral("正在加载几何干扰规则…"));
}

void MainWindow::requestGeometryProfile(const QString &workpieceId) {
    if (workpieceId.isEmpty() || client_ == nullptr || !backendReady_) {
        return;
    }
    if (client_->state() != BackendClient::State::Ready || clientBusy_) {
        QTimer::singleShot(20, this, [this, workpieceId]() { requestGeometryProfile(workpieceId); });
        return;
    }
    geometryWorkpieceId_ = workpieceId;
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("get_geometry_mask_profile"), {
        {QStringLiteral("workpiece_id"), workpieceId},
    });
}

void MainWindow::saveGeometryDraft(const QJsonObject &draft, int libraryRevision, int draftRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("save_geometry_mask_draft"), {
        {QStringLiteral("workpiece_id"), geometryWorkpieceId_},
        {QStringLiteral("base_library_revision"), libraryRevision},
        {QStringLiteral("base_draft_revision"), draftRevision},
        {QStringLiteral("draft"), draft},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
}

void MainWindow::publishGeometryWorkflow(const QJsonObject &draft, int libraryRevision, int draftRevision,
                                         const QString &overrideReason) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    geometryPublishAfterValidation_ = true;
    geometryPublishOverrideReason_ = overrideReason.trimmed();
    saveGeometryDraft(draft, libraryRevision, draftRevision);
}

void MainWindow::validateGeometryDraft(int libraryRevision, int draftRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("validate_geometry_mask_draft"), {
        {QStringLiteral("workpiece_id"), geometryWorkpieceId_},
        {QStringLiteral("base_library_revision"), libraryRevision},
        {QStringLiteral("base_draft_revision"), draftRevision},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
}

void MainWindow::pollGeometryValidation() {
    if (geometryValidationJobId_.isEmpty() || client_ == nullptr || !backendReady_ || clientBusy_
        || !pendingCommand_.isEmpty() || client_->state() != BackendClient::State::Ready) return;
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("get_geometry_mask_validation_job"), {
        {QStringLiteral("job_id"), geometryValidationJobId_},
    });
}

bool MainWindow::maybeContinueGeometryPublish(const QJsonObject &job) {
    if (!geometryPublishAfterValidation_) return false;
    const QString state = job.value(QStringLiteral("state")).toString();
    if (state == QStringLiteral("queued") || state == QStringLiteral("running")) return false;

    geometryPublishAfterValidation_ = false;
    const QJsonArray blocking = job.value(QStringLiteral("blocking_issues")).toArray();
    const QJsonArray warnings = job.value(QStringLiteral("warnings")).toArray();
    const bool hasRegression = job.value(QStringLiteral("regression")).toObject()
                                   .value(QStringLiteral("correct_to_wrong")).toInt(0) > 0;
    if (state != QStringLiteral("completed")) {
        showLibraryMessage(QStringLiteral("几何规则验证未完成，未自动发布"), true);
        geometryPublishOverrideReason_.clear();
        return true;
    }
    if (!blocking.isEmpty()) {
        showLibraryMessage(QStringLiteral("验证存在阻断项，请在右侧模板表中逐项处理后再发布"), true);
        geometryPublishOverrideReason_.clear();
        return true;
    }
    if ((!warnings.isEmpty() || hasRegression) && geometryPublishOverrideReason_.isEmpty()) {
        showLibraryMessage(QStringLiteral("验证有告警，请填写发布覆盖原因后点击发布规则"), true);
        return true;
    }

    const QString jobId = job.value(QStringLiteral("job_id")).toString();
    const int libraryRevision = job.value(QStringLiteral("base_library_revision")).toInt();
    const int draftRevision = job.value(QStringLiteral("base_draft_revision")).toInt();
    const QString reason = geometryPublishOverrideReason_;
    geometryPublishOverrideReason_.clear();
    QTimer::singleShot(0, this, [this, jobId, libraryRevision, draftRevision, reason]() {
        publishGeometryProfile(jobId, libraryRevision, draftRevision, reason);
    });
    return true;
}

void MainWindow::geometryJobAction(const QString &jobId, const QString &action) {
    if (client_ == nullptr || clientBusy_ || client_->state() != BackendClient::State::Ready) return;
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("geometry_mask_validation_job_action"), {
        {QStringLiteral("job_id"), jobId}, {QStringLiteral("action"), action},
    });
}

void MainWindow::publishGeometryProfile(const QString &jobId, int libraryRevision, int draftRevision,
                                         const QString &overrideReason) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    geometryPublishAfterValidation_ = false;
    geometryPublishOverrideReason_.clear();
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("publish_geometry_mask_profile"), {
        {QStringLiteral("workpiece_id"), geometryWorkpieceId_},
        {QStringLiteral("job_id"), jobId},
        {QStringLiteral("base_library_revision"), libraryRevision},
        {QStringLiteral("base_draft_revision"), draftRevision},
        {QStringLiteral("override_reason"), overrideReason},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
}

void MainWindow::rollbackGeometryProfile(int libraryRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("rollback_geometry_mask_profile"), {
        {QStringLiteral("workpiece_id"), geometryWorkpieceId_},
        {QStringLiteral("base_library_revision"), libraryRevision},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
}

void MainWindow::previewGeometryRule(const QJsonObject &request) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    QJsonObject fields = request;
    fields.insert(QStringLiteral("workpiece_id"), geometryWorkpieceId_);
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setPreviewBusy(true);
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("preview_geometry_mask_rule"), fields);
}

void MainWindow::resolveGeometryMigration(const QString &conflictId, const QJsonObject &resolution,
                                          int libraryRevision, int draftRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("resolve_geometry_mask_migration"), {
        {QStringLiteral("workpiece_id"), geometryWorkpieceId_},
        {QStringLiteral("conflict_id"), conflictId},
        {QStringLiteral("resolution"), resolution},
        {QStringLiteral("base_library_revision"), libraryRevision},
        {QStringLiteral("base_draft_revision"), draftRevision},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
}

void MainWindow::requestAnnotationSnapshot(const QString &workpieceId) {
    if (workpieceId.isEmpty() || client_ == nullptr || !backendReady_) {
        return;
    }
    if (client_->state() != BackendClient::State::Ready || clientBusy_) {
        QTimer::singleShot(20, this, [this, workpieceId]() { requestAnnotationSnapshot(workpieceId); });
        return;
    }
    annotationWorkpieceId_ = workpieceId;
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("get_workpiece_annotations"), {
        {QStringLiteral("workpiece_id"), workpieceId},
    });
}

void MainWindow::sendAnnotationMutation(const QString &command, const QJsonObject &fields) {
    if (client_ == nullptr || !backendReady_ || clientBusy_
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    sendPageCommand(CommandOwner::Geometry, command, fields);
    if (annotationManagerDialog_ != nullptr) {
        annotationManagerDialog_->setBusy(true);
    }
    showLibraryMessage(QStringLiteral("干扰标注正在保存…"));
}

void MainWindow::saveAnnotationGroups(const QJsonArray &groups, int baseRevision) {
    if (annotationWorkpieceId_.isEmpty()) return;
    sendAnnotationMutation(QStringLiteral("save_workpiece_annotations"), {
        {QStringLiteral("workpiece_id"), annotationWorkpieceId_},
        {QStringLiteral("groups"), groups},
        {QStringLiteral("base_revision"), baseRevision},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
        {QStringLiteral("progress_events"), true},
    });
}

void MainWindow::setAnnotationGroupEnabled(const QString &groupId, bool enabled, int baseRevision) {
    if (annotationWorkpieceId_.isEmpty()) return;
    sendAnnotationMutation(QStringLiteral("set_workpiece_annotation_group_enabled"), {
        {QStringLiteral("workpiece_id"), annotationWorkpieceId_},
        {QStringLiteral("group_id"), groupId},
        {QStringLiteral("enabled"), enabled},
        {QStringLiteral("base_revision"), baseRevision},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
}

void MainWindow::deleteAnnotationGroup(const QString &groupId, int baseRevision) {
    if (annotationWorkpieceId_.isEmpty()) return;
    sendAnnotationMutation(QStringLiteral("delete_workpiece_annotation_group"), {
        {QStringLiteral("workpiece_id"), annotationWorkpieceId_},
        {QStringLiteral("group_id"), groupId},
        {QStringLiteral("base_revision"), baseRevision},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
}

void MainWindow::reviewAnnotation(const QString &groupId, const QString &templateId,
                                  const QString &action, const QJsonArray &regions, int baseRevision) {
    if (annotationWorkpieceId_.isEmpty()) return;
    const QJsonObject annotation{
        {QStringLiteral("template_id"), templateId},
        {QStringLiteral("review_action"), action},
        {QStringLiteral("regions"), regions},
    };
    saveAnnotationGroups(QJsonArray{QJsonObject{
        {QStringLiteral("group_id"), groupId},
        {QStringLiteral("annotations"), QJsonArray{annotation}},
    }}, baseRevision);
}

void MainWindow::submitTemplateConfirmation(const QString &workpieceId, const QString &imagePath,
                                            const QString &orientation) {
    if (imagePath.isEmpty() || workpieceId.isEmpty()
        || client_ == nullptr || clientBusy_ || !backendReady_) {
        return;
    }
    QJsonObject mutation = uncertainConfirmationMutations_.value(
        pendingConfirmationRecordId_);
    if (!mutation.isEmpty()) {
        const bool identicalPayload = mutation.value(QStringLiteral("workpiece_id")).toString()
                == workpieceId
            && mutation.value(QStringLiteral("orientation")).toString() == orientation
            && mutation.value(QStringLiteral("image_path")).toString() == imagePath;
        if (!identicalPayload) {
            const QString originalOrientation = orientationText(
                mutation.value(QStringLiteral("orientation")).toString());
            const QString message = QStringLiteral(
                "上次%1入库结果未知，只能按原方向重试；请先刷新核对或点击原方向确认")
                                        .arg(originalOrientation);
            inspectionPage_->setRecordDisposition(
                pendingConfirmationRecordId_, BatchDisposition::SubmitFailed,
                QString(), message);
            pendingConfirmationRecordId_.clear();
            pendingConfirmationOrientation_.clear();
            pendingConfirmationMutation_ = QJsonObject();
            showLibraryMessage(message, true);
            updateButtonStates();
            return;
        }
    } else {
        mutation = QJsonObject{
            {QStringLiteral("workpiece_id"), workpieceId},
            {QStringLiteral("orientation"), orientation},
            {QStringLiteral("image_path"), imagePath},
            {QStringLiteral("operation_id"),
             QUuid::createUuid().toString(QUuid::WithoutBraces)},
        };
    }
    pendingConfirmationMutation_ = mutation;
    sendPageCommand(CommandOwner::Inspection, QStringLiteral("submit_confirmation"), mutation);
    confirmFrontButton_->setEnabled(false);
    confirmBackButton_->setEnabled(false);
    rejectConfirmationButton_->setEnabled(false);
    showLibraryMessage(QStringLiteral("已确认，正在后台排队入库…"));
}

void MainWindow::restartBackend() {
    if (manager_ != nullptr) {
        manager_->restart();
    }
}

void MainWindow::onBackendReady() {
    const bool firstReadySignal = !backendReadyHandled_;
    backendReadyHandled_ = true;
    backendReady_ = true;
    clientBusy_ = false;
    if (annotationManagerDialog_ != nullptr) {
        annotationManagerDialog_->setBusy(false);
    }
    if (geometryMaskManagerDialog_ != nullptr) {
        geometryMaskManagerDialog_->setBusy(false);
    }
    BackendStatusDetails details;
    details.state = BackendUiState::Ready;
    details.connectionDetail = QStringLiteral("已连接");
    details.modelDetail = QStringLiteral("已加载");
    details.canRestart = true;
    appHeader_->setBackendDetails(details);
    workpieceLibraryPage_->setBackendState(BackendUiState::Ready, QString());
    updateButtonStates();
    if (evolutionPollTimer_ != nullptr) {
        evolutionPollTimer_->start();
    }
    if (firstReadySignal) {
        QTimer::singleShot(0, this, [this]() {
            requestWorkpieceRefresh(false);
        });
    }
}

void MainWindow::onBackendLoading(const QString &message) {
    backendReadyHandled_ = false;
    backendReady_ = false;
    clientBusy_ = false;
    if (annotationManagerDialog_ != nullptr) {
        annotationManagerDialog_->setBusy(true);
    }
    if (geometryMaskManagerDialog_ != nullptr) {
        geometryMaskManagerDialog_->setBusy(true);
    }
    BackendStatusDetails details;
    details.state = BackendUiState::Loading;
    details.connectionDetail = QStringLiteral("已连接");
    details.modelDetail = message.isEmpty() ? QStringLiteral("模型加载中") : message;
    details.canRestart = manager_ != nullptr;
    appHeader_->setBackendDetails(details);
    workpieceLibraryPage_->setBackendState(BackendUiState::Loading, details.modelDetail);
    showLibraryMessage(message.isEmpty() ? QStringLiteral("正在加载模型，请稍候…") : message);
    updateButtonStates();
}

void MainWindow::onBackendUnavailable(const QString &reason) {
    const CommandOwner interruptedOwner = pendingOwner_;
    const QString interruptedTask = pendingCommand_;
    const bool hasPreservedWork = workpieceLibraryPage_->hasUnsavedChanges()
        || batchInFlight_ || clientBusy_
        || !pendingCommand_.isEmpty() || !inspectionImagePath_.isEmpty()
        || (batchResultsTableWidget_ != nullptr && batchResultsTableWidget_->rowCount() > 0)
        || annotationManagerDialog_ != nullptr || geometryMaskManagerDialog_ != nullptr;
    if (interruptedOwner == CommandOwner::UserRefresh) {
        deferredUserWorkpieceRefresh_ = true;
    }
    if (annotationManagerDialog_ != nullptr) {
        annotationManagerDialog_->setBusy(false);
        annotationManagerDialog_->setOperationError(
            QStringLiteral("后端连接中断，操作结果未知；请重连后刷新"));
    }
    if (geometryMaskManagerDialog_ != nullptr) {
        geometryMaskManagerDialog_->setBusy(false);
        geometryMaskManagerDialog_->setPreviewBusy(false);
        geometryMaskManagerDialog_->setOperationError(
            QStringLiteral("后端连接中断，操作结果未知；请重连后刷新"));
    }
    if (!pendingConfirmationRecordId_.isEmpty()) {
        if (!pendingConfirmationMutation_.isEmpty()) {
            uncertainConfirmationMutations_.insert(
                pendingConfirmationRecordId_, pendingConfirmationMutation_);
        }
        inspectionPage_->setRecordDisposition(
            pendingConfirmationRecordId_, BatchDisposition::SubmitFailed, QString(),
            QStringLiteral("连接中断，入库结果未知；请刷新后确认"));
        pendingConfirmationRecordId_.clear();
        pendingConfirmationOrientation_.clear();
        pendingConfirmationMutation_ = QJsonObject();
    }
    backendReadyHandled_ = false;
    backendReady_ = false;
    clientBusy_ = false;
    batchInFlight_ = false;
    clearPendingCommand();
    queuedOwner_ = CommandOwner::None;
    queuedCommand_.clear();
    queuedFields_ = QJsonObject();
    if (interruptedOwner == CommandOwner::Inspection
        && inspectionPage_ != nullptr && !interruptedTask.isEmpty()) {
        inspectionPage_->handleBackendFailure(interruptedTask,
                                              QStringLiteral("CONNECTION_LOST"), reason);
    }
    BackendStatusDetails details;
    details.state = BackendUiState::Error;
    details.connectionDetail = QStringLiteral("连接中断");
    details.modelDetail = QStringLiteral("状态未知");
    details.currentTask = interruptedTask;
    details.recentError = reason;
    details.canRestart = true;
    appHeader_->setBackendDetails(details);
    workpieceLibraryPage_->setBackendState(BackendUiState::Error, reason);
    showLibraryMessage(hasPreservedWork
                           ? QStringLiteral("后端连接中断，未完成操作结果未知；请重连后刷新：%1").arg(reason)
                           : reason,
                       true);
    if (evolutionPollTimer_ != nullptr) {
        evolutionPollTimer_->stop();
    }
    if (geometryPollTimer_ != nullptr) {
        geometryPollTimer_->stop();
    }
    inspectionPage_->setBackendAvailable(false, false, reason);
    updateButtonStates();
}

void MainWindow::pollEvolutionJobs() {
    if (batchInFlight_ || !backendReady_ || clientBusy_ || !pendingCommand_.isEmpty()
        || client_ == nullptr || client_->state() != BackendClient::State::Ready) {
        return;
    }
    sendPageCommand(CommandOwner::System, QStringLiteral("list_evolution_jobs"), QJsonObject());
}

void MainWindow::onClientStateChanged(BackendClient::State state, const QString &detail) {
    clientBusy_ = state == BackendClient::State::Busy;
    if (state != BackendClient::State::Ready && state != BackendClient::State::Busy) {
        backendReadyHandled_ = false;
    }
    BackendStatusDetails details;
    details.connectionDetail = detail;
    details.canRestart = manager_ != nullptr || state == BackendClient::State::Error;
    if (state == BackendClient::State::Ready) {
        backendReady_ = true;
        details.state = BackendUiState::Ready;
        details.modelDetail = QStringLiteral("已加载");
    } else if (state == BackendClient::State::Busy) {
        details.state = BackendUiState::Busy;
        details.modelDetail = QStringLiteral("已加载");
        details.currentTask = detail;
    } else if (state == BackendClient::State::Connecting
               || state == BackendClient::State::Handshaking) {
        details.state = BackendUiState::Loading;
        details.modelDetail = detail;
    } else if (state == BackendClient::State::Error) {
        details.state = BackendUiState::Error;
        details.modelDetail = QStringLiteral("状态未知");
        details.recentError = detail;
    } else {
        details.state = BackendUiState::Disconnected;
        details.modelDetail = QStringLiteral("未加载");
    }
    appHeader_->setBackendDetails(details);
    workpieceLibraryPage_->setBackendState(details.state, detail);
    updateButtonStates();
    if (state == BackendClient::State::Ready) {
        dispatchQueuedCommand();
        dispatchDeferredUserRefresh();
    }
}

void MainWindow::onClientProgress(const QString &command, const QJsonObject &progress) {
    if (command == QStringLiteral("save_workpiece_annotations")) {
        if (annotationManagerDialog_ != nullptr) {
            annotationManagerDialog_->setProgress(progress.value(QStringLiteral("phase")).toString(),
                                                  progress.value(QStringLiteral("completed")).toInt(),
                                                  progress.value(QStringLiteral("total")).toInt());
        }
        return;
    }
    if (command == QStringLiteral("register")) {
        workpieceLibraryPage_->setRegistrationProgress(progress, -1);
    }
}

void MainWindow::onClientResponse(const QString &command, const QJsonObject &response) {
    const CommandOwner responseOwner = pendingOwner_;
    const QJsonObject issuedFields = pendingFields_;
    clearPendingCommand();
    if (batchInFlight_ && command == QStringLiteral("predict")) {
        inspectionPage_->handleBackendResponse(command, response);
        return;
    }
    if (command == QStringLiteral("list_workpieces")) {
        const QString previousId = selectedWorkpieceId();
        const QJsonArray workpieces = response.value(QStringLiteral("workpieces")).toArray();
        workpieceSummaries_ = workpieces;
        QList<QPair<QString, QString>> items;
        QString requestedId = previousId;
        bool previousStillExists = false;
        for (const QJsonValue &value : workpieces) {
            const QJsonObject item = value.toObject();
            const QString id = item.value(QStringLiteral("id")).toString();
            const QString name = item.value(QStringLiteral("name")).toString();
            items.append(qMakePair(id, name));
            previousStillExists = previousStillExists || id == previousId;
        }
        if (!previousStillExists) {
            requestedId.clear();
        }
        appHeader_->setWorkpieces(items, requestedId);
        workpieceLibraryPage_->setWorkpieces(workpieces, appHeader_->currentWorkpieceId());
        if (responseOwner == CommandOwner::UserRefresh) {
            deferredUserWorkpieceRefresh_ = false;
        }
        showLibraryMessage(QStringLiteral("工件列表已刷新"));
        updateButtonStates();
        return;
    }
    if (command == QStringLiteral("get_workpiece_details")) {
        const QString requestedId = issuedFields.value(QStringLiteral("workpiece_id")).toString();
        if (requestedId.isEmpty()
            || requestedId == workpieceLibraryPage_->browsedWorkpieceId()) {
            workpieceLibraryPage_->handleBackendResponse(command, response);
        }
        return;
    }
    if (command == QStringLiteral("recycle_workpiece")) {
        lastPredictionWorkpieceId_.clear();
        lastPredictionImagePath_.clear();
        lastPredictionResponse_ = QJsonObject();
        workpieceLibraryPage_->handleBackendResponse(command, response);
        requestWorkpieceRefresh(false);
        return;
    }
    if (command == QStringLiteral("submit_confirmation")) {
        const QJsonObject job = response.value(QStringLiteral("job")).toObject();
        const bool jobFailed = job.value(QStringLiteral("state")).toString()
            == QStringLiteral("failed");
        const QString jobError = job.value(QStringLiteral("error")).toString(
            QStringLiteral("后台入库任务失败"));
        if (!pendingConfirmationRecordId_.isEmpty()) {
            const QString completedRecordId = pendingConfirmationRecordId_;
            inspectionPage_->setRecordDisposition(
                pendingConfirmationRecordId_,
                jobFailed ? BatchDisposition::SubmitFailed
                          : (pendingConfirmationOrientation_ == QStringLiteral("front")
                                 ? BatchDisposition::QueuedFront
                                 : BatchDisposition::QueuedBack),
                job.value(QStringLiteral("job_id")).toString(),
                jobFailed ? jobError : QString());
            pendingConfirmationRecordId_.clear();
            pendingConfirmationOrientation_.clear();
            pendingConfirmationMutation_ = QJsonObject();
            uncertainConfirmationMutations_.remove(completedRecordId);
        } else if (inspectionPage_ != nullptr
                   && (responseOwner == CommandOwner::Inspection
                       || responseOwner == CommandOwner::None)) {
            inspectionPage_->handleBackendResponse(command, response);
        }
        showLibraryMessage(jobFailed
                               ? QStringLiteral("确认入库失败：%1").arg(jobError)
                               : QStringLiteral("确认图片已进入后台入库队列"),
                           jobFailed);
        updateButtonStates();
        return;
    }
    if (command == QStringLiteral("get_geometry_mask_profile")) {
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryMaskManagerDialog_->setBusy(false);
        }
        showLibraryMessage(QStringLiteral("几何干扰规则已加载"));
        return;
    }
    if (command == QStringLiteral("preview_geometry_mask_rule")) {
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setRulePreview(response.value(QStringLiteral("preview")).toObject());
            geometryMaskManagerDialog_->setPreviewBusy(false);
        }
        showLibraryMessage(QStringLiteral("几何规则预览已更新"));
        return;
    }
    if (command == QStringLiteral("save_geometry_mask_draft")) {
        const bool continuePublishWorkflow = geometryPublishAfterValidation_;
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryMaskManagerDialog_->setBusy(false);
        }
        const QJsonObject savedProfile = response.value(QStringLiteral("profile")).toObject();
        showLibraryMessage(continuePublishWorkflow
                               ? QStringLiteral("草稿已保存，正在验证后发布…")
                               : QStringLiteral("几何规则草稿已保存"));
        if (continuePublishWorkflow) {
            const int libraryRevision = savedProfile.value(QStringLiteral("library_revision")).toInt();
            const int draftRevision = savedProfile.value(QStringLiteral("draft_revision")).toInt();
            QTimer::singleShot(0, this, [this, libraryRevision, draftRevision]() {
                validateGeometryDraft(libraryRevision, draftRevision);
            });
        }
        return;
    }
    if (command == QStringLiteral("resolve_geometry_mask_migration")) {
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryMaskManagerDialog_->setBusy(false);
        }
        showLibraryMessage(QStringLiteral("迁移冲突处置已保存"));
        return;
    }
    if (command == QStringLiteral("validate_geometry_mask_draft")) {
        const QJsonObject job = response.value(QStringLiteral("job")).toObject();
        geometryValidationJobId_ = job.value(QStringLiteral("job_id")).toString();
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setValidationJob(job);
            geometryMaskManagerDialog_->setBusy(false);
        }
        const bool workflowHandled = maybeContinueGeometryPublish(job);
        if (geometryPollTimer_ != nullptr && (job.value(QStringLiteral("state")).toString() == QStringLiteral("queued")
                                               || job.value(QStringLiteral("state")).toString() == QStringLiteral("running"))) {
            geometryPollTimer_->start();
        } else if (workflowHandled && geometryPollTimer_ != nullptr) {
            geometryPollTimer_->stop();
        }
        if (!workflowHandled || job.value(QStringLiteral("state")).toString() == QStringLiteral("queued")
            || job.value(QStringLiteral("state")).toString() == QStringLiteral("running")) {
            showLibraryMessage(QStringLiteral("几何规则验证任务已启动"));
        }
        return;
    }
    if (command == QStringLiteral("get_geometry_mask_validation_job")) {
        const QJsonObject job = response.value(QStringLiteral("job")).toObject();
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setValidationJob(job);
            geometryMaskManagerDialog_->setBusy(false);
        }
        const bool workflowHandled = maybeContinueGeometryPublish(job);
        const QString state = job.value(QStringLiteral("state")).toString();
        if (state != QStringLiteral("queued") && state != QStringLiteral("running")) {
            if (geometryPollTimer_ != nullptr) geometryPollTimer_->stop();
        }
        Q_UNUSED(workflowHandled);
        return;
    }
    if (command == QStringLiteral("geometry_mask_validation_job_action")) {
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setValidationJob(response.value(QStringLiteral("job")).toObject());
            geometryMaskManagerDialog_->setBusy(false);
        }
        return;
    }
    if (command == QStringLiteral("publish_geometry_mask_profile")
        || command == QStringLiteral("rollback_geometry_mask_profile")) {
        if (geometryPollTimer_ != nullptr) geometryPollTimer_->stop();
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryMaskManagerDialog_->setBusy(false);
        }
        showLibraryMessage(command == QStringLiteral("publish_geometry_mask_profile")
                              ? QStringLiteral("几何干扰规则已发布") : QStringLiteral("几何干扰规则已回退"));
        return;
    }
    if (command == QStringLiteral("get_workpiece_annotations")) {
        const QJsonObject annotations = response.value(QStringLiteral("annotations")).toObject();
        if (annotations.isEmpty()) {
            showLibraryMessage(QStringLiteral("后端返回的干扰标注快照为空"), true);
            return;
        }
        if (annotationManagerDialog_ == nullptr) {
            annotationManagerDialog_ = new AnnotationManagerDialog(this);
            annotationManagerDialog_->setAttribute(Qt::WA_DeleteOnClose);
            connect(annotationManagerDialog_, &AnnotationManagerDialog::saveRequested,
                    this, &MainWindow::saveAnnotationGroups);
            connect(annotationManagerDialog_, &AnnotationManagerDialog::setEnabledRequested,
                    this, &MainWindow::setAnnotationGroupEnabled);
            connect(annotationManagerDialog_, &AnnotationManagerDialog::deleteRequested,
                    this, &MainWindow::deleteAnnotationGroup);
            connect(annotationManagerDialog_, &AnnotationManagerDialog::reviewRequested,
                    this, &MainWindow::reviewAnnotation);
            connect(annotationManagerDialog_, &AnnotationManagerDialog::refreshRequested,
                    this, [this]() { requestAnnotationSnapshot(annotationWorkpieceId_); });
        }
        annotationWorkpieceId_ = annotations.value(QStringLiteral("workpiece_id")).toString(annotationWorkpieceId_);
        annotationManagerDialog_->setSnapshot(annotations);
        annotationManagerDialog_->setBusy(false);
        annotationManagerDialog_->show();
        annotationManagerDialog_->raise();
        annotationManagerDialog_->activateWindow();
        showLibraryMessage(QStringLiteral("干扰标注快照已加载"));
        return;
    }
    if (command == QStringLiteral("save_workpiece_annotations")
        || command == QStringLiteral("set_workpiece_annotation_group_enabled")
        || command == QStringLiteral("delete_workpiece_annotation_group")) {
        showLibraryMessage(QStringLiteral("干扰标注已保存，正在刷新生效状态…"));
        if (annotationManagerDialog_ != nullptr) {
            annotationManagerDialog_->setBusy(true);
        }
        const QString workpieceId = annotationWorkpieceId_;
        QTimer::singleShot(0, this, [this, workpieceId]() { requestAnnotationSnapshot(workpieceId); });
        return;
    }
    if (command == QStringLiteral("list_evolution_jobs")) {
        const QJsonArray jobs = response.value(QStringLiteral("jobs")).toArray();
        inspectionPage_->handleBackendResponse(command, response);
        workpieceLibraryPage_->setEvolutionJobs(jobs);
        if (!jobs.isEmpty()) {
            const QJsonObject latest = jobs.last().toObject();
            showLibraryMessage(QStringLiteral("后台入库：%1，进度 %2%%")
                                   .arg(latest.value(QStringLiteral("state")).toString())
                                   .arg(latest.value(QStringLiteral("progress")).toInt()));
        }
        return;
    }
    if (command == QStringLiteral("register")) {
        workpieceLibraryPage_->handleBackendResponse(command, response);
        QTimer::singleShot(0, this, [this]() {
            requestWorkpieceRefresh(true);
        });
        return;
    }
    if (command == QStringLiteral("predict")
        && (responseOwner == CommandOwner::Inspection
            || responseOwner == CommandOwner::None)) {
        lastPredictionWorkpieceId_ = pendingPredictionWorkpieceId_.isEmpty()
            ? selectedWorkpieceId() : pendingPredictionWorkpieceId_;
        lastPredictionImagePath_ = pendingPredictionImagePath_.isEmpty()
            ? inspectionPage_->singleImagePath() : pendingPredictionImagePath_;
        lastPredictionResponse_ = response;
        resultContext_ = ResultContext::Single;
        inspectionPage_->setMode(InspectionMode::Single);
        inspectionPage_->handleBackendResponse(command, response);
        updateButtonStates();
    }
}

void MainWindow::onClientCommandFailed(const QString &command, const QString &code,
                                       const QString &message) {
    const CommandOwner failureOwner = pendingOwner_;
    const QString issuedCommand = pendingCommand_;
    const QJsonObject issuedFields = pendingFields_;
    clearPendingCommand();
    const QString failedCommand = issuedCommand.isEmpty() ? command : issuedCommand;
    const auto matchesOwner = [failureOwner](CommandOwner expected) {
        return failureOwner == expected || failureOwner == CommandOwner::None;
    };
    if (failedCommand == QStringLiteral("register")
        && matchesOwner(CommandOwner::Library)) {
        workpieceLibraryPage_->handleBackendFailure(failedCommand, code, message);
        return;
    }
    if (batchInFlight_ && failedCommand == QStringLiteral("predict")
        && matchesOwner(CommandOwner::Inspection)) {
        inspectionPage_->handleBackendFailure(failedCommand, code, message);
        updateButtonStates();
        return;
    }
    if (failedCommand == QStringLiteral("list_workpieces")
               && failureOwner == CommandOwner::UserRefresh) {
        deferredUserWorkpieceRefresh_ = false;
        showLibraryMessage(
            QStringLiteral("刷新失败：%1；请手动重试").arg(message), true);
    } else if (failedCommand == QStringLiteral("get_workpiece_details")
               && matchesOwner(CommandOwner::Library)) {
        const QString requestedId = issuedFields.value(QStringLiteral("workpiece_id")).toString();
        if (requestedId.isEmpty()
            || requestedId == workpieceLibraryPage_->browsedWorkpieceId()) {
            workpieceLibraryPage_->handleBackendFailure(failedCommand, code, message);
        }
    } else if (failedCommand == QStringLiteral("predict")
               && matchesOwner(CommandOwner::Inspection)) {
        if (inspectionPage_ != nullptr
            && (failureOwner == CommandOwner::Inspection
                || failureOwner == CommandOwner::None)) {
            inspectionPage_->handleBackendFailure(failedCommand, code, message);
        }
    } else if (failedCommand == QStringLiteral("submit_confirmation")
               && matchesOwner(CommandOwner::Inspection)) {
        if (!pendingConfirmationRecordId_.isEmpty()) {
            const QString failedRecordId = pendingConfirmationRecordId_;
            inspectionPage_->setRecordDisposition(
                pendingConfirmationRecordId_, BatchDisposition::SubmitFailed,
                QString(), message);
            pendingConfirmationRecordId_.clear();
            pendingConfirmationOrientation_.clear();
            pendingConfirmationMutation_ = QJsonObject();
            uncertainConfirmationMutations_.remove(failedRecordId);
        } else if (inspectionPage_ != nullptr
                   && (failureOwner == CommandOwner::Inspection
                       || failureOwner == CommandOwner::None)) {
            inspectionPage_->handleBackendFailure(failedCommand, code, message);
        }
        showLibraryMessage(QStringLiteral("确认入库失败：%1").arg(message), true);
    } else if (matchesOwner(CommandOwner::Geometry)
               && (failedCommand.startsWith(QStringLiteral("get_geometry_mask"))
                   || failedCommand.startsWith(QStringLiteral("preview_geometry_mask"))
                   || failedCommand.startsWith(QStringLiteral("save_geometry_mask"))
                   || failedCommand.startsWith(QStringLiteral("validate_geometry_mask"))
                   || failedCommand.startsWith(QStringLiteral("publish_geometry_mask"))
                   || failedCommand.startsWith(QStringLiteral("resolve_geometry_mask"))
                   || failedCommand.startsWith(QStringLiteral("rollback_geometry_mask")))) {
        QString actionable = message;
        if (code == QStringLiteral("MISSING_DIRECTION_CALIBRATION")) {
            actionable = QStringLiteral("正面或反面标定未完成，请按左侧规则状态补齐后重试");
        } else if (code == QStringLiteral("MIGRATION_CONFLICT")) {
            actionable = QStringLiteral("请在右侧迁移冲突区明确选择保留、配对或分别保留");
        } else if (code == QStringLiteral("GEOMETRY_CONTEXT_MISMATCH")) {
            actionable = QStringLiteral("预览上下文已变化，过期结果已丢弃，请在当前模板重新绘制");
        } else if (code == QStringLiteral("PROFILE_CACHE_REVISION_MISMATCH")) {
            actionable = QStringLiteral("规则缓存修订不一致，请重新验证草稿后再发布");
        }
        const QString detail = QStringLiteral("几何干扰规则操作失败（%1）：%2").arg(code, actionable);
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setBusy(false);
            geometryMaskManagerDialog_->setPreviewBusy(false);
            geometryMaskManagerDialog_->setOperationError(detail);
        }
        geometryPublishAfterValidation_ = false;
        geometryPublishOverrideReason_.clear();
        if (geometryPollTimer_ != nullptr) geometryPollTimer_->stop();
        showLibraryMessage(detail, true);
    } else if (failedCommand == QStringLiteral("recycle_workpiece")
               && matchesOwner(CommandOwner::Library)) {
        workpieceLibraryPage_->handleBackendFailure(failedCommand, code, message);
    } else if (failedCommand == QStringLiteral("get_workpiece_annotations")
               && matchesOwner(CommandOwner::Geometry)) {
        showLibraryMessage(QStringLiteral("干扰标注加载失败：%1").arg(message), true);
    } else if (matchesOwner(CommandOwner::Geometry)
               && (failedCommand == QStringLiteral("save_workpiece_annotations")
                   || failedCommand == QStringLiteral("set_workpiece_annotation_group_enabled")
                   || failedCommand == QStringLiteral("delete_workpiece_annotation_group"))) {
        const QString workpieceId = annotationWorkpieceId_;
        const bool stale = code == QStringLiteral("STALE_WORKPIECE_REVISION");
        if (annotationManagerDialog_ != nullptr) {
            annotationManagerDialog_->setBusy(stale);
            if (!stale) {
                QString detail = message;
                if (code == QStringLiteral("MODEL_ERROR")) {
                    detail = QStringLiteral("局部特征递推失败，草稿未提交：%1").arg(message);
                } else if (code == QStringLiteral("INTERNAL_ERROR")) {
                    detail = QStringLiteral("后端递推异常，草稿未提交，请查看诊断日志：%1").arg(message);
                }
                annotationManagerDialog_->setOperationError(detail);
            }
        }
        if (stale) {
            showLibraryMessage(QStringLiteral("标注已在其他操作中更新，已重新加载"), true);
            QTimer::singleShot(0, this, [this, workpieceId]() { requestAnnotationSnapshot(workpieceId); });
        } else {
            showLibraryMessage(QStringLiteral("干扰标注保存失败：%1").arg(message), true);
        }
    } else {
        showLibraryMessage(message, true);
    }
    updateButtonStates();
}

void MainWindow::onClientTransportFailed(const QString &code, const QString &message) {
    Q_UNUSED(code)
    onBackendUnavailable(message);
}

void MainWindow::updateButtonStates() {
    const bool interactive = backendReady_ && !clientBusy_ && !batchInFlight_;
    inspectionPage_->setCurrentWorkpiece(appHeader_->currentWorkpieceId(),
                                         appHeader_->currentWorkpieceName());
    inspectionPage_->setBackendAvailable(backendReady_, clientBusy_,
                                         backendReady_ ? QString() : QStringLiteral("后端尚未就绪"));
    chooseBatchImagesButton_->setEnabled(interactive);
    batchPredictButton_->setEnabled(interactive && !selectedWorkpieceId().isEmpty()
                                    && !batchImagePaths_.isEmpty());
}

void MainWindow::updatePreview() {
    inspectionPage_->setMode(InspectionMode::Single);
    inspectionPage_->setSingleImagePath(inspectionImagePath_);
}

void MainWindow::clearInspectionState() {
    inspectionImagePath_.clear();
    updatePreview();
    resultLabel_->setText(QStringLiteral("尚未检测"));
    reviewLabel_->clear();
    evidenceTextEdit_->clear();
    clearPendingCommand();
    pendingPredictionWorkpieceId_.clear();
    pendingPredictionImagePath_.clear();
    lastPredictionWorkpieceId_.clear();
    lastPredictionImagePath_.clear();
    lastPredictionResponse_ = QJsonObject();
    updateButtonStates();
}

void MainWindow::clearBatchState() {
    batchImagePaths_.clear();
    batchInFlight_ = false;
    clearBatchResults();
    chooseBatchImagesButton_->setText(QStringLiteral("选择批量图片"));
    updateButtonStates();
}

void MainWindow::clearBatchResults() {
    const bool clearVisibleBatchResult = resultContext_ == ResultContext::Batch;
    pendingConfirmationRecordId_.clear();
    pendingConfirmationOrientation_.clear();
    pendingConfirmationMutation_ = QJsonObject();
    uncertainConfirmationMutations_.clear();
    if (inspectionPage_ != nullptr) {
        inspectionPage_->clearBatchState();
    }
    if (clearVisibleBatchResult) {
        resultContext_ = ResultContext::None;
        inspectionImagePath_.clear();
    }
    if (batchResultsTableWidget_ == nullptr) {
        return;
    }
    batchResultsTableWidget_->setRowCount(0);
    batchSummaryLabel_->clear();
}

bool MainWindow::validateImagePath(const QString &path, QString *error) const {
    if (path.isEmpty()) {
        if (error != nullptr) *error = QStringLiteral("请选择待测图片");
        return false;
    }
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix != QStringLiteral("png") && suffix != QStringLiteral("jpg")
        && suffix != QStringLiteral("jpeg") && suffix != QStringLiteral("bmp")) {
        if (error != nullptr) *error = QStringLiteral("图片格式不支持");
        return false;
    }
    QImageReader reader(path);
    if (!reader.canRead()) {
        if (error != nullptr) *error = QStringLiteral("图片无法读取");
        return false;
    }
    return true;
}

QStringList MainWindow::normalizedPaths(const QStringList &paths) const {
    QStringList normalized;
    for (const QString &path : paths) {
        const QString absolute = QFileInfo(path).absoluteFilePath();
        normalized.append(absolute.isEmpty() ? path : absolute);
    }
    return normalized;
}

QString MainWindow::selectedWorkpieceId() const {
    return appHeader_->currentWorkpieceId();
}

QString MainWindow::orientationText(const QString &label) const {
    if (label == QStringLiteral("front")) return QStringLiteral("正面");
    if (label == QStringLiteral("back")) return QStringLiteral("反面");
    if (label == QStringLiteral("uncertain")) return QStringLiteral("不确定");
    return label;
}

void MainWindow::showLibraryMessage(const QString &message, bool error) {
    QLabel *label = workpieceLibraryPage_ == nullptr ? nullptr
        : workpieceLibraryPage_->findChild<QLabel *>(QStringLiteral("libraryMessageLabel"));
    if (label == nullptr) return;
    label->setStyleSheet(error ? QStringLiteral("color: #b00020;") : QString());
    label->setText(message);
}
