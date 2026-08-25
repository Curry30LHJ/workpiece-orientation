#include "mainwindow.h"

#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QCryptographicHash>
#include <QBrush>
#include <QColor>
#include <QHeaderView>
#include <QAbstractItemView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonObject>
#include <QMessageBox>
#include <QPushButton>
#include <QSet>
#include <QTableWidget>
#include <QTableWidgetItem>
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

namespace {
const QStringList kImageFilters = {QStringLiteral("PNG/JPEG/BMP (*.png *.jpg *.jpeg *.bmp)")};

QStringList dialogFilters() {
    return kImageFilters;
}

QString imageContentKey(const QImage &image) {
    const QImage normalized = image.convertToFormat(QImage::Format_RGBA8888);
    const int width = normalized.width();
    const int height = normalized.height();
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(reinterpret_cast<const char *>(&width), sizeof(width));
    hash.addData(reinterpret_cast<const char *>(&height), sizeof(height));
    hash.addData(reinterpret_cast<const char *>(normalized.constBits()), normalized.sizeInBytes());
    return QString::fromLatin1(hash.result().toHex());
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

    deleteWorkpieceButton_ = new QPushButton(QStringLiteral("删除工件"), ui->libraryGroupBox);
    deleteWorkpieceButton_->setObjectName(QStringLiteral("deleteWorkpieceButton"));
    ui->libraryLayout->addWidget(deleteWorkpieceButton_, 0, 3);
    annotationEditorButton_ = new QPushButton(QStringLiteral("管理几何干扰规则"), ui->libraryGroupBox);
    annotationEditorButton_->setObjectName(QStringLiteral("annotationEditorButton"));
    ui->libraryLayout->addWidget(annotationEditorButton_, 8, 0, 1, 3);
    ui->registerButton->setEnabled(false);
    predictButton_->setEnabled(false);
    ui->refreshWorkpiecesButton->setEnabled(false);
    ui->chooseFrontTemplatesButton->setEnabled(false);
    ui->chooseBackTemplatesButton->setEnabled(false);
    chooseImageButton_->setEnabled(false);
    chooseBatchImagesButton_->setEnabled(false);
    batchPredictButton_->setEnabled(false);
    resultLabel_->setText(QStringLiteral("尚未检测"));
    reviewLabel_->clear();
    batchSummaryLabel_->clear();
    evidenceTextEdit_->clear();
    ui->libraryMessageLabel->clear();
    ui->templateWarningLabel->clear();
    ui->registrationProgressLabel->setText(QStringLiteral("建库进度：未开始"));
    ui->registrationProgressBar->setRange(0, 1);
    ui->registrationProgressBar->setValue(0);
    ui->registrationElapsedLabel->setText(QStringLiteral("耗时：0 ms"));
    registrationElapsedTimer_ = new QTimer(this);
    registrationElapsedTimer_->setInterval(100);
    connect(registrationElapsedTimer_, &QTimer::timeout,
            this, &MainWindow::updateRegistrationElapsed);
    evolutionPollTimer_ = new QTimer(this);
    evolutionPollTimer_->setInterval(5000);
    connect(evolutionPollTimer_, &QTimer::timeout, this, &MainWindow::pollEvolutionJobs);
    geometryPollTimer_ = new QTimer(this);
    geometryPollTimer_->setInterval(1000);
    connect(geometryPollTimer_, &QTimer::timeout, this, &MainWindow::pollGeometryValidation);
    batchResultsTableWidget_->setColumnCount(5);
    batchResultsTableWidget_->setHorizontalHeaderLabels({
        QStringLiteral("文件"), QStringLiteral("结果"), QStringLiteral("复检"),
        QStringLiteral("耗时（毫秒）"), QStringLiteral("处理状态")});
    batchResultsTableWidget_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    batchResultsTableWidget_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    batchResultsTableWidget_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    batchResultsTableWidget_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    batchResultsTableWidget_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    batchResultsTableWidget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    batchResultsTableWidget_->setSelectionBehavior(QAbstractItemView::SelectRows);
    batchResultsTableWidget_->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(batchResultsTableWidget_, &QTableWidget::currentCellChanged,
            this, [this](int row, int, int, int) {
                if (!changingBatchSelection_) {
                    selectBatchResult(row, true);
                }
            });
    updateTemplateLabels();
    clearBatchResults();
    updateButtonStates();
    connect(ui->chooseFrontTemplatesButton, &QPushButton::clicked,
            this, &MainWindow::chooseFrontTemplates);
    connect(ui->chooseBackTemplatesButton, &QPushButton::clicked,
            this, &MainWindow::chooseBackTemplates);
    connect(chooseImageButton_, &QPushButton::clicked,
            this, &MainWindow::chooseInspectionImage);
    connect(chooseBatchImagesButton_, &QPushButton::clicked,
            this, &MainWindow::chooseBatchImages);
    connect(ui->refreshWorkpiecesButton, &QPushButton::clicked,
            this, &MainWindow::refreshWorkpieces);
    connect(ui->registerButton, &QPushButton::clicked,
            this, &MainWindow::submitRegistration);
    connect(batchPredictButton_, &QPushButton::clicked,
            this, &MainWindow::submitBatchPrediction);
    connect(inspectionPage_, &InspectionPage::commandRequested,
            this, [this](const QString &command, const QJsonObject &fields) {
                sendPageCommand(CommandOwner::Inspection, command, fields);
            });
    connect(inspectionPage_, &InspectionPage::confirmationRequested,
            this, [this](const QString &, const QString &workpieceId,
                         const QString &imagePath, const QString &orientation) {
                submitTemplateConfirmation(workpieceId, imagePath, orientation);
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
        if (resultContext_ != ResultContext::Batch) {
            lastPredictionWorkpieceId_.clear();
            lastPredictionImagePath_.clear();
            lastPredictionResponse_ = QJsonObject();
        }
        updateButtonStates();
    });
    connect(deleteWorkpieceButton_, &QPushButton::clicked,
            this, &MainWindow::deleteSelectedWorkpiece);
    connect(confirmFrontButton_, &QPushButton::clicked, this, [this]() {
        if (inspectionPage_->mode() == InspectionMode::Batch) confirmFrontTemplate();
    });
    connect(confirmBackButton_, &QPushButton::clicked, this, [this]() {
        if (inspectionPage_->mode() == InspectionMode::Batch) confirmBackTemplate();
    });
    connect(rejectConfirmationButton_, &QPushButton::clicked, this, [this]() {
        if (inspectionPage_->mode() == InspectionMode::Batch) rejectTemplateConfirmation();
    });
    connect(annotationEditorButton_, &QPushButton::clicked,
            this, &MainWindow::openAnnotationManager);
    confirmFrontButton_->setEnabled(false);
    confirmBackButton_->setEnabled(false);
    rejectConfirmationButton_->setEnabled(false);
}

bool MainWindow::requestPage(AppPage page) {
    if (page == currentPage_) return true;
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
    pendingOwner_ = owner;
    pendingCommand_ = command;
    client_->sendRequest(command, fields);
}

void MainWindow::clearPendingCommand() {
    pendingOwner_ = CommandOwner::None;
    pendingCommand_.clear();
}

void MainWindow::setTemplatePaths(const QStringList &frontPaths, const QStringList &backPaths) {
    frontTemplatePaths_ = normalizedPaths(frontPaths);
    backTemplatePaths_ = normalizedPaths(backPaths);
    updateTemplateLabels();
    updateButtonStates();
}

void MainWindow::setWorkpieceName(const QString &name) {
    ui->workpieceNameEdit->setText(name);
    updateButtonStates();
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
    batchIndex_ = 0;
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
    replaceConfirmationHandler_ = std::move(handler);
}

void MainWindow::setBackendError(const QString &message) {
    backendReady_ = false;
    clientBusy_ = false;
    BackendStatusDetails details;
    details.state = BackendUiState::Error;
    details.connectionDetail = QStringLiteral("配置错误");
    details.modelDetail = QStringLiteral("未加载");
    details.recentError = message;
    details.canRestart = false;
    appHeader_->setBackendDetails(details);
    showLibraryMessage(message, true);
    updateButtonStates();
}

void MainWindow::chooseFrontTemplates() {
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, QStringLiteral("选择正面模板"), QString(), dialogFilters().constFirst());
    if (!paths.isEmpty()) {
        frontTemplatePaths_ = normalizedPaths(paths);
        updateTemplateLabels();
        updateButtonStates();
    }
}

void MainWindow::chooseBackTemplates() {
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, QStringLiteral("选择反面模板"), QString(), dialogFilters().constFirst());
    if (!paths.isEmpty()) {
        backTemplatePaths_ = normalizedPaths(paths);
        updateTemplateLabels();
        updateButtonStates();
    }
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
    requestWorkpieceRefresh(false);
}

void MainWindow::requestWorkpieceRefresh(bool preserveRegistrationSummary) {
    if (client_ == nullptr || !backendReady_ || clientBusy_
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    if (!preserveRegistrationSummary) {
        registrationSummaryVisible_ = false;
    }
    sendPageCommand(CommandOwner::System, QStringLiteral("list_workpieces"), QJsonObject());
    if (!preserveRegistrationSummary) {
        showLibraryMessage(QStringLiteral("正在刷新工件列表…"));
    }
}

void MainWindow::submitRegistration() {
    QString error;
    if (!validateRegistration(&error)) {
        showLibraryMessage(error, true);
        return;
    }
    if (client_ == nullptr || !backendReady_ || clientBusy_
        || client_->state() != BackendClient::State::Ready) {
        showLibraryMessage(QStringLiteral("后端尚未就绪"), true);
        return;
    }
    pendingWorkpieceName_ = ui->workpieceNameEdit->text().trimmed();
    pendingReplace_ = false;
    registrationInFlight_ = true;
    registrationSummaryVisible_ = false;
    startRegistrationProgress();
    sendRegistration(false);
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
    batchWorkpieceId_ = workpieceId;
    batchIndex_ = 0;
    clearBatchResults();
    batchSelectionPinned_ = false;
    batchCompletedSuccessfully_ = false;
    inspectionPage_->setMode(InspectionMode::Batch);
    resultLabel_->setText(QStringLiteral("批量检测进行中…"));
    reviewLabel_->clear();
    evidenceTextEdit_->clear();
    sendNextBatchPrediction();
}

void MainWindow::deleteSelectedWorkpiece() {
    const QString workpieceId = selectedWorkpieceId();
    if (workpieceId.isEmpty() || client_ == nullptr || clientBusy_ || !backendReady_) {
        return;
    }
    const QString name = appHeader_->currentWorkpieceName();
    const bool confirmed = QMessageBox::question(
        this, QStringLiteral("确认删除工件"),
        QStringLiteral("工件“%1”将移入可恢复回收区，是否继续？").arg(name),
        QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes;
    if (!confirmed) {
        return;
    }
    sendPageCommand(CommandOwner::Library, QStringLiteral("recycle_workpiece"), {
        {QStringLiteral("workpiece_id"), workpieceId},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
    showLibraryMessage(QStringLiteral("正在将工件移入回收区…"));
}

void MainWindow::confirmFrontTemplate() {
    submitCurrentConfirmation(QStringLiteral("front"));
}

void MainWindow::confirmBackTemplate() {
    submitCurrentConfirmation(QStringLiteral("back"));
}

void MainWindow::rejectTemplateConfirmation() {
    if (resultContext_ == ResultContext::Batch) {
        if (!currentBatchResultCanBeProcessed()) {
            return;
        }
        const int rejectedIndex = selectedBatchResultIndex_;
        BatchResult &result = batchResults_[rejectedIndex];
        result.state = BatchResultState::Rejected;
        result.submitError.clear();
        updateBatchRow(rejectedIndex);
        updateBatchSummary();
        const int nextIndex = preferredPendingBatchResult(rejectedIndex);
        if (nextIndex >= 0) {
            selectBatchResult(nextIndex, false);
        } else {
            currentResultTargetLabel_->setText(
                QStringLiteral("当前：%1（不入库）").arg(QFileInfo(result.imagePath).fileName()));
        }
        reviewLabel_->setText(QStringLiteral("本次结果不入库"));
        updateButtonStates();
        return;
    }
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
    const QString workpieceId = selectedWorkpieceId();
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

void MainWindow::submitCurrentConfirmation(const QString &orientation) {
    if (resultContext_ == ResultContext::Batch) {
        if (!currentBatchResultCanBeProcessed()) {
            return;
        }
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

void MainWindow::submitTemplateConfirmation(const QString &workpieceId, const QString &imagePath,
                                            const QString &orientation) {
    if (imagePath.isEmpty() || workpieceId.isEmpty()
        || client_ == nullptr || clientBusy_ || !backendReady_) {
        return;
    }
    sendPageCommand(CommandOwner::Inspection, QStringLiteral("submit_confirmation"), {
        {QStringLiteral("workpiece_id"), workpieceId},
        {QStringLiteral("orientation"), orientation},
        {QStringLiteral("image_path"), imagePath},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
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
    updateButtonStates();
    if (evolutionPollTimer_ != nullptr) {
        evolutionPollTimer_->start();
    }
    QTimer::singleShot(0, this, [this]() {
        requestWorkpieceRefresh(registrationSummaryVisible_);
    });
}

void MainWindow::onBackendLoading(const QString &message) {
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
    showLibraryMessage(message.isEmpty() ? QStringLiteral("正在加载模型，请稍候…") : message);
    updateButtonStates();
}

void MainWindow::onBackendUnavailable(const QString &reason) {
    const CommandOwner interruptedOwner = pendingOwner_;
    const QString interruptedTask = pendingCommand_;
    const bool interruptedBatchCommand = batchInFlight_
        || pendingConfirmationBatchIndex_ >= 0;
    const bool hasPreservedWork = registrationInFlight_ || batchInFlight_ || clientBusy_
        || !pendingCommand_.isEmpty() || !inspectionImagePath_.isEmpty()
        || (batchResultsTableWidget_ != nullptr && batchResultsTableWidget_->rowCount() > 0)
        || annotationManagerDialog_ != nullptr || geometryMaskManagerDialog_ != nullptr;
    stopRegistrationProgress();
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
    if (pendingConfirmationBatchIndex_ >= 0
        && pendingConfirmationBatchIndex_ < batchResults_.size()) {
        const int failedIndex = pendingConfirmationBatchIndex_;
        BatchResult &result = batchResults_[failedIndex];
        result.state = BatchResultState::SubmitFailed;
        result.submitError = QStringLiteral("连接中断，入库结果未知；请刷新后确认");
        pendingConfirmationBatchIndex_ = -1;
        pendingConfirmationOrientation_.clear();
        updateBatchRow(failedIndex);
        selectBatchResult(failedIndex, false);
    }
    backendReady_ = false;
    clientBusy_ = false;
    registrationInFlight_ = false;
    batchInFlight_ = false;
    pendingReplace_ = false;
    clearPendingCommand();
    if (interruptedOwner == CommandOwner::Inspection
        && !interruptedBatchCommand && inspectionPage_ != nullptr) {
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
    if (!backendReady_ || clientBusy_ || !pendingCommand_.isEmpty()
        || client_ == nullptr || client_->state() != BackendClient::State::Ready) {
        return;
    }
    sendPageCommand(CommandOwner::System, QStringLiteral("list_evolution_jobs"), QJsonObject());
}

void MainWindow::onClientStateChanged(BackendClient::State state, const QString &detail) {
    clientBusy_ = state == BackendClient::State::Busy;
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
    updateButtonStates();
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
    if (command != QStringLiteral("register") || !registrationInFlight_) {
        return;
    }
    const QString phase = progress.value(QStringLiteral("phase")).toString();
    const int total = progress.value(QStringLiteral("total")).toInt(1);
    const int completed = progress.value(QStringLiteral("completed")).toInt(0);
    ui->registrationProgressBar->setRange(0, qMax(1, total));
    ui->registrationProgressBar->setValue(qBound(0, completed, qMax(1, total)));
    if (phase == QStringLiteral("validating")) {
        ui->registrationProgressLabel->setText(QStringLiteral("建库进度：正在校验图片"));
    } else if (phase == QStringLiteral("copying")) {
        ui->registrationProgressLabel->setText(QStringLiteral("建库进度：正在写入模板"));
    } else if (phase == QStringLiteral("features")) {
        ui->registrationProgressLabel->setText(QStringLiteral("建库进度：正在提取特征（%1/%2）")
                                                    .arg(completed).arg(total));
    } else if (phase == QStringLiteral("committing")) {
        ui->registrationProgressLabel->setText(QStringLiteral("建库进度：正在提交"));
    }
    updateRegistrationElapsed();
}

void MainWindow::onClientResponse(const QString &command, const QJsonObject &response) {
    const CommandOwner responseOwner = pendingOwner_;
    const QString issuedCommand = pendingCommand_;
    clearPendingCommand();
    if (batchInFlight_ && command == QStringLiteral("predict")) {
        appendBatchResult(response);
        ++batchIndex_;
        if (batchIndex_ >= batchImagePaths_.size()) {
            finishBatchPrediction();
        } else {
            QTimer::singleShot(0, this, [this]() { sendNextBatchPrediction(); });
        }
        return;
    }
    if (command == QStringLiteral("list_workpieces")) {
        const QString previousId = selectedWorkpieceId();
        const QJsonArray workpieces = response.value(QStringLiteral("workpieces")).toArray();
        QList<QPair<QString, QString>> items;
        QString requestedId = previousId;
        bool previousStillExists = false;
        for (const QJsonValue &value : workpieces) {
            const QJsonObject item = value.toObject();
            const QString id = item.value(QStringLiteral("id")).toString();
            const QString name = item.value(QStringLiteral("name")).toString();
            items.append(qMakePair(id, name));
            previousStillExists = previousStillExists || id == previousId;
            if (!pendingWorkpieceName_.isEmpty() && name == pendingWorkpieceName_) {
                requestedId = id;
            }
        }
        if (!previousStillExists && pendingWorkpieceName_.isEmpty()) {
            requestedId.clear();
        }
        appHeader_->setWorkpieces(items, requestedId);
        if (issuedCommand == QStringLiteral("list_workpieces")
            || command == QStringLiteral("list_workpieces")) {
            pendingWorkpieceName_.clear();
        }
        if (!registrationSummaryVisible_) {
            showLibraryMessage(QStringLiteral("工件列表已刷新"));
        }
        updateButtonStates();
        return;
    }
    if (command == QStringLiteral("recycle_workpiece")) {
        lastPredictionWorkpieceId_.clear();
        lastPredictionImagePath_.clear();
        lastPredictionResponse_ = QJsonObject();
        showLibraryMessage(QStringLiteral("工件已移入回收区，可在后端恢复"));
        requestWorkpieceRefresh(false);
        return;
    }
    if (command == QStringLiteral("submit_confirmation")) {
        if (pendingConfirmationBatchIndex_ >= 0
            && pendingConfirmationBatchIndex_ < batchResults_.size()) {
            const int completedIndex = pendingConfirmationBatchIndex_;
            BatchResult &result = batchResults_[completedIndex];
            result.state = pendingConfirmationOrientation_ == QStringLiteral("front")
                ? BatchResultState::QueuedFront : BatchResultState::QueuedBack;
            result.submitError.clear();
            pendingConfirmationBatchIndex_ = -1;
            pendingConfirmationOrientation_.clear();
            updateBatchRow(completedIndex);
            updateBatchSummary();
            const int nextIndex = preferredPendingBatchResult(completedIndex);
            if (nextIndex >= 0) {
                selectBatchResult(nextIndex, false);
            } else {
                selectBatchResult(completedIndex, false);
            }
        } else if (inspectionPage_ != nullptr
                   && (responseOwner == CommandOwner::Inspection
                       || responseOwner == CommandOwner::None)) {
            inspectionPage_->handleBackendResponse(command, response);
        }
        showLibraryMessage(QStringLiteral("确认图片已进入后台入库队列"));
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
        if (!jobs.isEmpty()) {
            const QJsonObject latest = jobs.last().toObject();
            showLibraryMessage(QStringLiteral("后台入库：%1，进度 %2%%")
                                   .arg(latest.value(QStringLiteral("state")).toString())
                                   .arg(latest.value(QStringLiteral("progress")).toInt()));
        }
        return;
    }
    if (command == QStringLiteral("register")) {
        stopRegistrationProgress();
        registrationInFlight_ = false;
        const QJsonObject counts = response.value(QStringLiteral("template_counts")).toObject();
        const int frontCount = counts.value(QStringLiteral("front")).toInt(frontTemplatePaths_.size());
        const int backCount = counts.value(QStringLiteral("back")).toInt(backTemplatePaths_.size());
        const double elapsedMs = response.value(QStringLiteral("elapsed_ms")).toDouble(
            registrationElapsedClock_.isValid() ? registrationElapsedClock_.elapsed() : 0.0);
        showLibraryMessage(QStringLiteral("工件库建立成功：正面 %1 张，反面 %2 张，耗时 %3 ms")
                              .arg(frontCount).arg(backCount).arg(QString::number(elapsedMs, 'f', 1)));
        registrationSummaryVisible_ = true;
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
    clearPendingCommand();
    if (command == QStringLiteral("register") && registrationInFlight_
        && code == QStringLiteral("WORKPIECE_EXISTS")
        && !pendingReplace_) {
        const bool confirmed = replaceConfirmationHandler_
            ? replaceConfirmationHandler_(pendingWorkpieceName_)
            : QMessageBox::question(this, QStringLiteral("确认覆盖"),
                                     QStringLiteral("工件“%1”已存在，是否覆盖？").arg(pendingWorkpieceName_),
                                     QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes;
        if (confirmed) {
            pendingReplace_ = true;
            QTimer::singleShot(0, this, [this]() { sendRegistration(true); });
        } else {
            stopRegistrationProgress();
            registrationInFlight_ = false;
            showLibraryMessage(QStringLiteral("已取消覆盖"));
        }
        return;
    }
    if (batchInFlight_ && command == QStringLiteral("predict")) {
        batchInFlight_ = false;
        batchCompletedSuccessfully_ = false;
        const QString failedFile = batchIndex_ >= 0 && batchIndex_ < batchImagePaths_.size()
            ? QFileInfo(batchImagePaths_.at(batchIndex_)).fileName() : QStringLiteral("未知");
        batchSummaryLabel_->setText(
            QStringLiteral("批量检测未完成：已完成 %1/%2，失败文件：%3，原因：%4")
                .arg(batchResults_.size())
                .arg(batchImagePaths_.size())
                .arg(failedFile, message));
        resultLabel_->setText(QStringLiteral("批量检测未完成"));
        updateButtonStates();
        return;
    }
    if (command == QStringLiteral("register")) {
        stopRegistrationProgress();
        registrationInFlight_ = false;
        showLibraryMessage(message, true);
    } else if (command == QStringLiteral("predict")) {
        if (inspectionPage_ != nullptr
            && (failureOwner == CommandOwner::Inspection
                || failureOwner == CommandOwner::None)) {
            inspectionPage_->handleBackendFailure(command, code, message);
        }
    } else if (command == QStringLiteral("submit_confirmation")) {
        if (pendingConfirmationBatchIndex_ >= 0
            && pendingConfirmationBatchIndex_ < batchResults_.size()) {
            const int failedIndex = pendingConfirmationBatchIndex_;
            BatchResult &result = batchResults_[failedIndex];
            result.state = BatchResultState::SubmitFailed;
            result.submitError = message;
            pendingConfirmationBatchIndex_ = -1;
            pendingConfirmationOrientation_.clear();
            updateBatchRow(failedIndex);
            selectBatchResult(failedIndex, false);
            updateBatchSummary();
        } else if (inspectionPage_ != nullptr
                   && (failureOwner == CommandOwner::Inspection
                       || failureOwner == CommandOwner::None)) {
            inspectionPage_->handleBackendFailure(command, code, message);
        }
        showLibraryMessage(QStringLiteral("确认入库失败：%1").arg(message), true);
    } else if (command.startsWith(QStringLiteral("get_geometry_mask"))
               || command.startsWith(QStringLiteral("preview_geometry_mask"))
               || command.startsWith(QStringLiteral("save_geometry_mask"))
               || command.startsWith(QStringLiteral("validate_geometry_mask"))
               || command.startsWith(QStringLiteral("publish_geometry_mask"))
               || command.startsWith(QStringLiteral("resolve_geometry_mask"))
               || command.startsWith(QStringLiteral("rollback_geometry_mask"))) {
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
    } else if (command == QStringLiteral("recycle_workpiece")) {
        showLibraryMessage(QStringLiteral("删除失败：%1").arg(message), true);
    } else if (command == QStringLiteral("get_workpiece_annotations")) {
        showLibraryMessage(QStringLiteral("干扰标注加载失败：%1").arg(message), true);
    } else if (command == QStringLiteral("save_workpiece_annotations")
               || command == QStringLiteral("set_workpiece_annotation_group_enabled")
               || command == QStringLiteral("delete_workpiece_annotation_group")) {
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
    const bool interactive = backendReady_ && !clientBusy_ && !batchInFlight_ && !registrationInFlight_;
    inspectionPage_->setCurrentWorkpiece(appHeader_->currentWorkpieceId(),
                                         appHeader_->currentWorkpieceName());
    inspectionPage_->setBackendAvailable(backendReady_, !interactive,
                                         backendReady_ ? QString() : QStringLiteral("后端尚未就绪"));
    ui->refreshWorkpiecesButton->setEnabled(interactive);
    ui->chooseFrontTemplatesButton->setEnabled(interactive);
    ui->chooseBackTemplatesButton->setEnabled(interactive);
    chooseBatchImagesButton_->setEnabled(interactive);
    ui->registerButton->setEnabled(interactive && !ui->workpieceNameEdit->text().trimmed().isEmpty()
                                   && !frontTemplatePaths_.isEmpty() && !backTemplatePaths_.isEmpty());
    batchPredictButton_->setEnabled(interactive && !selectedWorkpieceId().isEmpty()
                                    && !batchImagePaths_.isEmpty());
    if (deleteWorkpieceButton_ != nullptr) {
        deleteWorkpieceButton_->setEnabled(interactive && !selectedWorkpieceId().isEmpty());
    }
    if (annotationEditorButton_ != nullptr) {
        annotationEditorButton_->setEnabled(interactive && !selectedWorkpieceId().isEmpty());
    }
    const bool confirmationAvailable = currentBatchResultCanBeProcessed();
    if (confirmFrontButton_ != nullptr
        && inspectionPage_->mode() == InspectionMode::Batch) {
        confirmFrontButton_->setEnabled(confirmationAvailable);
        confirmBackButton_->setEnabled(confirmationAvailable);
        rejectConfirmationButton_->setEnabled(confirmationAvailable);
    }
    if (resultContext_ == ResultContext::Batch
        && selectedBatchResultIndex_ >= 0 && selectedBatchResultIndex_ < batchResults_.size()) {
        const BatchResult &result = batchResults_.at(selectedBatchResultIndex_);
        const QString fileName = QFileInfo(result.imagePath).fileName();
        QString detail;
        if (batchInFlight_) {
            detail = QStringLiteral("批量检测完成后可处理");
        } else if (!batchCompletedSuccessfully_) {
            detail = QStringLiteral("批量检测未完整完成，结果仅供查看");
        } else if (selectedWorkpieceId() != result.workpieceId) {
            detail = QStringLiteral("结果来自其他工件，请切回原工件后处理");
        } else if (!backendReady_ || client_ == nullptr
                   || client_->state() != BackendClient::State::Ready) {
            detail = QStringLiteral("后端不可用，结果已保留");
        } else {
            detail = batchResultStateText(result.state);
            if (result.needsReview
                && (result.state == BatchResultState::Pending
                    || result.state == BatchResultState::SubmitFailed)) {
                detail = QStringLiteral("建议复检，%1").arg(detail);
            }
        }
        currentResultTargetLabel_->setText(
            QStringLiteral("当前：%1（%2）").arg(fileName, detail));
    }
}

void MainWindow::updateTemplateLabels() {
    ui->frontTemplatesLabel->setText(QStringLiteral("正面已选择 %1 张").arg(frontTemplatePaths_.size()));
    ui->backTemplatesLabel->setText(QStringLiteral("反面已选择 %1 张").arg(backTemplatePaths_.size()));
    ui->frontTemplatesFilesLabel->setText(frontTemplatePaths_.join(QLatin1Char('\n')));
    ui->backTemplatesFilesLabel->setText(backTemplatePaths_.join(QLatin1Char('\n')));
    QStringList warnings;
    if (frontTemplatePaths_.size() < 3 || backTemplatePaths_.size() < 3) {
        warnings.append(QStringLiteral("模板较少，建议补充更多角度，但仍可建库"));
    }
    if (frontTemplatePaths_.size() + backTemplatePaths_.size() > 30) {
        warnings.append(QStringLiteral("模板较多，建库和检测耗时可能增加"));
    }
    ui->templateWarningLabel->setText(warnings.join(QStringLiteral("\n")));
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
    batchIndex_ = 0;
    batchWorkpieceId_.clear();
    clearBatchResults();
    chooseBatchImagesButton_->setText(QStringLiteral("选择批量图片"));
    updateButtonStates();
}

void MainWindow::clearBatchResults() {
    const bool clearVisibleBatchResult = resultContext_ == ResultContext::Batch;
    batchFrontCount_ = 0;
    batchBackCount_ = 0;
    batchUncertainCount_ = 0;
    batchReviewCount_ = 0;
    batchResults_.clear();
    selectedBatchResultIndex_ = -1;
    pendingConfirmationBatchIndex_ = -1;
    pendingConfirmationOrientation_.clear();
    changingBatchSelection_ = false;
    batchSelectionPinned_ = false;
    batchCompletedSuccessfully_ = false;
    if (clearVisibleBatchResult) {
        resultContext_ = ResultContext::None;
        inspectionImagePath_.clear();
        updatePreview();
        inspectionPage_->setMode(InspectionMode::Batch);
        inspectionPage_->setSingleImagePath(QString());
        currentImageLabel_->clear();
        currentResultTargetLabel_->clear();
        resultLabel_->setText(QStringLiteral("尚未检测"));
        reviewLabel_->clear();
        evidenceTextEdit_->clear();
    }
    if (batchResultsTableWidget_ == nullptr) {
        return;
    }
    batchResultsTableWidget_->setRowCount(0);
    batchSummaryLabel_->clear();
}

void MainWindow::startRegistrationProgress() {
    registrationElapsedClock_.start();
    registrationElapsedTimer_->start();
    const int total = frontTemplatePaths_.size() + backTemplatePaths_.size();
    ui->registrationProgressBar->setRange(0, qMax(1, total));
    ui->registrationProgressBar->setValue(0);
    ui->registrationProgressLabel->setText(QStringLiteral("建库进度：准备中（共 %1 张）").arg(total));
    ui->registrationElapsedLabel->setText(QStringLiteral("耗时：0 ms"));
}

void MainWindow::stopRegistrationProgress() {
    if (registrationElapsedTimer_ != nullptr) {
        registrationElapsedTimer_->stop();
    }
    if (registrationElapsedClock_.isValid()) {
        ui->registrationElapsedLabel->setText(QStringLiteral("耗时：%1 ms")
                                                   .arg(registrationElapsedClock_.elapsed()));
    }
}

void MainWindow::updateRegistrationElapsed() {
    if (registrationInFlight_ && registrationElapsedClock_.isValid()) {
        ui->registrationElapsedLabel->setText(QStringLiteral("耗时：%1 ms")
                                                   .arg(registrationElapsedClock_.elapsed()));
    }
}

void MainWindow::sendRegistration(bool replace) {
    if (client_ == nullptr || client_->state() != BackendClient::State::Ready) {
        return;
    }
    QJsonArray front;
    QJsonArray back;
    for (const QString &path : frontTemplatePaths_) {
        front.append(path);
    }
    for (const QString &path : backTemplatePaths_) {
        back.append(path);
    }
    sendPageCommand(CommandOwner::Library, QStringLiteral("register"), {
        {QStringLiteral("name"), pendingWorkpieceName_},
        {QStringLiteral("replace"), replace},
        {QStringLiteral("front_images"), front},
        {QStringLiteral("back_images"), back},
        {QStringLiteral("progress_events"), true},
    });
    showLibraryMessage(replace ? QStringLiteral("正在覆盖并建立工件库…") : QStringLiteral("正在建立工件库…"));
}

void MainWindow::sendNextBatchPrediction() {
    if (!batchInFlight_ || client_ == nullptr || client_->state() != BackendClient::State::Ready) {
        return;
    }
    if (batchIndex_ >= batchImagePaths_.size()) {
        finishBatchPrediction();
        return;
    }
    pendingPredictionWorkpieceId_ = batchWorkpieceId_;
    pendingPredictionImagePath_ = batchImagePaths_.at(batchIndex_);
    sendPageCommand(CommandOwner::Inspection, QStringLiteral("predict"), {
        {QStringLiteral("workpiece_id"), pendingPredictionWorkpieceId_},
        {QStringLiteral("image_path"), batchImagePaths_.at(batchIndex_)},
    });
    batchSummaryLabel_->setText(
        QStringLiteral("已完成 %1/%2，当前文件：%3")
            .arg(batchResults_.size())
            .arg(batchImagePaths_.size())
            .arg(QFileInfo(batchImagePaths_.at(batchIndex_)).fileName()));
}

void MainWindow::renderPredictionResult(const QString &imagePath, const QJsonObject &response,
                                        const QString &sourceText) {
    InspectionRecord record;
    record.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    record.imagePath = QFileInfo(imagePath).absoluteFilePath();
    record.workpieceId = sourceText == QStringLiteral("批量结果")
        ? batchWorkpieceId_ : selectedWorkpieceId();
    record.response = response;
    record.label = response.value(QStringLiteral("label")).toString();
    record.needsReview = response.value(QStringLiteral("needs_review")).toBool();
    record.elapsedMs = response.value(QStringLiteral("elapsed_ms")).toDouble();
    record.completedAt = QDateTime::currentDateTime();
    inspectionImagePath_ = record.imagePath;
    inspectionPage_->setMode(sourceText == QStringLiteral("批量结果")
                                 ? InspectionMode::Batch : InspectionMode::Single);
    inspectionPage_->showSingleResult(record);
    updateButtonStates();
}

void MainWindow::appendBatchResult(const QJsonObject &response) {
    BatchResult result;
    result.imagePath = batchImagePaths_.at(batchIndex_);
    result.workpieceId = batchWorkpieceId_;
    result.response = response;
    result.label = response.value(QStringLiteral("label")).toString();
    result.needsReview = response.value(QStringLiteral("needs_review")).toBool();
    result.elapsedMs = response.value(QStringLiteral("elapsed_ms")).toDouble();
    batchResults_.append(result);

    const int row = batchResults_.size() - 1;
    batchResultsTableWidget_->insertRow(row);
    updateBatchRow(row);
    if (selectedBatchResultIndex_ < 0) {
        selectBatchResult(row, false);
    }

    if (result.label == QStringLiteral("front")) {
        ++batchFrontCount_;
    } else if (result.label == QStringLiteral("back")) {
        ++batchBackCount_;
    } else {
        ++batchUncertainCount_;
    }
    if (result.needsReview) {
        ++batchReviewCount_;
    }
    updateBatchSummary();
}

void MainWindow::updateBatchRow(int index) {
    if (index < 0 || index >= batchResults_.size()) {
        return;
    }
    const BatchResult &result = batchResults_.at(index);
    auto *fileItem = new QTableWidgetItem(QFileInfo(result.imagePath).fileName());
    fileItem->setToolTip(result.imagePath);
    batchResultsTableWidget_->setItem(index, 0, fileItem);
    batchResultsTableWidget_->setItem(
        index, 1, new QTableWidgetItem(orientationText(result.label)));
    batchResultsTableWidget_->setItem(
        index, 2, new QTableWidgetItem(result.needsReview ? QStringLiteral("是")
                                                         : QStringLiteral("否")));
    batchResultsTableWidget_->setItem(
        index, 3, new QTableWidgetItem(QString::number(result.elapsedMs, 'f', 1)));
    batchResultsTableWidget_->setItem(
        index, 4, new QTableWidgetItem(batchResultStateText(result.state)));
    if (result.needsReview) {
        const QBrush warningBrush(QColor(255, 244, 204));
        for (int column = 0; column < batchResultsTableWidget_->columnCount(); ++column) {
            batchResultsTableWidget_->item(index, column)->setBackground(warningBrush);
        }
    }
}

QString MainWindow::batchResultStateText(BatchResultState state) const {
    switch (state) {
    case BatchResultState::Pending:
        return QStringLiteral("待处理");
    case BatchResultState::Submitting:
        return QStringLiteral("提交中");
    case BatchResultState::QueuedFront:
        return QStringLiteral("正面已排队");
    case BatchResultState::QueuedBack:
        return QStringLiteral("反面已排队");
    case BatchResultState::Rejected:
        return QStringLiteral("不入库");
    case BatchResultState::SubmitFailed:
        return QStringLiteral("提交失败");
    }
    return QString();
}

bool MainWindow::currentBatchResultCanBeProcessed() const {
    if (!batchCompletedSuccessfully_ || batchInFlight_ || !backendReady_ || clientBusy_
        || registrationInFlight_ || client_ == nullptr
        || client_->state() != BackendClient::State::Ready
        || selectedBatchResultIndex_ < 0 || selectedBatchResultIndex_ >= batchResults_.size()) {
        return false;
    }
    const BatchResult &result = batchResults_.at(selectedBatchResultIndex_);
    if (selectedWorkpieceId() != result.workpieceId) {
        return false;
    }
    return result.state == BatchResultState::Pending
        || result.state == BatchResultState::SubmitFailed;
}

void MainWindow::selectBatchResult(int index, bool userInitiated) {
    if (index < 0 || index >= batchResults_.size()) {
        return;
    }
    selectedBatchResultIndex_ = index;
    resultContext_ = ResultContext::Batch;
    if (userInitiated) {
        batchSelectionPinned_ = true;
    }
    changingBatchSelection_ = true;
    batchResultsTableWidget_->setCurrentCell(index, 0);
    changingBatchSelection_ = false;
    const BatchResult &result = batchResults_.at(index);
    renderPredictionResult(result.imagePath, result.response, QStringLiteral("批量结果"));
}

int MainWindow::preferredPendingBatchResult(int afterIndex) const {
    if (batchResults_.isEmpty()) {
        return -1;
    }
    const int count = batchResults_.size();
    const int start = (afterIndex >= 0 && afterIndex < count) ? (afterIndex + 1) % count : 0;
    const auto isPending = [](const BatchResult &result) {
        return result.state == BatchResultState::Pending
            || result.state == BatchResultState::SubmitFailed;
    };
    for (int reviewPass = 0; reviewPass < 2; ++reviewPass) {
        for (int offset = 0; offset < count; ++offset) {
            const int index = (start + offset) % count;
            const BatchResult &result = batchResults_.at(index);
            if (isPending(result) && (!reviewPass ? result.needsReview : true)) {
                return index;
            }
        }
    }
    return -1;
}

void MainWindow::updateBatchSummary() {
    const int completed = batchResults_.size();
    const int total = batchImagePaths_.size();
    if (batchInFlight_) {
        batchSummaryLabel_->setText(
            QStringLiteral("已完成 %1/%2：正面 %3，反面 %4，不确定 %5，建议复检 %6")
                .arg(completed)
                .arg(total)
                .arg(batchFrontCount_)
                .arg(batchBackCount_)
                .arg(batchUncertainCount_)
                .arg(batchReviewCount_));
        return;
    }
    int processed = 0;
    for (const BatchResult &result : batchResults_) {
        if (result.state == BatchResultState::QueuedFront
            || result.state == BatchResultState::QueuedBack
            || result.state == BatchResultState::Rejected) {
            ++processed;
        }
    }
    const int unprocessed = qMax(0, completed - processed);
    batchSummaryLabel_->setText(
        QStringLiteral("共 %1 张：正面 %2，反面 %3，不确定 %4，建议复检 %5，已处理 %6，未处理 %7")
            .arg(total)
            .arg(batchFrontCount_)
            .arg(batchBackCount_)
            .arg(batchUncertainCount_)
            .arg(batchReviewCount_)
            .arg(processed)
            .arg(unprocessed));
}

void MainWindow::finishBatchPrediction() {
    batchInFlight_ = false;
    batchCompletedSuccessfully_ = true;
    batchSelectionPinned_ = false;
    clearPendingCommand();
    resultLabel_->setText(QStringLiteral("批量检测完成"));
    updateBatchSummary();
    const int preferredIndex = preferredPendingBatchResult();
    if (preferredIndex >= 0) {
        selectBatchResult(preferredIndex, false);
    }
    updateButtonStates();
}

bool MainWindow::validateRegistration(QString *error) const {
    if (ui->workpieceNameEdit->text().trimmed().isEmpty()) {
        if (error != nullptr) *error = QStringLiteral("请输入工件名称");
        return false;
    }
    if (frontTemplatePaths_.isEmpty() || backTemplatePaths_.isEmpty()) {
        if (error != nullptr) *error = QStringLiteral("正面和反面都至少需要选择 1 张图片");
        return false;
    }
    QSet<QString> paths;
    QSet<QString> contents;
    for (const QString &path : frontTemplatePaths_ + backTemplatePaths_) {
        const QString absolute = QFileInfo(path).absoluteFilePath();
        const QString pathKey = absolute.toCaseFolded();
        if (paths.contains(pathKey)) {
            if (error != nullptr) *error = QStringLiteral("模板图片不能重复");
            return false;
        }
        paths.insert(pathKey);
        QString imageError;
        if (!validateImagePath(absolute, &imageError)) {
            if (error != nullptr) *error = QStringLiteral("模板图片无效：%1").arg(absolute);
            return false;
        }
        QImageReader reader(absolute);
        const QImage image = reader.read();
        const QString contentKey = imageContentKey(image);
        if (contents.contains(contentKey)) {
            if (error != nullptr) *error = QStringLiteral("模板图片内容不能重复");
            return false;
        }
        contents.insert(contentKey);
    }
    return true;
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
    ui->libraryMessageLabel->setStyleSheet(error ? QStringLiteral("color: #b00020;") : QString());
    ui->libraryMessageLabel->setText(message);
}
