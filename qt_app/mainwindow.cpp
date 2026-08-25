#include "mainwindow.h"

#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QCryptographicHash>
#include <QBrush>
#include <QColor>
#include <QComboBox>
#include <QHeaderView>
#include <QAbstractItemView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonObject>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QSet>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTimer>
#include <QUuid>

#include "backendclient.h"
#include "backendprocessmanager.h"
#include "annotationeditor.h"
#include "annotationmanager.h"
#include "geometrymaskmanager.h"

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
    deleteWorkpieceButton_ = new QPushButton(QStringLiteral("删除工件"), ui->libraryGroupBox);
    deleteWorkpieceButton_->setObjectName(QStringLiteral("deleteWorkpieceButton"));
    ui->libraryLayout->addWidget(deleteWorkpieceButton_, 0, 3);
    confirmFrontButton_ = new QPushButton(QStringLiteral("确认正面并入库"), ui->resultGroupBox);
    confirmFrontButton_->setObjectName(QStringLiteral("confirmFrontButton"));
    confirmBackButton_ = new QPushButton(QStringLiteral("确认反面并入库"), ui->resultGroupBox);
    confirmBackButton_->setObjectName(QStringLiteral("confirmBackButton"));
    rejectConfirmationButton_ = new QPushButton(QStringLiteral("不入库"), ui->resultGroupBox);
    rejectConfirmationButton_->setObjectName(QStringLiteral("rejectConfirmationButton"));
    annotationEditorButton_ = new QPushButton(QStringLiteral("管理几何干扰规则"), ui->libraryGroupBox);
    annotationEditorButton_->setObjectName(QStringLiteral("annotationEditorButton"));
    ui->libraryLayout->addWidget(annotationEditorButton_, 8, 0, 1, 3);
    ui->resultLayout->addWidget(confirmFrontButton_);
    ui->resultLayout->addWidget(confirmBackButton_);
    ui->resultLayout->addWidget(rejectConfirmationButton_);
    ui->registerButton->setEnabled(false);
    ui->predictButton->setEnabled(false);
    ui->refreshWorkpiecesButton->setEnabled(false);
    ui->chooseFrontTemplatesButton->setEnabled(false);
    ui->chooseBackTemplatesButton->setEnabled(false);
    ui->chooseImageButton->setEnabled(false);
    ui->chooseBatchImagesButton->setEnabled(false);
    ui->batchPredictButton->setEnabled(false);
    ui->restartBackendButton->setEnabled(manager_ != nullptr);
    ui->resultLabel->setText(QStringLiteral("尚未检测"));
    ui->reviewLabel->clear();
    ui->batchSummaryLabel->clear();
    ui->evidenceTextEdit->clear();
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
    ui->batchResultsTableWidget->setColumnCount(5);
    ui->batchResultsTableWidget->setHorizontalHeaderLabels({
        QStringLiteral("文件"), QStringLiteral("结果"), QStringLiteral("复检"),
        QStringLiteral("耗时（毫秒）"), QStringLiteral("处理状态")});
    ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    ui->batchResultsTableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->batchResultsTableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->batchResultsTableWidget->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(ui->batchResultsTableWidget, &QTableWidget::currentCellChanged,
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
    connect(ui->chooseImageButton, &QPushButton::clicked,
            this, &MainWindow::chooseInspectionImage);
    connect(ui->chooseBatchImagesButton, &QPushButton::clicked,
            this, &MainWindow::chooseBatchImages);
    connect(ui->refreshWorkpiecesButton, &QPushButton::clicked,
            this, &MainWindow::refreshWorkpieces);
    connect(ui->registerButton, &QPushButton::clicked,
            this, &MainWindow::submitRegistration);
    connect(ui->predictButton, &QPushButton::clicked,
            this, &MainWindow::submitPrediction);
    connect(ui->batchPredictButton, &QPushButton::clicked,
            this, &MainWindow::submitBatchPrediction);
    connect(ui->restartBackendButton, &QPushButton::clicked,
            this, &MainWindow::restartBackend);
    connect(ui->workpieceComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        if (annotationManagerDialog_ != nullptr) {
            annotationManagerDialog_->close();
            annotationManagerDialog_.clear();
            annotationWorkpieceId_.clear();
        }
        if (resultContext_ != ResultContext::Batch) {
            lastPredictionWorkpieceId_.clear();
            lastPredictionImagePath_.clear();
            lastPredictionResponse_ = QJsonObject();
        }
        updateButtonStates();
    });
    connect(deleteWorkpieceButton_, &QPushButton::clicked,
            this, &MainWindow::deleteSelectedWorkpiece);
    connect(confirmFrontButton_, &QPushButton::clicked,
            this, &MainWindow::confirmFrontTemplate);
    connect(confirmBackButton_, &QPushButton::clicked,
            this, &MainWindow::confirmBackTemplate);
    connect(rejectConfirmationButton_, &QPushButton::clicked,
            this, &MainWindow::rejectTemplateConfirmation);
    connect(annotationEditorButton_, &QPushButton::clicked,
            this, &MainWindow::openAnnotationManager);
    confirmFrontButton_->setEnabled(false);
    confirmBackButton_->setEnabled(false);
    rejectConfirmationButton_->setEnabled(false);
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
    resultContext_ = ResultContext::None;
    lastPredictionImagePath_.clear();
    lastPredictionWorkpieceId_.clear();
    lastPredictionResponse_ = QJsonObject();
    ui->currentImageLabel->setText(
        QStringLiteral("当前图片：%1（单图）").arg(QFileInfo(inspectionImagePath_).fileName()));
    ui->currentResultTargetLabel->clear();
    updatePreview();
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
    ui->chooseBatchImagesButton->setText(batchImagePaths_.isEmpty()
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
    ui->backendStatusLabel->setText(QStringLiteral("后端：配置错误"));
    showLibraryMessage(message, true);
    ui->restartBackendButton->setEnabled(false);
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
    pendingCommand_ = QStringLiteral("list_workpieces");
    client_->sendRequest(QStringLiteral("list_workpieces"));
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
    if (client_ == nullptr || !backendReady_ || clientBusy_
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    const QString workpieceId = selectedWorkpieceId();
    if (workpieceId.isEmpty()) {
        ui->resultLabel->setText(QStringLiteral("请先选择工件"));
        return;
    }
    QString error;
    if (!validateImagePath(inspectionImagePath_, &error)) {
        ui->resultLabel->setText(error);
        return;
    }
    pendingPredictionWorkpieceId_ = workpieceId;
    pendingPredictionImagePath_ = inspectionImagePath_;
    pendingCommand_ = QStringLiteral("predict");
    client_->sendRequest(QStringLiteral("predict"), {
        {QStringLiteral("workpiece_id"), workpieceId},
        {QStringLiteral("image_path"), inspectionImagePath_},
    });
    ui->resultLabel->setText(QStringLiteral("正在检测…"));
    ui->reviewLabel->clear();
    ui->evidenceTextEdit->clear();
}

void MainWindow::submitBatchPrediction() {
    if (client_ == nullptr || !backendReady_ || clientBusy_ || batchInFlight_
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    const QString workpieceId = selectedWorkpieceId();
    if (workpieceId.isEmpty()) {
        ui->batchSummaryLabel->setText(QStringLiteral("请先选择工件"));
        return;
    }
    if (batchImagePaths_.isEmpty()) {
        ui->batchSummaryLabel->setText(QStringLiteral("请先选择批量图片"));
        return;
    }
    for (const QString &path : batchImagePaths_) {
        QString error;
        if (!validateImagePath(path, &error)) {
            ui->batchSummaryLabel->setText(QStringLiteral("批量检测无法开始：%1").arg(error));
            return;
        }
    }
    batchInFlight_ = true;
    batchWorkpieceId_ = workpieceId;
    batchIndex_ = 0;
    clearBatchResults();
    batchSelectionPinned_ = false;
    batchCompletedSuccessfully_ = false;
    ui->resultLabel->setText(QStringLiteral("批量检测进行中…"));
    ui->reviewLabel->clear();
    ui->evidenceTextEdit->clear();
    sendNextBatchPrediction();
}

void MainWindow::deleteSelectedWorkpiece() {
    const QString workpieceId = selectedWorkpieceId();
    if (workpieceId.isEmpty() || client_ == nullptr || clientBusy_ || !backendReady_) {
        return;
    }
    const QString name = ui->workpieceComboBox->currentText();
    const bool confirmed = QMessageBox::question(
        this, QStringLiteral("确认删除工件"),
        QStringLiteral("工件“%1”将移入可恢复回收区，是否继续？").arg(name),
        QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes;
    if (!confirmed) {
        return;
    }
    pendingCommand_ = QStringLiteral("recycle_workpiece");
    client_->sendRequest(QStringLiteral("recycle_workpiece"), {
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
            ui->currentResultTargetLabel->setText(
                QStringLiteral("当前：%1（不入库）").arg(QFileInfo(result.imagePath).fileName()));
        }
        ui->reviewLabel->setText(QStringLiteral("本次结果不入库"));
        updateButtonStates();
        return;
    }
    lastPredictionImagePath_.clear();
    lastPredictionWorkpieceId_.clear();
    lastPredictionResponse_ = QJsonObject();
    confirmFrontButton_->setEnabled(false);
    confirmBackButton_->setEnabled(false);
    rejectConfirmationButton_->setEnabled(false);
    ui->reviewLabel->setText(QStringLiteral("本次结果不入库"));
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
    pendingCommand_ = QStringLiteral("get_geometry_mask_profile");
    client_->sendRequest(QStringLiteral("get_geometry_mask_profile"), {
        {QStringLiteral("workpiece_id"), workpieceId},
    });
}

void MainWindow::saveGeometryDraft(const QJsonObject &draft, int libraryRevision, int draftRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    pendingCommand_ = QStringLiteral("save_geometry_mask_draft");
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    client_->sendRequest(QStringLiteral("save_geometry_mask_draft"), {
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
    pendingCommand_ = QStringLiteral("validate_geometry_mask_draft");
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    client_->sendRequest(QStringLiteral("validate_geometry_mask_draft"), {
        {QStringLiteral("workpiece_id"), geometryWorkpieceId_},
        {QStringLiteral("base_library_revision"), libraryRevision},
        {QStringLiteral("base_draft_revision"), draftRevision},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
}

void MainWindow::pollGeometryValidation() {
    if (geometryValidationJobId_.isEmpty() || client_ == nullptr || !backendReady_ || clientBusy_
        || !pendingCommand_.isEmpty() || client_->state() != BackendClient::State::Ready) return;
    pendingCommand_ = QStringLiteral("get_geometry_mask_validation_job");
    client_->sendRequest(QStringLiteral("get_geometry_mask_validation_job"), {
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
    pendingCommand_ = QStringLiteral("geometry_mask_validation_job_action");
    client_->sendRequest(QStringLiteral("geometry_mask_validation_job_action"), {
        {QStringLiteral("job_id"), jobId}, {QStringLiteral("action"), action},
    });
}

void MainWindow::publishGeometryProfile(const QString &jobId, int libraryRevision, int draftRevision,
                                         const QString &overrideReason) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    geometryPublishAfterValidation_ = false;
    geometryPublishOverrideReason_.clear();
    pendingCommand_ = QStringLiteral("publish_geometry_mask_profile");
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    client_->sendRequest(QStringLiteral("publish_geometry_mask_profile"), {
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
    pendingCommand_ = QStringLiteral("rollback_geometry_mask_profile");
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    client_->sendRequest(QStringLiteral("rollback_geometry_mask_profile"), {
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
    pendingCommand_ = QStringLiteral("preview_geometry_mask_rule");
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setPreviewBusy(true);
    client_->sendRequest(QStringLiteral("preview_geometry_mask_rule"), fields);
}

void MainWindow::resolveGeometryMigration(const QString &conflictId, const QJsonObject &resolution,
                                          int libraryRevision, int draftRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    pendingCommand_ = QStringLiteral("resolve_geometry_mask_migration");
    if (geometryMaskManagerDialog_ != nullptr) geometryMaskManagerDialog_->setBusy(true);
    client_->sendRequest(QStringLiteral("resolve_geometry_mask_migration"), {
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
    pendingCommand_ = QStringLiteral("get_workpiece_annotations");
    client_->sendRequest(QStringLiteral("get_workpiece_annotations"), {
        {QStringLiteral("workpiece_id"), workpieceId},
    });
}

void MainWindow::sendAnnotationMutation(const QString &command, const QJsonObject &fields) {
    if (client_ == nullptr || !backendReady_ || clientBusy_
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    pendingCommand_ = command;
    client_->sendRequest(command, fields);
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
    pendingCommand_ = QStringLiteral("submit_confirmation");
    client_->sendRequest(QStringLiteral("submit_confirmation"), {
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
    ui->backendStatusLabel->setText(QStringLiteral("后端：已连接"));
    ui->restartBackendButton->setEnabled(true);
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
    ui->backendStatusLabel->setText(QStringLiteral("后端：模型加载中"));
    showLibraryMessage(message.isEmpty() ? QStringLiteral("正在加载模型，请稍候…") : message);
    updateButtonStates();
}

void MainWindow::onBackendUnavailable(const QString &reason) {
    const bool hasPreservedWork = registrationInFlight_ || batchInFlight_ || clientBusy_
        || !pendingCommand_.isEmpty() || !inspectionImagePath_.isEmpty()
        || (ui->batchResultsTableWidget != nullptr && ui->batchResultsTableWidget->rowCount() > 0)
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
    pendingCommand_.clear();
    ui->backendStatusLabel->setText(QStringLiteral("后端：不可用"));
    showLibraryMessage(hasPreservedWork
                           ? QStringLiteral("后端连接中断，未完成操作结果未知；请重连后刷新：%1").arg(reason)
                           : reason,
                       true);
    ui->restartBackendButton->setEnabled(true);
    if (evolutionPollTimer_ != nullptr) {
        evolutionPollTimer_->stop();
    }
    if (geometryPollTimer_ != nullptr) {
        geometryPollTimer_->stop();
    }
    updateButtonStates();
}

void MainWindow::pollEvolutionJobs() {
    if (!backendReady_ || clientBusy_ || !pendingCommand_.isEmpty()
        || client_ == nullptr || client_->state() != BackendClient::State::Ready) {
        return;
    }
    pendingCommand_ = QStringLiteral("list_evolution_jobs");
    client_->sendRequest(QStringLiteral("list_evolution_jobs"));
}

void MainWindow::onClientStateChanged(BackendClient::State state, const QString &detail) {
    Q_UNUSED(detail)
    clientBusy_ = state == BackendClient::State::Busy;
    if (state == BackendClient::State::Ready) {
        backendReady_ = true;
        ui->backendStatusLabel->setText(QStringLiteral("后端：已连接"));
    } else if (state == BackendClient::State::Disconnected || state == BackendClient::State::Error) {
        ui->restartBackendButton->setEnabled(manager_ != nullptr || state == BackendClient::State::Error);
    }
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
        ui->workpieceComboBox->blockSignals(true);
        ui->workpieceComboBox->clear();
        const QJsonArray workpieces = response.value(QStringLiteral("workpieces")).toArray();
        for (const QJsonValue &value : workpieces) {
            const QJsonObject item = value.toObject();
            ui->workpieceComboBox->addItem(item.value(QStringLiteral("name")).toString(),
                                           item.value(QStringLiteral("id")).toString());
        }
        int index = ui->workpieceComboBox->findData(previousId);
        if (index < 0 && !pendingWorkpieceName_.isEmpty()) {
            index = ui->workpieceComboBox->findText(pendingWorkpieceName_);
        }
        if (index >= 0) {
            ui->workpieceComboBox->setCurrentIndex(index);
        }
        ui->workpieceComboBox->blockSignals(false);
        if (pendingCommand_ == QStringLiteral("list_workpieces")) {
            pendingCommand_.clear();
            pendingWorkpieceName_.clear();
        }
        if (!registrationSummaryVisible_) {
            showLibraryMessage(QStringLiteral("工件列表已刷新"));
        }
        updateButtonStates();
        return;
    }
    if (command == QStringLiteral("recycle_workpiece")) {
        pendingCommand_.clear();
        lastPredictionWorkpieceId_.clear();
        lastPredictionImagePath_.clear();
        lastPredictionResponse_ = QJsonObject();
        showLibraryMessage(QStringLiteral("工件已移入回收区，可在后端恢复"));
        requestWorkpieceRefresh(false);
        return;
    }
    if (command == QStringLiteral("submit_confirmation")) {
        pendingCommand_.clear();
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
        }
        showLibraryMessage(QStringLiteral("确认图片已进入后台入库队列"));
        updateButtonStates();
        return;
    }
    if (command == QStringLiteral("get_geometry_mask_profile")) {
        pendingCommand_.clear();
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryMaskManagerDialog_->setBusy(false);
        }
        showLibraryMessage(QStringLiteral("几何干扰规则已加载"));
        return;
    }
    if (command == QStringLiteral("preview_geometry_mask_rule")) {
        pendingCommand_.clear();
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setRulePreview(response.value(QStringLiteral("preview")).toObject());
            geometryMaskManagerDialog_->setPreviewBusy(false);
        }
        showLibraryMessage(QStringLiteral("几何规则预览已更新"));
        return;
    }
    if (command == QStringLiteral("save_geometry_mask_draft")) {
        pendingCommand_.clear();
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
        pendingCommand_.clear();
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryMaskManagerDialog_->setBusy(false);
        }
        showLibraryMessage(QStringLiteral("迁移冲突处置已保存"));
        return;
    }
    if (command == QStringLiteral("validate_geometry_mask_draft")) {
        pendingCommand_.clear();
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
        pendingCommand_.clear();
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
        pendingCommand_.clear();
        if (geometryMaskManagerDialog_ != nullptr) {
            geometryMaskManagerDialog_->setValidationJob(response.value(QStringLiteral("job")).toObject());
            geometryMaskManagerDialog_->setBusy(false);
        }
        return;
    }
    if (command == QStringLiteral("publish_geometry_mask_profile")
        || command == QStringLiteral("rollback_geometry_mask_profile")) {
        pendingCommand_.clear();
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
            pendingCommand_.clear();
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
        pendingCommand_.clear();
        showLibraryMessage(QStringLiteral("干扰标注快照已加载"));
        return;
    }
    if (command == QStringLiteral("save_workpiece_annotations")
        || command == QStringLiteral("set_workpiece_annotation_group_enabled")
        || command == QStringLiteral("delete_workpiece_annotation_group")) {
        pendingCommand_.clear();
        showLibraryMessage(QStringLiteral("干扰标注已保存，正在刷新生效状态…"));
        if (annotationManagerDialog_ != nullptr) {
            annotationManagerDialog_->setBusy(true);
        }
        const QString workpieceId = annotationWorkpieceId_;
        QTimer::singleShot(0, this, [this, workpieceId]() { requestAnnotationSnapshot(workpieceId); });
        return;
    }
    if (command == QStringLiteral("list_evolution_jobs")) {
        pendingCommand_.clear();
        const QJsonArray jobs = response.value(QStringLiteral("jobs")).toArray();
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
        pendingCommand_.clear();
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
    if (command == QStringLiteral("predict")) {
        lastPredictionWorkpieceId_ = pendingPredictionWorkpieceId_;
        lastPredictionImagePath_ = pendingPredictionImagePath_;
        lastPredictionResponse_ = response;
        resultContext_ = ResultContext::Single;
        renderPredictionResult(lastPredictionImagePath_, response, QStringLiteral("单图"));
        pendingCommand_.clear();
    }
}

void MainWindow::onClientCommandFailed(const QString &command, const QString &code,
                                       const QString &message) {
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
            pendingCommand_.clear();
            showLibraryMessage(QStringLiteral("已取消覆盖"));
        }
        return;
    }
    if (batchInFlight_ && command == QStringLiteral("predict")) {
        batchInFlight_ = false;
        batchCompletedSuccessfully_ = false;
        pendingCommand_.clear();
        const QString failedFile = batchIndex_ >= 0 && batchIndex_ < batchImagePaths_.size()
            ? QFileInfo(batchImagePaths_.at(batchIndex_)).fileName() : QStringLiteral("未知");
        ui->batchSummaryLabel->setText(
            QStringLiteral("批量检测未完成：已完成 %1/%2，失败文件：%3，原因：%4")
                .arg(batchResults_.size())
                .arg(batchImagePaths_.size())
                .arg(failedFile, message));
        ui->resultLabel->setText(QStringLiteral("批量检测未完成"));
        updateButtonStates();
        return;
    }
    if (command == QStringLiteral("register")) {
        stopRegistrationProgress();
        registrationInFlight_ = false;
        pendingCommand_.clear();
        showLibraryMessage(message, true);
    } else if (command == QStringLiteral("predict")) {
        pendingCommand_.clear();
        ui->resultLabel->setText(QStringLiteral("检测失败：%1").arg(message));
    } else if (command == QStringLiteral("submit_confirmation")) {
        pendingCommand_.clear();
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
        pendingCommand_.clear();
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
        pendingCommand_.clear();
        showLibraryMessage(QStringLiteral("删除失败：%1").arg(message), true);
    } else if (command == QStringLiteral("get_workpiece_annotations")) {
        pendingCommand_.clear();
        showLibraryMessage(QStringLiteral("干扰标注加载失败：%1").arg(message), true);
    } else if (command == QStringLiteral("save_workpiece_annotations")
               || command == QStringLiteral("set_workpiece_annotation_group_enabled")
               || command == QStringLiteral("delete_workpiece_annotation_group")) {
        const QString workpieceId = annotationWorkpieceId_;
        const bool stale = code == QStringLiteral("STALE_WORKPIECE_REVISION");
        pendingCommand_.clear();
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
        if (pendingCommand_ == command) {
            pendingCommand_.clear();
        }
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
    ui->refreshWorkpiecesButton->setEnabled(interactive);
    ui->chooseFrontTemplatesButton->setEnabled(interactive);
    ui->chooseBackTemplatesButton->setEnabled(interactive);
    ui->chooseImageButton->setEnabled(interactive);
    ui->chooseBatchImagesButton->setEnabled(interactive);
    ui->registerButton->setEnabled(interactive && !ui->workpieceNameEdit->text().trimmed().isEmpty()
                                   && !frontTemplatePaths_.isEmpty() && !backTemplatePaths_.isEmpty());
    ui->predictButton->setEnabled(interactive && !selectedWorkpieceId().isEmpty()
                                  && !inspectionImagePath_.isEmpty());
    ui->batchPredictButton->setEnabled(interactive && !selectedWorkpieceId().isEmpty()
                                       && !batchImagePaths_.isEmpty());
    if (deleteWorkpieceButton_ != nullptr) {
        deleteWorkpieceButton_->setEnabled(interactive && !selectedWorkpieceId().isEmpty());
    }
    if (annotationEditorButton_ != nullptr) {
        annotationEditorButton_->setEnabled(interactive && !selectedWorkpieceId().isEmpty());
    }
    const bool confirmationAvailable = resultContext_ == ResultContext::Batch
        ? currentBatchResultCanBeProcessed()
        : interactive && resultContext_ == ResultContext::Single
            && !lastPredictionImagePath_.isEmpty() && !lastPredictionWorkpieceId_.isEmpty();
    if (confirmFrontButton_ != nullptr) {
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
        ui->currentResultTargetLabel->setText(
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
    if (inspectionImagePath_.isEmpty()) {
        ui->imagePreviewLabel->setPixmap(QPixmap());
        ui->imagePreviewLabel->setText(QStringLiteral("请选择待测图片"));
        return;
    }
    QImageReader reader(inspectionImagePath_);
    const QImage image = reader.read();
    if (image.isNull()) {
        ui->imagePreviewLabel->setPixmap(QPixmap());
        ui->imagePreviewLabel->setText(
            QStringLiteral("图片无法读取：%1").arg(QFileInfo(inspectionImagePath_).fileName()));
        return;
    }
    ui->imagePreviewLabel->setText(QString());
    ui->imagePreviewLabel->setPixmap(QPixmap::fromImage(image).scaled(
        ui->imagePreviewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::clearInspectionState() {
    inspectionImagePath_.clear();
    updatePreview();
    ui->resultLabel->setText(QStringLiteral("尚未检测"));
    ui->reviewLabel->clear();
    ui->evidenceTextEdit->clear();
    pendingCommand_.clear();
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
    ui->chooseBatchImagesButton->setText(QStringLiteral("选择批量图片"));
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
        ui->currentImageLabel->clear();
        ui->currentResultTargetLabel->clear();
        ui->resultLabel->setText(QStringLiteral("尚未检测"));
        ui->reviewLabel->clear();
        ui->evidenceTextEdit->clear();
    }
    if (ui == nullptr || ui->batchResultsTableWidget == nullptr) {
        return;
    }
    ui->batchResultsTableWidget->setRowCount(0);
    ui->batchSummaryLabel->clear();
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
    pendingCommand_ = QStringLiteral("register");
    client_->sendRequest(QStringLiteral("register"), {
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
    pendingCommand_ = QStringLiteral("batch_predict");
    pendingPredictionWorkpieceId_ = batchWorkpieceId_;
    pendingPredictionImagePath_ = batchImagePaths_.at(batchIndex_);
    client_->sendRequest(QStringLiteral("predict"), {
        {QStringLiteral("workpiece_id"), pendingPredictionWorkpieceId_},
        {QStringLiteral("image_path"), batchImagePaths_.at(batchIndex_)},
    });
    ui->batchSummaryLabel->setText(
        QStringLiteral("已完成 %1/%2，当前文件：%3")
            .arg(batchResults_.size())
            .arg(batchImagePaths_.size())
            .arg(QFileInfo(batchImagePaths_.at(batchIndex_)).fileName()));
}

void MainWindow::renderPredictionResult(const QString &imagePath, const QJsonObject &response,
                                        const QString &sourceText) {
    if (!imagePath.isEmpty()) {
        inspectionImagePath_ = QFileInfo(imagePath).absoluteFilePath();
        updatePreview();
        const QString fileName = QFileInfo(inspectionImagePath_).fileName();
        ui->currentImageLabel->setText(
            QStringLiteral("当前图片：%1（%2）").arg(fileName, sourceText));
        ui->currentResultTargetLabel->setText(
            QStringLiteral("当前：%1（%2）").arg(fileName, sourceText));
    }

    const QString label = response.value(QStringLiteral("label")).toString();
    ui->resultLabel->setText(QStringLiteral("检测结果：%1").arg(orientationText(label)));
    ui->reviewLabel->clear();
    QStringList lines;
    lines << QStringLiteral("全局得分：正面 %1，反面 %2")
                 .arg(formatScore(response.value(QStringLiteral("global_scores")).toObject(),
                                  QStringLiteral("front")),
                      formatScore(response.value(QStringLiteral("global_scores")).toObject(),
                                  QStringLiteral("back")));
    lines << QStringLiteral("全局间隔：%1")
                 .arg(response.value(QStringLiteral("global_margin")).toDouble());
    lines << QStringLiteral("局部预测：%1")
                 .arg(orientationText(response.value(QStringLiteral("local_prediction")).toString()));
    lines << QStringLiteral("局部得分：正面 %1，反面 %2")
                 .arg(formatScore(response.value(QStringLiteral("local_scores")).toObject(),
                                  QStringLiteral("front")),
                      formatScore(response.value(QStringLiteral("local_scores")).toObject(),
                                  QStringLiteral("back")));
    lines << QStringLiteral("局部间隔：%1")
                 .arg(response.value(QStringLiteral("local_margin")).toDouble());
    lines << QStringLiteral("决策来源：%1")
                 .arg(decisionSourceText(response.value(QStringLiteral("decision_source")).toString()));
    lines << QStringLiteral("耗时（毫秒）：%1")
                 .arg(response.value(QStringLiteral("elapsed_ms")).toDouble());
    const QJsonObject geometryMask = response.value(QStringLiteral("geometry_mask")).toObject();
    const QString geometryStatus = geometryMask.value(QStringLiteral("status")).toString();
    if (geometryStatus == QStringLiteral("active")) {
        lines << QStringLiteral("几何遮罩：已启用（版本 %1）")
                     .arg(geometryMask.value(QStringLiteral("profile_revision")).toInt());
    } else if (geometryStatus == QStringLiteral("low_confidence")
               || geometryStatus == QStringLiteral("unavailable")
               || geometryStatus == QStringLiteral("unreadable")) {
        lines << QStringLiteral("几何遮罩：未启用（需复核，%1）").arg(geometryStatus);
        ui->reviewLabel->setText(QStringLiteral("遮罩未启用，需复核；保留原始识别结果"));
    }
    if (geometryStatus.isEmpty() || geometryStatus == QStringLiteral("active")) {
        ui->reviewLabel->setText(response.value(QStringLiteral("needs_review")).toBool()
                                     ? QStringLiteral("建议人工复检") : QString());
    }
    ui->evidenceTextEdit->setPlainText(lines.join(QLatin1Char('\n')));
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
    ui->batchResultsTableWidget->insertRow(row);
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
    ui->batchResultsTableWidget->setItem(index, 0, fileItem);
    ui->batchResultsTableWidget->setItem(
        index, 1, new QTableWidgetItem(orientationText(result.label)));
    ui->batchResultsTableWidget->setItem(
        index, 2, new QTableWidgetItem(result.needsReview ? QStringLiteral("是")
                                                         : QStringLiteral("否")));
    ui->batchResultsTableWidget->setItem(
        index, 3, new QTableWidgetItem(QString::number(result.elapsedMs, 'f', 1)));
    ui->batchResultsTableWidget->setItem(
        index, 4, new QTableWidgetItem(batchResultStateText(result.state)));
    if (result.needsReview) {
        const QBrush warningBrush(QColor(255, 244, 204));
        for (int column = 0; column < ui->batchResultsTableWidget->columnCount(); ++column) {
            ui->batchResultsTableWidget->item(index, column)->setBackground(warningBrush);
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
    ui->batchResultsTableWidget->setCurrentCell(index, 0);
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
        ui->batchSummaryLabel->setText(
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
    ui->batchSummaryLabel->setText(
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
    pendingCommand_.clear();
    ui->resultLabel->setText(QStringLiteral("批量检测完成"));
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
    return ui->workpieceComboBox->currentData().toString();
}

QString MainWindow::orientationText(const QString &label) const {
    if (label == QStringLiteral("front")) return QStringLiteral("正面");
    if (label == QStringLiteral("back")) return QStringLiteral("反面");
    if (label == QStringLiteral("uncertain")) return QStringLiteral("不确定");
    return label;
}

QString MainWindow::decisionSourceText(const QString &source) const {
    if (source == QStringLiteral("global")) return QStringLiteral("全局特征");
    if (source == QStringLiteral("local_override")) return QStringLiteral("局部特征覆盖");
    return source;
}

QString MainWindow::formatScore(const QJsonObject &scores, const QString &key) const {
    return QString::number(scores.value(key).toDouble(), 'g', 8);
}

void MainWindow::showLibraryMessage(const QString &message, bool error) {
    ui->libraryMessageLabel->setStyleSheet(error ? QStringLiteral("color: #b00020;") : QString());
    ui->libraryMessageLabel->setText(message);
}
