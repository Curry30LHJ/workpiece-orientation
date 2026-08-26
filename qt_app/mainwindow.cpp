#include "mainwindow.h"

#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QMessageBox>
#include <QCloseEvent>
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
#include "geometryrulespage.h"
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
    geometryRulesPage_ = new GeometryRulesPage(ui->geometryPageHost);
    geometryRulesPage_->setObjectName(QStringLiteral("geometryRulesPage"));
    geometryRulesPage_->setBackendAvailable(false, QStringLiteral("后端尚未就绪"));
    ui->geometryPageHostLayout->addWidget(geometryRulesPage_);
    connect(geometryRulesPage_, &GeometryRulesPage::saveDraftRequested,
            this, &MainWindow::saveGeometryDraft);
    connect(geometryRulesPage_, &GeometryRulesPage::validateRequested,
            this, &MainWindow::validateGeometryDraft);
    connect(geometryRulesPage_, &GeometryRulesPage::validationJobActionRequested,
            this, &MainWindow::geometryJobAction);
    connect(geometryRulesPage_, &GeometryRulesPage::publishRequested,
            this, &MainWindow::publishGeometryProfile);
    connect(geometryRulesPage_, &GeometryRulesPage::publishWorkflowRequested,
            this, &MainWindow::publishGeometryWorkflow);
    connect(geometryRulesPage_, &GeometryRulesPage::rollbackRequested,
            this, &MainWindow::rollbackGeometryProfile);
    connect(geometryRulesPage_, &GeometryRulesPage::previewRequested,
            this, &MainWindow::previewGeometryRule);
    connect(geometryRulesPage_, &GeometryRulesPage::migrationResolutionRequested,
            this, &MainWindow::resolveGeometryMigration);
    connect(geometryRulesPage_, &GeometryRulesPage::snapshotRequested,
            this, &MainWindow::requestGeometryProfile);
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
                applyDetectionWorkpieceChange(workpieceId);
            });
    connect(workpieceLibraryPage_, &WorkpieceLibraryPage::taskStatusChanged,
            this, [this](const QString &title, const QString &phase,
                         int completed, int total, qint64 elapsedMs) {
                globalTaskStatus_->setRunning(title, phase, completed, total,
                                              qMax<qint64>(0, elapsedMs));
                if (phase == QStringLiteral("failed")) {
                    globalTaskStatus_->setMessage(
                        TaskStatusWidget::MessageKind::Error,
                        elapsedMs < 0
                            ? QStringLiteral("%1 · %2/%3 · 耗时未知 · 任务失败")
                                  .arg(phase).arg(completed).arg(total)
                            : QStringLiteral("%1 · %2/%3 · 耗时 %4 ms · 任务失败")
                                  .arg(phase).arg(completed).arg(total).arg(elapsedMs));
                } else if (phase == QStringLiteral("cancelled")) {
                    globalTaskStatus_->setMessage(
                        TaskStatusWidget::MessageKind::Warning,
                        elapsedMs < 0
                            ? QStringLiteral("建库已取消 · %1/%2 · 耗时未知")
                                  .arg(completed).arg(total)
                            : QStringLiteral("建库已取消 · %1/%2 · 耗时 %3 ms")
                                  .arg(completed).arg(total).arg(elapsedMs));
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
    connect(appHeader_, &AppHeader::currentWorkpieceRequested, this,
            [this](const QString &workpieceId) {
        if (!applyDetectionWorkpieceChange(workpieceId)) return;
        if (annotationManagerDialog_ != nullptr) {
            annotationManagerDialog_->close();
            annotationManagerDialog_.clear();
            annotationWorkpieceId_.clear();
        }
    });
    connect(annotationEditorButton_, &QPushButton::clicked,
            this, &MainWindow::openAnnotationManager);
    confirmFrontButton_->setEnabled(false);
    confirmBackButton_->setEnabled(false);
    rejectConfirmationButton_->setEnabled(false);
}

MainWindow::GeometryDirtyDecision MainWindow::promptForDirtyGeometry() {
    QMessageBox prompt(this);
    prompt.setWindowTitle(QStringLiteral("未保存的几何草稿"));
    prompt.setText(QStringLiteral("几何规则中有未保存的修改，如何处理？"));
    QAbstractButton *save = prompt.addButton(
        QStringLiteral("保存并继续"), QMessageBox::AcceptRole);
    QAbstractButton *discard = prompt.addButton(
        QStringLiteral("放弃修改"), QMessageBox::DestructiveRole);
    QAbstractButton *cancel = prompt.addButton(
        QStringLiteral("取消"), QMessageBox::RejectRole);
    prompt.exec();
    if (prompt.clickedButton() == discard) return GeometryDirtyDecision::Discard;
    if (prompt.clickedButton() == save) return GeometryDirtyDecision::Save;
    Q_UNUSED(cancel)
    return GeometryDirtyDecision::Cancel;
}

void MainWindow::closeEvent(QCloseEvent *event) {
    if (bypassCloseGuard_) {
        bypassCloseGuard_ = false;
        QMainWindow::closeEvent(event);
        return;
    }
    if (pendingNavigationKind_ != PendingNavigationKind::None) {
        event->ignore();
        return;
    }
    if (geometryRulesPage_ == nullptr || !geometryRulesPage_->hasUnsavedChanges()) {
        QMainWindow::closeEvent(event);
        return;
    }
    const GeometryDirtyDecision decision = promptForDirtyGeometry();
    if (decision == GeometryDirtyDecision::Cancel) {
        event->ignore();
        return;
    }
    if (decision == GeometryDirtyDecision::Discard) {
        geometryRulesPage_->discardUnsavedChanges();
    } else if (decision == GeometryDirtyDecision::Save) {
        beginPendingGeometryNavigation(PendingNavigationKind::CloseWindow);
        event->ignore();
        return;
    }
    QMainWindow::closeEvent(event);
}

bool MainWindow::beginPendingGeometryNavigation(PendingNavigationKind kind,
                                                AppPage page,
                                                const QString &workpieceId) {
    if (pendingNavigationKind_ != PendingNavigationKind::None
        || geometryRulesPage_ == nullptr || client_ == nullptr || !backendReady_
        || (client_->state() != BackendClient::State::Ready
            && client_->state() != BackendClient::State::Busy)) {
        if (geometryRulesPage_ != nullptr) {
            geometryRulesPage_->setOperationError(
                QStringLiteral("当前无法保存草稿，请检查后端连接后重试"));
        }
        return false;
    }
    pendingNavigationKind_ = kind;
    pendingNavigationPage_ = page;
    pendingNavigationWorkpieceId_ = workpieceId;
    setGeometryOperationEditingLocked(true);
    geometryRulesPage_->setEnabled(false);
    if (tryStartPendingGeometrySave()) return true;
    cancelPendingGeometryNavigation();
    setGeometryOperationEditingLocked(false);
    return false;
}

bool MainWindow::tryStartPendingGeometrySave() {
    if (pendingNavigationKind_ == PendingNavigationKind::None) return false;
    if (stagedGeometrySaveIntent_ == GeometrySaveIntent::Navigation) {
        dispatchStagedGeometryDraftSave();
        return true;
    }
    if (geometrySaveIntent_ == GeometrySaveIntent::Navigation
        && pendingCommand_ == QStringLiteral("save_geometry_mask_draft")) {
        return true;
    }
    if (geometryRulesPage_ == nullptr || client_ == nullptr || !backendReady_
        || (client_->state() != BackendClient::State::Ready
            && client_->state() != BackendClient::State::Busy)) {
        return false;
    }
    geometryRulesPage_->requestSaveDraft();
    return stagedGeometrySaveIntent_ == GeometrySaveIntent::Navigation
        || (geometrySaveIntent_ == GeometrySaveIntent::Navigation
            && pendingCommand_ == QStringLiteral("save_geometry_mask_draft"));
}

void MainWindow::cancelPendingGeometryNavigation() {
    pendingNavigationKind_ = PendingNavigationKind::None;
    pendingNavigationPage_ = AppPage::Inspection;
    pendingNavigationWorkpieceId_.clear();
    pendingNavigationWorkpieceListResponse_ = QJsonObject();
    pendingNavigationRefreshTransactionId_ = 0;
    pendingNavigationResponseIncludedMandatory_ = false;
    if (stagedGeometrySaveIntent_ == GeometrySaveIntent::Navigation) {
        clearStagedGeometryDraftSave();
    }
    if (geometrySaveIntent_ == GeometrySaveIntent::Navigation) {
        geometrySaveIntent_ = GeometrySaveIntent::None;
    }
    if (geometryRulesPage_ != nullptr) geometryRulesPage_->setEnabled(true);
}

void MainWindow::completePendingGeometryNavigation() {
    const PendingNavigationKind kind = pendingNavigationKind_;
    const AppPage page = pendingNavigationPage_;
    const QString workpieceId = pendingNavigationWorkpieceId_;
    const QJsonObject listResponse = pendingNavigationWorkpieceListResponse_;
    const quint64 refreshTransactionId = pendingNavigationRefreshTransactionId_;
    const bool includedMandatory = pendingNavigationResponseIncludedMandatory_;
    cancelPendingGeometryNavigation();

    if (kind == PendingNavigationKind::Page) {
        setCurrentPageUnchecked(page);
        if (!listResponse.isEmpty()) {
            applyWorkpieceListResponse(listResponse, refreshTransactionId,
                                       includedMandatory);
        }
        if (page == AppPage::GeometryRules) ensureGeometryProfileForCurrentWorkpiece();
    } else if (kind == PendingNavigationKind::Workpiece) {
        if (!listResponse.isEmpty()) {
            applyWorkpieceListResponse(listResponse, refreshTransactionId,
                                       includedMandatory, workpieceId);
        } else {
            applyDetectionWorkpieceChange(workpieceId);
        }
    } else if (kind == PendingNavigationKind::CloseWindow) {
        bypassCloseGuard_ = true;
        close();
    }
    setGeometryOperationEditingLocked(false);
}

void MainWindow::setCurrentPageUnchecked(AppPage page) {
    ui->mainPageStack->setCurrentIndex(static_cast<int>(page));
    currentPage_ = page;
    appHeader_->setCurrentPage(page);
}

void MainWindow::refreshDetectionWorkpieceConsumers() {
    if (!currentDetectionWorkpieceId_.isEmpty()) {
        appHeader_->setCurrentWorkpieceId(currentDetectionWorkpieceId_);
    }
    inspectionPage_->setCurrentWorkpiece(appHeader_->currentWorkpieceId(),
                                         appHeader_->currentWorkpieceName());
    workpieceLibraryPage_->setWorkpieces(workpieceSummaries_,
                                         appHeader_->currentWorkpieceId());
    if (resultContext_ != ResultContext::Batch) {
        lastPredictionWorkpieceId_.clear();
        lastPredictionImagePath_.clear();
        lastPredictionResponse_ = QJsonObject();
    }
    updateButtonStates();
}

void MainWindow::applyWorkpieceListResponse(const QJsonObject &response,
                                            quint64 refreshTransactionId,
                                            bool includedMandatoryRefresh,
                                            const QString &preferredTarget) {
    const QString previousId = selectedWorkpieceId();
    const QJsonArray workpieces = response.value(QStringLiteral("workpieces")).toArray();
    QList<QPair<QString, QString>> items;
    QString requestedId = preferredTarget.isEmpty() ? previousId : preferredTarget;
    bool previousStillExists = false;
    bool requestedStillExists = false;
    for (const QJsonValue &value : workpieces) {
        const QJsonObject item = value.toObject();
        const QString id = item.value(QStringLiteral("id")).toString();
        const QString name = item.value(QStringLiteral("name")).toString();
        items.append(qMakePair(id, name));
        previousStillExists = previousStillExists || id == previousId;
        requestedStillExists = requestedStillExists || id == requestedId;
    }
    if (!requestedStillExists) {
        requestedId = previousStillExists ? previousId : QString();
    }
    const QString proposedId = requestedId.isEmpty() && !items.isEmpty()
        ? items.constFirst().first : requestedId;
    if (geometryRulesPage_ != nullptr && geometryRulesPage_->hasUnsavedChanges()
        && proposedId != currentDetectionWorkpieceId_) {
        pendingNavigationWorkpieceListResponse_ = response;
        pendingNavigationRefreshTransactionId_ = refreshTransactionId;
        pendingNavigationResponseIncludedMandatory_ = includedMandatoryRefresh;
        if (!applyDetectionWorkpieceChange(proposedId)) {
            if (pendingNavigationKind_ != PendingNavigationKind::Workpiece) {
                pendingNavigationWorkpieceListResponse_ = QJsonObject();
                pendingNavigationRefreshTransactionId_ = 0;
                pendingNavigationResponseIncludedMandatory_ = false;
                showLibraryMessage(QStringLiteral("已取消工件切换，几何规则草稿仍保留"));
            }
            return;
        }
        pendingNavigationWorkpieceListResponse_ = QJsonObject();
        pendingNavigationRefreshTransactionId_ = 0;
        pendingNavigationResponseIncludedMandatory_ = false;
    }
    workpieceSummaries_ = workpieces;
    appHeader_->setWorkpieces(items, requestedId);
    if (currentDetectionWorkpieceId_ != appHeader_->currentWorkpieceId()) {
        applyDetectionWorkpieceChange(appHeader_->currentWorkpieceId());
    } else {
        refreshDetectionWorkpieceConsumers();
    }
    if (includedMandatoryRefresh && refreshTransactionId != 0
        && refreshTransactionId == activeMandatoryRefreshTransactionId_) {
        mandatoryRefreshRetryRequired_ = false;
        const QString browsedId = workpieceLibraryPage_->browsedWorkpieceId();
        if (!browsedId.isEmpty()) {
            activeMandatoryDetailsWorkpieceId_ = browsedId;
            sendPageCommand(CommandOwner::Library,
                            QStringLiteral("get_workpiece_details"),
                            {{QStringLiteral("workpiece_id"), browsedId}},
                            refreshTransactionId);
        } else {
            activeMandatoryDetailsWorkpieceId_.clear();
            activeMandatoryRefreshTransactionId_ = 0;
        }
    }
    showLibraryMessage(QStringLiteral("工件列表已刷新"));
    updateButtonStates();
}

bool MainWindow::applyDetectionWorkpieceChange(const QString &workpieceId) {
    if (pendingNavigationKind_ != PendingNavigationKind::None) {
        if (!currentDetectionWorkpieceId_.isEmpty()) {
            appHeader_->setCurrentWorkpieceId(currentDetectionWorkpieceId_);
        }
        return false;
    }
    if (workpieceId == currentDetectionWorkpieceId_) {
        if (!workpieceId.isEmpty()) appHeader_->setCurrentWorkpieceId(workpieceId);
        return true;
    }
    if (geometryRulesPage_ != nullptr && geometryRulesPage_->hasUnsavedChanges()
        && workpieceId != geometryWorkpieceId_) {
        const GeometryDirtyDecision decision = promptForDirtyGeometry();
        if (decision == GeometryDirtyDecision::Cancel) {
            if (!currentDetectionWorkpieceId_.isEmpty()) {
                appHeader_->setCurrentWorkpieceId(currentDetectionWorkpieceId_);
            }
            return false;
        }
        if (decision == GeometryDirtyDecision::Discard) {
            geometryRulesPage_->discardUnsavedChanges();
        } else if (decision == GeometryDirtyDecision::Save) {
            beginPendingGeometryNavigation(PendingNavigationKind::Workpiece,
                                           AppPage::Inspection, workpieceId);
            if (!currentDetectionWorkpieceId_.isEmpty()) {
                appHeader_->setCurrentWorkpieceId(currentDetectionWorkpieceId_);
            }
            return false;
        }
    }
    currentDetectionWorkpieceId_ = workpieceId;
    ++geometryTargetGeneration_;
    if (!workpieceId.isEmpty()) appHeader_->setCurrentWorkpieceId(workpieceId);
    if (annotationManagerDialog_ != nullptr) {
        annotationManagerDialog_->close();
        annotationManagerDialog_.clear();
        annotationWorkpieceId_.clear();
    }
    refreshDetectionWorkpieceConsumers();
    if (currentPage_ == AppPage::GeometryRules
        && !geometryRulesPage_->hasUnsavedChanges()) {
        ensureGeometryProfileForCurrentWorkpiece();
    }
    return true;
}

bool MainWindow::requestPage(AppPage page) {
    if (pendingNavigationKind_ != PendingNavigationKind::None) {
        appHeader_->setCurrentPage(currentPage_);
        return false;
    }
    if (page == currentPage_) {
        if (page == AppPage::GeometryRules) ensureGeometryProfileForCurrentWorkpiece();
        return true;
    }
    if (geometryRulesPage_ != nullptr && geometryRulesPage_->hasUnsavedChanges()) {
        const bool leavingGeometry = currentPage_ == AppPage::GeometryRules
            && page != AppPage::GeometryRules;
        const bool enteringDifferentGeometry = page == AppPage::GeometryRules
            && geometryWorkpieceId_ != currentDetectionWorkpieceId_;
        if (leavingGeometry || enteringDifferentGeometry) {
            const GeometryDirtyDecision decision = promptForDirtyGeometry();
            if (decision == GeometryDirtyDecision::Cancel) {
                appHeader_->setCurrentPage(currentPage_);
                return false;
            }
            if (decision == GeometryDirtyDecision::Discard) {
                geometryRulesPage_->discardUnsavedChanges();
            } else if (decision == GeometryDirtyDecision::Save) {
                beginPendingGeometryNavigation(PendingNavigationKind::Page, page);
                appHeader_->setCurrentPage(currentPage_);
                return false;
            }
        }
    }
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
    setCurrentPageUnchecked(page);
    if (page == AppPage::GeometryRules) ensureGeometryProfileForCurrentWorkpiece();
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
                                 const QJsonObject &fields,
                                 quint64 refreshTransactionId) {
    if (client_ == nullptr || command.isEmpty()) return;

    quint64 effectiveRefreshTransactionId = refreshTransactionId;
    if (owner == CommandOwner::Library
        && command == QStringLiteral("get_workpiece_details")
        && effectiveRefreshTransactionId == 0
        && activeMandatoryRefreshTransactionId_ != 0
        && !activeMandatoryDetailsWorkpieceId_.isEmpty()
        && !mandatoryDetailsRetryRequired_) {
        effectiveRefreshTransactionId = activeMandatoryRefreshTransactionId_;
        activeMandatoryDetailsWorkpieceId_ = fields.value(
            QStringLiteral("workpiece_id")).toString();
    }
    const quint64 geometryTargetGeneration =
        owner == CommandOwner::Geometry
            && command == QStringLiteral("get_geometry_mask_profile")
        ? geometryTargetGeneration_ : 0;
    const QueuedCommandIntent intent{
        owner, command, fields, effectiveRefreshTransactionId,
        geometryTargetGeneration};
    if (owner == CommandOwner::Library
        && command == QStringLiteral("register")
        && fields.value(QStringLiteral("replace")).toBool()) {
        hasReplaceRegistrationContinuation_ = true;
        replaceRegistrationContinuationFields_ = fields;
    } else if (owner == CommandOwner::Library
               && command == QStringLiteral("get_workpiece_details")) {
        hasLatestDetailsIntent_ = true;
        latestDetailsFields_ = fields;
        latestDetailsRefreshTransactionId_ = effectiveRefreshTransactionId;
    } else if (command == QStringLiteral("list_workpieces")) {
        if (owner == CommandOwner::UserRefresh) {
            deferredUserWorkpieceRefresh_ = true;
        } else {
            mandatoryWorkpieceRefresh_ = true;
            queuedMandatoryRefreshTransactionId_ = effectiveRefreshTransactionId;
        }
    } else if (owner == CommandOwner::System) {
        bool replaced = false;
        for (int index = 0; index < queuedInternalCommands_.size(); ++index) {
            if (queuedInternalCommands_.at(index).command == command) {
                queuedInternalCommands_[index] = intent;
                replaced = true;
                break;
            }
        }
        if (!replaced) queuedInternalCommands_.enqueue(intent);
    } else {
        queuedMutationCommands_.enqueue(intent);
    }
    dispatchQueuedCommand();
}

void MainWindow::dispatchQueuedCommand() {
    if (client_ == nullptr || clientBusy_ || !pendingCommand_.isEmpty()
        || client_->state() != BackendClient::State::Ready) {
        return;
    }

    if (hasReplaceRegistrationContinuation_) {
        const QJsonObject fields = replaceRegistrationContinuationFields_;
        hasReplaceRegistrationContinuation_ = false;
        replaceRegistrationContinuationFields_ = QJsonObject();
        issuePageCommand({CommandOwner::Library, QStringLiteral("register"), fields});
        return;
    }
    if (batchInFlight_) {
        for (int index = 0; index < queuedMutationCommands_.size(); ++index) {
            const QueuedCommandIntent &candidate = queuedMutationCommands_.at(index);
            if (candidate.owner == CommandOwner::Inspection
                && candidate.command == QStringLiteral("predict")) {
                issuePageCommand(queuedMutationCommands_.takeAt(index));
                return;
            }
        }
        return;
    }
    if (mandatoryWorkpieceRefresh_ || deferredUserWorkpieceRefresh_) {
        const bool includesMandatory = mandatoryWorkpieceRefresh_;
        const bool includesUser = deferredUserWorkpieceRefresh_;
        const quint64 refreshTransactionId = includesMandatory
            ? queuedMandatoryRefreshTransactionId_ : 0;
        mandatoryWorkpieceRefresh_ = false;
        queuedMandatoryRefreshTransactionId_ = 0;
        deferredUserWorkpieceRefresh_ = false;
        issuePageCommand({includesMandatory ? CommandOwner::System
                                            : CommandOwner::UserRefresh,
                          QStringLiteral("list_workpieces"), QJsonObject(),
                          refreshTransactionId},
                         includesMandatory, includesUser);
        return;
    }
    if (hasLatestDetailsIntent_) {
        const QJsonObject fields = latestDetailsFields_;
        const quint64 refreshTransactionId = latestDetailsRefreshTransactionId_;
        hasLatestDetailsIntent_ = false;
        latestDetailsFields_ = QJsonObject();
        latestDetailsRefreshTransactionId_ = 0;
        const QString workpieceId = fields.value(QStringLiteral("workpiece_id")).toString();
        if (workpieceLibraryPage_ != nullptr
            && workpieceId == workpieceLibraryPage_->browsedWorkpieceId()) {
            issuePageCommand({CommandOwner::Library,
                              QStringLiteral("get_workpiece_details"), fields,
                              refreshTransactionId});
            return;
        }
    }
    if (!queuedMutationCommands_.isEmpty()) {
        issuePageCommand(queuedMutationCommands_.dequeue());
        return;
    }
    if (!queuedInternalCommands_.isEmpty()) {
        issuePageCommand(queuedInternalCommands_.dequeue());
    }
}

bool MainWindow::dispatchGeometryWorkflowContinuation() {
    if (geometryWorkflowContinuationCommand_.isEmpty() || client_ == nullptr
        || clientBusy_ || !pendingCommand_.isEmpty()
        || client_->state() != BackendClient::State::Ready) {
        return false;
    }
    const QString command = geometryWorkflowContinuationCommand_;
    const QJsonObject fields = geometryWorkflowContinuationFields_;
    if (command == QStringLiteral("validate_geometry_mask_draft")) {
        bindGeometryValidationContext(
            fields.value(QStringLiteral("workpiece_id")).toString(),
            fields.value(QStringLiteral("base_library_revision")).toInt(-1),
            fields.value(QStringLiteral("base_draft_revision")).toInt(-1));
    }
    issuePageCommand({CommandOwner::Geometry, command, fields});
    geometryWorkflowContinuationCommand_.clear();
    geometryWorkflowContinuationFields_ = QJsonObject();
    return true;
}

void MainWindow::stageGeometryWorkflowContinuation(const QString &command,
                                                   const QJsonObject &fields) {
    if (!geometryWorkflowContinuationCommand_.isEmpty()) return;
    geometryWorkflowContinuationCommand_ = command;
    geometryWorkflowContinuationFields_ = fields;
}

void MainWindow::clearGeometryWorkflowTarget() {
    geometryWorkflowWorkpieceId_.clear();
    geometryWorkflowLibraryRevision_ = -1;
    geometryWorkflowDraftRevision_ = -1;
    geometryWorkflowJobId_.clear();
    geometryWorkflowContinuationCommand_.clear();
    geometryWorkflowContinuationFields_ = QJsonObject();
    updateGeometryEditingLock();
}

void MainWindow::setGeometryOperationEditingLocked(bool locked) {
    geometryOperationEditingLocked_ = locked;
    updateGeometryEditingLock();
}

void MainWindow::setGeometryProfileLoadGeneration(quint64 generation) {
    geometryProfileLoadGeneration_ = generation;
    updateGeometryEditingLock();
}

void MainWindow::updateGeometryEditingLock() {
    if (geometryRulesPage_ == nullptr) return;
    geometryRulesPage_->setEditingLocked(
        geometryOperationEditingLocked_
        || geometryProfileLoadGeneration_ != 0
        || !geometryWorkflowWorkpieceId_.isEmpty());
}

void MainWindow::issuePageCommand(const QueuedCommandIntent &intent,
                                  bool includesMandatoryRefresh,
                                  bool includesUserRefresh) {
    pendingOwner_ = intent.owner;
    pendingCommand_ = intent.command;
    pendingFields_ = intent.fields;
    pendingRefreshTransactionId_ = intent.refreshTransactionId;
    pendingGeometryTargetGeneration_ = intent.geometryTargetGeneration;
    pendingRefreshIncludesMandatory_ = includesMandatoryRefresh;
    pendingRefreshIncludesUser_ = includesUserRefresh;
    client_->sendRequest(intent.command, intent.fields);
}

void MainWindow::clearPendingCommand() {
    pendingOwner_ = CommandOwner::None;
    pendingCommand_.clear();
    pendingFields_ = QJsonObject();
    pendingRefreshTransactionId_ = 0;
    pendingGeometryTargetGeneration_ = 0;
    pendingRefreshIncludesMandatory_ = false;
    pendingRefreshIncludesUser_ = false;
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
    queuedMutationCommands_.clear();
    queuedInternalCommands_.clear();
    hasReplaceRegistrationContinuation_ = false;
    replaceRegistrationContinuationFields_ = QJsonObject();
    hasLatestDetailsIntent_ = false;
    latestDetailsFields_ = QJsonObject();
    latestDetailsRefreshTransactionId_ = 0;
    mandatoryWorkpieceRefresh_ = false;
    mandatoryRefreshRetryRequired_ = false;
    mandatoryDetailsRetryRequired_ = false;
    activeMandatoryDetailsWorkpieceId_.clear();
    activeMandatoryRefreshTransactionId_ = 0;
    queuedMandatoryRefreshTransactionId_ = 0;
    deferredUserWorkpieceRefresh_ = false;
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
    if (mandatoryRefreshRetryRequired_ || mandatoryDetailsRetryRequired_) {
        mandatoryRefreshRetryRequired_ = false;
        mandatoryDetailsRetryRequired_ = false;
        activeMandatoryDetailsWorkpieceId_.clear();
        if (activeMandatoryRefreshTransactionId_ == 0) {
            activeMandatoryRefreshTransactionId_ = ++nextMandatoryRefreshTransactionId_;
        }
        mandatoryWorkpieceRefresh_ = true;
        queuedMandatoryRefreshTransactionId_ = activeMandatoryRefreshTransactionId_;
    }
    deferredUserWorkpieceRefresh_ = true;
    if (batchInFlight_ || clientBusy_ || !pendingCommand_.isEmpty()
        || client_->state() != BackendClient::State::Ready) {
        showLibraryMessage(QStringLiteral("当前操作完成后刷新工件列表…"));
    }
    dispatchDeferredUserRefresh();
}

void MainWindow::requestWorkpieceRefresh(bool preserveRegistrationSummary,
                                         CommandOwner owner) {
    if (client_ == nullptr || !backendReady_) {
        return;
    }
    quint64 refreshTransactionId = 0;
    if (owner != CommandOwner::UserRefresh) {
        refreshTransactionId = ++nextMandatoryRefreshTransactionId_;
        activeMandatoryRefreshTransactionId_ = refreshTransactionId;
        activeMandatoryDetailsWorkpieceId_.clear();
        mandatoryRefreshRetryRequired_ = false;
        mandatoryDetailsRetryRequired_ = false;
    }
    sendPageCommand(owner, QStringLiteral("list_workpieces"), QJsonObject(),
                    refreshTransactionId);
    if (!preserveRegistrationSummary) {
        showLibraryMessage(QStringLiteral("正在刷新工件列表…"));
    }
}

void MainWindow::dispatchDeferredUserRefresh() {
    if (batchInFlight_ || client_ == nullptr || !backendReady_) return;
    dispatchQueuedCommand();
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
    showGeometryRules();
}

void MainWindow::ensureGeometryProfileForCurrentWorkpiece() {
    const QString workpieceId = currentDetectionWorkpieceId_.isEmpty()
        ? appHeader_->currentWorkpieceId() : currentDetectionWorkpieceId_;
    if (workpieceId.isEmpty() || geometryRulesPage_ == nullptr
        || !geometryWorkflowWorkpieceId_.isEmpty()) {
        return;
    }
    if (geometryRulesPage_->hasUnsavedChanges()
        && !geometryForceProfileReload_) {
        return;
    }
    const QString snapshotWorkpieceId = geometryRulesPage_->snapshot()
        .value(QStringLiteral("workpiece_id")).toString();
    if (!geometryForceProfileReload_
        && geometryWorkpieceId_ == workpieceId
        && snapshotWorkpieceId == workpieceId) {
        return;
    }
    if (pendingCommand_ == QStringLiteral("get_geometry_mask_profile")
        && pendingFields_.value(QStringLiteral("workpiece_id")).toString()
            == workpieceId
        && pendingGeometryTargetGeneration_ == geometryTargetGeneration_) {
        return;
    }
    for (const QueuedCommandIntent &intent : queuedMutationCommands_) {
        if (intent.command == QStringLiteral("get_geometry_mask_profile")
            && intent.fields.value(QStringLiteral("workpiece_id")).toString()
                == workpieceId
            && intent.geometryTargetGeneration == geometryTargetGeneration_) {
            return;
        }
    }
    requestGeometryProfile(workpieceId);
}

void MainWindow::requestGeometryProfile(const QString &workpieceId) {
    if (workpieceId.isEmpty() || client_ == nullptr || !backendReady_) {
        return;
    }
    geometryRequestedWorkpieceId_ = workpieceId;
    setGeometryProfileLoadGeneration(geometryTargetGeneration_);
    if (geometryRulesPage_ != nullptr) geometryRulesPage_->setBusy(true);
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("get_geometry_mask_profile"), {
        {QStringLiteral("workpiece_id"), workpieceId},
    });
}

void MainWindow::saveGeometryDraft(const QJsonObject &draft, int libraryRevision, int draftRevision) {
    const GeometrySaveIntent intent = pendingNavigationKind_ != PendingNavigationKind::None
        ? GeometrySaveIntent::Navigation : GeometrySaveIntent::Normal;
    startGeometryDraftSave(draft, libraryRevision, draftRevision, intent);
}

bool MainWindow::startGeometryDraftSave(const QJsonObject &draft, int libraryRevision,
                                        int draftRevision, GeometrySaveIntent intent) {
    const QString workpieceId = intent == GeometrySaveIntent::PublishWorkflow
        ? geometryWorkflowWorkpieceId_ : geometryWorkpieceId_;
    if (workpieceId.isEmpty() || client_ == nullptr || !backendReady_
        || stagedGeometrySaveIntent_ != GeometrySaveIntent::None
        || geometrySaveIntent_ != GeometrySaveIntent::None
        || (client_->state() != BackendClient::State::Ready
            && client_->state() != BackendClient::State::Busy)) {
        return false;
    }
    stagedGeometrySaveIntent_ = intent;
    stagedGeometrySaveWorkpieceId_ = workpieceId;
    stagedGeometrySaveDraft_ = draft;
    stagedGeometrySaveLibraryRevision_ = libraryRevision;
    stagedGeometrySaveDraftRevision_ = draftRevision;
    setGeometryOperationEditingLocked(true);
    dispatchStagedGeometryDraftSave();
    return true;
}

bool MainWindow::dispatchStagedGeometryDraftSave() {
    if (stagedGeometrySaveIntent_ == GeometrySaveIntent::None
        || client_ == nullptr || clientBusy_ || !pendingCommand_.isEmpty()
        || client_->state() != BackendClient::State::Ready) {
        return false;
    }
    const GeometrySaveIntent intent = stagedGeometrySaveIntent_;
    const QJsonObject fields{
        {QStringLiteral("workpiece_id"), stagedGeometrySaveWorkpieceId_},
        {QStringLiteral("base_library_revision"), stagedGeometrySaveLibraryRevision_},
        {QStringLiteral("base_draft_revision"), stagedGeometrySaveDraftRevision_},
        {QStringLiteral("draft"), stagedGeometrySaveDraft_},
        {QStringLiteral("operation_id"),
         QUuid::createUuid().toString(QUuid::WithoutBraces)},
    };
    geometrySaveIntent_ = intent;
    if (geometryRulesPage_ != nullptr) geometryRulesPage_->setBusy(true);
    issuePageCommand({CommandOwner::Geometry,
                      QStringLiteral("save_geometry_mask_draft"), fields});
    if (pendingCommand_ != QStringLiteral("save_geometry_mask_draft")) {
        geometrySaveIntent_ = GeometrySaveIntent::None;
        if (geometryRulesPage_ != nullptr) geometryRulesPage_->setBusy(false);
        return false;
    }
    clearStagedGeometryDraftSave();
    return true;
}

void MainWindow::clearStagedGeometryDraftSave() {
    stagedGeometrySaveIntent_ = GeometrySaveIntent::None;
    stagedGeometrySaveWorkpieceId_.clear();
    stagedGeometrySaveDraft_ = QJsonObject();
    stagedGeometrySaveLibraryRevision_ = -1;
    stagedGeometrySaveDraftRevision_ = -1;
}

void MainWindow::publishGeometryWorkflow(const QJsonObject &draft, int libraryRevision, int draftRevision,
                                         const QString &overrideReason) {
    Q_UNUSED(overrideReason);
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || !backendReady_
        || !geometryWorkflowWorkpieceId_.isEmpty()
        || stagedGeometrySaveIntent_ != GeometrySaveIntent::None
        || (client_->state() != BackendClient::State::Ready
            && client_->state() != BackendClient::State::Busy)) return;
    geometryWorkflowWorkpieceId_ = geometryWorkpieceId_;
    geometryWorkflowLibraryRevision_ = libraryRevision;
    geometryWorkflowDraftRevision_ = draftRevision;
    geometryWorkflowJobId_.clear();
    geometryPublishAfterValidation_ = false;
    geometryPublishOverrideReason_.clear();
    if (!startGeometryDraftSave(draft, libraryRevision, draftRevision,
                                GeometrySaveIntent::PublishWorkflow)) {
        geometryPublishOverrideReason_.clear();
        clearGeometryWorkflowTarget();
    }
}

void MainWindow::validateGeometryDraft(int libraryRevision, int draftRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    bindGeometryValidationContext(geometryWorkpieceId_, libraryRevision,
                                  draftRevision);
    if (geometryRulesPage_ != nullptr) {
        geometryRulesPage_->clearPublishContinuation();
        geometryRulesPage_->setBusy(true);
    }
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

void MainWindow::applyGeometryValidationJob(const QJsonObject &job) {
    const QString state = job.value(QStringLiteral("state")).toString();
    const bool active = state == QStringLiteral("queued")
        || state == QStringLiteral("running");
    const QString jobId = job.value(QStringLiteral("job_id")).toString();
    const QJsonObject progress = job.value(QStringLiteral("progress")).toObject();
    const int total = qMax(1, progress.value(QStringLiteral("total")).toInt(1));
    const int completed = qBound(0, progress.value(QStringLiteral("completed")).toInt(), total);
    const qint64 elapsedMs = qMax<qint64>(
        0, progress.value(QStringLiteral("elapsed_ms"))
               .toVariant().toLongLong());

    if (!jobId.isEmpty() && !geometryValidationContextWorkpieceId_.isEmpty()
        && (geometryValidationContextJobId_.isEmpty()
            || geometryValidationContextJobId_ == jobId)) {
        geometryValidationContextJobId_ = jobId;
    }
    if (geometryRulesPage_ != nullptr
        && geometryValidationMatchesPage(job)) {
        geometryRulesPage_->setValidationJob(job);
        geometryRulesPage_->setBusy(false);
    }
    if (globalTaskStatus_ != nullptr) {
        globalTaskStatus_->setRunning(
            QStringLiteral("几何规则验证"),
            state == QStringLiteral("queued") ? QStringLiteral("排队中")
                                               : QStringLiteral("验证中"),
            completed, total, elapsedMs);
    }

    if (active) {
        if (!jobId.isEmpty()) geometryValidationJobId_ = jobId;
        if (geometryPollTimer_ != nullptr) geometryPollTimer_->start();
        return;
    }

    geometryValidationJobId_.clear();
    if (geometryPollTimer_ != nullptr) geometryPollTimer_->stop();
    if (globalTaskStatus_ == nullptr) return;
    if (state == QStringLiteral("completed")) {
        globalTaskStatus_->setMessage(
            TaskStatusWidget::MessageKind::Success,
            QStringLiteral("几何规则验证完成 · %1/%2").arg(completed).arg(total));
    } else if (state == QStringLiteral("cancelled")
               || state == QStringLiteral("canceled")) {
        globalTaskStatus_->setMessage(
            TaskStatusWidget::MessageKind::Warning,
            QStringLiteral("几何规则验证已取消 · %1/%2").arg(completed).arg(total));
    } else {
        globalTaskStatus_->setMessage(
            TaskStatusWidget::MessageKind::Error,
            QStringLiteral("几何规则验证未完成 · %1/%2").arg(completed).arg(total));
    }
}

void MainWindow::bindGeometryValidationContext(const QString &workpieceId,
                                               int libraryRevision,
                                               int draftRevision) {
    geometryValidationContextWorkpieceId_ = workpieceId;
    geometryValidationContextLibraryRevision_ = libraryRevision;
    geometryValidationContextDraftRevision_ = draftRevision;
    geometryValidationContextTargetGeneration_ = geometryTargetGeneration_;
    geometryValidationContextJobId_.clear();
    geometryValidationJobId_.clear();
    if (geometryPollTimer_ != nullptr) geometryPollTimer_->stop();
}

void MainWindow::clearGeometryValidationContext() {
    geometryValidationContextWorkpieceId_.clear();
    geometryValidationContextLibraryRevision_ = -1;
    geometryValidationContextDraftRevision_ = -1;
    geometryValidationContextTargetGeneration_ = 0;
    geometryValidationContextJobId_.clear();
}

bool MainWindow::geometryValidationMatchesPage(const QJsonObject &job) const {
    if (geometryValidationContextWorkpieceId_.isEmpty()) return true;
    if (geometryRulesPage_ == nullptr
        || geometryWorkpieceId_ != geometryValidationContextWorkpieceId_) {
        return false;
    }
    const bool contextGenerationIsCurrent =
        geometryValidationContextTargetGeneration_ == geometryTargetGeneration_;
    const bool activeWorkflowOwnsContext =
        geometryWorkflowWorkpieceId_ == geometryValidationContextWorkpieceId_;
    if (!contextGenerationIsCurrent && !activeWorkflowOwnsContext) return false;
    const QJsonObject snapshot = geometryRulesPage_->snapshot();
    if (snapshot.value(QStringLiteral("workpiece_id")).toString()
            != geometryValidationContextWorkpieceId_
        || snapshot.value(QStringLiteral("library_revision")).toInt(-1)
            != geometryValidationContextLibraryRevision_
        || snapshot.value(QStringLiteral("draft_revision")).toInt(-1)
            != geometryValidationContextDraftRevision_) {
        return false;
    }
    const QString jobId = job.value(QStringLiteral("job_id")).toString();
    return (geometryValidationContextJobId_.isEmpty() || jobId.isEmpty()
            || jobId == geometryValidationContextJobId_)
        && job.value(QStringLiteral("base_library_revision")).toInt(-1)
            == geometryValidationContextLibraryRevision_
        && job.value(QStringLiteral("base_draft_revision")).toInt(-1)
            == geometryValidationContextDraftRevision_;
}

bool MainWindow::maybeContinueGeometryPublish(const QJsonObject &job) {
    if (!geometryPublishAfterValidation_) return false;
    const QString state = job.value(QStringLiteral("state")).toString();
    const QString jobId = job.value(QStringLiteral("job_id")).toString();
    if (!jobId.isEmpty()) geometryWorkflowJobId_ = jobId;
    if (state == QStringLiteral("queued") || state == QStringLiteral("running")) return false;

    geometryPublishAfterValidation_ = false;
    const QJsonArray blocking = job.value(QStringLiteral("blocking_issues")).toArray();
    const QJsonArray warnings = job.value(QStringLiteral("warnings")).toArray();
    const bool hasRegression = job.value(QStringLiteral("regression")).toObject()
                                   .value(QStringLiteral("correct_to_wrong")).toInt(0) > 0;
    if (state != QStringLiteral("completed")) {
        setGeometryOperationEditingLocked(false);
        showLibraryMessage(QStringLiteral("几何规则验证未完成，未自动发布"), true);
        geometryPublishOverrideReason_.clear();
        clearGeometryWorkflowTarget();
        ensureGeometryProfileForCurrentWorkpiece();
        return true;
    }
    if (!blocking.isEmpty()) {
        setGeometryOperationEditingLocked(false);
        showLibraryMessage(QStringLiteral("验证存在阻断项，请在右侧模板表中逐项处理后再发布"), true);
        geometryPublishOverrideReason_.clear();
        clearGeometryWorkflowTarget();
        ensureGeometryProfileForCurrentWorkpiece();
        return true;
    }
    if ((!warnings.isEmpty() || hasRegression) && geometryPublishOverrideReason_.isEmpty()) {
        showLibraryMessage(QStringLiteral("验证有告警，请填写发布覆盖原因后点击发布规则"), true);
        return true;
    }

    const QString reason = geometryPublishOverrideReason_;
    geometryPublishOverrideReason_.clear();
    stageGeometryWorkflowContinuation(
        QStringLiteral("publish_geometry_mask_profile"),
        {{QStringLiteral("workpiece_id"), geometryWorkflowWorkpieceId_},
         {QStringLiteral("job_id"), geometryWorkflowJobId_},
         {QStringLiteral("base_library_revision"), geometryWorkflowLibraryRevision_},
         {QStringLiteral("base_draft_revision"), geometryWorkflowDraftRevision_},
         {QStringLiteral("override_reason"), reason},
         {QStringLiteral("operation_id"),
          QUuid::createUuid().toString(QUuid::WithoutBraces)}});
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
    const bool continuingWorkflow = !geometryWorkflowWorkpieceId_.isEmpty();
    const bool boundValidation = !continuingWorkflow
        && !geometryValidationContextWorkpieceId_.isEmpty()
        && jobId == geometryValidationContextJobId_
        && libraryRevision == geometryValidationContextLibraryRevision_
        && draftRevision == geometryValidationContextDraftRevision_;
    const QString workpieceId = continuingWorkflow
        ? geometryWorkflowWorkpieceId_
        : boundValidation ? geometryValidationContextWorkpieceId_
                          : geometryWorkpieceId_;
    if (continuingWorkflow
        && (jobId != geometryWorkflowJobId_
            || libraryRevision != geometryWorkflowLibraryRevision_
            || draftRevision != geometryWorkflowDraftRevision_)) {
        return;
    }
    if (workpieceId.isEmpty() || client_ == nullptr) return;
    geometryPublishAfterValidation_ = false;
    geometryPublishOverrideReason_.clear();
    if (geometryRulesPage_ != nullptr) {
        setGeometryOperationEditingLocked(true);
        geometryRulesPage_->setBusy(true);
    }
    const QJsonObject fields{
        {QStringLiteral("workpiece_id"), workpieceId},
        {QStringLiteral("job_id"), jobId},
        {QStringLiteral("base_library_revision"), libraryRevision},
        {QStringLiteral("base_draft_revision"), draftRevision},
        {QStringLiteral("override_reason"), overrideReason},
        {QStringLiteral("operation_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
    };
    if (continuingWorkflow) {
        stageGeometryWorkflowContinuation(
            QStringLiteral("publish_geometry_mask_profile"), fields);
        dispatchGeometryWorkflowContinuation();
        return;
    }
    if (clientBusy_ || client_->state() != BackendClient::State::Ready) {
        setGeometryOperationEditingLocked(false);
        if (geometryRulesPage_ != nullptr) geometryRulesPage_->setBusy(false);
        return;
    }
    sendPageCommand(CommandOwner::Geometry,
                    QStringLiteral("publish_geometry_mask_profile"), fields);
}

void MainWindow::rollbackGeometryProfile(int libraryRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    if (geometryRulesPage_ != nullptr) {
        setGeometryOperationEditingLocked(true);
        geometryRulesPage_->setBusy(true);
    }
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
    if (geometryRulesPage_ != nullptr) geometryRulesPage_->setPreviewBusy(true);
    sendPageCommand(CommandOwner::Geometry, QStringLiteral("preview_geometry_mask_rule"), fields);
}

void MainWindow::resolveGeometryMigration(const QString &conflictId, const QJsonObject &resolution,
                                          int libraryRevision, int draftRevision) {
    if (geometryWorkpieceId_.isEmpty() || client_ == nullptr || clientBusy_
        || client_->state() != BackendClient::State::Ready) return;
    if (geometryRulesPage_ != nullptr) {
        setGeometryOperationEditingLocked(true);
        geometryRulesPage_->setBusy(true);
    }
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
    if (geometryRulesPage_ != nullptr) {
        geometryRulesPage_->setBackendAvailable(true, QString());
        geometryRulesPage_->setBusy(false);
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
    if (geometryPollTimer_ != nullptr && !geometryValidationJobId_.isEmpty()) {
        geometryPollTimer_->start();
    }
    if (currentPage_ == AppPage::GeometryRules) {
        ensureGeometryProfileForCurrentWorkpiece();
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
    if (geometryRulesPage_ != nullptr) {
        geometryRulesPage_->setBackendAvailable(false,
            message.isEmpty() ? QStringLiteral("后端模型加载中") : message);
        geometryRulesPage_->setBusy(true);
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
    const quint64 interruptedRefreshTransactionId = pendingRefreshTransactionId_;
    const bool interruptedMandatoryRefresh = pendingRefreshIncludesMandatory_;
    const bool interruptedUserRefresh = pendingRefreshIncludesUser_;
    const bool recoverableGeometryPublishChain = geometryPublishAfterValidation_
        && !geometryValidationJobId_.isEmpty();
    const bool hadGeometryPublishWorkflow = geometryPublishAfterValidation_
        || !geometryWorkflowWorkpieceId_.isEmpty();
    const bool interruptedGeometryProfileMutation = interruptedOwner == CommandOwner::Geometry
        && (interruptedTask == QStringLiteral("save_geometry_mask_draft")
            || interruptedTask == QStringLiteral("publish_geometry_mask_profile")
            || interruptedTask == QStringLiteral("rollback_geometry_mask_profile")
            || interruptedTask == QStringLiteral("resolve_geometry_mask_migration"));
    if (interruptedGeometryProfileMutation) {
        geometryForceProfileReload_ = true;
    }
    if (interruptedTask == QStringLiteral("publish_geometry_mask_profile")) {
        clearGeometryValidationContext();
    }
    const bool hasPreservedWork = workpieceLibraryPage_->hasUnsavedChanges()
        || batchInFlight_ || clientBusy_
        || !pendingCommand_.isEmpty() || !inspectionImagePath_.isEmpty()
        || (batchResultsTableWidget_ != nullptr && batchResultsTableWidget_->rowCount() > 0)
        || annotationManagerDialog_ != nullptr
        || (geometryRulesPage_ != nullptr && geometryRulesPage_->hasUnsavedChanges())
        || !geometryValidationJobId_.isEmpty() || geometryPublishAfterValidation_
        || pendingNavigationKind_ != PendingNavigationKind::None
        || geometrySaveIntent_ != GeometrySaveIntent::None;
    if (interruptedTask == QStringLiteral("list_workpieces")) {
        mandatoryWorkpieceRefresh_ = mandatoryWorkpieceRefresh_
            || interruptedMandatoryRefresh;
        if (interruptedMandatoryRefresh) {
            queuedMandatoryRefreshTransactionId_ = interruptedRefreshTransactionId;
        }
        deferredUserWorkpieceRefresh_ = deferredUserWorkpieceRefresh_
            || interruptedUserRefresh || interruptedOwner == CommandOwner::UserRefresh;
    } else if (interruptedTask == QStringLiteral("get_workpiece_details")
               && interruptedRefreshTransactionId != 0
               && interruptedRefreshTransactionId
                      == activeMandatoryRefreshTransactionId_) {
        mandatoryWorkpieceRefresh_ = true;
        queuedMandatoryRefreshTransactionId_ = interruptedRefreshTransactionId;
        mandatoryDetailsRetryRequired_ = false;
        activeMandatoryDetailsWorkpieceId_.clear();
    }
    if (annotationManagerDialog_ != nullptr) {
        annotationManagerDialog_->setBusy(false);
        annotationManagerDialog_->setOperationError(
            QStringLiteral("后端连接中断，操作结果未知；请重连后刷新"));
    }
    cancelPendingGeometryNavigation();
    geometrySaveIntent_ = GeometrySaveIntent::None;
    clearStagedGeometryDraftSave();
    geometryWorkflowContinuationCommand_.clear();
    geometryWorkflowContinuationFields_ = QJsonObject();
    setGeometryProfileLoadGeneration(0);
    geometryRequestedWorkpieceId_.clear();
    if (hadGeometryPublishWorkflow && !recoverableGeometryPublishChain) {
        geometryPublishAfterValidation_ = false;
        geometryPublishOverrideReason_.clear();
        if (geometryRulesPage_ != nullptr) {
            geometryRulesPage_->clearPublishContinuation();
        }
    }
    if (!recoverableGeometryPublishChain) clearGeometryWorkflowTarget();
    if (geometryRulesPage_ != nullptr) {
        geometryRulesPage_->setBackendAvailable(false, reason);
        geometryRulesPage_->setBusy(false);
        geometryRulesPage_->setPreviewBusy(false);
        setGeometryOperationEditingLocked(recoverableGeometryPublishChain);
        geometryRulesPage_->setOperationError(
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
    queuedMutationCommands_.clear();
    queuedInternalCommands_.clear();
    hasReplaceRegistrationContinuation_ = false;
    replaceRegistrationContinuationFields_ = QJsonObject();
    hasLatestDetailsIntent_ = false;
    latestDetailsFields_ = QJsonObject();
    latestDetailsRefreshTransactionId_ = 0;
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
    if (state == BackendClient::State::Ready
        && dispatchStagedGeometryDraftSave()) {
        return;
    }
    if (state == BackendClient::State::Ready
        && dispatchGeometryWorkflowContinuation()) {
        return;
    }
    if (state == BackendClient::State::Ready
        && pendingNavigationKind_ != PendingNavigationKind::None) {
        tryStartPendingGeometrySave();
    }
    if (state == BackendClient::State::Ready && backendReadyHandled_) {
        dispatchQueuedCommand();
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
    const quint64 responseGeometryTargetGeneration = pendingGeometryTargetGeneration_;
    const quint64 responseRefreshTransactionId = pendingRefreshTransactionId_;
    const bool responseIncludedMandatoryRefresh = pendingRefreshIncludesMandatory_;
    clearPendingCommand();
    if (batchInFlight_ && command == QStringLiteral("predict")) {
        inspectionPage_->handleBackendResponse(command, response);
        return;
    }
    if (command == QStringLiteral("list_workpieces")) {
        if (pendingNavigationKind_ != PendingNavigationKind::None) {
            pendingNavigationWorkpieceListResponse_ = response;
            pendingNavigationRefreshTransactionId_ = responseRefreshTransactionId;
            pendingNavigationResponseIncludedMandatory_ =
                responseIncludedMandatoryRefresh;
            return;
        }
        applyWorkpieceListResponse(response, responseRefreshTransactionId,
                                   responseIncludedMandatoryRefresh);
        return;
    }
    if (command == QStringLiteral("get_workpiece_details")) {
        const QString requestedId = issuedFields.value(QStringLiteral("workpiece_id")).toString();
        const QString browsedId = workpieceLibraryPage_->browsedWorkpieceId();
        const QString responseId = response.value(QStringLiteral("workpiece")).toObject()
            .value(QStringLiteral("id")).toString();
        const bool matchesCurrentBrowse = requestedId.isEmpty() || requestedId == browsedId;
        if (matchesCurrentBrowse) {
            workpieceLibraryPage_->handleBackendResponse(command, response);
        }
        if (responseRefreshTransactionId != 0
            && responseRefreshTransactionId == activeMandatoryRefreshTransactionId_
            && !requestedId.isEmpty() && requestedId == browsedId
            && responseId == requestedId) {
            mandatoryDetailsRetryRequired_ = false;
            activeMandatoryDetailsWorkpieceId_.clear();
            activeMandatoryRefreshTransactionId_ = 0;
        }
        return;
    }
    if (command == QStringLiteral("recycle_workpiece")) {
        const QString recycledWorkpieceId = issuedFields.value(
            QStringLiteral("workpiece_id")).toString();
        if (!recycledWorkpieceId.isEmpty()
            && recycledWorkpieceId == lastPredictionWorkpieceId_) {
            lastPredictionWorkpieceId_.clear();
            lastPredictionImagePath_.clear();
            lastPredictionResponse_ = QJsonObject();
        }
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
        const QJsonObject profile = response.value(QStringLiteral("profile")).toObject();
        const QString issuedWorkpieceId = issuedFields
            .value(QStringLiteral("workpiece_id")).toString();
        const QString profileWorkpieceId = profile
            .value(QStringLiteral("workpiece_id")).toString();
        const bool currentResponse = responseGeometryTargetGeneration != 0
            && responseGeometryTargetGeneration == geometryTargetGeneration_
            && issuedWorkpieceId == currentDetectionWorkpieceId_
            && profileWorkpieceId == issuedWorkpieceId;
        if (!currentResponse) {
            ensureGeometryProfileForCurrentWorkpiece();
            return;
        }
        const bool reconcilingUnknownMutation = geometryForceProfileReload_;
        const bool preserveDirtyDraft = reconcilingUnknownMutation
            && geometryRulesPage_ != nullptr
            && geometryRulesPage_->hasUnsavedChanges()
            && geometryRulesPage_->draft()
                != profile.value(QStringLiteral("draft")).toObject();
        if (geometryRulesPage_ != nullptr) {
            if (!preserveDirtyDraft) {
                geometryRulesPage_->setSnapshot(profile);
            } else {
                geometryRulesPage_->reconcileSnapshotKeepingDraft(profile);
            }
            geometryRulesPage_->setBusy(false);
        }
        geometryForceProfileReload_ = false;
        geometryWorkpieceId_ = profileWorkpieceId;
        if (geometryRequestedWorkpieceId_ == profileWorkpieceId) {
            geometryRequestedWorkpieceId_.clear();
        }
        if (geometryProfileLoadGeneration_ == responseGeometryTargetGeneration) {
            setGeometryProfileLoadGeneration(0);
        }
        if (preserveDirtyDraft && geometryRulesPage_ != nullptr) {
            const QString message = QStringLiteral(
                "上次操作结果未知，远端草稿与本地修改不一致；已保留本地草稿，请处理冲突后再保存");
            geometryRulesPage_->setOperationError(message);
            showLibraryMessage(message, true);
        } else {
            showLibraryMessage(reconcilingUnknownMutation
                                   ? QStringLiteral("已重新加载几何规则并完成结果对账")
                                   : QStringLiteral("几何干扰规则已加载"));
        }
        return;
    }
    if (command == QStringLiteral("preview_geometry_mask_rule")) {
        if (geometryRulesPage_ != nullptr) {
            geometryRulesPage_->setRulePreview(response.value(QStringLiteral("preview")).toObject());
            geometryRulesPage_->setPreviewBusy(false);
        }
        showLibraryMessage(QStringLiteral("几何规则预览已更新"));
        return;
    }
    if (command == QStringLiteral("save_geometry_mask_draft")) {
        const GeometrySaveIntent saveIntent = geometrySaveIntent_;
        geometrySaveIntent_ = GeometrySaveIntent::None;
        const bool continuePublishWorkflow = saveIntent == GeometrySaveIntent::PublishWorkflow;
        if (continuePublishWorkflow) geometryPublishAfterValidation_ = true;
        if (geometryRulesPage_ != nullptr) {
            geometryRulesPage_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryRulesPage_->setBusy(false);
            if (saveIntent == GeometrySaveIntent::Normal) {
                setGeometryOperationEditingLocked(false);
            }
        }
        const QJsonObject savedProfile = response.value(QStringLiteral("profile")).toObject();
        showLibraryMessage(continuePublishWorkflow
                               ? QStringLiteral("草稿已保存，正在验证后发布…")
                               : QStringLiteral("几何规则草稿已保存"));
        if (saveIntent == GeometrySaveIntent::Navigation) {
            completePendingGeometryNavigation();
        } else if (continuePublishWorkflow) {
            const int libraryRevision = savedProfile.value(QStringLiteral("library_revision")).toInt();
            const int draftRevision = savedProfile.value(QStringLiteral("draft_revision")).toInt();
            geometryWorkflowLibraryRevision_ = libraryRevision;
            geometryWorkflowDraftRevision_ = draftRevision;
            stageGeometryWorkflowContinuation(
                QStringLiteral("validate_geometry_mask_draft"),
                {{QStringLiteral("workpiece_id"), geometryWorkflowWorkpieceId_},
                 {QStringLiteral("base_library_revision"), libraryRevision},
                 {QStringLiteral("base_draft_revision"), draftRevision},
                 {QStringLiteral("operation_id"),
                  QUuid::createUuid().toString(QUuid::WithoutBraces)}});
        }
        return;
    }
    if (command == QStringLiteral("resolve_geometry_mask_migration")) {
        if (geometryRulesPage_ != nullptr) {
            geometryRulesPage_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryRulesPage_->setBusy(false);
            setGeometryOperationEditingLocked(false);
        }
        showLibraryMessage(QStringLiteral("迁移冲突处置已保存"));
        return;
    }
    if (command == QStringLiteral("validate_geometry_mask_draft")) {
        const QJsonObject job = response.value(QStringLiteral("job")).toObject();
        applyGeometryValidationJob(job);
        const bool workflowHandled = maybeContinueGeometryPublish(job);
        if (!workflowHandled || job.value(QStringLiteral("state")).toString() == QStringLiteral("queued")
            || job.value(QStringLiteral("state")).toString() == QStringLiteral("running")) {
            showLibraryMessage(QStringLiteral("几何规则验证任务已启动"));
        }
        return;
    }
    if (command == QStringLiteral("get_geometry_mask_validation_job")) {
        const QJsonObject job = response.value(QStringLiteral("job")).toObject();
        applyGeometryValidationJob(job);
        const bool workflowHandled = maybeContinueGeometryPublish(job);
        Q_UNUSED(workflowHandled);
        return;
    }
    if (command == QStringLiteral("geometry_mask_validation_job_action")) {
        const QJsonObject job = response.value(QStringLiteral("job")).toObject();
        applyGeometryValidationJob(job);
        maybeContinueGeometryPublish(job);
        const QString action = issuedFields.value(QStringLiteral("action"))
                                   .toString().toLower();
        if (action == QStringLiteral("cancel")
            || action == QStringLiteral("cancelled")
            || action == QStringLiteral("canceled")) {
            clearGeometryValidationContext();
        }
        return;
    }
    if (command == QStringLiteral("publish_geometry_mask_profile")
        || command == QStringLiteral("rollback_geometry_mask_profile")) {
        if (geometryPollTimer_ != nullptr) geometryPollTimer_->stop();
        if (geometryRulesPage_ != nullptr) {
            geometryRulesPage_->setSnapshot(response.value(QStringLiteral("profile")).toObject());
            geometryRulesPage_->setBusy(false);
            setGeometryOperationEditingLocked(false);
        }
        showLibraryMessage(command == QStringLiteral("publish_geometry_mask_profile")
                              ? QStringLiteral("几何干扰规则已发布") : QStringLiteral("几何干扰规则已回退"));
        if (command == QStringLiteral("publish_geometry_mask_profile")) {
            clearGeometryValidationContext();
            clearGeometryWorkflowTarget();
            ensureGeometryProfileForCurrentWorkpiece();
        }
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
        requestWorkpieceRefresh(true);
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
    const quint64 failedGeometryTargetGeneration = pendingGeometryTargetGeneration_;
    const quint64 failedRefreshTransactionId = pendingRefreshTransactionId_;
    const bool failedListIncludedMandatoryRefresh = pendingRefreshIncludesMandatory_;
    const bool failedListIncludedUserRefresh = pendingRefreshIncludesUser_;
    clearPendingCommand();
    const QString failedCommand = issuedCommand.isEmpty() ? command : issuedCommand;
    if (failedCommand == QStringLiteral("save_geometry_mask_draft")) {
        const GeometrySaveIntent failedSaveIntent = geometrySaveIntent_;
        geometrySaveIntent_ = GeometrySaveIntent::None;
        if (failedSaveIntent == GeometrySaveIntent::Navigation) {
            cancelPendingGeometryNavigation();
        } else if (failedSaveIntent == GeometrySaveIntent::PublishWorkflow) {
            geometryPublishAfterValidation_ = false;
            geometryPublishOverrideReason_.clear();
        }
    }
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
        && failedListIncludedMandatoryRefresh) {
        mandatoryRefreshRetryRequired_ = true;
        showLibraryMessage(
            QStringLiteral("刷新失败：%1；当前列表可能已过期，请点击“刷新工件列表”重试")
                .arg(message),
            true);
    } else if (failedCommand == QStringLiteral("list_workpieces")
               && (failureOwner == CommandOwner::UserRefresh
                   || failedListIncludedUserRefresh)) {
        showLibraryMessage(QStringLiteral("刷新失败：%1；请手动重试").arg(message), true);
    } else if (failedCommand == QStringLiteral("get_workpiece_details")
               && matchesOwner(CommandOwner::Library)) {
        const QString requestedId = issuedFields.value(QStringLiteral("workpiece_id")).toString();
        if (requestedId.isEmpty()
            || requestedId == workpieceLibraryPage_->browsedWorkpieceId()) {
            workpieceLibraryPage_->handleBackendFailure(failedCommand, code, message);
            if (failedRefreshTransactionId != 0
                && failedRefreshTransactionId
                       == activeMandatoryRefreshTransactionId_
                && !requestedId.isEmpty()) {
                mandatoryDetailsRetryRequired_ = true;
                activeMandatoryDetailsWorkpieceId_ = requestedId;
                showLibraryMessage(
                    QStringLiteral("详情刷新失败：%1；请点击“刷新工件列表”重试")
                        .arg(message),
                    true);
            }
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
                    || failedCommand == QStringLiteral("geometry_mask_validation_job_action")
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
        const bool staleProfileFailure = failedCommand
                == QStringLiteral("get_geometry_mask_profile")
            && geometryProfileLoadGeneration_ != 0
            && failedGeometryTargetGeneration != geometryProfileLoadGeneration_;
        const bool failedActiveWorkflow = !geometryWorkflowWorkpieceId_.isEmpty();
        const bool validationFailure =
            failedCommand == QStringLiteral("validate_geometry_mask_draft")
            || failedCommand == QStringLiteral("get_geometry_mask_validation_job")
            || failedCommand == QStringLiteral("geometry_mask_validation_job_action");
        const bool publishFailure =
            failedCommand == QStringLiteral("publish_geometry_mask_profile");
        const bool failedWorkflowCommand = failedActiveWorkflow
            && (failedCommand == QStringLiteral("save_geometry_mask_draft")
                || validationFailure || publishFailure);
        const bool terminatesValidation = validationFailure || publishFailure
            || failedWorkflowCommand;
        if (geometryRulesPage_ != nullptr) {
            if (failedCommand == QStringLiteral("publish_geometry_mask_profile")) {
                geometryRulesPage_->clearPublishContinuation();
            }
            geometryRulesPage_->setBusy(false);
            geometryRulesPage_->setPreviewBusy(false);
            if (!staleProfileFailure) {
                if (failedCommand == QStringLiteral("get_geometry_mask_profile")) {
                    setGeometryProfileLoadGeneration(0);
                    geometryRequestedWorkpieceId_.clear();
                }
                setGeometryOperationEditingLocked(false);
            }
        }
        if (terminatesValidation) {
            geometryPublishAfterValidation_ = false;
            geometryPublishOverrideReason_.clear();
            if (failedActiveWorkflow) clearGeometryWorkflowTarget();
            clearGeometryValidationContext();
            geometryValidationJobId_.clear();
            if (geometryPollTimer_ != nullptr) geometryPollTimer_->stop();
        }
        if (geometryRulesPage_ != nullptr) geometryRulesPage_->setOperationError(detail);
        if (failedWorkflowCommand) ensureGeometryProfileForCurrentWorkpiece();
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
