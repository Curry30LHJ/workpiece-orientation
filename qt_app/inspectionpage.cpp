#include "inspectionpage.h"

#include "ui_inspectionpage.h"

#include <QApplication>
#include <QButtonGroup>
#include <QFileInfo>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QListWidgetItem>
#include <QPixmap>
#include <QPushButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QUuid>
#include <QVBoxLayout>
#include <QWheelEvent>

#include "inspectionimageview.h"

namespace {
QString geometryDescription(const QJsonObject &response) {
    const QString status = response.value(QStringLiteral("geometry_mask"))
                               .toObject().value(QStringLiteral("status")).toString();
    if (status == QStringLiteral("active")) return QStringLiteral("已应用");
    if (status.isEmpty()) return QStringLiteral("未提供");
    return QStringLiteral("未应用（%1）").arg(status);
}
}

InspectionPage::InspectionPage(QWidget *parent)
    : QWidget(parent), ui(new Ui::InspectionPage) {
    ui->setupUi(this);
    ui->inspectionSplitter->setSizes({650, 350});
    ui->batchReviewSplitter->setSizes({430, 230});
    ui->batchResultsTableWidget->setColumnCount(5);
    ui->batchResultsTableWidget->setHorizontalHeaderLabels({
        QStringLiteral("文件"), QStringLiteral("结果"), QStringLiteral("复检"),
        QStringLiteral("耗时（毫秒）"), QStringLiteral("处理状态")});
    ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(
        0, QHeaderView::Stretch);
    for (int column = 1; column < 5; ++column) {
        ui->batchResultsTableWidget->horizontalHeader()->setSectionResizeMode(
            column, QHeaderView::ResizeToContents);
    }
    ui->batchResultsTableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->batchResultsTableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->batchResultsTableWidget->setSelectionMode(QAbstractItemView::SingleSelection);
    auto *filterGroup = new QButtonGroup(this);
    filterGroup->setExclusive(true);
    filterGroup->addButton(ui->allBatchFilterButton);
    filterGroup->addButton(ui->needsReviewBatchFilterButton);
    filterGroup->addButton(ui->unprocessedBatchFilterButton);
    filterGroup->addButton(ui->failedBatchFilterButton);
    ui->rawEvidenceContainer->setChecked(false);
    ui->rawEvidenceTextEdit->setVisible(false);
    connect(ui->rawEvidenceContainer, &QGroupBox::toggled,
            ui->rawEvidenceTextEdit, &QWidget::setVisible);
    connect(ui->inspectionImageView, &InspectionImageView::imageDropped,
            this, [this](const QString &path) {
                setMode(InspectionMode::Single);
                setSingleImagePath(path);
            });
    connect(ui->predictButton, &QPushButton::clicked,
            this, &InspectionPage::requestPrediction);
    connect(ui->confirmFrontButton, &QPushButton::clicked,
            this, [this]() {
                requestConfirmation(QStringLiteral("front"));
            });
    connect(ui->confirmBackButton, &QPushButton::clicked,
            this, [this]() {
                requestConfirmation(QStringLiteral("back"));
            });
    connect(ui->rejectConfirmationButton, &QPushButton::clicked,
            this, &InspectionPage::rejectCurrentRecord);
    connect(ui->stopBatchButton, &QPushButton::clicked,
            this, &InspectionPage::requestBatchStop);
    connect(ui->allBatchFilterButton, &QPushButton::clicked,
            this, [this]() { setBatchFilter(BatchFilter::All); });
    connect(ui->needsReviewBatchFilterButton, &QPushButton::clicked,
            this, [this]() { setBatchFilter(BatchFilter::NeedsReview); });
    connect(ui->unprocessedBatchFilterButton, &QPushButton::clicked,
            this, [this]() { setBatchFilter(BatchFilter::Unprocessed); });
    connect(ui->failedBatchFilterButton, &QPushButton::clicked,
            this, [this]() { setBatchFilter(BatchFilter::Failed); });
    connect(ui->batchResultsTableWidget, &QTableWidget::currentCellChanged,
            this, [this](int row, int, int, int) {
                if (changingBatchSelection_ || row < 0) return;
                QTableWidgetItem *item = ui->batchResultsTableWidget->item(row, 0);
                if (item != nullptr) {
                    selectBatchRecord(item->data(Qt::UserRole).toString(), true);
                }
            });
    connect(ui->recentInspectionList, &QListWidget::itemActivated,
            this, [this](QListWidgetItem *item) {
                if (item != nullptr) selectRecentRecord(item->data(Qt::UserRole).toString());
            });
    connect(ui->recentInspectionList, &QListWidget::itemClicked,
            this, [this](QListWidgetItem *item) {
                if (item != nullptr) selectRecentRecord(item->data(Qt::UserRole).toString());
            });
    connect(ui->zoomInButton, &QPushButton::clicked, this, [this]() {
        QWheelEvent event(QPointF(), QPointF(), QPoint(), QPoint(0, 120),
                          Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(ui->inspectionImageView->viewport(), &event);
    });
    connect(ui->zoomOutButton, &QPushButton::clicked, this, [this]() {
        QWheelEvent event(QPointF(), QPointF(), QPoint(), QPoint(0, -120),
                          Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(ui->inspectionImageView->viewport(), &event);
    });
    connect(ui->resetViewButton, &QPushButton::clicked,
            ui->inspectionImageView, &InspectionImageView::resetView);
    ui->rejectConfirmationButton->setProperty("role", QStringLiteral("secondary"));
    ui->rejectConfirmationButton->setStyleSheet(
        QStringLiteral("background: transparent; border: 1px solid #6B7280;"));
    renderActiveState();
}

InspectionPage::~InspectionPage() {
    delete ui;
}

InspectionPage::ModeState &InspectionPage::activeState() {
    return mode_ == InspectionMode::Single ? singleState_ : batchState_;
}

const InspectionPage::ModeState &InspectionPage::activeState() const {
    return mode_ == InspectionMode::Single ? singleState_ : batchState_;
}

void InspectionPage::setMode(InspectionMode mode) {
    if (mode_ == mode) return;
    mode_ = mode;
    renderActiveState();
}

InspectionMode InspectionPage::mode() const {
    return mode_;
}

InspectionUiState InspectionPage::uiState() const {
    if (backendStatusKnown_ && !backendAvailable_) {
        return InspectionUiState::BackendUnavailable;
    }
    return activeState().uiState;
}

void InspectionPage::setCurrentWorkpiece(const QString &id, const QString &name) {
    currentWorkpieceId_ = id;
    currentWorkpieceName_ = name;
    renderActiveState();
}

void InspectionPage::setBackendAvailable(bool available, bool busy, const QString &reason) {
    backendStatusKnown_ = true;
    backendAvailable_ = available;
    backendBusy_ = busy;
    backendReason_ = reason;
    renderActiveState();
    if (available && !busy && batchRunning_ && currentBatchRequestId_.isEmpty()
        && !stopRequested_ && batchCursor_ < batchRecordOrder_.size()) {
        requestNextBatchPrediction();
    }
}

void InspectionPage::setSingleImagePath(const QString &path) {
    ModeState &state = activeState();
    state.imagePath = QFileInfo(path).absoluteFilePath();
    state.hasRecord = false;
    state.visibleRecord = InspectionRecord();
    state.uiState = InspectionUiState::Idle;
    state.message.clear();
    renderActiveState();
}

void InspectionPage::clearBatchState() {
    batchState_ = ModeState();
    batchRecords_.clear();
    batchRecordOrder_.clear();
    currentBatchRequestId_.clear();
    selectedRecordId_.clear();
    batchWorkpieceId_.clear();
    batchFilter_ = BatchFilter::All;
    batchCursor_ = 0;
    batchSelectionPinned_ = false;
    batchRunning_ = false;
    stopRequested_ = false;
    ui->batchResultsTableWidget->setRowCount(0);
    ui->batchSummaryLabel->clear();
    ui->batchProgressBar->setRange(0, 1);
    ui->batchProgressBar->setValue(0);
    ui->allBatchFilterButton->setChecked(true);
    if (mode_ == InspectionMode::Batch) {
        renderActiveState();
    }
}

void InspectionPage::beginBatch(const QStringList &paths, const QString &workpieceId) {
    QStringList normalized;
    normalized.reserve(paths.size());
    for (const QString &path : paths) {
        const QString absolute = QFileInfo(path).absoluteFilePath();
        QImageReader reader(absolute);
        if (!QFileInfo::exists(absolute) || !reader.canRead()) {
            clearBatchState();
            mode_ = InspectionMode::Batch;
            batchState_.uiState = InspectionUiState::Failed;
            batchState_.message = QStringLiteral("图片无法读取：%1")
                                      .arg(QFileInfo(absolute).fileName());
            renderActiveState();
            return;
        }
        normalized.append(absolute);
    }

    clearBatchState();
    mode_ = InspectionMode::Batch;
    batchWorkpieceId_ = workpieceId;
    for (const QString &path : normalized) {
        InspectionRecord record;
        record.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        record.imagePath = path;
        record.workpieceId = workpieceId;
        batchRecordOrder_.append(record.id);
        batchRecords_.insert(record.id, record);
    }
    batchState_.imagePath = normalized.isEmpty() ? QString() : normalized.constFirst();
    batchState_.uiState = normalized.isEmpty()
        ? InspectionUiState::Idle : InspectionUiState::Running;
    batchState_.message = normalized.isEmpty()
        ? QStringLiteral("批量列表为空") : QStringLiteral("正在批量检测…");
    batchRunning_ = !normalized.isEmpty();
    rebuildBatchTable();
    updateBatchSummary();
    renderActiveState();
    if (batchRunning_) requestNextBatchPrediction();
}

void InspectionPage::setBatchFilter(BatchFilter filter) {
    batchFilter_ = filter;
    ui->allBatchFilterButton->setChecked(filter == BatchFilter::All);
    ui->needsReviewBatchFilterButton->setChecked(filter == BatchFilter::NeedsReview);
    ui->unprocessedBatchFilterButton->setChecked(filter == BatchFilter::Unprocessed);
    ui->failedBatchFilterButton->setChecked(filter == BatchFilter::Failed);
    rebuildBatchTable();
}

BatchFilter InspectionPage::batchFilter() const {
    return batchFilter_;
}

QString InspectionPage::selectedRecordId() const {
    return mode_ == InspectionMode::Batch
        ? selectedRecordId_ : singleState_.visibleRecord.id;
}

QStringList InspectionPage::batchRecordIds() const {
    return batchRecordOrder_;
}

void InspectionPage::requestBatchStop() {
    if (!batchRunning_) return;
    stopRequested_ = true;
    if (currentBatchRequestId_.isEmpty()) finishBatch(true);
    updateActionAvailability();
}

int InspectionPage::completedBatchCount() const {
    int completed = 0;
    for (const QString &recordId : batchRecordOrder_) {
        if (batchRecords_.value(recordId).completedAt.isValid()) ++completed;
    }
    return completed;
}

int InspectionPage::failedBatchCount() const {
    int failed = 0;
    for (const QString &recordId : batchRecordOrder_) {
        const InspectionRecord record = batchRecords_.value(recordId);
        if (record.disposition == BatchDisposition::PredictionFailed
            || record.disposition == BatchDisposition::SubmitFailed
            || !record.error.isEmpty()) {
            ++failed;
        }
    }
    return failed;
}

void InspectionPage::setRecordDisposition(const QString &recordId,
                                           BatchDisposition disposition,
                                           const QString &evolutionJobId,
                                           const QString &error) {
    const bool resolvesPendingConfirmation = recordId == pendingConfirmationRecordId_
        && disposition != BatchDisposition::Submitting;
    if (batchRecords_.contains(recordId)) {
        InspectionRecord record = batchRecords_.value(recordId);
        record.disposition = disposition;
        if (!evolutionJobId.isEmpty()) record.evolutionJobId = evolutionJobId;
        record.error = error;
        record.submissionError = error;
        batchRecords_.insert(recordId, record);
        if (selectedRecordId_ == recordId) {
            batchState_.visibleRecord = record;
            batchState_.hasRecord = record.completedAt.isValid();
        }
        rebuildBatchTable();
        updateBatchSummary();
        if (selectedRecordId_ == recordId
            && (disposition == BatchDisposition::QueuedFront
                || disposition == BatchDisposition::QueuedBack
                || disposition == BatchDisposition::Rejected)) {
            selectPreferredBatchRecord(recordId);
        } else {
            renderActiveState();
        }
        if (resolvesPendingConfirmation) {
            pendingConfirmationRecordId_.clear();
            pendingConfirmationOrientation_.clear();
        }
        return;
    }
    updateRecordDisposition(recordId, disposition, error);
    if (recentRecords_.contains(recordId) && !evolutionJobId.isEmpty()) {
        InspectionRecord record = recentRecords_.value(recordId);
        record.evolutionJobId = evolutionJobId;
        recentRecords_.insert(recordId, record);
        if (singleState_.visibleRecord.id == recordId) singleState_.visibleRecord = record;
    }
    if (resolvesPendingConfirmation) {
        pendingConfirmationRecordId_.clear();
        pendingConfirmationOrientation_.clear();
    }
    renderActiveState();
}

QString InspectionPage::singleImagePath() const {
    return activeState().imagePath;
}

void InspectionPage::showSingleResult(const InspectionRecord &sourceRecord) {
    InspectionRecord record = sourceRecord;
    if (record.id.isEmpty()) {
        record.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    }
    record.imagePath = QFileInfo(record.imagePath).absoluteFilePath();
    if (record.label.isEmpty()) {
        record.label = record.response.value(QStringLiteral("label")).toString();
    }
    record.needsReview = record.needsReview
        || record.response.value(QStringLiteral("needs_review")).toBool();
    if (record.elapsedMs <= 0.0) {
        record.elapsedMs = record.response.value(QStringLiteral("elapsed_ms")).toDouble();
    }
    if (!record.completedAt.isValid()) {
        record.completedAt = QDateTime::currentDateTime();
    }
    ModeState &state = activeState();
    state.imagePath = record.imagePath;
    state.visibleRecord = record;
    state.hasRecord = true;
    state.uiState = record.error.isEmpty()
        ? (record.needsReview ? InspectionUiState::NeedsReview : InspectionUiState::Completed)
        : InspectionUiState::Failed;
    state.message = record.error;
    if (mode_ == InspectionMode::Single) {
        storeRecentRecord(record);
    }
    renderActiveState();
}

void InspectionPage::showSingleFailure(const QString &message) {
    ModeState &state = activeState();
    InspectionRecord record = state.visibleRecord;
    record.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    record.imagePath = state.imagePath;
    record.workpieceId = currentWorkpieceId_;
    record.completedAt = QDateTime::currentDateTime();
    record.error = message;
    record.disposition = BatchDisposition::PredictionFailed;
    state.visibleRecord = record;
    state.hasRecord = true;
    state.uiState = InspectionUiState::Failed;
    state.message = message;
    if (mode_ == InspectionMode::Single) storeRecentRecord(record);
    renderActiveState();
}

void InspectionPage::selectRecentRecord(const QString &recordId) {
    if (!recentRecords_.contains(recordId)) return;
    mode_ = InspectionMode::Single;
    const InspectionRecord record = recentRecords_.value(recordId);
    singleState_.imagePath = record.imagePath;
    singleState_.visibleRecord = record;
    singleState_.hasRecord = true;
    singleState_.uiState = record.error.isEmpty()
        ? (record.needsReview ? InspectionUiState::NeedsReview : InspectionUiState::Completed)
        : InspectionUiState::Failed;
    singleState_.message = record.error;
    renderActiveState();
}

void InspectionPage::handleBackendResponse(const QString &command,
                                           const QJsonObject &response) {
    if (command == QStringLiteral("predict")) {
        if (mode_ == InspectionMode::Batch && !currentBatchRequestId_.isEmpty()) {
            InspectionRecord *record = batchRecord(currentBatchRequestId_);
            if (record == nullptr) return;
            record->response = response;
            record->label = response.value(QStringLiteral("label")).toString();
            record->needsReview = response.value(QStringLiteral("needs_review")).toBool();
            record->elapsedMs = response.value(QStringLiteral("elapsed_ms")).toDouble();
            record->completedAt = QDateTime::currentDateTime();
            record->error.clear();
            const QString completedRecordId = currentBatchRequestId_;
            currentBatchRequestId_.clear();
            ++batchCursor_;
            if (selectedRecordId_ == completedRecordId) {
                batchState_.visibleRecord = *record;
                batchState_.hasRecord = true;
            }
            rebuildBatchTable();
            updateBatchSummary();
            renderActiveState();
            if (stopRequested_) {
                finishBatch(true);
            } else if (batchCursor_ >= batchRecordOrder_.size()) {
                finishBatch(false);
            } else {
                requestNextBatchPrediction();
            }
            return;
        }
        InspectionRecord record;
        record.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        record.imagePath = pendingPredictionImagePath_.isEmpty()
            ? activeState().imagePath : pendingPredictionImagePath_;
        record.workpieceId = pendingPredictionWorkpieceId_.isEmpty()
            ? currentWorkpieceId_ : pendingPredictionWorkpieceId_;
        record.response = response;
        record.label = response.value(QStringLiteral("label")).toString();
        record.needsReview = response.value(QStringLiteral("needs_review")).toBool();
        record.elapsedMs = response.value(QStringLiteral("elapsed_ms")).toDouble();
        record.completedAt = QDateTime::currentDateTime();
        pendingPredictionImagePath_.clear();
        pendingPredictionWorkpieceId_.clear();
        showSingleResult(record);
        return;
    }
    if (command == QStringLiteral("submit_confirmation")) {
        const QString recordId = pendingConfirmationRecordId_;
        if (recordId.isEmpty()) return;
        const QJsonObject job = response.value(QStringLiteral("job")).toObject();
        const QString jobId = job.value(QStringLiteral("job_id")).toString();
        const QString state = job.value(QStringLiteral("state")).toString(QStringLiteral("queued"));
        if (state == QStringLiteral("failed")) {
            setRecordDisposition(recordId, BatchDisposition::SubmitFailed, jobId,
                                 job.value(QStringLiteral("error")).toString());
        } else {
            setRecordDisposition(
                recordId,
                pendingConfirmationOrientation_ == QStringLiteral("back")
                    ? BatchDisposition::QueuedBack : BatchDisposition::QueuedFront,
                jobId);
        }
        pendingConfirmationRecordId_.clear();
        pendingConfirmationOrientation_.clear();
        return;
    }
    if (command == QStringLiteral("list_evolution_jobs")) {
        const QJsonArray jobs = response.value(QStringLiteral("jobs")).toArray();
        for (const QJsonValue &value : jobs) {
            const QJsonObject job = value.toObject();
            const QString jobId = job.value(QStringLiteral("job_id")).toString();
            for (const QString &recordId : recentRecordOrder_) {
                InspectionRecord record = recentRecords_.value(recordId);
                if (record.evolutionJobId != jobId) continue;
                record.response.insert(QStringLiteral("evolution_state"),
                                       job.value(QStringLiteral("state")));
                if (job.value(QStringLiteral("state")).toString() == QStringLiteral("failed")) {
                    record.disposition = BatchDisposition::SubmitFailed;
                    record.submissionError = job.value(QStringLiteral("error")).toString();
                }
                recentRecords_.insert(recordId, record);
                if (singleState_.visibleRecord.id == recordId) singleState_.visibleRecord = record;
            }
            for (const QString &recordId : batchRecordOrder_) {
                InspectionRecord record = batchRecords_.value(recordId);
                if (record.evolutionJobId != jobId) continue;
                const QString state = job.value(QStringLiteral("state")).toString();
                record.response.insert(QStringLiteral("evolution_state"), state);
                if (state == QStringLiteral("failed")) {
                    record.disposition = BatchDisposition::SubmitFailed;
                    record.error = job.value(QStringLiteral("error")).toString();
                    record.submissionError = record.error;
                }
                batchRecords_.insert(recordId, record);
                if (selectedRecordId_ == recordId) batchState_.visibleRecord = record;
            }
        }
        rebuildRecentList();
        rebuildBatchTable();
        updateBatchSummary();
        renderActiveState();
    }
}

void InspectionPage::handleBackendFailure(const QString &command, const QString &code,
                                          const QString &message) {
    if (command == QStringLiteral("submit_confirmation")
        && !pendingConfirmationRecordId_.isEmpty()) {
        setRecordDisposition(pendingConfirmationRecordId_,
                             BatchDisposition::SubmitFailed, QString(), message);
        pendingConfirmationRecordId_.clear();
        pendingConfirmationOrientation_.clear();
        renderActiveState();
        return;
    }
    if (command == QStringLiteral("predict")) {
        if (mode_ == InspectionMode::Batch && !currentBatchRequestId_.isEmpty()) {
            InspectionRecord *record = batchRecord(currentBatchRequestId_);
            if (record == nullptr) return;
            const bool transportFailure = code == QStringLiteral("CONNECTION_LOST")
                || code == QStringLiteral("TRANSPORT_ERROR");
            record->completedAt = QDateTime::currentDateTime();
            record->disposition = BatchDisposition::PredictionFailed;
            record->error = transportFailure
                ? QStringLiteral("结果未知：%1").arg(message) : message;
            const QString failedRecordId = currentBatchRequestId_;
            currentBatchRequestId_.clear();
            ++batchCursor_;
            if (selectedRecordId_ == failedRecordId) {
                batchState_.visibleRecord = *record;
                batchState_.hasRecord = true;
            }
            rebuildBatchTable();
            updateBatchSummary();
            renderActiveState();
            if (transportFailure) {
                backendStatusKnown_ = true;
                backendAvailable_ = false;
                backendReason_ = message;
                finishBatch(true);
            } else if (stopRequested_ || batchCursor_ >= batchRecordOrder_.size()) {
                finishBatch(stopRequested_);
            } else if (backendAvailable_ && !backendBusy_) {
                requestNextBatchPrediction();
            }
            return;
        }
        showSingleFailure(message);
        if (code == QStringLiteral("CONNECTION_LOST")
            || code == QStringLiteral("TRANSPORT_ERROR")) {
            backendStatusKnown_ = true;
            backendAvailable_ = false;
            backendReason_ = message;
            renderActiveState();
        }
    }
}

void InspectionPage::requestPrediction() {
    const QString imagePath = activeState().imagePath;
    if (mode_ != InspectionMode::Single || imagePath.isEmpty()
        || ui->inspectionImageView->imagePath() != imagePath
        || currentWorkpieceId_.isEmpty() || !backendAvailable_ || backendBusy_) {
        return;
    }
    pendingPredictionImagePath_ = imagePath;
    pendingPredictionWorkpieceId_ = currentWorkpieceId_;
    activeState().uiState = InspectionUiState::Running;
    activeState().message = QStringLiteral("正在检测…");
    renderActiveState();
    emit commandRequested(QStringLiteral("predict"), QJsonObject{
        {QStringLiteral("workpiece_id"), currentWorkpieceId_},
        {QStringLiteral("image_path"), imagePath},
    });
}

void InspectionPage::requestNextBatchPrediction() {
    if (!batchRunning_ || stopRequested_ || !currentBatchRequestId_.isEmpty()) return;
    if (batchCursor_ >= batchRecordOrder_.size()) {
        finishBatch(false);
        return;
    }
    currentBatchRequestId_ = batchRecordOrder_.at(batchCursor_);
    const InspectionRecord record = batchRecords_.value(currentBatchRequestId_);
    batchState_.uiState = InspectionUiState::Running;
    batchState_.message = QStringLiteral("正在检测：%1")
                              .arg(QFileInfo(record.imagePath).fileName());
    updateBatchSummary();
    emit commandRequested(QStringLiteral("predict"), QJsonObject{
        {QStringLiteral("workpiece_id"), batchWorkpieceId_},
        {QStringLiteral("image_path"), record.imagePath},
    });
}

void InspectionPage::finishBatch(bool stopped) {
    if (!batchRunning_) return;
    batchRunning_ = false;
    currentBatchRequestId_.clear();
    batchSelectionPinned_ = false;
    batchState_.uiState = failedBatchCount() > 0
        ? InspectionUiState::Failed : InspectionUiState::Completed;
    batchState_.message = stopped
        ? QStringLiteral("批量检测已停止")
        : (failedBatchCount() > 0 ? QStringLiteral("批量检测完成，存在失败记录")
                                  : QStringLiteral("批量检测完成"));
    selectPreferredBatchRecord();
    updateBatchSummary();
    renderActiveState();
    emit batchFinished(stopped);
}

void InspectionPage::requestConfirmation(const QString &orientation) {
    const InspectionRecord record = mode_ == InspectionMode::Batch
        ? batchRecords_.value(selectedRecordId_) : singleState_.visibleRecord;
    const bool hasRecord = mode_ == InspectionMode::Batch
        ? batchRecords_.contains(selectedRecordId_) && record.completedAt.isValid()
        : singleState_.hasRecord;
    if (!hasRecord || record.id.isEmpty()
        || record.imagePath.isEmpty() || record.workpieceId.isEmpty()
        || record.workpieceId != currentWorkpieceId_
        || !backendAvailable_ || backendBusy_) {
        return;
    }
    pendingConfirmationRecordId_ = record.id;
    pendingConfirmationOrientation_ = orientation;
    setRecordDisposition(record.id, BatchDisposition::Submitting);
    renderActiveState();
    emit confirmationRequested(record.id, record.workpieceId,
                               record.imagePath, orientation);
}

void InspectionPage::rejectCurrentRecord() {
    const QString recordId = mode_ == InspectionMode::Batch
        ? selectedRecordId_ : singleState_.visibleRecord.id;
    if (recordId.isEmpty()) return;
    setRecordDisposition(recordId, BatchDisposition::Rejected);
    renderActiveState();
    emit rejectionRequested(recordId);
}

void InspectionPage::storeRecentRecord(const InspectionRecord &record) {
    recentRecordOrder_.removeAll(record.id);
    recentRecordOrder_.prepend(record.id);
    recentRecords_.insert(record.id, record);
    while (recentRecordOrder_.size() > 50) {
        recentRecords_.remove(recentRecordOrder_.takeLast());
    }
    rebuildRecentList();
}

void InspectionPage::rebuildRecentList() {
    ui->recentInspectionList->clear();
    for (const QString &recordId : recentRecordOrder_) {
        const InspectionRecord record = recentRecords_.value(recordId);
        const QString review = record.needsReview ? QStringLiteral(" · 需复检") : QString();
        const QString result = record.error.isEmpty()
            ? orientationText(record.label) : QStringLiteral("失败");
        const QString time = record.completedAt.isValid()
            ? record.completedAt.toString(QStringLiteral("HH:mm:ss")) : QStringLiteral("--:--:--");
        auto *item = new QListWidgetItem(
            QStringLiteral("%1 · %2%3 · %4\n%5")
                .arg(QFileInfo(record.imagePath).fileName(), result, review, time,
                     dispositionText(record)),
            ui->recentInspectionList);
        item->setData(Qt::UserRole, record.id);
        const QPixmap thumbnail(record.imagePath);
        if (!thumbnail.isNull()) {
            item->setIcon(QIcon(thumbnail.scaled(64, 48, Qt::KeepAspectRatio,
                                                 Qt::SmoothTransformation)));
        }
        item->setToolTip(record.imagePath);
    }
}

void InspectionPage::rebuildBatchTable() {
    changingBatchSelection_ = true;
    ui->batchResultsTableWidget->setRowCount(0);
    QStringList visibleIds;
    for (const QString &recordId : batchRecordOrder_) {
        const InspectionRecord record = batchRecords_.value(recordId);
        if (!matchesFilter(record)) continue;
        visibleIds.append(recordId);
        const int row = ui->batchResultsTableWidget->rowCount();
        ui->batchResultsTableWidget->insertRow(row);
        const QString prediction = record.completedAt.isValid() && !record.label.isEmpty()
            ? orientationText(record.label) : QStringLiteral("—");
        const QString review = record.completedAt.isValid()
            ? (record.needsReview ? QStringLiteral("是") : QStringLiteral("否"))
            : QStringLiteral("—");
        const QString elapsed = record.completedAt.isValid()
            ? QString::number(record.elapsedMs, 'f', 1) : QStringLiteral("—");
        const QStringList values{
            QFileInfo(record.imagePath).fileName(), prediction, review, elapsed,
            dispositionText(record)};
        for (int column = 0; column < values.size(); ++column) {
            auto *item = new QTableWidgetItem(values.at(column));
            item->setData(Qt::UserRole, record.id);
            if (column == 0) item->setToolTip(record.imagePath);
            ui->batchResultsTableWidget->setItem(row, column, item);
        }
    }

    QString targetId = visibleIds.contains(selectedRecordId_)
        ? selectedRecordId_ : QString();
    if (targetId.isEmpty()) {
        for (const QString &recordId : visibleIds) {
            if (batchRecords_.value(recordId).needsReview) {
                targetId = recordId;
                break;
            }
        }
    }
    if (targetId.isEmpty()) {
        for (const QString &recordId : visibleIds) {
            if (batchRecords_.value(recordId).disposition == BatchDisposition::Pending) {
                targetId = recordId;
                break;
            }
        }
    }
    if (targetId.isEmpty() && !visibleIds.isEmpty()) targetId = visibleIds.constFirst();
    selectedRecordId_ = targetId;
    if (!targetId.isEmpty()) {
        for (int row = 0; row < ui->batchResultsTableWidget->rowCount(); ++row) {
            if (ui->batchResultsTableWidget->item(row, 0)->data(Qt::UserRole).toString()
                == targetId) {
                ui->batchResultsTableWidget->setCurrentCell(row, 0);
                break;
            }
        }
        const InspectionRecord record = batchRecords_.value(targetId);
        batchState_.imagePath = record.imagePath;
        batchState_.visibleRecord = record;
        batchState_.hasRecord = record.completedAt.isValid();
    } else {
        batchState_.visibleRecord = InspectionRecord();
        batchState_.hasRecord = false;
    }
    changingBatchSelection_ = false;
}

void InspectionPage::selectBatchRecord(const QString &recordId, bool userInitiated) {
    const InspectionRecord *record = batchRecord(recordId);
    if (record == nullptr) return;
    selectedRecordId_ = recordId;
    if (userInitiated) batchSelectionPinned_ = true;
    batchState_.imagePath = record->imagePath;
    batchState_.visibleRecord = *record;
    batchState_.hasRecord = record->completedAt.isValid();
    changingBatchSelection_ = true;
    for (int row = 0; row < ui->batchResultsTableWidget->rowCount(); ++row) {
        QTableWidgetItem *item = ui->batchResultsTableWidget->item(row, 0);
        if (item != nullptr && item->data(Qt::UserRole).toString() == recordId) {
            ui->batchResultsTableWidget->setCurrentCell(row, 0);
            break;
        }
    }
    changingBatchSelection_ = false;
    renderActiveState();
}

void InspectionPage::selectPreferredBatchRecord(const QString &afterRecordId) {
    if (batchRecordOrder_.isEmpty()) return;
    const int afterIndex = batchRecordOrder_.indexOf(afterRecordId);
    const int start = afterIndex >= 0 ? (afterIndex + 1) % batchRecordOrder_.size() : 0;
    for (int reviewPass = 0; reviewPass < 2; ++reviewPass) {
        for (int offset = 0; offset < batchRecordOrder_.size(); ++offset) {
            const QString recordId = batchRecordOrder_.at(
                (start + offset) % batchRecordOrder_.size());
            const InspectionRecord record = batchRecords_.value(recordId);
            if (!matchesFilter(record) || !record.completedAt.isValid()
                || record.response.isEmpty()
                || (record.disposition != BatchDisposition::Pending
                    && record.disposition != BatchDisposition::SubmitFailed)) {
                continue;
            }
            if (reviewPass == 0 && !record.needsReview) continue;
            selectBatchRecord(recordId, false);
            return;
        }
    }
    rebuildBatchTable();
}

InspectionRecord *InspectionPage::batchRecord(const QString &recordId) {
    auto it = batchRecords_.find(recordId);
    return it == batchRecords_.end() ? nullptr : &it.value();
}

const InspectionRecord *InspectionPage::batchRecord(const QString &recordId) const {
    auto it = batchRecords_.constFind(recordId);
    return it == batchRecords_.constEnd() ? nullptr : &it.value();
}

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

void InspectionPage::updateBatchSummary() {
    const int total = batchRecordOrder_.size();
    const int completed = completedBatchCount();
    const int failed = failedBatchCount();
    int review = 0;
    for (const QString &recordId : batchRecordOrder_) {
        const InspectionRecord record = batchRecords_.value(recordId);
        if (record.completedAt.isValid() && record.needsReview) ++review;
    }
    const int successful = qMax(0, completed - failed);
    ui->batchSummaryLabel->setText(
        QStringLiteral("已处理 %1/%2，成功 %3，需复检 %4，失败 %5")
            .arg(completed).arg(total).arg(successful).arg(review).arg(failed));
    ui->batchProgressBar->setRange(0, qMax(1, total));
    ui->batchProgressBar->setValue(completed);
}

void InspectionPage::renderActiveState() {
    const bool batchMode = mode_ == InspectionMode::Batch;
    ui->chooseImageButton->setVisible(!batchMode);
    ui->predictButton->setVisible(!batchMode);
    ui->chooseBatchImagesButton->setVisible(batchMode);
    ui->batchPredictButton->setVisible(batchMode);
    ui->stopBatchButton->setVisible(batchMode);
    ui->batchSummaryLabel->setVisible(batchMode);
    ui->batchProgressBar->setVisible(batchMode);
    ui->allBatchFilterButton->setVisible(batchMode);
    ui->needsReviewBatchFilterButton->setVisible(batchMode);
    ui->unprocessedBatchFilterButton->setVisible(batchMode);
    ui->failedBatchFilterButton->setVisible(batchMode);
    ui->batchResultsTableWidget->setVisible(batchMode);
    ui->recentInspectionTitleLabel->setVisible(!batchMode);
    ui->recentInspectionList->setVisible(!batchMode);
    const ModeState &state = activeState();
    if (state.imagePath.isEmpty()) {
        ui->inspectionImageView->setImagePath(QString());
        ui->currentImageLabel->setText(QStringLiteral("请选择待测图片或拖放图片到此处"));
    } else if (ui->inspectionImageView->setImagePath(state.imagePath)) {
        ui->currentImageLabel->setText(
            QStringLiteral("当前图片：%1").arg(QFileInfo(state.imagePath).fileName()));
    } else {
        ui->currentImageLabel->setText(
            QStringLiteral("图片无法读取：%1").arg(QFileInfo(state.imagePath).fileName()));
    }
    if (state.hasRecord) {
        renderRecord(state.visibleRecord);
    } else {
        ui->resultLabel->setText(QStringLiteral("尚未检测"));
        ui->reviewLabel->clear();
        ui->evidenceTextEdit->clear();
        ui->rawEvidenceTextEdit->clear();
        ui->currentResultTargetLabel->clear();
    }
    const InspectionUiState visibleState = uiState();
    if (visibleState == InspectionUiState::Running) {
        ui->resultMessageLabel->setText(QStringLiteral("正在检测…"));
    } else if (visibleState == InspectionUiState::BackendUnavailable) {
        ui->resultMessageLabel->setText(
            backendReason_.isEmpty() ? QStringLiteral("后端不可用") : backendReason_);
    } else if (visibleState == InspectionUiState::Failed) {
        ui->resultMessageLabel->setText(
            state.message.isEmpty() ? QStringLiteral("检测失败") : state.message);
    } else {
        ui->resultMessageLabel->clear();
    }
    updateActionAvailability();
}

void InspectionPage::renderRecord(const InspectionRecord &record) {
    ui->resultLabel->setText(
        QStringLiteral("检测结果：%1").arg(orientationText(record.label)));
    ui->reviewLabel->setText(record.needsReview
                                 ? QStringLiteral("建议人工复检") : QString());
    ui->evidenceTextEdit->setPlainText(evidenceSummary(record));
    ui->rawEvidenceTextEdit->setPlainText(rawEvidence(record));
    QString detail = dispositionText(record);
    if (mode_ == InspectionMode::Batch && record.workpieceId != currentWorkpieceId_) {
        detail = QStringLiteral("结果来自其他工件，请切回原工件后处理");
    } else if (mode_ == InspectionMode::Batch && batchRunning_) {
        detail = QStringLiteral("批量检测完成后可处理");
    } else if (!backendAvailable_) {
        detail = QStringLiteral("后端不可用，结果已保留");
    } else if (record.needsReview
               && (record.disposition == BatchDisposition::Pending
                   || record.disposition == BatchDisposition::SubmitFailed)) {
        detail = QStringLiteral("建议复检，%1").arg(detail);
    }
    ui->currentResultTargetLabel->setText(
        QStringLiteral("当前：%1（%2）")
            .arg(QFileInfo(record.imagePath).fileName(), detail));
    updateActionOrder(record.label);
}

void InspectionPage::updateActionOrder(const QString &predictedOrientation) {
    QVBoxLayout *layout = ui->confirmationActionsLayout;
    layout->removeWidget(ui->confirmFrontButton);
    layout->removeWidget(ui->confirmBackButton);
    layout->removeWidget(ui->rejectConfirmationButton);
    if (predictedOrientation == QStringLiteral("back")) {
        ui->confirmBackButton->setText(QStringLiteral("确认反面"));
        ui->confirmFrontButton->setText(QStringLiteral("修正为正面"));
        layout->insertWidget(0, ui->confirmBackButton);
        layout->insertWidget(1, ui->confirmFrontButton);
    } else {
        ui->confirmFrontButton->setText(QStringLiteral("确认正面"));
        ui->confirmBackButton->setText(QStringLiteral("修正为反面"));
        layout->insertWidget(0, ui->confirmFrontButton);
        layout->insertWidget(1, ui->confirmBackButton);
    }
    layout->insertWidget(2, ui->rejectConfirmationButton);
}

void InspectionPage::updateActionAvailability() {
    const ModeState &state = activeState();
    const bool interactive = backendAvailable_ && !backendBusy_;
    ui->chooseImageButton->setEnabled(interactive);
    ui->chooseBatchImagesButton->setEnabled(interactive && !batchRunning_);
    ui->batchPredictButton->setEnabled(interactive && !batchRunning_);
    ui->stopBatchButton->setEnabled(batchRunning_ && !stopRequested_);
    ui->predictButton->setEnabled(interactive && mode_ == InspectionMode::Single
                                  && !state.imagePath.isEmpty()
                                  && ui->inspectionImageView->imagePath() == state.imagePath
                                  && !currentWorkpieceId_.isEmpty());
    const bool singleRecordReady = interactive && mode_ == InspectionMode::Single
        && state.hasRecord && state.visibleRecord.error.isEmpty()
        && state.visibleRecord.workpieceId == currentWorkpieceId_
        && (state.visibleRecord.disposition == BatchDisposition::Pending
            || state.visibleRecord.disposition == BatchDisposition::SubmitFailed);
    if (mode_ == InspectionMode::Single) {
        ui->confirmFrontButton->setEnabled(singleRecordReady);
        ui->confirmBackButton->setEnabled(singleRecordReady);
        ui->rejectConfirmationButton->setEnabled(singleRecordReady);
    } else {
        const InspectionRecord record = batchRecords_.value(selectedRecordId_);
        const bool batchRecordReady = interactive && !batchRunning_
            && batchRecords_.contains(selectedRecordId_)
            && record.completedAt.isValid() && !record.response.isEmpty()
            && record.workpieceId == currentWorkpieceId_
            && (record.disposition == BatchDisposition::Pending
                || record.disposition == BatchDisposition::SubmitFailed);
        ui->confirmFrontButton->setEnabled(batchRecordReady);
        ui->confirmBackButton->setEnabled(batchRecordReady);
        ui->rejectConfirmationButton->setEnabled(batchRecordReady);
    }
}

void InspectionPage::updateRecordDisposition(const QString &recordId,
                                             BatchDisposition disposition,
                                             const QString &error) {
    if (!recentRecords_.contains(recordId)) return;
    InspectionRecord record = recentRecords_.value(recordId);
    record.disposition = disposition;
    record.submissionError = error;
    recentRecords_.insert(recordId, record);
    if (singleState_.visibleRecord.id == recordId) singleState_.visibleRecord = record;
    rebuildRecentList();
}

QString InspectionPage::orientationText(const QString &label) const {
    if (label == QStringLiteral("front")) return QStringLiteral("正面");
    if (label == QStringLiteral("back")) return QStringLiteral("反面");
    if (label == QStringLiteral("uncertain")) return QStringLiteral("不确定");
    return label;
}

QString InspectionPage::dispositionText(const InspectionRecord &record) const {
    const QString evolutionState = record.response.value(
        QStringLiteral("evolution_state")).toString();
    if (record.disposition == BatchDisposition::PredictionFailed) {
        return record.error.startsWith(QStringLiteral("结果未知"))
            ? QStringLiteral("结果未知") : QStringLiteral("预测失败");
    }
    if (record.disposition == BatchDisposition::SubmitFailed) {
        return batchRecords_.contains(record.id)
            ? QStringLiteral("提交失败") : QStringLiteral("写入失败");
    }
    if (record.disposition == BatchDisposition::Rejected) return QStringLiteral("不入库");
    if (record.disposition == BatchDisposition::QueuedFront) return QStringLiteral("正面已排队");
    if (record.disposition == BatchDisposition::QueuedBack) return QStringLiteral("反面已排队");
    if (record.disposition == BatchDisposition::Submitting
        || evolutionState == QStringLiteral("running")) {
        return QStringLiteral("正在更新缓存");
    }
    if (evolutionState == QStringLiteral("completed")) return QStringLiteral("已参与预测");
    if (!record.completedAt.isValid()) return QStringLiteral("待检测");
    return QStringLiteral("等待写入");
}

QString InspectionPage::evidenceSummary(const InspectionRecord &record) const {
    const QJsonObject response = record.response;
    const QString globalPrediction = response.value(
        QStringLiteral("global_prediction")).toString(record.label);
    const QString localPrediction = response.value(
        QStringLiteral("local_prediction")).toString();
    const QString decisionSource = response.value(
        QStringLiteral("decision_source")).toString();
    QString globalLine = globalPrediction.isEmpty()
        ? QStringLiteral("全局特征：未提供独立结论")
        : QStringLiteral("全局特征：支持%1").arg(orientationText(globalPrediction));
    QString localLine = localPrediction.isEmpty()
        ? QStringLiteral("局部匹配：未提供独立结论")
        : QStringLiteral("局部匹配：支持%1").arg(orientationText(localPrediction));
    if (decisionSource == QStringLiteral("global")) {
        globalLine.append(QStringLiteral("（采用此结果）"));
    } else if (decisionSource == QStringLiteral("local_override")) {
        localLine.append(QStringLiteral("（采用此结果）"));
    }
    QStringList lines{globalLine, localLine,
                      QStringLiteral("几何规则：%1").arg(geometryDescription(response))};
    if (record.needsReview) {
        const QString reason = response.value(QStringLiteral("review_reason"))
                                   .toString(QStringLiteral("证据需人工复核"));
        lines.append(QStringLiteral("复检原因：%1").arg(reason));
    }
    return lines.join(QLatin1Char('\n'));
}

QString InspectionPage::rawEvidence(const InspectionRecord &record) const {
    QStringList lines;
    const QJsonObject response = record.response;
    if (response.contains(QStringLiteral("global_scores"))) {
        lines.append(QStringLiteral("全局原始得分：%1")
                         .arg(QString::fromUtf8(QJsonDocument(
                             response.value(QStringLiteral("global_scores")).toObject())
                                                   .toJson(QJsonDocument::Compact))));
    }
    if (response.contains(QStringLiteral("global_margin"))) {
        lines.append(QStringLiteral("全局原始间隔：%1")
                         .arg(QString::number(
                             response.value(QStringLiteral("global_margin")).toDouble(), 'g', 8)));
    }
    if (response.contains(QStringLiteral("local_scores"))) {
        lines.append(QStringLiteral("局部候选得分：%1")
                         .arg(QString::fromUtf8(QJsonDocument(
                             response.value(QStringLiteral("local_scores")).toObject())
                                                   .toJson(QJsonDocument::Compact))));
    }
    if (response.contains(QStringLiteral("local_margin"))) {
        lines.append(QStringLiteral("局部原始间隔：%1")
                         .arg(QString::number(
                             response.value(QStringLiteral("local_margin")).toDouble(), 'g', 8)));
    }
    const double elapsed = record.elapsedMs > 0.0
        ? record.elapsedMs : response.value(QStringLiteral("elapsed_ms")).toDouble();
    lines.append(QStringLiteral("耗时（毫秒）：%1").arg(QString::number(elapsed, 'g', 10)));
    return lines.join(QLatin1Char('\n'));
}
