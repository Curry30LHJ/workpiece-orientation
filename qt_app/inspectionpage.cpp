#include "inspectionpage.h"

#include "ui_inspectionpage.h"

#include <QApplication>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QListWidgetItem>
#include <QPixmap>
#include <QPushButton>
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
                if (mode_ == InspectionMode::Single) requestConfirmation(QStringLiteral("front"));
            });
    connect(ui->confirmBackButton, &QPushButton::clicked,
            this, [this]() {
                if (mode_ == InspectionMode::Single) requestConfirmation(QStringLiteral("back"));
            });
    connect(ui->rejectConfirmationButton, &QPushButton::clicked,
            this, [this]() {
                if (mode_ == InspectionMode::Single) rejectCurrentRecord();
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
    updateActionAvailability();
}

void InspectionPage::setBackendAvailable(bool available, bool busy, const QString &reason) {
    backendStatusKnown_ = true;
    backendAvailable_ = available;
    backendBusy_ = busy;
    backendReason_ = reason;
    renderActiveState();
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
    if (mode_ == InspectionMode::Batch) {
        renderActiveState();
    }
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
        if (recordId.isEmpty() || !recentRecords_.contains(recordId)) return;
        InspectionRecord record = recentRecords_.value(recordId);
        const QJsonObject job = response.value(QStringLiteral("job")).toObject();
        record.evolutionJobId = job.value(QStringLiteral("job_id")).toString();
        const QString state = job.value(QStringLiteral("state")).toString(QStringLiteral("queued"));
        if (state == QStringLiteral("failed")) {
            record.disposition = BatchDisposition::SubmitFailed;
            record.submissionError = job.value(QStringLiteral("error")).toString();
        } else {
            record.disposition = pendingConfirmationOrientation_ == QStringLiteral("back")
                ? BatchDisposition::QueuedBack : BatchDisposition::QueuedFront;
            record.submissionError.clear();
            record.response.insert(QStringLiteral("evolution_state"), state);
        }
        recentRecords_.insert(recordId, record);
        pendingConfirmationRecordId_.clear();
        pendingConfirmationOrientation_.clear();
        if (singleState_.visibleRecord.id == recordId) singleState_.visibleRecord = record;
        rebuildRecentList();
        renderActiveState();
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
        }
        rebuildRecentList();
        renderActiveState();
    }
}

void InspectionPage::handleBackendFailure(const QString &command, const QString &code,
                                          const QString &message) {
    if (command == QStringLiteral("submit_confirmation")
        && !pendingConfirmationRecordId_.isEmpty()) {
        updateRecordDisposition(pendingConfirmationRecordId_,
                                BatchDisposition::SubmitFailed, message);
        pendingConfirmationRecordId_.clear();
        pendingConfirmationOrientation_.clear();
        renderActiveState();
        return;
    }
    if (command == QStringLiteral("predict")) {
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

void InspectionPage::requestConfirmation(const QString &orientation) {
    const InspectionRecord record = singleState_.visibleRecord;
    if (!singleState_.hasRecord || record.id.isEmpty()
        || record.imagePath.isEmpty() || record.workpieceId.isEmpty()
        || record.workpieceId != currentWorkpieceId_
        || !backendAvailable_ || backendBusy_) {
        return;
    }
    pendingConfirmationRecordId_ = record.id;
    pendingConfirmationOrientation_ = orientation;
    updateRecordDisposition(record.id, BatchDisposition::Submitting);
    renderActiveState();
    emit confirmationRequested(record.id, record.workpieceId,
                               record.imagePath, orientation);
}

void InspectionPage::rejectCurrentRecord() {
    if (!singleState_.hasRecord || singleState_.visibleRecord.id.isEmpty()) return;
    const QString recordId = singleState_.visibleRecord.id;
    updateRecordDisposition(recordId, BatchDisposition::Rejected);
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

void InspectionPage::renderActiveState() {
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
    ui->currentResultTargetLabel->setText(
        QStringLiteral("当前：%1（%2）")
            .arg(QFileInfo(record.imagePath).fileName(), dispositionText(record)));
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
    if (record.disposition == BatchDisposition::SubmitFailed) return QStringLiteral("写入失败");
    if (record.disposition == BatchDisposition::Rejected) return QStringLiteral("不入库");
    if (record.disposition == BatchDisposition::Submitting
        || evolutionState == QStringLiteral("running")) {
        return QStringLiteral("正在更新缓存");
    }
    if (evolutionState == QStringLiteral("completed")) return QStringLiteral("已参与预测");
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
