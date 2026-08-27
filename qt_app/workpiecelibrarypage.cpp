#include "workpiecelibrarypage.h"

#include "ui_workpiecelibrarypage.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QFileDialog>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QSet>
#include <QSignalBlocker>
#include <QStyle>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QUuid>

namespace {

const QString kImageFilter = QStringLiteral("PNG/JPEG/BMP (*.png *.jpg *.jpeg *.bmp)");

QString imageContentKey(const QImage &image) {
    const QImage normalized = image.convertToFormat(QImage::Format_RGBA8888);
    const int width = normalized.width();
    const int height = normalized.height();
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(reinterpret_cast<const char *>(&width), sizeof(width));
    hash.addData(reinterpret_cast<const char *>(&height), sizeof(height));
    hash.addData(reinterpret_cast<const char *>(normalized.constBits()),
                 normalized.sizeInBytes());
    return QString::fromLatin1(hash.result().toHex());
}

QString directionText(const QString &direction) {
    return direction == QStringLiteral("front") ? QStringLiteral("正面")
         : direction == QStringLiteral("back") ? QStringLiteral("反面")
                                                 : direction;
}

QString phaseText(const QString &phase) {
    if (phase == QStringLiteral("queued")) return QStringLiteral("排队中");
    if (phase == QStringLiteral("validating")) return QStringLiteral("校验图片");
    if (phase == QStringLiteral("copying")) return QStringLiteral("写入模板");
    if (phase == QStringLiteral("features")) return QStringLiteral("提取特征");
    if (phase == QStringLiteral("fast_originals")) return QStringLiteral("提取快速特征");
    if (phase == QStringLiteral("fast_augmentation")) return QStringLiteral("生成旋转增强");
    if (phase == QStringLiteral("fast_ridge")) return QStringLiteral("构建快速判别器");
    if (phase == QStringLiteral("committing")) return QStringLiteral("提交更新");
    if (phase == QStringLiteral("active")) return QStringLiteral("已生效");
    return phase;
}

bool fastCacheCapabilityUnavailable(const QJsonObject &cache) {
    return cache.value(QStringLiteral("error")).toString().contains(
        QStringLiteral("FAST_CACHE_CAPABILITY_UNAVAILABLE"));
}

QString fastCacheStateText(const QJsonObject &cache) {
    const QString state = cache.value(QStringLiteral("state")).toString();
    if (state == QStringLiteral("ready")) return QStringLiteral("已就绪");
    if (state == QStringLiteral("queued")) return QStringLiteral("排队中");
    if (state == QStringLiteral("running") || state == QStringLiteral("building")) {
        return QStringLiteral("构建中");
    }
    if (state == QStringLiteral("failed")) return QStringLiteral("构建失败");
    if (state == QStringLiteral("not_ready")) {
        return fastCacheCapabilityUnavailable(cache)
            ? QStringLiteral("能力不可用") : QStringLiteral("未就绪");
    }
    return state.isEmpty() ? QStringLiteral("未提供") : state;
}

QString fastCacheDescription(const QJsonObject &cache) {
    if (cache.isEmpty()) return QString();
    QStringList parts{QStringLiteral("快速缓存：%1").arg(fastCacheStateText(cache))};
    if (cache.value(QStringLiteral("completed")).isDouble()
        && cache.value(QStringLiteral("total")).isDouble()) {
        parts.append(QStringLiteral("进度 %1/%2")
                         .arg(cache.value(QStringLiteral("completed")).toInt())
                         .arg(cache.value(QStringLiteral("total")).toInt()));
    }
    if (cache.value(QStringLiteral("elapsed_ms")).isDouble()) {
        parts.append(QStringLiteral("耗时 %1 ms")
                         .arg(QString::number(
                             cache.value(QStringLiteral("elapsed_ms")).toDouble(), 'g', 10)));
    }
    const QString error = cache.value(QStringLiteral("error")).toString();
    if (!error.isEmpty()) parts.append(QStringLiteral("提示：%1").arg(error));
    return parts.join(QStringLiteral(" · "));
}

QString fastCacheMessageKind(const QJsonObject &cache) {
    const QString state = cache.value(QStringLiteral("state")).toString();
    if (state == QStringLiteral("ready")) return QStringLiteral("success");
    if (state == QStringLiteral("failed") || fastCacheCapabilityUnavailable(cache)) {
        return QStringLiteral("error");
    }
    return cache.isEmpty() ? QStringLiteral("neutral") : QStringLiteral("warning");
}

QDateTime jobSubmittedAt(const QJsonObject &job) {
    const QJsonValue timestamp = job.value(QStringLiteral("last_submitted_at"));
    if (timestamp.isDouble()) {
        return QDateTime::fromMSecsSinceEpoch(
            static_cast<qint64>(timestamp.toDouble() * 1000.0), Qt::UTC);
    }
    const QString value = timestamp.toString();
    QDateTime submittedAt = QDateTime::fromString(value, Qt::ISODateWithMs);
    if (!submittedAt.isValid()) submittedAt = QDateTime::fromString(value, Qt::ISODate);
    return submittedAt;
}

bool shouldReplaceSelectedJob(const QJsonObject &selected,
                              const QDateTime &selectedAt,
                              const QDateTime &candidateAt) {
    if (selected.isEmpty()) return true;
    if (candidateAt.isValid() && !selectedAt.isValid()) return true;
    if (!candidateAt.isValid() && !selectedAt.isValid()) return true;
    return candidateAt.isValid() && selectedAt.isValid() && candidateAt > selectedAt;
}

QString templatePathSummary(const QStringList &paths) {
    if (paths.isEmpty()) return QStringLiteral("未选择");
    const QString firstName = QFileInfo(paths.constFirst()).fileName();
    if (paths.size() == 1) return firstName;
    return QStringLiteral("%1 等 %2 张").arg(firstName).arg(paths.size());
}

} // namespace

WorkpieceLibraryPage::WorkpieceLibraryPage(QWidget *parent)
    : QWidget(parent), ui(new Ui::WorkpieceLibraryPage) {
    ui->setupUi(this);
    ui->librarySplitter->setStretchFactor(0, 1);
    ui->librarySplitter->setStretchFactor(1, 3);
    ui->librarySplitter->setChildrenCollapsible(false);
    ui->libraryWorkpieceList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->libraryWorkpieceList->setTextElideMode(Qt::ElideRight);
    ui->templateDetailsTable->horizontalHeader()->setStretchLastSection(true);
    ui->evolutionJobsTable->horizontalHeader()->setStretchLastSection(true);
    ui->workpieceDetailsSummaryLabel->setWordWrap(true);
    ui->registrationProgressBar->setRange(0, 1);
    ui->registrationProgressBar->setValue(0);
    registrationElapsedTimer_.setInterval(100);

    connect(&registrationElapsedTimer_, &QTimer::timeout,
            this, &WorkpieceLibraryPage::updateRegistrationElapsed);
    connect(ui->chooseFrontTemplatesButton, &QPushButton::clicked,
            this, &WorkpieceLibraryPage::chooseFrontTemplates);
    connect(ui->chooseBackTemplatesButton, &QPushButton::clicked,
            this, &WorkpieceLibraryPage::chooseBackTemplates);
    connect(ui->registerButton, &QPushButton::clicked,
            this, &WorkpieceLibraryPage::submitRegistration);
    connect(ui->refreshWorkpiecesButton, &QPushButton::clicked,
            this, &WorkpieceLibraryPage::refreshWorkpieces);
    connect(ui->librarySearchEdit, &QLineEdit::textChanged,
            this, &WorkpieceLibraryPage::filterWorkpieces);
    connect(ui->libraryWorkpieceList, &QListWidget::currentItemChanged,
            this, [this](QListWidgetItem *, QListWidgetItem *) {
                browseSelectedWorkpiece();
            });
    connect(ui->setCurrentWorkpieceButton, &QPushButton::clicked,
            this, &WorkpieceLibraryPage::activateBrowsedWorkpiece);
    connect(ui->deleteWorkpieceButton, &QPushButton::clicked,
            this, &WorkpieceLibraryPage::recycleBrowsedWorkpiece);
    connect(ui->recycleNameConfirmationEdit, &QLineEdit::textChanged,
            this, [this]() { updateControlStates(); });
    connect(ui->workpieceNameEdit, &QLineEdit::textChanged, this, [this]() {
        setDirty(true);
        updateControlStates();
    });
    auto updatePrimaryRole = [this](int tabIndex) {
        for (QPushButton *button : {ui->setCurrentWorkpieceButton,
                                    ui->refreshWorkpiecesButton,
                                    ui->registerButton}) {
            button->setProperty("role", QStringLiteral("secondary"));
            button->style()->unpolish(button);
            button->style()->polish(button);
        }
        QPushButton *primary = tabIndex == 1 ? ui->registerButton
            : (tabIndex == 2 ? ui->refreshWorkpiecesButton
                             : ui->setCurrentWorkpieceButton);
        primary->setProperty("role", QStringLiteral("primary"));
        primary->style()->unpolish(primary);
        primary->style()->polish(primary);
    };
    connect(ui->libraryTabWidget, &QTabWidget::currentChanged,
            this, updatePrimaryRole);
    updatePrimaryRole(ui->libraryTabWidget->currentIndex());
    ui->deleteWorkpieceButton->setProperty("role", QStringLiteral("danger"));
    ui->templateWarningLabel->setProperty("messageKind", QStringLiteral("warning"));
    ui->libraryMessageLabel->setProperty("messageKind", QStringLiteral("neutral"));
    QWidget::setTabOrder(ui->librarySearchEdit, ui->libraryWorkpieceList);
    QWidget::setTabOrder(ui->libraryWorkpieceList, ui->setCurrentWorkpieceButton);
    QWidget::setTabOrder(ui->setCurrentWorkpieceButton, ui->refreshWorkpiecesButton);
    QWidget::setTabOrder(ui->refreshWorkpiecesButton, ui->libraryTabWidget);

    updateTemplateUi();
    updateControlStates();
}

WorkpieceLibraryPage::~WorkpieceLibraryPage() {
    delete ui;
}

void WorkpieceLibraryPage::setWorkpieces(
    const QJsonArray &workpieces, const QString &currentDetectionWorkpieceId) {
    workpieces_ = workpieces;
    currentDetectionWorkpieceId_ = currentDetectionWorkpieceId;
    bool browsedStillExists = false;
    for (const QJsonValue &value : workpieces_) {
        browsedStillExists = browsedStillExists
            || value.toObject().value(QStringLiteral("id")).toString() == browsedWorkpieceId_;
    }
    if (!browsedStillExists) {
        browsedWorkpieceId_.clear();
        workpieceDetails_ = QJsonObject();
        ui->templateDetailsTable->setRowCount(0);
        ui->workpieceDetailsSummaryLabel->setText(QStringLiteral("请选择工件查看详情"));
    }
    rebuildWorkpieceList();
    updateControlStates();
}

void WorkpieceLibraryPage::setWorkpieceDetails(const QJsonObject &details) {
    const QString id = details.value(QStringLiteral("id")).toString(
        details.value(QStringLiteral("workpiece_id")).toString());
    if (!browsedWorkpieceId_.isEmpty() && id != browsedWorkpieceId_) return;
    workpieceDetails_ = details;
    const QJsonObject counts = details.value(QStringLiteral("template_counts")).toObject();
    QString summary =
        QStringLiteral("%1 · 正面 %2 张 · 反面 %3 张 · 几何规则 %4 条 · %5")
            .arg(details.value(QStringLiteral("name")).toString())
            .arg(counts.value(QStringLiteral("front")).toInt())
            .arg(counts.value(QStringLiteral("back")).toInt())
            .arg(details.value(QStringLiteral("geometry_rule_count")).toInt())
            .arg(details.value(QStringLiteral("detectable")).toBool()
                     ? QStringLiteral("可检测") : QStringLiteral("不可检测"));
    const QJsonObject fastCache = details.value(QStringLiteral("fast_cache")).toObject();
    const QString fastCacheText = fastCacheDescription(fastCache);
    if (!fastCacheText.isEmpty()) summary.append(QLatin1Char('\n') + fastCacheText);
    ui->workpieceDetailsSummaryLabel->setProperty(
        "messageKind", fastCacheMessageKind(fastCache));
    ui->workpieceDetailsSummaryLabel->style()->unpolish(ui->workpieceDetailsSummaryLabel);
    ui->workpieceDetailsSummaryLabel->style()->polish(ui->workpieceDetailsSummaryLabel);
    ui->workpieceDetailsSummaryLabel->setText(summary);
    const QJsonArray templates = details.value(QStringLiteral("templates")).toArray();
    ui->templateDetailsTable->setRowCount(templates.size());
    for (int row = 0; row < templates.size(); ++row) {
        const QJsonObject item = templates.at(row).toObject();
        auto *templateItem = new QTableWidgetItem(item.value(QStringLiteral("template_id")).toString());
        templateItem->setData(Qt::UserRole, item.value(QStringLiteral("template_id")));
        const QString previewPath = item.value(QStringLiteral("preview_path")).toString();
        if (item.value(QStringLiteral("readable")).toBool() && !previewPath.isEmpty()) {
            templateItem->setIcon(QIcon(previewPath));
        }
        ui->templateDetailsTable->setItem(row, 0, templateItem);
        ui->templateDetailsTable->setItem(
            row, 1, new QTableWidgetItem(directionText(
                        item.value(QStringLiteral("direction")).toString())));
        ui->templateDetailsTable->setItem(
            row, 2, new QTableWidgetItem(item.value(QStringLiteral("source")).toString()));
        ui->templateDetailsTable->setItem(
            row, 3, new QTableWidgetItem(item.value(QStringLiteral("added_at")).toString()));
        ui->templateDetailsTable->setItem(
            row, 4, new QTableWidgetItem(item.value(QStringLiteral("readable")).toBool()
                                             ? QStringLiteral("可读取")
                                             : QStringLiteral("无法读取")));
    }
    ui->templateDetailsTable->resizeColumnsToContents();
}

void WorkpieceLibraryPage::setEvolutionJobs(const QJsonArray &jobs) {
    for (const QJsonValue &value : jobs) {
        const QJsonObject incoming = value.toObject();
        const QString jobId = incoming.value(QStringLiteral("job_id")).toString();
        if (jobId.isEmpty()) continue;
        if (!evolutionJobsById_.contains(jobId)) evolutionJobOrder_.append(jobId);
        QJsonObject merged = evolutionJobsById_.value(jobId);
        for (auto it = incoming.begin(); it != incoming.end(); ++it) {
            merged.insert(it.key(), it.value());
        }
        evolutionJobsById_.insert(jobId, merged);
    }
    rebuildEvolutionTable();

    QJsonObject selectedActive;
    QJsonObject selectedTerminal;
    QDateTime selectedActiveAt;
    QDateTime selectedTerminalAt;
    for (const QJsonValue &value : jobs) {
        const QJsonObject job = value.toObject();
        const QString state = job.value(QStringLiteral("state")).toString();
        const QDateTime submittedAt = jobSubmittedAt(job);
        if (state == QStringLiteral("queued") || state == QStringLiteral("building")) {
            if (shouldReplaceSelectedJob(selectedActive, selectedActiveAt, submittedAt)) {
                selectedActive = job;
                selectedActiveAt = submittedAt;
            }
        } else if (state == QStringLiteral("completed") || state == QStringLiteral("failed")) {
            if (shouldReplaceSelectedJob(selectedTerminal, selectedTerminalAt, submittedAt)) {
                selectedTerminal = job;
                selectedTerminalAt = submittedAt;
            }
        }
    }
    const QJsonObject job = selectedActive.isEmpty() ? selectedTerminal : selectedActive;
    if (!job.isEmpty()) {
        const QString state = job.value(QStringLiteral("state")).toString();
        const QString phase = state == QStringLiteral("failed")
            ? state : job.value(QStringLiteral("phase")).toString(
                          state == QStringLiteral("completed")
                              ? QStringLiteral("active") : state);
        emit taskStatusChanged(
            QStringLiteral("后台模板强化"),
            phase,
            job.value(QStringLiteral("completed")).toInt(),
            job.value(QStringLiteral("total")).toInt(),
            job.value(QStringLiteral("elapsed_ms")).isDouble()
                ? static_cast<qint64>(job.value(QStringLiteral("elapsed_ms")).toDouble()) : -1);
    }
}

void WorkpieceLibraryPage::setBackendState(BackendUiState state,
                                           const QString &detail) {
    backendState_ = state;
    if (state == BackendUiState::Error || state == BackendUiState::Disconnected) {
        const bool registrationInterrupted = registrationInFlight_;
        if (registrationInterrupted) {
            publishRegistrationFailure();
            registrationInFlight_ = false;
            stopRegistrationProgress();
        }
        const QString reason = detail.isEmpty() ? QStringLiteral("后端不可用") : detail;
        showMessage(registrationInterrupted
                        ? QStringLiteral("建库连接中断，结果未知；草稿已保留：%1").arg(reason)
                        : reason,
                    true);
    }
    updateControlStates();
}

void WorkpieceLibraryPage::setRegistrationProgress(const QJsonObject &progress,
                                                    qint64 elapsedMs) {
    const QString phase = progress.value(QStringLiteral("phase")).toString();
    const int total = progress.value(QStringLiteral("total")).toInt(1);
    const int completed = progress.value(QStringLiteral("completed")).toInt();
    const qint64 effectiveElapsed = elapsedMs >= 0 ? elapsedMs
        : (registrationElapsedClock_.isValid() ? registrationElapsedClock_.elapsed() : 0);
    ui->registrationProgressBar->setRange(0, qMax(1, total));
    ui->registrationProgressBar->setValue(qBound(0, completed, qMax(1, total)));
    ui->registrationProgressLabel->setText(
        QStringLiteral("建库进度：%1（%2/%3）")
            .arg(phaseText(phase)).arg(completed).arg(total));
    ui->registrationElapsedLabel->setText(
        QStringLiteral("耗时：%1 ms").arg(effectiveElapsed));
    publishRegistrationTaskStatus(phase, completed, total, effectiveElapsed);
}

void WorkpieceLibraryPage::setRegistrationResult(const QJsonObject &response) {
    stopRegistrationProgress();
    registrationInFlight_ = false;
    const QJsonObject counts = response.value(QStringLiteral("template_counts")).toObject();
    const int frontCount = counts.value(QStringLiteral("front")).toInt(
        savedRegistrationFields_.value(QStringLiteral("front_images")).toArray().size());
    const int backCount = counts.value(QStringLiteral("back")).toInt(
        savedRegistrationFields_.value(QStringLiteral("back_images")).toArray().size());
    const qint64 elapsedMs = static_cast<qint64>(response.value(QStringLiteral("elapsed_ms")).toDouble(
        registrationElapsedClock_.isValid() ? registrationElapsedClock_.elapsed() : 0));
    QString result = QStringLiteral("工件库建立成功：正面 %1 张，反面 %2 张，耗时 %3 ms")
                         .arg(frontCount).arg(backCount).arg(elapsedMs);
    const QString fastCacheState = response.value(
        QStringLiteral("fast_cache_state")).toString();
    QJsonObject fastCache;
    if (!fastCacheState.isEmpty()) {
        fastCache.insert(QStringLiteral("state"), fastCacheState);
        result.append(QStringLiteral(" · %1").arg(fastCacheDescription(fastCache)));
    }
    const QString fastCacheRevision = response.value(
        QStringLiteral("fast_cache_revision")).toString();
    if (!fastCacheRevision.isEmpty()) {
        result.append(QStringLiteral(" · 版本 %1").arg(fastCacheRevision));
    }
    ui->latestRegistrationResultLabel->setProperty(
        "messageKind", fastCacheMessageKind(fastCache));
    ui->latestRegistrationResultLabel->style()->unpolish(
        ui->latestRegistrationResultLabel);
    ui->latestRegistrationResultLabel->style()->polish(
        ui->latestRegistrationResultLabel);
    ui->latestRegistrationResultLabel->setText(result);
    showMessage(result);
    if (draftMatchesSavedRegistration()) setDirty(false);
    publishRegistrationTaskStatus(QStringLiteral("active"), frontCount + backCount,
                                  frontCount + backCount, elapsedMs);
    updateControlStates();
}

void WorkpieceLibraryPage::handleBackendResponse(const QString &command,
                                                  const QJsonObject &response) {
    if (command == QStringLiteral("register")) {
        setRegistrationResult(response);
    } else if (command == QStringLiteral("get_workpiece_details")) {
        setWorkpieceDetails(response.value(QStringLiteral("workpiece")).toObject());
    } else if (command == QStringLiteral("recycle_workpiece")) {
        showMessage(QStringLiteral("工件已移入回收区，可由后端恢复"));
    }
}

void WorkpieceLibraryPage::handleBackendFailure(const QString &command,
                                                 const QString &code,
                                                 const QString &message) {
    if (command == QStringLiteral("register")
        && code == QStringLiteral("WORKPIECE_EXISTS")
        && registrationInFlight_
        && !savedRegistrationFields_.value(QStringLiteral("replace")).toBool()) {
        const QString name = savedRegistrationFields_.value(QStringLiteral("name")).toString();
        const bool confirmed = replaceConfirmationHandler_
            ? replaceConfirmationHandler_(name)
            : QMessageBox::question(this, QStringLiteral("确认覆盖"),
                                     QStringLiteral("工件“%1”已存在，是否覆盖？").arg(name),
                                     QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes;
        if (confirmed) {
            QJsonObject retry = savedRegistrationFields_;
            retry.insert(QStringLiteral("replace"), true);
            savedRegistrationFields_ = retry;
            emit commandRequested(QStringLiteral("register"), retry);
            showMessage(QStringLiteral("正在覆盖并建立工件库…"));
        } else {
            publishRegistrationTaskStatus(QStringLiteral("cancelled"),
                                          registrationTaskCompleted_,
                                          registrationTaskTotal_,
                                          currentRegistrationElapsedMs());
            registrationInFlight_ = false;
            stopRegistrationProgress();
            showMessage(QStringLiteral("已取消覆盖"));
            updateControlStates();
        }
        return;
    }
    if (command == QStringLiteral("register")) {
        if (registrationInFlight_) publishRegistrationFailure();
        registrationInFlight_ = false;
        stopRegistrationProgress();
        updateControlStates();
    }
    setOperationError(code, message);
}

void WorkpieceLibraryPage::setOperationError(const QString &code,
                                              const QString &message) {
    showMessage(code.isEmpty() ? message
                              : QStringLiteral("%1（%2）").arg(message, code), true);
}

void WorkpieceLibraryPage::setTemplatePaths(const QStringList &front,
                                             const QStringList &back) {
    frontTemplatePaths_ = normalizedPaths(front);
    backTemplatePaths_ = normalizedPaths(back);
    setDirty(true);
    updateTemplateUi();
    updateControlStates();
}

void WorkpieceLibraryPage::setWorkpieceName(const QString &name) {
    ui->workpieceNameEdit->setText(name);
}

void WorkpieceLibraryPage::setReplaceConfirmationHandler(
    std::function<bool(const QString &)> handler) {
    replaceConfirmationHandler_ = std::move(handler);
}

bool WorkpieceLibraryPage::hasUnsavedChanges() const {
    return dirty_;
}

bool WorkpieceLibraryPage::hasActiveRegistration() const {
    return registrationInFlight_;
}

bool WorkpieceLibraryPage::hasActiveEvolutionTask() const {
    for (const QJsonObject &job : evolutionJobsById_) {
        const QString state = job.value(QStringLiteral("state")).toString();
        if (state == QStringLiteral("queued") || state == QStringLiteral("building")
            || state == QStringLiteral("running")) {
            return true;
        }
    }
    return false;
}

void WorkpieceLibraryPage::discardEditingDraft() {
    const QSignalBlocker blocker(ui->workpieceNameEdit);
    ui->workpieceNameEdit->clear();
    frontTemplatePaths_.clear();
    backTemplatePaths_.clear();
    updateTemplateUi();
    setDirty(false);
    showMessage(QStringLiteral("编辑草稿已放弃"));
    updateControlStates();
}

QString WorkpieceLibraryPage::browsedWorkpieceId() const {
    return browsedWorkpieceId_;
}

void WorkpieceLibraryPage::chooseFrontTemplates() {
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, QStringLiteral("选择正面模板"), QString(), kImageFilter);
    if (paths.isEmpty()) return;
    frontTemplatePaths_ = normalizedPaths(paths);
    setDirty(true);
    updateTemplateUi();
    updateControlStates();
}

void WorkpieceLibraryPage::chooseBackTemplates() {
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, QStringLiteral("选择反面模板"), QString(), kImageFilter);
    if (paths.isEmpty()) return;
    backTemplatePaths_ = normalizedPaths(paths);
    setDirty(true);
    updateTemplateUi();
    updateControlStates();
}

void WorkpieceLibraryPage::submitRegistration() {
    QString error;
    if (!validateRegistration(&error)) {
        showMessage(error, true);
        return;
    }
    if (backendState_ != BackendUiState::Ready) {
        showMessage(QStringLiteral("后端尚未就绪"), true);
        return;
    }
    savedRegistrationFields_ = registrationFields(false);
    registrationInFlight_ = true;
    startRegistrationProgress();
    publishRegistrationTaskStatus(
        QStringLiteral("queued"), 0,
        frontTemplatePaths_.size() + backTemplatePaths_.size(), 0);
    updateControlStates();
    emit commandRequested(QStringLiteral("register"), savedRegistrationFields_);
    showMessage(QStringLiteral("正在建立工件库…"));
}

void WorkpieceLibraryPage::refreshWorkpieces() {
    emit commandRequested(QStringLiteral("list_workpieces"), QJsonObject());
}

void WorkpieceLibraryPage::filterWorkpieces() {
    rebuildWorkpieceList();
    updateControlStates();
}

void WorkpieceLibraryPage::browseSelectedWorkpiece() {
    QListWidgetItem *item = ui->libraryWorkpieceList->currentItem();
    const QString id = item == nullptr ? QString() : item->data(Qt::UserRole).toString();
    if (id.isEmpty() || id == browsedWorkpieceId_) {
        updateControlStates();
        return;
    }
    browsedWorkpieceId_ = id;
    workpieceDetails_ = QJsonObject();
    ui->templateDetailsTable->setRowCount(0);
    ui->workpieceDetailsSummaryLabel->setText(QStringLiteral("正在加载工件详情…"));
    ui->recycleNameConfirmationEdit->clear();
    updateControlStates();
    emit commandRequested(QStringLiteral("get_workpiece_details"), {
        {QStringLiteral("workpiece_id"), browsedWorkpieceId_},
    });
}

void WorkpieceLibraryPage::activateBrowsedWorkpiece() {
    if (!browsedWorkpieceId_.isEmpty()) {
        emit setCurrentWorkpieceRequested(browsedWorkpieceId_);
    }
}

void WorkpieceLibraryPage::recycleBrowsedWorkpiece() {
    if (browsedWorkpieceId_.isEmpty()
        || ui->recycleNameConfirmationEdit->text() != browsedDisplayName()) {
        return;
    }
    emit commandRequested(QStringLiteral("recycle_workpiece"), {
        {QStringLiteral("workpiece_id"), browsedWorkpieceId_},
        {QStringLiteral("operation_id"),
         QUuid::createUuid().toString(QUuid::WithoutBraces)},
    });
    showMessage(QStringLiteral("正在将工件移入回收区…"));
}

void WorkpieceLibraryPage::updateRegistrationElapsed() {
    if (!registrationInFlight_ || !registrationElapsedClock_.isValid()) return;
    ui->registrationElapsedLabel->setText(
        QStringLiteral("耗时：%1 ms").arg(registrationElapsedClock_.elapsed()));
}

QStringList WorkpieceLibraryPage::normalizedPaths(const QStringList &paths) const {
    QStringList normalized;
    normalized.reserve(paths.size());
    for (const QString &path : paths) {
        const QString absolute = QFileInfo(path).absoluteFilePath();
        normalized.append(absolute.isEmpty() ? path : absolute);
    }
    return normalized;
}

bool WorkpieceLibraryPage::loadValidImage(const QString &path, QImage *image) const {
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix != QStringLiteral("png") && suffix != QStringLiteral("jpg")
        && suffix != QStringLiteral("jpeg") && suffix != QStringLiteral("bmp")) {
        return false;
    }
    QImageReader reader(path);
    if (!reader.canRead()) return false;
    const QImage decoded = reader.read();
    if (decoded.isNull()) return false;
    if (image != nullptr) *image = decoded;
    return true;
}

bool WorkpieceLibraryPage::validateRegistration(QString *error) const {
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
        const QString pathKey = QFileInfo(path).absoluteFilePath().toCaseFolded();
        if (paths.contains(pathKey)) {
            if (error != nullptr) *error = QStringLiteral("模板图片不能重复");
            return false;
        }
        paths.insert(pathKey);
        QImage image;
        if (!loadValidImage(path, &image)) {
            if (error != nullptr) *error = QStringLiteral("模板图片无效或无法读取：%1").arg(path);
            return false;
        }
        const QString contentKey = imageContentKey(image);
        if (contents.contains(contentKey)) {
            if (error != nullptr) *error = QStringLiteral("模板图片内容不能重复");
            return false;
        }
        contents.insert(contentKey);
    }
    return true;
}

QJsonObject WorkpieceLibraryPage::registrationFields(bool replace) const {
    return QJsonObject{
        {QStringLiteral("name"), ui->workpieceNameEdit->text().trimmed()},
        {QStringLiteral("replace"), replace},
        {QStringLiteral("front_images"), QJsonArray::fromStringList(frontTemplatePaths_)},
        {QStringLiteral("back_images"), QJsonArray::fromStringList(backTemplatePaths_)},
        {QStringLiteral("progress_events"), true},
    };
}

bool WorkpieceLibraryPage::draftMatchesSavedRegistration() const {
    if (savedRegistrationFields_.isEmpty()) return false;
    QJsonObject current = registrationFields(
        savedRegistrationFields_.value(QStringLiteral("replace")).toBool());
    return current == savedRegistrationFields_;
}

void WorkpieceLibraryPage::updateTemplateUi() {
    ui->frontTemplatesLabel->setText(
        QStringLiteral("正面已选择 %1 张").arg(frontTemplatePaths_.size()));
    ui->backTemplatesLabel->setText(
        QStringLiteral("反面已选择 %1 张").arg(backTemplatePaths_.size()));
    ui->frontTemplatesFilesLabel->setText(templatePathSummary(frontTemplatePaths_));
    ui->backTemplatesFilesLabel->setText(templatePathSummary(backTemplatePaths_));
    ui->frontTemplatesFilesLabel->setToolTip(frontTemplatePaths_.join(QLatin1Char('\n')));
    ui->backTemplatesFilesLabel->setToolTip(backTemplatePaths_.join(QLatin1Char('\n')));
    QStringList warnings;
    if (frontTemplatePaths_.size() < 3 || backTemplatePaths_.size() < 3) {
        warnings.append(QStringLiteral("模板较少，建议补充更多角度，但仍可建库"));
    }
    if (frontTemplatePaths_.size() + backTemplatePaths_.size() > 30) {
        warnings.append(QStringLiteral("模板较多，建库和检测耗时可能增加"));
    }
    ui->templateWarningLabel->setText(warnings.join(QLatin1Char('\n')));
    ui->templateWarningLabel->setVisible(!warnings.isEmpty());
}

void WorkpieceLibraryPage::updateControlStates() {
    const bool ready = backendState_ == BackendUiState::Ready;
    const bool interactive = ready && !registrationInFlight_;
    ui->refreshWorkpiecesButton->setEnabled(interactive);
    ui->chooseFrontTemplatesButton->setEnabled(interactive);
    ui->chooseBackTemplatesButton->setEnabled(interactive);
    ui->registerButton->setEnabled(
        interactive && !ui->workpieceNameEdit->text().trimmed().isEmpty()
        && !frontTemplatePaths_.isEmpty() && !backTemplatePaths_.isEmpty());
    ui->setCurrentWorkpieceButton->setEnabled(
        !browsedWorkpieceId_.isEmpty()
        && browsedWorkpieceId_ != currentDetectionWorkpieceId_);
    ui->deleteWorkpieceButton->setEnabled(
        interactive && !browsedWorkpieceId_.isEmpty()
        && ui->recycleNameConfirmationEdit->text() == browsedDisplayName());
    ui->registerButton->setToolTip(!ready
        ? QStringLiteral("后端未就绪")
        : (registrationInFlight_ ? QStringLiteral("建库任务正在运行")
                                 : QStringLiteral("使用全部已选模板建立工件库")));
    ui->deleteWorkpieceButton->setToolTip(ui->deleteWorkpieceButton->isEnabled()
        ? QStringLiteral("将当前浏览工件移入可恢复回收区")
        : QStringLiteral("请输入完整工件名称后才能移入回收区"));
}

void WorkpieceLibraryPage::setDirty(bool dirty) {
    if (dirty_ == dirty) return;
    dirty_ = dirty;
    emit dirtyChanged(dirty_);
}

void WorkpieceLibraryPage::showMessage(const QString &message, bool error) {
    ui->libraryMessageLabel->setProperty(
        "messageKind", error ? QStringLiteral("error") : QStringLiteral("neutral"));
    ui->libraryMessageLabel->style()->unpolish(ui->libraryMessageLabel);
    ui->libraryMessageLabel->style()->polish(ui->libraryMessageLabel);
    ui->libraryMessageLabel->setText(message);
}

void WorkpieceLibraryPage::rebuildWorkpieceList() {
    const QString filter = ui->librarySearchEdit->text().trimmed();
    const QSignalBlocker blocker(ui->libraryWorkpieceList);
    ui->libraryWorkpieceList->clear();
    int selectedRow = -1;
    for (const QJsonValue &value : workpieces_) {
        const QJsonObject item = value.toObject();
        const QString id = item.value(QStringLiteral("id")).toString();
        const QString name = item.value(QStringLiteral("name")).toString();
        if (!filter.isEmpty() && !name.contains(filter, Qt::CaseInsensitive)
            && !id.contains(filter, Qt::CaseInsensitive)) {
            continue;
        }
        const QJsonObject counts = item.value(QStringLiteral("template_counts")).toObject();
        const QString text = QStringLiteral("%1%2\n正 %3 / 反 %4 · 规则 %5 · %6 · %7")
                                 .arg(name,
                                      id == currentDetectionWorkpieceId_
                                          ? QStringLiteral("  [当前检测]") : QString())
                                 .arg(counts.value(QStringLiteral("front")).toInt())
                                 .arg(counts.value(QStringLiteral("back")).toInt())
                                 .arg(item.value(QStringLiteral("geometry_rule_count")).toInt())
                                 .arg(item.value(QStringLiteral("detectable")).toBool()
                                          ? QStringLiteral("可检测") : QStringLiteral("不可检测"))
                                 .arg(item.value(QStringLiteral("updated_at")).toString());
        auto *listItem = new QListWidgetItem(text, ui->libraryWorkpieceList);
        listItem->setData(Qt::UserRole, id);
        listItem->setToolTip(QStringLiteral("%1\nID: %2").arg(text, id));
        if (id == browsedWorkpieceId_) selectedRow = ui->libraryWorkpieceList->count() - 1;
    }
    ui->libraryWorkpieceList->setCurrentRow(selectedRow);
    if (selectedRow < 0 && !browsedWorkpieceId_.isEmpty()) {
        browsedWorkpieceId_.clear();
        workpieceDetails_ = QJsonObject();
        ui->templateDetailsTable->setRowCount(0);
        ui->workpieceDetailsSummaryLabel->setText(QStringLiteral("请选择工件查看详情"));
        ui->recycleNameConfirmationEdit->clear();
    }
}

void WorkpieceLibraryPage::rebuildEvolutionTable() {
    ui->evolutionJobsTable->setRowCount(evolutionJobOrder_.size());
    for (int row = 0; row < evolutionJobOrder_.size(); ++row) {
        const QString jobId = evolutionJobOrder_.at(row);
        const QJsonObject job = evolutionJobsById_.value(jobId);
        const int completed = job.value(QStringLiteral("completed")).toInt();
        const int total = job.value(QStringLiteral("total")).toInt();
        ui->evolutionJobsTable->setItem(row, 0, new QTableWidgetItem(jobId));
        ui->evolutionJobsTable->setItem(
            row, 1, new QTableWidgetItem(job.value(QStringLiteral("workpiece_id")).toString()));
        ui->evolutionJobsTable->setItem(
            row, 2, new QTableWidgetItem(job.value(QStringLiteral("state")).toString()));
        ui->evolutionJobsTable->setItem(
            row, 3, new QTableWidgetItem(phaseText(
                        job.value(QStringLiteral("phase")).toString())));
        ui->evolutionJobsTable->setItem(
            row, 4, new QTableWidgetItem(QStringLiteral("%1/%2").arg(completed).arg(total)));
        ui->evolutionJobsTable->setItem(
            row, 5, new QTableWidgetItem(job.value(QStringLiteral("error")).toString()));
    }
    ui->evolutionJobsTable->resizeColumnsToContents();
}

QString WorkpieceLibraryPage::browsedDisplayName() const {
    for (const QJsonValue &value : workpieces_) {
        const QJsonObject item = value.toObject();
        if (item.value(QStringLiteral("id")).toString() == browsedWorkpieceId_) {
            return item.value(QStringLiteral("name")).toString();
        }
    }
    return QString();
}

void WorkpieceLibraryPage::startRegistrationProgress() {
    registrationElapsedClock_.start();
    registrationElapsedTimer_.start();
    const int total = frontTemplatePaths_.size() + backTemplatePaths_.size();
    ui->registrationProgressBar->setRange(0, qMax(1, total));
    ui->registrationProgressBar->setValue(0);
    ui->registrationProgressLabel->setText(
        QStringLiteral("建库进度：准备中（共 %1 张）").arg(total));
    ui->registrationElapsedLabel->setText(QStringLiteral("耗时：0 ms"));
}

void WorkpieceLibraryPage::stopRegistrationProgress() {
    registrationElapsedTimer_.stop();
    if (registrationElapsedClock_.isValid()) {
        ui->registrationElapsedLabel->setText(
            QStringLiteral("耗时：%1 ms").arg(registrationElapsedClock_.elapsed()));
    }
}

void WorkpieceLibraryPage::publishRegistrationTaskStatus(
    const QString &phase, int completed, int total, qint64 elapsedMs) {
    registrationTaskPhase_ = phase;
    registrationTaskCompleted_ = completed;
    registrationTaskTotal_ = total;
    registrationTaskElapsedMs_ = elapsedMs;
    emit taskStatusChanged(QStringLiteral("建立工件库"), phase,
                           completed, total, elapsedMs);
}

void WorkpieceLibraryPage::publishRegistrationFailure() {
    publishRegistrationTaskStatus(QStringLiteral("failed"),
                                  registrationTaskCompleted_,
                                  registrationTaskTotal_,
                                  currentRegistrationElapsedMs());
}

qint64 WorkpieceLibraryPage::currentRegistrationElapsedMs() const {
    if (!registrationElapsedClock_.isValid()) return registrationTaskElapsedMs_;
    return qMax(registrationTaskElapsedMs_, registrationElapsedClock_.elapsed());
}
