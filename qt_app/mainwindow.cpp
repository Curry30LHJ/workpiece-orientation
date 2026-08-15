#include "mainwindow.h"

#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonObject>
#include <QMessageBox>
#include <QPixmap>
#include <QSet>
#include <QTimer>

#include "backendclient.h"
#include "backendprocessmanager.h"

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
    ui->registerButton->setEnabled(false);
    ui->predictButton->setEnabled(false);
    ui->refreshWorkpiecesButton->setEnabled(false);
    ui->chooseFrontTemplatesButton->setEnabled(false);
    ui->chooseBackTemplatesButton->setEnabled(false);
    ui->chooseImageButton->setEnabled(false);
    ui->restartBackendButton->setEnabled(manager_ != nullptr);
    ui->resultLabel->setText(QStringLiteral("尚未检测"));
    ui->reviewLabel->clear();
    ui->evidenceTextEdit->clear();
    ui->libraryMessageLabel->clear();
    updateTemplateLabels();
    updateButtonStates();
    connect(ui->chooseFrontTemplatesButton, &QPushButton::clicked,
            this, &MainWindow::chooseFrontTemplates);
    connect(ui->chooseBackTemplatesButton, &QPushButton::clicked,
            this, &MainWindow::chooseBackTemplates);
    connect(ui->chooseImageButton, &QPushButton::clicked,
            this, &MainWindow::chooseInspectionImage);
    connect(ui->refreshWorkpiecesButton, &QPushButton::clicked,
            this, &MainWindow::refreshWorkpieces);
    connect(ui->registerButton, &QPushButton::clicked,
            this, &MainWindow::submitRegistration);
    connect(ui->predictButton, &QPushButton::clicked,
            this, &MainWindow::submitPrediction);
    connect(ui->restartBackendButton, &QPushButton::clicked,
            this, &MainWindow::restartBackend);
}

void MainWindow::connectBackendSignals() {
    if (client_ == nullptr) {
        return;
    }
    connect(client_, &BackendClient::handshakeSucceeded,
            this, &MainWindow::onBackendReady);
    connect(client_, &BackendClient::stateChanged,
            this, &MainWindow::onClientStateChanged);
    connect(client_, &BackendClient::responseReceived,
            this, &MainWindow::onClientResponse);
    connect(client_, &BackendClient::requestFailed,
            this, &MainWindow::onClientRequestFailed);
    connect(client_, &BackendClient::connectionLost,
            this, &MainWindow::onConnectionLost);
    if (manager_ != nullptr) {
        connect(manager_, &BackendProcessManager::backendReady,
                this, &MainWindow::onBackendReady);
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
    updatePreview();
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
        this, QStringLiteral("选择正面模板（5 张）"), QString(), dialogFilters().constFirst());
    if (!paths.isEmpty()) {
        frontTemplatePaths_ = normalizedPaths(paths);
        updateTemplateLabels();
        updateButtonStates();
    }
}

void MainWindow::chooseBackTemplates() {
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, QStringLiteral("选择反面模板（5 张）"), QString(), dialogFilters().constFirst());
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

void MainWindow::refreshWorkpieces() {
    if (client_ == nullptr || !backendReady_ || clientBusy_
        || client_->state() != BackendClient::State::Ready) {
        return;
    }
    pendingCommand_ = QStringLiteral("list_workpieces");
    client_->sendRequest(QStringLiteral("list_workpieces"));
    showLibraryMessage(QStringLiteral("正在刷新工件列表…"));
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
    pendingCommand_ = QStringLiteral("predict");
    client_->sendRequest(QStringLiteral("predict"), {
        {QStringLiteral("workpiece_id"), workpieceId},
        {QStringLiteral("image_path"), inspectionImagePath_},
    });
    ui->resultLabel->setText(QStringLiteral("正在检测…"));
    ui->reviewLabel->clear();
    ui->evidenceTextEdit->clear();
}

void MainWindow::restartBackend() {
    if (manager_ != nullptr) {
        manager_->restart();
    }
}

void MainWindow::onBackendReady() {
    backendReady_ = true;
    clientBusy_ = false;
    ui->backendStatusLabel->setText(QStringLiteral("后端：已连接"));
    ui->restartBackendButton->setEnabled(true);
    updateButtonStates();
    QTimer::singleShot(0, this, [this]() { refreshWorkpieces(); });
}

void MainWindow::onBackendUnavailable(const QString &reason) {
    backendReady_ = false;
    clientBusy_ = false;
    ui->backendStatusLabel->setText(QStringLiteral("后端：不可用"));
    showLibraryMessage(reason, true);
    clearInspectionState();
    ui->restartBackendButton->setEnabled(true);
    updateButtonStates();
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

void MainWindow::onClientResponse(const QString &command, const QJsonObject &response) {
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
        showLibraryMessage(QStringLiteral("工件列表已刷新"));
        updateButtonStates();
        return;
    }
    if (command == QStringLiteral("register")) {
        pendingCommand_.clear();
        registrationInFlight_ = false;
        showLibraryMessage(QStringLiteral("工件库建立成功"));
        QTimer::singleShot(0, this, [this]() { refreshWorkpieces(); });
        return;
    }
    if (command == QStringLiteral("predict")) {
        const QString label = response.value(QStringLiteral("label")).toString();
        ui->resultLabel->setText(QStringLiteral("检测结果：%1").arg(orientationText(label)));
        QStringList lines;
        lines << QStringLiteral("全局得分：正面 %1，反面 %2")
                     .arg(formatScore(response.value(QStringLiteral("global_scores")).toObject(), QStringLiteral("front")),
                          formatScore(response.value(QStringLiteral("global_scores")).toObject(), QStringLiteral("back")));
        lines << QStringLiteral("全局间隔：%1").arg(response.value(QStringLiteral("global_margin")).toDouble());
        lines << QStringLiteral("局部预测：%1").arg(orientationText(response.value(QStringLiteral("local_prediction")).toString()));
        lines << QStringLiteral("局部得分：正面 %1，反面 %2")
                     .arg(formatScore(response.value(QStringLiteral("local_scores")).toObject(), QStringLiteral("front")),
                          formatScore(response.value(QStringLiteral("local_scores")).toObject(), QStringLiteral("back")));
        lines << QStringLiteral("局部间隔：%1").arg(response.value(QStringLiteral("local_margin")).toDouble());
        lines << QStringLiteral("决策来源：%1").arg(decisionSourceText(response.value(QStringLiteral("decision_source")).toString()));
        lines << QStringLiteral("耗时（毫秒）：%1").arg(response.value(QStringLiteral("elapsed_ms")).toDouble());
        ui->evidenceTextEdit->setPlainText(lines.join(QLatin1Char('\n')));
        ui->reviewLabel->setText(response.value(QStringLiteral("needs_review")).toBool()
                                     ? QStringLiteral("建议人工复检") : QString());
        pendingCommand_.clear();
    }
}

void MainWindow::onClientRequestFailed(const QString &code, const QString &message) {
    if (registrationInFlight_ && code == QStringLiteral("WORKPIECE_EXISTS")
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
            registrationInFlight_ = false;
            pendingCommand_.clear();
            showLibraryMessage(QStringLiteral("已取消覆盖"));
        }
        return;
    }
    if (registrationInFlight_ || pendingCommand_ == QStringLiteral("register")) {
        registrationInFlight_ = false;
        pendingCommand_.clear();
        showLibraryMessage(message, true);
    } else if (pendingCommand_ == QStringLiteral("predict")) {
        pendingCommand_.clear();
        ui->resultLabel->setText(QStringLiteral("检测失败：%1").arg(message));
    } else {
        showLibraryMessage(message, true);
    }
    updateButtonStates();
}

void MainWindow::onConnectionLost(const QString &reason) {
    onBackendUnavailable(reason);
}

void MainWindow::updateButtonStates() {
    const bool interactive = backendReady_ && !clientBusy_;
    ui->refreshWorkpiecesButton->setEnabled(interactive);
    ui->chooseFrontTemplatesButton->setEnabled(interactive);
    ui->chooseBackTemplatesButton->setEnabled(interactive);
    ui->chooseImageButton->setEnabled(interactive);
    ui->registerButton->setEnabled(interactive && !ui->workpieceNameEdit->text().trimmed().isEmpty()
                                   && frontTemplatePaths_.size() == 5 && backTemplatePaths_.size() == 5);
    ui->predictButton->setEnabled(interactive && !selectedWorkpieceId().isEmpty()
                                  && !inspectionImagePath_.isEmpty());
}

void MainWindow::updateTemplateLabels() {
    ui->frontTemplatesLabel->setText(QStringLiteral("正面模板：已选择（%1/5）").arg(frontTemplatePaths_.size()));
    ui->backTemplatesLabel->setText(QStringLiteral("反面模板：已选择（%1/5）").arg(backTemplatePaths_.size()));
    ui->frontTemplatesFilesLabel->setText(frontTemplatePaths_.join(QLatin1Char('\n')));
    ui->backTemplatesFilesLabel->setText(backTemplatePaths_.join(QLatin1Char('\n')));
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
        ui->imagePreviewLabel->setText(QStringLiteral("图片无法读取"));
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
    updateButtonStates();
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
    });
    showLibraryMessage(replace ? QStringLiteral("正在覆盖并建立工件库…") : QStringLiteral("正在建立工件库…"));
}

bool MainWindow::validateRegistration(QString *error) const {
    if (ui->workpieceNameEdit->text().trimmed().isEmpty()) {
        if (error != nullptr) *error = QStringLiteral("请输入工件名称");
        return false;
    }
    if (frontTemplatePaths_.size() != 5 || backTemplatePaths_.size() != 5) {
        if (error != nullptr) *error = QStringLiteral("正面和反面都必须选择 5 张图片");
        return false;
    }
    QSet<QString> paths;
    for (const QString &path : frontTemplatePaths_ + backTemplatePaths_) {
        const QString absolute = QFileInfo(path).absoluteFilePath();
        if (paths.contains(absolute)) {
            if (error != nullptr) *error = QStringLiteral("模板图片不能重复");
            return false;
        }
        paths.insert(absolute);
        if (!QFileInfo::exists(absolute)) {
            if (error != nullptr) *error = QStringLiteral("模板图片不存在：%1").arg(absolute);
            return false;
        }
    }
    return true;
}

bool MainWindow::validateImagePath(const QString &path, QString *error) const {
    if (path.isEmpty()) {
        if (error != nullptr) *error = QStringLiteral("请选择待测图片");
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
        if (!absolute.isEmpty() && !normalized.contains(absolute)) {
            normalized.append(absolute);
        }
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
