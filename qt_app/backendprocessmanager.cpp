#include "backendprocessmanager.h"

#include <QJsonObject>
#include <QTimer>
#include <QtGlobal>

#include "backendclient.h"
#include "processlauncher.h"

BackendProcessManager::BackendProcessManager(const AppConfig &config, BackendClient *client,
                                             ProcessLauncher *launcher, QObject *parent)
    : QObject(parent), config_(config), client_(client), launcher_(launcher),
      retryTimer_(new QTimer(this)), startupTimer_(new QTimer(this)),
      stopEscalationTimer_(new QTimer(this)) {
    if (launcher_ == nullptr) {
        launcher_ = new QProcessLauncher(this);
        ownsLauncher_ = true;
    }
    retryTimer_->setInterval(500);
    startupTimer_->setSingleShot(true);
    stopEscalationTimer_->setSingleShot(true);
    connect(retryTimer_, &QTimer::timeout, this, &BackendProcessManager::tryConnect);
    connect(startupTimer_, &QTimer::timeout, this, &BackendProcessManager::onStartupTimeout);
    connect(stopEscalationTimer_, &QTimer::timeout,
            this, &BackendProcessManager::onStopEscalationTimeout);
    connect(client_, &BackendClient::handshakeSucceeded, this, &BackendProcessManager::onHandshakeSucceeded);
    connect(client_, &BackendClient::transportFailed, this, &BackendProcessManager::onTransportFailed);
    connect(client_, &BackendClient::responseReceived, this, &BackendProcessManager::onResponseReceived);
    connect(launcher_, &ProcessLauncher::failed, this, &BackendProcessManager::onProcessFailed);
    connect(launcher_, &ProcessLauncher::finished, this, &BackendProcessManager::onProcessFinished);
}

BackendProcessManager::~BackendProcessManager() {
    shutdownOwnedService();
}

void BackendProcessManager::start() {
    shuttingDown_ = false;
    launchRequested_ = false;
    startupTimer_->start(config_.startupTimeoutMs);
    tryConnect();
}

void BackendProcessManager::restart() {
    if (shuttingDown_ || restartPhase_ != RestartPhase::Idle) {
        return;
    }
    retryTimer_->stop();
    startupTimer_->stop();
    if (!owned_) {
        launchRequested_ = false;
        startupTimer_->start(config_.startupTimeoutMs);
        client_->connectToService(config_.host, config_.port, config_.requestTimeoutMs);
        return;
    }
    if (!launcher_->isRunning()) {
        owned_ = false;
        launchRequested_ = false;
        emit serviceOwnershipChanged(false);
        startupTimer_->start(config_.startupTimeoutMs);
        launchBackend();
        return;
    }
    restartPhase_ = RestartPhase::GracefulStop;
    if (client_->state() == BackendClient::State::Ready) {
        client_->sendRequest(QStringLiteral("shutdown"));
    }
    stopEscalationTimer_->start(stopEscalationIntervalMs());
}

void BackendProcessManager::shutdownOwnedService() {
    if (!owned_ || shuttingDown_) {
        return;
    }
    shuttingDown_ = true;
    retryTimer_->stop();
    startupTimer_->stop();
    stopEscalationTimer_->stop();
    restartPhase_ = RestartPhase::Idle;
    if (client_->state() == BackendClient::State::Ready) {
        client_->sendRequest(QStringLiteral("shutdown"));
    } else if (launcher_->isRunning()) {
        stoppingOwnedProcess_ = true;
        launcher_->terminate();
    }
}

bool BackendProcessManager::ownedByThisSession() const {
    return owned_;
}

void BackendProcessManager::tryConnect() {
    if (client_->state() == BackendClient::State::Ready
        || client_->state() == BackendClient::State::Connecting
        || client_->state() == BackendClient::State::Handshaking
        || client_->state() == BackendClient::State::Busy) {
        return;
    }
    client_->connectToService(config_.host, config_.port, config_.requestTimeoutMs);
}

void BackendProcessManager::onHandshakeSucceeded() {
    retryTimer_->stop();
    startupTimer_->stop();
    emit backendReady();
}

void BackendProcessManager::onTransportFailed(const QString &code, const QString &message) {
    if (shuttingDown_ || restartPhase_ != RestartPhase::Idle) {
        return;
    }
    if (code == QStringLiteral("MODEL_LOADING")) {
        emit backendLoading(message);
        retryTimer_->start();
        return;
    }
    if (code == QStringLiteral("HANDSHAKE_FAILED") || code == QStringLiteral("SERVER_BUSY")) {
        markUnavailable(message);
        return;
    }
    if (code == QStringLiteral("CONNECTION_ERROR") || code == QStringLiteral("CONNECTION_LOST")
        || code == QStringLiteral("TIMEOUT")) {
        if (!launchRequested_) {
            launchBackend();
        }
        retryTimer_->start();
        return;
    }
    markUnavailable(message);
}

void BackendProcessManager::onResponseReceived(const QString &command, const QJsonObject &response) {
    Q_UNUSED(response)
    if (restartPhase_ != RestartPhase::Idle && command == QStringLiteral("shutdown")) {
        return;
    }
    if (shuttingDown_ && command == QStringLiteral("shutdown")) {
        shuttingDown_ = false;
        owned_ = false;
        emit serviceOwnershipChanged(false);
        client_->disconnectFromService();
        if (launcher_->isRunning()) {
            stoppingOwnedProcess_ = true;
            launcher_->terminate();
        }
    }
}

void BackendProcessManager::onStartupTimeout() {
    if (restartPhase_ != RestartPhase::Idle) {
        return;
    }
    retryTimer_->stop();
    if (!launchRequested_) {
        launchBackend();
        startupTimer_->start(config_.startupTimeoutMs);
        return;
    }
    if (owned_ && launcher_->isRunning()) {
        stoppingOwnedProcess_ = true;
        launcher_->terminate();
    }
    markUnavailable(QStringLiteral("后端启动超时"));
}

void BackendProcessManager::onStopEscalationTimeout() {
    if (!launcher_->isRunning()) {
        return;
    }
    if (restartPhase_ == RestartPhase::GracefulStop) {
        restartPhase_ = RestartPhase::TerminateWait;
        launcher_->terminate();
        if (restartPhase_ == RestartPhase::TerminateWait && launcher_->isRunning()) {
            stopEscalationTimer_->start(stopEscalationIntervalMs());
        }
        return;
    }
    if (restartPhase_ == RestartPhase::TerminateWait) {
        restartPhase_ = RestartPhase::KillWait;
        launcher_->kill();
    }
}

void BackendProcessManager::onProcessFailed(const QString &message) {
    if (restartPhase_ != RestartPhase::Idle) {
        return;
    }
    retryTimer_->stop();
    markUnavailable(QStringLiteral("后端进程启动失败：%1").arg(message));
}

void BackendProcessManager::onProcessFinished(int exitCode) {
    if (restartPhase_ != RestartPhase::Idle) {
        Q_UNUSED(exitCode)
        relaunchAfterRestartExit();
        return;
    }
    if (stoppingOwnedProcess_) {
        stoppingOwnedProcess_ = false;
        return;
    }
    if (!shuttingDown_ && client_->state() != BackendClient::State::Ready) {
        markUnavailable(QStringLiteral("后端进程已退出，代码 %1").arg(exitCode));
    }
}

void BackendProcessManager::relaunchAfterRestartExit() {
    stopEscalationTimer_->stop();
    client_->disconnectFromService();
    restartPhase_ = RestartPhase::Idle;
    stoppingOwnedProcess_ = false;
    if (owned_) {
        owned_ = false;
        emit serviceOwnershipChanged(false);
    }
    launchRequested_ = false;
    startupTimer_->start(config_.startupTimeoutMs);
    launchBackend();
}

void BackendProcessManager::launchBackend() {
    launchRequested_ = true;
    owned_ = true;
    emit serviceOwnershipChanged(true);
    if (!launcher_->start(config_.pythonExecutable, backendArguments(), config_.projectRoot)) {
        markUnavailable(QStringLiteral("无法启动后端进程"));
        return;
    }
    retryTimer_->start();
}

void BackendProcessManager::markUnavailable(const QString &reason) {
    emit backendUnavailable(reason);
}

QStringList BackendProcessManager::backendArguments() const {
    return {
        config_.backendScript,
        QStringLiteral("--host"), config_.host.toString(),
        QStringLiteral("--port"), QString::number(config_.port),
        QStringLiteral("--project-root"), config_.projectRoot,
        QStringLiteral("--model-dir"), config_.modelDir,
        QStringLiteral("--library-dir"), config_.libraryDir,
        QStringLiteral("--local-search-mode"), config_.localSearchMode,
    };
}

int BackendProcessManager::stopEscalationIntervalMs() const {
    return qBound(100, config_.requestTimeoutMs, 2000);
}
