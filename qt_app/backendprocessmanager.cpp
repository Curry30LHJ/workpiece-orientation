#include "backendprocessmanager.h"

#include <QJsonObject>
#include <QTimer>

#include "backendclient.h"
#include "processlauncher.h"

BackendProcessManager::BackendProcessManager(const AppConfig &config, BackendClient *client,
                                             ProcessLauncher *launcher, QObject *parent)
    : QObject(parent), config_(config), client_(client), launcher_(launcher),
      retryTimer_(new QTimer(this)), startupTimer_(new QTimer(this)) {
    if (launcher_ == nullptr) {
        launcher_ = new QProcessLauncher(this);
        ownsLauncher_ = true;
    }
    retryTimer_->setInterval(50);
    startupTimer_->setSingleShot(true);
    connect(retryTimer_, &QTimer::timeout, this, &BackendProcessManager::tryConnect);
    connect(startupTimer_, &QTimer::timeout, this, &BackendProcessManager::onStartupTimeout);
    connect(client_, &BackendClient::handshakeSucceeded, this, &BackendProcessManager::onHandshakeSucceeded);
    connect(client_, &BackendClient::requestFailed, this, &BackendProcessManager::onRequestFailed);
    connect(client_, &BackendClient::connectionLost, this, &BackendProcessManager::onConnectionLost);
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
    if (client_->state() == BackendClient::State::Ready) {
        client_->connectToService(config_.host, config_.port, config_.requestTimeoutMs);
        return;
    }
    if (owned_ && launcher_->isRunning()) {
        stoppingOwnedProcess_ = true;
        launcher_->terminate();
    }
    owned_ = false;
    launchRequested_ = false;
    emit serviceOwnershipChanged(false);
    start();
}

void BackendProcessManager::shutdownOwnedService() {
    if (!owned_ || shuttingDown_) {
        return;
    }
    shuttingDown_ = true;
    retryTimer_->stop();
    startupTimer_->stop();
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

void BackendProcessManager::onRequestFailed(const QString &code, const QString &message) {
    if (shuttingDown_) {
        return;
    }
    if (code == QStringLiteral("HANDSHAKE_FAILED") || code == QStringLiteral("SERVER_BUSY")) {
        markUnavailable(message);
        return;
    }
    if (code == QStringLiteral("CONNECTION_ERROR") || code == QStringLiteral("TIMEOUT")) {
        if (!launchRequested_) {
            launchBackend();
        }
        retryTimer_->start();
        return;
    }
    markUnavailable(message);
}

void BackendProcessManager::onConnectionLost(const QString &reason) {
    if (!shuttingDown_) {
        markUnavailable(reason);
    }
}

void BackendProcessManager::onResponseReceived(const QString &command, const QJsonObject &response) {
    Q_UNUSED(response)
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

void BackendProcessManager::onProcessFailed(const QString &message) {
    retryTimer_->stop();
    markUnavailable(QStringLiteral("后端进程启动失败：%1").arg(message));
}

void BackendProcessManager::onProcessFinished(int exitCode) {
    if (stoppingOwnedProcess_) {
        stoppingOwnedProcess_ = false;
        return;
    }
    if (!shuttingDown_ && client_->state() != BackendClient::State::Ready) {
        markUnavailable(QStringLiteral("后端进程已退出，代码 %1").arg(exitCode));
    }
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
