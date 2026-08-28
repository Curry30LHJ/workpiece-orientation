#include "backendprocessmanager.h"

#include <QAbstractSocket>
#include <QCoreApplication>
#include <QDir>
#include <QJsonObject>
#include <QTimer>
#include <QUuid>
#include <QtGlobal>

#include "backendclient.h"
#include "processlauncher.h"

namespace {
const QString kIdentityAction = QStringLiteral(
    "请确认后端版本、版本类型、计算设备和模型文件与当前应用一致");
const QString kConflictAction = QStringLiteral(
    "请关闭占用该端口的其他后端服务后重试");
const QString kStartAction = QStringLiteral(
    "请检查后端程序及依赖文件是否完整，然后重新启动应用");
const QString kTimeoutAction = QStringLiteral("请查看后端日志并重试");
}

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
    connect(client_, &BackendClient::handshakeLoading,
            this, &BackendProcessManager::onHandshakeLoading);
    connect(client_, &BackendClient::handshakeSucceeded,
            this, &BackendProcessManager::onHandshakeSucceeded);
    connect(client_, &BackendClient::transportFailed,
            this, &BackendProcessManager::onTransportFailed);
    connect(client_, &BackendClient::stateChanged, this,
            [this](BackendClient::State state, const QString &) {
        if (!launchRequested_ && state == BackendClient::State::Handshaking) {
            prelaunchProbeConnected_ = true;
        }
    });
    connect(client_, &BackendClient::responseReceived,
            this, &BackendProcessManager::onResponseReceived);
    connect(launcher_, &ProcessLauncher::failed, this, &BackendProcessManager::onProcessFailed);
    connect(launcher_, &ProcessLauncher::finished, this, &BackendProcessManager::onProcessFinished);
}

BackendProcessManager::~BackendProcessManager() {
    shutdownOwnedService();
}

void BackendProcessManager::start() {
    ++startupGeneration_;
    retryTimer_->stop();
    startupTimer_->stop();
    stopEscalationTimer_->stop();
    client_->disconnectFromService();
    shuttingDown_ = false;
    terminalFailure_ = false;
    launchRequested_ = false;
    launchedProcess_ = false;
    prelaunchProbeConnected_ = false;
    reusingExternalDevelopmentService_ = false;
    launchInstanceToken_.clear();
    readyInstanceToken_.clear();
    backendProgress_ = 0;
    startupTimer_->start(config_.startupTimeoutMs);
    tryConnect();
}

void BackendProcessManager::restart() {
    if (shuttingDown_ || restartPhase_ != RestartPhase::Idle) {
        return;
    }
    ++startupGeneration_;
    retryTimer_->stop();
    startupTimer_->stop();
    terminalFailure_ = false;
    backendProgress_ = 0;
    if (!owned_) {
        client_->disconnectFromService();
        launchRequested_ = false;
        launchedProcess_ = false;
        prelaunchProbeConnected_ = false;
        launchInstanceToken_.clear();
        readyInstanceToken_.clear();
        startupTimer_->start(config_.startupTimeoutMs);
        tryConnect();
        return;
    }
    if (!launcher_->isRunning()) {
        client_->disconnectFromService();
        owned_ = false;
        launchedProcess_ = false;
        launchRequested_ = false;
        readyInstanceToken_.clear();
        emit serviceOwnershipChanged(false);
        startupTimer_->start(config_.startupTimeoutMs);
        launchBackend();
        return;
    }
    restartPhase_ = RestartPhase::GracefulStop;
    if (client_->state() == BackendClient::State::Ready && canControlOwnedProcess()) {
        client_->sendRequest(QStringLiteral("shutdown"),
                             {{QStringLiteral("instance_token"), launchInstanceToken_}});
    }
    stopEscalationTimer_->start(stopEscalationIntervalMs());
}

void BackendProcessManager::shutdownOwnedService() {
    ++startupGeneration_;
    retryTimer_->stop();
    startupTimer_->stop();
    stopEscalationTimer_->stop();
    restartPhase_ = RestartPhase::Idle;
    if (shuttingDown_) {
        client_->disconnectFromService();
        return;
    }
    shuttingDown_ = true;
    if (!owned_) {
        client_->disconnectFromService();
        return;
    }
    if (client_->state() == BackendClient::State::Ready && canControlOwnedProcess()) {
        restartPhase_ = RestartPhase::GracefulStop;
        client_->sendRequest(QStringLiteral("shutdown"),
                             {{QStringLiteral("instance_token"), launchInstanceToken_}});
        stopEscalationTimer_->start(stopEscalationIntervalMs());
    } else if (launcher_->isRunning() && canControlOwnedProcess()) {
        stoppingOwnedProcess_ = true;
        launcher_->terminate();
    }
}

bool BackendProcessManager::ownedByThisSession() const {
    return owned_ && canControlOwnedProcess();
}

void BackendProcessManager::tryConnect() {
    if (shuttingDown_ || restartPhase_ != RestartPhase::Idle) {
        return;
    }
    if (client_->state() == BackendClient::State::Ready
        || client_->state() == BackendClient::State::Connecting
        || client_->state() == BackendClient::State::Handshaking
        || client_->state() == BackendClient::State::Busy) {
        return;
    }
    client_->connectToService(config_.host, config_.port,
                              config_.requestTimeoutMs, startupGeneration_);
}

void BackendProcessManager::onHandshakeLoading(quint64 generation,
                                               const QJsonObject &metadata) {
    if (generation != startupGeneration_ || shuttingDown_
        || restartPhase_ != RestartPhase::Idle) {
        return;
    }
    if (config_.launchMode == BackendLaunchMode::PackagedExecutable
        && !launchRequested_) {
        markUnavailable(QStringLiteral("检测到其他后端实例占用当前端口"),
                        QStringLiteral("BACKEND_INSTANCE_CONFLICT"),
                        kConflictAction, configuredLogPath());
        return;
    }
    QString reason;
    if (!identityMatches(metadata, &reason)) {
        owned_ = false;
        launchedProcess_ = false;
        emit serviceOwnershipChanged(false);
        markUnavailable(reason, QStringLiteral("BACKEND_IDENTITY_MISMATCH"),
                        kIdentityAction, configuredLogPath());
        return;
    }
    retryTimer_->stop();
    const QString phase = metadata.value(QStringLiteral("phase")).toString();
    const int progress = metadata.value(QStringLiteral("progress")).toInt(-1);
    if (phase.isEmpty() || progress < 0 || progress > 100) {
        markUnavailable(QStringLiteral("后端加载状态无效"),
                        QStringLiteral("BACKEND_PROTOCOL_ERROR"),
                        kStartAction, configuredLogPath());
        return;
    }
    backendProgress_ = qMax(backendProgress_, progress);
    emit backendLoading(phase,
                        metadata.value(QStringLiteral("message")).toString(),
                        backendProgress_);
}

void BackendProcessManager::onHandshakeSucceeded(quint64 generation,
                                                 const QJsonObject &metadata) {
    if (generation != startupGeneration_ || shuttingDown_
        || restartPhase_ != RestartPhase::Idle) {
        return;
    }
    if (config_.launchMode == BackendLaunchMode::PackagedExecutable
        && !launchRequested_) {
        markUnavailable(QStringLiteral("检测到其他后端实例占用当前端口"),
                        QStringLiteral("BACKEND_INSTANCE_CONFLICT"),
                        kConflictAction, configuredLogPath());
        return;
    }
    QString reason;
    if (!identityMatches(metadata, &reason)) {
        owned_ = false;
        launchedProcess_ = false;
        emit serviceOwnershipChanged(false);
        markUnavailable(reason, QStringLiteral("BACKEND_IDENTITY_MISMATCH"),
                        kIdentityAction, configuredLogPath());
        return;
    }
    retryTimer_->stop();
    startupTimer_->stop();
    if (backendProgress_ == 0) {
        emit backendLoading(QStringLiteral("loading_runtime"),
                            QStringLiteral("正在加载运行环境"), 0);
    }
    backendProgress_ = 100;
    readyInstanceToken_ = metadata.value(QStringLiteral("instance_token")).toString();
    reusingExternalDevelopmentService_ = !launchRequested_
        && config_.launchMode == BackendLaunchMode::PythonScript;
    if (launchRequested_ && launchedProcess_) {
        owned_ = true;
        emit serviceOwnershipChanged(true);
    }
    emit backendReady();
}

void BackendProcessManager::onTransportFailed(quint64 generation, const QString &code,
                                              const QString &message,
                                              const QJsonObject &details) {
    if (generation != startupGeneration_ || shuttingDown_
        || restartPhase_ != RestartPhase::Idle) {
        return;
    }
    if (config_.launchMode == BackendLaunchMode::PackagedExecutable
        && !launchRequested_ && prelaunchProbeConnected_) {
        markUnavailable(QStringLiteral("检测到其他后端实例占用当前端口"),
                        QStringLiteral("BACKEND_INSTANCE_CONFLICT"),
                        kConflictAction, configuredLogPath());
        return;
    }
    const bool connectionRefused = code == QStringLiteral("CONNECTION_ERROR")
        && details.value(QStringLiteral("socket_error")).toInt(-1)
            == int(QAbstractSocket::ConnectionRefusedError);
    if (config_.launchMode == BackendLaunchMode::PackagedExecutable
        && !launchRequested_ && !connectionRefused) {
        markUnavailable(message, QStringLiteral("BACKEND_CONNECTION_ERROR"),
                        kStartAction, configuredLogPath());
        return;
    }
    if (code == QStringLiteral("CONNECTION_ERROR")
        || code == QStringLiteral("CONNECTION_LOST")
        || code == QStringLiteral("TIMEOUT")) {
        if (!launchRequested_ && reusingExternalDevelopmentService_) {
            startupTimer_->start(config_.startupTimeoutMs);
            retryTimer_->start();
            QTimer::singleShot(0, this, &BackendProcessManager::tryConnect);
            return;
        }
        if (!launchRequested_) {
            launchBackend();
        }
        retryTimer_->start();
        return;
    }
    if (!details.isEmpty()) {
        markUnavailable(message, code,
                        details.value(QStringLiteral("action")).toString(),
                        details.value(QStringLiteral("log_path")).toString());
        return;
    }
    const QString action = code == QStringLiteral("HANDSHAKE_FAILED")
        ? kIdentityAction : kStartAction;
    markUnavailable(message, code, action, configuredLogPath());
}

void BackendProcessManager::onResponseReceived(const QString &command,
                                               const QJsonObject &response) {
    Q_UNUSED(response)
    if (command != QStringLiteral("shutdown")) {
        return;
    }
    if (restartPhase_ != RestartPhase::Idle) {
        return;
    }
    if (shuttingDown_ && canControlOwnedProcess()) {
        client_->disconnectFromService();
        if (launcher_->isRunning()) {
            stoppingOwnedProcess_ = true;
            launcher_->terminate();
        }
        owned_ = false;
        emit serviceOwnershipChanged(false);
    }
}

void BackendProcessManager::onStartupTimeout() {
    if (restartPhase_ != RestartPhase::Idle || shuttingDown_) {
        return;
    }
    retryTimer_->stop();
    if (config_.launchMode == BackendLaunchMode::PackagedExecutable
        && !launchRequested_ && prelaunchProbeConnected_) {
        markUnavailable(QStringLiteral("检测到其他后端实例占用当前端口"),
                        QStringLiteral("BACKEND_INSTANCE_CONFLICT"),
                        kConflictAction, configuredLogPath());
        return;
    }
    if (!launchRequested_) {
        launchBackend();
        startupTimer_->start(config_.startupTimeoutMs);
        return;
    }
    markUnavailable(QStringLiteral("后端启动超时"),
                    QStringLiteral("BACKEND_STARTUP_TIMEOUT"),
                    kTimeoutAction, configuredLogPath());
}

void BackendProcessManager::onStopEscalationTimeout() {
    if (!launcher_->isRunning() || !canControlOwnedProcess()) {
        return;
    }
    if (restartPhase_ == RestartPhase::GracefulStop) {
        restartPhase_ = RestartPhase::TerminateWait;
        launcher_->terminate();
        if (launcher_->isRunning()) {
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
    if (shuttingDown_ || terminalFailure_
        || restartPhase_ != RestartPhase::Idle) {
        return;
    }
    retryTimer_->stop();
    launchedProcess_ = false;
    markUnavailable(QStringLiteral("后端进程启动失败：%1").arg(message),
                    QStringLiteral("BACKEND_START_FAILED"),
                    kStartAction, configuredLogPath());
}

void BackendProcessManager::onProcessFinished(int exitCode) {
    launchedProcess_ = false;
    if (shuttingDown_) {
        Q_UNUSED(exitCode)
        stopEscalationTimer_->stop();
        restartPhase_ = RestartPhase::Idle;
        stoppingOwnedProcess_ = false;
        if (owned_) {
            owned_ = false;
            emit serviceOwnershipChanged(false);
        }
        return;
    }
    if (terminalFailure_) {
        return;
    }
    if (restartPhase_ != RestartPhase::Idle) {
        Q_UNUSED(exitCode)
        relaunchAfterRestartExit();
        return;
    }
    if (stoppingOwnedProcess_) {
        stoppingOwnedProcess_ = false;
        owned_ = false;
        emit serviceOwnershipChanged(false);
        return;
    }
    if (!shuttingDown_) {
        markUnavailable(QStringLiteral("后端进程已退出，代码 %1").arg(exitCode),
                        QStringLiteral("BACKEND_PROCESS_EXITED"),
                        kStartAction, configuredLogPath());
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
    readyInstanceToken_.clear();
    startupTimer_->start(config_.startupTimeoutMs);
    launchBackend();
}

void BackendProcessManager::launchBackend() {
    terminalFailure_ = false;
    launchRequested_ = true;
    launchedProcess_ = true;
    prelaunchProbeConnected_ = false;
    reusingExternalDevelopmentService_ = false;
    owned_ = false;
    readyInstanceToken_.clear();
    launchInstanceToken_ = QUuid::createUuid().toString(QUuid::WithoutBraces);
    backendProgress_ = 0;
    emit backendLoading(QStringLiteral("starting_process"),
                        QStringLiteral("正在启动后端"), 0);
    if (!launcher_->start(backendProgram(), backendArguments(), config_.projectRoot)) {
        launchedProcess_ = false;
        markUnavailable(QStringLiteral("无法启动后端进程"),
                        QStringLiteral("BACKEND_START_FAILED"),
                        kStartAction, configuredLogPath());
        return;
    }
    retryTimer_->start();
}

void BackendProcessManager::markUnavailable(const QString &reason, const QString &code,
                                            const QString &action,
                                            const QString &logPath) {
    retryTimer_->stop();
    startupTimer_->stop();
    terminalFailure_ = true;
    ++startupGeneration_;
    client_->disconnectFromService();
    emit backendUnavailable(reason, code, action, logPath);
}

QString BackendProcessManager::backendProgram() const {
    return config_.launchMode == BackendLaunchMode::PackagedExecutable
        ? config_.backendExecutable : config_.pythonExecutable;
}

QStringList BackendProcessManager::backendArguments() const {
    QStringList arguments;
    if (config_.launchMode == BackendLaunchMode::PythonScript) {
        arguments.append(config_.backendScript);
    }
    arguments.append({
        QStringLiteral("--host"), config_.host.toString(),
        QStringLiteral("--port"), QString::number(config_.port),
        QStringLiteral("--project-root"), config_.projectRoot,
        QStringLiteral("--model-dir"), config_.modelDir,
    });
    if (config_.launchMode == BackendLaunchMode::PackagedExecutable) {
        arguments.append({
            QStringLiteral("--paddle-config"), config_.paddleConfigPath,
            QStringLiteral("--data-root"), config_.dataRoot,
        });
    } else {
        arguments.append({QStringLiteral("--library-dir"), config_.libraryDir});
    }
    arguments.append({
        QStringLiteral("--compute-device"), config_.computeDevice,
        QStringLiteral("--model-sha256"), config_.modelSha256,
        QStringLiteral("--edition"), config_.edition,
        QStringLiteral("--package-version"), config_.packageVersion,
        QStringLiteral("--instance-token"), launchInstanceToken_,
        QStringLiteral("--parent-pid"), QString::number(QCoreApplication::applicationPid()),
        QStringLiteral("--local-search-mode"), config_.localSearchMode,
        QStringLiteral("--inference-mode"), config_.inferenceMode,
    });
    return arguments;
}

bool BackendProcessManager::identityMatches(const QJsonObject &metadata,
                                            QString *reason) const {
    const auto mismatch = [&](const QString &field, const QString &expected) {
        if (metadata.value(field).toString() == expected) {
            return false;
        }
        if (reason != nullptr) {
            *reason = QStringLiteral("后端身份不匹配：%1").arg(field);
        }
        return true;
    };
    if (mismatch(QStringLiteral("package_version"), config_.packageVersion)
        || mismatch(QStringLiteral("edition"), config_.edition)
        || mismatch(QStringLiteral("compute_device"), config_.computeDevice)
        || mismatch(QStringLiteral("model_fingerprint"), config_.modelSha256)) {
        return false;
    }
    if (!launchRequested_) {
        return true;
    }
    return !mismatch(QStringLiteral("instance_token"), launchInstanceToken_);
}

bool BackendProcessManager::canControlOwnedProcess() const {
    return owned_ && launchedProcess_ && !launchInstanceToken_.isEmpty()
        && readyInstanceToken_ == launchInstanceToken_;
}

QString BackendProcessManager::configuredLogPath() const {
    return QDir(config_.dataRoot).filePath(QStringLiteral("logs"));
}

int BackendProcessManager::stopEscalationIntervalMs() const {
    return qBound(100, config_.requestTimeoutMs, 2000);
}
