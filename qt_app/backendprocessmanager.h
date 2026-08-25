#pragma once

#include <QObject>
#include <QJsonObject>
#include <QStringList>

#include "appconfig.h"

class BackendClient;
class ProcessLauncher;
class QTimer;

class BackendProcessManager : public QObject {
    Q_OBJECT

public:
    BackendProcessManager(const AppConfig &config, BackendClient *client,
                          ProcessLauncher *launcher = nullptr, QObject *parent = nullptr);
    ~BackendProcessManager() override;

    void start();
    void restart();
    void shutdownOwnedService();
    bool ownedByThisSession() const;

signals:
    void backendReady();
    void backendLoading(const QString &message);
    void backendUnavailable(const QString &reason);
    void serviceOwnershipChanged(bool owned);

private slots:
    void tryConnect();
    void onHandshakeSucceeded();
    void onTransportFailed(const QString &code, const QString &message);
    void onResponseReceived(const QString &command, const QJsonObject &response);
    void onStartupTimeout();
    void onStopEscalationTimeout();
    void onProcessFailed(const QString &message);
    void onProcessFinished(int exitCode);

private:
    enum class RestartPhase { Idle, GracefulStop, TerminateWait, KillWait };

    void launchBackend();
    void relaunchAfterRestartExit();
    void markUnavailable(const QString &reason);
    QStringList backendArguments() const;
    int stopEscalationIntervalMs() const;

    AppConfig config_;
    BackendClient *client_;
    ProcessLauncher *launcher_;
    bool ownsLauncher_ = false;
    bool owned_ = false;
    bool launchRequested_ = false;
    bool shuttingDown_ = false;
    bool stoppingOwnedProcess_ = false;
    RestartPhase restartPhase_ = RestartPhase::Idle;
    QTimer *retryTimer_;
    QTimer *startupTimer_;
    QTimer *stopEscalationTimer_;
};
