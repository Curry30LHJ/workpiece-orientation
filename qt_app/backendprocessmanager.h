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
    void backendLoading(const QString &phase, const QString &message, int progress);
    void backendUnavailable(const QString &reason, const QString &code = QString(),
                            const QString &action = QString(),
                            const QString &logPath = QString());
    void serviceOwnershipChanged(bool owned);

private slots:
    void tryConnect();
    void onHandshakeLoading(quint64 generation, const QJsonObject &metadata);
    void onHandshakeSucceeded(quint64 generation, const QJsonObject &metadata);
    void onTransportFailed(quint64 generation, const QString &code,
                           const QString &message, const QJsonObject &details);
    void onResponseReceived(const QString &command, const QJsonObject &response);
    void onStartupTimeout();
    void onStopEscalationTimeout();
    void onProcessFailed(const QString &message);
    void onProcessFinished(int exitCode);

private:
    enum class RestartPhase { Idle, GracefulStop, TerminateWait, KillWait };

    void launchBackend();
    void relaunchAfterRestartExit();
    void markUnavailable(const QString &reason, const QString &code,
                         const QString &action, const QString &logPath = QString());
    QString backendProgram() const;
    QStringList backendArguments() const;
    bool identityMatches(const QJsonObject &metadata, QString *reason) const;
    bool canControlOwnedProcess() const;
    QString configuredLogPath() const;
    int stopEscalationIntervalMs() const;

    AppConfig config_;
    BackendClient *client_;
    ProcessLauncher *launcher_;
    bool ownsLauncher_ = false;
    bool owned_ = false;
    bool launchedProcess_ = false;
    bool launchRequested_ = false;
    bool reusingExternalDevelopmentService_ = false;
    bool shuttingDown_ = false;
    bool stoppingOwnedProcess_ = false;
    RestartPhase restartPhase_ = RestartPhase::Idle;
    quint64 startupGeneration_ = 0;
    QString launchInstanceToken_;
    QString readyInstanceToken_;
    int backendProgress_ = 0;
    QTimer *retryTimer_;
    QTimer *startupTimer_;
    QTimer *stopEscalationTimer_;
};
