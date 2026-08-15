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
    void backendUnavailable(const QString &reason);
    void serviceOwnershipChanged(bool owned);

private slots:
    void tryConnect();
    void onHandshakeSucceeded();
    void onRequestFailed(const QString &code, const QString &message);
    void onConnectionLost(const QString &reason);
    void onResponseReceived(const QString &command, const QJsonObject &response);
    void onStartupTimeout();
    void onProcessFailed(const QString &message);
    void onProcessFinished(int exitCode);

private:
    void launchBackend();
    void markUnavailable(const QString &reason);
    QStringList backendArguments() const;

    AppConfig config_;
    BackendClient *client_;
    ProcessLauncher *launcher_;
    bool ownsLauncher_ = false;
    bool owned_ = false;
    bool launchRequested_ = false;
    bool shuttingDown_ = false;
    bool stoppingOwnedProcess_ = false;
    QTimer *retryTimer_;
    QTimer *startupTimer_;
};
