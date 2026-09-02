#pragma once

#include <QObject>
#include <QString>

class BackendProcessManager;
class QTimer;

class StartupSmokeController : public QObject {
    Q_OBJECT
public:
    explicit StartupSmokeController(BackendProcessManager *manager, int timeoutMs,
                                     QObject *parent = nullptr);

signals:
    void finished(int exitCode);

private slots:
    void onReady();
    void onUnavailable(const QString &reason, const QString &code,
                      const QString &action, const QString &logPath);
    void onTimeout();

private:
    void finish(int exitCode);
    BackendProcessManager *manager_;
    QTimer *timer_;
    bool finished_ = false;
};
