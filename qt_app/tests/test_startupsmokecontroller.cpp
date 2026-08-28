#include <QtTest/QtTest>

#include "../startupsmokecontroller.h"
#include "../backendprocessmanager.h"
#include "../backendclient.h"
#include "../processlauncher.h"

class NoopProcessLauncher : public ProcessLauncher {
    Q_OBJECT
public:
    using ProcessLauncher::ProcessLauncher;
    bool start(const QString &, const QStringList &, const QString &) override { return true; }
    void terminate() override {}
    void kill() override {}
    bool isRunning() const override { return false; }
};

static AppConfig smokeConfig() {
    AppConfig config;
    config.pythonExecutable = QStringLiteral("python.exe");
    config.backendScript = QStringLiteral("service.py");
    config.projectRoot = QStringLiteral(".");
    config.modelDir = QStringLiteral("models");
    config.libraryDir = QStringLiteral("runtime_library");
    config.host = QHostAddress::LocalHost;
    config.port = 37651;
    config.startupTimeoutMs = 1000;
    return config;
}

class TestStartupSmokeController : public QObject {
    Q_OBJECT
private slots:
    void readyFinishesWithZeroOnce() {
        BackendClient client; NoopProcessLauncher launcher;
        BackendProcessManager manager(smokeConfig(), &client, &launcher);
        StartupSmokeController controller(&manager, 1000);
        QSignalSpy finishedSpy(&controller, &StartupSmokeController::finished);
        emit manager.backendReady(); emit manager.backendReady();
        QCOMPARE(finishedSpy.count(), 1);
        QCOMPARE(finishedSpy.at(0).at(0).toInt(), 0);
    }
    void failureAndTimeoutUseDistinctExitCodes() {
        BackendClient client; NoopProcessLauncher launcher;
        BackendProcessManager manager(smokeConfig(), &client, &launcher);
        StartupSmokeController failed(&manager, 1000);
        QSignalSpy failedSpy(&failed, &StartupSmokeController::finished);
        emit manager.backendUnavailable(QStringLiteral("模型缺失"));
        QCOMPARE(failedSpy.at(0).at(0).toInt(), 2);
        StartupSmokeController timedOut(&manager, 1);
        QSignalSpy timeoutSpy(&timedOut, &StartupSmokeController::finished);
        QTRY_COMPARE_WITH_TIMEOUT(timeoutSpy.count(), 1, 100);
        QCOMPARE(timeoutSpy.at(0).at(0).toInt(), 3);
    }
};

QTEST_MAIN(TestStartupSmokeController)
#include "test_startupsmokecontroller.moc"
