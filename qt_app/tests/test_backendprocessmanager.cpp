#include <QtTest/QtTest>

#include <QJsonDocument>
#include <QJsonObject>
#include <QTcpServer>
#include <QTcpSocket>

#include "../appconfig.h"
#include "../backendclient.h"
#include "../backendprocessmanager.h"
#include "../processlauncher.h"

class FakeProcessLauncher : public ProcessLauncher {
    Q_OBJECT

public:
    explicit FakeProcessLauncher(QObject *parent = nullptr) : ProcessLauncher(parent) {}

    bool start(const QString &program, const QStringList &arguments, const QString &workingDirectory) override {
        lastProgram = program;
        lastArguments = arguments;
        lastWorkingDirectory = workingDirectory;
        ++startCalls;
        running = true;
        emit startRequested();
        emit started();
        return true;
    }

    void terminate() override {
        ++terminateCalls;
        running = false;
        emit finished(0);
    }

    void kill() override {
        ++killCalls;
        running = false;
        emit finished(-1);
    }

    bool isRunning() const override { return running; }

    int startCalls = 0;
    int terminateCalls = 0;
    int killCalls = 0;
    bool running = false;
    QString lastProgram;
    QStringList lastArguments;
    QString lastWorkingDirectory;

signals:
    void startRequested();
};

class HandshakeServer : public QObject {
    Q_OBJECT

public:
    explicit HandshakeServer(QObject *parent = nullptr) : QObject(parent) {
        connect(&server, &QTcpServer::newConnection, this, &HandshakeServer::acceptConnection);
    }

    bool listen(quint16 port) { return server.listen(QHostAddress::LocalHost, port); }
    quint16 port() const { return server.serverPort(); }

private slots:
    void acceptConnection() {
        socket = server.nextPendingConnection();
        connect(socket, &QTcpSocket::readyRead, this, &HandshakeServer::readRequest);
    }

    void readRequest() {
        buffer += socket->readAll();
        while (buffer.contains('\n')) {
            const int newline = buffer.indexOf('\n');
            const QJsonDocument request = QJsonDocument::fromJson(buffer.left(newline));
            buffer.remove(0, newline + 1);
            if (!request.isObject()) {
                continue;
            }
            const QJsonObject object = request.object();
            const QString id = object.value(QStringLiteral("request_id")).toString();
            if (object.value(QStringLiteral("command")).toString() == QStringLiteral("hello")) {
                send({{"version", 1}, {"request_id", id}, {"ok", true},
                      {"service", "workpiece-orientation"}, {"ready", true}});
            } else if (object.value(QStringLiteral("command")).toString() == QStringLiteral("shutdown")) {
                send({{"version", 1}, {"request_id", id}, {"ok", true}});
                socket->disconnectFromHost();
            }
        }
    }

private:
    void send(const QJsonObject &object) {
        socket->write(QJsonDocument(object).toJson(QJsonDocument::Compact) + "\n");
        socket->flush();
    }

    QTcpServer server;
    QTcpSocket *socket = nullptr;
    QByteArray buffer;
};

class TestBackendProcessManager : public QObject {
    Q_OBJECT

private:
    static AppConfig configFor(quint16 port, int startupTimeoutMs = 300) {
        AppConfig config;
        config.pythonExecutable = QStringLiteral("python.exe");
        config.backendScript = QStringLiteral("service.py");
        config.projectRoot = QStringLiteral(".");
        config.modelDir = QStringLiteral("models");
        config.libraryDir = QStringLiteral("runtime_library");
        config.host = QHostAddress::LocalHost;
        config.port = port;
        config.startupTimeoutMs = startupTimeoutMs;
        config.requestTimeoutMs = 100;
        return config;
    }

    static quint16 unusedPort() {
        QTcpServer probe;
        if (!probe.listen(QHostAddress::LocalHost, 0)) {
            return 0;
        }
        const quint16 port = probe.serverPort();
        probe.close();
        return port;
    }

private slots:
    void reusesExistingServiceWithoutLaunchingProcess() {
        HandshakeServer server;
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);

        manager.start();

        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1000);
        QCOMPARE(launcher.startCalls, 0);
        QVERIFY(!manager.ownedByThisSession());
    }

    void launchesPythonOnlyAfterInitialConnectionFailure() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1000);
        config.localSearchMode = QStringLiteral("exhaustive");
        BackendProcessManager manager(config, &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            QVERIFY(server.listen(port));
        });

        manager.start();

        QTRY_COMPARE_WITH_TIMEOUT(launcher.startCalls, 1, 1000);
        QCOMPARE(launcher.lastArguments.first(), QStringLiteral("service.py"));
        const int modeIndex = launcher.lastArguments.indexOf(
            QStringLiteral("--local-search-mode")
        );
        QVERIFY(modeIndex >= 0);
        QCOMPARE(
            launcher.lastArguments.value(modeIndex + 1),
            QStringLiteral("exhaustive")
        );
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        QVERIFY(manager.ownedByThisSession());
    }

    void startupTimesOutAfterConfiguredDeadline() {
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(unusedPort(), 150), &client, &launcher);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);

        manager.start();

        QTRY_COMPARE_WITH_TIMEOUT(unavailableSpy.count(), 1, 1000);
        QCOMPARE(launcher.startCalls, 1);
        QVERIFY(unavailableSpy.at(0).at(0).toString().contains(QStringLiteral("超时")));
    }

    void restartReconnectsExternalServiceWithoutTerminatingIt() {
        HandshakeServer server;
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1000);

        manager.restart();

        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 2, 1000);
        QCOMPARE(launcher.startCalls, 0);
        QCOMPARE(launcher.terminateCalls, 0);
    }

    void shutdownDoesNotTerminateExternalService() {
        HandshakeServer server;
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1000);

        manager.shutdownOwnedService();

        QTest::qWait(100);
        QCOMPARE(launcher.terminateCalls, 0);
    }
};

QTEST_MAIN(TestBackendProcessManager)
#include "test_backendprocessmanager.moc"
