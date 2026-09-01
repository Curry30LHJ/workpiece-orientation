#include <QtTest/QtTest>

#include <QJsonDocument>
#include <QJsonObject>
#include <QDir>
#include <QFileInfo>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>

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
        if (!delayExit) {
            finish(0);
        }
    }

    void kill() override {
        ++killCalls;
        if (!delayExit) {
            finish(-1);
        }
    }

    void release() override { ++releaseCalls; }

    void finish(int exitCode) {
        running = false;
        emit finished(exitCode);
    }

    bool isRunning() const override { return running; }

    int startCalls = 0;
    int terminateCalls = 0;
    int killCalls = 0;
    int releaseCalls = 0;
    bool running = false;
    bool delayExit = false;
    QString lastProgram;
    QStringList lastArguments;
    QString lastWorkingDirectory;

signals:
    void startRequested();
};

class HandshakeServer : public QObject {
    Q_OBJECT

public:
    enum class HelloBehavior {
        Ready,
        ErrorResponse,
        BadJson,
        WrongProtocol,
        WrongService,
        NoResponse,
        CloseImmediately,
    };

    explicit HandshakeServer(QObject *parent = nullptr) : QObject(parent) {
        connect(&server, &QTcpServer::newConnection, this, &HandshakeServer::acceptConnection);
    }

    bool listen(quint16 port) { return server.listen(QHostAddress::LocalHost, port); }
    bool isListening() const { return server.isListening(); }
    quint16 port() const { return server.serverPort(); }
    void setLoadingResponses(int count) { loadingResponses = count; }
    QString shutdownToken() const { return lastShutdownToken; }
    void setRespondToShutdown(bool respond) { respondToShutdown = respond; }
    void setHelloBehavior(HelloBehavior behavior) { helloBehavior = behavior; }
    void disconnectClient() {
        if (socket != nullptr) socket->abort();
    }
    void setIdentity(const QString &version, const QString &edition,
                     const QString &device, const QString &modelSha256,
                     const QString &instanceToken) {
        packageVersion = version;
        packageEdition = edition;
        computeDevice = device;
        modelFingerprint = modelSha256;
        token = instanceToken;
    }
    void setNativeMetadata(const QString &serviceVersion,
                           const QString &modelSha256, int featureDim) {
        ppBackend = QStringLiteral("native_cpp");
        nativeServiceVersion = serviceVersion;
        nativeModelSha256 = modelSha256;
        nativeFeatureDim = featureDim;
    }

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
                if (helloBehavior == HelloBehavior::ErrorResponse) {
                    send({{"version", 1}, {"request_id", id}, {"ok", false},
                          {"error", QJsonObject{{"code", "STARTUP_REJECTED"},
                                                {"message", "rejected"}}}});
                    continue;
                }
                if (helloBehavior == HelloBehavior::BadJson) {
                    socket->write("not-json\n");
                    socket->flush();
                    continue;
                }
                if (helloBehavior == HelloBehavior::WrongProtocol) {
                    send({{"version", 2}, {"request_id", id}, {"ok", true},
                          {"service", "workpiece-orientation"}, {"ready", true}});
                    continue;
                }
                if (helloBehavior == HelloBehavior::WrongService) {
                    send({{"version", 1}, {"request_id", id}, {"ok", true},
                          {"service", "foreign-service"}, {"ready", true}});
                    continue;
                }
                if (helloBehavior == HelloBehavior::NoResponse) {
                    continue;
                }
                if (helloBehavior == HelloBehavior::CloseImmediately) {
                    socket->abort();
                    continue;
                }
                if (loadingResponses > 0) {
                    --loadingResponses;
                    QJsonObject response{{"version", 1}, {"request_id", id}, {"ok", true},
                                         {"service", "workpiece-orientation"},
                                         {"ready", false}, {"status", "loading"},
                                         {"phase", "loading_model"},
                                         {"message", "模型加载中"}, {"progress", 50},
                                         {"package_version", packageVersion},
                                         {"edition", packageEdition},
                                         {"compute_device", computeDevice},
                                         {"model_fingerprint", modelFingerprint},
                                         {"instance_token", token}};
                    appendNativeMetadata(&response);
                    send(response);
                } else {
                    QJsonObject response{{"version", 1}, {"request_id", id}, {"ok", true},
                                         {"service", "workpiece-orientation"},
                                         {"ready", true},
                                         {"package_version", packageVersion},
                                         {"edition", packageEdition},
                                         {"compute_device", computeDevice},
                                         {"model_fingerprint", modelFingerprint},
                                         {"instance_token", token}};
                    appendNativeMetadata(&response);
                    send(response);
                }
            } else if (object.value(QStringLiteral("command")).toString() == QStringLiteral("shutdown")) {
                lastShutdownToken = object.value(QStringLiteral("instance_token")).toString();
                if (respondToShutdown) {
                    send({{"version", 1}, {"request_id", id}, {"ok", true}});
                    socket->disconnectFromHost();
                }
            }
        }
    }

private:
    void appendNativeMetadata(QJsonObject *response) const {
        if (response == nullptr || ppBackend.isEmpty()) {
            return;
        }
        response->insert(QStringLiteral("pp_backend"), ppBackend);
        response->insert(QStringLiteral("native_service_version"), nativeServiceVersion);
        response->insert(QStringLiteral("native_model_sha256"), nativeModelSha256);
        response->insert(QStringLiteral("feature_dim"), nativeFeatureDim);
    }

    void send(const QJsonObject &object) {
        socket->write(QJsonDocument(object).toJson(QJsonDocument::Compact) + "\n");
        socket->flush();
    }

    QTcpServer server;
    QTcpSocket *socket = nullptr;
    QByteArray buffer;
    int loadingResponses = 0;
    QString packageVersion = QStringLiteral("dev");
    QString packageEdition = QStringLiteral("dev");
    QString computeDevice = QStringLiteral("gpu");
    QString modelFingerprint;
    QString token;
    QString ppBackend;
    QString nativeServiceVersion;
    QString nativeModelSha256;
    int nativeFeatureDim = -1;
    QString lastShutdownToken;
    bool respondToShutdown = true;
    HelloBehavior helloBehavior = HelloBehavior::Ready;
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
        config.dataRoot = QStringLiteral("data");
        return config;
    }

    static AppConfig packagedConfigFor(quint16 port, int startupTimeoutMs = 1000) {
        AppConfig config = configFor(port, startupTimeoutMs);
        config.launchMode = BackendLaunchMode::PackagedExecutable;
        config.backendExecutable = QStringLiteral("backend/orientation_backend.exe");
        config.paddleConfigPath = QStringLiteral("backend/resources/inference_general.yaml");
        config.dataRoot = QStringLiteral("data");
        config.modelSha256 = QString(64, QLatin1Char('a'));
        config.computeDevice = QStringLiteral("gpu");
        config.edition = QStringLiteral("gpu");
        config.packageVersion = QStringLiteral("1.0.0");
        config.inferenceMode = QStringLiteral("fast_geometry");
        return config;
    }

    static QString argumentValue(const QStringList &arguments, const QString &name) {
        const int index = arguments.indexOf(name);
        return index >= 0 && index + 1 < arguments.size() ? arguments.at(index + 1) : QString();
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
    void packagedModeLaunchesBackendExeWithoutScriptArgument() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = packagedConfigFor(port);
        BackendProcessManager manager(config, &client, &launcher);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(
                config.packageVersion, config.edition, config.computeDevice,
                config.modelSha256,
                argumentValue(launcher.lastArguments, QStringLiteral("--instance-token")));
            QVERIFY(server.listen(port));
        });
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(launcher.startCalls, 1, 1000);
        QCOMPARE(launcher.lastProgram, config.backendExecutable);
        QVERIFY(!launcher.lastArguments.contains(config.backendScript));
        QCOMPARE(argumentValue(launcher.lastArguments, QStringLiteral("--data-root")), config.dataRoot);
        QCOMPARE(argumentValue(launcher.lastArguments, QStringLiteral("--compute-device")), QStringLiteral("gpu"));
        QCOMPARE(argumentValue(launcher.lastArguments, QStringLiteral("--model-sha256")), config.modelSha256);
        QVERIFY(!argumentValue(launcher.lastArguments, QStringLiteral("--instance-token")).isEmpty());
        QCOMPARE(launcher.lastArguments, QStringList({
            QStringLiteral("--host"), config.host.toString(),
            QStringLiteral("--port"), QString::number(config.port),
            QStringLiteral("--project-root"), config.projectRoot,
            QStringLiteral("--model-dir"), config.modelDir,
            QStringLiteral("--paddle-config"), config.paddleConfigPath,
            QStringLiteral("--data-root"), config.dataRoot,
            QStringLiteral("--compute-device"), config.computeDevice,
            QStringLiteral("--pp-backend"), config.ppBackend,
            QStringLiteral("--model-sha256"), config.modelSha256,
            QStringLiteral("--edition"), config.edition,
            QStringLiteral("--package-version"), config.packageVersion,
            QStringLiteral("--instance-token"),
            argumentValue(launcher.lastArguments, QStringLiteral("--instance-token")),
            QStringLiteral("--parent-pid"),
            argumentValue(launcher.lastArguments, QStringLiteral("--parent-pid")),
            QStringLiteral("--local-search-mode"), config.localSearchMode,
            QStringLiteral("--inference-mode"), config.inferenceMode,
        }));
    }

    void nativeBackendArgumentsAndMetadataArePublished() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1500);
        config.ppBackend = QStringLiteral("native_cpp");
        config.nativePpExecutable = QStringLiteral("native/ppshitu_rec_service.exe");
        config.computeDevice = QStringLiteral("cpu");
        config.inferenceMode = QStringLiteral("fast_geometry");
        BackendProcessManager manager(config, &client, &launcher);
        QSignalSpy metadataSpy(&manager, &BackendProcessManager::backendMetadataUpdated);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            server.setNativeMetadata(QStringLiteral("ppshitu-native-cpp/1"),
                                     QString(64, QLatin1Char('b')), 512);
            QVERIFY(server.listen(port));
        });

        manager.start();

        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 2000);
        QCOMPARE(metadataSpy.count(), 1);
        const QJsonObject metadata = metadataSpy.at(0).at(0).toJsonObject();
        QCOMPARE(metadata.value(QStringLiteral("pp_backend")).toString(),
                 QStringLiteral("native_cpp"));
        QCOMPARE(metadata.value(QStringLiteral("native_service_version")).toString(),
                 QStringLiteral("ppshitu-native-cpp/1"));
        QCOMPARE(metadata.value(QStringLiteral("feature_dim")).toInt(), 512);
        QCOMPARE(argumentValue(launcher.lastArguments, QStringLiteral("--pp-backend")),
                 QStringLiteral("native_cpp"));
        QCOMPARE(argumentValue(launcher.lastArguments,
                               QStringLiteral("--native-pp-executable")),
                 config.nativePpExecutable);
    }

    void mismatchedExternalPackageIsRejectedWithoutTermination() {
        HandshakeServer server;
        server.setIdentity(QStringLiteral("0.9.0"), QStringLiteral("gpu"),
                           QStringLiteral("gpu"), QString(64, QLatin1Char('a')),
                           QStringLiteral("foreign-token"));
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(packagedConfigFor(server.port()), &client, &launcher);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(unavailableSpy.count(), 1, 1000);
        QCOMPARE(launcher.startCalls, 0);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
    }

    void compatibleForeignPackagedServiceIsInstanceConflict() {
        HandshakeServer server;
        const AppConfig config = packagedConfigFor(0);
        server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                           config.modelSha256, QStringLiteral("foreign-token"));
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig listeningConfig = config;
        listeningConfig.port = server.port();
        BackendProcessManager manager(listeningConfig, &client, &launcher);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);

        manager.start();

        QTRY_COMPARE_WITH_TIMEOUT(unavailableSpy.count(), 1, 1000);
        QCOMPARE(unavailableSpy.at(0).at(1).toString(),
                 QStringLiteral("BACKEND_INSTANCE_CONFLICT"));
        QCOMPARE(launcher.startCalls, 0);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
    }

    void packagedTcpContactFailuresAreInstanceConflicts_data() {
        QTest::addColumn<int>("behavior");
        QTest::newRow("hello-ok-false")
            << int(HandshakeServer::HelloBehavior::ErrorResponse);
        QTest::newRow("bad-json") << int(HandshakeServer::HelloBehavior::BadJson);
        QTest::newRow("wrong-protocol")
            << int(HandshakeServer::HelloBehavior::WrongProtocol);
        QTest::newRow("wrong-service")
            << int(HandshakeServer::HelloBehavior::WrongService);
        QTest::newRow("accepted-timeout")
            << int(HandshakeServer::HelloBehavior::NoResponse);
        QTest::newRow("immediate-close")
            << int(HandshakeServer::HelloBehavior::CloseImmediately);
    }

    void packagedTcpContactFailuresAreInstanceConflicts() {
        QFETCH(int, behavior);
        HandshakeServer server;
        server.setHelloBehavior(static_cast<HandshakeServer::HelloBehavior>(behavior));
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = packagedConfigFor(server.port());
        if (static_cast<HandshakeServer::HelloBehavior>(behavior)
            == HandshakeServer::HelloBehavior::NoResponse) {
            config.startupTimeoutMs = 50;
            config.requestTimeoutMs = 500;
        }
        BackendProcessManager manager(config, &client, &launcher);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);

        manager.start();

        QTRY_COMPARE_WITH_TIMEOUT(unavailableSpy.count(), 1, 1500);
        QCOMPARE(unavailableSpy.at(0).at(1).toString(),
                 QStringLiteral("BACKEND_INSTANCE_CONFLICT"));
        QCOMPARE(launcher.startCalls, 0);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
    }

    void staleGenerationSignalsAreIgnoredAfterRestart() {
        HandshakeServer server;
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(server.port(), 1000), &client, &launcher);
        QSignalSpy loadingSpy(&manager, &BackendProcessManager::backendLoading);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1000);
        manager.restart();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 2, 1000);
        const int loadingCountBeforeStaleSignals = loadingSpy.count();

        const QJsonObject staleMetadata{{"phase", "loading_model"},
                                        {"message", "旧加载"}, {"progress", 90}};
        emit client.handshakeLoading(1, staleMetadata);
        emit client.handshakeSucceeded(1, staleMetadata);
        emit client.transportFailed(1, QStringLiteral("TIMEOUT"),
                                    QStringLiteral("旧失败"));
        QTest::qWait(50);

        QCOMPARE(loadingSpy.count(), loadingCountBeforeStaleSignals);
        QCOMPARE(readySpy.count(), 2);
        QCOMPARE(unavailableSpy.count(), 0);
    }

    void developmentModeReusesCompatibleExternalToken() {
        HandshakeServer server;
        AppConfig config = configFor(0);
        server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                           config.modelSha256, QStringLiteral("external"));
        QVERIFY(server.listen(0));
        config.port = server.port();
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(config, &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);

        manager.start();

        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1000);
        QCOMPARE(launcher.startCalls, 0);
        QVERIFY(!manager.ownedByThisSession());
    }

    void serviceFailureDetailsRemainVerbatim() {
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(unusedPort(), 1000), &client, &launcher);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);
        manager.start();

        emit client.transportFailed(
            1, QStringLiteral("SERVICE_FAILURE"), QStringLiteral("服务原始消息"),
            QJsonObject{{"code", "SERVICE_FAILURE"}, {"action", "原始动作"},
                        {"log_path", "D:/service/raw.log"}});

        QCOMPARE(unavailableSpy.count(), 1);
        QCOMPARE(unavailableSpy.at(0).at(0).toString(), QStringLiteral("服务原始消息"));
        QCOMPARE(unavailableSpy.at(0).at(1).toString(), QStringLiteral("SERVICE_FAILURE"));
        QCOMPARE(unavailableSpy.at(0).at(2).toString(), QStringLiteral("原始动作"));
        QCOMPARE(unavailableSpy.at(0).at(3).toString(), QStringLiteral("D:/service/raw.log"));
    }

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
        config.inferenceMode = QStringLiteral("fast_geometry");
        BackendProcessManager manager(config, &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
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
        const int inferenceModeIndex = launcher.lastArguments.indexOf(
            QStringLiteral("--inference-mode")
        );
        QVERIFY(inferenceModeIndex >= 0);
        QCOMPARE(
            launcher.lastArguments.value(inferenceModeIndex + 1),
            QStringLiteral("fast_geometry")
        );
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        QVERIFY(manager.ownedByThisSession());
    }

    void retriesLoadingHandshakeWithoutStartingAnotherProcess() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        server.setLoadingResponses(1);
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1500);
        BackendProcessManager manager(config, &client, &launcher);
        QSignalSpy loadingSpy(&manager, &BackendProcessManager::backendLoading);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            QVERIFY(server.listen(port));
        });

        manager.start();

        QTRY_COMPARE_WITH_TIMEOUT(loadingSpy.count(), 1, 2500);
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 3500);
        QCOMPARE(launcher.startCalls, 1);
        QCOMPARE(unavailableSpy.count(), 0);
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
        QCOMPARE(unavailableSpy.at(0).at(1).toString(),
                 QStringLiteral("BACKEND_STARTUP_TIMEOUT"));
        QCOMPARE(unavailableSpy.at(0).at(2).toString(),
                 QStringLiteral("请查看后端日志并重试"));
        QCOMPARE(unavailableSpy.at(0).at(3).toString(),
                 QStringLiteral("data/logs"));
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
        QCOMPARE(launcher.killCalls, 0);
    }

    void ownedRestartWaitsForExitBeforeLaunchingReplacement() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1000);
        config.localSearchMode = QStringLiteral("exhaustive");
        BackendProcessManager manager(config, &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            if (!server.isListening() && !server.listen(port)) {
                QCOMPARE(server.port(), port);
            }
        });
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        QCOMPARE(launcher.startCalls, 1);
        QVERIFY(manager.ownedByThisSession());
        launcher.delayExit = true;

        manager.restart();
        QTest::qWait(50);

        QCOMPARE(launcher.startCalls, 1);
        QCOMPARE(launcher.terminateCalls, 0);
        launcher.finish(0);
        QTRY_COMPARE_WITH_TIMEOUT(launcher.startCalls, 2, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 2, 1500);
        const int modeIndex = launcher.lastArguments.indexOf(
            QStringLiteral("--local-search-mode")
        );
        QVERIFY(modeIndex >= 0);
        QCOMPARE(
            launcher.lastArguments.value(modeIndex + 1),
            QStringLiteral("exhaustive")
        );
        QVERIFY(manager.ownedByThisSession());
    }

    void ownedRestartEscalatesTerminateThenKillAndRelaunchesOnce() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1000);
        BackendProcessManager manager(config, &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            if (!server.isListening() && !server.listen(port)) {
                QCOMPARE(server.port(), port);
            }
        });
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        launcher.delayExit = true;

        manager.restart();
        manager.restart();

        QTRY_COMPARE_WITH_TIMEOUT(launcher.terminateCalls, 1, 1000);
        QCOMPARE(launcher.startCalls, 1);
        QTRY_COMPARE_WITH_TIMEOUT(launcher.killCalls, 1, 1000);
        QCOMPARE(launcher.startCalls, 1);
        launcher.finish(-1);
        QTRY_COMPARE_WITH_TIMEOUT(launcher.startCalls, 2, 1000);
        QTest::qWait(150);
        QCOMPARE(launcher.startCalls, 2);
    }

    void domainCommandFailureDoesNotMarkBackendUnavailable() {
        HandshakeServer server;
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1000);

        emit client.commandFailed(QStringLiteral("publish_geometry_mask_profile"),
                                  QStringLiteral("GEOMETRY_VALIDATION_FAILED"),
                                  QStringLiteral("validation rejected"));
        QTest::qWait(50);

        QCOMPARE(unavailableSpy.count(), 0);
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

    void developmentExternalDisconnectRetriesWithoutLaunching() {
        HandshakeServer server;
        QVERIFY(server.listen(0));
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1000);

        server.disconnectClient();

        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 2, 1500);
        QCOMPARE(launcher.startCalls, 0);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
    }

    void ownedShutdownIncludesMatchingInstanceToken() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1000);
        BackendProcessManager manager(config, &client, &launcher);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            QVERIFY(server.listen(port));
        });
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        const QString launchedToken = argumentValue(
            launcher.lastArguments, QStringLiteral("--instance-token"));

        manager.shutdownOwnedService();

        QTRY_COMPARE_WITH_TIMEOUT(server.shutdownToken(), launchedToken, 1000);
    }

    void shutdownAckWaitsForGracefulProcessExit() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1000);
        BackendProcessManager manager(config, &client, &launcher);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            QVERIFY(server.listen(port));
        });
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        launcher.delayExit = true;
        const QString launchedToken = argumentValue(
            launcher.lastArguments, QStringLiteral("--instance-token"));

        manager.shutdownOwnedService();

        QTRY_COMPARE_WITH_TIMEOUT(server.shutdownToken(), launchedToken, 1000);
        QTest::qWait(40);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
        launcher.finish(0);
        QTest::qWait(200);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
    }

    void shutdownWithoutResponseEscalatesTerminateThenKillOnce() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        server.setRespondToShutdown(false);
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1000);
        BackendProcessManager manager(config, &client, &launcher);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            QVERIFY(server.listen(port));
        });
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        launcher.delayExit = true;
        const QString launchedToken = argumentValue(
            launcher.lastArguments, QStringLiteral("--instance-token"));

        manager.shutdownOwnedService();

        QTRY_COMPARE_WITH_TIMEOUT(server.shutdownToken(), launchedToken, 1000);
        QTest::qWait(40);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
        QTRY_COMPARE_WITH_TIMEOUT(launcher.terminateCalls, 1, 500);
        QCOMPARE(launcher.killCalls, 0);
        QTRY_COMPARE_WITH_TIMEOUT(launcher.killCalls, 1, 500);
        QTest::qWait(150);
        QCOMPARE(launcher.terminateCalls, 1);
        QCOMPARE(launcher.killCalls, 1);
    }

    void ownedNotReadyShutdownEscalatesTerminateThenKill() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        AppConfig config = configFor(port, 1000);
        BackendProcessManager manager(config, &client, &launcher);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            QVERIFY(server.listen(port));
        });
        QSignalSpy readySpy(&manager, &BackendProcessManager::backendReady);
        QSignalSpy shutdownSpy(&manager, &BackendProcessManager::shutdownFinished);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        QVERIFY(manager.ownedByThisSession());
        launcher.delayExit = true;
        client.disconnectFromService();
        QCOMPARE(client.state(), BackendClient::State::Disconnected);

        manager.shutdownOwnedService();

        QCOMPARE(launcher.terminateCalls, 1);
        QCOMPARE(launcher.killCalls, 0);
        QCOMPARE(shutdownSpy.count(), 0);
        QTRY_COMPARE_WITH_TIMEOUT(launcher.killCalls, 1, 500);
        QCOMPARE(shutdownSpy.count(), 1);
        QTest::qWait(150);
        QCOMPARE(launcher.terminateCalls, 1);
        QCOMPARE(launcher.killCalls, 1);
        QCOMPARE(shutdownSpy.count(), 1);
    }

    void ownedDestructorUsesImmediateKillFallback() {
        const quint16 port = unusedPort();
        HandshakeServer server;
        BackendClient client;
        FakeProcessLauncher launcher;
        launcher.delayExit = true;
        AppConfig config = configFor(port, 1000);
        auto *manager = new BackendProcessManager(config, &client, &launcher);
        QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
            server.setIdentity(config.packageVersion, config.edition, config.computeDevice,
                               config.modelSha256,
                               argumentValue(launcher.lastArguments,
                                             QStringLiteral("--instance-token")));
            QVERIFY(server.listen(port));
        });
        QSignalSpy readySpy(manager, &BackendProcessManager::backendReady);
        manager->start();
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1500);
        QVERIFY(manager->ownedByThisSession());

        delete manager;

        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 1);
        QVERIFY(server.shutdownToken().isEmpty());
    }

    void unverifiedRunningProcessIsReleasedDuringManagerTeardown() {
        const quint16 port = unusedPort();
        BackendClient client;
        FakeProcessLauncher launcher;
        launcher.delayExit = true;
        auto *manager = new BackendProcessManager(configFor(port, 1000),
                                                  &client, &launcher);
        manager->start();
        QTRY_COMPARE_WITH_TIMEOUT(launcher.startCalls, 1, 1000);
        QVERIFY(launcher.isRunning());
        QVERIFY(!manager->ownedByThisSession());

        delete manager;

        QCOMPARE(launcher.releaseCalls, 1);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
        QVERIFY(launcher.isRunning());
    }

    void releasedQProcessSurvivesLauncherDestruction() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString markerPath = dir.filePath(QStringLiteral("released.txt"));
        const QString helperProgram = QStringLiteral(
            "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe");
        QString escapedMarkerPath = QDir::toNativeSeparators(markerPath);
        escapedMarkerPath.replace(QLatin1Char('\''), QStringLiteral("''"));
        const QString helperCommand = QStringLiteral(
            "Start-Sleep -Milliseconds 750; "
            "[System.IO.File]::WriteAllText('%1', 'released')")
            .arg(escapedMarkerPath);
        auto *launcher = new QProcessLauncher;
        QSignalSpy startedSpy(launcher, &ProcessLauncher::started);
        QVERIFY(launcher->start(helperProgram,
                                {QStringLiteral("-NoLogo"),
                                 QStringLiteral("-NoProfile"),
                                 QStringLiteral("-NonInteractive"),
                                 QStringLiteral("-Command"), helperCommand},
                                dir.path()));
        QTRY_COMPARE_WITH_TIMEOUT(startedSpy.count(), 1, 1000);
        QVERIFY(launcher->isRunning());

        launcher->release();
        delete launcher;

        QTRY_VERIFY_WITH_TIMEOUT(QFileInfo::exists(markerPath), 5000);
    }

    void stopDuringLaunchIgnoresLateProcessCallbacks() {
        const quint16 port = unusedPort();
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(port, 1000), &client, &launcher);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(launcher.startCalls, 1, 1000);

        manager.shutdownOwnedService();
        emit launcher.failed(QStringLiteral("late launch failure"));
        launcher.finish(23);
        QTest::qWait(50);

        QCOMPARE(unavailableSpy.count(), 0);
        QCOMPARE(launcher.terminateCalls, 0);
        QCOMPARE(launcher.killCalls, 0);
    }

    void terminalFailureIgnoresLaterProcessExit() {
        const quint16 port = unusedPort();
        BackendClient client;
        FakeProcessLauncher launcher;
        BackendProcessManager manager(configFor(port, 1000), &client, &launcher);
        QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);
        manager.start();
        QTRY_COMPARE_WITH_TIMEOUT(launcher.startCalls, 1, 1000);

        emit client.transportFailed(
            1, QStringLiteral("SERVICE_FAILURE"), QStringLiteral("terminal failure"),
            QJsonObject{{QStringLiteral("action"), QStringLiteral("stop")},
                        {QStringLiteral("log_path"), QStringLiteral("data/logs")}});
        QTRY_COMPARE_WITH_TIMEOUT(unavailableSpy.count(), 1, 1000);

        launcher.finish(37);
        QTest::qWait(50);

        QCOMPARE(unavailableSpy.count(), 1);
    }
};

QTEST_MAIN(TestBackendProcessManager)
#include "test_backendprocessmanager.moc"
