#include <QtTest/QtTest>

#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTcpSocket>

#include "../backendclient.h"

class FakeTcpServer : public QObject {
    Q_OBJECT

public:
    explicit FakeTcpServer(QObject *parent = nullptr) : QObject(parent) {
        connect(&server, &QTcpServer::newConnection, this, &FakeTcpServer::acceptConnection);
    }

    bool start() { return server.listen(QHostAddress::LocalHost, 0); }
    quint16 port() const { return server.serverPort(); }
    const QList<QJsonObject> &requests() const { return received; }
    int connectionCount() const { return connectionCount_; }

    void sendJson(const QJsonObject &object) {
        QVERIFY(socket != nullptr);
        socket->write(QJsonDocument(object).toJson(QJsonDocument::Compact) + "\n");
        socket->flush();
    }

    void sendFragments(const QList<QByteArray> &fragments) {
        QVERIFY(socket != nullptr);
        for (const QByteArray &fragment : fragments) {
            socket->write(fragment);
            socket->flush();
            QTest::qWait(5);
        }
    }

    void closeClient() {
        if (socket != nullptr) {
            socket->disconnectFromHost();
        }
    }

signals:
    void requestReceived(const QJsonObject &request);

private slots:
    void acceptConnection() {
        ++connectionCount_;
        socket = server.nextPendingConnection();
        connect(socket, &QTcpSocket::readyRead, this, &FakeTcpServer::readRequests);
    }

    void readRequests() {
        buffer += socket->readAll();
        while (buffer.contains('\n')) {
            const int newline = buffer.indexOf('\n');
            const QByteArray line = buffer.left(newline);
            buffer.remove(0, newline + 1);
            const QJsonDocument document = QJsonDocument::fromJson(line);
            if (document.isObject()) {
                received.append(document.object());
                emit requestReceived(document.object());
            }
        }
    }

private:
    QTcpServer server;
    QTcpSocket *socket = nullptr;
    QByteArray buffer;
    QList<QJsonObject> received;
    int connectionCount_ = 0;
};

class TestBackendClient : public QObject {
    Q_OBJECT

private:
    static QJsonObject helloResponse(const QString &requestId, const QString &service = QStringLiteral("workpiece-orientation")) {
        return {{"version", 1}, {"request_id", requestId}, {"ok", true}, {"service", service}, {"ready", true}};
    }

    static QJsonObject loadingHelloResponse(const QString &requestId) {
        return {{"version", 1}, {"request_id", requestId}, {"ok", true},
                {"service", "workpiece-orientation"}, {"ready", false},
                {"status", "loading"}, {"phase", "loading_model"},
                {"message", "模型加载中"}, {"progress", 50}};
    }

    static void connectWithHello(FakeTcpServer &server, BackendClient &client) {
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server, [&](const QJsonObject &request) {
            if (request.value(QStringLiteral("command")).toString() == QStringLiteral("hello")) {
                server.sendJson(helloResponse(request.value(QStringLiteral("request_id")).toString()));
            }
        });
        client.connectToService(QHostAddress::LocalHost, server.port(), 1000);
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() == 1, 1000);
    }

private slots:
    void loadingHandshakeStaysConnectedUntilReady() {
        FakeTcpServer server;
        QVERIFY(server.start());
        int helloCount = 0;
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server,
                         [&](const QJsonObject &request) {
            if (request.value(QStringLiteral("command")).toString() != QStringLiteral("hello")) {
                return;
            }
            ++helloCount;
            const QString id = request.value(QStringLiteral("request_id")).toString();
            QJsonObject response{{"version", 1}, {"request_id", id}, {"ok", true},
                                 {"service", "workpiece-orientation"},
                                 {"package_version", "1.0.0"}, {"edition", "gpu"},
                                 {"compute_device", "gpu"},
                                 {"model_fingerprint", QString(64, QLatin1Char('a'))},
                                 {"instance_token", "launch-123"}};
            response.insert(QStringLiteral("ready"), helloCount > 2);
            response.insert(QStringLiteral("status"), helloCount > 2 ? "ready" : "loading");
            response.insert(QStringLiteral("phase"), helloCount > 1 ? "loading_model" : "loading_runtime");
            response.insert(QStringLiteral("message"), "正在加载");
            response.insert(QStringLiteral("progress"), helloCount > 2 ? 100 : helloCount * 30);
            server.sendJson(response);
        });
        BackendClient client;
        QSignalSpy loadingSpy(&client, &BackendClient::handshakeLoading);
        QSignalSpy readySpy(&client, &BackendClient::handshakeSucceeded);
        QSignalSpy transportSpy(&client, &BackendClient::transportFailed);
        client.connectToService(QHostAddress::LocalHost, server.port(), 1000, 7);
        QTRY_COMPARE_WITH_TIMEOUT(loadingSpy.count(), 2, 2500);
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 2500);
        QCOMPARE(server.connectionCount(), 1);
        QCOMPARE(transportSpy.count(), 0);
        QCOMPARE(readySpy.at(0).at(0).toULongLong(), quint64(7));
        QCOMPARE(readySpy.at(0).at(1).toJsonObject()
                     .value(QStringLiteral("package_version")).toString(),
                 QStringLiteral("1.0.0"));
    }

    void loadingReceiverCanDisconnectWithoutRetryingHello() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server,
                         [&](const QJsonObject &request) {
            server.sendJson(loadingHelloResponse(
                request.value(QStringLiteral("request_id")).toString()));
        });
        QObject::connect(&client, &BackendClient::handshakeLoading, &client,
                         [&](quint64, const QJsonObject &) {
            client.disconnectFromService();
        });

        client.connectToService(QHostAddress::LocalHost, server.port(), 1000, 1);

        QTRY_COMPARE_WITH_TIMEOUT(server.requests().size(), 1, 1000);
        QTest::qWait(600);
        QCOMPARE(server.requests().size(), 1);
        QCOMPARE(client.state(), BackendClient::State::Disconnected);
    }

    void loadingReceiverCanReconnectWithoutStaleHelloOrRequestId() {
        FakeTcpServer loadingServer;
        FakeTcpServer replacementServer;
        QVERIFY(loadingServer.start());
        QVERIFY(replacementServer.start());
        BackendClient client;
        QSignalSpy readySpy(&client, &BackendClient::handshakeSucceeded);
        QSignalSpy transportSpy(&client, &BackendClient::transportFailed);
        QObject::connect(&loadingServer, &FakeTcpServer::requestReceived,
                         &loadingServer, [&](const QJsonObject &request) {
            loadingServer.sendJson(loadingHelloResponse(
                request.value(QStringLiteral("request_id")).toString()));
        });
        QObject::connect(&client, &BackendClient::handshakeLoading, &client,
                         [&](quint64 generation, const QJsonObject &) {
            if (generation == 1) {
                client.connectToService(QHostAddress::LocalHost,
                                        replacementServer.port(), 1000, 2);
            }
        });

        client.connectToService(QHostAddress::LocalHost, loadingServer.port(), 1000, 1);

        QTRY_COMPARE_WITH_TIMEOUT(replacementServer.requests().size(), 1, 1000);
        const QString replacementRequestId = replacementServer.requests().constFirst()
            .value(QStringLiteral("request_id")).toString();
        QVERIFY(!replacementRequestId.isEmpty());
        QTest::qWait(600);
        QCOMPARE(replacementServer.requests().size(), 1);
        replacementServer.sendJson(helloResponse(replacementRequestId));
        QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 1000);
        QCOMPARE(readySpy.at(0).at(0).toULongLong(), quint64(2));
        QCOMPARE(transportSpy.count(), 0);
    }

    void buffersPartialAndMultipleResponses() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy handshakeSpy(&client, &BackendClient::handshakeSucceeded);
        connectWithHello(server, client);
        QTRY_COMPARE_WITH_TIMEOUT(handshakeSpy.count(), 1, 1000);
        QSignalSpy responseSpy(&client, &BackendClient::responseReceived);

        const QString requestId = client.sendRequest(QStringLiteral("list_workpieces"));
        QVERIFY(!requestId.isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() == 2, 1000);
        const QJsonObject response{{"version", 1}, {"request_id", requestId}, {"ok", true}, {"workpieces", QJsonArray()}};
        const QByteArray encoded = QJsonDocument(response).toJson(QJsonDocument::Compact) + "\n";
        server.sendFragments({encoded.left(7), encoded.mid(7)});

        QTRY_COMPARE_WITH_TIMEOUT(responseSpy.count(), 1, 1000);
        QCOMPARE(responseSpy.at(0).at(0).toString(), QStringLiteral("list_workpieces"));
    }

    void rejectsWrongServiceIdentity() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy failureSpy(&client, &BackendClient::requestFailed);
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server, [&](const QJsonObject &request) {
            server.sendJson(helloResponse(request.value(QStringLiteral("request_id")).toString(), QStringLiteral("other")));
        });

        client.connectToService(QHostAddress::LocalHost, server.port(), 1000);

        QTRY_VERIFY_WITH_TIMEOUT(failureSpy.count() == 1, 1000);
        QCOMPARE(failureSpy.at(0).at(0).toString(), QStringLiteral("HANDSHAKE_FAILED"));
    }

    void rejectsMismatchedRequestId() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy handshakeSpy(&client, &BackendClient::handshakeSucceeded);
        connectWithHello(server, client);
        QSignalSpy failureSpy(&client, &BackendClient::requestFailed);
        QTRY_COMPARE_WITH_TIMEOUT(handshakeSpy.count(), 1, 1000);

        const QString requestId = client.sendRequest(QStringLiteral("list_workpieces"));
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() == 2, 1000);
        server.sendJson({{"version", 1}, {"request_id", requestId + QStringLiteral("-wrong")}, {"ok", true}});

        QTRY_VERIFY_WITH_TIMEOUT(failureSpy.count() == 1, 1000);
        QCOMPARE(failureSpy.at(0).at(0).toString(), QStringLiteral("PROTOCOL_ERROR"));
    }

    void emitsServerBusy() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy failureSpy(&client, &BackendClient::requestFailed);
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server, [&](const QJsonObject &request) {
            server.sendJson({
                {"version", 1},
                {"request_id", request.value(QStringLiteral("request_id")).toString()},
                {"ok", false},
                {"error", QJsonObject{{"code", "SERVER_BUSY"}, {"message", "busy"}}},
            });
        });

        client.connectToService(QHostAddress::LocalHost, server.port(), 1000);

        QTRY_VERIFY_WITH_TIMEOUT(failureSpy.count() == 1, 1000);
        QCOMPARE(failureSpy.at(0).at(0).toString(), QStringLiteral("SERVER_BUSY"));
    }

    void reportsModelLoadingWithoutTransportFailure() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy loadingSpy(&client, &BackendClient::handshakeLoading);
        QSignalSpy failureSpy(&client, &BackendClient::transportFailed);
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server, [&](const QJsonObject &request) {
            if (request.value(QStringLiteral("command")).toString() == QStringLiteral("hello")) {
                server.sendJson(loadingHelloResponse(request.value(QStringLiteral("request_id")).toString()));
            }
        });

        client.connectToService(QHostAddress::LocalHost, server.port(), 1000);

        QTRY_COMPARE_WITH_TIMEOUT(loadingSpy.count(), 1, 1000);
        QCOMPARE(loadingSpy.at(0).at(1).toJsonObject()
                     .value(QStringLiteral("message")).toString(),
                 QStringLiteral("模型加载中"));
        QCOMPARE(failureSpy.count(), 0);
    }

    void rejectsLoadingProgressOutsideProtocolRange() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy transportSpy(&client, &BackendClient::transportFailed);
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server,
                         [&](const QJsonObject &request) {
            QJsonObject response = loadingHelloResponse(
                request.value(QStringLiteral("request_id")).toString());
            response.insert(QStringLiteral("progress"), 101);
            server.sendJson(response);
        });

        client.connectToService(QHostAddress::LocalHost, server.port(), 1000, 9);

        QTRY_COMPARE_WITH_TIMEOUT(transportSpy.count(), 1, 1000);
        QCOMPARE(transportSpy.at(0).at(0).toULongLong(), quint64(9));
        QCOMPARE(transportSpy.at(0).at(1).toString(), QStringLiteral("PROTOCOL_ERROR"));
    }

    void handshakeFailurePreservesStructuredDetails() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy transportSpy(&client, &BackendClient::transportFailed);
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server,
                         [&](const QJsonObject &request) {
            server.sendJson({
                {"version", 1},
                {"request_id", request.value(QStringLiteral("request_id")).toString()},
                {"ok", false},
                {"error", QJsonObject{{"code", "SERVICE_FAILURE"},
                                      {"message", "服务原始消息"},
                                      {"action", "原始动作"},
                                      {"log_path", "D:/service/raw.log"}}},
            });
        });

        client.connectToService(QHostAddress::LocalHost, server.port(), 1000, 11);

        QTRY_COMPARE_WITH_TIMEOUT(transportSpy.count(), 1, 1000);
        QCOMPARE(transportSpy.at(0).at(0).toULongLong(), quint64(11));
        QCOMPARE(transportSpy.at(0).at(1).toString(), QStringLiteral("SERVICE_FAILURE"));
        const QJsonObject details = transportSpy.at(0).at(3).toJsonObject();
        QCOMPARE(details.value(QStringLiteral("action")).toString(), QStringLiteral("原始动作"));
        QCOMPARE(details.value(QStringLiteral("log_path")).toString(),
                 QStringLiteral("D:/service/raw.log"));
    }

    void timesOutOneOutstandingRequest() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy handshakeSpy(&client, &BackendClient::handshakeSucceeded);
        connectWithHello(server, client);
        QTRY_COMPARE_WITH_TIMEOUT(handshakeSpy.count(), 1, 1000);
        QSignalSpy failureSpy(&client, &BackendClient::requestFailed);

        QVERIFY(!client.sendRequest(QStringLiteral("list_workpieces")).isEmpty());

        QTRY_VERIFY_WITH_TIMEOUT(failureSpy.count() == 1, 1500);
        QCOMPARE(failureSpy.at(0).at(0).toString(), QStringLiteral("TIMEOUT"));
    }

    void progressEventKeepsRequestBusyAndCompletesNormally() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy handshakeSpy(&client, &BackendClient::handshakeSucceeded);
        connectWithHello(server, client);
        QTRY_COMPARE_WITH_TIMEOUT(handshakeSpy.count(), 1, 1000);
        QSignalSpy progressSpy(&client, &BackendClient::progressReceived);
        QSignalSpy responseSpy(&client, &BackendClient::responseReceived);

        const QString requestId = client.sendRequest(QStringLiteral("register"));
        QVERIFY(!requestId.isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() == 2, 1000);
        server.sendJson({
            {"version", 1}, {"request_id", requestId}, {"event", "progress"},
            {"command", "register"},
            {"progress", QJsonObject{{"phase", "features"}, {"completed", 1}, {"total", 2}}},
        });

        QTRY_COMPARE_WITH_TIMEOUT(progressSpy.count(), 1, 1000);
        QCOMPARE(client.state(), BackendClient::State::Busy);
        QCOMPARE(progressSpy.at(0).at(0).toString(), QStringLiteral("register"));
        server.sendJson({
            {"version", 1}, {"request_id", requestId}, {"ok", true},
            {"template_counts", QJsonObject{{"front", 1}, {"back", 1}}},
        });

        QTRY_COMPARE_WITH_TIMEOUT(responseSpy.count(), 1, 1000);
        QCOMPARE(client.state(), BackendClient::State::Ready);
    }

    void progressEventRefreshesInactivityTimeout() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy handshakeSpy(&client, &BackendClient::handshakeSucceeded);
        QObject::connect(&server, &FakeTcpServer::requestReceived, &server, [&](const QJsonObject &request) {
            if (request.value(QStringLiteral("command")).toString() == QStringLiteral("hello")) {
                server.sendJson(helloResponse(request.value(QStringLiteral("request_id")).toString()));
            }
        });
        client.connectToService(QHostAddress::LocalHost, server.port(), 200);
        QTRY_COMPARE_WITH_TIMEOUT(handshakeSpy.count(), 1, 1000);
        QSignalSpy failureSpy(&client, &BackendClient::requestFailed);
        const QString requestId = client.sendRequest(QStringLiteral("register"));
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() == 2, 1000);
        QTest::qWait(120);
        server.sendJson({
            {"version", 1}, {"request_id", requestId}, {"event", "progress"},
            {"command", "register"},
            {"progress", QJsonObject{{"phase", "features"}, {"completed", 1}, {"total", 2}}},
        });
        QTRY_VERIFY_WITH_TIMEOUT(failureSpy.count() == 0, 150);
        QTest::qWait(100);
        QVERIFY(failureSpy.isEmpty());
        server.sendJson({{"version", 1}, {"request_id", requestId}, {"ok", true}});
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
    }

    void disconnectClearsPendingRequest() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy handshakeSpy(&client, &BackendClient::handshakeSucceeded);
        connectWithHello(server, client);
        QTRY_COMPARE_WITH_TIMEOUT(handshakeSpy.count(), 1, 1000);
        QSignalSpy lostSpy(&client, &BackendClient::connectionLost);

        QVERIFY(!client.sendRequest(QStringLiteral("list_workpieces")).isEmpty());
        server.closeClient();

        QTRY_COMPARE_WITH_TIMEOUT(lostSpy.count(), 1, 1000);
        QCOMPARE(client.state(), BackendClient::State::Disconnected);
    }

    void domainErrorKeepsReadyAndUsesCommandFailureChannel() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy handshakeSpy(&client, &BackendClient::handshakeSucceeded);
        connectWithHello(server, client);
        QTRY_COMPARE_WITH_TIMEOUT(handshakeSpy.count(), 1, 1000);
        QSignalSpy commandSpy(&client, &BackendClient::commandFailed);
        QSignalSpy transportSpy(&client, &BackendClient::transportFailed);

        const QString requestId = client.sendRequest(QStringLiteral("publish_geometry_mask_profile"));
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() == 2, 1000);
        server.sendJson({
            {"version", 1}, {"request_id", requestId}, {"ok", false},
            {"error", QJsonObject{{"code", "GEOMETRY_VALIDATION_FAILED"},
                                  {"message", "validation rejected"}}},
        });

        QTRY_COMPARE_WITH_TIMEOUT(commandSpy.count(), 1, 1000);
        QCOMPARE(commandSpy.at(0).at(0).toString(), QStringLiteral("publish_geometry_mask_profile"));
        QCOMPARE(commandSpy.at(0).at(1).toString(), QStringLiteral("GEOMETRY_VALIDATION_FAILED"));
        QCOMPARE(transportSpy.count(), 0);
        QCOMPARE(client.state(), BackendClient::State::Ready);
        QVERIFY(!client.sendRequest(QStringLiteral("list_workpieces")).isEmpty());
    }

    void localRejectionIncludesAttemptedCommand() {
        BackendClient client;
        QSignalSpy commandSpy(&client, &BackendClient::commandFailed);

        QVERIFY(client.sendRequest(QStringLiteral("predict")).isEmpty());

        QCOMPARE(commandSpy.count(), 1);
        QCOMPARE(commandSpy.at(0).at(0).toString(), QStringLiteral("predict"));
        QCOMPARE(commandSpy.at(0).at(1).toString(), QStringLiteral("NOT_READY"));
    }

    void malformedResponseUsesTransportFailureChannel() {
        FakeTcpServer server;
        QVERIFY(server.start());
        BackendClient client;
        QSignalSpy handshakeSpy(&client, &BackendClient::handshakeSucceeded);
        connectWithHello(server, client);
        QTRY_COMPARE_WITH_TIMEOUT(handshakeSpy.count(), 1, 1000);
        QSignalSpy commandSpy(&client, &BackendClient::commandFailed);
        QSignalSpy transportSpy(&client, &BackendClient::transportFailed);

        QVERIFY(!client.sendRequest(QStringLiteral("predict")).isEmpty());
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() == 2, 1000);
        server.sendFragments({QByteArray("{bad-json}\n")});

        QTRY_COMPARE_WITH_TIMEOUT(transportSpy.count(), 1, 1000);
        QCOMPARE(transportSpy.at(0).at(1).toString(), QStringLiteral("PROTOCOL_ERROR"));
        QCOMPARE(commandSpy.count(), 0);
        QVERIFY(client.state() != BackendClient::State::Ready);
    }
};

QTEST_MAIN(TestBackendClient)
#include "test_backendclient.moc"
