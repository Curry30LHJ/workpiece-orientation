#include <QtTest/QtTest>

#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QTemporaryDir>
#include <QTcpServer>
#include <QTcpSocket>
#include <QImage>

#include "../backendclient.h"
#include "../backendprocessmanager.h"
#include "../mainwindow.h"
#include "../processlauncher.h"

class PassiveLauncher : public ProcessLauncher {
public:
    explicit PassiveLauncher(QObject *parent = nullptr) : ProcessLauncher(parent) {}
    bool start(const QString &, const QStringList &, const QString &) override { return true; }
    void terminate() override {}
    void kill() override {}
    bool isRunning() const override { return false; }
};

class RegistrationServer : public QObject {
    Q_OBJECT

public:
    explicit RegistrationServer(QObject *parent = nullptr) : QObject(parent) {
        connect(&server_, &QTcpServer::newConnection, this, &RegistrationServer::acceptConnection);
    }

    bool listen() { return server_.listen(QHostAddress::LocalHost, 0); }
    quint16 port() const { return server_.serverPort(); }
    QList<QJsonObject> requests() const { return requests_; }

private slots:
    void acceptConnection() {
        socket_ = server_.nextPendingConnection();
        connect(socket_, &QTcpSocket::readyRead, this, &RegistrationServer::readRequests);
    }

    void readRequests() {
        buffer_ += socket_->readAll();
        while (buffer_.contains('\n')) {
            const int newline = buffer_.indexOf('\n');
            const QJsonDocument document = QJsonDocument::fromJson(buffer_.left(newline));
            buffer_.remove(0, newline + 1);
            if (!document.isObject()) {
                continue;
            }
            const QJsonObject request = document.object();
            requests_.append(request);
            const QString command = request.value(QStringLiteral("command")).toString();
            const QString requestId = request.value(QStringLiteral("request_id")).toString();
            if (command == QStringLiteral("hello")) {
                send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                      {"service", "workpiece-orientation"}, {"ready", true}});
            } else if (command == QStringLiteral("register") && registerAttempts_++ == 0) {
                send({{"version", 1}, {"request_id", requestId}, {"ok", false},
                      {"error", QJsonObject{{"code", "WORKPIECE_EXISTS"}, {"message", "exists"}}}});
            } else if (command == QStringLiteral("register")) {
                send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                      {"workpiece", QJsonObject{{"id", "id-1"}, {"name", "M7"}}}});
            } else if (command == QStringLiteral("list_workpieces")) {
                send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                      {"workpieces", QJsonArray{QJsonObject{{"id", "id-1"}, {"name", "M7"}}}}});
            }
        }
    }

private:
    void send(const QJsonObject &object) {
        socket_->write(QJsonDocument(object).toJson(QJsonDocument::Compact) + "\n");
        socket_->flush();
    }

    QTcpServer server_;
    QTcpSocket *socket_ = nullptr;
    QByteArray buffer_;
    QList<QJsonObject> requests_;
    int registerAttempts_ = 0;
};

class TestMainWindow : public QObject {
    Q_OBJECT

private:
    static AppConfig configFor(quint16 port) {
        AppConfig config;
        config.host = QHostAddress::LocalHost;
        config.port = port;
        config.startupTimeoutMs = 200;
        config.requestTimeoutMs = 500;
        config.backendScript = QStringLiteral("service.py");
        return config;
    }

    static QStringList writeImages(QTemporaryDir &dir, const QString &prefix, int count) {
        QStringList paths;
        for (int i = 0; i < count; ++i) {
            const QString path = dir.filePath(QStringLiteral("%1-%2.png").arg(prefix).arg(i));
            QImage image(32, 24, QImage::Format_RGB32);
            image.fill(Qt::white);
            if (!image.save(path)) {
                return {};
            }
            paths.append(path);
        }
        return paths;
    }

private slots:
    void requiresExactlyFiveUniqueImagesPerSide() {
        QTemporaryDir dir;
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(37651), &client, &launcher);
        MainWindow window(&client, &manager);
        emit manager.backendReady();
        window.setWorkpieceName(QStringLiteral("M7"));
        window.setTemplatePaths(writeImages(dir, QStringLiteral("front"), 4),
                                writeImages(dir, QStringLiteral("back"), 5));

        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration", Qt::DirectConnection));
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text().contains(QStringLiteral("5")));
    }

    void confirmsBeforeSendingReplaceTrue() {
        RegistrationServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        window.setReplaceConfirmationHandler([](const QString &) { return true; });
        QTemporaryDir dir;
        window.setWorkpieceName(QStringLiteral("M7"));
        window.setTemplatePaths(writeImages(dir, QStringLiteral("front"), 5),
                                writeImages(dir, QStringLiteral("back"), 5));
        QObject::connect(&client, &BackendClient::handshakeSucceeded, &manager, [&manager]() {
            emit manager.backendReady();
        });
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() >= 2, 1000);
        QTest::qWait(20);
        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration", Qt::DirectConnection));
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() >= 4, 1500);

        int registerRequests = 0;
        bool sawReplaceFalse = false;
        bool sawReplaceTrue = false;
        for (const QJsonObject &request : server.requests()) {
            if (request.value(QStringLiteral("command")).toString() != QStringLiteral("register")) {
                continue;
            }
            ++registerRequests;
            if (request.value(QStringLiteral("replace")).toBool()) {
                sawReplaceTrue = true;
            } else {
                sawReplaceFalse = true;
            }
        }
        QCOMPARE(registerRequests, 2);
        QVERIFY(sawReplaceFalse);
        QVERIFY(sawReplaceTrue);
    }

    void busyStateDisablesRegisterAndPredict() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.stateChanged(BackendClient::State::Busy, QStringLiteral("busy"));
        QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("registerButton"))->isEnabled());
        QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("predictButton"))->isEnabled());
    }

    void disconnectClearsPreviousPredictionAndEnablesRestart() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.handshakeSucceeded();
        window.setInspectionImagePath(QStringLiteral("C:/tmp/sample.png"));
        emit client.responseReceived(QStringLiteral("predict"),
                                     QJsonObject{{"label", "front"}, {"needs_review", false}});
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("resultLabel"))->text().contains(QStringLiteral("正面")));
        emit client.connectionLost(QStringLiteral("lost"));
        QCOMPARE(window.findChild<QLabel *>(QStringLiteral("resultLabel"))->text(), QStringLiteral("尚未检测"));
        QCOMPARE(window.findChild<QLabel *>(QStringLiteral("imagePreviewLabel"))->text(), QStringLiteral("请选择待测图片"));
        QVERIFY(window.findChild<QPushButton *>(QStringLiteral("restartBackendButton"))->isEnabled());
    }

    void predictionShowsRawEvidenceWithoutPercentConfidence() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.handshakeSucceeded();
        emit client.responseReceived(QStringLiteral("predict"), QJsonObject{
            {"label", "back"}, {"global_scores", QJsonObject{{"front", 0.91}, {"back", 0.88}}},
            {"global_margin", 0.03}, {"local_prediction", "front"},
            {"local_scores", QJsonObject{{"front", 12.4}, {"back", 5.1}}},
            {"local_margin", 7.3}, {"decision_source", "local_override"},
            {"needs_review", true}, {"elapsed_ms", 248.5}});
        const QString evidence = window.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))->toPlainText();
        QVERIFY(evidence.contains(QStringLiteral("0.91")));
        QVERIFY(evidence.contains(QStringLiteral("248.5")));
        QVERIFY(!evidence.contains(QStringLiteral("%")));
        QCOMPARE(window.findChild<QLabel *>(QStringLiteral("reviewLabel"))->text(), QStringLiteral("建议人工复检"));
    }
};

QTEST_MAIN(TestMainWindow)
#include "test_mainwindow.moc"
