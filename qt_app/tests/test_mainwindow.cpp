#include <QtTest/QtTest>

#include <QJsonDocument>
#include <QDateTime>
#include <QJsonArray>
#include <QJsonObject>
#include <QComboBox>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QQueue>
#include <QSet>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QTemporaryDir>
#include <QTimer>
#include <QTcpServer>
#include <QTcpSocket>
#include <QImage>

#include "../backendclient.h"
#include "../backendprocessmanager.h"
#include "../inspectionimageview.h"
#include "../inspectionpage.h"
#include "../inspectiontypes.h"
#include "../mainwindow.h"
#include "../processlauncher.h"

class MessageBoxTextCapture : public QObject {
public:
    QString text;
    bool wasShown = false;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override {
        auto *dialog = qobject_cast<QMessageBox *>(watched);
        if (dialog == nullptr || event->type() != QEvent::Show) {
            return false;
        }
        wasShown = true;
        text = dialog->text();
        QTimer::singleShot(0, dialog, [dialog]() { dialog->done(QMessageBox::No); });
        return true;
    }
};

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
                      {"workpiece", QJsonObject{{"id", "id-1"}, {"name", "M7"}}},
                      {"template_counts", QJsonObject{{"front", 1}, {"back", 12}}},
                      {"elapsed_ms", 321.5}});
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

class BatchPredictionServer : public QObject {
    Q_OBJECT

public:
    explicit BatchPredictionServer(QObject *parent = nullptr) : QObject(parent) {
        connect(&server_, &QTcpServer::newConnection, this, &BatchPredictionServer::acceptConnection);
    }

    bool listen() { return server_.listen(QHostAddress::LocalHost, 0); }
    quint16 port() const { return server_.serverPort(); }
    int predictionCount() const { return predictionCount_; }
    int confirmationCount() const { return confirmationCount_; }
    int listWorkpieceCount() const { return listWorkpieceCount_; }
    int recycleCount() const { return recycleCount_; }
    QList<QJsonObject> requests() const { return requests_; }
    void setReviewRows(const QSet<int> &rows) { reviewRows_ = rows; }
    void setHoldPredictions(bool hold) { holdPredictions_ = hold; }
    void setHoldConfirmations(bool hold) { holdConfirmations_ = hold; }
    void setHoldWorkpieceResponses(bool hold) { holdWorkpieceResponses_ = hold; }
    void failNextWorkpieceRefresh() { failNextWorkpieceRefresh_ = true; }
    void failPredictionRow(int zeroBasedRow) { failedPredictionRow_ = zeroBasedRow; }
    void failNextConfirmation() { failNextConfirmation_ = true; }
    void failNextConfirmationJob() { failNextConfirmationJob_ = true; }
    void disconnectClient() {
        if (socket_ != nullptr) socket_->abort();
    }
    void replyNextPrediction() {
        if (pendingPredictions_.isEmpty()) {
            return;
        }
        const PendingPrediction pending = pendingPredictions_.dequeue();
        sendPrediction(pending.requestId, pending.oneBasedIndex);
    }

private slots:
    void acceptConnection() {
        socket_ = server_.nextPendingConnection();
        connect(socket_, &QTcpSocket::readyRead, this, &BatchPredictionServer::readRequests);
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
            } else if (command == QStringLiteral("list_workpieces")) {
                ++listWorkpieceCount_;
                if (holdWorkpieceResponses_) {
                    continue;
                }
                if (failNextWorkpieceRefresh_) {
                    failNextWorkpieceRefresh_ = false;
                    send({{"version", 1}, {"request_id", requestId}, {"ok", false},
                          {"error", QJsonObject{{"code", "REFRESH_FAILED"},
                                                {"message", "refresh failed"}}}});
                } else {
                    send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                          {"workpieces", QJsonArray{QJsonObject{{"id", "m1"}, {"name", "M1"}}}}});
                }
            } else if (command == QStringLiteral("recycle_workpiece")) {
                ++recycleCount_;
                send({{"version", 1}, {"request_id", requestId}, {"ok", true}});
            } else if (command == QStringLiteral("predict")) {
                ++predictionCount_;
                const int zeroBasedRow = predictionCount_ - 1;
                if (zeroBasedRow == failedPredictionRow_) {
                    send({{"version", 1}, {"request_id", requestId}, {"ok", false},
                          {"error", QJsonObject{{"code", "MODEL_ERROR"},
                                                {"message", "predict failed"}}}});
                    continue;
                }
                if (holdPredictions_) {
                    pendingPredictions_.enqueue(PendingPrediction{requestId, predictionCount_});
                } else {
                    sendPrediction(requestId, predictionCount_);
                }
            } else if (command == QStringLiteral("submit_confirmation")) {
                ++confirmationCount_;
                if (holdConfirmations_) {
                    continue;
                }
                if (failNextConfirmation_) {
                    failNextConfirmation_ = false;
                    send({{"version", 1}, {"request_id", requestId}, {"ok", false},
                          {"error", QJsonObject{{"code", "MODEL_ERROR"},
                                                {"message", "queue failed"}}}});
                } else if (failNextConfirmationJob_) {
                    failNextConfirmationJob_ = false;
                    send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                          {"job", QJsonObject{{"job_id", "job-failed"},
                                               {"state", "failed"},
                                               {"error", "job failed"}}}});
                } else {
                    send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                          {"job", QJsonObject{{"job_id",
                                               QStringLiteral("job-%1").arg(confirmationCount_)}}}});
                }
            }
        }
    }

private:
    struct PendingPrediction {
        QString requestId;
        int oneBasedIndex = 0;
    };

    void sendPrediction(const QString &requestId, int oneBasedIndex) {
        const double frontScore = 0.95 - 0.10 * oneBasedIndex;
        send({
            {"version", 1}, {"request_id", requestId}, {"ok", true},
            {"label", oneBasedIndex % 2 == 0 ? "back" : "front"},
            {"global_scores", QJsonObject{{"front", frontScore}, {"back", 1.0 - frontScore}}},
            {"global_margin", qAbs(frontScore - (1.0 - frontScore))},
            {"local_prediction", "front"},
            {"local_scores", QJsonObject{{"front", 8.0 + oneBasedIndex}, {"back", 1.0}}},
            {"local_margin", 7.0 + oneBasedIndex}, {"decision_source", "global"},
            {"needs_review", reviewRows_.contains(oneBasedIndex - 1)},
            {"elapsed_ms", 12.5 + oneBasedIndex},
        });
    }

    void send(const QJsonObject &object) {
        socket_->write(QJsonDocument(object).toJson(QJsonDocument::Compact) + "\n");
        socket_->flush();
    }

    QTcpServer server_;
    QTcpSocket *socket_ = nullptr;
    QByteArray buffer_;
    QList<QJsonObject> requests_;
    QQueue<PendingPrediction> pendingPredictions_;
    QSet<int> reviewRows_;
    bool holdPredictions_ = false;
    bool holdConfirmations_ = false;
    bool holdWorkpieceResponses_ = false;
    bool failNextWorkpieceRefresh_ = false;
    bool failNextConfirmation_ = false;
    bool failNextConfirmationJob_ = false;
    int failedPredictionRow_ = -1;
    int predictionCount_ = 0;
    int confirmationCount_ = 0;
    int listWorkpieceCount_ = 0;
    int recycleCount_ = 0;
};

class GeometryWorkflowServer : public QObject {
    Q_OBJECT

public:
    explicit GeometryWorkflowServer(QObject *parent = nullptr) : QObject(parent) {
        connect(&server_, &QTcpServer::newConnection, this, &GeometryWorkflowServer::acceptConnection);
    }

    bool listen() { return server_.listen(QHostAddress::LocalHost, 0); }
    quint16 port() const { return server_.serverPort(); }
    QStringList workflowCommands() const { return workflowCommands_; }

private slots:
    void acceptConnection() {
        socket_ = server_.nextPendingConnection();
        connect(socket_, &QTcpSocket::readyRead, this, &GeometryWorkflowServer::readRequests);
    }

    void readRequests() {
        buffer_ += socket_->readAll();
        while (buffer_.contains('\n')) {
            const int newline = buffer_.indexOf('\n');
            const QJsonDocument document = QJsonDocument::fromJson(buffer_.left(newline));
            buffer_.remove(0, newline + 1);
            if (!document.isObject()) continue;
            const QJsonObject request = document.object();
            const QString command = request.value(QStringLiteral("command")).toString();
            const QString requestId = request.value(QStringLiteral("request_id")).toString();
            if (command == QStringLiteral("hello")) {
                send(QJsonObject{{QStringLiteral("version"), 1}, {QStringLiteral("request_id"), requestId},
                                 {QStringLiteral("ok"), true}, {QStringLiteral("service"), QStringLiteral("workpiece-orientation")},
                                 {QStringLiteral("ready"), true}});
            } else if (command == QStringLiteral("list_workpieces")) {
                const QJsonArray workpieces{QJsonObject{
                    {QStringLiteral("id"), QStringLiteral("m1")},
                    {QStringLiteral("name"), QStringLiteral("M1")}}};
                send(QJsonObject{{QStringLiteral("version"), 1},
                                 {QStringLiteral("request_id"), requestId},
                                 {QStringLiteral("ok"), true},
                                 {QStringLiteral("workpieces"), workpieces}});
            } else if (command == QStringLiteral("get_geometry_mask_profile")) {
                sendSuccess(requestId, QJsonObject{{QStringLiteral("profile"), profile(1, 1, 0)}});
            } else if (command == QStringLiteral("save_geometry_mask_draft")) {
                workflowCommands_.append(command);
                sendSuccess(requestId, QJsonObject{{QStringLiteral("profile"), profile(1, 2, 0)}});
            } else if (command == QStringLiteral("validate_geometry_mask_draft")) {
                workflowCommands_.append(command);
                sendSuccess(requestId, QJsonObject{{QStringLiteral("job"), QJsonObject{
                    {QStringLiteral("job_id"), QStringLiteral("job-1")},
                    {QStringLiteral("state"), QStringLiteral("completed")},
                    {QStringLiteral("base_library_revision"), 1},
                    {QStringLiteral("base_draft_revision"), 2},
                    {QStringLiteral("blocking_issues"), QJsonArray()},
                    {QStringLiteral("warnings"), QJsonArray()}}}});
            } else if (command == QStringLiteral("publish_geometry_mask_profile")) {
                workflowCommands_.append(command);
                sendSuccess(requestId, QJsonObject{{QStringLiteral("profile"), profile(1, 2, 1)}});
            }
        }
    }

private:
    static QJsonObject profile(int libraryRevision, int draftRevision, int activeRevision) {
        const QJsonObject geometry{{QStringLiteral("cx"), 0.0}, {QStringLiteral("cy"), 0.0},
                                   {QStringLiteral("r"), 0.7}, {QStringLiteral("angle_deg"), 0.0}};
        const QJsonObject calibration{{QStringLiteral("state"), QStringLiteral("ready")},
                                      {QStringLiteral("geometry"), geometry},
                                      {QStringLiteral("seed_geometry"), geometry},
                                      {QStringLiteral("reference_template"), QJsonObject{
                                          {QStringLiteral("template_id"), QStringLiteral("front:00.png")},
                                          {QStringLiteral("direction"), QStringLiteral("front")},
                                          {QStringLiteral("width"), 120}, {QStringLiteral("height"), 120}}}};
        QJsonObject backCalibration = calibration;
        backCalibration.insert(QStringLiteral("reference_template"), QJsonObject{
            {QStringLiteral("template_id"), QStringLiteral("back:00.png")},
            {QStringLiteral("direction"), QStringLiteral("back")},
            {QStringLiteral("width"), 120}, {QStringLiteral("height"), 120}});
        const QJsonObject draft{
            {QStringLiteral("schema_version"), 2},
            {QStringLiteral("rules"), QJsonArray{QJsonObject{
                {QStringLiteral("rule_id"), QStringLiteral("glare")},
                {QStringLiteral("name"), QStringLiteral("中心反光")},
                {QStringLiteral("shape"), QStringLiteral("circle")},
                {QStringLiteral("mode"), QStringLiteral("inside")},
                {QStringLiteral("margin_ratio"), 0.0},
                {QStringLiteral("enabled"), true}}}},
            {QStringLiteral("directions"), QJsonObject{
                {QStringLiteral("front"), QJsonObject{{QStringLiteral("anchor"), QJsonValue()},
                    {QStringLiteral("calibrations"), QJsonObject{{QStringLiteral("glare"), calibration}}},
                    {QStringLiteral("template_reviews"), QJsonObject()}}},
                {QStringLiteral("back"), QJsonObject{{QStringLiteral("anchor"), QJsonValue()},
                    {QStringLiteral("calibrations"), QJsonObject{{QStringLiteral("glare"), backCalibration}}},
                    {QStringLiteral("template_reviews"), QJsonObject()}}}}},
            {QStringLiteral("migration"), QJsonObject{{QStringLiteral("conflicts"), QJsonArray()},
                                                        {QStringLiteral("resolutions"), QJsonArray()}}}};
        return QJsonObject{{QStringLiteral("workpiece_id"), QStringLiteral("m1")},
                           {QStringLiteral("library_revision"), libraryRevision},
                           {QStringLiteral("draft_revision"), draftRevision},
                           {QStringLiteral("active_revision"), activeRevision},
                           {QStringLiteral("draft"), draft}, {QStringLiteral("templates"), QJsonArray()}};
    }

    void sendSuccess(const QString &requestId, const QJsonObject &fields) {
        QJsonObject response = fields;
        response.insert(QStringLiteral("version"), 1);
        response.insert(QStringLiteral("request_id"), requestId);
        response.insert(QStringLiteral("ok"), true);
        send(response);
    }

    void send(const QJsonObject &object) {
        socket_->write(QJsonDocument(object).toJson(QJsonDocument::Compact) + "\n");
        socket_->flush();
    }

    QTcpServer server_;
    QTcpSocket *socket_ = nullptr;
    QByteArray buffer_;
    QStringList workflowCommands_;
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
            const uint marker = static_cast<uint>(qHash(prefix)) + static_cast<uint>(i + 1);
            image.setPixel(0, 0, qRgb(marker & 0xffu, (marker >> 8) & 0xffu, (marker >> 16) & 0xffu));
            if (!image.save(path)) {
                return {};
            }
            paths.append(path);
        }
        return paths;
    }

    static void startBatch(BatchPredictionServer &server, BackendClient &client,
                           MainWindow &window, const QStringList &paths,
                           const QJsonArray &workpieces = QJsonArray{
                               QJsonObject{{"id", "m1"}, {"name", "M1"}}}) {
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        emit client.responseReceived(QStringLiteral("list_workpieces"),
                                     QJsonObject{{"workpieces", workpieces}});
        window.setBatchImagePaths(paths);
        QVERIFY(QMetaObject::invokeMethod(&window, "submitBatchPrediction",
                                          Qt::DirectConnection));
    }

    static void waitForBatchCompletion(MainWindow &window, int expected) {
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(inspectionPage->completedBatchCount(), expected, 5000);
    }

private slots:
    void deleteConfirmationShowsDisplayNameInsteadOfInternalId() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.handshakeSucceeded();
        emit client.responseReceived(
            QStringLiteral("list_workpieces"),
            QJsonObject{{QStringLiteral("workpieces"),
                         QJsonArray{QJsonObject{
                             {QStringLiteral("id"), QStringLiteral("internal-workpiece-id")},
                             {QStringLiteral("name"), QStringLiteral("泵体 A")}}}}});

        auto *deleteButton = window.findChild<QPushButton *>(
            QStringLiteral("deleteWorkpieceButton"));
        QVERIFY(deleteButton != nullptr);
        QVERIFY(deleteButton->isEnabled());
        window.show();
        QCoreApplication::processEvents();

        MessageBoxTextCapture capture;
        qApp->installEventFilter(&capture);
        deleteButton->click();
        qApp->removeEventFilter(&capture);

        QVERIFY(capture.wasShown);
        QVERIFY(capture.text.contains(QStringLiteral("泵体 A")));
        QVERIFY(!capture.text.contains(QStringLiteral("internal-workpiece-id")));
    }

    void successfulCallbackQueuesSerialRefreshUntilReady() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        MainWindow window(&client, nullptr);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listWorkpieceCount(), 1, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);

        client.sendRequest(QStringLiteral("recycle_workpiece"), QJsonObject{
            {QStringLiteral("workpiece_id"), QStringLiteral("m1")},
            {QStringLiteral("operation_id"), QStringLiteral("test-recycle")},
        });

        QTRY_COMPARE_WITH_TIMEOUT(server.recycleCount(), 1, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listWorkpieceCount(), 2, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
    }

    void failedCallbackQueuesSerialRefreshUntilReady() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.failPredictionRow(0);
        BackendClient client;
        MainWindow window(&client, nullptr);
        QTemporaryDir dir;
        const QString imagePath = writeImages(
            dir, QStringLiteral("failed-followup"), 1).constFirst();
        QVERIFY(!imagePath.isEmpty());
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listWorkpieceCount(), 1, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        window.setInspectionImagePath(imagePath);
        bool failureCallbackRan = false;
        connect(&client, &BackendClient::commandFailed, &window,
                [&window, &failureCallbackRan](const QString &command,
                                                const QString &, const QString &) {
            if (command != QStringLiteral("predict")) return;
            failureCallbackRan = true;
            QMetaObject::invokeMethod(&window, "refreshWorkpieces", Qt::DirectConnection);
        });

        QVERIFY(QMetaObject::invokeMethod(&window, "submitPrediction", Qt::DirectConnection));

        QTRY_COMPARE_WITH_TIMEOUT(server.predictionCount(), 1, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(failureCallbackRan, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(window.findChild<InspectionPage *>()->uiState(),
                                  InspectionUiState::Failed, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listWorkpieceCount(), 2, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
    }

    void startsOnInspectionAndPreservesHeaderAcrossNavigation() {
        MainWindow window;
        auto *stack = window.findChild<QStackedWidget *>(
            QStringLiteral("mainPageStack"));
        QVERIFY(stack != nullptr);
        QCOMPARE(stack->currentIndex(), 0);
        QVERIFY(QMetaObject::invokeMethod(&window, "showWorkpieceLibrary",
                                          Qt::DirectConnection));
        QCOMPARE(stack->currentIndex(), 1);
        QVERIFY(window.findChild<QLabel *>(
                    QStringLiteral("backendStatusLabel")) != nullptr);
        QVERIFY(window.findChild<QComboBox *>(
                    QStringLiteral("workpieceComboBox")) != nullptr);
    }

    void acceptsUnequalTemplateCountsAndShowsLowCountWarning() {
        QTemporaryDir dir;
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(37651), &client, &launcher);
        MainWindow window(&client, &manager);
        QMetaObject::invokeMethod(&window, "onBackendReady", Qt::DirectConnection);
        window.setWorkpieceName(QStringLiteral("M7"));
        window.setTemplatePaths(writeImages(dir, QStringLiteral("front"), 1),
                                writeImages(dir, QStringLiteral("back"), 12));

        QCOMPARE(window.findChild<QLabel *>(QStringLiteral("frontTemplatesLabel"))->text(),
                 QStringLiteral("正面已选择 1 张"));
        QCOMPARE(window.findChild<QLabel *>(QStringLiteral("backTemplatesLabel"))->text(),
                 QStringLiteral("反面已选择 12 张"));
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("templateWarningLabel"))->text().contains(QStringLiteral("模板较少")));
        QVERIFY(window.findChild<QPushButton *>(QStringLiteral("registerButton"))->isEnabled());
    }

    void exposesLifecycleConfirmationAndAnnotationControls() {
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(37651), &client, &launcher);
        MainWindow window(&client, &manager);
        QVERIFY(window.findChild<QPushButton *>(QStringLiteral("deleteWorkpieceButton")) != nullptr);
        QVERIFY(window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton")) != nullptr);
        QVERIFY(window.findChild<QPushButton *>(QStringLiteral("confirmBackButton")) != nullptr);
        auto *geometryButton = window.findChild<QPushButton *>(QStringLiteral("annotationEditorButton"));
        QVERIFY(geometryButton != nullptr);
        QCOMPARE(geometryButton->text(), QStringLiteral("管理几何干扰规则"));
    }

    void allowsMoreThanThirtyTemplatesWithPerformanceWarning() {
        QTemporaryDir dir;
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(37651), &client, &launcher);
        MainWindow window(&client, &manager);
        QMetaObject::invokeMethod(&window, "onBackendReady", Qt::DirectConnection);
        window.setWorkpieceName(QStringLiteral("M7"));
        window.setTemplatePaths(writeImages(dir, QStringLiteral("front"), 31),
                                writeImages(dir, QStringLiteral("back"), 1));

        QVERIFY(window.findChild<QLabel *>(QStringLiteral("templateWarningLabel"))->text().contains(QStringLiteral("耗时")));
        QVERIFY(window.findChild<QPushButton *>(QStringLiteral("registerButton"))->isEnabled());
    }

    void registrationCompletionShowsActualTemplateCounts() {
        RegistrationServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        window.setReplaceConfirmationHandler([](const QString &) { return true; });
        QTemporaryDir dir;
        window.setWorkpieceName(QStringLiteral("M7"));
        window.setTemplatePaths(writeImages(dir, QStringLiteral("front"), 1),
                                writeImages(dir, QStringLiteral("back"), 12));
        QObject::connect(&client, &BackendClient::handshakeSucceeded, &manager, [&manager]() {
            emit manager.backendReady();
        });
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(server.requests().size() >= 2, 1000);
        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration", Qt::DirectConnection));

        QTRY_VERIFY_WITH_TIMEOUT(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text().contains(QStringLiteral("正面 1 张")), 1500);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text().contains(QStringLiteral("反面 12 张")));
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
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("registerButton"))->isEnabled());
        QVERIFY(!inspectionPage->findChild<QPushButton *>(
                     QStringLiteral("predictButton"))->isEnabled());
    }

    void loadingStateShowsStatusAndKeepsActionsDisabled() {
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(37651), &client, &launcher);
        MainWindow window(&client, &manager);

        emit manager.backendLoading(QStringLiteral("模型加载中"));

        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("backendStatusLabel"))->text().contains(QStringLiteral("模型加载中")));
        QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("registerButton"))->isEnabled());
        QVERIFY(!inspectionPage->findChild<QPushButton *>(
                     QStringLiteral("predictButton"))->isEnabled());
    }

    void transportFailurePreservesPreviousPredictionAndMarksOutcomeUnknown() {
        QTemporaryDir dir;
        const QString imagePath = writeImages(dir, QStringLiteral("preserved"), 1).constFirst();
        QVERIFY(!imagePath.isEmpty());
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.handshakeSucceeded();
        window.setInspectionImagePath(imagePath);
        emit client.responseReceived(QStringLiteral("predict"),
                                     QJsonObject{{"label", "front"}, {"needs_review", false}});
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        QVERIFY(inspectionPage->findChild<QLabel *>(
                    QStringLiteral("resultLabel"))->text().contains(QStringLiteral("正面")));
        emit client.transportFailed(QStringLiteral("CONNECTION_LOST"), QStringLiteral("lost"));
        QVERIFY(inspectionPage->findChild<QLabel *>(
                    QStringLiteral("resultLabel"))->text().contains(QStringLiteral("正面")));
        QCOMPARE(inspectionPage->findChild<InspectionImageView *>()->imagePath(), imagePath);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text().contains(QStringLiteral("未知")));
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
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        const QString evidence = inspectionPage->findChild<QTextEdit *>(
            QStringLiteral("rawEvidenceTextEdit"))->toPlainText();
        QVERIFY(evidence.contains(QStringLiteral("0.91")));
        QVERIFY(evidence.contains(QStringLiteral("248.5")));
        QVERIFY(!evidence.contains(QStringLiteral("%")));
        QCOMPARE(inspectionPage->findChild<QLabel *>(
                     QStringLiteral("reviewLabel"))->text(), QStringLiteral("建议人工复检"));
    }

    void batchPredictionSendsEachImageAndShowsSummary() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QStringList paths = writeImages(dir, QStringLiteral("batch"), 3);

        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        emit client.responseReceived(QStringLiteral("list_workpieces"), QJsonObject{
            {"workpieces", QJsonArray{QJsonObject{{"id", "m1"}, {"name", "M1"}}}},
        });
        window.setBatchImagePaths(paths);

        QPushButton *batchButton = window.findChild<QPushButton *>(QStringLiteral("batchPredictButton"));
        QVERIFY(batchButton != nullptr);
        QVERIFY(batchButton->isEnabled());
        QVERIFY(QMetaObject::invokeMethod(&window, "submitBatchPrediction", Qt::DirectConnection));

        QTRY_COMPARE_WITH_TIMEOUT(server.predictionCount(), 3, 1500);
        QTableWidget *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        QVERIFY(table != nullptr);
        waitForBatchCompletion(window, 3);
        QCOMPARE(table->rowCount(), 3);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("batchSummaryLabel"))->text().contains(QStringLiteral("3")));

        emit client.commandFailed(QStringLiteral("predict"), QStringLiteral("MODEL_ERROR"),
                                  QStringLiteral("synthetic failure"));
        QCOMPARE(table->rowCount(), 3);
    }

    void selectingBatchRowShowsItsImageAndEvidence() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QStringList paths = writeImages(dir, QStringLiteral("select"), 3);

        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        emit client.responseReceived(QStringLiteral("list_workpieces"), QJsonObject{
            {"workpieces", QJsonArray{QJsonObject{{"id", "m1"}, {"name", "M1"}}}},
        });
        window.setBatchImagePaths(paths);
        QVERIFY(QMetaObject::invokeMethod(&window, "submitBatchPrediction", Qt::DirectConnection));

        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 3);
        QCOMPARE(table->rowCount(), 3);
        QCOMPARE(table->columnCount(), 5);
        table->setCurrentCell(1, 0);

        QTRY_VERIFY(window.findChild<QLabel *>(QStringLiteral("currentImageLabel"))
                        ->text().contains(QStringLiteral("select-1.png")));
        QVERIFY(window.findChild<QTextEdit *>(QStringLiteral("rawEvidenceTextEdit"))
                    ->toPlainText().contains(QStringLiteral("0.75")));
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"))
                    ->text().contains(QStringLiteral("select-1.png")));
    }

    void laterBatchResponsesDoNotReplaceManualSelection() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setHoldPredictions(true);
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("held"), 4));

        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        QTRY_VERIFY(server.predictionCount() >= 1);
        server.replyNextPrediction();
        QTRY_COMPARE(inspectionPage->completedBatchCount(), 1);
        QCOMPARE(table->rowCount(), 4);
        QTRY_VERIFY(server.predictionCount() >= 2);
        server.replyNextPrediction();
        QTRY_COMPARE(inspectionPage->completedBatchCount(), 2);
        QCOMPARE(table->rowCount(), 4);
        table->setCurrentCell(0, 0);
        QTRY_VERIFY(server.predictionCount() >= 3);
        server.replyNextPrediction();
        QTRY_COMPARE(inspectionPage->completedBatchCount(), 3);
        QCOMPARE(table->rowCount(), 4);

        QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentImageLabel"))
                    ->text().contains(QStringLiteral("held-0.png")));
        QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->isEnabled());
    }

    void userRefreshWaitsForBatchAndStillRuns() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setHoldPredictions(true);
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QStringList paths = writeImages(dir, QStringLiteral("queued-system"), 3);
        startBatch(server, client, window, paths);
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        QTRY_COMPARE(server.predictionCount(), 1);
        const int refreshCountBeforeClick = server.listWorkpieceCount();

        QVERIFY(QMetaObject::invokeMethod(&window, "refreshWorkpieces",
                                          Qt::DirectConnection));
        server.replyNextPrediction();

        QTRY_COMPARE_WITH_TIMEOUT(server.predictionCount(), 2, 1000);
        server.replyNextPrediction();
        QTRY_COMPARE_WITH_TIMEOUT(server.predictionCount(), 3, 1000);
        server.replyNextPrediction();
        QTRY_COMPARE_WITH_TIMEOUT(inspectionPage->completedBatchCount(), 3, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listWorkpieceCount(),
                                  refreshCountBeforeClick + 1, 1000);
    }

    void deferredUserRefreshSurvivesTransportUntilSuccess() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setHoldPredictions(true);
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(
            dir, QStringLiteral("refresh-reconnect"), 2));
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        QTRY_COMPARE(server.predictionCount(), 1);

        QVERIFY(QMetaObject::invokeMethod(&window, "refreshWorkpieces",
                                          Qt::DirectConnection));
        QVERIFY(QMetaObject::invokeMethod(&window, "refreshWorkpieces",
                                          Qt::DirectConnection));
        server.replyNextPrediction();
        QTRY_COMPARE(server.predictionCount(), 2);
        server.setHoldWorkpieceResponses(true);
        const int refreshCountBeforeAttempt = server.listWorkpieceCount();
        server.replyNextPrediction();
        QTRY_COMPARE(inspectionPage->completedBatchCount(), 2);
        QTRY_COMPARE(server.listWorkpieceCount(), refreshCountBeforeAttempt + 1);
        const int refreshCountBeforeReconnect = server.listWorkpieceCount();

        server.disconnectClient();
        QTRY_VERIFY(client.state() == BackendClient::State::Disconnected
                    || client.state() == BackendClient::State::Error);
        server.setHoldWorkpieceResponses(false);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listWorkpieceCount(),
                                  refreshCountBeforeReconnect + 2, 1000);
        const int refreshCountAfterSuccess = server.listWorkpieceCount();

        emit client.stateChanged(BackendClient::State::Ready,
                                 QStringLiteral("still ready"));
        QTest::qWait(100);
        QCOMPARE(server.listWorkpieceCount(), refreshCountAfterSuccess);
    }

    void userRefreshCommandFailureStopsAutomaticRetry() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_VERIFY(server.listWorkpieceCount() >= 1);
        QTRY_VERIFY(client.state() == BackendClient::State::Ready);
        server.failNextWorkpieceRefresh();
        const int refreshCountBeforeAttempt = server.listWorkpieceCount();

        QVERIFY(QMetaObject::invokeMethod(&window, "refreshWorkpieces",
                                          Qt::DirectConnection));

        QTRY_COMPARE(server.listWorkpieceCount(), refreshCountBeforeAttempt + 1);
        QTRY_VERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))
                        ->text().contains(QStringLiteral("刷新失败")));
        QTRY_VERIFY(client.state() == BackendClient::State::Ready);
        const int refreshCountAfterFailure = server.listWorkpieceCount();
        emit client.stateChanged(BackendClient::State::Ready,
                                 QStringLiteral("still ready"));
        QTest::qWait(100);
        QCOMPARE(server.listWorkpieceCount(), refreshCountAfterFailure);
    }

    void queuedInternalRefreshDoesNotStrandBatchPrediction() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setHoldPredictions(true);
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QStringList paths = writeImages(dir, QStringLiteral("queued-internal"), 3);
        startBatch(server, client, window, paths);
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        QTRY_COMPARE(server.predictionCount(), 1);

        QVERIFY(QMetaObject::invokeMethod(
            &window, "onClientResponse", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("recycle_workpiece")),
            Q_ARG(QJsonObject, QJsonObject())));
        server.replyNextPrediction();

        QTRY_COMPARE_WITH_TIMEOUT(server.predictionCount(), 2, 1000);
        server.replyNextPrediction();
        QTRY_COMPARE_WITH_TIMEOUT(server.predictionCount(), 3, 1000);
        server.replyNextPrediction();
        QTRY_COMPARE_WITH_TIMEOUT(inspectionPage->completedBatchCount(), 3, 1000);
    }

    void batchCompletionSelectsFirstReviewResult() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setReviewRows(QSet<int>{1, 2});
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("review"), 3));

        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 3);
        QTRY_COMPARE(table->currentRow(), 1);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"))
                    ->text().contains(QStringLiteral("review-1.png")));
    }

    void batchWithoutReviewSelectsFirstResult() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("normal"), 3));

        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 3);
        QTRY_COMPARE(table->currentRow(), 0);
    }

    void confirmingBatchResultUpdatesStatusAndAdvances() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setReviewRows(QSet<int>{0, 1});
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("confirm"), 3));
        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 3);
        QTRY_COMPARE(table->currentRow(), 0);

        window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->click();
        QTRY_COMPARE(server.confirmationCount(), 1);
        QTRY_COMPARE(table->item(0, 4)->text(), QStringLiteral("正面已排队"));
        QTRY_COMPARE(table->currentRow(), 1);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("batchSummaryLabel"))
                    ->text().contains(QStringLiteral("已处理 3/3")));
        table->setCurrentCell(0, 0);
        auto *frontButton = window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
        QTRY_VERIFY(!frontButton->isEnabled());
        frontButton->click();
        QCOMPARE(server.confirmationCount(), 1);
    }

    void filteredBatchConfirmationUsesStableRecordIdentity() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setReviewRows(QSet<int>{1});
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QStringList paths = writeImages(dir, QStringLiteral("filtered-confirm"), 3);
        startBatch(server, client, window, paths);
        auto *inspectionPage = window.findChild<InspectionPage *>();
        auto *table = window.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        QVERIFY(inspectionPage != nullptr);
        waitForBatchCompletion(window, 3);

        inspectionPage->setBatchFilter(BatchFilter::NeedsReview);
        QCOMPARE(table->rowCount(), 1);
        table->setCurrentCell(0, 0);
        window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->click();
        QTRY_COMPARE(server.confirmationCount(), 1);

        QJsonObject confirmation;
        for (const QJsonObject &request : server.requests()) {
            if (request.value(QStringLiteral("command")).toString()
                == QStringLiteral("submit_confirmation")) {
                confirmation = request;
            }
        }
        QCOMPARE(confirmation.value(QStringLiteral("image_path")).toString(), paths.at(1));
        QVERIFY(!confirmation.value(QStringLiteral("operation_id")).toString().isEmpty());
    }

    void rejectingBatchResultIsLocalAndAdvances() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setReviewRows(QSet<int>{0, 1});
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("reject"), 3));
        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 3);
        const int before = server.confirmationCount();

        window.findChild<QPushButton *>(QStringLiteral("rejectConfirmationButton"))->click();
        QCOMPARE(server.confirmationCount(), before);
        QCOMPARE(table->item(0, 4)->text(), QStringLiteral("不入库"));
        QCOMPARE(table->currentRow(), 1);
    }

    void failedBatchConfirmationStaysSelectedAndCanRetry() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setReviewRows(QSet<int>{0});
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QStringList paths = writeImages(dir, QStringLiteral("failed"), 2);
        startBatch(server, client, window, paths);
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 2);
        server.failNextConfirmation();
        const QString recordId = table->item(0, 0)->data(Qt::UserRole).toString();

        window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->click();
        QTRY_COMPARE(table->item(0, 4)->text(), QStringLiteral("提交失败"));
        QCOMPARE(table->currentRow(), 0);
        QCOMPARE(inspectionPage->selectedRecordId(), recordId);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))
                    ->text().contains(QStringLiteral("queue failed")));
        auto *backButton = window.findChild<QPushButton *>(QStringLiteral("confirmBackButton"));
        QTRY_VERIFY(backButton->isEnabled());
        backButton->click();
        QTRY_COMPARE(server.confirmationCount(), 2);

        QList<QJsonObject> confirmations;
        for (const QJsonObject &request : server.requests()) {
            if (request.value(QStringLiteral("command")).toString()
                == QStringLiteral("submit_confirmation")) {
                confirmations.append(request);
            }
        }
        QCOMPARE(confirmations.size(), 2);
        QCOMPARE(confirmations.at(0).value(QStringLiteral("image_path")).toString(),
                 paths.at(0));
        QCOMPARE(confirmations.at(1).value(QStringLiteral("image_path")).toString(),
                 paths.at(0));
        QCOMPARE(confirmations.at(0).value(QStringLiteral("orientation")).toString(),
                 QStringLiteral("front"));
        QCOMPARE(confirmations.at(1).value(QStringLiteral("orientation")).toString(),
                 QStringLiteral("back"));
        QVERIFY(confirmations.at(0).value(QStringLiteral("operation_id")).toString()
                != confirmations.at(1).value(QStringLiteral("operation_id")).toString());
    }

    void failedConfirmationJobReportsFailureAndCanRetry() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setReviewRows(QSet<int>{0});
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(
            dir, QStringLiteral("failed-job"), 1));
        auto *table = window.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 1);
        QTRY_VERIFY(client.state() == BackendClient::State::Ready);
        auto *frontButton = window.findChild<QPushButton *>(
            QStringLiteral("confirmFrontButton"));
        QTRY_VERIFY(frontButton->isEnabled());
        server.failNextConfirmationJob();

        frontButton->click();

        QTRY_COMPARE(table->item(0, 4)->text(), QStringLiteral("提交失败"));
        QCOMPARE(table->currentRow(), 0);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text()
                    .contains(QStringLiteral("job failed")));
        QTRY_VERIFY(frontButton->isEnabled());
    }

    void workpieceSwitchPreservesBatchButBlocksConfirmationUntilSwitchedBack() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QJsonArray workpieces{
            QJsonObject{{"id", "m1"}, {"name", "M1"}},
            QJsonObject{{"id", "m2"}, {"name", "M2"}}};
        startBatch(server, client, window, writeImages(dir, QStringLiteral("switch"), 2),
                   workpieces);
        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        auto *combo = window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
        auto *front = window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
        waitForBatchCompletion(window, 2);
        QTRY_VERIFY(front->isEnabled());

        combo->setCurrentIndex(1);
        QCOMPARE(table->rowCount(), 2);
        QVERIFY(!front->isEnabled());
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"))
                    ->text().contains(QStringLiteral("其他工件")));
        combo->setCurrentIndex(0);
        QTRY_VERIFY(front->isEnabled());
    }

    void switchingWorkpieceDuringBatchDoesNotRetargetPendingRequests() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setHoldPredictions(true);
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QJsonArray workpieces{
            QJsonObject{{"id", "m1"}, {"name", "M1"}},
            QJsonObject{{"id", "m2"}, {"name", "M2"}}};
        startBatch(server, client, window, writeImages(dir, QStringLiteral("fixed-workpiece"), 2),
                   workpieces);
        QTRY_COMPARE(server.predictionCount(), 1);

        window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"))->setCurrentIndex(1);
        server.replyNextPrediction();
        QTRY_COMPARE(server.predictionCount(), 2);

        QList<QJsonObject> predictionRequests;
        for (const QJsonObject &request : server.requests()) {
            if (request.value(QStringLiteral("command")).toString() == QStringLiteral("predict")) {
                predictionRequests.append(request);
            }
        }
        QCOMPARE(predictionRequests.size(), 2);
        QCOMPARE(predictionRequests.at(0).value(QStringLiteral("workpiece_id")).toString(),
                 QStringLiteral("m1"));
        QCOMPARE(predictionRequests.at(1).value(QStringLiteral("workpiece_id")).toString(),
                 QStringLiteral("m1"));
    }

    void disconnectPreservesBatchEvidenceAndDisablesConfirmation() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("disconnect"), 2));
        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 2);
        table->setCurrentCell(1, 0);
        const QString evidence = window.findChild<QTextEdit *>(
            QStringLiteral("evidenceTextEdit"))->toPlainText();

        emit client.transportFailed(QStringLiteral("CONNECTION_LOST"), QStringLiteral("lost"));
        QCOMPARE(table->rowCount(), 2);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentImageLabel"))
                    ->text().contains(QStringLiteral("disconnect-1.png")));
        QCOMPARE(window.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))
                     ->toPlainText(), evidence);
        QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->isEnabled());
    }

    void disconnectDuringBatchConfirmationMarksOutcomeUnknown() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setHoldConfirmations(true);
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("confirm-lost"), 2));
        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 2);

        window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->click();
        QTRY_COMPARE(server.confirmationCount(), 1);
        emit client.transportFailed(QStringLiteral("CONNECTION_LOST"), QStringLiteral("lost"));

        QCOMPARE(table->item(0, 4)->text(), QStringLiteral("提交失败"));
        QCOMPARE(table->currentRow(), 0);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))
                    ->text().contains(QStringLiteral("结果未知")));
        QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->isEnabled());
    }

    void transportUnknownConfirmationRequiresIdenticalPayload() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.setHoldConfirmations(true);
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(
            dir, QStringLiteral("confirm-idempotent"), 1));
        waitForBatchCompletion(window, 1);
        auto *frontButton = window.findChild<QPushButton *>(
            QStringLiteral("confirmFrontButton"));
        auto *backButton = window.findChild<QPushButton *>(
            QStringLiteral("confirmBackButton"));
        QVERIFY(frontButton != nullptr);
        QVERIFY(backButton != nullptr);
        QTRY_VERIFY(frontButton->isEnabled());

        frontButton->click();
        QTRY_COMPARE(server.confirmationCount(), 1);
        server.disconnectClient();
        QTRY_VERIFY(client.state() == BackendClient::State::Disconnected
                    || client.state() == BackendClient::State::Error);
        server.setHoldConfirmations(false);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_VERIFY(frontButton->isEnabled());

        backButton->click();
        QTest::qWait(50);
        QCOMPARE(server.confirmationCount(), 1);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))
                    ->text().contains(QStringLiteral("正面")));
        QTRY_VERIFY(frontButton->isEnabled());

        frontButton->click();
        QTRY_COMPARE(server.confirmationCount(), 2);
        QList<QJsonObject> confirmations;
        for (const QJsonObject &request : server.requests()) {
            if (request.value(QStringLiteral("command")).toString()
                == QStringLiteral("submit_confirmation")) {
                confirmations.append(request);
            }
        }
        QCOMPARE(confirmations.size(), 2);
        const QString firstOperationId = confirmations.at(0)
            .value(QStringLiteral("operation_id")).toString();
        QVERIFY(!firstOperationId.isEmpty());
        QCOMPARE(confirmations.at(1).value(QStringLiteral("operation_id")).toString(),
                 firstOperationId);
    }

    void replacingBatchSelectionClearsOldRowsAndContext() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("old"), 2));
        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 2);
        QVERIFY(!window.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))
                     ->toPlainText().isEmpty());

        window.setBatchImagePaths(writeImages(dir, QStringLiteral("new"), 3));
        QCOMPARE(table->rowCount(), 0);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("currentResultTargetLabel"))->text().isEmpty());
        QVERIFY(window.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))
                    ->toPlainText().isEmpty());
        QVERIFY(window.findChild<InspectionImageView *>()->imagePath().isEmpty());
        QVERIFY(!window.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"))->isEnabled());
    }

    void replacingBatchImagesPreservesPriorSingleResult() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        const QString singlePath = writeImages(
            dir, QStringLiteral("single-preserved"), 1).constFirst();
        const QStringList oldBatchPaths = writeImages(
            dir, QStringLiteral("old-batch"), 2);
        const QStringList newBatchPaths = writeImages(
            dir, QStringLiteral("new-batch"), 2);
        QVERIFY(!singlePath.isEmpty());
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        InspectionRecord singleRecord;
        singleRecord.id = QStringLiteral("single-before-batch");
        singleRecord.imagePath = singlePath;
        singleRecord.workpieceId = QStringLiteral("m1");
        singleRecord.label = QStringLiteral("front");
        singleRecord.response = QJsonObject{
            {QStringLiteral("label"), QStringLiteral("front")},
            {QStringLiteral("global_prediction"), QStringLiteral("front")},
            {QStringLiteral("global_margin"), 0.42}};
        singleRecord.completedAt = QDateTime::currentDateTime();
        inspectionPage->setMode(InspectionMode::Single);
        inspectionPage->showSingleResult(singleRecord);
        const QString singleEvidence = inspectionPage->findChild<QTextEdit *>(
            QStringLiteral("evidenceTextEdit"))->toPlainText();
        startBatch(server, client, window, oldBatchPaths);
        waitForBatchCompletion(window, oldBatchPaths.size());
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);

        window.setBatchImagePaths(newBatchPaths);
        inspectionPage->setMode(InspectionMode::Single);

        QCOMPARE(inspectionPage->singleImagePath(), singlePath);
        QVERIFY(inspectionPage->findChild<QLabel *>(QStringLiteral("resultLabel"))->text()
                    .contains(QStringLiteral("正面")));
        QCOMPARE(inspectionPage->findChild<QTextEdit *>(
                     QStringLiteral("evidenceTextEdit"))->toPlainText(), singleEvidence);
    }

    void midBatchFailureKeepsCompletedRowsAndShowsExactProgress() {
        BatchPredictionServer server;
        QVERIFY(server.listen());
        server.failPredictionRow(2);
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QTemporaryDir dir;
        startBatch(server, client, window, writeImages(dir, QStringLiteral("partial"), 4));
        auto *table = window.findChild<QTableWidget *>(QStringLiteral("batchResultsTableWidget"));
        waitForBatchCompletion(window, 4);
        bool foundFailedRecord = false;
        for (int row = 0; row < table->rowCount(); ++row) {
            if (table->item(row, 0)->text() == QStringLiteral("partial-2.png")) {
                foundFailedRecord = true;
                QCOMPARE(table->item(row, 4)->text(), QStringLiteral("预测失败"));
            }
        }
        QVERIFY(foundFailedRecord);
        const QString summary = window.findChild<QLabel *>(
            QStringLiteral("batchSummaryLabel"))->text();
        QTRY_VERIFY(summary.contains(QStringLiteral("已处理 4/4")));
        QVERIFY(summary.contains(QStringLiteral("失败 1")));
    }

    void unreadablePreviewClearsPreviousPixmapAndNamesFile() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        QTemporaryDir dir;
        const QString validPath = writeImages(dir, QStringLiteral("preview"), 1).first();
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        auto *preview = inspectionPage->findChild<InspectionImageView *>();
        QVERIFY(preview != nullptr);
        window.setInspectionImagePath(validPath);
        QCOMPARE(preview->imagePath(), validPath);

        window.setInspectionImagePath(dir.filePath(QStringLiteral("broken.png")));
        QVERIFY(preview->imagePath().isEmpty());
        QVERIFY(inspectionPage->findChild<QLabel *>(QStringLiteral("currentImageLabel"))
                    ->text().contains(QStringLiteral("broken.png")));
    }

    void returningToInspectionPreservesDisplayedImageAndResult() {
        QTemporaryDir dir;
        const QString imagePath = writeImages(dir, QStringLiteral("return"), 1).constFirst();
        QVERIFY(!imagePath.isEmpty());
        MainWindow window;
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        InspectionRecord record;
        record.id = QStringLiteral("return-record");
        record.imagePath = imagePath;
        record.workpieceId = QStringLiteral("m1");
        record.label = QStringLiteral("front");
        record.response = QJsonObject{{QStringLiteral("label"), QStringLiteral("front")},
                                      {QStringLiteral("global_margin"), 0.12},
                                      {QStringLiteral("local_margin"), 3.4}};
        record.completedAt = QDateTime::currentDateTime();
        inspectionPage->showSingleResult(record);

        QVERIFY(QMetaObject::invokeMethod(&window, "showWorkpieceLibrary", Qt::DirectConnection));
        QVERIFY(QMetaObject::invokeMethod(&window, "showInspection", Qt::DirectConnection));

        QCOMPARE(inspectionPage->singleImagePath(), imagePath);
        QVERIFY(inspectionPage->findChild<QLabel *>(QStringLiteral("resultLabel"))
                    ->text().contains(QStringLiteral("正面")));
    }

    void switchingModesDoesNotOverwriteEitherResult() {
        QTemporaryDir dir;
        const QString singlePath = writeImages(dir, QStringLiteral("single-mode"), 1).constFirst();
        const QString batchPath = writeImages(dir, QStringLiteral("batch-mode"), 1).constFirst();
        QVERIFY(!singlePath.isEmpty());
        QVERIFY(!batchPath.isEmpty());
        MainWindow window;
        auto *inspectionPage = window.findChild<InspectionPage *>();
        QVERIFY(inspectionPage != nullptr);
        InspectionRecord singleRecord;
        singleRecord.id = QStringLiteral("single-mode-record");
        singleRecord.imagePath = singlePath;
        singleRecord.label = QStringLiteral("front");
        singleRecord.response = QJsonObject{{QStringLiteral("label"), QStringLiteral("front")}};
        singleRecord.completedAt = QDateTime::currentDateTime();
        inspectionPage->setMode(InspectionMode::Single);
        inspectionPage->showSingleResult(singleRecord);
        InspectionRecord batchRecord = singleRecord;
        batchRecord.id = QStringLiteral("batch-mode-record");
        batchRecord.imagePath = batchPath;
        batchRecord.label = QStringLiteral("back");
        batchRecord.response = QJsonObject{{QStringLiteral("label"), QStringLiteral("back")}};
        inspectionPage->setMode(InspectionMode::Batch);
        inspectionPage->showSingleResult(batchRecord);

        inspectionPage->setMode(InspectionMode::Single);
        QCOMPARE(inspectionPage->singleImagePath(), singlePath);
        QVERIFY(inspectionPage->findChild<QLabel *>(QStringLiteral("resultLabel"))
                    ->text().contains(QStringLiteral("正面")));
        inspectionPage->setMode(InspectionMode::Batch);
        QCOMPARE(inspectionPage->singleImagePath(), batchPath);
        QVERIFY(inspectionPage->findChild<QLabel *>(QStringLiteral("resultLabel"))
                    ->text().contains(QStringLiteral("反面")));
    }

    void geometryPublishWorkflowSendsSaveValidatePublishInOrder() {
        GeometryWorkflowServer server;
        QVERIFY(server.listen());
        BackendClient client;
        PassiveLauncher launcher;
        BackendProcessManager manager(configFor(server.port()), &client, &launcher);
        MainWindow window(&client, &manager);
        QObject::connect(&client, &BackendClient::handshakeSucceeded, &manager, [&manager]() {
            emit manager.backendReady();
        });
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        auto *geometryButton = window.findChild<QPushButton *>(QStringLiteral("annotationEditorButton"));
        QTRY_VERIFY_WITH_TIMEOUT(geometryButton != nullptr && geometryButton->isEnabled(), 1000);
        geometryButton->click();
        auto *workflowButton = window.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
        QTRY_VERIFY_WITH_TIMEOUT(workflowButton != nullptr && workflowButton->isEnabled(), 1000);
        workflowButton->click();

        QTRY_COMPARE_WITH_TIMEOUT(server.workflowCommands().size(), 3, 2000);
        QCOMPARE(server.workflowCommands(), QStringList({
            QStringLiteral("save_geometry_mask_draft"),
            QStringLiteral("validate_geometry_mask_draft"),
            QStringLiteral("publish_geometry_mask_profile")}));
    }
};

QTEST_MAIN(TestMainWindow)
#include "test_mainwindow.moc"
