#include <QtTest/QtTest>

#include <QJsonDocument>
#include <QDateTime>
#include <QJsonArray>
#include <QJsonObject>
#include <QComboBox>
#include <QLabel>
#include <QListWidget>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QProgressBar>
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
#include "../geometryrulespage.h"
#include "../inspectionimageview.h"
#include "../inspectionpage.h"
#include "../inspectiontypes.h"
#include "../mainwindow.h"
#include "../processlauncher.h"
#include "../workpiecelibrarypage.h"

class MessageBoxButtonChooser : public QObject {
public:
    explicit MessageBoxButtonChooser(const QString &text)
        : QObject(qApp), text_(text) {
        qApp->installEventFilter(this);
    }

protected:
    bool eventFilter(QObject *watched, QEvent *event) override {
        auto *dialog = qobject_cast<QMessageBox *>(watched);
        if (dialog == nullptr || event->type() != QEvent::Show) return false;
        qApp->removeEventFilter(this);
        const QString text = text_;
        QTimer::singleShot(0, dialog, [dialog, text]() {
            for (QAbstractButton *button : dialog->buttons()) {
                if (button->text() == text) {
                    button->click();
                    return;
                }
            }
        });
        deleteLater();
        return true;
    }

private:
    QString text_;
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
    int registerCount() const { return registerAttempts_; }

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

class LibraryIntentServer : public QObject {
    Q_OBJECT

public:
    explicit LibraryIntentServer(QObject *parent = nullptr) : QObject(parent) {
        connect(&server_, &QTcpServer::newConnection,
                this, &LibraryIntentServer::acceptConnection);
    }

    bool listen() { return server_.listen(QHostAddress::LocalHost, 0); }
    quint16 port() const { return server_.serverPort(); }
    void holdFirstRegister() { holdFirstRegister_ = true; }
    void holdRecycle() { holdRecycle_ = true; }
    void failNextListRefresh() { failNextListRefresh_ = true; }
    void failNextDetailsRefresh() { failNextDetailsRefresh_ = true; }
    void disconnectNextListRefresh() { disconnectNextListRefresh_ = true; }
    int registerCount() const { return registerCount_; }
    int recycleCount() const { return recycleCount_; }
    int listCount() const { return listCount_; }
    QStringList detailsWorkpieceIds() const { return detailsWorkpieceIds_; }
    QList<QJsonObject> requests() const { return requests_; }

    bool replyFirstRegisterExists() {
        if (heldRegisterRequestId_.isEmpty()) return false;
        send({{"version", 1}, {"request_id", heldRegisterRequestId_}, {"ok", false},
              {"error", QJsonObject{{"code", "WORKPIECE_EXISTS"},
                                     {"message", "exists"}}}});
        heldRegisterRequestId_.clear();
        return true;
    }

    bool replyRecycleSuccess() {
        if (heldRecycleRequestId_.isEmpty()) return false;
        send({{"version", 1}, {"request_id", heldRecycleRequestId_}, {"ok", true}});
        heldRecycleRequestId_.clear();
        return true;
    }

private slots:
    void acceptConnection() {
        socket_ = server_.nextPendingConnection();
        connect(socket_, &QTcpSocket::readyRead,
                this, &LibraryIntentServer::readRequests);
    }

    void readRequests() {
        buffer_ += socket_->readAll();
        while (buffer_.contains('\n')) {
            const int newline = buffer_.indexOf('\n');
            const QJsonDocument document = QJsonDocument::fromJson(buffer_.left(newline));
            buffer_.remove(0, newline + 1);
            if (!document.isObject()) continue;
            const QJsonObject request = document.object();
            requests_.append(request);
            const QString command = request.value(QStringLiteral("command")).toString();
            const QString requestId = request.value(QStringLiteral("request_id")).toString();
            if (command == QStringLiteral("hello")) {
                send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                      {"service", "workpiece-orientation"}, {"ready", true}});
            } else if (command == QStringLiteral("list_workpieces")) {
                ++listCount_;
                if (disconnectNextListRefresh_) {
                    disconnectNextListRefresh_ = false;
                    socket_->abort();
                } else if (failNextListRefresh_) {
                    failNextListRefresh_ = false;
                    send({{"version", 1}, {"request_id", requestId}, {"ok", false},
                          {"error", QJsonObject{{"code", "REFRESH_FAILED"},
                                                 {"message", "refresh failed"}}}});
                } else {
                    send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                          {"workpieces", QJsonArray{
                              QJsonObject{{"id", "m1"}, {"name", "M1"},
                                          {"template_counts", QJsonObject{{"front", 2}, {"back", 3}}}},
                              QJsonObject{{"id", "m2"}, {"name", "M2"},
                                          {"template_counts", QJsonObject{{"front", 4}, {"back", 5}}}},
                          }}});
                }
            } else if (command == QStringLiteral("get_workpiece_details")) {
                const QString workpieceId = request.value(
                    QStringLiteral("workpiece_id")).toString();
                detailsWorkpieceIds_.append(workpieceId);
                if (failNextDetailsRefresh_) {
                    failNextDetailsRefresh_ = false;
                    send({{"version", 1}, {"request_id", requestId}, {"ok", false},
                          {"error", QJsonObject{{"code", "DETAILS_FAILED"},
                                                 {"message", "details failed"}}}});
                } else {
                    ++detailsSuccessCount_;
                    send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                          {"workpiece", QJsonObject{
                              {"id", workpieceId},
                              {"name", workpieceId.toUpper()},
                              {"template_counts", QJsonObject{
                                  {"front", 5 + detailsSuccessCount_ * 2},
                                  {"back", 6 + detailsSuccessCount_ * 2}}},
                              {"geometry_rule_count", 2},
                              {"detectable", true}}}});
                }
            } else if (command == QStringLiteral("register")) {
                ++registerCount_;
                if (registerCount_ == 1 && holdFirstRegister_) {
                    heldRegisterRequestId_ = requestId;
                } else {
                    send({{"version", 1}, {"request_id", requestId}, {"ok", true},
                          {"workpiece", QJsonObject{{"id", "registered"},
                                                     {"name", request.value("name")}}},
                          {"template_counts", QJsonObject{
                              {"front", request.value("front_images").toArray().size()},
                              {"back", request.value("back_images").toArray().size()}}},
                          {"elapsed_ms", 17}});
                }
            } else if (command == QStringLiteral("recycle_workpiece")) {
                ++recycleCount_;
                if (holdRecycle_) {
                    heldRecycleRequestId_ = requestId;
                } else {
                    send({{"version", 1}, {"request_id", requestId}, {"ok", true}});
                }
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
    QStringList detailsWorkpieceIds_;
    QString heldRegisterRequestId_;
    QString heldRecycleRequestId_;
    bool holdFirstRegister_ = false;
    bool holdRecycle_ = false;
    bool failNextListRefresh_ = false;
    bool failNextDetailsRefresh_ = false;
    bool disconnectNextListRefresh_ = false;
    int registerCount_ = 0;
    int recycleCount_ = 0;
    int listCount_ = 0;
    int detailsSuccessCount_ = 0;
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
    QStringList geometryProfileWorkpieceIds() const {
        return geometryProfileWorkpieceIds_;
    }
    QStringList validationPollJobIds() const { return validationPollJobIds_; }
    void holdGeometryProfiles(bool hold = true) { holdGeometryProfiles_ = hold; }
    bool replyNextGeometryProfile() {
        if (heldGeometryProfiles_.isEmpty()) return false;
        const QPair<QString, QString> pending = heldGeometryProfiles_.dequeue();
        sendSuccess(pending.first, QJsonObject{{QStringLiteral("profile"),
            profile(1, 1, 0, pending.second)}});
        return true;
    }

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
                const QJsonArray workpieces{
                    QJsonObject{{QStringLiteral("id"), QStringLiteral("m1")},
                                {QStringLiteral("name"), QStringLiteral("M1")}},
                    QJsonObject{{QStringLiteral("id"), QStringLiteral("m2")},
                                {QStringLiteral("name"), QStringLiteral("M2")}}};
                send(QJsonObject{{QStringLiteral("version"), 1},
                                 {QStringLiteral("request_id"), requestId},
                                 {QStringLiteral("ok"), true},
                                 {QStringLiteral("workpieces"), workpieces}});
            } else if (command == QStringLiteral("get_geometry_mask_profile")) {
                const QString workpieceId = request.value(QStringLiteral("workpiece_id")).toString();
                geometryProfileWorkpieceIds_.append(workpieceId);
                if (holdGeometryProfiles_) {
                    heldGeometryProfiles_.enqueue(qMakePair(requestId, workpieceId));
                } else {
                    sendSuccess(requestId, QJsonObject{{QStringLiteral("profile"),
                        profile(1, 1, 0, workpieceId)}});
                }
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
            } else if (command == QStringLiteral("get_geometry_mask_validation_job")) {
                const QString jobId = request.value(QStringLiteral("job_id")).toString();
                validationPollJobIds_.append(jobId);
                sendSuccess(requestId, QJsonObject{{QStringLiteral("job"), QJsonObject{
                    {QStringLiteral("job_id"), jobId},
                    {QStringLiteral("state"), QStringLiteral("running")},
                    {QStringLiteral("progress"), QJsonObject{
                        {QStringLiteral("completed"), 2},
                        {QStringLiteral("total"), 5}}}}}});
            } else if (command == QStringLiteral("publish_geometry_mask_profile")) {
                workflowCommands_.append(command);
                sendSuccess(requestId, QJsonObject{{QStringLiteral("profile"), profile(1, 2, 1)}});
            }
        }
    }

public:
    static QJsonObject profile(int libraryRevision, int draftRevision, int activeRevision,
                               const QString &workpieceId = QStringLiteral("m1")) {
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
        return QJsonObject{{QStringLiteral("workpiece_id"), workpieceId},
                           {QStringLiteral("library_revision"), libraryRevision},
                           {QStringLiteral("draft_revision"), draftRevision},
                           {QStringLiteral("active_revision"), activeRevision},
                           {QStringLiteral("draft"), draft}, {QStringLiteral("templates"), QJsonArray()}};
    }

private:
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
    QStringList geometryProfileWorkpieceIds_;
    QStringList validationPollJobIds_;
    QQueue<QPair<QString, QString>> heldGeometryProfiles_;
    bool holdGeometryProfiles_ = false;
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

    static void chooseDirtyNavigationOption(const QString &text) {
        new MessageBoxButtonChooser(text);
    }

    static QTimer *geometryValidationTimer(MainWindow &window) {
        for (QTimer *timer : window.findChildren<QTimer *>()) {
            if (timer->interval() == 1000) return timer;
        }
        return nullptr;
    }

    static QJsonObject validationJob(const QString &state, int completed = 2,
                                     int total = 5) {
        return QJsonObject{
            {QStringLiteral("job_id"), QStringLiteral("job-lifecycle")},
            {QStringLiteral("state"), state},
            {QStringLiteral("base_library_revision"), 1},
            {QStringLiteral("base_draft_revision"), 2},
            {QStringLiteral("progress"), QJsonObject{
                {QStringLiteral("completed"), completed},
                {QStringLiteral("total"), total}}},
            {QStringLiteral("blocking_issues"), QJsonArray()},
            {QStringLiteral("warnings"), QJsonArray()}};
    }

    static bool deliverValidationJob(MainWindow &window, const QString &command,
                                     const QJsonObject &job) {
        const QJsonObject response{{QStringLiteral("job"), job}};
        return QMetaObject::invokeMethod(
            &window, "onClientResponse", Qt::DirectConnection,
            Q_ARG(QString, command), Q_ARG(QJsonObject, response));
    }

private slots:
    void geometryPageIsEmbeddedAndPersistent() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        auto *stack = window.findChild<QStackedWidget *>(
            QStringLiteral("mainPageStack"));
        auto *host = window.findChild<QWidget *>(
            QStringLiteral("geometryPageHost"));
        QVERIFY(stack != nullptr);
        QVERIFY(host != nullptr);
        QCOMPARE(stack->count(), 3);

        QVERIFY(window.requestPage(AppPage::GeometryRules));
        auto *page = window.findChild<QWidget *>(
            QStringLiteral("geometryRulesPage"));
        QVERIFY(page != nullptr);
        QVERIFY(host->isAncestorOf(page));
        QVERIFY(!page->isWindow());

        QVERIFY(window.requestPage(AppPage::Inspection));
        QVERIFY(window.requestPage(AppPage::GeometryRules));
        QCOMPARE(window.findChild<QWidget *>(QStringLiteral("geometryRulesPage")), page);
        QCOMPARE(stack->count(), 3);
    }

    void geometrySnapshotLoadsOnlyOnFirstEntryForSameWorkpiece() {
        GeometryWorkflowServer server;
        QVERIFY(server.listen());
        BackendClient client;
        MainWindow window(&client, nullptr);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(window.findChild<QComboBox *>(
            QStringLiteral("workpieceComboBox"))->currentData().toString()
            == QStringLiteral("m1"), 1000);

        QVERIFY(window.requestPage(AppPage::GeometryRules));
        QTRY_COMPARE_WITH_TIMEOUT(server.geometryProfileWorkpieceIds().size(), 1, 1000);
        QCOMPARE(server.geometryProfileWorkpieceIds().first(), QStringLiteral("m1"));
        auto *page = window.findChild<GeometryRulesPage *>(
            QStringLiteral("geometryRulesPage"));
        QVERIFY(page != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(page->snapshot().value(
            QStringLiteral("workpiece_id")).toString(), QStringLiteral("m1"), 1000);

        QVERIFY(window.requestPage(AppPage::Inspection));
        QVERIFY(window.requestPage(AppPage::GeometryRules));
        QTest::qWait(50);
        QCOMPARE(server.geometryProfileWorkpieceIds().size(), 1);
    }

    void dirtyGeometryPageGuardsNavigationChoices() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        auto *page = window.findChild<GeometryRulesPage *>(
            QStringLiteral("geometryRulesPage"));
        auto *stack = window.findChild<QStackedWidget *>(
            QStringLiteral("mainPageStack"));
        QVERIFY(page != nullptr);
        QVERIFY(stack != nullptr);
        page->setSnapshot(GeometryWorkflowServer::profile(1, 1, 0));
        QVERIFY(window.requestPage(AppPage::GeometryRules));
        page->findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
        QVERIFY(page->hasUnsavedChanges());

        chooseDirtyNavigationOption(QStringLiteral("取消"));
        QVERIFY(!window.requestPage(AppPage::Inspection));
        QCOMPARE(stack->currentIndex(), static_cast<int>(AppPage::GeometryRules));
        QVERIFY(page->hasUnsavedChanges());

        chooseDirtyNavigationOption(QStringLiteral("放弃修改"));
        QVERIFY(window.requestPage(AppPage::Inspection));
        QCOMPARE(stack->currentIndex(), static_cast<int>(AppPage::Inspection));
        QVERIFY(!page->hasUnsavedChanges());

        QVERIFY(window.requestPage(AppPage::GeometryRules));
        page->findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
        chooseDirtyNavigationOption(QStringLiteral("保留并离开"));
        QVERIFY(window.requestPage(AppPage::Inspection));
        QVERIFY(page->hasUnsavedChanges());
    }

    void dirtyGeometryPageGuardsDetectionTargetChoices() {
        const QJsonObject workpieces{{QStringLiteral("workpieces"), QJsonArray{
            QJsonObject{{QStringLiteral("id"), QStringLiteral("m1")},
                        {QStringLiteral("name"), QStringLiteral("M1")}},
            QJsonObject{{QStringLiteral("id"), QStringLiteral("m2")},
                        {QStringLiteral("name"), QStringLiteral("M2")}}
        }}};

        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.responseReceived(QStringLiteral("list_workpieces"), workpieces);
        auto *combo = window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
        auto *page = window.findChild<GeometryRulesPage *>(
            QStringLiteral("geometryRulesPage"));
        auto *stack = window.findChild<QStackedWidget *>(
            QStringLiteral("mainPageStack"));
        QVERIFY(combo != nullptr);
        QVERIFY(page != nullptr);
        QVERIFY(stack != nullptr);
        QCOMPARE(combo->currentData().toString(), QStringLiteral("m1"));
        page->setSnapshot(GeometryWorkflowServer::profile(1, 1, 0));
        QVERIFY(window.requestPage(AppPage::GeometryRules));

        page->findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
        chooseDirtyNavigationOption(QStringLiteral("取消"));
        combo->setCurrentIndex(combo->findData(QStringLiteral("m2")));
        QCOMPARE(combo->currentData().toString(), QStringLiteral("m1"));
        QCOMPARE(stack->currentIndex(), static_cast<int>(AppPage::GeometryRules));
        QVERIFY(page->hasUnsavedChanges());

        chooseDirtyNavigationOption(QStringLiteral("放弃修改"));
        combo->setCurrentIndex(combo->findData(QStringLiteral("m2")));
        QCOMPARE(combo->currentData().toString(), QStringLiteral("m2"));
        QVERIFY(!page->hasUnsavedChanges());

        combo->setCurrentIndex(combo->findData(QStringLiteral("m1")));
        page->setSnapshot(GeometryWorkflowServer::profile(1, 1, 0));
        page->findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
        chooseDirtyNavigationOption(QStringLiteral("保留并离开"));
        combo->setCurrentIndex(combo->findData(QStringLiteral("m2")));
        QCOMPARE(combo->currentData().toString(), QStringLiteral("m2"));
        QCOMPARE(stack->currentIndex(), static_cast<int>(AppPage::Inspection));
        QVERIFY(page->hasUnsavedChanges());
    }

    void browsingLibraryDoesNotRetargetInspection() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.responseReceived(QStringLiteral("list_workpieces"), QJsonObject{
            {QStringLiteral("workpieces"), QJsonArray{
                QJsonObject{{QStringLiteral("id"), QStringLiteral("m1")},
                            {QStringLiteral("name"), QStringLiteral("M1")},
                            {QStringLiteral("template_counts"), QJsonObject{{"front", 5}, {"back", 5}}}},
                QJsonObject{{QStringLiteral("id"), QStringLiteral("m2")},
                            {QStringLiteral("name"), QStringLiteral("M2")},
                            {QStringLiteral("template_counts"), QJsonObject{{"front", 7}, {"back", 9}}}},
            }},
        });
        auto *headerTarget = window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
        auto *library = window.findChild<WorkpieceLibraryPage *>();
        auto *list = window.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        QVERIFY(headerTarget != nullptr);
        QVERIFY(library != nullptr);
        QVERIFY(list != nullptr);
        headerTarget->setCurrentIndex(headerTarget->findData(QStringLiteral("m1")));
        QVERIFY(window.requestPage(AppPage::WorkpieceLibrary));

        list->setCurrentRow(1);

        QCOMPARE(library->browsedWorkpieceId(), QStringLiteral("m2"));
        QCOMPARE(headerTarget->currentData().toString(), QStringLiteral("m1"));
    }

    void explicitSetCurrentUpdatesHeaderAndInspection() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.responseReceived(QStringLiteral("list_workpieces"), QJsonObject{
            {QStringLiteral("workpieces"), QJsonArray{
                QJsonObject{{QStringLiteral("id"), QStringLiteral("m1")},
                            {QStringLiteral("name"), QStringLiteral("M1")}},
                QJsonObject{{QStringLiteral("id"), QStringLiteral("m2")},
                            {QStringLiteral("name"), QStringLiteral("M2")}},
            }},
        });
        auto *headerTarget = window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
        auto *list = window.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        auto *activate = window.findChild<QPushButton *>(
            QStringLiteral("setCurrentWorkpieceButton"));
        QVERIFY(headerTarget != nullptr);
        QVERIFY(list != nullptr);
        QVERIFY(activate != nullptr);
        headerTarget->setCurrentIndex(headerTarget->findData(QStringLiteral("m1")));
        list->setCurrentRow(1);

        activate->click();

        QCOMPARE(headerTarget->currentData().toString(), QStringLiteral("m2"));
        auto *inspection = window.findChild<InspectionPage *>();
        QVERIFY(inspection != nullptr);
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString imagePath = writeImages(
            directory, QStringLiteral("inspection"), 1).first();
        inspection->setBackendAvailable(true, false, QString());
        inspection->setSingleImagePath(imagePath);
        QSignalSpy predictionSpy(inspection, &InspectionPage::commandRequested);
        auto *predict = inspection->findChild<QPushButton *>(
            QStringLiteral("predictButton"));
        QVERIFY(predict != nullptr);
        QVERIFY(predict->isEnabled());
        predict->click();
        QCOMPARE(predictionSpy.count(), 1);
        QCOMPARE(predictionSpy.first().at(1).toJsonObject()
                     .value(QStringLiteral("workpiece_id")).toString(),
                 QStringLiteral("m2"));
    }

    void registrationContinuesWhileInspectionPageIsVisible() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        BackendClient client;
        MainWindow window(&client, nullptr);
        auto *library = window.findChild<WorkpieceLibraryPage *>();
        QVERIFY(library != nullptr);
        library->setBackendState(BackendUiState::Ready, QString());
        library->setWorkpieceName(QStringLiteral("M-running"));
        library->setTemplatePaths(writeImages(directory, QStringLiteral("front"), 2),
                                  writeImages(directory, QStringLiteral("back"), 3));
        QVERIFY(QMetaObject::invokeMethod(library, "submitRegistration",
                                          Qt::DirectConnection));
        QVERIFY(window.requestPage(AppPage::Inspection));

        const QJsonObject progress{
            {QStringLiteral("phase"), QStringLiteral("features")},
            {QStringLiteral("completed"), 3},
            {QStringLiteral("total"), 5},
        };
        QVERIFY(QMetaObject::invokeMethod(
            &window, "onClientProgress", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("register")),
            Q_ARG(QJsonObject, progress)));

        auto *globalDetail = window.findChild<QLabel *>(QStringLiteral("globalTaskDetailLabel"));
        QVERIFY(globalDetail != nullptr);
        QVERIFY(globalDetail->text().contains(QStringLiteral("3/5")));
        QCOMPARE(window.findChild<QStackedWidget *>(QStringLiteral("mainPageStack"))->currentIndex(),
                 static_cast<int>(AppPage::Inspection));
    }

    void detailsFailureDoesNotOverwriteInspectionResult() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.responseReceived(QStringLiteral("predict"),
                                     QJsonObject{{QStringLiteral("label"), QStringLiteral("front")},
                                                 {QStringLiteral("needs_review"), false}});
        auto *result = window.findChild<QLabel *>(QStringLiteral("resultLabel"));
        QVERIFY(result != nullptr);
        const QString before = result->text();

        QVERIFY(QMetaObject::invokeMethod(
            &window, "onClientCommandFailed", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("get_workpiece_details")),
            Q_ARG(QString, QStringLiteral("WORKPIECE_NOT_FOUND")),
            Q_ARG(QString, QStringLiteral("详情不存在"))));

        QCOMPARE(result->text(), before);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text()
                    .contains(QStringLiteral("详情不存在")));
    }

    void reconnectPreservesLibraryDraftAndEvolutionRows() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        BackendClient client;
        MainWindow window(&client, nullptr);
        auto *library = window.findChild<WorkpieceLibraryPage *>();
        QVERIFY(library != nullptr);
        library->setWorkpieceName(QStringLiteral("M-draft"));
        library->setTemplatePaths(writeImages(directory, QStringLiteral("front"), 1),
                                  writeImages(directory, QStringLiteral("back"), 1));
        emit client.responseReceived(QStringLiteral("list_evolution_jobs"), QJsonObject{
            {QStringLiteral("jobs"), QJsonArray{QJsonObject{
                {QStringLiteral("job_id"), QStringLiteral("job-preserved")},
                {QStringLiteral("state"), QStringLiteral("failed")},
                {QStringLiteral("phase"), QStringLiteral("features")},
                {QStringLiteral("completed"), 1},
                {QStringLiteral("total"), 2},
                {QStringLiteral("error"), QStringLiteral("failed")},
            }}},
        });
        auto *globalDetail = window.findChild<QLabel *>(
            QStringLiteral("globalTaskDetailLabel"));
        QVERIFY(globalDetail != nullptr);
        QVERIFY(globalDetail->text().contains(QStringLiteral("任务失败")));

        QVERIFY(QMetaObject::invokeMethod(
            &window, "onBackendUnavailable", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("lost"))));
        QVERIFY(QMetaObject::invokeMethod(&window, "onBackendReady", Qt::DirectConnection));

        QVERIFY(library->hasUnsavedChanges());
        QCOMPARE(window.findChild<QLineEdit *>(QStringLiteral("workpieceNameEdit"))->text(),
                 QStringLiteral("M-draft"));
        QCOMPARE(window.findChild<QTableWidget *>(QStringLiteral("evolutionJobsTable"))->rowCount(),
                 1);
    }

    void failedTaskStatusRetainsCountsAndKnownOrUnknownElapsed() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        auto *library = window.findChild<WorkpieceLibraryPage *>();
        auto *detail = window.findChild<QLabel *>(QStringLiteral("globalTaskDetailLabel"));
        auto *progress = window.findChild<QProgressBar *>(
            QStringLiteral("globalTaskProgressBar"));
        QVERIFY(library != nullptr);
        QVERIFY(detail != nullptr);
        QVERIFY(progress != nullptr);

        library->setEvolutionJobs(QJsonArray{QJsonObject{
            {QStringLiteral("job_id"), QStringLiteral("known-failure")},
            {QStringLiteral("state"), QStringLiteral("failed")},
            {QStringLiteral("completed"), 2},
            {QStringLiteral("total"), 5},
            {QStringLiteral("elapsed_ms"), 1234},
        }});
        QCOMPARE(progress->value(), 2);
        QCOMPARE(progress->maximum(), 5);
        QVERIFY(detail->text().contains(QStringLiteral("2/5")));
        QVERIFY(detail->text().contains(QStringLiteral("1234 ms")));
        QCOMPARE(detail->parentWidget()->property("messageKind").toString(),
                 QStringLiteral("error"));

        library->setEvolutionJobs(QJsonArray{QJsonObject{
            {QStringLiteral("job_id"), QStringLiteral("unknown-failure")},
            {QStringLiteral("state"), QStringLiteral("failed")},
            {QStringLiteral("completed"), 4},
            {QStringLiteral("total"), 6},
            {QStringLiteral("elapsed_ms"), QJsonValue(QJsonValue::Null)},
        }});
        QCOMPARE(progress->value(), 4);
        QCOMPARE(progress->maximum(), 6);
        QVERIFY(detail->text().contains(QStringLiteral("4/6")));
        QVERIFY(detail->text().contains(QStringLiteral("耗时未知")));
        QCOMPARE(detail->parentWidget()->property("messageKind").toString(),
                 QStringLiteral("error"));
    }

    void decliningOverwriteShowsCancelledTaskAndAllowsImmediateResubmit() {
        RegistrationServer server;
        QVERIFY(server.listen());
        BackendClient client;
        MainWindow window(&client, nullptr);
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        window.setReplaceConfirmationHandler([](const QString &) { return false; });
        window.setWorkpieceName(QStringLiteral("M-cancel"));
        window.setTemplatePaths(
            writeImages(directory, QStringLiteral("cancel-front"), 1),
            writeImages(directory, QStringLiteral("cancel-back"), 2));
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);

        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration",
                                          Qt::DirectConnection));
        QTRY_COMPARE_WITH_TIMEOUT(server.registerCount(), 1, 1000);
        auto *detail = window.findChild<QLabel *>(QStringLiteral("globalTaskDetailLabel"));
        auto *progress = window.findChild<QProgressBar *>(
            QStringLiteral("globalTaskProgressBar"));
        auto *registerButton = window.findChild<QPushButton *>(
            QStringLiteral("registerButton"));
        QVERIFY(detail != nullptr);
        QVERIFY(progress != nullptr);
        QVERIFY(registerButton != nullptr);
        QTRY_VERIFY_WITH_TIMEOUT(detail->text().contains(QStringLiteral("建库已取消")), 1000);
        QVERIFY(detail->text().contains(QStringLiteral("0/3")));
        QVERIFY(detail->text().contains(QStringLiteral("耗时")));
        QVERIFY(!detail->text().contains(QStringLiteral("已用")));
        QVERIFY(!detail->text().contains(QStringLiteral("cancelled")));
        QCOMPARE(progress->value(), 0);
        QCOMPARE(progress->maximum(), 3);
        QCOMPARE(detail->parentWidget()->property("messageKind").toString(),
                 QStringLiteral("warning"));
        QTRY_VERIFY_WITH_TIMEOUT(registerButton->isEnabled(), 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);

        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration",
                                          Qt::DirectConnection));
        QTRY_COMPARE_WITH_TIMEOUT(server.registerCount(), 2, 1000);
    }

    void dirtyLibraryDraftCanCancelNavigation() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        auto *library = window.findChild<WorkpieceLibraryPage *>();
        auto *stack = window.findChild<QStackedWidget *>(QStringLiteral("mainPageStack"));
        QVERIFY(library != nullptr);
        QVERIFY(stack != nullptr);
        window.show();
        QCoreApplication::processEvents();
        QVERIFY(window.requestPage(AppPage::WorkpieceLibrary));

        const QStringList submittedFront = writeImages(
            directory, QStringLiteral("navigation-front"), 2);
        const QStringList submittedBack = writeImages(
            directory, QStringLiteral("navigation-back"), 3);
        library->setBackendState(BackendUiState::Ready, QString());
        library->setReplaceConfirmationHandler([](const QString &) { return true; });
        library->setWorkpieceName(QStringLiteral("keep-me"));
        library->setTemplatePaths(submittedFront, submittedBack);
        QSignalSpy commandSpy(library, &WorkpieceLibraryPage::commandRequested);
        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration",
                                          Qt::DirectConnection));
        QCOMPARE(commandSpy.count(), 1);

        chooseDirtyNavigationOption(QStringLiteral("取消"));
        QVERIFY(!window.requestPage(AppPage::Inspection));
        QCOMPARE(stack->currentIndex(), static_cast<int>(AppPage::WorkpieceLibrary));
        QVERIFY(library->hasUnsavedChanges());

        chooseDirtyNavigationOption(QStringLiteral("保留并离开"));
        QVERIFY(window.requestPage(AppPage::Inspection));
        QVERIFY(library->hasUnsavedChanges());

        QVERIFY(window.requestPage(AppPage::WorkpieceLibrary));
        chooseDirtyNavigationOption(QStringLiteral("放弃修改"));
        QVERIFY(window.requestPage(AppPage::Inspection));
        QVERIFY(!library->hasUnsavedChanges());

        library->handleBackendFailure(QStringLiteral("register"),
                                      QStringLiteral("WORKPIECE_EXISTS"),
                                      QStringLiteral("exists"));
        QCOMPARE(commandSpy.count(), 2);
        const QJsonObject replacement = commandSpy.at(1).at(1).toJsonObject();
        QCOMPARE(replacement.value(QStringLiteral("replace")).toBool(), true);
        QCOMPARE(replacement.value(QStringLiteral("name")).toString(),
                 QStringLiteral("keep-me"));
        QCOMPARE(replacement.value(QStringLiteral("front_images")).toArray(),
                 QJsonArray::fromStringList(submittedFront));
        QCOMPARE(replacement.value(QStringLiteral("back_images")).toArray(),
                 QJsonArray::fromStringList(submittedBack));
    }

    void deleteRequiresExactBrowsedDisplayName() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        emit client.responseReceived(
            QStringLiteral("list_workpieces"),
            QJsonObject{{QStringLiteral("workpieces"),
                         QJsonArray{QJsonObject{
                             {QStringLiteral("id"), QStringLiteral("internal-workpiece-id")},
                             {QStringLiteral("name"), QStringLiteral("泵体 A")}}}}});

        auto *library = window.findChild<WorkpieceLibraryPage *>();
        auto *list = window.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        auto *confirmation = window.findChild<QLineEdit *>(
            QStringLiteral("recycleNameConfirmationEdit"));
        auto *deleteButton = window.findChild<QPushButton *>(
            QStringLiteral("deleteWorkpieceButton"));
        QVERIFY(library != nullptr);
        QVERIFY(list != nullptr);
        QVERIFY(confirmation != nullptr);
        QVERIFY(deleteButton != nullptr);
        library->setBackendState(BackendUiState::Ready, QString());
        list->setCurrentRow(0);
        confirmation->setText(QStringLiteral("泵体"));
        QVERIFY(!deleteButton->isEnabled());
        confirmation->setText(QStringLiteral("泵体 A"));
        QVERIFY(deleteButton->isEnabled());

        QSignalSpy commandSpy(library, &WorkpieceLibraryPage::commandRequested);
        deleteButton->click();
        QCOMPARE(commandSpy.count(), 1);
        QCOMPARE(commandSpy.first().at(0).toString(),
                 QStringLiteral("recycle_workpiece"));
        QCOMPARE(commandSpy.first().at(1).toJsonObject()
                     .value(QStringLiteral("workpiece_id")).toString(),
                 QStringLiteral("internal-workpiece-id"));
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
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration", Qt::DirectConnection));

        auto *result = window.findChild<QLabel *>(
            QStringLiteral("latestRegistrationResultLabel"));
        QVERIFY(result != nullptr);
        QTRY_VERIFY_WITH_TIMEOUT(result->text().contains(QStringLiteral("正面 1 张")), 1500);
        QVERIFY(result->text().contains(QStringLiteral("反面 12 张")));
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

    void overwriteContinuationKeepsMandatoryRefreshAndLatestBrowseIntent() {
        LibraryIntentServer server;
        QVERIFY(server.listen());
        server.holdFirstRegister();
        BackendClient client;
        MainWindow window(&client, nullptr);
        window.setReplaceConfirmationHandler([](const QString &) { return true; });
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList front = writeImages(
            directory, QStringLiteral("route-front"), 2);
        const QStringList back = writeImages(
            directory, QStringLiteral("route-back"), 3);
        window.setWorkpieceName(QStringLiteral("M-route"));
        window.setTemplatePaths(front, back);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 1, 1000);
        auto *library = window.findChild<WorkpieceLibraryPage *>();
        auto *list = window.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        QVERIFY(library != nullptr);
        QVERIFY(list != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(list->count(), 2, 1000);
        QVERIFY(window.requestPage(AppPage::WorkpieceLibrary));

        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration",
                                          Qt::DirectConnection));
        QTRY_COMPARE_WITH_TIMEOUT(server.registerCount(), 1, 1000);
        list->setCurrentRow(1);
        QCOMPARE(library->browsedWorkpieceId(), QStringLiteral("m2"));
        QVERIFY(server.replyFirstRegisterExists());

        QTRY_COMPARE_WITH_TIMEOUT(server.registerCount(), 2, 1500);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 2, 1500);
        QTRY_VERIFY_WITH_TIMEOUT(
            server.detailsWorkpieceIds().contains(QStringLiteral("m2")), 1500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);

        QList<QJsonObject> routed;
        bool afterInitialRegister = false;
        for (const QJsonObject &request : server.requests()) {
            const QString command = request.value(QStringLiteral("command")).toString();
            if (command == QStringLiteral("register")) afterInitialRegister = true;
            if (afterInitialRegister
                && (command == QStringLiteral("register")
                    || command == QStringLiteral("list_workpieces")
                    || command == QStringLiteral("get_workpiece_details"))) {
                routed.append(request);
            }
        }
        QCOMPARE(routed.size(), 4);
        QCOMPARE(routed.at(0).value(QStringLiteral("command")).toString(),
                 QStringLiteral("register"));
        QCOMPARE(routed.at(0).value(QStringLiteral("replace")).toBool(), false);
        QCOMPARE(routed.at(1).value(QStringLiteral("command")).toString(),
                 QStringLiteral("register"));
        QCOMPARE(routed.at(1).value(QStringLiteral("replace")).toBool(), true);
        QCOMPARE(routed.at(1).value(QStringLiteral("front_images")).toArray(),
                 QJsonArray::fromStringList(front));
        QCOMPARE(routed.at(1).value(QStringLiteral("back_images")).toArray(),
                 QJsonArray::fromStringList(back));
        QCOMPARE(routed.at(2).value(QStringLiteral("command")).toString(),
                 QStringLiteral("list_workpieces"));
        QCOMPARE(routed.at(3).value(QStringLiteral("command")).toString(),
                 QStringLiteral("get_workpiece_details"));
        QCOMPARE(routed.at(3).value(QStringLiteral("workpiece_id")).toString(),
                 QStringLiteral("m2"));
    }

    void recycleSuccessKeepsMandatoryRefreshAndLatestBrowseIntent() {
        LibraryIntentServer server;
        QVERIFY(server.listen());
        server.holdRecycle();
        BackendClient client;
        MainWindow window(&client, nullptr);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 1, 1000);
        auto *list = window.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        auto *confirmation = window.findChild<QLineEdit *>(
            QStringLiteral("recycleNameConfirmationEdit"));
        auto *recycle = window.findChild<QPushButton *>(
            QStringLiteral("deleteWorkpieceButton"));
        QVERIFY(list != nullptr);
        QVERIFY(confirmation != nullptr);
        QVERIFY(recycle != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(list->count(), 2, 1000);
        QVERIFY(window.requestPage(AppPage::WorkpieceLibrary));
        list->setCurrentRow(0);
        QTRY_VERIFY_WITH_TIMEOUT(
            server.detailsWorkpieceIds().contains(QStringLiteral("m1")), 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        confirmation->setText(QStringLiteral("M1"));
        QVERIFY(recycle->isEnabled());

        recycle->click();
        QTRY_COMPARE_WITH_TIMEOUT(server.recycleCount(), 1, 1000);
        list->setCurrentRow(1);
        QVERIFY(server.replyRecycleSuccess());

        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 2, 1500);
        QTRY_VERIFY_WITH_TIMEOUT(
            server.detailsWorkpieceIds().contains(QStringLiteral("m2")), 1500);

        QStringList routed;
        bool afterRecycle = false;
        for (const QJsonObject &request : server.requests()) {
            const QString command = request.value(QStringLiteral("command")).toString();
            if (command == QStringLiteral("recycle_workpiece")) afterRecycle = true;
            if (afterRecycle
                && (command == QStringLiteral("recycle_workpiece")
                    || command == QStringLiteral("list_workpieces")
                    || command == QStringLiteral("get_workpiece_details"))) {
                routed.append(command);
            }
        }
        QCOMPARE(routed, QStringList({QStringLiteral("recycle_workpiece"),
                                     QStringLiteral("list_workpieces"),
                                     QStringLiteral("get_workpiece_details")}));
    }

    void failedMandatoryRefreshWaitsForExplicitRetryAndReloadsBrowsedDetails() {
        LibraryIntentServer server;
        QVERIFY(server.listen());
        server.holdFirstRegister();
        BackendClient client;
        MainWindow window(&client, nullptr);
        window.setReplaceConfirmationHandler([](const QString &) { return true; });
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        window.setWorkpieceName(QStringLiteral("M-refresh-retry"));
        window.setTemplatePaths(
            writeImages(directory, QStringLiteral("retry-front"), 1),
            writeImages(directory, QStringLiteral("retry-back"), 2));
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 1, 1000);
        auto *list = window.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        auto *refresh = window.findChild<QPushButton *>(
            QStringLiteral("refreshWorkpiecesButton"));
        auto *message = window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"));
        QVERIFY(list != nullptr);
        QVERIFY(refresh != nullptr);
        QVERIFY(message != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(list->count(), 2, 1000);
        QVERIFY(window.requestPage(AppPage::WorkpieceLibrary));
        list->setCurrentRow(0);
        QTRY_VERIFY_WITH_TIMEOUT(
            server.detailsWorkpieceIds().contains(QStringLiteral("m1")), 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        const int detailsBeforeRegistration = server.detailsWorkpieceIds().size();

        server.failNextListRefresh();
        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration",
                                          Qt::DirectConnection));
        QTRY_COMPARE_WITH_TIMEOUT(server.registerCount(), 1, 1000);
        QVERIFY(server.replyFirstRegisterExists());
        QTRY_COMPARE_WITH_TIMEOUT(server.registerCount(), 2, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 2, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(message->text().contains(QStringLiteral("刷新失败")), 1000);
        QVERIFY(message->text().contains(QStringLiteral("刷新工件列表")));
        QVERIFY(refresh->isEnabled());
        QTest::qWait(100);
        QCOMPARE(server.listCount(), 2);

        refresh->click();
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 3, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(
            server.detailsWorkpieceIds().size() > detailsBeforeRegistration, 1000);
        QCOMPARE(server.detailsWorkpieceIds().last(), QStringLiteral("m1"));

        QStringList routed;
        bool afterRegister = false;
        for (const QJsonObject &request : server.requests()) {
            const QString command = request.value(QStringLiteral("command")).toString();
            if (command == QStringLiteral("register")) afterRegister = true;
            if (afterRegister
                && (command == QStringLiteral("register")
                    || command == QStringLiteral("list_workpieces")
                    || command == QStringLiteral("get_workpiece_details"))) {
                routed.append(command);
            }
        }
        QCOMPARE(routed, QStringList({QStringLiteral("register"),
                                     QStringLiteral("register"),
                                     QStringLiteral("list_workpieces"),
                                     QStringLiteral("list_workpieces"),
                                     QStringLiteral("get_workpiece_details")}));
    }

    void failedMandatoryDetailsWaitsForOneExplicitListAndDetailsRetry() {
        LibraryIntentServer server;
        QVERIFY(server.listen());
        BackendClient client;
        MainWindow window(&client, nullptr);
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        window.setWorkpieceName(QStringLiteral("M-details-retry"));
        window.setTemplatePaths(
            writeImages(directory, QStringLiteral("details-front"), 1),
            writeImages(directory, QStringLiteral("details-back"), 2));
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 1, 1000);
        auto *list = window.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        auto *refresh = window.findChild<QPushButton *>(
            QStringLiteral("refreshWorkpiecesButton"));
        auto *message = window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"));
        auto *summary = window.findChild<QLabel *>(
            QStringLiteral("workpieceDetailsSummaryLabel"));
        QVERIFY(list != nullptr);
        QVERIFY(refresh != nullptr);
        QVERIFY(message != nullptr);
        QVERIFY(summary != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(list->count(), 2, 1000);
        QVERIFY(window.requestPage(AppPage::WorkpieceLibrary));
        list->setCurrentRow(0);
        QTRY_COMPARE_WITH_TIMEOUT(server.detailsWorkpieceIds().size(), 1, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(summary->text().contains(QStringLiteral("正面 7 张")), 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);

        server.failNextDetailsRefresh();
        QVERIFY(QMetaObject::invokeMethod(&window, "submitRegistration",
                                          Qt::DirectConnection));
        QTRY_COMPARE_WITH_TIMEOUT(server.registerCount(), 1, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 2, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.detailsWorkpieceIds().size(), 2, 1000);
        QCOMPARE(server.detailsWorkpieceIds().last(), QStringLiteral("m1"));
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(message->text().contains(QStringLiteral("刷新失败")), 1000);
        QVERIFY(message->text().contains(QStringLiteral("刷新工件列表")));

        const int listsAfterFailure = server.listCount();
        const int detailsAfterFailure = server.detailsWorkpieceIds().size();
        emit client.stateChanged(BackendClient::State::Ready, QStringLiteral("still ready"));
        emit client.stateChanged(BackendClient::State::Ready, QStringLiteral("still ready"));
        QTest::qWait(100);
        QCOMPARE(server.listCount(), listsAfterFailure);
        QCOMPARE(server.detailsWorkpieceIds().size(), detailsAfterFailure);

        refresh->click();
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), listsAfterFailure + 1, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.detailsWorkpieceIds().size(),
                                  detailsAfterFailure + 1, 1000);
        QCOMPARE(server.detailsWorkpieceIds().last(), QStringLiteral("m1"));
        QTRY_VERIFY_WITH_TIMEOUT(summary->text().contains(QStringLiteral("正面 9 张")), 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);

        QStringList routed;
        bool afterRegister = false;
        for (const QJsonObject &request : server.requests()) {
            const QString command = request.value(QStringLiteral("command")).toString();
            if (command == QStringLiteral("register")) afterRegister = true;
            if (afterRegister
                && (command == QStringLiteral("register")
                    || command == QStringLiteral("list_workpieces")
                    || command == QStringLiteral("get_workpiece_details"))) {
                routed.append(command);
            }
        }
        QCOMPARE(routed, QStringList({QStringLiteral("register"),
                                     QStringLiteral("list_workpieces"),
                                     QStringLiteral("get_workpiece_details"),
                                     QStringLiteral("list_workpieces"),
                                     QStringLiteral("get_workpiece_details")}));
        const int finalListCount = server.listCount();
        const int finalDetailsCount = server.detailsWorkpieceIds().size();
        emit client.stateChanged(BackendClient::State::Ready, QStringLiteral("still ready"));
        QTest::qWait(100);
        QCOMPARE(server.listCount(), finalListCount);
        QCOMPARE(server.detailsWorkpieceIds().size(), finalDetailsCount);
    }

    void recycleRefreshTransportFailureRecoversOnceAfterReconnect() {
        LibraryIntentServer server;
        QVERIFY(server.listen());
        BackendClient client;
        MainWindow window(&client, nullptr);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 1, 1000);
        auto *list = window.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        auto *confirmation = window.findChild<QLineEdit *>(
            QStringLiteral("recycleNameConfirmationEdit"));
        auto *recycle = window.findChild<QPushButton *>(
            QStringLiteral("deleteWorkpieceButton"));
        QVERIFY(list != nullptr);
        QVERIFY(confirmation != nullptr);
        QVERIFY(recycle != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(list->count(), 2, 1000);
        QVERIFY(window.requestPage(AppPage::WorkpieceLibrary));
        list->setCurrentRow(0);
        QTRY_VERIFY_WITH_TIMEOUT(
            server.detailsWorkpieceIds().contains(QStringLiteral("m1")), 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        confirmation->setText(QStringLiteral("M1"));
        QVERIFY(recycle->isEnabled());

        server.disconnectNextListRefresh();
        recycle->click();
        QTRY_COMPARE_WITH_TIMEOUT(server.recycleCount(), 1, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 2, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Disconnected
                                     || client.state() == BackendClient::State::Error,
                                 1000);

        const int detailsBeforeReconnect = server.detailsWorkpieceIds().size();
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listCount(), 3, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(
            server.detailsWorkpieceIds().size() > detailsBeforeReconnect, 1000);
        QTest::qWait(100);
        QCOMPARE(server.listCount(), 3);
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
                                  refreshCountBeforeReconnect + 1, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(
            window.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))
                ->text().contains(QStringLiteral("工件列表已刷新")),
            1000);
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

    void mandatoryRefreshWaitsUntilBatchPredictionSequenceCompletes() {
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
        const int refreshCountBefore = server.listWorkpieceCount();

        QVERIFY(QMetaObject::invokeMethod(
            &window, "onClientResponse", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("recycle_workpiece")),
            Q_ARG(QJsonObject, QJsonObject())));
        server.replyNextPrediction();

        QTRY_COMPARE_WITH_TIMEOUT(server.predictionCount(), 2, 1000);
        QCOMPARE(server.listWorkpieceCount(), refreshCountBefore);
        server.replyNextPrediction();
        QTRY_COMPARE_WITH_TIMEOUT(server.predictionCount(), 3, 1000);
        QCOMPARE(server.listWorkpieceCount(), refreshCountBefore);
        server.replyNextPrediction();
        QTRY_COMPARE_WITH_TIMEOUT(inspectionPage->completedBatchCount(), 3, 1000);
        QTRY_COMPARE_WITH_TIMEOUT(server.listWorkpieceCount(), refreshCountBefore + 1, 1000);

        QStringList routed;
        bool inBatch = false;
        for (const QJsonObject &request : server.requests()) {
            const QString command = request.value(QStringLiteral("command")).toString();
            if (command == QStringLiteral("predict")) inBatch = true;
            if (inBatch && (command == QStringLiteral("predict")
                            || command == QStringLiteral("list_workpieces"))) {
                routed.append(command);
            }
        }
        QCOMPARE(routed, QStringList({QStringLiteral("predict"),
                                     QStringLiteral("predict"),
                                     QStringLiteral("predict"),
                                     QStringLiteral("list_workpieces")}));
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

    void hiddenGeometryPageKeepsValidationPolling() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        QTimer *timer = geometryValidationTimer(window);
        QVERIFY(timer != nullptr);
        QVERIFY(window.requestPage(AppPage::GeometryRules));
        QVERIFY(deliverValidationJob(
            window, QStringLiteral("validate_geometry_mask_draft"),
            validationJob(QStringLiteral("running"))));
        QVERIFY(timer->isActive());

        QVERIFY(window.requestPage(AppPage::Inspection));

        QVERIFY(timer->isActive());
    }

    void reconnectResumesTheSameGeometryValidationJob() {
        GeometryWorkflowServer server;
        QVERIFY(server.listen());
        BackendClient client;
        MainWindow window(&client, nullptr);
        client.connectToService(QHostAddress::LocalHost, server.port(), 500);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTimer *timer = geometryValidationTimer(window);
        QVERIFY(timer != nullptr);
        QVERIFY(deliverValidationJob(
            window, QStringLiteral("validate_geometry_mask_draft"),
            validationJob(QStringLiteral("running"))));
        QVERIFY(timer->isActive());

        QVERIFY(QMetaObject::invokeMethod(
            &window, "onBackendUnavailable", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("lost"))));
        QVERIFY(!timer->isActive());
        QVERIFY(QMetaObject::invokeMethod(&window, "onBackendReady", Qt::DirectConnection));

        QVERIFY(timer->isActive());
        QTRY_VERIFY_WITH_TIMEOUT(
            server.validationPollJobIds().contains(QStringLiteral("job-lifecycle")), 1500);
    }

    void terminalAndCancelledGeometryValidationStopPolling() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        QTimer *timer = geometryValidationTimer(window);
        QVERIFY(timer != nullptr);
        QVERIFY(deliverValidationJob(
            window, QStringLiteral("validate_geometry_mask_draft"),
            validationJob(QStringLiteral("running"))));
        QVERIFY(timer->isActive());
        QVERIFY(deliverValidationJob(
            window, QStringLiteral("get_geometry_mask_validation_job"),
            validationJob(QStringLiteral("completed"), 5, 5)));
        QVERIFY(!timer->isActive());

        QVERIFY(deliverValidationJob(
            window, QStringLiteral("validate_geometry_mask_draft"),
            validationJob(QStringLiteral("running"))));
        QVERIFY(timer->isActive());
        QVERIFY(deliverValidationJob(
            window, QStringLiteral("geometry_mask_validation_job_action"),
            validationJob(QStringLiteral("cancelled"))));
        QVERIFY(!timer->isActive());
    }

    void geometryValidationProgressUsesGlobalTaskStatus() {
        BackendClient client;
        MainWindow window(&client, nullptr);
        QVERIFY(deliverValidationJob(
            window, QStringLiteral("validate_geometry_mask_draft"),
            validationJob(QStringLiteral("running"), 2, 5)));

        auto *title = window.findChild<QLabel *>(QStringLiteral("globalTaskTitleLabel"));
        auto *detail = window.findChild<QLabel *>(QStringLiteral("globalTaskDetailLabel"));
        auto *progress = window.findChild<QProgressBar *>(
            QStringLiteral("globalTaskProgressBar"));
        QVERIFY(title != nullptr);
        QVERIFY(detail != nullptr);
        QVERIFY(progress != nullptr);
        QVERIFY(title->text().contains(QStringLiteral("几何规则验证")));
        QVERIFY(detail->text().contains(QStringLiteral("2/5")));
        QCOMPARE(progress->maximum(), 5);
        QCOMPARE(progress->value(), 2);
    }

    void idleGeometryPageDoesNotCountAsPreservedDisconnectWork() {
        BackendClient idleClient;
        MainWindow idleWindow(&idleClient, nullptr);
        QVERIFY(QMetaObject::invokeMethod(
            &idleWindow, "onBackendUnavailable", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("lost"))));
        auto *idleMessage = idleWindow.findChild<QLabel *>(
            QStringLiteral("libraryMessageLabel"));
        QVERIFY(idleMessage != nullptr);
        QVERIFY(!idleMessage->text().contains(QStringLiteral("未完成操作结果未知")));

        BackendClient dirtyClient;
        MainWindow dirtyWindow(&dirtyClient, nullptr);
        auto *dirtyPage = dirtyWindow.findChild<GeometryRulesPage *>(
            QStringLiteral("geometryRulesPage"));
        QVERIFY(dirtyPage != nullptr);
        dirtyPage->setSnapshot(GeometryWorkflowServer::profile(1, 1, 0));
        dirtyPage->findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
        QVERIFY(dirtyPage->hasUnsavedChanges());
        QVERIFY(QMetaObject::invokeMethod(
            &dirtyWindow, "onBackendUnavailable", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("lost"))));
        QVERIFY(dirtyWindow.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))
                    ->text().contains(QStringLiteral("未完成操作结果未知")));

        BackendClient activeClient;
        MainWindow activeWindow(&activeClient, nullptr);
        QVERIFY(deliverValidationJob(
            activeWindow, QStringLiteral("validate_geometry_mask_draft"),
            validationJob(QStringLiteral("running"))));
        QVERIFY(QMetaObject::invokeMethod(
            &activeWindow, "onBackendUnavailable", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("lost"))));
        QVERIFY(activeWindow.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))
                    ->text().contains(QStringLiteral("未完成操作结果未知")));
    }

    void staleGeometryProfileCannotOverwriteAcceptedTarget() {
        GeometryWorkflowServer server;
        QVERIFY(server.listen());
        server.holdGeometryProfiles();
        BackendClient client;
        MainWindow window(&client, nullptr);
        client.connectToService(QHostAddress::LocalHost, server.port(), 5000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        auto *combo = window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
        auto *page = window.findChild<GeometryRulesPage *>(
            QStringLiteral("geometryRulesPage"));
        QVERIFY(combo != nullptr);
        QVERIFY(page != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(combo->currentData().toString(), QStringLiteral("m1"), 1000);
        QVERIFY(window.requestPage(AppPage::GeometryRules));
        QTRY_COMPARE_WITH_TIMEOUT(server.geometryProfileWorkpieceIds().size(), 1, 1000);
        combo->setCurrentIndex(combo->findData(QStringLiteral("m2")));
        page->setSnapshot(GeometryWorkflowServer::profile(2, 2, 0, QStringLiteral("m2")));

        QVERIFY(server.replyNextGeometryProfile());
        QTRY_COMPARE_WITH_TIMEOUT(server.geometryProfileWorkpieceIds().size(), 2, 1000);

        QCOMPARE(combo->currentData().toString(), QStringLiteral("m2"));
        QCOMPARE(page->snapshot().value(QStringLiteral("workpiece_id")).toString(),
                 QStringLiteral("m2"));
    }

    void rapidGeometryTargetRoundTripRefetchesCurrentGeneration() {
        GeometryWorkflowServer server;
        QVERIFY(server.listen());
        server.holdGeometryProfiles();
        BackendClient client;
        MainWindow window(&client, nullptr);
        client.connectToService(QHostAddress::LocalHost, server.port(), 5000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        auto *combo = window.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
        QVERIFY(combo != nullptr);
        QTRY_COMPARE_WITH_TIMEOUT(combo->currentData().toString(), QStringLiteral("m1"), 1000);
        QVERIFY(window.requestPage(AppPage::GeometryRules));
        QTRY_COMPARE_WITH_TIMEOUT(server.geometryProfileWorkpieceIds().size(), 1, 1000);

        combo->setCurrentIndex(combo->findData(QStringLiteral("m2")));
        combo->setCurrentIndex(combo->findData(QStringLiteral("m1")));
        QCOMPARE(combo->currentData().toString(), QStringLiteral("m1"));

        QVERIFY(server.replyNextGeometryProfile());
        QTRY_COMPARE_WITH_TIMEOUT(server.geometryProfileWorkpieceIds().size(), 2, 1000);
        QVERIFY(server.replyNextGeometryProfile());

        QTRY_COMPARE_WITH_TIMEOUT(server.geometryProfileWorkpieceIds().size(), 3, 1000);
        QCOMPARE(server.geometryProfileWorkpieceIds(),
                 QStringList({QStringLiteral("m1"), QStringLiteral("m2"),
                              QStringLiteral("m1")}));
    }

    void reconnectRetriesUnloadedGeometryProfileForCurrentTargetOnce() {
        GeometryWorkflowServer server;
        QVERIFY(server.listen());
        server.holdGeometryProfiles();
        BackendClient client;
        MainWindow window(&client, nullptr);
        client.connectToService(QHostAddress::LocalHost, server.port(), 5000);
        QTRY_VERIFY_WITH_TIMEOUT(client.state() == BackendClient::State::Ready, 1000);
        QTRY_VERIFY_WITH_TIMEOUT(window.findChild<QComboBox *>(
            QStringLiteral("workpieceComboBox"))->currentData().toString()
            == QStringLiteral("m1"), 1000);
        QVERIFY(window.requestPage(AppPage::GeometryRules));
        QTRY_COMPARE_WITH_TIMEOUT(server.geometryProfileWorkpieceIds().size(), 1, 1000);

        QVERIFY(QMetaObject::invokeMethod(
            &window, "onBackendUnavailable", Qt::DirectConnection,
            Q_ARG(QString, QStringLiteral("lost"))));
        client.disconnectFromService();
        client.connectToService(QHostAddress::LocalHost, server.port(), 5000);
        QTRY_VERIFY_WITH_TIMEOUT(
            client.state() == BackendClient::State::Ready
                || client.state() == BackendClient::State::Busy,
            1000);

        QTRY_COMPARE_WITH_TIMEOUT(server.geometryProfileWorkpieceIds().size(), 2, 1500);
        QCOMPARE(server.geometryProfileWorkpieceIds(),
                 QStringList({QStringLiteral("m1"), QStringLiteral("m1")}));
    }

    void listRefreshRemovingCurrentTargetHonorsGeometryDirtyChoice() {
        const QJsonObject initial{{QStringLiteral("workpieces"), QJsonArray{
            QJsonObject{{QStringLiteral("id"), QStringLiteral("m1")},
                        {QStringLiteral("name"), QStringLiteral("M1")}},
            QJsonObject{{QStringLiteral("id"), QStringLiteral("m2")},
                        {QStringLiteral("name"), QStringLiteral("M2")}}}}};
        const QJsonObject removed{{QStringLiteral("workpieces"), QJsonArray{
            QJsonObject{{QStringLiteral("id"), QStringLiteral("m2")},
                        {QStringLiteral("name"), QStringLiteral("M2")}}}}};

        BackendClient cancelClient;
        MainWindow cancelWindow(&cancelClient, nullptr);
        emit cancelClient.responseReceived(QStringLiteral("list_workpieces"), initial);
        auto *cancelCombo = cancelWindow.findChild<QComboBox *>(
            QStringLiteral("workpieceComboBox"));
        auto *cancelPage = cancelWindow.findChild<GeometryRulesPage *>(
            QStringLiteral("geometryRulesPage"));
        QVERIFY(cancelCombo != nullptr);
        QVERIFY(cancelPage != nullptr);
        cancelPage->setSnapshot(GeometryWorkflowServer::profile(1, 1, 0));
        QVERIFY(cancelWindow.requestPage(AppPage::GeometryRules));
        cancelPage->findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
        chooseDirtyNavigationOption(QStringLiteral("取消"));
        emit cancelClient.responseReceived(QStringLiteral("list_workpieces"), removed);
        QCOMPARE(cancelCombo->currentData().toString(), QStringLiteral("m1"));
        QVERIFY(cancelPage->hasUnsavedChanges());

        BackendClient discardClient;
        MainWindow discardWindow(&discardClient, nullptr);
        emit discardClient.responseReceived(QStringLiteral("list_workpieces"), initial);
        auto *discardCombo = discardWindow.findChild<QComboBox *>(
            QStringLiteral("workpieceComboBox"));
        auto *discardPage = discardWindow.findChild<GeometryRulesPage *>(
            QStringLiteral("geometryRulesPage"));
        QVERIFY(discardCombo != nullptr);
        QVERIFY(discardPage != nullptr);
        discardPage->setSnapshot(GeometryWorkflowServer::profile(1, 1, 0));
        QVERIFY(discardWindow.requestPage(AppPage::GeometryRules));
        discardPage->findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
        chooseDirtyNavigationOption(QStringLiteral("放弃修改"));
        emit discardClient.responseReceived(QStringLiteral("list_workpieces"), removed);
        QCOMPARE(discardCombo->currentData().toString(), QStringLiteral("m2"));
        QVERIFY(!discardPage->hasUnsavedChanges());
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
