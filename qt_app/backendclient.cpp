#include "backendclient.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QTcpSocket>
#include <QTimer>
#include <QUuid>

namespace {
constexpr int kProtocolVersion = 1;
constexpr int kMaxMessageBytes = 1024 * 1024;
const QString kServiceName = QStringLiteral("workpiece-orientation");
}

BackendClient::BackendClient(QObject *parent)
    : QObject(parent), socket_(new QTcpSocket(this)), requestTimer_(new QTimer(this)) {
    requestTimer_->setSingleShot(true);
    connect(socket_, &QTcpSocket::connected, this, &BackendClient::onConnected);
    connect(socket_, &QTcpSocket::readyRead, this, &BackendClient::onReadyRead);
    connect(socket_, &QTcpSocket::disconnected, this, &BackendClient::onDisconnected);
    connect(socket_, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(onSocketError(QAbstractSocket::SocketError)));
    connect(requestTimer_, &QTimer::timeout, this, &BackendClient::onRequestTimeout);
}

BackendClient::~BackendClient() {
    socket_->abort();
}

BackendClient::State BackendClient::state() const {
    return state_;
}

void BackendClient::connectToService(const QHostAddress &host, quint16 port, int requestTimeoutMs) {
    suppressConnectionLost_ = true;
    socket_->abort();
    readBuffer_.clear();
    handshakeRequestId_.clear();
    clearPending();
    suppressConnectionLost_ = false;
    transportFailureReported_ = false;
    requestTimeoutMs_ = requestTimeoutMs > 0 ? requestTimeoutMs : 120000;
    setState(State::Connecting, QStringLiteral("正在连接后端"));
    socket_->connectToHost(host, port);
}

QString BackendClient::sendRequest(const QString &command, const QJsonObject &fields) {
    if (pending_ != nullptr || state_ == State::Busy) {
        emitCommandFailure(command, QStringLiteral("BUSY"), QStringLiteral("已有请求正在执行"));
        return QString();
    }
    if (state_ != State::Ready) {
        emitCommandFailure(command, QStringLiteral("NOT_READY"), QStringLiteral("后端尚未就绪"));
        return QString();
    }
    const QString requestId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    QJsonObject request = fields;
    request.insert(QStringLiteral("version"), kProtocolVersion);
    request.insert(QStringLiteral("request_id"), requestId);
    request.insert(QStringLiteral("command"), command);
    pending_ = std::make_unique<PendingRequest>(PendingRequest{requestId, command});
    setState(State::Busy, QStringLiteral("正在处理 %1").arg(command));
    sendJson(request);
    requestTimer_->start(requestTimeoutMs_);
    return requestId;
}

void BackendClient::disconnectFromService() {
    requestTimer_->stop();
    clearPending();
    handshakeRequestId_.clear();
    suppressConnectionLost_ = true;
    socket_->abort();
    setState(State::Disconnected, QStringLiteral("已断开后端"));
}

void BackendClient::onConnected() {
    setState(State::Handshaking, QStringLiteral("正在验证后端"));
    handshakeRequestId_ = QUuid::createUuid().toString(QUuid::WithoutBraces);
    sendJson({
        {QStringLiteral("version"), kProtocolVersion},
        {QStringLiteral("request_id"), handshakeRequestId_},
        {QStringLiteral("command"), QStringLiteral("hello")},
    });
    requestTimer_->start(requestTimeoutMs_);
}

void BackendClient::onReadyRead() {
    readBuffer_ += socket_->readAll();
    if (readBuffer_.size() > kMaxMessageBytes && !readBuffer_.contains('\n')) {
        failTransport(QStringLiteral("MESSAGE_TOO_LARGE"), QStringLiteral("后端响应超过 1 MiB"));
        return;
    }
    while (readBuffer_.contains('\n')) {
        const int newline = readBuffer_.indexOf('\n');
        const QByteArray line = readBuffer_.left(newline);
        readBuffer_.remove(0, newline + 1);
        if (line.size() > kMaxMessageBytes) {
            failTransport(QStringLiteral("MESSAGE_TOO_LARGE"), QStringLiteral("后端响应超过 1 MiB"));
            return;
        }
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            failTransport(QStringLiteral("PROTOCOL_ERROR"), QStringLiteral("后端返回了无效 JSON"));
            return;
        }
        handleResponse(document.object());
        if (state_ == State::Error || state_ == State::Disconnected) {
            return;
        }
    }
}

void BackendClient::onDisconnected() {
    const bool wasConnected = state_ != State::Disconnected;
    const bool reportConnectionLost = !suppressConnectionLost_;
    suppressConnectionLost_ = false;
    requestTimer_->stop();
    clearPending();
    handshakeRequestId_.clear();
    setState(State::Disconnected, QStringLiteral("后端连接已断开"));
    if (wasConnected && reportConnectionLost) {
        if (!transportFailureReported_) {
            transportFailureReported_ = true;
            emit transportFailed(QStringLiteral("CONNECTION_LOST"), QStringLiteral("后端连接已断开"));
        }
        emit connectionLost(QStringLiteral("后端连接已断开"));
    }
}

void BackendClient::onSocketError(QAbstractSocket::SocketError error) {
    Q_UNUSED(error)
    if (state_ == State::Disconnected) {
        return;
    }
    failTransport(QStringLiteral("CONNECTION_ERROR"), socket_->errorString());
}

void BackendClient::onRequestTimeout() {
    if (!handshakeRequestId_.isEmpty()) {
        failTransport(QStringLiteral("TIMEOUT"), QStringLiteral("后端握手超时"));
    } else if (pending_ != nullptr) {
        failTransport(QStringLiteral("TIMEOUT"), QStringLiteral("后端请求超时"));
    }
}

void BackendClient::setState(State state, const QString &detail) {
    if (state_ == state && detail.isEmpty()) {
        return;
    }
    state_ = state;
    emit stateChanged(state_, detail);
}

void BackendClient::sendJson(const QJsonObject &object) {
    socket_->write(QJsonDocument(object).toJson(QJsonDocument::Compact) + "\n");
    socket_->flush();
}

void BackendClient::handleResponse(const QJsonObject &response) {
    if (response.value(QStringLiteral("version")).toInt(-1) != kProtocolVersion) {
        failTransport(QStringLiteral("PROTOCOL_ERROR"), QStringLiteral("后端协议版本不匹配"));
        return;
    }
    const QString responseId = response.value(QStringLiteral("request_id")).toString();
    if (!handshakeRequestId_.isEmpty()) {
        if (responseId != handshakeRequestId_) {
            failTransport(QStringLiteral("PROTOCOL_ERROR"), QStringLiteral("握手请求标识不匹配"));
            return;
        }
        requestTimer_->stop();
        handshakeRequestId_.clear();
        if (!response.value(QStringLiteral("ok")).toBool()) {
            const QJsonObject error = response.value(QStringLiteral("error")).toObject();
            const QString code = error.value(QStringLiteral("code")).toString(QStringLiteral("HANDSHAKE_FAILED"));
            failTransport(code, error.value(QStringLiteral("message")).toString(QStringLiteral("后端身份验证失败")));
            return;
        }
        if (response.value(QStringLiteral("service")).toString() != kServiceName) {
            failTransport(QStringLiteral("HANDSHAKE_FAILED"), QStringLiteral("后端身份验证失败"));
            return;
        }
        if (response.value(QStringLiteral("ok")).toBool()
            && !response.value(QStringLiteral("ready")).toBool()
            && response.value(QStringLiteral("status")).toString() == QStringLiteral("loading")) {
            failTransport(QStringLiteral("MODEL_LOADING"),
                          response.value(QStringLiteral("message")).toString(QStringLiteral("模型加载中")));
            return;
        }
        if (!response.value(QStringLiteral("ready")).toBool()) {
            failTransport(QStringLiteral("HANDSHAKE_FAILED"), QStringLiteral("后端身份验证失败"));
            return;
        }
        setState(State::Ready, QStringLiteral("后端已就绪"));
        emit handshakeSucceeded();
        return;
    }
    if (pending_ == nullptr || responseId != pending_->id) {
        failTransport(QStringLiteral("PROTOCOL_ERROR"), QStringLiteral("业务请求标识不匹配"));
        return;
    }
    const QString command = pending_->command;
    if (response.value(QStringLiteral("event")).toString() == QStringLiteral("progress")) {
        if (response.value(QStringLiteral("command")).toString() != command
            || !response.value(QStringLiteral("progress")).isObject()) {
            failTransport(QStringLiteral("PROTOCOL_ERROR"), QStringLiteral("后端进度事件无效"));
            return;
        }
        const QJsonObject progress = response.value(QStringLiteral("progress")).toObject();
        const int total = progress.value(QStringLiteral("total")).toInt(-1);
        const int completed = progress.value(QStringLiteral("completed")).toInt(-1);
        if (progress.value(QStringLiteral("phase")).toString().isEmpty()
            || total <= 0 || completed < 0 || completed > total) {
            failTransport(QStringLiteral("PROTOCOL_ERROR"), QStringLiteral("后端进度数据无效"));
            return;
        }
        emit progressReceived(command, progress);
        requestTimer_->start(requestTimeoutMs_);
        return;
    }
    requestTimer_->stop();
    clearPending();
    if (!response.value(QStringLiteral("ok")).toBool()) {
        const QJsonObject error = response.value(QStringLiteral("error")).toObject();
        emitCommandFailure(command,
                           error.value(QStringLiteral("code")).toString(QStringLiteral("INTERNAL_ERROR")),
                           error.value(QStringLiteral("message")).toString(QStringLiteral("后端请求失败")));
    } else {
        emit responseReceived(command, response);
    }
    setState(State::Ready, QStringLiteral("后端已就绪"));
}

void BackendClient::emitCommandFailure(const QString &command, const QString &code,
                                       const QString &message) {
    emit commandFailed(command, code, message);
    emit requestFailed(code, message);
}

void BackendClient::failTransport(const QString &code, const QString &message, bool closeSocket) {
    if (transportFailureReported_) {
        return;
    }
    transportFailureReported_ = true;
    requestTimer_->stop();
    clearPending();
    handshakeRequestId_.clear();
    setState(State::Error, message);
    emit transportFailed(code, message);
    emit requestFailed(code, message);
    if (closeSocket) {
        suppressConnectionLost_ = code == QStringLiteral("MODEL_LOADING");
        socket_->abort();
    }
}

void BackendClient::clearPending() {
    pending_.reset();
}
