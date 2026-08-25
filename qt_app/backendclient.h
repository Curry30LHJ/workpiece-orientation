#pragma once

#include <QHostAddress>
#include <QJsonObject>
#include <QObject>
#include <QString>

#include <memory>

class QJsonObject;
class QTcpSocket;
class QTimer;

class BackendClient : public QObject {
    Q_OBJECT

public:
    enum class State { Disconnected, Connecting, Handshaking, Ready, Busy, Error };
    Q_ENUM(State)

    explicit BackendClient(QObject *parent = nullptr);
    ~BackendClient() override;

    State state() const;
    void connectToService(const QHostAddress &host, quint16 port, int requestTimeoutMs = 120000);
    QString sendRequest(const QString &command, const QJsonObject &fields = QJsonObject());
    void disconnectFromService();

signals:
    void stateChanged(BackendClient::State state, const QString &detail);
    void handshakeSucceeded();
    void progressReceived(const QString &command, const QJsonObject &progress);
    void responseReceived(const QString &command, const QJsonObject &response);
    void commandFailed(const QString &command, const QString &code, const QString &message);
    void transportFailed(const QString &code, const QString &message);
    // Compatibility signals for older integrations. New production code uses the
    // command/transport-specific channels above.
    void requestFailed(const QString &code, const QString &message);
    void connectionLost(const QString &reason);

private slots:
    void onConnected();
    void onReadyRead();
    void onDisconnected();
    void onSocketError(QAbstractSocket::SocketError error);
    void onRequestTimeout();

private:
    struct PendingRequest {
        QString id;
        QString command;
    };

    void setState(State state, const QString &detail);
    void sendJson(const QJsonObject &object);
    void handleResponse(const QJsonObject &response);
    void emitCommandFailure(const QString &command, const QString &code, const QString &message);
    void failTransport(const QString &code, const QString &message, bool closeSocket = true);
    void clearPending();

    QTcpSocket *socket_;
    QTimer *requestTimer_;
    State state_ = State::Disconnected;
    QByteArray readBuffer_;
    QString handshakeRequestId_;
    std::unique_ptr<PendingRequest> pending_;
    bool suppressConnectionLost_ = false;
    bool transportFailureReported_ = false;
    int requestTimeoutMs_ = 120000;
};
