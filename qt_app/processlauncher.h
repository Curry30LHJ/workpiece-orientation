#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

class QProcess;

class ProcessLauncher : public QObject {
    Q_OBJECT

public:
    explicit ProcessLauncher(QObject *parent = nullptr) : QObject(parent) {}
    ~ProcessLauncher() override = default;

    virtual bool start(const QString &program, const QStringList &arguments, const QString &workingDirectory) = 0;
    virtual void terminate() = 0;
    virtual void kill() = 0;
    virtual bool isRunning() const = 0;

signals:
    void started();
    void finished(int exitCode);
    void failed(const QString &message);
};

class QProcessLauncher final : public ProcessLauncher {
    Q_OBJECT

public:
    explicit QProcessLauncher(QObject *parent = nullptr);

    bool start(const QString &program, const QStringList &arguments, const QString &workingDirectory) override;
    void terminate() override;
    void kill() override;
    bool isRunning() const override;

private:
    QProcess *process_;
};
