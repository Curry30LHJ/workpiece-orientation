#include "processlauncher.h"

#include <QProcess>

QProcessLauncher::QProcessLauncher(QObject *parent)
    : ProcessLauncher(parent), process_(new QProcess(this)) {
    connect(process_, &QProcess::started, this, &QProcessLauncher::started);
    connect(process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [this](int exitCode, QProcess::ExitStatus) { emit finished(exitCode); });
    connect(process_, &QProcess::errorOccurred, this,
            [this](QProcess::ProcessError) { emit failed(process_->errorString()); });
}

bool QProcessLauncher::start(const QString &program, const QStringList &arguments, const QString &workingDirectory) {
    if (process_ == nullptr) return false;
    process_->setStandardOutputFile(QProcess::nullDevice());
    process_->setStandardErrorFile(QProcess::nullDevice());
    process_->setProgram(program);
    process_->setArguments(arguments);
    process_->setWorkingDirectory(workingDirectory);
    process_->start();
    return true;
}

void QProcessLauncher::terminate() {
    if (process_ != nullptr) process_->terminate();
}

void QProcessLauncher::kill() {
    if (process_ != nullptr) process_->kill();
}

void QProcessLauncher::release() {
    if (process_ == nullptr) return;
    QProcess *releasedProcess = process_;
    process_ = nullptr;
    QObject::disconnect(releasedProcess, nullptr, this, nullptr);
    releasedProcess->setParent(nullptr);
    if (releasedProcess->state() == QProcess::NotRunning) {
        releasedProcess->deleteLater();
        return;
    }
    connect(releasedProcess,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            releasedProcess, &QObject::deleteLater);
}

bool QProcessLauncher::isRunning() const {
    return process_ != nullptr && process_->state() != QProcess::NotRunning;
}
