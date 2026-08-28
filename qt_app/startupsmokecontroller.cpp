#include "startupsmokecontroller.h"

#include <QTimer>

#include "backendprocessmanager.h"

StartupSmokeController::StartupSmokeController(BackendProcessManager *manager, int timeoutMs,
                                               QObject *parent)
    : QObject(parent), manager_(manager), timer_(new QTimer(this)) {
    timer_->setSingleShot(true);
    connect(manager_, &BackendProcessManager::backendReady, this, &StartupSmokeController::onReady);
    connect(manager_, &BackendProcessManager::backendUnavailable, this,
            &StartupSmokeController::onUnavailable);
    connect(timer_, &QTimer::timeout, this, &StartupSmokeController::onTimeout);
    timer_->start(qMax(1, timeoutMs));
}

void StartupSmokeController::onReady() { finish(0); }

void StartupSmokeController::onUnavailable(const QString &, const QString &, const QString &, const QString &) {
    finish(2);
}

void StartupSmokeController::onTimeout() { finish(3); }

void StartupSmokeController::finish(int exitCode) {
    if (finished_) return;
    finished_ = true;
    timer_->stop();
    disconnect(manager_, nullptr, this, nullptr);
    emit finished(exitCode);
}
