#pragma once

#include <QElapsedTimer>
#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>
#include <QTimer>
#include <QWidget>

#include <functional>

#include "appheader.h"

class QImage;

namespace Ui {
class WorkpieceLibraryPage;
}

class WorkpieceLibraryPage : public QWidget {
    Q_OBJECT

public:
    explicit WorkpieceLibraryPage(QWidget *parent = nullptr);
    ~WorkpieceLibraryPage() override;

    void setWorkpieces(const QJsonArray &workpieces,
                       const QString &currentDetectionWorkpieceId);
    void setWorkpieceDetails(const QJsonObject &details);
    void setEvolutionJobs(const QJsonArray &jobs);
    void setBackendState(BackendUiState state, const QString &detail);
    void setRegistrationProgress(const QJsonObject &progress, qint64 elapsedMs);
    void setRegistrationResult(const QJsonObject &response);
    void handleBackendResponse(const QString &command,
                               const QJsonObject &response);
    void handleBackendFailure(const QString &command, const QString &code,
                              const QString &message);
    void setOperationError(const QString &code, const QString &message);
    void setTemplatePaths(const QStringList &front, const QStringList &back);
    void setWorkpieceName(const QString &name);
    void setReplaceConfirmationHandler(std::function<bool(const QString &)> handler);
    bool hasUnsavedChanges() const;
    void discardEditingDraft();
    QString browsedWorkpieceId() const;

signals:
    void commandRequested(const QString &command, const QJsonObject &fields);
    void setCurrentWorkpieceRequested(const QString &workpieceId);
    void dirtyChanged(bool dirty);
    void taskStatusChanged(const QString &title, const QString &phase,
                           int completed, int total, qint64 elapsedMs);

private slots:
    void chooseFrontTemplates();
    void chooseBackTemplates();
    void submitRegistration();
    void refreshWorkpieces();
    void filterWorkpieces();
    void browseSelectedWorkpiece();
    void activateBrowsedWorkpiece();
    void recycleBrowsedWorkpiece();
    void updateRegistrationElapsed();

private:
    QStringList normalizedPaths(const QStringList &paths) const;
    bool loadValidImage(const QString &path, QImage *image) const;
    bool validateRegistration(QString *error) const;
    QJsonObject registrationFields(bool replace) const;
    bool draftMatchesSavedRegistration() const;
    void updateTemplateUi();
    void updateControlStates();
    void setDirty(bool dirty);
    void showMessage(const QString &message, bool error = false);
    void rebuildWorkpieceList();
    void rebuildEvolutionTable();
    QString browsedDisplayName() const;
    void startRegistrationProgress();
    void stopRegistrationProgress();
    void publishRegistrationTaskStatus(const QString &phase, int completed,
                                       int total, qint64 elapsedMs);
    void publishRegistrationFailure();

    Ui::WorkpieceLibraryPage *ui;
    QJsonArray workpieces_;
    QJsonObject workpieceDetails_;
    QHash<QString, QJsonObject> evolutionJobsById_;
    QStringList evolutionJobOrder_;
    QString currentDetectionWorkpieceId_;
    QString browsedWorkpieceId_;
    QStringList frontTemplatePaths_;
    QStringList backTemplatePaths_;
    BackendUiState backendState_ = BackendUiState::Disconnected;
    bool dirty_ = false;
    bool registrationInFlight_ = false;
    QJsonObject savedRegistrationFields_;
    std::function<bool(const QString &)> replaceConfirmationHandler_;
    QElapsedTimer registrationElapsedClock_;
    QTimer registrationElapsedTimer_;
    QString registrationTaskPhase_;
    int registrationTaskCompleted_ = 0;
    int registrationTaskTotal_ = 0;
    qint64 registrationTaskElapsedMs_ = -1;
};
