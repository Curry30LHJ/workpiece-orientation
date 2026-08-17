#pragma once

#include <QMainWindow>
#include <QElapsedTimer>
#include <QJsonObject>
#include <QPointer>
#include <QJsonArray>
#include <QStringList>
#include <QTimer>

#include <functional>

#include "backendclient.h"

class BackendProcessManager;
class QPushButton;
class AnnotationManagerDialog;
class GeometryMaskManagerDialog;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    MainWindow(BackendClient *client, BackendProcessManager *manager, QWidget *parent = nullptr);
    ~MainWindow() override;

    void setTemplatePaths(const QStringList &frontPaths, const QStringList &backPaths);
    void setWorkpieceName(const QString &name);
    void setInspectionImagePath(const QString &path);
    void setBatchImagePaths(const QStringList &paths);
    void setReplaceConfirmationHandler(std::function<bool(const QString &)> handler);
    void setBackendError(const QString &message);

private slots:
    void chooseFrontTemplates();
    void chooseBackTemplates();
    void chooseInspectionImage();
    void chooseBatchImages();
    void refreshWorkpieces();
    void submitRegistration();
    void submitPrediction();
    void submitBatchPrediction();
    void deleteSelectedWorkpiece();
    void confirmFrontTemplate();
    void confirmBackTemplate();
    void rejectTemplateConfirmation();
    void openAnnotationEditor();
    void openAnnotationManager();
    void openGeometryMaskManager();
    void saveAnnotationGroups(const QJsonArray &groups, int baseRevision);
    void setAnnotationGroupEnabled(const QString &groupId, bool enabled, int baseRevision);
    void deleteAnnotationGroup(const QString &groupId, int baseRevision);
    void reviewAnnotation(const QString &groupId, const QString &templateId,
                          const QString &action, const QJsonArray &regions, int baseRevision);
    void restartBackend();
    void pollEvolutionJobs();
    void pollGeometryValidation();
    void saveGeometryDraft(const QJsonObject &draft, int libraryRevision, int draftRevision);
    void validateGeometryDraft(int libraryRevision, int draftRevision);
    void geometryJobAction(const QString &jobId, const QString &action);
    void publishGeometryProfile(const QString &jobId, int libraryRevision, int draftRevision,
                                const QString &overrideReason);
    void rollbackGeometryProfile(int libraryRevision);

    void onBackendReady();
    void onBackendLoading(const QString &message);
    void onBackendUnavailable(const QString &reason);
    void onClientStateChanged(BackendClient::State state, const QString &detail);
    void onClientProgress(const QString &command, const QJsonObject &progress);
    void onClientResponse(const QString &command, const QJsonObject &response);
    void onClientRequestFailed(const QString &code, const QString &message);
    void onConnectionLost(const QString &reason);

private:
    void initializeUi();
    void connectBackendSignals();
    void updateButtonStates();
    void updateTemplateLabels();
    void updatePreview();
    void clearInspectionState();
    void clearBatchState();
    void clearBatchResults();
    void sendRegistration(bool replace);
    void requestWorkpieceRefresh(bool preserveRegistrationSummary);
    void sendNextBatchPrediction();
    void appendBatchResult(const QJsonObject &response);
    void finishBatchPrediction();
    void submitTemplateConfirmation(const QString &imagePath, const QString &orientation);
    void startRegistrationProgress();
    void stopRegistrationProgress();
    void updateRegistrationElapsed();
    bool validateRegistration(QString *error) const;
    bool validateImagePath(const QString &path, QString *error) const;
    QStringList normalizedPaths(const QStringList &paths) const;
    QString selectedWorkpieceId() const;
    QString orientationText(const QString &label) const;
    QString decisionSourceText(const QString &source) const;
    QString formatScore(const QJsonObject &scores, const QString &key) const;
    void showLibraryMessage(const QString &message, bool error = false);
    void requestAnnotationSnapshot(const QString &workpieceId);
    void sendAnnotationMutation(const QString &command, const QJsonObject &fields);
    void requestGeometryProfile(const QString &workpieceId);

    Ui::MainWindow *ui;
    BackendClient *client_ = nullptr;
    BackendProcessManager *manager_ = nullptr;
    QStringList frontTemplatePaths_;
    QStringList backTemplatePaths_;
    QString inspectionImagePath_;
    QStringList batchImagePaths_;
    int batchIndex_ = 0;
    QString pendingCommand_;
    QString pendingWorkpieceName_;
    bool pendingReplace_ = false;
    bool registrationInFlight_ = false;
    bool registrationSummaryVisible_ = false;
    bool batchInFlight_ = false;
    int batchFrontCount_ = 0;
    int batchBackCount_ = 0;
    int batchUncertainCount_ = 0;
    int batchReviewCount_ = 0;
    bool backendReady_ = false;
    bool clientBusy_ = false;
    QElapsedTimer registrationElapsedClock_;
    QTimer *registrationElapsedTimer_ = nullptr;
    QTimer *evolutionPollTimer_ = nullptr;
    QTimer *geometryPollTimer_ = nullptr;
    std::function<bool(const QString &)> replaceConfirmationHandler_;
    QPushButton *deleteWorkpieceButton_ = nullptr;
    QPushButton *confirmFrontButton_ = nullptr;
    QPushButton *confirmBackButton_ = nullptr;
    QPushButton *rejectConfirmationButton_ = nullptr;
    QPushButton *annotationEditorButton_ = nullptr;
    QString lastPredictionWorkpieceId_;
    QString lastPredictionImagePath_;
    QJsonObject lastPredictionResponse_;
    QString pendingPredictionWorkpieceId_;
    QString pendingPredictionImagePath_;
    QString batchWorkpieceId_;
    QPointer<AnnotationManagerDialog> annotationManagerDialog_;
    QString annotationWorkpieceId_;
    QPointer<GeometryMaskManagerDialog> geometryMaskManagerDialog_;
    QString geometryWorkpieceId_;
    QString geometryValidationJobId_;
};
