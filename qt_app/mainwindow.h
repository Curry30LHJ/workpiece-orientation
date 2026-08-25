#pragma once

#include <QMainWindow>
#include <QElapsedTimer>
#include <QJsonObject>
#include <QPointer>
#include <QJsonArray>
#include <QStringList>
#include <QTimer>

#include <functional>

#include "appheader.h"
#include "backendclient.h"

class BackendProcessManager;
class InspectionPage;
class QLabel;
class QPushButton;
class QTableWidget;
class QTextEdit;
class AnnotationManagerDialog;
class GeometryMaskManagerDialog;
class TaskStatusWidget;

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
    bool requestPage(AppPage page);

private slots:
    void showInspection();
    void showWorkpieceLibrary();
    void showGeometryRules();
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
    void publishGeometryWorkflow(const QJsonObject &draft, int libraryRevision, int draftRevision,
                                 const QString &overrideReason);
    void validateGeometryDraft(int libraryRevision, int draftRevision);
    void geometryJobAction(const QString &jobId, const QString &action);
    void publishGeometryProfile(const QString &jobId, int libraryRevision, int draftRevision,
                                const QString &overrideReason);
    void rollbackGeometryProfile(int libraryRevision);
    void previewGeometryRule(const QJsonObject &request);
    void resolveGeometryMigration(const QString &conflictId, const QJsonObject &resolution,
                                  int libraryRevision, int draftRevision);

    void onBackendReady();
    void onBackendLoading(const QString &message);
    void onBackendUnavailable(const QString &reason);
    void onClientStateChanged(BackendClient::State state, const QString &detail);
    void onClientProgress(const QString &command, const QJsonObject &progress);
    void onClientResponse(const QString &command, const QJsonObject &response);
    void onClientCommandFailed(const QString &command, const QString &code, const QString &message);
    void onClientTransportFailed(const QString &code, const QString &message);

private:
    enum class CommandOwner { None, System, Inspection, Library, Geometry };

    enum class ResultContext {
        None,
        Single,
        Batch,
    };

    void initializeUi();
    void connectBackendSignals();
    void sendPageCommand(CommandOwner owner, const QString &command,
                         const QJsonObject &fields = QJsonObject());
    void dispatchQueuedCommand();
    void clearPendingCommand();
    void updateButtonStates();
    void updateTemplateLabels();
    void updatePreview();
    void clearInspectionState();
    void clearBatchState();
    void clearBatchResults();
    void sendRegistration(bool replace);
    void requestWorkpieceRefresh(bool preserveRegistrationSummary);
    void submitTemplateConfirmation(const QString &workpieceId, const QString &imagePath,
                                    const QString &orientation);
    void startRegistrationProgress();
    void stopRegistrationProgress();
    void updateRegistrationElapsed();
    bool validateRegistration(QString *error) const;
    bool validateImagePath(const QString &path, QString *error) const;
    QStringList normalizedPaths(const QStringList &paths) const;
    QString selectedWorkpieceId() const;
    QString orientationText(const QString &label) const;
    void showLibraryMessage(const QString &message, bool error = false);
    void requestAnnotationSnapshot(const QString &workpieceId);
    void sendAnnotationMutation(const QString &command, const QJsonObject &fields);
    void requestGeometryProfile(const QString &workpieceId);
    bool maybeContinueGeometryPublish(const QJsonObject &job);

    Ui::MainWindow *ui;
    AppHeader *appHeader_ = nullptr;
    TaskStatusWidget *globalTaskStatus_ = nullptr;
    InspectionPage *inspectionPage_ = nullptr;
    AppPage currentPage_ = AppPage::Inspection;
    BackendClient *client_ = nullptr;
    BackendProcessManager *manager_ = nullptr;
    QStringList frontTemplatePaths_;
    QStringList backTemplatePaths_;
    QString inspectionImagePath_;
    QStringList batchImagePaths_;
    CommandOwner pendingOwner_ = CommandOwner::None;
    QString pendingCommand_;
    CommandOwner queuedOwner_ = CommandOwner::None;
    QString queuedCommand_;
    QJsonObject queuedFields_;
    QString pendingWorkpieceName_;
    bool pendingReplace_ = false;
    bool registrationInFlight_ = false;
    bool registrationSummaryVisible_ = false;
    bool batchInFlight_ = false;
    bool backendReady_ = false;
    bool backendReadyHandled_ = false;
    bool clientBusy_ = false;
    QElapsedTimer registrationElapsedClock_;
    QTimer *registrationElapsedTimer_ = nullptr;
    QTimer *evolutionPollTimer_ = nullptr;
    QTimer *geometryPollTimer_ = nullptr;
    std::function<bool(const QString &)> replaceConfirmationHandler_;
    QPushButton *deleteWorkpieceButton_ = nullptr;
    QPushButton *chooseImageButton_ = nullptr;
    QPushButton *predictButton_ = nullptr;
    QPushButton *chooseBatchImagesButton_ = nullptr;
    QPushButton *batchPredictButton_ = nullptr;
    QPushButton *confirmFrontButton_ = nullptr;
    QPushButton *confirmBackButton_ = nullptr;
    QPushButton *rejectConfirmationButton_ = nullptr;
    QPushButton *annotationEditorButton_ = nullptr;
    QLabel *currentImageLabel_ = nullptr;
    QLabel *resultLabel_ = nullptr;
    QLabel *reviewLabel_ = nullptr;
    QLabel *batchSummaryLabel_ = nullptr;
    QLabel *currentResultTargetLabel_ = nullptr;
    QTextEdit *evidenceTextEdit_ = nullptr;
    QTableWidget *batchResultsTableWidget_ = nullptr;
    QString lastPredictionWorkpieceId_;
    QString lastPredictionImagePath_;
    QJsonObject lastPredictionResponse_;
    QString pendingPredictionWorkpieceId_;
    QString pendingPredictionImagePath_;
    QString pendingConfirmationRecordId_;
    QString pendingConfirmationOrientation_;
    ResultContext resultContext_ = ResultContext::None;
    QPointer<AnnotationManagerDialog> annotationManagerDialog_;
    QString annotationWorkpieceId_;
    QPointer<GeometryMaskManagerDialog> geometryMaskManagerDialog_;
    QString geometryWorkpieceId_;
    QString geometryValidationJobId_;
    bool geometryPublishAfterValidation_ = false;
    QString geometryPublishOverrideReason_;
};
