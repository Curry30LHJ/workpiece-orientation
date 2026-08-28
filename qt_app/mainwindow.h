#pragma once

#include <QMainWindow>
#include <QHash>
#include <QJsonObject>
#include <QQueue>
#include <QJsonArray>
#include <QStringList>
#include <QTimer>

#include <functional>

#include "appheader.h"
#include "backendclient.h"

class BackendProcessManager;
class InspectionPage;
class WorkpieceLibraryPage;
class QLabel;
class QPushButton;
class QTableWidget;
class QTextEdit;
class GeometryRulesPage;
class TaskStatusWidget;
class QCloseEvent;

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

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void showInspection();
    void showWorkpieceLibrary();
    void showGeometryRules();
    void chooseInspectionImage();
    void chooseBatchImages();
    void refreshWorkpieces();
    void submitRegistration();
    void submitPrediction();
    void submitBatchPrediction();
    void confirmFrontTemplate();
    void confirmBackTemplate();
    void rejectTemplateConfirmation();
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
    void onBackendLoading(const QString &phase, const QString &message, int progress);
    void onBackendUnavailable(const QString &reason, const QString &code = QString(),
                              const QString &action = QString(),
                              const QString &logPath = QString());
    void onClientStateChanged(BackendClient::State state, const QString &detail);
    void onClientProgress(const QString &command, const QJsonObject &progress);
    void onClientResponse(const QString &command, const QJsonObject &response);
    void onClientCommandFailed(const QString &command, const QString &code, const QString &message);
    void onClientTransportFailed(const QString &code, const QString &message);

private:
    enum class CommandOwner { None, System, UserRefresh, Inspection, Library, Geometry };
    enum class GeometryDirtyDecision { Save, Discard, Cancel };
    enum class ActiveTaskCloseDecision { ExitApplication, ContinueRunning };
    enum class PendingNavigationKind { None, Page, Workpiece, CloseWindow };
    enum class GeometrySaveIntent { None, Normal, Navigation, PublishWorkflow };

    struct QueuedCommandIntent {
        CommandOwner owner = CommandOwner::None;
        QString command;
        QJsonObject fields;
        quint64 refreshTransactionId = 0;
        quint64 geometryTargetGeneration = 0;
    };

    enum class ResultContext {
        None,
        Single,
        Batch,
    };

    void initializeUi();
    void connectBackendSignals();
    void presentBackendState(const BackendStatusDetails &details);
    void sendPageCommand(CommandOwner owner, const QString &command,
                         const QJsonObject &fields = QJsonObject(),
                         quint64 refreshTransactionId = 0);
    void dispatchQueuedCommand();
    bool dispatchGeometryWorkflowContinuation();
    void stageGeometryWorkflowContinuation(const QString &command,
                                           const QJsonObject &fields);
    void clearGeometryWorkflowTarget();
    void setGeometryOperationEditingLocked(bool locked);
    void setGeometryProfileLoadGeneration(quint64 generation);
    void updateGeometryEditingLock();
    void issuePageCommand(const QueuedCommandIntent &intent,
                          bool includesMandatoryRefresh = false,
                          bool includesUserRefresh = false);
    void clearPendingCommand();
    void updateButtonStates();
    void updatePreview();
    void clearInspectionState();
    void clearBatchState();
    void clearBatchResults();
    void requestWorkpieceRefresh(bool preserveRegistrationSummary,
                                 CommandOwner owner = CommandOwner::System);
    void dispatchDeferredUserRefresh();
    void submitTemplateConfirmation(const QString &workpieceId, const QString &imagePath,
                                    const QString &orientation);
    bool validateImagePath(const QString &path, QString *error) const;
    QStringList normalizedPaths(const QStringList &paths) const;
    QString selectedWorkpieceId() const;
    QString orientationText(const QString &label) const;
    void showLibraryMessage(const QString &message, bool error = false);
    void requestGeometryProfile(const QString &workpieceId);
    bool startGeometryDraftSave(const QJsonObject &draft, int libraryRevision,
                                int draftRevision, GeometrySaveIntent intent);
    bool dispatchStagedGeometryDraftSave();
    void clearStagedGeometryDraftSave();
    bool maybeContinueGeometryPublish(const QJsonObject &job);
    void applyGeometryValidationJob(const QJsonObject &job);
    void bindGeometryValidationContext(const QString &workpieceId,
                                       int libraryRevision, int draftRevision);
    void clearGeometryValidationContext();
    bool geometryValidationMatchesPage(const QJsonObject &job) const;
    GeometryDirtyDecision promptForDirtyGeometry();
    ActiveTaskCloseDecision promptForActiveTaskClose();
    bool hasActiveTask() const;
    bool beginPendingGeometryNavigation(PendingNavigationKind kind,
                                        AppPage page = AppPage::Inspection,
                                        const QString &workpieceId = QString());
    bool tryStartPendingGeometrySave();
    void cancelPendingGeometryNavigation();
    void completePendingGeometryNavigation();
    void applyWorkpieceListResponse(const QJsonObject &response,
                                    quint64 refreshTransactionId,
                                    bool includedMandatoryRefresh,
                                    const QString &preferredTarget = QString());
    void setCurrentPageUnchecked(AppPage page);
    bool applyDetectionWorkpieceChange(const QString &workpieceId);
    void refreshDetectionWorkpieceConsumers();
    void ensureGeometryProfileForCurrentWorkpiece();

    Ui::MainWindow *ui;
    AppHeader *appHeader_ = nullptr;
    TaskStatusWidget *globalTaskStatus_ = nullptr;
    InspectionPage *inspectionPage_ = nullptr;
    WorkpieceLibraryPage *workpieceLibraryPage_ = nullptr;
    AppPage currentPage_ = AppPage::Inspection;
    BackendClient *client_ = nullptr;
    BackendProcessManager *manager_ = nullptr;
    QString inspectionImagePath_;
    QStringList batchImagePaths_;
    QJsonArray workpieceSummaries_;
    CommandOwner pendingOwner_ = CommandOwner::None;
    QString pendingCommand_;
    QJsonObject pendingFields_;
    QQueue<QueuedCommandIntent> queuedMutationCommands_;
    QQueue<QueuedCommandIntent> queuedInternalCommands_;
    bool hasReplaceRegistrationContinuation_ = false;
    QJsonObject replaceRegistrationContinuationFields_;
    bool hasLatestDetailsIntent_ = false;
    QJsonObject latestDetailsFields_;
    quint64 latestDetailsRefreshTransactionId_ = 0;
    bool mandatoryWorkpieceRefresh_ = false;
    bool mandatoryRefreshRetryRequired_ = false;
    bool mandatoryDetailsRetryRequired_ = false;
    QString activeMandatoryDetailsWorkpieceId_;
    quint64 nextMandatoryRefreshTransactionId_ = 0;
    quint64 activeMandatoryRefreshTransactionId_ = 0;
    quint64 queuedMandatoryRefreshTransactionId_ = 0;
    quint64 pendingRefreshTransactionId_ = 0;
    bool deferredUserWorkpieceRefresh_ = false;
    bool pendingRefreshIncludesMandatory_ = false;
    bool pendingRefreshIncludesUser_ = false;
    bool batchInFlight_ = false;
    bool backendReady_ = false;
    bool backendReadyHandled_ = false;
    BackendUiState backendPresentationState_ = BackendUiState::Disconnected;
    bool backendEverReady_ = false;
    QString backendRecoveryDetail_;
    bool backendFailureIsRecoverable_ = false;
    bool clientBusy_ = false;
    QTimer *evolutionPollTimer_ = nullptr;
    QTimer *geometryPollTimer_ = nullptr;
    QPushButton *chooseImageButton_ = nullptr;
    QPushButton *predictButton_ = nullptr;
    QPushButton *chooseBatchImagesButton_ = nullptr;
    QPushButton *batchPredictButton_ = nullptr;
    QPushButton *confirmFrontButton_ = nullptr;
    QPushButton *confirmBackButton_ = nullptr;
    QPushButton *rejectConfirmationButton_ = nullptr;
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
    QJsonObject pendingConfirmationMutation_;
    QHash<QString, QJsonObject> uncertainConfirmationMutations_;
    ResultContext resultContext_ = ResultContext::None;
    GeometryRulesPage *geometryRulesPage_ = nullptr;
    QString geometryWorkpieceId_;
    QString geometryRequestedWorkpieceId_;
    QString currentDetectionWorkpieceId_;
    quint64 geometryTargetGeneration_ = 0;
    quint64 pendingGeometryTargetGeneration_ = 0;
    quint64 geometryProfileLoadGeneration_ = 0;
    bool geometryOperationEditingLocked_ = false;
    bool geometryForceProfileReload_ = false;
    QString geometryValidationJobId_;
    QString geometryValidationContextWorkpieceId_;
    int geometryValidationContextLibraryRevision_ = -1;
    int geometryValidationContextDraftRevision_ = -1;
    quint64 geometryValidationContextTargetGeneration_ = 0;
    QString geometryValidationContextJobId_;
    PendingNavigationKind pendingNavigationKind_ = PendingNavigationKind::None;
    AppPage pendingNavigationPage_ = AppPage::Inspection;
    QString pendingNavigationWorkpieceId_;
    QJsonObject pendingNavigationWorkpieceListResponse_;
    quint64 pendingNavigationRefreshTransactionId_ = 0;
    bool pendingNavigationResponseIncludedMandatory_ = false;
    bool skipDirtyCloseGuardOnce_ = false;
    bool backendShutdownPending_ = false;
    bool backendShutdownComplete_ = false;
    GeometrySaveIntent geometrySaveIntent_ = GeometrySaveIntent::None;
    GeometrySaveIntent stagedGeometrySaveIntent_ = GeometrySaveIntent::None;
    QString stagedGeometrySaveWorkpieceId_;
    QJsonObject stagedGeometrySaveDraft_;
    int stagedGeometrySaveLibraryRevision_ = -1;
    int stagedGeometrySaveDraftRevision_ = -1;
    bool geometryPublishAfterValidation_ = false;
    QString geometryPublishOverrideReason_;
    QString geometryWorkflowWorkpieceId_;
    int geometryWorkflowLibraryRevision_ = -1;
    int geometryWorkflowDraftRevision_ = -1;
    QString geometryWorkflowJobId_;
    QString geometryWorkflowContinuationCommand_;
    QJsonObject geometryWorkflowContinuationFields_;
};
