#pragma once

#include <QWidget>
#include <QList>
#include <QJsonArray>
#include <QJsonObject>

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QProgressBar;
class QSpinBox;
class QTableWidget;
class QTextEdit;
class GeometryRuleCanvas;

class GeometryRulesPage : public QWidget {
    Q_OBJECT

public:
    explicit GeometryRulesPage(QWidget *parent = nullptr);

    void setSnapshot(const QJsonObject &snapshot);
    void setValidationJob(const QJsonObject &job);
    void setBusy(bool busy);
    void setPreviewBusy(bool busy);
    void setRulePreview(const QJsonObject &preview);
    void setOperationError(const QString &message);
    QJsonObject draft() const { return draft_; }
    QJsonObject snapshot() const { return snapshot_; }
    bool hasUnsavedChanges() const { return dirty_; }
    void discardUnsavedChanges();
    void requestSaveDraft();

signals:
    void unsavedChangesChanged(bool dirty);
    void snapshotRequested(const QString &workpieceId);
    void saveDraftRequested(const QJsonObject &draft, int libraryRevision, int draftRevision);
    void publishWorkflowRequested(const QJsonObject &draft, int libraryRevision, int draftRevision,
                                  const QString &overrideReason);
    void validateRequested(int libraryRevision, int draftRevision);
    void validationJobRequested(const QString &jobId);
    void validationJobActionRequested(const QString &jobId, const QString &action);
    void publishRequested(const QString &jobId, int libraryRevision, int draftRevision,
                          const QString &overrideReason);
    void rollbackRequested(int libraryRevision);
    void previewRequested(const QJsonObject &request);
    void migrationResolutionRequested(const QString &conflictId, const QJsonObject &resolution,
                                      int libraryRevision, int draftRevision);

private slots:
    void addRule();
    void deleteRule();
    void saveDraft();
    void validateDraft();
    void publishDraft();
    void publishWorkflow();
    void rollbackDraft();
    void refreshRuleList();
    void refreshTemplatePreview();
    void refreshTemplateReview();
    void setAnchorFromCanvas();
    void updatePublishState();
    void undoDraft();
    void redoDraft();
    void resetCurrentRule();
    void reloadDraft();
    void copyActiveToDraft();
    void requestPreviewFromShape(const QJsonObject &shape);
    void candidateChanged(int index);
    void selectValidationTemplate(int row);
    void resolveMigrationConflict();

private:
    struct MissingCalibration {
        bool valid = false;
        QString ruleId;
        QString direction;
    };

    QString direction() const;
    QString currentTemplateId() const;
    QString currentEditContextKey() const;
    QJsonObject directionObject() const;
    bool usesLogicalRuleSchema() const;
    QJsonArray logicalRules() const;
    QString currentRuleId() const;
    QJsonObject currentCalibration() const;
    void setCurrentCalibration(const QJsonObject &calibration);
    QString calibrationState(const QString &side, const QString &ruleId) const;
    void ensureDirectionObject(const QString &name);
    void restoreSnapshotDraft();
    void setCurrentRuleFromEditor();
    void loadCurrentRuleIntoEditor();
    QJsonObject currentRule() const;
    int currentRuleIndex() const;
    QJsonObject currentDirectionAnchor() const;
    QJsonObject canvasShapeForRule(const QJsonObject &rule) const;
    QJsonObject anchorFromCanvasShape(const QJsonObject &shape) const;
    QJsonObject ruleGeometryFromCanvasShape(const QJsonObject &shape) const;
    void setDirty(bool dirty);
    void markDraftDirty();
    void applyDraftMutation(const QJsonObject &next);
    void refreshEditor();
    QJsonObject previewRequest(int anchorCandidateIndex = -1, int ruleCandidateIndex = -1,
                               const QJsonObject &seedShape = QJsonObject()) const;
    bool applyFittedBoundaryForCurrentTemplate();
    void setTemplateDiagnostics(const QString &text);
    void refreshDiagnostics();
    void setTemplateReview(const QString &state, const QString &reason);
    MissingCalibration firstMissingEnabledCalibration() const;
    void selectRuleById(const QString &ruleId);
    void refreshMigrationPanel();
    QJsonObject selectedMigrationConflict() const;

    QJsonObject snapshot_;
    QJsonObject draft_;
    QJsonObject job_;
    bool busy_ = false;
    bool previewBusy_ = false;
    bool dirty_ = false;
    bool manualAnchorCapture_ = false;
    QComboBox *directionCombo_ = nullptr;
    QComboBox *templateCombo_ = nullptr;
    QComboBox *shapeCombo_ = nullptr;
    QComboBox *modeCombo_ = nullptr;
    QComboBox *anchorCandidateCombo_ = nullptr;
    QComboBox *ruleCandidateCombo_ = nullptr;
    QComboBox *reviewStateCombo_ = nullptr;
    QComboBox *versionCombo_ = nullptr;
    QComboBox *migrationConflictCombo_ = nullptr;
    QComboBox *migrationActionCombo_ = nullptr;
    QComboBox *migrationSurvivorCombo_ = nullptr;
    QComboBox *migrationFrontCombo_ = nullptr;
    QComboBox *migrationBackCombo_ = nullptr;
    QLineEdit *ruleNameEdit_ = nullptr;
    QLineEdit *overrideReasonEdit_ = nullptr;
    QLineEdit *reviewReasonEdit_ = nullptr;
    QSpinBox *marginSpin_ = nullptr;
    QDoubleSpinBox *rotationSpin_ = nullptr;
    QListWidget *ruleList_ = nullptr;
    QLabel *revisionLabel_ = nullptr;
    QLabel *statusLabel_ = nullptr;
    QProgressBar *progressBar_ = nullptr;
    QTableWidget *validationTable_ = nullptr;
    QTextEdit *diagnostics_ = nullptr;
    QLabel *validationHintLabel_ = nullptr;
    QPushButton *saveButton_ = nullptr;
    QPushButton *validateButton_ = nullptr;
    QPushButton *publishButton_ = nullptr;
    QPushButton *publishWorkflowButton_ = nullptr;
    QPushButton *rollbackButton_ = nullptr;
    QPushButton *setAnchorButton_ = nullptr;
    QPushButton *manualAnchorButton_ = nullptr;
    QPushButton *undoButton_ = nullptr;
    QPushButton *redoButton_ = nullptr;
    QPushButton *resetRuleButton_ = nullptr;
    QPushButton *reloadDraftButton_ = nullptr;
    QPushButton *copyActiveToDraftButton_ = nullptr;
    QPushButton *resolveMigrationButton_ = nullptr;
    QLabel *dirtyLabel_ = nullptr;
    GeometryRuleCanvas *canvas_ = nullptr;
    QList<QJsonObject> undoHistory_;
    QList<QJsonObject> redoHistory_;
    QString editorDirection_;
    bool suppressPreviewRequests_ = false;
    QJsonObject lastRulePreview_;
    QJsonObject lastPreviewRuleSignature_;
    QString lastPreviewWorkpieceId_;
    QString lastPreviewRuleId_;
    QString lastPreviewDirection_;
    QString lastPreviewTemplateId_;
    int lastPreviewLibraryRevision_ = -1;
    int lastPreviewDraftRevision_ = -1;
    QString manualEditContextKey_;
    QString pendingPreviewContextKey_;
    QString validationDiagnostics_;
    QString templateDiagnostics_;
};
