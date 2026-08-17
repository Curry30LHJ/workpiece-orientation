#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QProgressBar;
class QSpinBox;
class QTextEdit;
class GeometryRuleCanvas;

class GeometryMaskManagerDialog : public QDialog {
    Q_OBJECT

public:
    explicit GeometryMaskManagerDialog(QWidget *parent = nullptr);

    void setSnapshot(const QJsonObject &snapshot);
    void setValidationJob(const QJsonObject &job);
    void setBusy(bool busy);
    void setOperationError(const QString &message);
    QJsonObject draft() const { return draft_; }
    QJsonObject snapshot() const { return snapshot_; }

signals:
    void snapshotRequested(const QString &workpieceId);
    void saveDraftRequested(const QJsonObject &draft, int libraryRevision, int draftRevision);
    void validateRequested(int libraryRevision, int draftRevision);
    void validationJobRequested(const QString &jobId);
    void validationJobActionRequested(const QString &jobId, const QString &action);
    void publishRequested(const QString &jobId, int libraryRevision, int draftRevision,
                          const QString &overrideReason);
    void rollbackRequested(int libraryRevision);

private slots:
    void addRule();
    void deleteRule();
    void saveDraft();
    void validateDraft();
    void publishDraft();
    void rollbackDraft();
    void refreshRuleList();
    void updatePublishState();

private:
    QString direction() const;
    QJsonObject directionObject() const;
    void ensureDirectionObject(const QString &name);
    void setCurrentRuleFromEditor();
    void loadCurrentRuleIntoEditor();
    QJsonObject currentRule() const;
    int currentRuleIndex() const;

    QJsonObject snapshot_;
    QJsonObject draft_;
    QJsonObject job_;
    bool busy_ = false;
    QComboBox *directionCombo_ = nullptr;
    QComboBox *shapeCombo_ = nullptr;
    QComboBox *modeCombo_ = nullptr;
    QLineEdit *ruleNameEdit_ = nullptr;
    QLineEdit *overrideReasonEdit_ = nullptr;
    QSpinBox *marginSpin_ = nullptr;
    QListWidget *ruleList_ = nullptr;
    QLabel *revisionLabel_ = nullptr;
    QLabel *statusLabel_ = nullptr;
    QProgressBar *progressBar_ = nullptr;
    QTextEdit *diagnostics_ = nullptr;
    QPushButton *saveButton_ = nullptr;
    QPushButton *validateButton_ = nullptr;
    QPushButton *publishButton_ = nullptr;
    QPushButton *rollbackButton_ = nullptr;
    QPushButton *cancelButton_ = nullptr;
    GeometryRuleCanvas *canvas_ = nullptr;
};
