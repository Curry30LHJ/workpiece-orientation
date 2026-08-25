#pragma once

#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>

#include <functional>

#include "annotationcanvas.h"

class QListWidget;
class QTableWidget;
class QTextEdit;
class QLabel;
class QPushButton;

class AnnotationManagerDialog : public QDialog {
    Q_OBJECT

public:
    explicit AnnotationManagerDialog(QWidget *parent = nullptr);

    void setSnapshot(const QJsonObject &snapshot);
    void setBusy(bool busy);
    void setProgress(const QString &phase, int completed, int total);
    void setOperationError(const QString &message);
    void setDeleteConfirmationHandler(std::function<bool(const QString &name, int affectedTemplates)> handler);
    QString workpieceId() const { return workpieceId_; }
    int baseRevision() const { return baseRevision_; }
    QJsonObject snapshot() const { return snapshot_; }

signals:
    void refreshRequested();
    void saveRequested(const QJsonArray &groups, int baseRevision);
    void setEnabledRequested(const QString &groupId, bool enabled, int baseRevision);
    void deleteRequested(const QString &groupId, int baseRevision);
    void reviewRequested(const QString &groupId, const QString &templateId,
                         const QString &action, const QJsonArray &regions, int baseRevision);

private slots:
    void selectGroup(int row);
    void selectTemplate(int row, int column);
    void deleteSelectedGroup();
    void toggleSelectedGroup();
    void addAnnotationGroup();
    void editSelectedAnnotation();
    void renameAnnotationGroup();
    void acceptSelectedAnnotation();
    void correctSelectedAnnotation();
    void rejectSelectedAnnotation();
    void absentSelectedAnnotation();
    void repropagateSelectedAnnotation();

private:
    QJsonObject selectedGroup() const;
    QJsonObject selectedTarget() const;
    QString selectedGroupId() const;
    QString selectedTemplateId() const;
    QJsonArray selectedRegions() const;
    void renderGroupList();
    void renderTargets();
    void renderSelectedTarget();
    void setButtonState();
    static QString statusText(const QJsonObject &group);
    static QString targetStatusText(const QString &state);
    static QString diagnosticReasonText(const QString &reasonCode);
    static QJsonArray regionArray(const QJsonObject &target);
    static QList<AnnotationRegionView> toCanvasRegions(const QJsonArray &regions,
                                                       const QString &provenance,
                                                       const QString &status);

    QJsonObject snapshot_;
    QString workpieceId_;
    int baseRevision_ = -1;
    bool busy_ = false;
    QListWidget *groupList_ = nullptr;
    QTableWidget *templateTable_ = nullptr;
    AnnotationCanvas *previewCanvas_ = nullptr;
    QTextEdit *diagnosticsText_ = nullptr;
    QLabel *versionLabel_ = nullptr;
    QLabel *progressLabel_ = nullptr;
    QLabel *operationStatusLabel_ = nullptr;
    QPushButton *addGroupButton_ = nullptr;
    QPushButton *renameGroupButton_ = nullptr;
    QPushButton *editButton_ = nullptr;
    QPushButton *toggleGroupButton_ = nullptr;
    QPushButton *deleteGroupButton_ = nullptr;
    QPushButton *acceptButton_ = nullptr;
    QPushButton *correctButton_ = nullptr;
    QPushButton *rejectButton_ = nullptr;
    QPushButton *absentButton_ = nullptr;
    QPushButton *repropagateButton_ = nullptr;
    std::function<bool(const QString &, int)> deleteConfirmationHandler_;
};
