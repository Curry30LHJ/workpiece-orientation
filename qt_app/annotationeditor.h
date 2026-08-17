#pragma once

#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>
#include <QList>
#include <QMessageBox>
#include <QRectF>
#include <QStringList>

#include <functional>

#include "annotationcanvas.h"

class QComboBox;
class QLineEdit;

class AnnotationEditorDialog : public QDialog {
    Q_OBJECT

public:
    explicit AnnotationEditorDialog(const QStringList &frontPaths, const QStringList &backPaths,
                                    QWidget *parent = nullptr);
    explicit AnnotationEditorDialog(const QJsonObject &snapshot, const QString &groupId,
                                    const QString &templateId, QWidget *parent = nullptr);

    QString groupName() const;
    QString orientation() const;
    int templateIndex() const;
    QRectF nativeRegion() const;
    QJsonObject annotationPatch() const;
    bool hasUnsavedChanges() const { return unsavedChanges_; }
    void setUnsavedPromptHandler(std::function<QMessageBox::StandardButton()> handler);

private slots:
    void loadSelectedImage();
    void markDirty();
    void deleteSelectedRegion();

private:
    struct TemplateEntry {
        QString id;
        QString orientation;
        int index = -1;
        QString path;
        QJsonArray regions;
    };

    void addTemplate(const QString &id, const QString &orientation, int index,
                     const QString &path, const QJsonArray &regions);
    void loadEntry(int index);
    bool confirmSwitch();
    void syncCurrentEntry() const;
    QJsonArray regionsToJson(const QList<AnnotationRegionView> &regions) const;
    QList<AnnotationRegionView> regionsFromJson(const QJsonArray &regions) const;

    AnnotationCanvas *canvas_ = nullptr;
    QComboBox *templateCombo_ = nullptr;
    QLineEdit *groupNameEdit_ = nullptr;
    mutable QList<TemplateEntry> templates_;
    QString groupId_;
    int currentEntry_ = -1;
    bool loading_ = false;
    bool unsavedChanges_ = false;
    std::function<QMessageBox::StandardButton()> unsavedPromptHandler_;
};
