#include "annotationmanager.h"

#include "annotationcanvas.h"
#include "annotationeditor.h"

#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonDocument>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QAbstractItemView>
#include <QTableWidget>
#include <QTextEdit>
#include <QVBoxLayout>

namespace {
QVariant jsonValue(const QJsonObject &value) {
    return QString::fromUtf8(QJsonDocument(value).toJson(QJsonDocument::Compact));
}

QJsonObject objectValue(const QVariant &value) {
    return QJsonDocument::fromJson(value.toString().toUtf8()).object();
}
}

AnnotationManagerDialog::AnnotationManagerDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle(QStringLiteral("干扰标注管理"));
    resize(1200, 760);
    auto *root = new QVBoxLayout(this);
    versionLabel_ = new QLabel(this);
    versionLabel_->setObjectName(QStringLiteral("annotationVersionLabel"));
    root->addWidget(versionLabel_);
    progressLabel_ = new QLabel(this);
    progressLabel_->setObjectName(QStringLiteral("annotationProgressLabel"));
    root->addWidget(progressLabel_);
    operationStatusLabel_ = new QLabel(this);
    operationStatusLabel_->setObjectName(QStringLiteral("annotationOperationStatusLabel"));
    root->addWidget(operationStatusLabel_);
    auto *columns = new QHBoxLayout();
    groupList_ = new QListWidget(this);
    groupList_->setObjectName(QStringLiteral("annotationGroupList"));
    groupList_->setMinimumWidth(220);
    columns->addWidget(groupList_);
    templateTable_ = new QTableWidget(this);
    templateTable_->setObjectName(QStringLiteral("annotationTemplateTable"));
    templateTable_->setColumnCount(5);
    templateTable_->setHorizontalHeaderLabels({QStringLiteral("模板"), QStringLiteral("方向"),
                                                QStringLiteral("状态"), QStringLiteral("来源"), QStringLiteral("区域")});
    templateTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    templateTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    columns->addWidget(templateTable_, 2);
    auto *previewColumn = new QVBoxLayout();
    previewCanvas_ = new AnnotationCanvas(this);
    previewCanvas_->setObjectName(QStringLiteral("annotationPreviewCanvas"));
    previewCanvas_->setEditable(false);
    previewColumn->addWidget(previewCanvas_, 3);
    diagnosticsText_ = new QTextEdit(this);
    diagnosticsText_->setObjectName(QStringLiteral("annotationDiagnosticsText"));
    diagnosticsText_->setReadOnly(true);
    previewColumn->addWidget(diagnosticsText_, 2);
    columns->addLayout(previewColumn, 3);
    root->addLayout(columns, 1);

    auto *controls = new QGridLayout();
    addGroupButton_ = new QPushButton(QStringLiteral("新增类型"), this);
    addGroupButton_->setObjectName(QStringLiteral("addAnnotationGroupButton"));
    renameGroupButton_ = new QPushButton(QStringLiteral("重命名"), this);
    renameGroupButton_->setObjectName(QStringLiteral("renameAnnotationGroupButton"));
    editButton_ = new QPushButton(QStringLiteral("编辑区域"), this);
    editButton_->setObjectName(QStringLiteral("editAnnotationButton"));
    toggleGroupButton_ = new QPushButton(this);
    toggleGroupButton_->setObjectName(QStringLiteral("toggleAnnotationGroupButton"));
    deleteGroupButton_ = new QPushButton(QStringLiteral("删除类型"), this);
    deleteGroupButton_->setObjectName(QStringLiteral("deleteAnnotationGroupButton"));
    acceptButton_ = new QPushButton(QStringLiteral("接受"), this);
    acceptButton_->setObjectName(QStringLiteral("acceptAnnotationButton"));
    correctButton_ = new QPushButton(QStringLiteral("修正"), this);
    correctButton_->setObjectName(QStringLiteral("correctAnnotationButton"));
    rejectButton_ = new QPushButton(QStringLiteral("拒绝"), this);
    rejectButton_->setObjectName(QStringLiteral("rejectAnnotationButton"));
    absentButton_ = new QPushButton(QStringLiteral("标记不存在"), this);
    absentButton_->setObjectName(QStringLiteral("absentAnnotationButton"));
    repropagateButton_ = new QPushButton(QStringLiteral("重新递推"), this);
    repropagateButton_->setObjectName(QStringLiteral("repropagateAnnotationButton"));
    const QList<QPushButton *> buttons{addGroupButton_, renameGroupButton_, editButton_, toggleGroupButton_, deleteGroupButton_,
                                       acceptButton_, correctButton_, rejectButton_, absentButton_, repropagateButton_};
    for (int index = 0; index < buttons.size(); ++index) {
        controls->addWidget(buttons.at(index), index / 5, index % 5);
    }
    root->addLayout(controls);

    connect(groupList_, &QListWidget::currentRowChanged, this, &AnnotationManagerDialog::selectGroup);
    connect(templateTable_, &QTableWidget::cellClicked, this, &AnnotationManagerDialog::selectTemplate);
    connect(templateTable_, &QTableWidget::currentCellChanged, this,
            [this](int, int, int, int) { renderSelectedTarget(); setButtonState(); });
    connect(addGroupButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::addAnnotationGroup);
    connect(renameGroupButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::renameAnnotationGroup);
    connect(editButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::editSelectedAnnotation);
    connect(toggleGroupButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::toggleSelectedGroup);
    connect(deleteGroupButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::deleteSelectedGroup);
    connect(acceptButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::acceptSelectedAnnotation);
    connect(correctButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::correctSelectedAnnotation);
    connect(rejectButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::rejectSelectedAnnotation);
    connect(absentButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::absentSelectedAnnotation);
    connect(repropagateButton_, &QPushButton::clicked, this, &AnnotationManagerDialog::repropagateSelectedAnnotation);
    setButtonState();
}

void AnnotationManagerDialog::setDeleteConfirmationHandler(
    std::function<bool(const QString &, int)> handler) {
    deleteConfirmationHandler_ = std::move(handler);
}

void AnnotationManagerDialog::setSnapshot(const QJsonObject &snapshot) {
    snapshot_ = snapshot;
    workpieceId_ = snapshot.value(QStringLiteral("workpiece_id")).toString();
    baseRevision_ = snapshot.value(QStringLiteral("revision")).toInt(-1);
    versionLabel_->setText(QStringLiteral("工件 %1：草稿修订 %2，当前识别使用修订 %3")
                               .arg(workpieceId_)
                               .arg(snapshot.value(QStringLiteral("annotation_revision")).toInt())
                               .arg(snapshot.value(QStringLiteral("active_annotation_revision")).toInt()));
    if (snapshot.value(QStringLiteral("annotation_revision")).toInt()
        != snapshot.value(QStringLiteral("active_annotation_revision")).toInt()) {
        versionLabel_->setText(versionLabel_->text() + QStringLiteral("；草稿尚未参与预测，当前仍使用修订前稳定掩码"));
    }
    renderGroupList();
    progressLabel_->clear();
    operationStatusLabel_->clear();
    operationStatusLabel_->setStyleSheet(QString());
    setButtonState();
}

void AnnotationManagerDialog::setBusy(bool busy) {
    busy_ = busy;
    if (!busy_) progressLabel_->clear();
    setButtonState();
}

void AnnotationManagerDialog::setProgress(const QString &phase, int completed, int total) {
    if (phase != QStringLiteral("propagating_annotations") || total <= 0) return;
    progressLabel_->setText(QStringLiteral("正在递推干扰区域：%1 / %2，完成后将刷新模板列表")
                                .arg(qBound(0, completed, total)).arg(total));
}

void AnnotationManagerDialog::setOperationError(const QString &message) {
    operationStatusLabel_->setStyleSheet(QStringLiteral("color: #b00020;"));
    operationStatusLabel_->setText(message);
}

QString AnnotationManagerDialog::selectedGroupId() const {
    const QListWidgetItem *item = groupList_->currentItem();
    return item == nullptr ? QString() : item->data(Qt::UserRole).toString();
}

QString AnnotationManagerDialog::selectedTemplateId() const {
    const int row = templateTable_->currentRow();
    return row < 0 ? QString() : objectValue(templateTable_->item(row, 0)->data(Qt::UserRole))
        .value(QStringLiteral("template_id")).toString();
}

QJsonObject AnnotationManagerDialog::selectedGroup() const {
    const QString id = selectedGroupId();
    for (const QJsonValue &value : snapshot_.value(QStringLiteral("groups")).toArray()) {
        const QJsonObject group = value.toObject();
        if (group.value(QStringLiteral("group_id")).toString() == id) {
            return group;
        }
    }
    return {};
}

QJsonObject AnnotationManagerDialog::selectedTarget() const {
    const int row = templateTable_->currentRow();
    return row < 0 ? QJsonObject() : objectValue(templateTable_->item(row, 0)->data(Qt::UserRole));
}

QJsonArray AnnotationManagerDialog::selectedRegions() const {
    return regionArray(selectedTarget());
}

QString AnnotationManagerDialog::statusText(const QJsonObject &group) {
    if (!group.value(QStringLiteral("enabled")).toBool(true)) return QStringLiteral("已停用");
    const QString state = group.value(QStringLiteral("draft_state")).toString();
    if (state == QStringLiteral("active")) return QStringLiteral("已生效");
    if (state == QStringLiteral("needs_review")) return QStringLiteral("待复核");
    return QStringLiteral("无有效种子");
}

QString AnnotationManagerDialog::targetStatusText(const QString &state) {
    if (state == QStringLiteral("active")) return QStringLiteral("已生效");
    if (state == QStringLiteral("needs_review")) return QStringLiteral("待复核");
    if (state == QStringLiteral("confirmed_absent")) return QStringLiteral("已确认不存在");
    if (state == QStringLiteral("rejected")) return QStringLiteral("已拒绝候选");
    return QStringLiteral("未解析");
}

QString AnnotationManagerDialog::diagnosticReasonText(const QString &reasonCode) {
    if (reasonCode == QStringLiteral("projection_failed")) {
        return QStringLiteral("局部特征投影失败");
    }
    if (reasonCode == QStringLiteral("insufficient_projections")) {
        return QStringLiteral("有效对应关系不足");
    }
    if (reasonCode == QStringLiteral("target_unreadable")) {
        return QStringLiteral("目标模板无法读取");
    }
    if (reasonCode == QStringLiteral("source_disagreement")) {
        return QStringLiteral("来源投影不一致");
    }
    if (reasonCode == QStringLiteral("sources_agree")) {
        return QStringLiteral("来源投影一致");
    }
    if (reasonCode == QStringLiteral("manual")) {
        return QStringLiteral("人工标注");
    }
    return reasonCode.isEmpty() ? QStringLiteral("旧版本未记录该诊断") : reasonCode;
}

QJsonArray AnnotationManagerDialog::regionArray(const QJsonObject &target) {
    return target.value(QStringLiteral("regions")).toArray();
}

QList<AnnotationRegionView> AnnotationManagerDialog::toCanvasRegions(const QJsonArray &regions,
                                                                      const QString &provenance,
                                                                      const QString &status) {
    QList<AnnotationRegionView> values;
    for (const QJsonValue &value : regions) {
        const QJsonObject item = value.toObject();
        values.append({QRectF(item.value(QStringLiteral("x")).toDouble(), item.value(QStringLiteral("y")).toDouble(),
                              item.value(QStringLiteral("width")).toDouble(), item.value(QStringLiteral("height")).toDouble()),
                       item.value(QStringLiteral("provenance")).toString(provenance.isEmpty() ? QStringLiteral("manual") : provenance),
                       item.value(QStringLiteral("status")).toString(status.isEmpty() ? QStringLiteral("active") : status), false});
    }
    return values;
}

void AnnotationManagerDialog::renderGroupList() {
    const QString oldId = selectedGroupId();
    groupList_->blockSignals(true);
    groupList_->clear();
    for (const QJsonValue &value : snapshot_.value(QStringLiteral("groups")).toArray()) {
        const QJsonObject group = value.toObject();
        const QJsonObject summary = group.value(QStringLiteral("summary")).toObject();
        auto *item = new QListWidgetItem(QStringLiteral("%1  [%2]  人工%3/自动%4/复核%5")
                                             .arg(group.value(QStringLiteral("name")).toString(group.value(QStringLiteral("group_id")).toString()))
                                             .arg(statusText(group))
                                             .arg(summary.value(QStringLiteral("manual_count")).toInt())
                                             .arg(summary.value(QStringLiteral("automatic_count")).toInt())
                                             .arg(summary.value(QStringLiteral("needs_review_count")).toInt()), groupList_);
        item->setData(Qt::UserRole, group.value(QStringLiteral("group_id")).toString());
    }
    int row = -1;
    for (int index = 0; index < groupList_->count(); ++index) {
        if (groupList_->item(index)->data(Qt::UserRole).toString() == oldId) { row = index; break; }
    }
    if (row < 0 && groupList_->count() > 0) row = 0;
    groupList_->setCurrentRow(row);
    groupList_->blockSignals(false);
    renderTargets();
}

void AnnotationManagerDialog::renderTargets() {
    templateTable_->blockSignals(true);
    templateTable_->setRowCount(0);
    const QJsonObject group = selectedGroup();
    const QJsonArray targets = group.value(QStringLiteral("targets")).toArray();
    for (const QJsonValue &value : targets) {
        const QJsonObject target = value.toObject();
        const int row = templateTable_->rowCount();
        templateTable_->insertRow(row);
        auto *id = new QTableWidgetItem(target.value(QStringLiteral("template_id")).toString());
        id->setData(Qt::UserRole, jsonValue(target));
        templateTable_->setItem(row, 0, id);
        templateTable_->setItem(row, 1, new QTableWidgetItem(target.value(QStringLiteral("orientation")).toString()));
        templateTable_->setItem(row, 2, new QTableWidgetItem(targetStatusText(target.value(QStringLiteral("state")).toString())));
        templateTable_->setItem(row, 3, new QTableWidgetItem(target.value(QStringLiteral("provenance")).toString()));
        templateTable_->setItem(row, 4, new QTableWidgetItem(QString::number(regionArray(target).size())));
    }
    if (templateTable_->rowCount() > 0) templateTable_->selectRow(0);
    templateTable_->blockSignals(false);
    renderSelectedTarget();
    setButtonState();
}

void AnnotationManagerDialog::renderSelectedTarget() {
    const QJsonObject target = selectedTarget();
    QString path;
    bool readable = false;
    for (const QJsonValue &value : snapshot_.value(QStringLiteral("templates")).toArray()) {
        const QJsonObject item = value.toObject();
        if (item.value(QStringLiteral("template_id")).toString() == target.value(QStringLiteral("template_id")).toString()) {
            path = item.value(QStringLiteral("preview_path")).toString();
            readable = item.value(QStringLiteral("readable")).toBool(true);
            previewCanvas_->setImage(readable ? QImage(path) : QImage());
            if (readable && previewCanvas_->image().isNull()) {
                readable = false;
            }
            break;
        }
    }
    if (target.isEmpty()) {
        previewCanvas_->setImage(QImage());
        diagnosticsText_->setPlainText(QStringLiteral("请选择模板"));
        return;
    }
    previewCanvas_->setRegions(toCanvasRegions(regionArray(target),
                                               target.value(QStringLiteral("provenance")).toString(),
                                               target.value(QStringLiteral("state")).toString()));
    const QJsonObject diagnostics = target.value(QStringLiteral("diagnostics")).toObject();
    QStringList lines;
    lines << QStringLiteral("模板：%1").arg(target.value(QStringLiteral("template_id")).toString());
    lines << QStringLiteral("状态：%1").arg(targetStatusText(target.value(QStringLiteral("state")).toString()));
    lines << QStringLiteral("来源：%1").arg(target.value(QStringLiteral("provenance")).toString());
    if (!readable) {
        lines << QStringLiteral("模板图片无法读取");
    }
    const QString reasonCode = diagnostics.value(QStringLiteral("reason_code")).toString();
    lines << QStringLiteral("原因码：%1（%2）")
                  .arg(reasonCode.isEmpty() ? QStringLiteral("旧版本未记录该诊断") : reasonCode,
                       diagnosticReasonText(reasonCode));
    lines << QStringLiteral("来源模板：%1").arg(QString::fromUtf8(QJsonDocument(diagnostics.value(QStringLiteral("source_template_ids")).toArray()).toJson(QJsonDocument::Compact)));
    lines << QStringLiteral("置信度：%1，最大分歧：%2 px，面积占比：%3")
                  .arg(diagnostics.value(QStringLiteral("confidence")).toDouble())
                  .arg(diagnostics.value(QStringLiteral("max_spread_px")).toDouble())
                  .arg(diagnostics.value(QStringLiteral("area_ratio")).toDouble());
    lines << QStringLiteral("关键点：%1 → %2（保留 %3）")
                  .arg(diagnostics.value(QStringLiteral("keypoints_before")).toInt())
                  .arg(diagnostics.value(QStringLiteral("keypoints_after")).toInt())
                  .arg(diagnostics.value(QStringLiteral("remaining_ratio")).toDouble());
    diagnosticsText_->setPlainText(lines.join(QLatin1Char('\n')));
}

void AnnotationManagerDialog::selectGroup(int row) { Q_UNUSED(row) renderTargets(); }
void AnnotationManagerDialog::selectTemplate(int row, int column) { Q_UNUSED(row) Q_UNUSED(column) renderSelectedTarget(); }

void AnnotationManagerDialog::setButtonState() {
    const bool hasGroup = !selectedGroupId().isEmpty();
    const bool hasTarget = !selectedTemplateId().isEmpty();
    const bool enabled = selectedGroup().value(QStringLiteral("enabled")).toBool(true);
    addGroupButton_->setEnabled(!busy_);
    renameGroupButton_->setEnabled(!busy_ && hasGroup);
    editButton_->setEnabled(!busy_ && hasGroup && hasTarget);
    toggleGroupButton_->setEnabled(!busy_ && hasGroup);
    toggleGroupButton_->setText(enabled ? QStringLiteral("停用类型") : QStringLiteral("启用类型"));
    deleteGroupButton_->setEnabled(!busy_ && hasGroup);
    acceptButton_->setEnabled(!busy_ && hasGroup && hasTarget);
    correctButton_->setEnabled(!busy_ && hasGroup && hasTarget);
    rejectButton_->setEnabled(!busy_ && hasGroup && hasTarget);
    absentButton_->setEnabled(!busy_ && hasGroup && hasTarget);
    repropagateButton_->setEnabled(!busy_ && hasGroup);
}

void AnnotationManagerDialog::deleteSelectedGroup() {
    const QString id = selectedGroupId();
    if (id.isEmpty()) return;
    const QJsonObject group = selectedGroup();
    const int affected = group.value(QStringLiteral("targets")).toArray().size();
    const QString name = group.value(QStringLiteral("name")).toString(id);
    const bool confirmed = deleteConfirmationHandler_ ? deleteConfirmationHandler_(name, affected)
        : QMessageBox::question(this, QStringLiteral("确认删除类型"),
                                QStringLiteral("删除“%1”将移除 %2 个模板上的人工与自动区域，是否继续？").arg(name).arg(affected),
                                QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes;
    if (confirmed) emit deleteRequested(id, baseRevision_);
}

void AnnotationManagerDialog::toggleSelectedGroup() {
    if (!selectedGroupId().isEmpty()) emit setEnabledRequested(selectedGroupId(),
        !selectedGroup().value(QStringLiteral("enabled")).toBool(true), baseRevision_);
}

void AnnotationManagerDialog::addAnnotationGroup() {
    bool ok = false;
    const QString name = QInputDialog::getText(this, QStringLiteral("新增干扰类型"), QStringLiteral("类型名称"),
                                               QLineEdit::Normal, QString(), &ok).trimmed();
    if (!ok || name.isEmpty()) return;
    QJsonObject draftGroup{{"group_id", name}, {"name", name}, {"enabled", true},
                           {"annotations", QJsonArray{}}};
    QJsonObject editorSnapshot = snapshot_;
    editorSnapshot.insert(QStringLiteral("groups"), QJsonArray{draftGroup});
    const QJsonArray templates = snapshot_.value(QStringLiteral("templates")).toArray();
    const QString templateId = templates.isEmpty() ? QString() : templates.first().toObject()
        .value(QStringLiteral("template_id")).toString();
    AnnotationEditorDialog editor(editorSnapshot, name, templateId, this);
    if (editor.exec() == QDialog::Accepted) {
        emit saveRequested(QJsonArray{editor.annotationPatch()}, baseRevision_);
    }
}

void AnnotationManagerDialog::editSelectedAnnotation() {
    const QString groupId = selectedGroupId();
    const QString templateId = selectedTemplateId();
    if (groupId.isEmpty() || templateId.isEmpty()) return;
    AnnotationEditorDialog editor(snapshot_, groupId, templateId, this);
    if (editor.exec() != QDialog::Accepted) return;
    const QJsonArray annotations = editor.annotationPatch().value(QStringLiteral("annotations")).toArray();
    for (const QJsonValue &value : annotations) {
        const QJsonObject annotation = value.toObject();
        if (annotation.value(QStringLiteral("template_id")).toString() == templateId) {
            emit reviewRequested(groupId, templateId, QStringLiteral("correct"),
                                 annotation.value(QStringLiteral("regions")).toArray(), baseRevision_);
            return;
        }
    }
    emit reviewRequested(groupId, templateId, QStringLiteral("correct"), {}, baseRevision_);
}

void AnnotationManagerDialog::renameAnnotationGroup() {
    const QJsonObject group = selectedGroup();
    if (group.isEmpty()) return;
    bool ok = false;
    const QString name = QInputDialog::getText(this, QStringLiteral("重命名干扰类型"), QStringLiteral("类型名称"),
                                               QLineEdit::Normal, group.value("name").toString(), &ok).trimmed();
    if (!ok || name.isEmpty()) return;
    QJsonObject replacement = group;
    replacement.insert(QStringLiteral("name"), name);
    emit saveRequested(QJsonArray{replacement}, baseRevision_);
}

void AnnotationManagerDialog::acceptSelectedAnnotation() {
    emit reviewRequested(selectedGroupId(), selectedTemplateId(), QStringLiteral("accept"), selectedRegions(), baseRevision_);
}
void AnnotationManagerDialog::correctSelectedAnnotation() {
    emit reviewRequested(selectedGroupId(), selectedTemplateId(), QStringLiteral("correct"), selectedRegions(), baseRevision_);
}
void AnnotationManagerDialog::rejectSelectedAnnotation() {
    emit reviewRequested(selectedGroupId(), selectedTemplateId(), QStringLiteral("reject"), selectedRegions(), baseRevision_);
}
void AnnotationManagerDialog::absentSelectedAnnotation() {
    emit reviewRequested(selectedGroupId(), selectedTemplateId(), QStringLiteral("absent"), {}, baseRevision_);
}
void AnnotationManagerDialog::repropagateSelectedAnnotation() {
    emit reviewRequested(selectedGroupId(), selectedTemplateId(), QStringLiteral("repropagate"), selectedRegions(), baseRevision_);
}
