#include "annotationeditor.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

AnnotationEditorDialog::AnnotationEditorDialog(const QStringList &frontPaths, const QStringList &backPaths,
                                               QWidget *parent)
    : QDialog(parent) {
    setWindowTitle(QStringLiteral("标注干扰区域"));
    auto *layout = new QVBoxLayout(this);
    auto *form = new QFormLayout();
    groupNameEdit_ = new QLineEdit(this);
    groupNameEdit_->setText(QStringLiteral("干扰区域"));
    templateCombo_ = new QComboBox(this);
    for (int index = 0; index < frontPaths.size(); ++index) {
        addTemplate(QStringLiteral("front:%1").arg(index), QStringLiteral("front"), index, frontPaths.at(index), {});
    }
    for (int index = 0; index < backPaths.size(); ++index) {
        addTemplate(QStringLiteral("back:%1").arg(index), QStringLiteral("back"), index, backPaths.at(index), {});
    }
    form->addRow(QStringLiteral("干扰组名称"), groupNameEdit_);
    form->addRow(QStringLiteral("模板"), templateCombo_);
    layout->addLayout(form);
    canvas_ = new AnnotationCanvas(this);
    layout->addWidget(canvas_);
    auto *deleteButton = new QPushButton(QStringLiteral("删除选中区域"), this);
    layout->addWidget(deleteButton);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttons);
    connect(templateCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AnnotationEditorDialog::loadSelectedImage);
    connect(canvas_, &AnnotationCanvas::regionsChanged, this, &AnnotationEditorDialog::markDirty);
    connect(groupNameEdit_, &QLineEdit::textChanged, this, &AnnotationEditorDialog::markDirty);
    connect(deleteButton, &QPushButton::clicked, this, &AnnotationEditorDialog::deleteSelectedRegion);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    loadSelectedImage();
}

AnnotationEditorDialog::AnnotationEditorDialog(const QJsonObject &snapshot, const QString &groupId,
                                               const QString &templateId, QWidget *parent)
    : QDialog(parent), groupId_(groupId) {
    setWindowTitle(QStringLiteral("管理干扰区域"));
    auto *layout = new QVBoxLayout(this);
    auto *form = new QFormLayout();
    groupNameEdit_ = new QLineEdit(this);
    templateCombo_ = new QComboBox(this);
    QJsonObject selectedGroup;
    for (const QJsonValue &value : snapshot.value(QStringLiteral("groups")).toArray()) {
        const QJsonObject group = value.toObject();
        if (group.value(QStringLiteral("group_id")).toString() == groupId) {
            selectedGroup = group;
            break;
        }
    }
    groupNameEdit_->setText(selectedGroup.value(QStringLiteral("name")).toString(groupId));
    const QJsonArray targets = selectedGroup.value(QStringLiteral("targets")).toArray();
    for (const QJsonValue &value : snapshot.value(QStringLiteral("templates")).toArray()) {
        const QJsonObject item = value.toObject();
        const QString id = item.value(QStringLiteral("template_id")).toString();
        QJsonArray regions;
        for (const QJsonValue &targetValue : targets) {
            const QJsonObject target = targetValue.toObject();
            if (target.value(QStringLiteral("template_id")).toString() == id) {
                regions = target.value(QStringLiteral("regions")).toArray();
                break;
            }
        }
        addTemplate(id, item.value(QStringLiteral("orientation")).toString(),
                    item.value(QStringLiteral("index")).toInt(-1),
                    item.value(QStringLiteral("preview_path")).toString(), regions);
    }
    form->addRow(QStringLiteral("干扰组名称"), groupNameEdit_);
    form->addRow(QStringLiteral("模板"), templateCombo_);
    layout->addLayout(form);
    canvas_ = new AnnotationCanvas(this);
    layout->addWidget(canvas_);
    auto *deleteButton = new QPushButton(QStringLiteral("删除选中区域"), this);
    layout->addWidget(deleteButton);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttons);
    connect(templateCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AnnotationEditorDialog::loadSelectedImage);
    connect(canvas_, &AnnotationCanvas::regionsChanged, this, &AnnotationEditorDialog::markDirty);
    connect(groupNameEdit_, &QLineEdit::textChanged, this, &AnnotationEditorDialog::markDirty);
    connect(deleteButton, &QPushButton::clicked, this, &AnnotationEditorDialog::deleteSelectedRegion);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    const int initial = templateCombo_->findData(templateId);
    templateCombo_->setCurrentIndex(initial >= 0 ? initial : 0);
    loadSelectedImage();
}

void AnnotationEditorDialog::addTemplate(const QString &id, const QString &orientation, int index,
                                         const QString &path, const QJsonArray &regions) {
    templates_.append({id, orientation, index, path, regions});
    if (templateCombo_ != nullptr) {
        templateCombo_->addItem(orientation == QStringLiteral("front")
                                    ? QStringLiteral("正面 %1").arg(index + 1)
                                    : QStringLiteral("反面 %1").arg(index + 1), id);
    }
}

QString AnnotationEditorDialog::groupName() const { return groupNameEdit_->text().trimmed(); }

QString AnnotationEditorDialog::orientation() const {
    return currentEntry_ >= 0 && currentEntry_ < templates_.size() ? templates_.at(currentEntry_).orientation : QString();
}

int AnnotationEditorDialog::templateIndex() const {
    return currentEntry_ >= 0 && currentEntry_ < templates_.size() ? templates_.at(currentEntry_).index : -1;
}

QRectF AnnotationEditorDialog::nativeRegion() const {
    const QList<AnnotationRegionView> regions = canvas_->regions();
    return regions.isEmpty() ? QRectF() : regions.first().rect;
}

QJsonArray AnnotationEditorDialog::regionsToJson(const QList<AnnotationRegionView> &regions) const {
    QJsonArray values;
    for (const AnnotationRegionView &region : regions) {
        values.append(QJsonObject{{QStringLiteral("x"), region.rect.x()},
                                  {QStringLiteral("y"), region.rect.y()},
                                  {QStringLiteral("width"), region.rect.width()},
                                  {QStringLiteral("height"), region.rect.height()}});
    }
    return values;
}

QList<AnnotationRegionView> AnnotationEditorDialog::regionsFromJson(const QJsonArray &regions) const {
    QList<AnnotationRegionView> values;
    for (const QJsonValue &value : regions) {
        const QJsonObject item = value.toObject();
        values.append({QRectF(item.value(QStringLiteral("x")).toDouble(),
                              item.value(QStringLiteral("y")).toDouble(),
                              item.value(QStringLiteral("width")).toDouble(),
                              item.value(QStringLiteral("height")).toDouble()),
                       item.value(QStringLiteral("provenance")).toString(QStringLiteral("manual")),
                       item.value(QStringLiteral("status")).toString(QStringLiteral("active")), false});
    }
    return values;
}

QJsonObject AnnotationEditorDialog::annotationPatch() const {
    syncCurrentEntry();
    QJsonArray annotations;
    for (const TemplateEntry &entry : templates_) {
        if (entry.regions.isEmpty()) {
            continue;
        }
        annotations.append(QJsonObject{
            {QStringLiteral("template_id"), entry.id},
            {QStringLiteral("orientation"), entry.orientation},
            {QStringLiteral("index"), entry.index},
            {QStringLiteral("regions"), entry.regions},
            {QStringLiteral("trusted"), true},
        });
    }
    return {{QStringLiteral("group_id"), groupId_.isEmpty() ? groupName() : groupId_},
            {QStringLiteral("name"), groupName()},
            {QStringLiteral("annotations"), annotations}};
}

void AnnotationEditorDialog::setUnsavedPromptHandler(std::function<QMessageBox::StandardButton()> handler) {
    unsavedPromptHandler_ = std::move(handler);
}

bool AnnotationEditorDialog::confirmSwitch() {
    if (!unsavedChanges_) {
        return true;
    }
    const auto choice = unsavedPromptHandler_ ? unsavedPromptHandler_()
        : QMessageBox::question(this, QStringLiteral("未保存修改"), QStringLiteral("是否保存当前标注？"),
                                QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    if (choice == QMessageBox::Save) {
        syncCurrentEntry();
        unsavedChanges_ = false;
        return true;
    }
    if (choice == QMessageBox::Cancel) {
        return false;
    }
    unsavedChanges_ = false;
    return true;
}

void AnnotationEditorDialog::loadEntry(int index) {
    if (index < 0 || index >= templates_.size()) {
        canvas_->setImage(QImage());
        canvas_->setRegions({});
        currentEntry_ = -1;
        return;
    }
    currentEntry_ = index;
    const TemplateEntry &entry = templates_.at(index);
    canvas_->setImage(QImage(entry.path));
    canvas_->setRegions(regionsFromJson(entry.regions));
}

void AnnotationEditorDialog::loadSelectedImage() {
    const int index = templateCombo_->currentIndex();
    if (index == currentEntry_ && !loading_) {
        return;
    }
    const bool wasDirty = unsavedChanges_;
    if (!loading_ && !confirmSwitch()) {
        loading_ = true;
        templateCombo_->setCurrentIndex(currentEntry_ >= 0 ? currentEntry_ : index);
        loading_ = false;
        return;
    }
    if (!loading_ && !wasDirty) {
        syncCurrentEntry();
    }
    loading_ = true;
    loadEntry(index);
    loading_ = false;
}

void AnnotationEditorDialog::syncCurrentEntry() const {
    if (canvas_ == nullptr || currentEntry_ < 0 || currentEntry_ >= templates_.size()) {
        return;
    }
    templates_[currentEntry_].regions = regionsToJson(canvas_->regions());
}

void AnnotationEditorDialog::markDirty() {
    if (!loading_) {
        unsavedChanges_ = true;
    }
}

void AnnotationEditorDialog::deleteSelectedRegion() {
    canvas_->deleteSelectedRegion();
}
