#include "geometrymaskmanager.h"

#include "geometryrulecanvas.h"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QUuid>
#include <QtMath>

namespace {
QJsonObject emptyDirection() {
    return QJsonObject{{QStringLiteral("anchor"), QJsonValue()},
                       {QStringLiteral("rules"), QJsonArray()}};
}

QJsonArray rulesFor(const QJsonObject &direction) {
    return direction.value(QStringLiteral("rules")).toArray();
}
}

GeometryMaskManagerDialog::GeometryMaskManagerDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle(QStringLiteral("几何干扰规则管理"));
    resize(1100, 700);
    auto *root = new QVBoxLayout(this);
    auto *splitter = new QSplitter(Qt::Horizontal, this);
    root->addWidget(splitter);

    auto *left = new QWidget(splitter);
    auto *leftLayout = new QVBoxLayout(left);
    revisionLabel_ = new QLabel(left);
    statusLabel_ = new QLabel(left);
    leftLayout->addWidget(revisionLabel_);
    leftLayout->addWidget(statusLabel_);
    directionCombo_ = new QComboBox(left);
    directionCombo_->addItem(QStringLiteral("正面"), QStringLiteral("front"));
    directionCombo_->addItem(QStringLiteral("反面"), QStringLiteral("back"));
    leftLayout->addWidget(directionCombo_);
    templateCombo_ = new QComboBox(left);
    templateCombo_->setObjectName(QStringLiteral("templatePreviewCombo"));
    leftLayout->addWidget(templateCombo_);
    ruleList_ = new QListWidget(left);
    ruleList_->setObjectName(QStringLiteral("ruleList"));
    leftLayout->addWidget(ruleList_, 1);
    auto *ruleForm = new QFormLayout();
    ruleNameEdit_ = new QLineEdit(left);
    shapeCombo_ = new QComboBox(left);
    shapeCombo_->addItems({QStringLiteral("圆"), QStringLiteral("椭圆"), QStringLiteral("旋转矩形")});
    modeCombo_ = new QComboBox(left);
    modeCombo_->addItem(QStringLiteral("忽略内部"), QStringLiteral("inside"));
    modeCombo_->addItem(QStringLiteral("忽略外部"), QStringLiteral("outside"));
    marginSpin_ = new QSpinBox(left);
    marginSpin_->setRange(0, 94);
    marginSpin_->setValue(2);
    marginSpin_->setSuffix(QStringLiteral("%"));
    ruleForm->addRow(QStringLiteral("名称"), ruleNameEdit_);
    ruleForm->addRow(QStringLiteral("形状"), shapeCombo_);
    ruleForm->addRow(QStringLiteral("方向"), modeCombo_);
    ruleForm->addRow(QStringLiteral("安全边距"), marginSpin_);
    rotationSpin_ = new QDoubleSpinBox(left);
    rotationSpin_->setRange(-180.0, 180.0);
    rotationSpin_->setDecimals(1);
    rotationSpin_->setSuffix(QStringLiteral("°"));
    ruleForm->addRow(QStringLiteral("旋转角度"), rotationSpin_);
    leftLayout->addLayout(ruleForm);
    auto *ruleButtons = new QHBoxLayout();
    auto *addButton = new QPushButton(QStringLiteral("新增规则"), left);
    auto *deleteButton = new QPushButton(QStringLiteral("删除规则"), left);
    addButton->setObjectName(QStringLiteral("addRuleButton"));
    deleteButton->setObjectName(QStringLiteral("deleteRuleButton"));
    ruleButtons->addWidget(addButton);
    ruleButtons->addWidget(deleteButton);
    leftLayout->addLayout(ruleButtons);
    setAnchorButton_ = new QPushButton(QStringLiteral("将画布设为基准边界"), left);
    setAnchorButton_->setObjectName(QStringLiteral("setAnchorButton"));
    leftLayout->addWidget(setAnchorButton_);

    auto *center = new QWidget(splitter);
    auto *centerLayout = new QVBoxLayout(center);
    canvas_ = new GeometryRuleCanvas(center);
    canvas_->setObjectName(QStringLiteral("geometryRuleCanvas"));
    centerLayout->addWidget(canvas_, 1);
    auto *help = new QLabel(QStringLiteral("先选择方向和规则，再在画布中粗画；坐标按原图保存。"), center);
    help->setWordWrap(true);
    centerLayout->addWidget(help);

    auto *right = new QWidget(splitter);
    auto *rightLayout = new QVBoxLayout(right);
    progressBar_ = new QProgressBar(right);
    progressBar_->setRange(0, 1);
    diagnostics_ = new QTextEdit(right);
    diagnostics_->setReadOnly(true);
    rightLayout->addWidget(new QLabel(QStringLiteral("验证进度/诊断"), right));
    rightLayout->addWidget(progressBar_);
    rightLayout->addWidget(diagnostics_, 1);
    overrideReasonEdit_ = new QLineEdit(right);
    overrideReasonEdit_->setPlaceholderText(QStringLiteral("有告警或回归时填写覆盖原因"));
    rightLayout->addWidget(overrideReasonEdit_);

    auto *buttons = new QHBoxLayout();
    saveButton_ = new QPushButton(QStringLiteral("保存草稿"), this);
    validateButton_ = new QPushButton(QStringLiteral("验证草稿"), this);
    publishButton_ = new QPushButton(QStringLiteral("发布规则"), this);
    rollbackButton_ = new QPushButton(QStringLiteral("回退上一版本"), this);
    cancelButton_ = new QPushButton(QStringLiteral("关闭"), this);
    saveButton_->setObjectName(QStringLiteral("saveButton"));
    validateButton_->setObjectName(QStringLiteral("validateButton"));
    publishButton_->setObjectName(QStringLiteral("publishButton"));
    rollbackButton_->setObjectName(QStringLiteral("rollbackButton"));
    buttons->addWidget(saveButton_);
    buttons->addWidget(validateButton_);
    buttons->addWidget(publishButton_);
    buttons->addWidget(rollbackButton_);
    buttons->addWidget(cancelButton_);
    root->addLayout(buttons);

    connect(addButton, &QPushButton::clicked, this, &GeometryMaskManagerDialog::addRule);
    connect(deleteButton, &QPushButton::clicked, this, &GeometryMaskManagerDialog::deleteRule);
    connect(setAnchorButton_, &QPushButton::clicked, this, &GeometryMaskManagerDialog::setAnchorFromCanvas);
    connect(saveButton_, &QPushButton::clicked, this, &GeometryMaskManagerDialog::saveDraft);
    connect(validateButton_, &QPushButton::clicked, this, &GeometryMaskManagerDialog::validateDraft);
    connect(publishButton_, &QPushButton::clicked, this, &GeometryMaskManagerDialog::publishDraft);
    connect(rollbackButton_, &QPushButton::clicked, this, &GeometryMaskManagerDialog::rollbackDraft);
    connect(cancelButton_, &QPushButton::clicked, this, &QDialog::reject);
    connect(directionCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { refreshRuleList(); refreshTemplatePreview(); });
    connect(templateCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &GeometryMaskManagerDialog::refreshTemplatePreview);
    connect(ruleList_, &QListWidget::currentRowChanged, this, [this](int) { loadCurrentRuleIntoEditor(); });
    connect(overrideReasonEdit_, &QLineEdit::textChanged, this, &GeometryMaskManagerDialog::updatePublishState);
    connect(rotationSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double value) { canvas_->setRotationDegrees(value); });
    connect(canvas_, &GeometryRuleCanvas::shapeChanged, this, [this](const QJsonObject &shape) {
        const int index = currentRuleIndex();
        if (index < 0) return;
        QJsonObject directionValue = directionObject();
        QJsonArray rules = rulesFor(directionValue);
        QJsonObject rule = rules.at(index).toObject();
        QJsonObject geometry = ruleGeometryFromCanvasShape(shape);
        if (geometry.isEmpty()) return;
        rule.insert(QStringLiteral("geometry"), geometry);
        rules.replace(index, rule);
        directionValue.insert(QStringLiteral("rules"), rules);
        QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
        directions.insert(direction(), directionValue);
        draft_.insert(QStringLiteral("directions"), directions);
    });
    refreshRuleList();
    refreshTemplatePreview();
    updatePublishState();
}

QString GeometryMaskManagerDialog::direction() const {
    return directionCombo_ ? directionCombo_->currentData().toString() : QStringLiteral("front");
}

QJsonObject GeometryMaskManagerDialog::directionObject() const {
    const QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    return directions.value(direction()).toObject();
}

void GeometryMaskManagerDialog::ensureDirectionObject(const QString &name) {
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    if (!directions.contains(name)) directions.insert(name, emptyDirection());
    draft_.insert(QStringLiteral("directions"), directions);
}

int GeometryMaskManagerDialog::currentRuleIndex() const {
    return ruleList_ ? ruleList_->currentRow() : -1;
}

QJsonObject GeometryMaskManagerDialog::currentRule() const {
    const QJsonArray rules = rulesFor(directionObject());
    const int index = currentRuleIndex();
    return index >= 0 && index < rules.size() ? rules.at(index).toObject() : QJsonObject();
}

QJsonObject GeometryMaskManagerDialog::currentDirectionAnchor() const {
    return directionObject().value(QStringLiteral("anchor")).toObject();
}

QJsonObject GeometryMaskManagerDialog::anchorFromCanvasShape(const QJsonObject &shape) const {
    if (canvas_ == nullptr || canvas_->image().isNull()) return {};
    const qreal width = canvas_->image().width();
    const qreal height = canvas_->image().height();
    const QString type = shape.value(QStringLiteral("shape")).toString();
    QJsonObject coarse{{QStringLiteral("cx"), shape.value(QStringLiteral("cx")).toDouble() / width},
                       {QStringLiteral("cy"), shape.value(QStringLiteral("cy")).toDouble() / height},
                       {QStringLiteral("angle_deg"), shape.value(QStringLiteral("angle_deg")).toDouble()}};
    if (type == QStringLiteral("circle")) {
        coarse.insert(QStringLiteral("r"), shape.value(QStringLiteral("r")).toDouble() / qMin(width, height));
    } else if (type == QStringLiteral("ellipse")) {
        coarse.insert(QStringLiteral("rx"), shape.value(QStringLiteral("rx")).toDouble() / width);
        coarse.insert(QStringLiteral("ry"), shape.value(QStringLiteral("ry")).toDouble() / height);
    } else if (type == QStringLiteral("rotated_rectangle")) {
        coarse.insert(QStringLiteral("half_width"), shape.value(QStringLiteral("half_width")).toDouble() / width);
        coarse.insert(QStringLiteral("half_height"), shape.value(QStringLiteral("half_height")).toDouble() / height);
    } else {
        return {};
    }
    return QJsonObject{{QStringLiteral("shape"), type}, {QStringLiteral("coarse"), coarse}};
}

QJsonObject GeometryMaskManagerDialog::ruleGeometryFromCanvasShape(const QJsonObject &shape) const {
    if (canvas_ == nullptr || canvas_->image().isNull()) return {};
    const QJsonObject anchor = currentDirectionAnchor();
    const QJsonObject coarse = anchor.value(QStringLiteral("coarse")).toObject();
    if (anchor.isEmpty() || coarse.isEmpty()) return {};
    const qreal width = canvas_->image().width();
    const qreal height = canvas_->image().height();
    const QString anchorType = anchor.value(QStringLiteral("shape")).toString();
    const qreal anchorCx = coarse.value(QStringLiteral("cx")).toDouble() * width;
    const qreal anchorCy = coarse.value(QStringLiteral("cy")).toDouble() * height;
    const qreal anchorX = anchorType == QStringLiteral("rotated_rectangle")
        ? coarse.value(QStringLiteral("half_width")).toDouble() * width
        : anchorType == QStringLiteral("circle")
            ? coarse.value(QStringLiteral("r")).toDouble() * qMin(width, height)
            : coarse.value(QStringLiteral("rx")).toDouble() * width;
    const qreal anchorY = anchorType == QStringLiteral("rotated_rectangle")
        ? coarse.value(QStringLiteral("half_height")).toDouble() * height
        : anchorType == QStringLiteral("circle")
            ? anchorX
            : coarse.value(QStringLiteral("ry")).toDouble() * height;
    if (anchorX <= 0.0 || anchorY <= 0.0) return {};
    const qreal angle = qDegreesToRadians(coarse.value(QStringLiteral("angle_deg")).toDouble());
    const qreal dx = shape.value(QStringLiteral("cx")).toDouble() - anchorCx;
    const qreal dy = shape.value(QStringLiteral("cy")).toDouble() - anchorCy;
    const qreal localX = (dx * qCos(angle) + dy * qSin(angle)) / anchorX;
    const qreal localY = (-dx * qSin(angle) + dy * qCos(angle)) / anchorY;
    const qreal shapeAngle = shape.value(QStringLiteral("angle_deg")).toDouble()
        - coarse.value(QStringLiteral("angle_deg")).toDouble();
    QJsonObject geometry{{QStringLiteral("cx"), localX}, {QStringLiteral("cy"), localY},
                         {QStringLiteral("angle_deg"), shapeAngle}};
    const QString shapeType = shape.value(QStringLiteral("shape")).toString();
    if (shapeType == QStringLiteral("circle")) {
        geometry.insert(QStringLiteral("r"), shape.value(QStringLiteral("r")).toDouble() / qMin(anchorX, anchorY));
    } else if (shapeType == QStringLiteral("ellipse")) {
        geometry.insert(QStringLiteral("rx"), shape.value(QStringLiteral("rx")).toDouble() / anchorX);
        geometry.insert(QStringLiteral("ry"), shape.value(QStringLiteral("ry")).toDouble() / anchorY);
    } else if (shapeType == QStringLiteral("rotated_rectangle")) {
        geometry.insert(QStringLiteral("half_width"), shape.value(QStringLiteral("half_width")).toDouble() / anchorX);
        geometry.insert(QStringLiteral("half_height"), shape.value(QStringLiteral("half_height")).toDouble() / anchorY);
    } else {
        return {};
    }
    return geometry;
}

QJsonObject GeometryMaskManagerDialog::canvasShapeForRule(const QJsonObject &rule) const {
    if (canvas_ == nullptr || canvas_->image().isNull()) return {};
    const QJsonObject anchor = currentDirectionAnchor();
    const QJsonObject coarse = anchor.value(QStringLiteral("coarse")).toObject();
    const QJsonObject geometry = rule.value(QStringLiteral("geometry")).toObject();
    if (anchor.isEmpty() || coarse.isEmpty() || geometry.isEmpty()) return {};
    const qreal width = canvas_->image().width();
    const qreal height = canvas_->image().height();
    const QString anchorType = anchor.value(QStringLiteral("shape")).toString();
    const qreal ax = anchorType == QStringLiteral("rotated_rectangle") ? coarse.value(QStringLiteral("half_width")).toDouble() * width
        : anchorType == QStringLiteral("circle") ? coarse.value(QStringLiteral("r")).toDouble() * qMin(width, height)
        : coarse.value(QStringLiteral("rx")).toDouble() * width;
    const qreal ay = anchorType == QStringLiteral("rotated_rectangle") ? coarse.value(QStringLiteral("half_height")).toDouble() * height
        : anchorType == QStringLiteral("circle") ? ax
        : coarse.value(QStringLiteral("ry")).toDouble() * height;
    const qreal baseAngle = coarse.value(QStringLiteral("angle_deg")).toDouble();
    const qreal angle = qDegreesToRadians(baseAngle);
    const qreal localX = geometry.value(QStringLiteral("cx")).toDouble() * ax;
    const qreal localY = geometry.value(QStringLiteral("cy")).toDouble() * ay;
    QJsonObject shape{{QStringLiteral("shape"), rule.value(QStringLiteral("shape"))},
                      {QStringLiteral("cx"), coarse.value(QStringLiteral("cx")).toDouble() * width + localX * qCos(angle) - localY * qSin(angle)},
                      {QStringLiteral("cy"), coarse.value(QStringLiteral("cy")).toDouble() * height + localX * qSin(angle) + localY * qCos(angle)},
                      {QStringLiteral("angle_deg"), baseAngle + geometry.value(QStringLiteral("angle_deg")).toDouble()}};
    const QString type = rule.value(QStringLiteral("shape")).toString();
    if (type == QStringLiteral("circle")) shape.insert(QStringLiteral("r"), geometry.value(QStringLiteral("r")).toDouble() * qMin(ax, ay));
    else if (type == QStringLiteral("ellipse")) {
        shape.insert(QStringLiteral("rx"), geometry.value(QStringLiteral("rx")).toDouble() * ax);
        shape.insert(QStringLiteral("ry"), geometry.value(QStringLiteral("ry")).toDouble() * ay);
    } else if (type == QStringLiteral("rotated_rectangle")) {
        shape.insert(QStringLiteral("half_width"), geometry.value(QStringLiteral("half_width")).toDouble() * ax);
        shape.insert(QStringLiteral("half_height"), geometry.value(QStringLiteral("half_height")).toDouble() * ay);
    }
    return shape;
}

void GeometryMaskManagerDialog::setSnapshot(const QJsonObject &snapshot) {
    snapshot_ = snapshot;
    draft_ = snapshot.value(QStringLiteral("draft")).toObject();
    if (draft_.isEmpty()) {
        draft_ = QJsonObject{{QStringLiteral("schema_version"), 1},
                             {QStringLiteral("directions"), QJsonObject{{QStringLiteral("front"), emptyDirection()},
                                                                           {QStringLiteral("back"), emptyDirection()}}}};
    }
    ensureDirectionObject(QStringLiteral("front"));
    ensureDirectionObject(QStringLiteral("back"));
    job_ = QJsonObject();
    revisionLabel_->setText(QStringLiteral("库修订 %1，草稿修订 %2，活动 %3")
                                .arg(snapshot.value(QStringLiteral("library_revision")).toInt())
                                .arg(snapshot.value(QStringLiteral("draft_revision")).toInt())
                                .arg(snapshot.value(QStringLiteral("active_revision")).toInt(0)));
    statusLabel_->setText(snapshot.value(QStringLiteral("legacy_archived")).toBool()
                              ? QStringLiteral("旧位置标注已归档，请使用几何规则重新标定")
                              : QString());
    refreshRuleList();
    refreshTemplatePreview();
    updatePublishState();
}

void GeometryMaskManagerDialog::setValidationJob(const QJsonObject &job) {
    job_ = job;
    const QJsonObject progress = job.value(QStringLiteral("progress")).toObject();
    const int total = qMax(1, progress.value(QStringLiteral("total")).toInt(1));
    progressBar_->setRange(0, total);
    progressBar_->setValue(qBound(0, progress.value(QStringLiteral("completed")).toInt(), total));
    statusLabel_->setText(QStringLiteral("验证任务：%1").arg(job.value(QStringLiteral("state")).toString()));
    const QJsonArray warnings = job.value(QStringLiteral("warnings")).toArray();
    diagnostics_->setPlainText(warnings.isEmpty()
                                   ? QStringLiteral("暂无告警")
                                   : QString::fromUtf8(QJsonDocument(warnings).toJson(QJsonDocument::Indented)));
    updatePublishState();
}

void GeometryMaskManagerDialog::setBusy(bool busy) {
    busy_ = busy;
    saveButton_->setEnabled(!busy);
    validateButton_->setEnabled(!busy);
    rollbackButton_->setEnabled(!busy && snapshot_.value(QStringLiteral("previous_active_revision")).toInt(0) > 0);
    updatePublishState();
}

void GeometryMaskManagerDialog::setOperationError(const QString &message) {
    statusLabel_->setText(message);
    statusLabel_->setStyleSheet(QStringLiteral("color: #b00020;"));
}

void GeometryMaskManagerDialog::refreshRuleList() {
    if (!ruleList_) return;
    setCurrentRuleFromEditor();
    const int previous = ruleList_->currentRow();
    ruleList_->blockSignals(true);
    ruleList_->clear();
    const QJsonArray rules = rulesFor(directionObject());
    for (const QJsonValue &value : rules) {
        const QJsonObject rule = value.toObject();
        ruleList_->addItem(QStringLiteral("%1 [%2]").arg(rule.value(QStringLiteral("name")).toString(),
                                                          rule.value(QStringLiteral("enabled")).toBool(true)
                                                              ? QStringLiteral("启用") : QStringLiteral("停用")));
    }
    if (!rules.isEmpty()) ruleList_->setCurrentRow(qBound(0, previous, rules.size() - 1));
    ruleList_->blockSignals(false);
    loadCurrentRuleIntoEditor();
}

void GeometryMaskManagerDialog::refreshTemplatePreview() {
    if (!templateCombo_ || !canvas_) return;
    const QString side = direction();
    const QString previous = templateCombo_->currentData().toString();
    templateCombo_->blockSignals(true);
    templateCombo_->clear();
    const QJsonArray templates = snapshot_.value(QStringLiteral("templates")).toArray();
    for (const QJsonValue &value : templates) {
        const QJsonObject item = value.toObject();
        if (item.value(QStringLiteral("direction")).toString() != side) continue;
        templateCombo_->addItem(item.value(QStringLiteral("template_id")).toString(), item.value(QStringLiteral("path")));
    }
    int index = templateCombo_->findData(previous);
    if (index < 0) index = 0;
    if (templateCombo_->count() > 0) templateCombo_->setCurrentIndex(index);
    templateCombo_->blockSignals(false);
    const QString path = templateCombo_->currentData().toString();
    canvas_->setImage(path.isEmpty() ? QImage() : QImage(path));
    const QJsonObject rule = currentRule();
    if (!rule.isEmpty()) {
        canvas_->setCoarseShape(canvasShapeForRule(rule));
        rotationSpin_->setValue(rule.value(QStringLiteral("geometry")).toObject().value(QStringLiteral("angle_deg")).toDouble());
    }
}

void GeometryMaskManagerDialog::setAnchorFromCanvas() {
    const QJsonObject anchor = anchorFromCanvasShape(canvas_->coarseShape());
    if (anchor.isEmpty()) return;
    ensureDirectionObject(direction());
    QJsonObject side = directionObject();
    side.insert(QStringLiteral("anchor"), anchor);
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    directions.insert(direction(), side);
    draft_.insert(QStringLiteral("directions"), directions);
    refreshRuleList();
}

void GeometryMaskManagerDialog::setCurrentRuleFromEditor() {
    const int index = currentRuleIndex();
    if (index < 0) return;
    QJsonObject directionValue = directionObject();
    QJsonArray rules = rulesFor(directionValue);
    if (index >= rules.size()) return;
    QJsonObject rule = rules.at(index).toObject();
    rule.insert(QStringLiteral("name"), ruleNameEdit_->text().trimmed());
    rule.insert(QStringLiteral("shape"), shapeCombo_->currentIndex() == 0 ? QStringLiteral("circle")
                : shapeCombo_->currentIndex() == 1 ? QStringLiteral("ellipse") : QStringLiteral("rotated_rectangle"));
    rule.insert(QStringLiteral("mode"), modeCombo_->currentData().toString());
    rule.insert(QStringLiteral("margin_ratio"), marginSpin_->value() / 100.0);
    rules.replace(index, rule);
    directionValue.insert(QStringLiteral("rules"), rules);
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    directions.insert(direction(), directionValue);
    draft_.insert(QStringLiteral("directions"), directions);
}

void GeometryMaskManagerDialog::loadCurrentRuleIntoEditor() {
    const QJsonObject rule = currentRule();
    if (rule.isEmpty()) return;
    ruleNameEdit_->setText(rule.value(QStringLiteral("name")).toString());
    const QString shape = rule.value(QStringLiteral("shape")).toString();
    shapeCombo_->setCurrentIndex(shape == QStringLiteral("ellipse") ? 1 : shape == QStringLiteral("rotated_rectangle") ? 2 : 0);
    modeCombo_->setCurrentIndex(modeCombo_->findData(rule.value(QStringLiteral("mode")).toString()));
    marginSpin_->setValue(qRound(rule.value(QStringLiteral("margin_ratio")).toDouble(0.02) * 100.0));
    rotationSpin_->setValue(rule.value(QStringLiteral("geometry")).toObject().value(QStringLiteral("angle_deg")).toDouble());
    refreshTemplatePreview();
}

void GeometryMaskManagerDialog::addRule() {
    ensureDirectionObject(direction());
    QJsonObject directionValue = directionObject();
    QJsonArray rules = rulesFor(directionValue);
    if (directionValue.value(QStringLiteral("anchor")).isNull()) {
        directionValue.insert(QStringLiteral("anchor"), QJsonObject{{QStringLiteral("shape"), QStringLiteral("ellipse")},
                                                                      {QStringLiteral("coarse"), QJsonObject{{QStringLiteral("cx"), 0.5}, {QStringLiteral("cy"), 0.5},
                                                                                                               {QStringLiteral("rx"), 0.4}, {QStringLiteral("ry"), 0.4}}}});
    }
    rules.append(QJsonObject{{QStringLiteral("rule_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
                             {QStringLiteral("name"), QStringLiteral("新规则")},
                             {QStringLiteral("shape"), QStringLiteral("circle")},
                             {QStringLiteral("geometry"), QJsonObject{{QStringLiteral("cx"), 0.0}, {QStringLiteral("cy"), 0.0}, {QStringLiteral("r"), 0.5}}},
                             {QStringLiteral("mode"), QStringLiteral("inside")},
                             {QStringLiteral("margin_ratio"), 0.02}, {QStringLiteral("enabled"), true}});
    directionValue.insert(QStringLiteral("rules"), rules);
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    directions.insert(direction(), directionValue);
    draft_.insert(QStringLiteral("directions"), directions);
    refreshRuleList();
    ruleList_->setCurrentRow(rules.size() - 1);
}

void GeometryMaskManagerDialog::deleteRule() {
    setCurrentRuleFromEditor();
    const int index = currentRuleIndex();
    if (index < 0) return;
    QJsonObject directionValue = directionObject();
    QJsonArray rules = rulesFor(directionValue);
    rules.removeAt(index);
    directionValue.insert(QStringLiteral("rules"), rules);
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    directions.insert(direction(), directionValue);
    draft_.insert(QStringLiteral("directions"), directions);
    refreshRuleList();
}

void GeometryMaskManagerDialog::saveDraft() {
    setCurrentRuleFromEditor();
    emit saveDraftRequested(draft_, snapshot_.value(QStringLiteral("library_revision")).toInt(),
                             snapshot_.value(QStringLiteral("draft_revision")).toInt());
}

void GeometryMaskManagerDialog::validateDraft() {
    setCurrentRuleFromEditor();
    emit validateRequested(snapshot_.value(QStringLiteral("library_revision")).toInt(),
                          snapshot_.value(QStringLiteral("draft_revision")).toInt());
}

void GeometryMaskManagerDialog::publishDraft() {
    if (!publishButton_->isEnabled()) return;
    emit publishRequested(job_.value(QStringLiteral("job_id")).toString(),
                          snapshot_.value(QStringLiteral("library_revision")).toInt(),
                          snapshot_.value(QStringLiteral("draft_revision")).toInt(),
                          overrideReasonEdit_->text().trimmed());
}

void GeometryMaskManagerDialog::rollbackDraft() {
    if (busy_) return;
    emit rollbackRequested(snapshot_.value(QStringLiteral("library_revision")).toInt());
}

void GeometryMaskManagerDialog::updatePublishState() {
    const QString state = job_.value(QStringLiteral("state")).toString();
    const bool completed = state == QStringLiteral("completed")
        && job_.value(QStringLiteral("base_library_revision")).toInt() == snapshot_.value(QStringLiteral("library_revision")).toInt()
        && job_.value(QStringLiteral("base_draft_revision")).toInt() == snapshot_.value(QStringLiteral("draft_revision")).toInt();
    const bool hasWarning = !job_.value(QStringLiteral("warnings")).toArray().isEmpty()
        || job_.value(QStringLiteral("regression")).toObject().value(QStringLiteral("correct_to_wrong")).toInt() > 0;
    publishButton_->setEnabled(!busy_ && completed && (!hasWarning || !overrideReasonEdit_->text().trimmed().isEmpty()));
}
