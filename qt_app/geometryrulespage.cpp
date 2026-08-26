#include "geometryrulespage.h"

#include "geometryrulecanvas.h"

#include <QComboBox>
#include <QColor>
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
#include <QTableWidget>
#include <QAbstractItemView>
#include <QSignalBlocker>
#include <QSet>
#include <QStringList>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QUuid>
#include <QtMath>

namespace {
QJsonObject emptyDirection() {
    return QJsonObject{{QStringLiteral("anchor"), QJsonValue()},
                       {QStringLiteral("rules"), QJsonArray()}};
}

QJsonObject emptyLogicalDirection() {
    return QJsonObject{{QStringLiteral("anchor"), QJsonValue()},
                       {QStringLiteral("calibrations"), QJsonObject()},
                       {QStringLiteral("template_reviews"), QJsonObject()}};
}

QJsonArray rulesFor(const QJsonObject &direction) {
    return direction.value(QStringLiteral("rules")).toArray();
}

QJsonObject rulePreviewSignature(const QJsonObject &rule) {
    QJsonObject signature;
    for (const QString &key : {QStringLiteral("rule_id"), QStringLiteral("shape"),
                               QStringLiteral("mode"), QStringLiteral("margin_ratio"),
                               QStringLiteral("geometry"), QStringLiteral("seed_geometry")}) {
        if (rule.contains(key)) signature.insert(key, rule.value(key));
    }
    return signature;
}

QString orientationName(const QString &value) {
    if (value == QStringLiteral("front")) return QStringLiteral("正面");
    if (value == QStringLiteral("back")) return QStringLiteral("反面");
    if (value == QStringLiteral("uncertain")) return QStringLiteral("不确定");
    return value.isEmpty() ? QStringLiteral("未知") : value;
}
}

GeometryRulesPage::GeometryRulesPage(QWidget *parent) : QWidget(parent) {
    auto *root = new QVBoxLayout(this);
    auto *splitter = new QSplitter(Qt::Horizontal, this);
    root->addWidget(splitter);

    auto *left = new QWidget(splitter);
    auto *leftLayout = new QVBoxLayout(left);
    revisionLabel_ = new QLabel(left);
    statusLabel_ = new QLabel(left);
    statusLabel_->setObjectName(QStringLiteral("geometryStatusLabel"));
    leftLayout->addWidget(revisionLabel_);
    leftLayout->addWidget(statusLabel_);
    directionCombo_ = new QComboBox(left);
    directionCombo_->setObjectName(QStringLiteral("directionCombo"));
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
    shapeCombo_->setObjectName(QStringLiteral("shapeCombo"));
    shapeCombo_->addItem(QStringLiteral("选择形状"), QString());
    shapeCombo_->addItem(QStringLiteral("圆"), QStringLiteral("circle"));
    shapeCombo_->addItem(QStringLiteral("椭圆"), QStringLiteral("ellipse"));
    shapeCombo_->addItem(QStringLiteral("旋转矩形"), QStringLiteral("rotated_rectangle"));
    modeCombo_ = new QComboBox(left);
    modeCombo_->addItem(QStringLiteral("忽略内部"), QStringLiteral("inside"));
    modeCombo_->addItem(QStringLiteral("忽略外部"), QStringLiteral("outside"));
    marginSpin_ = new QSpinBox(left);
    marginSpin_->setObjectName(QStringLiteral("marginSpinBox"));
    marginSpin_->setRange(-94, 94);
    marginSpin_->setValue(0);
    marginSpin_->setSuffix(QStringLiteral("%"));
    marginSpin_->setToolTip(QStringLiteral("负值向内收缩，正值向外扩张；仅影响最终忽略边界。"));
    ruleForm->addRow(QStringLiteral("名称"), ruleNameEdit_);
    ruleForm->addRow(QStringLiteral("形状"), shapeCombo_);
    ruleForm->addRow(QStringLiteral("方向"), modeCombo_);
    ruleForm->addRow(QStringLiteral("边界偏移"), marginSpin_);
    rotationSpin_ = new QDoubleSpinBox(left);
    rotationSpin_->setRange(-180.0, 180.0);
    rotationSpin_->setDecimals(1);
    rotationSpin_->setSuffix(QStringLiteral("°"));
    ruleForm->addRow(QStringLiteral("旋转角度"), rotationSpin_);
    anchorCandidateCombo_ = new QComboBox(left);
    anchorCandidateCombo_->setObjectName(QStringLiteral("anchorCandidateCombo"));
    ruleForm->addRow(QStringLiteral("基准候选"), anchorCandidateCombo_);
    ruleCandidateCombo_ = new QComboBox(left);
    ruleCandidateCombo_->setObjectName(QStringLiteral("ruleCandidateCombo"));
    ruleForm->addRow(QStringLiteral("规则候选"), ruleCandidateCombo_);
    reviewStateCombo_ = new QComboBox(left);
    reviewStateCombo_->setObjectName(QStringLiteral("reviewStateCombo"));
    reviewStateCombo_->addItem(QStringLiteral("纳入"), QStringLiteral("included"));
    reviewStateCombo_->addItem(QStringLiteral("待复核"), QStringLiteral("review"));
    reviewStateCombo_->addItem(QStringLiteral("排除"), QStringLiteral("excluded"));
    reviewReasonEdit_ = new QLineEdit(left);
    reviewReasonEdit_->setObjectName(QStringLiteral("reviewReasonEdit"));
    ruleForm->addRow(QStringLiteral("模板复核"), reviewStateCombo_);
    ruleForm->addRow(QStringLiteral("复核原因"), reviewReasonEdit_);
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
    setAnchorButton_->setVisible(false);
    setAnchorButton_->setObjectName(QStringLiteral("setAnchorButton"));
    leftLayout->addWidget(setAnchorButton_);
    manualAnchorButton_ = new QPushButton(QStringLiteral("手动指定基准边界"), left);
    manualAnchorButton_->setObjectName(QStringLiteral("manualAnchorButton"));
    manualAnchorButton_->setVisible(false);
    leftLayout->addWidget(manualAnchorButton_);

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
    validationHintLabel_ = new QLabel(QStringLiteral("点击表格中的模板行可定位预览；低置信度/未配置模板请在左侧选择“排除”并填写原因，再重新验证。"), right);
    validationHintLabel_->setObjectName(QStringLiteral("validationHintLabel"));
    validationHintLabel_->setWordWrap(true);
    validationHintLabel_->setStyleSheet(QStringLiteral("color: #7a4b00;"));
    rightLayout->addWidget(validationHintLabel_);
    validationTable_ = new QTableWidget(right);
    validationTable_->setObjectName(QStringLiteral("validationTable"));
    validationTable_->setColumnCount(4);
    validationTable_->setHorizontalHeaderLabels({QStringLiteral("方向"), QStringLiteral("模板"),
                                                  QStringLiteral("状态"), QStringLiteral("待处理规则")});
    validationTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    validationTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    validationTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    validationTable_->setCursor(Qt::PointingHandCursor);
    rightLayout->addWidget(validationTable_, 1);

    auto *migrationForm = new QFormLayout();
    migrationConflictCombo_ = new QComboBox(right);
    migrationConflictCombo_->setObjectName(QStringLiteral("migrationConflictCombo"));
    migrationActionCombo_ = new QComboBox(right);
    migrationActionCombo_->setObjectName(QStringLiteral("migrationActionCombo"));
    migrationActionCombo_->addItem(QStringLiteral("仅保留所选来源"), QStringLiteral("keep_only"));
    migrationActionCombo_->addItem(QStringLiteral("正反来源配成一条规则"), QStringLiteral("pair"));
    migrationActionCombo_->addItem(QStringLiteral("分别保留"), QStringLiteral("keep_separate"));
    migrationSurvivorCombo_ = new QComboBox(right);
    migrationSurvivorCombo_->setObjectName(QStringLiteral("migrationSurvivorCombo"));
    migrationFrontCombo_ = new QComboBox(right);
    migrationFrontCombo_->setObjectName(QStringLiteral("migrationFrontCombo"));
    migrationBackCombo_ = new QComboBox(right);
    migrationBackCombo_->setObjectName(QStringLiteral("migrationBackCombo"));
    resolveMigrationButton_ = new QPushButton(QStringLiteral("应用迁移处置"), right);
    resolveMigrationButton_->setObjectName(QStringLiteral("resolveMigrationButton"));
    migrationForm->addRow(QStringLiteral("迁移冲突"), migrationConflictCombo_);
    migrationForm->addRow(QStringLiteral("处置方式"), migrationActionCombo_);
    migrationForm->addRow(QStringLiteral("保留来源"), migrationSurvivorCombo_);
    migrationForm->addRow(QStringLiteral("正面来源"), migrationFrontCombo_);
    migrationForm->addRow(QStringLiteral("反面来源"), migrationBackCombo_);
    rightLayout->addLayout(migrationForm);
    rightLayout->addWidget(resolveMigrationButton_);
    auto *overrideReasonLabel = new QLabel(QStringLiteral("发布覆盖原因（有告警或回归时必填）"), right);
    overrideReasonLabel->setObjectName(QStringLiteral("overrideReasonLabel"));
    rightLayout->addWidget(overrideReasonLabel);
    overrideReasonEdit_ = new QLineEdit(right);
    overrideReasonEdit_->setObjectName(QStringLiteral("overrideReasonEdit"));
    overrideReasonEdit_->setPlaceholderText(QStringLiteral("请说明已检查告警/回归并确认发布"));
    rightLayout->addWidget(overrideReasonEdit_);

    auto *buttons = new QHBoxLayout();
    saveButton_ = new QPushButton(QStringLiteral("保存草稿"), this);
    validateButton_ = new QPushButton(QStringLiteral("验证草稿"), this);
    publishButton_ = new QPushButton(QStringLiteral("发布规则"), this);
    publishWorkflowButton_ = new QPushButton(QStringLiteral("保存、验证并发布"), this);
    rollbackButton_ = new QPushButton(QStringLiteral("回退上一版本"), this);
    saveButton_->setObjectName(QStringLiteral("saveButton"));
    validateButton_->setObjectName(QStringLiteral("validateButton"));
    publishButton_->setObjectName(QStringLiteral("publishButton"));
    publishButton_->setVisible(false);
    publishWorkflowButton_->setObjectName(QStringLiteral("publishWorkflowButton"));
    rollbackButton_->setObjectName(QStringLiteral("rollbackButton"));
    buttons->addWidget(saveButton_);
    buttons->addWidget(validateButton_);
    buttons->addWidget(publishButton_);
    buttons->addWidget(publishWorkflowButton_);
    buttons->addWidget(rollbackButton_);
    root->addLayout(buttons);

    auto *editButtons = new QHBoxLayout();
    undoButton_ = new QPushButton(QStringLiteral("撤销"), left);
    undoButton_->setObjectName(QStringLiteral("undoButton"));
    redoButton_ = new QPushButton(QStringLiteral("重做"), left);
    redoButton_->setObjectName(QStringLiteral("redoButton"));
    resetRuleButton_ = new QPushButton(QStringLiteral("重置规则"), left);
    resetRuleButton_->setObjectName(QStringLiteral("resetRuleButton"));
    reloadDraftButton_ = new QPushButton(QStringLiteral("重新载入草稿"), left);
    reloadDraftButton_->setObjectName(QStringLiteral("reloadDraftButton"));
    editButtons->addWidget(undoButton_);
    editButtons->addWidget(redoButton_);
    editButtons->addWidget(resetRuleButton_);
    editButtons->addWidget(reloadDraftButton_);
    leftLayout->addLayout(editButtons);
    dirtyLabel_ = new QLabel(left);
    dirtyLabel_->setObjectName(QStringLiteral("dirtyLabel"));
    leftLayout->addWidget(dirtyLabel_);
    versionCombo_ = new QComboBox(left);
    versionCombo_->setObjectName(QStringLiteral("versionCombo"));
    versionCombo_->addItem(QStringLiteral("编辑草稿"), QStringLiteral("draft"));
    versionCombo_->addItem(QStringLiteral("活动版本（只读）"), QStringLiteral("active"));
    copyActiveToDraftButton_ = new QPushButton(QStringLiteral("复制活动版本到草稿"), left);
    copyActiveToDraftButton_->setObjectName(QStringLiteral("copyActiveToDraftButton"));
    leftLayout->addWidget(versionCombo_);
    leftLayout->addWidget(copyActiveToDraftButton_);

    connect(addButton, &QPushButton::clicked, this, &GeometryRulesPage::addRule);
    connect(deleteButton, &QPushButton::clicked, this, &GeometryRulesPage::deleteRule);
    connect(setAnchorButton_, &QPushButton::clicked, this, &GeometryRulesPage::setAnchorFromCanvas);
    connect(saveButton_, &QPushButton::clicked, this, &GeometryRulesPage::saveDraft);
    connect(validateButton_, &QPushButton::clicked, this, &GeometryRulesPage::validateDraft);
    connect(publishButton_, &QPushButton::clicked, this, &GeometryRulesPage::publishDraft);
    connect(publishWorkflowButton_, &QPushButton::clicked, this, &GeometryRulesPage::publishWorkflow);
    connect(resolveMigrationButton_, &QPushButton::clicked, this,
            &GeometryRulesPage::resolveMigrationConflict);
    connect(migrationConflictCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { refreshMigrationPanel(); });
    connect(migrationActionCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { refreshMigrationPanel(); });
    connect(rollbackButton_, &QPushButton::clicked, this, &GeometryRulesPage::rollbackDraft);
    connect(undoButton_, &QPushButton::clicked, this, &GeometryRulesPage::undoDraft);
    connect(redoButton_, &QPushButton::clicked, this, &GeometryRulesPage::redoDraft);
    connect(resetRuleButton_, &QPushButton::clicked, this, &GeometryRulesPage::resetCurrentRule);
    connect(reloadDraftButton_, &QPushButton::clicked, this, &GeometryRulesPage::reloadDraft);
    connect(copyActiveToDraftButton_, &QPushButton::clicked, this, &GeometryRulesPage::copyActiveToDraft);
    connect(directionCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) {
                refreshRuleList();
                refreshTemplatePreview();
                if (!suppressPreviewRequests_) requestPreviewFromShape(canvasShapeForRule(currentRule()));
            });
    connect(templateCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) {
                refreshTemplatePreview();
                if (!suppressPreviewRequests_) requestPreviewFromShape(canvasShapeForRule(currentRule()));
            });
    connect(ruleList_, &QListWidget::currentRowChanged, this, [this](int) { loadCurrentRuleIntoEditor(); });
    connect(validationTable_, &QTableWidget::cellClicked, this,
            [this](int row, int) { selectValidationTemplate(row); });
    connect(validationTable_, &QTableWidget::currentCellChanged, this,
            [this](int currentRow, int, int, int) {
                if (currentRow >= 0) selectValidationTemplate(currentRow);
            });
    connect(shapeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) {
                const QString shape = shapeCombo_->itemData(shapeCombo_->currentIndex()).toString();
                canvas_->setTool(shape == QStringLiteral("circle") ? GeometryRuleCanvas::Circle
                                   : shape == QStringLiteral("ellipse") ? GeometryRuleCanvas::Ellipse
                                                                          : shape == QStringLiteral("rotated_rectangle")
                                                                                 ? GeometryRuleCanvas::RotatedRectangle
                                                                                 : GeometryRuleCanvas::None);
                setCurrentRuleFromEditor();
            });
    connect(ruleNameEdit_, &QLineEdit::textChanged, this,
            [this](const QString &) { setCurrentRuleFromEditor(); });
    connect(modeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { setCurrentRuleFromEditor(); });
    connect(marginSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { setCurrentRuleFromEditor(); });
    connect(anchorCandidateCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) { Q_UNUSED(index); candidateChanged(anchorCandidateCombo_->currentIndex()); });
    connect(ruleCandidateCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) { candidateChanged(index); });
    connect(reviewStateCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { setTemplateReview(reviewStateCombo_->currentData().toString(), reviewReasonEdit_->text()); });
    connect(reviewReasonEdit_, &QLineEdit::textChanged, this,
            [this](const QString &text) { setTemplateReview(reviewStateCombo_->currentData().toString(), text); });
    connect(versionCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this, addButton, deleteButton](int index) {
                const bool readOnly = index == 1;
                for (QWidget *widget : {static_cast<QWidget *>(shapeCombo_), static_cast<QWidget *>(modeCombo_),
                                        static_cast<QWidget *>(ruleNameEdit_), static_cast<QWidget *>(marginSpin_),
                                        static_cast<QWidget *>(rotationSpin_), static_cast<QWidget *>(addButton),
                                        static_cast<QWidget *>(deleteButton),
                                        static_cast<QWidget *>(canvas_)}) {
                    if (widget) widget->setEnabled(!readOnly);
                }
                copyActiveToDraftButton_->setEnabled(readOnly);
            });
    connect(overrideReasonEdit_, &QLineEdit::textChanged, this, &GeometryRulesPage::updatePublishState);
    connect(rotationSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double value) { canvas_->setRotationDegrees(value); });
    connect(canvas_, &GeometryRuleCanvas::shapeChanged, this, [this](const QJsonObject &shape) {
        const int index = currentRuleIndex();
        if (index < 0) return;
        QJsonObject geometry = ruleGeometryFromCanvasShape(shape);
        if (geometry.isEmpty()) return;
        if (usesLogicalRuleSchema()) {
            QJsonObject calibration = currentCalibration();
            calibration.insert(QStringLiteral("seed_geometry"), geometry);
            calibration.insert(QStringLiteral("state"), QStringLiteral("needs_review"));
            setCurrentCalibration(calibration);
            manualEditContextKey_ = currentEditContextKey();
            return;
        }
        QJsonObject directionValue = directionObject();
        QJsonArray rules = rulesFor(directionValue);
        QJsonObject rule = rules.at(index).toObject();
        rule.insert(QStringLiteral("geometry"), geometry);
        rules.replace(index, rule);
        directionValue.insert(QStringLiteral("rules"), rules);
        QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
        directions.insert(direction(), directionValue);
        draft_.insert(QStringLiteral("directions"), directions);
        markDraftDirty();
    });
    connect(canvas_, &GeometryRuleCanvas::shapeCommitted, this,
            &GeometryRulesPage::requestPreviewFromShape);
    connect(manualAnchorButton_, &QPushButton::clicked, this, [this]() {
        manualAnchorCapture_ = true;
        manualAnchorButton_->setText(QStringLiteral("请在画布中绘制基准边界"));
        canvas_->setTool(GeometryRuleCanvas::Circle);
    });
    refreshRuleList();
    refreshTemplatePreview();
    updatePublishState();
}

QString GeometryRulesPage::direction() const {
    return directionCombo_ ? directionCombo_->currentData().toString() : QStringLiteral("front");
}

QString GeometryRulesPage::currentTemplateId() const {
    return templateCombo_ ? templateCombo_->currentData(Qt::UserRole + 1).toString() : QString();
}

QString GeometryRulesPage::currentEditContextKey() const {
    return QStringLiteral("%1|%2|%3|%4")
        .arg(currentRuleId(), direction(), currentTemplateId())
        .arg(snapshot_.value(QStringLiteral("draft_revision")).toInt());
}

QJsonObject GeometryRulesPage::directionObject() const {
    const QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    return directions.value(direction()).toObject();
}

bool GeometryRulesPage::usesLogicalRuleSchema() const {
    return draft_.value(QStringLiteral("schema_version")).toInt(1) == 2;
}

QJsonArray GeometryRulesPage::logicalRules() const {
    return usesLogicalRuleSchema()
        ? draft_.value(QStringLiteral("rules")).toArray()
        : rulesFor(directionObject());
}

QString GeometryRulesPage::currentRuleId() const {
    if (ruleList_ != nullptr && ruleList_->currentItem() != nullptr) {
        const QString stored = ruleList_->currentItem()->data(Qt::UserRole).toString();
        if (!stored.isEmpty()) return stored;
    }
    const QJsonArray rules = logicalRules();
    const int index = currentRuleIndex();
    return index >= 0 && index < rules.size()
        ? rules.at(index).toObject().value(QStringLiteral("rule_id")).toString()
        : QString();
}

QJsonObject GeometryRulesPage::currentCalibration() const {
    if (!usesLogicalRuleSchema()) return {};
    return directionObject().value(QStringLiteral("calibrations")).toObject()
        .value(currentRuleId()).toObject();
}

void GeometryRulesPage::setCurrentCalibration(const QJsonObject &calibration) {
    if (!usesLogicalRuleSchema()) return;
    const QString ruleId = currentRuleId();
    if (ruleId.isEmpty()) return;
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    QJsonObject side = directions.value(direction()).toObject();
    QJsonObject calibrations = side.value(QStringLiteral("calibrations")).toObject();
    calibrations.insert(ruleId, calibration);
    side.insert(QStringLiteral("calibrations"), calibrations);
    directions.insert(direction(), side);
    QJsonObject next = draft_;
    next.insert(QStringLiteral("directions"), directions);
    if (next == draft_) return;
    draft_ = next;
    markDraftDirty();
}

QString GeometryRulesPage::calibrationState(const QString &sideName, const QString &ruleId) const {
    const QJsonObject side = draft_.value(QStringLiteral("directions")).toObject()
        .value(sideName).toObject();
    const QJsonObject calibration = side.value(QStringLiteral("calibrations")).toObject()
        .value(ruleId).toObject();
    const QString state = calibration.value(QStringLiteral("state")).toString(QStringLiteral("missing"));
    if (state == QStringLiteral("ready")) return QStringLiteral("已标定");
    if (state == QStringLiteral("low_confidence")) return QStringLiteral("低置信度");
    if (state == QStringLiteral("needs_review")) return QStringLiteral("待复核");
    return QStringLiteral("缺失");
}

GeometryRulesPage::MissingCalibration
GeometryRulesPage::firstMissingEnabledCalibration() const {
    if (!usesLogicalRuleSchema()) return {};
    const QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    for (const QJsonValue &value : logicalRules()) {
        const QJsonObject rule = value.toObject();
        if (!rule.value(QStringLiteral("enabled")).toBool(true)) continue;
        const QString ruleId = rule.value(QStringLiteral("rule_id")).toString();
        for (const QString &sideName : {QStringLiteral("front"), QStringLiteral("back")}) {
            const QJsonObject calibration = directions.value(sideName).toObject()
                .value(QStringLiteral("calibrations")).toObject().value(ruleId).toObject();
            if (calibration.value(QStringLiteral("state")).toString() != QStringLiteral("ready")
                || calibration.value(QStringLiteral("geometry")).toObject().isEmpty()) {
                return MissingCalibration{true, ruleId, sideName};
            }
        }
    }
    return {};
}

void GeometryRulesPage::selectRuleById(const QString &ruleId) {
    if (ruleList_ == nullptr) return;
    for (int row = 0; row < ruleList_->count(); ++row) {
        if (ruleList_->item(row)->data(Qt::UserRole).toString() == ruleId) {
            ruleList_->setCurrentRow(row);
            return;
        }
    }
}

QJsonObject GeometryRulesPage::selectedMigrationConflict() const {
    if (migrationConflictCombo_ == nullptr) return {};
    const QString conflictId = migrationConflictCombo_->currentData().toString();
    for (const QJsonValue &value : draft_.value(QStringLiteral("migration")).toObject()
                                         .value(QStringLiteral("conflicts")).toArray()) {
        const QJsonObject conflict = value.toObject();
        if (conflict.value(QStringLiteral("conflict_id")).toString() == conflictId) return conflict;
    }
    return {};
}

void GeometryRulesPage::refreshMigrationPanel() {
    if (migrationConflictCombo_ == nullptr || migrationActionCombo_ == nullptr) return;
    const QString previousConflict = migrationConflictCombo_->currentData().toString();
    const QString previousAction = migrationActionCombo_->currentData().toString();
    const QSignalBlocker conflictBlocker(migrationConflictCombo_);
    const QSignalBlocker actionBlocker(migrationActionCombo_);
    const QSignalBlocker survivorBlocker(migrationSurvivorCombo_);
    const QSignalBlocker frontBlocker(migrationFrontCombo_);
    const QSignalBlocker backBlocker(migrationBackCombo_);
    migrationConflictCombo_->clear();
    const QJsonArray conflicts = draft_.value(QStringLiteral("migration")).toObject()
                                      .value(QStringLiteral("conflicts")).toArray();
    for (const QJsonValue &value : conflicts) {
        const QJsonObject conflict = value.toObject();
        const QString id = conflict.value(QStringLiteral("conflict_id")).toString();
        const QString type = conflict.value(QStringLiteral("type")).toString();
        migrationConflictCombo_->addItem(QStringLiteral("%1 · %2").arg(type, id), id);
    }
    int conflictIndex = migrationConflictCombo_->findData(previousConflict);
    if (conflictIndex < 0) conflictIndex = conflicts.isEmpty() ? -1 : 0;
    migrationConflictCombo_->setCurrentIndex(conflictIndex);
    int actionIndex = migrationActionCombo_->findData(previousAction);
    if (actionIndex < 0) actionIndex = 0;
    migrationActionCombo_->setCurrentIndex(actionIndex);

    migrationSurvivorCombo_->clear();
    migrationFrontCombo_->clear();
    migrationBackCombo_->clear();
    const QJsonObject conflict = selectedMigrationConflict();
    for (const QJsonValue &value : conflict.value(QStringLiteral("sources")).toArray()) {
        const QJsonObject source = value.toObject();
        const QString sourceId = source.value(QStringLiteral("source_rule_id")).toString();
        const QString logicalId = source.value(QStringLiteral("logical_rule_id")).toString();
        const QString side = source.value(QStringLiteral("direction")).toString();
        const QString label = QStringLiteral("%1 · %2 · %3")
            .arg(side, source.value(QStringLiteral("name")).toString(), sourceId);
        migrationSurvivorCombo_->addItem(label, sourceId);
        if (side == QStringLiteral("front")) migrationFrontCombo_->addItem(label, logicalId);
        if (side == QStringLiteral("back")) migrationBackCombo_->addItem(label, logicalId);
    }
    const QString action = migrationActionCombo_->currentData().toString();
    migrationSurvivorCombo_->setVisible(action == QStringLiteral("keep_only"));
    migrationFrontCombo_->setVisible(action == QStringLiteral("pair"));
    migrationBackCombo_->setVisible(action == QStringLiteral("pair"));
    const bool hasConflict = !conflict.isEmpty();
    migrationActionCombo_->setEnabled(hasConflict);
    resolveMigrationButton_->setEnabled(hasConflict);
}

void GeometryRulesPage::resolveMigrationConflict() {
    const QJsonObject conflict = selectedMigrationConflict();
    if (conflict.isEmpty()) return;
    const QString action = migrationActionCombo_->currentData().toString();
    QJsonObject resolution{{QStringLiteral("action"), action}};
    if (action == QStringLiteral("keep_only")) {
        const QString survivor = migrationSurvivorCombo_->currentData().toString();
        if (survivor.isEmpty()) return;
        resolution.insert(QStringLiteral("survivor_rule_id"), survivor);
    } else if (action == QStringLiteral("pair")) {
        const QString frontId = migrationFrontCombo_->currentData().toString();
        const QString backId = migrationBackCombo_->currentData().toString();
        if (frontId.isEmpty() || backId.isEmpty()) return;
        resolution.insert(QStringLiteral("front_rule_id"), frontId);
        resolution.insert(QStringLiteral("back_rule_id"), backId);
    }
    emit migrationResolutionRequested(
        conflict.value(QStringLiteral("conflict_id")).toString(), resolution,
        snapshot_.value(QStringLiteral("library_revision")).toInt(),
        snapshot_.value(QStringLiteral("draft_revision")).toInt());
}

void GeometryRulesPage::ensureDirectionObject(const QString &name) {
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    if (!directions.contains(name)) {
        directions.insert(name, usesLogicalRuleSchema() ? emptyLogicalDirection() : emptyDirection());
    }
    draft_.insert(QStringLiteral("directions"), directions);
}

int GeometryRulesPage::currentRuleIndex() const {
    return ruleList_ ? ruleList_->currentRow() : -1;
}

QJsonObject GeometryRulesPage::currentRule() const {
    const QJsonArray rules = logicalRules();
    const int index = currentRuleIndex();
    if (index < 0 || index >= rules.size()) return {};
    QJsonObject rule = rules.at(index).toObject();
    if (usesLogicalRuleSchema()) {
        const QJsonObject calibration = currentCalibration();
        for (const QString &key : {QStringLiteral("geometry"), QStringLiteral("seed_geometry"),
                                   QStringLiteral("reference_template"), QStringLiteral("diagnostics"),
                                   QStringLiteral("updated_at")}) {
            if (calibration.contains(key)) rule.insert(key, calibration.value(key));
        }
        const QString state = calibration.value(QStringLiteral("state")).toString(QStringLiteral("missing"));
        rule.insert(QStringLiteral("calibration_state"), state);
        rule.insert(QStringLiteral("editor_state"), state == QStringLiteral("ready")
            ? QStringLiteral("ready") : QStringLiteral("needs_reseed"));
    }
    return rule;
}

QJsonObject GeometryRulesPage::currentDirectionAnchor() const {
    return directionObject().value(QStringLiteral("anchor")).toObject();
}

QJsonObject GeometryRulesPage::anchorFromCanvasShape(const QJsonObject &shape) const {
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

QJsonObject GeometryRulesPage::ruleGeometryFromCanvasShape(const QJsonObject &shape) const {
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

QJsonObject GeometryRulesPage::canvasShapeForRule(const QJsonObject &rule) const {
    if (canvas_ == nullptr || canvas_->image().isNull()) return {};
    const QJsonObject anchor = currentDirectionAnchor();
    const QJsonObject coarse = anchor.value(QStringLiteral("coarse")).toObject();
    QJsonObject geometry = rule.value(QStringLiteral("geometry")).toObject();
    if (geometry.isEmpty()) geometry = rule.value(QStringLiteral("seed_geometry")).toObject();
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

void GeometryRulesPage::setSnapshot(const QJsonObject &snapshot) {
    const QString nextWorkpieceId = snapshot.value(QStringLiteral("workpiece_id")).toString();
    Q_UNUSED(nextWorkpieceId);
    lastRulePreview_ = QJsonObject();
    lastPreviewRuleSignature_ = QJsonObject();
    lastPreviewWorkpieceId_.clear();
    lastPreviewRuleId_.clear();
    lastPreviewDirection_.clear();
    lastPreviewTemplateId_.clear();
    lastPreviewLibraryRevision_ = -1;
    lastPreviewDraftRevision_ = -1;
    manualEditContextKey_.clear();
    pendingPreviewContextKey_.clear();
    snapshot_ = snapshot;
    restoreSnapshotDraft();
    job_ = QJsonObject();
    editorDirection_.clear();
    setDirty(false);
    undoHistory_.clear();
    redoHistory_.clear();
    if (versionCombo_ != nullptr) versionCombo_->setCurrentIndex(0);
    if (validationTable_ != nullptr) validationTable_->setRowCount(0);
    if (progressBar_ != nullptr) progressBar_->setValue(0);
    validationDiagnostics_ = QStringLiteral("等待验证");
    templateDiagnostics_.clear();
    refreshDiagnostics();
    if (validationHintLabel_ != nullptr) {
        validationHintLabel_->setText(QStringLiteral("点击表格中的模板行可定位预览；低置信度/未配置模板请在左侧选择“排除”并填写原因，再重新验证。"));
        validationHintLabel_->setStyleSheet(QStringLiteral("color: #7a4b00;"));
    }
    revisionLabel_->setText(QStringLiteral("库修订 %1，草稿修订 %2，活动 %3")
                                .arg(snapshot.value(QStringLiteral("library_revision")).toInt())
                                .arg(snapshot.value(QStringLiteral("draft_revision")).toInt())
                                .arg(snapshot.value(QStringLiteral("active_revision")).toInt(0)));
    statusLabel_->setText(snapshot.value(QStringLiteral("legacy_archived")).toBool()
                              ? QStringLiteral("旧位置标注已归档，请使用几何规则重新标定")
                              : QString());
    statusLabel_->setStyleSheet(QString());
    refreshMigrationPanel();
    refreshRuleList();
    refreshTemplatePreview();
    dirtyLabel_->setText(QStringLiteral("草稿已保存"));
    updatePublishState();
}

void GeometryRulesPage::restoreSnapshotDraft() {
    draft_ = snapshot_.value(QStringLiteral("draft")).toObject();
    if (draft_.isEmpty()) {
        draft_ = QJsonObject{{QStringLiteral("schema_version"), 1},
                             {QStringLiteral("directions"), QJsonObject{{QStringLiteral("front"), emptyDirection()},
                                                                            {QStringLiteral("back"), emptyDirection()}}}};
    }
    ensureDirectionObject(QStringLiteral("front"));
    ensureDirectionObject(QStringLiteral("back"));
}

void GeometryRulesPage::setValidationJob(const QJsonObject &job) {
    if (dirty_) return;
    job_ = job;
    if (validationHintLabel_ != nullptr) {
        validationHintLabel_->setText(QStringLiteral("点击表格中的模板行可定位预览；低置信度/未配置模板请在左侧选择“排除”并填写原因，再重新验证。"));
        validationHintLabel_->setStyleSheet(QStringLiteral("color: #7a4b00;"));
    }
    const QJsonObject progress = job.value(QStringLiteral("progress")).toObject();
    const int total = qMax(1, progress.value(QStringLiteral("total")).toInt(1));
    progressBar_->setRange(0, total);
    progressBar_->setValue(qBound(0, progress.value(QStringLiteral("completed")).toInt(), total));
    statusLabel_->setText(QStringLiteral("验证任务：%1").arg(job.value(QStringLiteral("state")).toString()));
    statusLabel_->setStyleSheet(QString());
    const QJsonArray warnings = job.value(QStringLiteral("warnings")).toArray();
    const QJsonArray blocking = job.value(QStringLiteral("blocking_issues")).toArray();
    QStringList validationLines;
    if (!warnings.isEmpty()) {
        validationLines.append(QStringLiteral("验证告警：\n%1")
            .arg(QString::fromUtf8(QJsonDocument(warnings).toJson(QJsonDocument::Indented))));
    }
    for (const QJsonValue &issueValue : blocking) {
        const QJsonObject issue = issueValue.toObject();
        const QString code = issue.value(QStringLiteral("code")).toString();
        if (code == QStringLiteral("geometry_fusion_regression")) {
            validationLines.append(QStringLiteral("发布阻断：干扰规则导致融合结果回归"));
            for (const QJsonValue &changeValue : issue.value(QStringLiteral("changed_predictions")).toArray()) {
                const QJsonObject change = changeValue.toObject();
                validationLines.append(QStringLiteral(
                    "模板: %1\n基线: %2\n候选全局: %3\n候选局部: %4\n候选最终: %5（%6）")
                    .arg(change.value(QStringLiteral("template_id")).toString(),
                         orientationName(change.value(QStringLiteral("baseline_predicted")).toString()),
                         orientationName(change.value(QStringLiteral("candidate_global_prediction")).toString()),
                         orientationName(change.value(QStringLiteral("candidate_local_prediction")).toString()),
                         orientationName(change.value(QStringLiteral("candidate_predicted")).toString()),
                         change.value(QStringLiteral("candidate_decision_source")).toString()));
            }
        } else {
            validationLines.append(QStringLiteral("发布阻断：%1").arg(code));
        }
    }
    if (validationLines.isEmpty()) validationLines.append(QStringLiteral("暂无告警"));
    validationDiagnostics_ = validationLines.join(QStringLiteral("\n\n"));
    refreshDiagnostics();
    validationTable_->setRowCount(0);
    QSet<QString> blockedTemplateIds;
    QSet<QString> fusionRegressionTemplateIds;
    for (const QJsonValue &issueValue : blocking) {
        const QJsonObject issue = issueValue.toObject();
        const bool fusionRegression = issue.value(QStringLiteral("code")).toString()
            == QStringLiteral("geometry_fusion_regression");
        for (const QJsonValue &templateValue : issue.value(QStringLiteral("templates")).toArray()) {
            const QString value = templateValue.toString();
            blockedTemplateIds.insert(value);
            if (fusionRegression) fusionRegressionTemplateIds.insert(value);
        }
        for (const QJsonValue &changeValue : issue.value(QStringLiteral("changed_predictions")).toArray()) {
            const QString value = changeValue.toObject().value(QStringLiteral("template_id")).toString();
            if (value.isEmpty()) continue;
            blockedTemplateIds.insert(value);
            if (fusionRegression) fusionRegressionTemplateIds.insert(value);
        }
        const QString templateId = issue.value(QStringLiteral("template_id")).toString();
        if (!templateId.isEmpty()) blockedTemplateIds.insert(templateId);
    }
    const QJsonObject report = job.value(QStringLiteral("report")).toObject();
    for (const QString &label : {QStringLiteral("front"), QStringLiteral("back")}) {
        const QJsonArray rows = report.value(label).toArray();
        for (const QJsonValue &value : rows) {
            const QJsonObject row = value.toObject();
            const int target = validationTable_->rowCount();
            validationTable_->insertRow(target);
            const QString templateId = row.value(QStringLiteral("template_id")).toString();
            auto *directionItem = new QTableWidgetItem(label);
            directionItem->setData(Qt::UserRole, label);
            auto *templateItem = new QTableWidgetItem(templateId);
            templateItem->setData(Qt::UserRole, templateId);
            validationTable_->setItem(target, 0, directionItem);
            validationTable_->setItem(target, 1, templateItem);
            auto *statusItem = new QTableWidgetItem(row.value(QStringLiteral("status")).toString());
            const QString status = row.value(QStringLiteral("status")).toString();
            const QString reason = row.value(QStringLiteral("reason_code")).toString();
            const QString statusText = status == QStringLiteral("active") ? QStringLiteral("已通过")
                : status == QStringLiteral("low_confidence") ? QStringLiteral("低置信度")
                : status == QStringLiteral("not_configured") ? QStringLiteral("未配置")
                : status == QStringLiteral("needs_reseed") ? QStringLiteral("需重新标定")
                : status == QStringLiteral("excluded") ? QStringLiteral("已排除")
                : status;
            statusItem->setText(statusText);
            statusItem->setToolTip(reason.isEmpty() ? QStringLiteral("点击此行定位模板，再在左侧完成复核") : reason);
            if (blockedTemplateIds.contains(templateId)) {
                statusItem->setBackground(QColor(QStringLiteral("#f8d7da")));
                statusItem->setToolTip(fusionRegressionTemplateIds.contains(templateId)
                    ? QStringLiteral("融合结果回归：候选全局与局部结论冲突，局部覆盖后造成误判；请查看右侧证据并重新验证")
                    : QStringLiteral("验证阻断：请定位此模板并明确排除或修正后重新验证"));
            } else if (status != QStringLiteral("active") && status != QStringLiteral("not_configured")) {
                statusItem->setBackground(QColor(QStringLiteral("#fff3cd")));
            }
            validationTable_->setItem(target, 2, statusItem);
            int pendingRules = 0;
            for (const QJsonValue &ruleValue : row.value(QStringLiteral("rules")).toArray()) {
                const QString ruleStatus = ruleValue.toObject().value(QStringLiteral("status")).toString();
                if (ruleStatus != QStringLiteral("active") && ruleStatus != QStringLiteral("excluded")) {
                    ++pendingRules;
                }
            }
            validationTable_->setItem(target, 3, new QTableWidgetItem(QString::number(pendingRules)));
        }
    }
    if (!dirty_ && canvas_ != nullptr) {
        const QSignalBlocker canvasBlocker(canvas_);
        canvas_->setCoarseShape(QJsonObject());
    }
    applyFittedBoundaryForCurrentTemplate();
    updatePublishState();
}

void GeometryRulesPage::selectValidationTemplate(int row) {
    if (validationTable_ == nullptr || templateCombo_ == nullptr || directionCombo_ == nullptr
        || row < 0 || row >= validationTable_->rowCount()) {
        return;
    }
    const QTableWidgetItem *directionItem = validationTable_->item(row, 0);
    const QTableWidgetItem *templateItem = validationTable_->item(row, 1);
    if (directionItem == nullptr || templateItem == nullptr) return;
    const QString selectedDirection = directionItem->data(Qt::UserRole).toString();
    const QString selectedTemplate = templateItem->data(Qt::UserRole).toString();
    if (selectedDirection.isEmpty() || selectedTemplate.isEmpty()) return;
    const int directionIndex = directionCombo_->findData(selectedDirection);
    if (directionIndex < 0) return;
    const QSignalBlocker directionBlocker(directionCombo_);
    const QSignalBlocker templateBlocker(templateCombo_);
    suppressPreviewRequests_ = true;
    if (directionCombo_->currentIndex() != directionIndex) directionCombo_->setCurrentIndex(directionIndex);
    refreshRuleList();
    refreshTemplatePreview();
    const int templateIndex = templateCombo_->findData(selectedTemplate, Qt::UserRole + 1);
    if (templateIndex < 0) {
        suppressPreviewRequests_ = false;
        return;
    }
    if (templateCombo_->currentIndex() != templateIndex) templateCombo_->setCurrentIndex(templateIndex);
    refreshRuleList();
    refreshTemplatePreview();
    const int finalTemplateIndex = templateCombo_->findData(selectedTemplate, Qt::UserRole + 1);
    if (finalTemplateIndex >= 0) templateCombo_->setCurrentIndex(finalTemplateIndex);
    refreshTemplateReview();
    suppressPreviewRequests_ = false;
    if (!dirty_) {
        const QSignalBlocker canvasBlocker(canvas_);
        canvas_->setCoarseShape(QJsonObject());
    }
    applyFittedBoundaryForCurrentTemplate();
}

void GeometryRulesPage::setBusy(bool busy) {
    busy_ = busy;
    saveButton_->setEnabled(!busy);
    validateButton_->setEnabled(!busy);
    rollbackButton_->setEnabled(!busy && snapshot_.value(QStringLiteral("previous_active_revision")).toInt(0) > 0);
    updatePublishState();
}

void GeometryRulesPage::setPreviewBusy(bool busy) {
    previewBusy_ = busy;
    if (directionCombo_) directionCombo_->setEnabled(!busy);
    if (templateCombo_) templateCombo_->setEnabled(!busy);
    if (anchorCandidateCombo_) anchorCandidateCombo_->setEnabled(!busy);
    if (ruleCandidateCombo_) ruleCandidateCombo_->setEnabled(!busy);
    if (shapeCombo_) shapeCombo_->setEnabled(!busy);
}

void GeometryRulesPage::setDirty(bool dirty) {
    if (dirty_ == dirty) return;
    dirty_ = dirty;
    emit unsavedChangesChanged(dirty_);
}

void GeometryRulesPage::markDraftDirty() {
    if (!dirty_) {
        job_ = QJsonObject();
        if (validationTable_ != nullptr) validationTable_->setRowCount(0);
        if (progressBar_ != nullptr) {
            progressBar_->setRange(0, 1);
            progressBar_->setValue(0);
        }
        validationDiagnostics_ = QStringLiteral("草稿已修改，旧验证结果已失效；请先保存草稿，再重新验证。");
        templateDiagnostics_.clear();
        refreshDiagnostics();
        if (validationHintLabel_ != nullptr) {
            validationHintLabel_->setText(
                QStringLiteral("草稿已修改：旧验证结果已失效，保存后请重新验证。"));
            validationHintLabel_->setStyleSheet(QStringLiteral("color: #b35c00; font-weight: 600;"));
        }
        if (statusLabel_ != nullptr) {
            statusLabel_->setText(QStringLiteral("草稿已修改：请先保存，再重新验证"));
            statusLabel_->setStyleSheet(QStringLiteral("color: #b35c00;"));
        }
    }
    setDirty(true);
    if (dirtyLabel_ != nullptr) dirtyLabel_->setText(QStringLiteral("草稿有未保存修改"));
    updatePublishState();
}

void GeometryRulesPage::applyDraftMutation(const QJsonObject &next) {
    if (next == draft_) return;
    undoHistory_.append(draft_);
    if (undoHistory_.size() > 100) undoHistory_.removeFirst();
    redoHistory_.clear();
    draft_ = next;
    markDraftDirty();
    refreshEditor();
}

void GeometryRulesPage::refreshEditor() {
    refreshRuleList();
    refreshTemplatePreview();
    dirtyLabel_->setText(dirty_ ? QStringLiteral("草稿有未保存修改") : QStringLiteral("草稿已保存"));
    undoButton_->setEnabled(!undoHistory_.isEmpty());
    redoButton_->setEnabled(!redoHistory_.isEmpty());
    updatePublishState();
}

void GeometryRulesPage::setOperationError(const QString &message) {
    statusLabel_->setText(message);
    statusLabel_->setStyleSheet(QStringLiteral("color: #b00020;"));
}

void GeometryRulesPage::refreshRuleList() {
    if (!ruleList_) return;
    if (editorDirection_ == direction()) setCurrentRuleFromEditor();
    const QString previousId = currentRuleId();
    const int previous = ruleList_->currentRow();
    ruleList_->blockSignals(true);
    ruleList_->clear();
    const QJsonArray rules = logicalRules();
    int selectedRow = -1;
    int row = 0;
    for (const QJsonValue &value : rules) {
        const QJsonObject rule = value.toObject();
        const QString ruleId = rule.value(QStringLiteral("rule_id")).toString();
        QString text;
        if (usesLogicalRuleSchema()) {
            const QString enabled = rule.value(QStringLiteral("enabled")).toBool(true)
                ? QStringLiteral("启用") : QStringLiteral("停用");
            text = QStringLiteral("%1 [%2] 正面:%3 反面:%4")
                .arg(rule.value(QStringLiteral("name")).toString(), enabled,
                     calibrationState(QStringLiteral("front"), ruleId),
                     calibrationState(QStringLiteral("back"), ruleId));
        } else {
            const QString state = rule.value(QStringLiteral("editor_state")).toString()
                    == QStringLiteral("needs_reseed")
                ? QStringLiteral("待重画")
                : rule.value(QStringLiteral("enabled")).toBool(true)
                    ? QStringLiteral("启用")
                    : QStringLiteral("停用");
            text = QStringLiteral("%1 [%2]").arg(rule.value(QStringLiteral("name")).toString(), state);
        }
        auto *item = new QListWidgetItem(text, ruleList_);
        item->setData(Qt::UserRole, ruleId);
        if (!previousId.isEmpty() && ruleId == previousId) selectedRow = row;
        ++row;
    }
    if (!rules.isEmpty()) {
        if (selectedRow < 0) selectedRow = qBound(0, previous, rules.size() - 1);
        ruleList_->setCurrentRow(selectedRow);
    }
    ruleList_->blockSignals(false);
    loadCurrentRuleIntoEditor();
}

void GeometryRulesPage::refreshTemplatePreview() {
    if (!templateCombo_ || !canvas_) return;
    const QSignalBlocker canvasBlocker(canvas_);
    const QString side = direction();
    const QString previous = templateCombo_->currentData().toString();
    templateCombo_->blockSignals(true);
    templateCombo_->clear();
    const QJsonArray templates = snapshot_.value(QStringLiteral("templates")).toArray();
    for (const QJsonValue &value : templates) {
        const QJsonObject item = value.toObject();
        if (item.value(QStringLiteral("direction")).toString() != side) continue;
        templateCombo_->addItem(item.value(QStringLiteral("template_id")).toString(),
                                item.value(QStringLiteral("path")).toString());
        templateCombo_->setItemData(templateCombo_->count() - 1,
                                     item.value(QStringLiteral("template_id")).toString(), Qt::UserRole + 1);
    }
    int index = templateCombo_->findData(previous);
    if (index < 0) index = 0;
    if (templateCombo_->count() > 0) templateCombo_->setCurrentIndex(index);
    templateCombo_->blockSignals(false);
    const QString path = templateCombo_->currentData().toString();
    canvas_->setImage(path.isEmpty() ? QImage() : QImage(path));
    anchorCandidateCombo_->clear();
    ruleCandidateCombo_->clear();
    setTemplateDiagnostics(QStringLiteral("等待当前模板预览"));
    const QJsonObject rule = currentRule();
    if (!rule.isEmpty()) {
        const QJsonObject guideShape = canvasShapeForRule(rule);
        canvas_->setCoarseShape(manualEditContextKey_ == currentEditContextKey()
                                    ? guideShape : QJsonObject());
        const QString shape = rule.value(QStringLiteral("shape")).toString();
        canvas_->setTool(shape == QStringLiteral("ellipse") ? GeometryRuleCanvas::Ellipse
                           : shape == QStringLiteral("rotated_rectangle") ? GeometryRuleCanvas::RotatedRectangle
                                                                            : GeometryRuleCanvas::Circle);
        const QSignalBlocker rotationBlocker(rotationSpin_);
        rotationSpin_->setValue(rule.value(QStringLiteral("geometry")).toObject().value(QStringLiteral("angle_deg")).toDouble());
    } else {
        canvas_->setCoarseShape(QJsonObject());
        canvas_->setFitOverlay(QJsonObject());
        canvas_->setTool(GeometryRuleCanvas::None);
        setTemplateDiagnostics(QStringLiteral("请先选择一条逻辑规则"));
    }
    applyFittedBoundaryForCurrentTemplate();
    refreshTemplateReview();
}

bool GeometryRulesPage::applyFittedBoundaryForCurrentTemplate() {
    if (canvas_ == nullptr || templateCombo_ == nullptr) return false;
    canvas_->setFitOverlay(QJsonObject());
    const QString side = direction();
    const QString templateId = currentTemplateId();
    const QJsonObject rule = currentRule();
    const QString ruleId = rule.value(QStringLiteral("rule_id")).toString();
    if (ruleId.isEmpty()) return false;

    const bool validationMatches = !job_.isEmpty()
        && job_.value(QStringLiteral("base_library_revision")).toInt(-1)
            == snapshot_.value(QStringLiteral("library_revision")).toInt()
        && job_.value(QStringLiteral("base_draft_revision")).toInt(-1)
            == snapshot_.value(QStringLiteral("draft_revision")).toInt();
    if (validationMatches) {
        const QJsonArray rows = job_.value(QStringLiteral("report")).toObject().value(side).toArray();
        for (const QJsonValue &rowValue : rows) {
            const QJsonObject row = rowValue.toObject();
            if (row.value(QStringLiteral("template_id")).toString() != templateId) continue;
            QJsonObject selectedRule;
            for (const QJsonValue &ruleValue : row.value(QStringLiteral("rules")).toArray()) {
                const QJsonObject candidate = ruleValue.toObject();
                if (candidate.value(QStringLiteral("rule_id")).toString() == ruleId) {
                    selectedRule = candidate;
                    break;
                }
            }
            if (selectedRule.isEmpty()) {
                setTemplateDiagnostics(QStringLiteral("当前规则在此模板无拟合报告"));
                return false;
            }
            const bool hasBoundary = !selectedRule.value(QStringLiteral("effective_shape")).toObject().isEmpty()
                || !selectedRule.value(QStringLiteral("fitted_shape")).toObject().isEmpty();
            canvas_->setFitOverlay(hasBoundary ? selectedRule : QJsonObject());
            if (hasBoundary) {
                QStringList lines{QStringLiteral("当前规则的验证拟合结果")};
                if (selectedRule.contains(QStringLiteral("selected_candidate_index"))) {
                    lines.append(QStringLiteral("候选: %1").arg(selectedRule.value(QStringLiteral("selected_candidate_index")).toInt()));
                }
                if (selectedRule.contains(QStringLiteral("edge_support"))) {
                    lines.append(QStringLiteral("边缘支持: %1").arg(selectedRule.value(QStringLiteral("edge_support")).toDouble(), 0, 'f', 3));
                }
                if (selectedRule.contains(QStringLiteral("fit_residual"))) {
                    lines.append(QStringLiteral("残差: %1").arg(selectedRule.value(QStringLiteral("fit_residual")).toDouble(), 0, 'f', 3));
                }
                setTemplateDiagnostics(lines.join(QLatin1Char('\n')));
            } else {
                QString reason = selectedRule.value(QStringLiteral("reason_code")).toString();
                if (reason.isEmpty()) reason = row.value(QStringLiteral("reason_code")).toString();
                if (reason.isEmpty()) reason = row.value(QStringLiteral("status")).toString();
                setTemplateDiagnostics(QStringLiteral("该模板未拟合出有效边界：%1\n"
                                                       "请重新标定、调整候选，或明确排除该模板。")
                                           .arg(reason.isEmpty() ? QStringLiteral("未知原因") : reason));
            }
            return true;
        }
    }

    const bool cachedPreviewMatches = !lastRulePreview_.isEmpty()
        && lastPreviewWorkpieceId_ == snapshot_.value(QStringLiteral("workpiece_id")).toString()
        && lastPreviewRuleId_ == ruleId
        && lastPreviewDirection_ == side
        && lastPreviewTemplateId_ == templateId
        && lastPreviewLibraryRevision_ == snapshot_.value(QStringLiteral("library_revision")).toInt()
        && lastPreviewDraftRevision_ == snapshot_.value(QStringLiteral("draft_revision")).toInt()
        && lastPreviewRuleSignature_ == rulePreviewSignature(rule);
    if (cachedPreviewMatches) {
        const QJsonObject fit = lastRulePreview_.value(QStringLiteral("rule_fit")).toObject();
        const bool hasBoundary = !fit.value(QStringLiteral("effective_shape")).toObject().isEmpty()
            || !fit.value(QStringLiteral("fitted_shape")).toObject().isEmpty();
        canvas_->setFitOverlay(hasBoundary ? fit : QJsonObject());
        if (hasBoundary) {
            setTemplateDiagnostics(QStringLiteral("当前模板的最近拟合边界：\n%1")
                .arg(QString::fromUtf8(QJsonDocument(lastRulePreview_).toJson(QJsonDocument::Indented))));
        } else {
            QString reason = lastRulePreview_.value(QStringLiteral("reason_code")).toString();
            if (reason.isEmpty()) reason = lastRulePreview_.value(QStringLiteral("status")).toString();
            setTemplateDiagnostics(QStringLiteral("该模板未拟合出有效边界：%1\n请重新标定或调整候选。")
                                       .arg(reason.isEmpty() ? QStringLiteral("未知原因") : reason));
        }
        return true;
    }

    QJsonObject reference;
    if (usesLogicalRuleSchema()) {
        reference = currentCalibration().value(QStringLiteral("reference_template")).toObject();
    } else {
        reference = directionObject().value(QStringLiteral("reference_template")).toObject();
    }
    if (reference.value(QStringLiteral("template_id")).toString() == templateId) {
        const QJsonObject savedShape = canvasShapeForRule(rule);
        if (!savedShape.isEmpty()) {
            canvas_->setFitOverlay(savedShape);
            setTemplateDiagnostics(QStringLiteral("显示当前方向参考模板的已保存拟合边界"));
            return true;
        }
    }
    setTemplateDiagnostics(QStringLiteral("当前规则在此模板暂无精确拟合边界"));
    return false;
}

void GeometryRulesPage::setTemplateDiagnostics(const QString &text) {
    templateDiagnostics_ = text;
    refreshDiagnostics();
}

void GeometryRulesPage::refreshDiagnostics() {
    if (diagnostics_ == nullptr) return;
    QStringList sections;
    if (!validationDiagnostics_.isEmpty()) sections.append(validationDiagnostics_);
    if (!templateDiagnostics_.isEmpty()) sections.append(templateDiagnostics_);
    diagnostics_->setPlainText(sections.join(QStringLiteral("\n\n----------------\n\n")));
}

void GeometryRulesPage::refreshTemplateReview() {
    if (templateCombo_ == nullptr || reviewStateCombo_ == nullptr || reviewReasonEdit_ == nullptr) return;
    const QString templateId = templateCombo_->currentData(Qt::UserRole + 1).toString();
    const QJsonObject reviews = directionObject().value(QStringLiteral("template_reviews")).toObject();
    const QJsonObject review = reviews.value(templateId).toObject();
    const QSignalBlocker reviewBlocker(reviewStateCombo_);
    const QSignalBlocker reasonBlocker(reviewReasonEdit_);
    const int reviewIndex = reviewStateCombo_->findData(review.value(QStringLiteral("state")).toString());
    reviewStateCombo_->setCurrentIndex(reviewIndex >= 0 ? reviewIndex : 0);
    reviewReasonEdit_->setText(review.value(QStringLiteral("reason")).toString());
}

void GeometryRulesPage::setAnchorFromCanvas() {
    QJsonObject anchor = anchorFromCanvasShape(canvas_->coarseShape());
    if (anchor.isEmpty()) return;
    ensureDirectionObject(direction());
    QJsonObject side = directionObject();
    anchor.insert(QStringLiteral("mode"), QStringLiteral("manual"));
    side.insert(QStringLiteral("anchor"), anchor);
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    directions.insert(direction(), side);
    draft_.insert(QStringLiteral("directions"), directions);
    markDraftDirty();
    refreshRuleList();
}

void GeometryRulesPage::setCurrentRuleFromEditor() {
    if (!editorDirection_.isEmpty() && editorDirection_ != direction()) return;
    const int index = currentRuleIndex();
    if (index < 0) return;
    QJsonObject directionValue = directionObject();
    QJsonArray rules = logicalRules();
    if (index >= rules.size()) return;
    QJsonObject rule = rules.at(index).toObject();
    const QString editedName = ruleNameEdit_->text();
    const QString selectedShape = shapeCombo_->currentData().toString();
    const QString selectedMode = modeCombo_->currentData().toString();
    const int selectedMarginPercent = marginSpin_->value();
    const bool shapeChanged = !selectedShape.isEmpty()
        && selectedShape != rule.value(QStringLiteral("shape")).toString();
    const bool editorChanged = editedName != rule.value(QStringLiteral("name")).toString()
        || shapeChanged
        || selectedMode != rule.value(QStringLiteral("mode")).toString()
        || selectedMarginPercent
            != qRound(rule.value(QStringLiteral("margin_ratio")).toDouble(0.0) * 100.0);
    if (!editorChanged) return;
    rule.insert(QStringLiteral("name"), editedName.trimmed());
    if (!selectedShape.isEmpty()) rule.insert(QStringLiteral("shape"), selectedShape);
    rule.insert(QStringLiteral("mode"), modeCombo_->currentData().toString());
    rule.insert(QStringLiteral("margin_ratio"), marginSpin_->value() / 100.0);
    rule.insert(QStringLiteral("margin_semantics"), QStringLiteral("signed_boundary_v2"));
    rules.replace(index, rule);
    QJsonObject next = draft_;
    if (usesLogicalRuleSchema()) {
        next.insert(QStringLiteral("rules"), rules);
        if (shapeChanged) {
            QJsonObject directions = next.value(QStringLiteral("directions")).toObject();
            const QString ruleId = rule.value(QStringLiteral("rule_id")).toString();
            for (const QString &sideName : {QStringLiteral("front"), QStringLiteral("back")}) {
                QJsonObject side = directions.value(sideName).toObject();
                QJsonObject calibrations = side.value(QStringLiteral("calibrations")).toObject();
                calibrations.remove(ruleId);
                side.insert(QStringLiteral("calibrations"), calibrations);
                directions.insert(sideName, side);
            }
            next.insert(QStringLiteral("directions"), directions);
        }
    } else {
        if (!rule.contains(QStringLiteral("seed_geometry"))) {
            rule.insert(QStringLiteral("seed_geometry"), rule.value(QStringLiteral("geometry")));
        }
        if (!rule.contains(QStringLiteral("editor_state"))) {
            rule.insert(QStringLiteral("editor_state"), QStringLiteral("ready"));
        }
        rules.replace(index, rule);
        directionValue.insert(QStringLiteral("rules"), rules);
        QJsonObject directions = next.value(QStringLiteral("directions")).toObject();
        directions.insert(direction(), directionValue);
        next.insert(QStringLiteral("directions"), directions);
    }
    if (next == draft_) return;
    if (shapeChanged && usesLogicalRuleSchema()) {
        applyDraftMutation(next);
        return;
    }
    draft_ = next;
    markDraftDirty();
}

void GeometryRulesPage::loadCurrentRuleIntoEditor() {
    editorDirection_ = direction();
    const QJsonObject rule = currentRule();
    if (rule.isEmpty()) return;
    const QSignalBlocker nameBlocker(ruleNameEdit_);
    const QSignalBlocker shapeBlocker(shapeCombo_);
    const QSignalBlocker modeBlocker(modeCombo_);
    const QSignalBlocker marginBlocker(marginSpin_);
    const QSignalBlocker rotationBlocker(rotationSpin_);
    const QSignalBlocker reviewBlocker(reviewStateCombo_);
    const QSignalBlocker reasonBlocker(reviewReasonEdit_);
    ruleNameEdit_->setText(rule.value(QStringLiteral("name")).toString());
    const QString shape = rule.value(QStringLiteral("shape")).toString();
    const int shapeIndex = shapeCombo_->findData(shape);
    shapeCombo_->setCurrentIndex(shapeIndex >= 0 ? shapeIndex : 0);
    modeCombo_->setCurrentIndex(modeCombo_->findData(rule.value(QStringLiteral("mode")).toString()));
    marginSpin_->setValue(qRound(rule.value(QStringLiteral("margin_ratio")).toDouble(0.0) * 100.0));
    rotationSpin_->setValue(rule.value(QStringLiteral("geometry")).toObject().value(QStringLiteral("angle_deg")).toDouble());
    refreshTemplatePreview();
}

void GeometryRulesPage::addRule() {
    ensureDirectionObject(direction());
    if (usesLogicalRuleSchema()) {
        QJsonArray rules = logicalRules();
        rules.append(QJsonObject{{QStringLiteral("rule_id"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
                                 {QStringLiteral("name"), QStringLiteral("新规则")},
                                 {QStringLiteral("shape"), QStringLiteral("circle")},
                                 {QStringLiteral("mode"), QStringLiteral("inside")},
                                 {QStringLiteral("margin_ratio"), 0.0},
                                 {QStringLiteral("margin_semantics"), QStringLiteral("signed_boundary_v2")},
                                 {QStringLiteral("enabled"), true}});
        draft_.insert(QStringLiteral("rules"), rules);
        markDraftDirty();
        refreshRuleList();
        ruleList_->setCurrentRow(rules.size() - 1);
        return;
    }
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
                             {QStringLiteral("margin_ratio"), 0.0},
                             {QStringLiteral("margin_semantics"), QStringLiteral("signed_boundary_v2")},
                             {QStringLiteral("enabled"), true},
                             {QStringLiteral("seed_geometry"), QJsonObject{{QStringLiteral("cx"), 0.0}, {QStringLiteral("cy"), 0.0}, {QStringLiteral("r"), 0.5}}},
                             {QStringLiteral("editor_state"), QStringLiteral("ready")}});
    directionValue.insert(QStringLiteral("rules"), rules);
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    directions.insert(direction(), directionValue);
    draft_.insert(QStringLiteral("directions"), directions);
    markDraftDirty();
    refreshRuleList();
    ruleList_->setCurrentRow(rules.size() - 1);
}

void GeometryRulesPage::deleteRule() {
    setCurrentRuleFromEditor();
    const int index = currentRuleIndex();
    if (index < 0) return;
    QJsonObject directionValue = directionObject();
    QJsonArray rules = logicalRules();
    if (index >= rules.size()) return;
    const QString ruleId = rules.at(index).toObject().value(QStringLiteral("rule_id")).toString();
    rules.removeAt(index);
    if (usesLogicalRuleSchema()) {
        QJsonObject next = draft_;
        next.insert(QStringLiteral("rules"), rules);
        QJsonObject directions = next.value(QStringLiteral("directions")).toObject();
        for (const QString &sideName : {QStringLiteral("front"), QStringLiteral("back")}) {
            QJsonObject side = directions.value(sideName).toObject();
            QJsonObject calibrations = side.value(QStringLiteral("calibrations")).toObject();
            calibrations.remove(ruleId);
            side.insert(QStringLiteral("calibrations"), calibrations);
            QJsonObject reviews = side.value(QStringLiteral("rule_reviews")).toObject();
            reviews.remove(ruleId);
            if (!reviews.isEmpty() || side.contains(QStringLiteral("rule_reviews"))) {
                side.insert(QStringLiteral("rule_reviews"), reviews);
            }
            directions.insert(sideName, side);
        }
        next.insert(QStringLiteral("directions"), directions);
        applyDraftMutation(next);
        return;
    }
    directionValue.insert(QStringLiteral("rules"), rules);
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    directions.insert(direction(), directionValue);
    draft_.insert(QStringLiteral("directions"), directions);
    markDraftDirty();
    refreshRuleList();
}

void GeometryRulesPage::requestPreviewFromShape(const QJsonObject &shape) {
    if (shape.isEmpty() || previewBusy_) return;
    if (manualAnchorCapture_) {
        const QJsonObject anchor = anchorFromCanvasShape(shape);
        if (anchor.isEmpty()) return;
        QJsonObject side = directionObject();
        side.insert(QStringLiteral("anchor"), anchor);
        QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
        directions.insert(direction(), side);
        QJsonObject next = draft_;
        next.insert(QStringLiteral("directions"), directions);
        manualAnchorCapture_ = false;
        manualAnchorButton_->setVisible(false);
        applyDraftMutation(next);
        return;
    }
    const QJsonObject request = previewRequest(anchorCandidateCombo_->currentIndex(),
                                               ruleCandidateCombo_->currentIndex(), shape);
    if (request.isEmpty()) return;
    canvas_->setFitOverlay(QJsonObject());
    pendingPreviewContextKey_ = currentEditContextKey();
    setPreviewBusy(true);
    emit previewRequested(request);
}

QJsonObject GeometryRulesPage::previewRequest(int anchorCandidateIndex, int ruleCandidateIndex,
                                                      const QJsonObject &seedShape) const {
    const QString templateId = templateCombo_->currentData(Qt::UserRole + 1).toString().isEmpty()
        ? templateCombo_->currentText() : templateCombo_->currentData(Qt::UserRole + 1).toString();
    if (templateId.isEmpty() || currentRuleIndex() < 0) return {};
    QJsonObject shape = seedShape;
    if (shape.isEmpty()) shape = canvas_->coarseShape();
    if (shape.isEmpty()) shape = canvasShapeForRule(currentRule());
    if (shape.isEmpty()) return {};
    QJsonObject request{{QStringLiteral("base_library_revision"), snapshot_.value(QStringLiteral("library_revision"))},
                        {QStringLiteral("base_draft_revision"), snapshot_.value(QStringLiteral("draft_revision"))},
                        {QStringLiteral("rule_id"), currentRuleId()},
                        {QStringLiteral("direction"), direction()},
                        {QStringLiteral("template_id"), templateId},
                        {QStringLiteral("seed_shape"), shape},
                        {QStringLiteral("mode"), modeCombo_->currentData().toString()},
                        {QStringLiteral("margin_ratio"), marginSpin_->value() / 100.0}};
    if (anchorCandidateIndex >= 0) request.insert(QStringLiteral("anchor_candidate_index"), anchorCandidateIndex);
    if (ruleCandidateIndex >= 0) request.insert(QStringLiteral("rule_candidate_index"), ruleCandidateIndex);
    return request;
}

void GeometryRulesPage::candidateChanged(int index) {
    Q_UNUSED(index);
    if (previewBusy_) return;
    const QJsonObject request = previewRequest(anchorCandidateCombo_->currentIndex(), ruleCandidateCombo_->currentIndex());
    if (request.isEmpty()) return;
    pendingPreviewContextKey_ = currentEditContextKey();
    setPreviewBusy(true);
    emit previewRequested(request);
}

void GeometryRulesPage::setRulePreview(const QJsonObject &preview) {
    if (preview.value(QStringLiteral("direction")).toString() != direction()) return;
    const QString expectedTemplate = templateCombo_->currentData(Qt::UserRole + 1).toString();
    if (!expectedTemplate.isEmpty() && preview.value(QStringLiteral("template_id")).toString() != expectedTemplate) return;
    const QString previewRuleId = preview.value(QStringLiteral("rule_id")).toString();
    if (usesLogicalRuleSchema() && previewRuleId.isEmpty()) return;
    if (!previewRuleId.isEmpty() && previewRuleId != currentRuleId()) return;
    if (preview.contains(QStringLiteral("base_library_revision"))
        && preview.value(QStringLiteral("base_library_revision")).toInt(-1)
            != snapshot_.value(QStringLiteral("library_revision")).toInt()) return;
    if (preview.contains(QStringLiteral("base_draft_revision"))
        && preview.value(QStringLiteral("base_draft_revision")).toInt(-1)
            != snapshot_.value(QStringLiteral("draft_revision")).toInt()) return;
    if (!pendingPreviewContextKey_.isEmpty()
        && pendingPreviewContextKey_ != currentEditContextKey()) return;
    manualEditContextKey_.clear();
    const QJsonObject patch = preview.value(QStringLiteral("profile_patch")).toObject();
    if (!patch.isEmpty()) {
        QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
        QJsonObject side = directions.value(direction()).toObject();
        if (patch.contains(QStringLiteral("anchor"))) side.insert(QStringLiteral("anchor"), patch.value(QStringLiteral("anchor")));
        const int index = currentRuleIndex();
        if (usesLogicalRuleSchema()) {
            const QString ruleId = currentRuleId();
            QJsonObject calibrations = side.value(QStringLiteral("calibrations")).toObject();
            QJsonObject calibration = calibrations.value(ruleId).toObject();
            for (const QString &key : {QStringLiteral("geometry"), QStringLiteral("seed_geometry"),
                                       QStringLiteral("reference_template")}) {
                if (patch.contains(key)) calibration.insert(key, patch.value(key));
            }
            calibration.insert(QStringLiteral("state"), QStringLiteral("ready"));
            calibration.insert(QStringLiteral("diagnostics"), preview.value(QStringLiteral("rule_fit")));
            calibrations.insert(ruleId, calibration);
            side.insert(QStringLiteral("calibrations"), calibrations);
            QJsonArray rules = logicalRules();
            if (index >= 0 && index < rules.size()) {
                QJsonObject rule = rules.at(index).toObject();
                if (patch.contains(QStringLiteral("margin_ratio"))) rule.insert(QStringLiteral("margin_ratio"), patch.value(QStringLiteral("margin_ratio")));
                if (patch.contains(QStringLiteral("margin_semantics"))) rule.insert(QStringLiteral("margin_semantics"), patch.value(QStringLiteral("margin_semantics")));
                rules.replace(index, rule);
                draft_.insert(QStringLiteral("rules"), rules);
            }
        } else {
            if (patch.contains(QStringLiteral("reference_template"))) {
                side.insert(QStringLiteral("reference_template"), patch.value(QStringLiteral("reference_template")));
            }
            QJsonArray rules = side.value(QStringLiteral("rules")).toArray();
            if (index >= 0 && index < rules.size()) {
                QJsonObject rule = rules.at(index).toObject();
                if (patch.contains(QStringLiteral("geometry"))) rule.insert(QStringLiteral("geometry"), patch.value(QStringLiteral("geometry")));
                if (patch.contains(QStringLiteral("seed_geometry"))) rule.insert(QStringLiteral("seed_geometry"), patch.value(QStringLiteral("seed_geometry")));
                if (patch.contains(QStringLiteral("margin_ratio"))) rule.insert(QStringLiteral("margin_ratio"), patch.value(QStringLiteral("margin_ratio")));
                if (patch.contains(QStringLiteral("margin_semantics"))) rule.insert(QStringLiteral("margin_semantics"), patch.value(QStringLiteral("margin_semantics")));
                if (rule.value(QStringLiteral("editor_state")).toString() == QStringLiteral("needs_reseed")) {
                    rule.insert(QStringLiteral("enabled"), true);
                }
                rule.insert(QStringLiteral("editor_state"), QStringLiteral("ready"));
                rules.replace(index, rule);
                side.insert(QStringLiteral("rules"), rules);
            }
        }
        directions.insert(direction(), side);
        QJsonObject next = draft_;
        next.insert(QStringLiteral("directions"), directions);
        applyDraftMutation(next);
    }
    const QJsonObject anchorFit = preview.value(QStringLiteral("anchor_fit")).toObject();
    const QJsonArray anchorCandidates = anchorFit.value(QStringLiteral("candidates")).toArray();
    const QJsonObject ruleFit = preview.value(QStringLiteral("rule_fit")).toObject();
    const QJsonArray ruleCandidates = ruleFit.value(QStringLiteral("candidates")).toArray();
    anchorCandidateCombo_->blockSignals(true);
    anchorCandidateCombo_->clear();
    for (int i = 0; i < anchorCandidates.size(); ++i) anchorCandidateCombo_->addItem(QStringLiteral("候选 %1").arg(i + 1), i);
    anchorCandidateCombo_->setCurrentIndex(anchorFit.value(QStringLiteral("selected_candidate_index")).toInt(0));
    anchorCandidateCombo_->blockSignals(false);
    ruleCandidateCombo_->blockSignals(true);
    ruleCandidateCombo_->clear();
    for (int i = 0; i < ruleCandidates.size(); ++i) ruleCandidateCombo_->addItem(QStringLiteral("候选 %1").arg(i + 1), i);
    ruleCandidateCombo_->setCurrentIndex(ruleFit.value(QStringLiteral("selected_candidate_index")).toInt(0));
    ruleCandidateCombo_->blockSignals(false);
    lastRulePreview_ = preview;
    lastPreviewRuleSignature_ = rulePreviewSignature(currentRule());
    lastPreviewWorkpieceId_ = snapshot_.value(QStringLiteral("workpiece_id")).toString();
    lastPreviewRuleId_ = currentRuleId();
    lastPreviewDirection_ = direction();
    lastPreviewTemplateId_ = expectedTemplate;
    lastPreviewLibraryRevision_ = snapshot_.value(QStringLiteral("library_revision")).toInt();
    lastPreviewDraftRevision_ = snapshot_.value(QStringLiteral("draft_revision")).toInt();
    pendingPreviewContextKey_.clear();
    const bool hasBoundary = !ruleFit.value(QStringLiteral("effective_shape")).toObject().isEmpty()
        || !ruleFit.value(QStringLiteral("fitted_shape")).toObject().isEmpty();
    canvas_->setFitOverlay(hasBoundary ? ruleFit : QJsonObject());
    setTemplateDiagnostics(QString::fromUtf8(QJsonDocument(preview).toJson(QJsonDocument::Indented)));
    const QString previewStatus = preview.value(QStringLiteral("status")).toString();
    const bool active = previewStatus == QStringLiteral("active");
    if (active) {
        statusLabel_->setText(QStringLiteral("当前模板拟合成功，请检查绿色边界"));
        statusLabel_->setStyleSheet(QStringLiteral("color: #008000;"));
    } else if (previewStatus == QStringLiteral("low_confidence")) {
        statusLabel_->setText(QStringLiteral("当前模板未拟合出有效边界，请在当前方向重新粗画；必要时重设基准边界"));
        statusLabel_->setStyleSheet(QStringLiteral("color: #b35c00;"));
    }
    manualAnchorButton_->setVisible(!active);
    setPreviewBusy(false);
}

void GeometryRulesPage::setTemplateReview(const QString &state, const QString &reason) {
    const QString templateId = templateCombo_->currentData(Qt::UserRole + 1).toString();
    if (templateId.isEmpty() || state.isEmpty()) return;
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    QJsonObject side = directions.value(direction()).toObject();
    QJsonObject reviews = side.value(QStringLiteral("template_reviews")).toObject();
    if (state == QStringLiteral("included") && reason.trimmed().isEmpty()) {
        reviews.remove(templateId);
    } else {
        reviews.insert(templateId, QJsonObject{{QStringLiteral("state"), state},
                                                {QStringLiteral("reason"), reason.trimmed()}});
    }
    side.insert(QStringLiteral("template_reviews"), reviews);
    directions.insert(direction(), side);
    QJsonObject next = draft_;
    next.insert(QStringLiteral("directions"), directions);
    if (next == draft_) return;
    draft_ = next;
    markDraftDirty();
}

void GeometryRulesPage::saveDraft() {
    setCurrentRuleFromEditor();
    emit saveDraftRequested(draft_, snapshot_.value(QStringLiteral("library_revision")).toInt(),
                             snapshot_.value(QStringLiteral("draft_revision")).toInt());
}

void GeometryRulesPage::requestSaveDraft() {
    saveDraft();
}

void GeometryRulesPage::discardUnsavedChanges() {
    reloadDraft();
}

void GeometryRulesPage::validateDraft() {
    setCurrentRuleFromEditor();
    emit validateRequested(snapshot_.value(QStringLiteral("library_revision")).toInt(),
                          snapshot_.value(QStringLiteral("draft_revision")).toInt());
}

void GeometryRulesPage::publishDraft() {
    if (!publishButton_->isEnabled()) return;
    emit publishRequested(job_.value(QStringLiteral("job_id")).toString(),
                          snapshot_.value(QStringLiteral("library_revision")).toInt(),
                          snapshot_.value(QStringLiteral("draft_revision")).toInt(),
                          overrideReasonEdit_->text().trimmed());
}

void GeometryRulesPage::publishWorkflow() {
    if (publishWorkflowButton_ == nullptr || !publishWorkflowButton_->isEnabled()) return;
    setCurrentRuleFromEditor();
    const QJsonArray conflicts = draft_.value(QStringLiteral("migration")).toObject()
                                     .value(QStringLiteral("conflicts")).toArray();
    if (!conflicts.isEmpty()) {
        statusLabel_->setText(QStringLiteral("请先在右侧处理迁移冲突，再保存、验证并发布"));
        statusLabel_->setStyleSheet(QStringLiteral("color: #b00020;"));
        return;
    }
    const MissingCalibration missing = firstMissingEnabledCalibration();
    if (missing.valid) {
        const int directionIndex = directionCombo_->findData(missing.direction);
        if (directionIndex >= 0) directionCombo_->setCurrentIndex(directionIndex);
        selectRuleById(missing.ruleId);
        statusLabel_->setText(missing.direction == QStringLiteral("front")
            ? QStringLiteral("请先完成正面标定") : QStringLiteral("请先完成反面标定"));
        statusLabel_->setStyleSheet(QStringLiteral("color: #b00020;"));
        return;
    }
    emit publishWorkflowRequested(draft_,
                                  snapshot_.value(QStringLiteral("library_revision")).toInt(),
                                  snapshot_.value(QStringLiteral("draft_revision")).toInt(),
                                  overrideReasonEdit_->text().trimmed());
}

void GeometryRulesPage::rollbackDraft() {
    if (busy_) return;
    emit rollbackRequested(snapshot_.value(QStringLiteral("library_revision")).toInt());
}

void GeometryRulesPage::updatePublishState() {
    const QString state = job_.value(QStringLiteral("state")).toString();
    const bool completed = state == QStringLiteral("completed")
        && job_.value(QStringLiteral("base_library_revision")).toInt() == snapshot_.value(QStringLiteral("library_revision")).toInt()
        && job_.value(QStringLiteral("base_draft_revision")).toInt() == snapshot_.value(QStringLiteral("draft_revision")).toInt();
    const bool hasWarning = !job_.value(QStringLiteral("warnings")).toArray().isEmpty()
        || job_.value(QStringLiteral("regression")).toObject().value(QStringLiteral("correct_to_wrong")).toInt() > 0;
    const bool hasBlocking = !job_.value(QStringLiteral("blocking_issues")).toArray().isEmpty();
    const bool readOnly = versionCombo_ != nullptr && versionCombo_->currentData().toString() == QStringLiteral("active");
    const bool hasOverrideReason = !overrideReasonEdit_->text().trimmed().isEmpty();
    const bool needsOverrideReason = completed && hasWarning && !hasBlocking && !hasOverrideReason;
    publishButton_->setEnabled(!busy_ && !readOnly && completed && !hasBlocking
                               && (!hasWarning || hasOverrideReason));
    if (validateButton_ != nullptr) {
        validateButton_->setEnabled(!busy_ && !readOnly && !dirty_);
        validateButton_->setToolTip(dirty_
            ? QStringLiteral("请先保存当前草稿，再重新验证") : QString());
    }
    if (publishWorkflowButton_ != nullptr) {
        publishWorkflowButton_->setEnabled(!busy_ && !readOnly);
        publishWorkflowButton_->setToolTip(QStringLiteral("自动保存当前草稿并验证；无阻断项时继续发布"));
    }
    if (needsOverrideReason) {
        statusLabel_->setText(QStringLiteral("验证完成：存在可覆盖告警，请填写发布覆盖原因"));
        statusLabel_->setStyleSheet(QStringLiteral("color: #b35c00;"));
        publishButton_->setToolTip(QStringLiteral("请先填写“发布覆盖原因（有告警或回归时必填）”"));
        overrideReasonEdit_->setToolTip(QStringLiteral("告警允许覆盖，但必须记录人工确认原因"));
    } else if (completed && hasBlocking) {
        statusLabel_->setText(QStringLiteral("验证完成：存在阻断项，处理后才能发布"));
        statusLabel_->setStyleSheet(QStringLiteral("color: #c00000;"));
        publishButton_->setToolTip(QStringLiteral("存在阻断项，当前不能发布"));
    } else if (completed && hasWarning && hasOverrideReason) {
        statusLabel_->setText(QStringLiteral("验证完成：覆盖原因已填写，可以发布"));
        statusLabel_->setStyleSheet(QStringLiteral("color: #008000;"));
        publishButton_->setToolTip(QString());
    } else {
        publishButton_->setToolTip(QString());
    }
}

void GeometryRulesPage::undoDraft() {
    if (undoHistory_.isEmpty()) return;
    redoHistory_.append(draft_);
    draft_ = undoHistory_.takeLast();
    editorDirection_.clear();
    markDraftDirty();
    refreshEditor();
}

void GeometryRulesPage::redoDraft() {
    if (redoHistory_.isEmpty()) return;
    undoHistory_.append(draft_);
    draft_ = redoHistory_.takeLast();
    editorDirection_.clear();
    markDraftDirty();
    refreshEditor();
}

void GeometryRulesPage::resetCurrentRule() {
    const int index = currentRuleIndex();
    if (index < 0) return;
    if (usesLogicalRuleSchema()) {
        QJsonObject calibration = currentCalibration();
        const QJsonValue seed = calibration.value(QStringLiteral("seed_geometry"));
        if (!seed.isUndefined()) calibration.insert(QStringLiteral("geometry"), seed);
        calibration.insert(QStringLiteral("state"), QStringLiteral("ready"));
        QJsonObject next = draft_;
        QJsonObject directions = next.value(QStringLiteral("directions")).toObject();
        QJsonObject side = directions.value(direction()).toObject();
        QJsonObject calibrations = side.value(QStringLiteral("calibrations")).toObject();
        calibrations.insert(currentRuleId(), calibration);
        side.insert(QStringLiteral("calibrations"), calibrations);
        directions.insert(direction(), side);
        next.insert(QStringLiteral("directions"), directions);
        applyDraftMutation(next);
        return;
    }
    QJsonObject side = directionObject();
    QJsonArray rules = side.value(QStringLiteral("rules")).toArray();
    QJsonObject rule = rules.at(index).toObject();
    const QJsonValue seed = rule.value(QStringLiteral("seed_geometry"));
    if (!seed.isUndefined()) rule.insert(QStringLiteral("geometry"), seed);
    rule.insert(QStringLiteral("editor_state"), QStringLiteral("ready"));
    rules.replace(index, rule);
    side.insert(QStringLiteral("rules"), rules);
    QJsonObject directions = draft_.value(QStringLiteral("directions")).toObject();
    directions.insert(direction(), side);
    QJsonObject next = draft_;
    next.insert(QStringLiteral("directions"), directions);
    applyDraftMutation(next);
}

void GeometryRulesPage::reloadDraft() {
    restoreSnapshotDraft();
    editorDirection_.clear();
    undoHistory_.clear();
    redoHistory_.clear();
    setDirty(false);
    job_ = QJsonObject();
    if (validationTable_ != nullptr) validationTable_->setRowCount(0);
    if (progressBar_ != nullptr) {
        progressBar_->setRange(0, 1);
        progressBar_->setValue(0);
    }
    validationDiagnostics_ = QStringLiteral("等待重新验证");
    templateDiagnostics_.clear();
    refreshDiagnostics();
    if (validationHintLabel_ != nullptr) {
        validationHintLabel_->setText(QStringLiteral("已恢复保存的草稿；请重新验证以生成最新结果。"));
        validationHintLabel_->setStyleSheet(QStringLiteral("color: #7a4b00;"));
    }
    if (statusLabel_ != nullptr) {
        statusLabel_->setText(QStringLiteral("已重新载入已保存草稿，请重新验证"));
        statusLabel_->setStyleSheet(QString());
    }
    refreshEditor();
}

void GeometryRulesPage::copyActiveToDraft() {
    const QJsonObject active = snapshot_.value(QStringLiteral("active")).toObject();
    if (active.isEmpty()) return;
    applyDraftMutation(active);
    versionCombo_->setCurrentIndex(0);
}
