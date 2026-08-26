#include <QtTest/QtTest>

#include <QDialog>
#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>
#include <QImage>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTextEdit>

#include "../geometryrulespage.h"
#include "../geometryrulecanvas.h"

QJsonObject profileSnapshot(int libraryRevision, int draftRevision, int activeRevision);
QJsonObject configuredProfileSnapshot(const QString &imagePath, int draftRevision = 0);
QJsonObject configuredV2ProfileSnapshot(const QString &frontPath, const QString &backPath);

class TestGeometryRulesPage : public QObject {
    Q_OBJECT

private slots:
    void pageIsAChildWidgetAndHasNoDialogCloseAction();
    void dirtyStateEmitsOnlyOnChangeAndDiscardRestoresSnapshot();
    void editorFieldChangeMarksDirtyAndDiscardRestores_data();
    void editorFieldChangeMarksDirtyAndDiscardRestores();
    void emptySnapshotDiscardRestoresNormalizedDraft();
    void requestSaveDraftUsesExistingSavePath();
    void normalStateExposesOnePrimaryWorkflowAction();
    void disabledWorkflowShowsTheExactReasonNearby();
    void backendUnavailableDisablesWorkflowWithoutClearingState();
    void lockedDirectionChangeDoesNotRequestPreviewOrStickSelectors();
    void advancedDiagnosticsAreCollapsedByDefault();
    void migrationControlsAppearOnlyWhenConflictExists();
    void issueRowStoresAndSelectsExactContext();
    void templateLevelIssueDoesNotInventRuleSelection();
    void validationStatesUseHumanReadableLabels_data();
    void validationStatesUseHumanReadableLabels();
    void standaloneIssuesUseHumanReadableStatusAndReason();
    void leaveOneOutIssueCreatesOneLocatableRowPerTemplate();
    void unmatchedStandaloneTemplateDoesNotGuessDirectionFromItsName();
    void directionIndexIssueResolvesTemplateWithoutGuessing();
    void regressionIssueUsesExpectedDirectionToLocateTemplate();
    void publishDisabledUntilCompletedValidationMatchesDraft();
    void draftEditInvalidatesVisibleValidationResult();
    void lateValidationUpdateStaysHiddenAfterDraftEdit();
    void warningPublishRequiresOverrideReason();
    void warningPublishRequirementIsVisible();
    void warningContinuationUsesPrimaryWithoutResaving();
    void warningWithoutJobIdentityStartsANewWorkflow();
    void newWarningJobClearsPreviousOverrideReason();
    void draftMutationClearsWarningOverrideReason();
    void newSnapshotClearsWarningOverrideReason();
    void clearingContinuationClearsWarningOverrideReason();
    void ordinaryWorkflowNeverEmitsAStaleOverrideReason();
    void saveAndDeleteRulesUpdateDraft();
    void templatePreviewLoadsRepresentativeImage();
    void validationRowSelectsTemplateForReview();
    void validationRowSelectsTemplateAcrossDirections();
    void publishWorkflowButtonEmitsDraft();
    void signedMarginSpinBoxAcceptsNegativeZeroAndPositive();
    void newRuleUsesSignedMarkerAndZeroDefault();
    void savedDraftHidesGuideAndShowsFittedBoundary();
    void validationTemplateShowsFittedBoundaryInsteadOfCoarseGuide();
    void layerVisibilityChangesDoNotDirtyDraft();
    void layerVisibilityPersistsAcrossTemplateRefresh();
    void directionAndTemplateSelectorsStayWithCanvas();
    void validationSummaryStaysInRightColumn();
    void publishDisabledReasonStaysInRightColumn();
    void primaryWorkflowStaysInRightColumn();
    void failedValidationTemplateDoesNotFallBackToCoarseGuide();
    void selectingTemplateRequestsPreviewWhileSavedGuideStaysHidden();
    void ruleInventoryDoesNotChangeAcrossValidationDirections();
    void logicalRuleWithOnlyFrontCalibrationRemainsVisibleOnBack();
    void deleteLogicalRuleRemovesBothCalibrations();
    void lowConfidenceBackPreviewExplainsHowToRetry();
    void noSelectedRuleClearsEveryRuleOverlay();
    void savedCalibrationShowsFittedBoundaryOnlyForReferenceTemplate();
    void validationDiagnosticsComeFromSelectedNestedRule();
    void fusionRegressionRemainsVisibleWhileInspectingTemplateFit();
    void publishWorkflowNavigatesToMissingDirectionCalibration();
    void migrationConflictKeepOnlyEmitsExplicitResolution();
    void allSupportedShapesUseTheSavePath();
    void shapeChangeDoesNotSaveStaleGeometry_data();
    void shapeChangeDoesNotSaveStaleGeometry();
    void numericShapeProducesBackendCompatiblePayload_data();
    void numericShapeProducesBackendCompatiblePayload();
    void emptyDraftHasNoDefaultDrawingTool();
    void signedMarginKeepsItsSignForInsideAndOutside();
};

void TestGeometryRulesPage::pageIsAChildWidgetAndHasNoDialogCloseAction() {
    QWidget host;
    GeometryRulesPage page(&host);
    QVERIFY(qobject_cast<QDialog *>(&page) == nullptr);
    QVERIFY(!page.isWindow());
    const auto buttons = page.findChildren<QPushButton *>();
    for (QPushButton *button : buttons) {
        QVERIFY2(button->text() != QStringLiteral("关闭"),
                 "embedded page must not retain the dialog close button");
    }
}

void TestGeometryRulesPage::dirtyStateEmitsOnlyOnChangeAndDiscardRestoresSnapshot() {
    GeometryRulesPage page;
    const QJsonObject saved = profileSnapshot(3, 4, 1);
    QSignalSpy dirtySpy(&page, &GeometryRulesPage::unsavedChangesChanged);
    QSignalSpy saveSpy(&page, &GeometryRulesPage::saveDraftRequested);
    page.setSnapshot(saved);
    QVERIFY(!page.hasUnsavedChanges());
    QCOMPARE(dirtySpy.count(), 0);

    page.findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
    QVERIFY(page.hasUnsavedChanges());
    QCOMPARE(dirtySpy.count(), 1);
    QCOMPARE(dirtySpy.at(0).at(0).toBool(), true);
    page.findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
    QCOMPARE(dirtySpy.count(), 1);

    page.discardUnsavedChanges();
    QVERIFY(!page.hasUnsavedChanges());
    QCOMPARE(page.draft(), saved.value(QStringLiteral("draft")).toObject());
    QCOMPARE(dirtySpy.count(), 2);
    QCOMPARE(dirtySpy.at(1).at(0).toBool(), false);
    QCOMPARE(saveSpy.count(), 0);

    page.discardUnsavedChanges();
    QCOMPARE(dirtySpy.count(), 2);
}

void TestGeometryRulesPage::editorFieldChangeMarksDirtyAndDiscardRestores_data() {
    QTest::addColumn<QString>("field");
    QTest::newRow("name") << QStringLiteral("name");
    QTest::newRow("shape") << QStringLiteral("shape");
    QTest::newRow("mode") << QStringLiteral("mode");
    QTest::newRow("margin") << QStringLiteral("margin");
}

void TestGeometryRulesPage::editorFieldChangeMarksDirtyAndDiscardRestores() {
    QFETCH(QString, field);
    GeometryRulesPage page;
    const QJsonObject saved = configuredProfileSnapshot(QString());
    page.setSnapshot(saved);
    const QJsonObject savedDraft = page.draft();
    auto *shape = page.findChild<QComboBox *>(QStringLiteral("shapeCombo"));
    auto *margin = page.findChild<QSpinBox *>(QStringLiteral("marginSpinBox"));
    QLineEdit *name = nullptr;
    QComboBox *mode = nullptr;
    for (QLineEdit *candidate : page.findChildren<QLineEdit *>()) {
        if (candidate->text() == QStringLiteral("中心反光")) name = candidate;
    }
    for (QComboBox *candidate : page.findChildren<QComboBox *>()) {
        if (candidate->findData(QStringLiteral("inside")) >= 0
            && candidate->findData(QStringLiteral("outside")) >= 0) {
            mode = candidate;
        }
    }
    QVERIFY(name != nullptr);
    QVERIFY(shape != nullptr);
    QVERIFY(mode != nullptr);
    QVERIFY(margin != nullptr);

    if (field == QStringLiteral("name")) {
        name->setText(QStringLiteral("中心高光"));
    } else if (field == QStringLiteral("shape")) {
        shape->setCurrentIndex(shape->findData(QStringLiteral("ellipse")));
    } else if (field == QStringLiteral("mode")) {
        mode->setCurrentIndex(mode->findData(QStringLiteral("outside")));
    } else {
        margin->setValue(3);
    }

    QVERIFY(page.hasUnsavedChanges());
    page.discardUnsavedChanges();
    QVERIFY(!page.hasUnsavedChanges());
    QCOMPARE(page.draft(), savedDraft);
}

void TestGeometryRulesPage::emptySnapshotDiscardRestoresNormalizedDraft() {
    GeometryRulesPage page;
    page.setSnapshot(QJsonObject{{QStringLiteral("workpiece_id"), QStringLiteral("m-empty")},
                                 {QStringLiteral("library_revision"), 1},
                                 {QStringLiteral("draft_revision"), 0}});
    const QJsonObject normalized = page.draft();
    QVERIFY(!normalized.isEmpty());
    const QJsonObject directions = normalized.value(QStringLiteral("directions")).toObject();
    QVERIFY(directions.contains(QStringLiteral("front")));
    QVERIFY(directions.contains(QStringLiteral("back")));
    page.findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
    QVERIFY(page.hasUnsavedChanges());

    page.discardUnsavedChanges();

    QVERIFY(!page.hasUnsavedChanges());
    QCOMPARE(page.draft(), normalized);
}

void TestGeometryRulesPage::requestSaveDraftUsesExistingSavePath() {
    GeometryRulesPage page;
    page.setSnapshot(profileSnapshot(7, 8, 1));
    page.findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
    QSignalSpy saveSpy(&page, &GeometryRulesPage::saveDraftRequested);

    page.requestSaveDraft();

    QCOMPARE(saveSpy.count(), 1);
    const QList<QVariant> arguments = saveSpy.takeFirst();
    QCOMPARE(arguments.at(0).toJsonObject(), page.draft());
    QCOMPARE(arguments.at(1).toInt(), 7);
    QCOMPARE(arguments.at(2).toInt(), 8);
}

void TestGeometryRulesPage::normalStateExposesOnePrimaryWorkflowAction() {
    GeometryRulesPage page;
    page.setSnapshot(configuredProfileSnapshot(QString()));
    page.setBackendAvailable(true, QString());
    page.show();
    QCoreApplication::processEvents();

    auto *primary = page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    QVERIFY(primary != nullptr);
    QCOMPARE(primary->property("role").toString(), QStringLiteral("primary"));
    int visiblePrimaryCount = 0;
    for (QPushButton *button : page.findChildren<QPushButton *>()) {
        if (button->isVisibleTo(&page)
            && button->property("role").toString() == QStringLiteral("primary")) {
            ++visiblePrimaryCount;
        }
    }
    QCOMPARE(visiblePrimaryCount, 1);
}

void TestGeometryRulesPage::disabledWorkflowShowsTheExactReasonNearby() {
    GeometryRulesPage page;
    page.setSnapshot(configuredProfileSnapshot(QString()));
    page.setBackendAvailable(false, QStringLiteral("后端断开"));

    auto *primary = page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    auto *reason = page.findChild<QLabel *>(QStringLiteral("publishDisabledReasonLabel"));
    QVERIFY(primary != nullptr);
    QVERIFY(reason != nullptr);
    QVERIFY(!primary->isEnabled());
    QCOMPARE(reason->text(), QStringLiteral("后端断开"));
}

void TestGeometryRulesPage::backendUnavailableDisablesWorkflowWithoutClearingState() {
    GeometryRulesPage page;
    page.setSnapshot(configuredProfileSnapshot(QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-preserved")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 1},
        {QStringLiteral("base_draft_revision"), 0},
        {QStringLiteral("progress"), QJsonObject{{QStringLiteral("completed"), 1},
                                                   {QStringLiteral("total"), 1}}},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                        {QStringLiteral("status"), QStringLiteral("active")}}
        }}}}
    });
    page.setBackendAvailable(true, QString());
    const QJsonObject draftBefore = page.draft();
    const QJsonObject snapshotBefore = page.snapshot();
    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *canvas = page.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(table != nullptr);
    QVERIFY(canvas != nullptr);
    const int rowsBefore = table->rowCount();
    const QJsonObject guideBefore = canvas->coarseShape();
    const QJsonObject fittedBefore = canvas->fittedShape();
    const QJsonObject effectiveBefore = canvas->effectiveShape();

    page.setBackendAvailable(false, QStringLiteral("后端断开"));

    QCOMPARE(page.draft(), draftBefore);
    QCOMPARE(page.snapshot(), snapshotBefore);
    QCOMPARE(table->rowCount(), rowsBefore);
    QCOMPARE(canvas->coarseShape(), guideBefore);
    QCOMPARE(canvas->fittedShape(), fittedBefore);
    QCOMPARE(canvas->effectiveShape(), effectiveBefore);
    QVERIFY(!page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"))->isEnabled());
    QCOMPARE(page.findChild<QLabel *>(QStringLiteral("publishDisabledReasonLabel"))->text(),
             QStringLiteral("后端断开"));
}

void TestGeometryRulesPage::lockedDirectionChangeDoesNotRequestPreviewOrStickSelectors() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    auto *direction = page.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    auto *templates = page.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    QVERIFY(direction != nullptr);
    QVERIFY(templates != nullptr);
    QSignalSpy previewSpy(&page, &GeometryRulesPage::previewRequested);

    page.setEditingLocked(true);
    direction->setCurrentIndex(direction->findData(QStringLiteral("back")));

    QCOMPARE(previewSpy.count(), 0);
    QVERIFY(direction->isEnabled());
    QVERIFY(templates->isEnabled());
}

void TestGeometryRulesPage::advancedDiagnosticsAreCollapsedByDefault() {
    GeometryRulesPage page;
    page.show();
    QCoreApplication::processEvents();

    auto *advanced = page.findChild<QGroupBox *>(QStringLiteral("advancedGeometryGroup"));
    auto *content = page.findChild<QWidget *>(QStringLiteral("advancedGeometryContent"));
    QVERIFY(advanced != nullptr);
    QVERIFY(content != nullptr);
    QVERIFY(advanced->isCheckable());
    QVERIFY(!advanced->isChecked());
    QVERIFY(!content->isVisibleTo(&page));

    const QStringList advancedControlNames{
        QStringLiteral("anchorCandidateCombo"),
        QStringLiteral("ruleCandidateCombo"),
        QStringLiteral("reviewStateCombo"),
        QStringLiteral("reviewReasonEdit"),
        QStringLiteral("setAnchorButton"),
        QStringLiteral("manualAnchorButton"),
        QStringLiteral("versionCombo"),
        QStringLiteral("copyActiveToDraftButton"),
        QStringLiteral("saveButton"),
        QStringLiteral("validateButton"),
        QStringLiteral("rollbackButton"),
        QStringLiteral("migrationPanel"),
    };
    for (const QString &name : advancedControlNames) {
        QWidget *control = page.findChild<QWidget *>(name);
        QVERIFY2(control != nullptr, qPrintable(name));
        QVERIFY2(content->isAncestorOf(control), qPrintable(name));
    }
    auto *primary = page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    QVERIFY(primary != nullptr);
    QVERIFY(!content->isAncestorOf(primary));
}

void TestGeometryRulesPage::migrationControlsAppearOnlyWhenConflictExists() {
    GeometryRulesPage page;
    page.setSnapshot(configuredProfileSnapshot(QString()));
    page.show();
    QCoreApplication::processEvents();
    auto *panel = page.findChild<QWidget *>(QStringLiteral("migrationPanel"));
    QVERIFY(panel != nullptr);
    QVERIFY(!panel->isVisibleTo(&page));

    QJsonObject snapshot = configuredProfileSnapshot(QString());
    QJsonObject draft = snapshot.value(QStringLiteral("draft")).toObject();
    draft.insert(QStringLiteral("migration"), QJsonObject{
        {QStringLiteral("conflicts"), QJsonArray{QJsonObject{
            {QStringLiteral("conflict_id"), QStringLiteral("legacy-1")},
            {QStringLiteral("type"), QStringLiteral("missing_direction")},
            {QStringLiteral("sources"), QJsonArray()}}}}});
    snapshot.insert(QStringLiteral("draft"), draft);
    page.setSnapshot(snapshot);
    QCoreApplication::processEvents();
    QVERIFY(panel->isVisibleTo(&page));
}

void TestGeometryRulesPage::issueRowStoresAndSelectsExactContext() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-context")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("progress"), QJsonObject{{QStringLiteral("completed"), 1},
                                                   {QStringLiteral("total"), 1}}},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                        {QStringLiteral("status"), QStringLiteral("low_confidence")},
                        {QStringLiteral("rules"), QJsonArray{
                            QJsonObject{{QStringLiteral("rule_id"), QStringLiteral("glare")},
                                        {QStringLiteral("status"), QStringLiteral("active")}},
                            QJsonObject{{QStringLiteral("rule_id"), QStringLiteral("intrusion")},
                                        {QStringLiteral("status"), QStringLiteral("failed")},
                                        {QStringLiteral("reason_code"), QStringLiteral("edge_support_low")}}
                        }}}
        }}}}
    });

    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *direction = page.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    auto *templates = page.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    auto *rules = page.findChild<QListWidget *>(QStringLiteral("ruleList"));
    QVERIFY(table != nullptr);
    QVERIFY(direction != nullptr);
    QVERIFY(templates != nullptr);
    QVERIFY(rules != nullptr);
    QCOMPARE(table->rowCount(), 1);
    QTableWidgetItem *identity = table->item(0, 0);
    QVERIFY(identity != nullptr);
    QCOMPARE(identity->data(GeometryRulesPage::DirectionRole).toString(), QStringLiteral("front"));
    QCOMPARE(identity->data(GeometryRulesPage::TemplateIdRole).toString(), QStringLiteral("front:front.png"));
    QCOMPARE(identity->data(GeometryRulesPage::RuleIdRole).toString(), QStringLiteral("intrusion"));
    QCOMPARE(identity->data(GeometryRulesPage::ReasonCodeRole).toString(), QStringLiteral("edge_support_low"));

    QVERIFY(QMetaObject::invokeMethod(&page, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 0)));
    QCOMPARE(direction->currentData().toString(), QStringLiteral("front"));
    QCOMPARE(templates->currentData(Qt::UserRole + 1).toString(), QStringLiteral("front:front.png"));
    QVERIFY(rules->currentItem() != nullptr);
    QCOMPARE(rules->currentItem()->data(Qt::UserRole).toString(), QStringLiteral("intrusion"));
}

void TestGeometryRulesPage::templateLevelIssueDoesNotInventRuleSelection() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-template")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("back"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("back:back.png")},
                        {QStringLiteral("status"), QStringLiteral("low_confidence")},
                        {QStringLiteral("reason_code"), QStringLiteral("source_disagreement")}}
        }}}}
    });
    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *rules = page.findChild<QListWidget *>(QStringLiteral("ruleList"));
    QVERIFY(table != nullptr);
    QVERIFY(rules != nullptr);
    QCOMPARE(table->rowCount(), 1);
    QCOMPARE(table->item(0, 0)->data(GeometryRulesPage::RuleIdRole).toString(), QString());

    QVERIFY(QMetaObject::invokeMethod(&page, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 0)));
    QCOMPARE(rules->currentRow(), -1);
}

void TestGeometryRulesPage::validationStatesUseHumanReadableLabels_data() {
    QTest::addColumn<QString>("state");
    QTest::addColumn<QString>("reason");
    QTest::addColumn<QString>("expected");
    QTest::newRow("active") << QStringLiteral("active") << QString() << QStringLiteral("通过");
    QTest::newRow("low confidence") << QStringLiteral("low_confidence") << QString() << QStringLiteral("低置信度");
    QTest::newRow("not configured") << QStringLiteral("not_configured") << QString() << QStringLiteral("未配置");
    QTest::newRow("needs reseed") << QStringLiteral("needs_reseed") << QString() << QStringLiteral("拟合失败");
    QTest::newRow("failed") << QStringLiteral("failed") << QString() << QStringLiteral("拟合失败");
    QTest::newRow("excluded") << QStringLiteral("excluded") << QStringLiteral("人工排除反光")
                               << QStringLiteral("已排除");
}

void TestGeometryRulesPage::validationStatesUseHumanReadableLabels() {
    QFETCH(QString, state);
    QFETCH(QString, reason);
    QFETCH(QString, expected);
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    QJsonObject reportRow{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                          {QStringLiteral("status"), state}};
    if (state == QStringLiteral("excluded")) {
        reportRow.insert(QStringLiteral("review_reason"), reason);
    } else {
        reportRow.insert(QStringLiteral("reason_code"), reason);
    }
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-state")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{reportRow}}}}
    });
    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    QVERIFY(table != nullptr);
    QCOMPARE(table->rowCount(), 1);
    QCOMPARE(table->item(0, 2)->text(), expected);
    if (state == QStringLiteral("excluded")) {
        QVERIFY(table->item(0, 3)->text().contains(reason));
    }
}

void TestGeometryRulesPage::standaloneIssuesUseHumanReadableStatusAndReason() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-standalone")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("warnings"), QJsonArray{QJsonObject{
            {QStringLiteral("orientation"), QStringLiteral("front")},
            {QStringLiteral("index"), 0},
            {QStringLiteral("code"), QStringLiteral("keypoint_retention_low")}}}},
        {QStringLiteral("blocking_issues"), QJsonArray{QJsonObject{
            {QStringLiteral("orientation"), QStringLiteral("back")},
            {QStringLiteral("index"), 0},
            {QStringLiteral("code"), QStringLiteral("geometry_mask_too_large")}}}}
    });

    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    QVERIFY(table != nullptr);
    QCOMPARE(table->rowCount(), 2);
    QCOMPARE(table->item(0, 2)->text(), QStringLiteral("告警"));
    QCOMPARE(table->item(0, 3)->text(), QStringLiteral("关键点保留率偏低"));
    QCOMPARE(table->item(1, 2)->text(), QStringLiteral("阻断"));
    QCOMPARE(table->item(1, 3)->text(), QStringLiteral("忽略区域过大"));

    QTableWidgetItem *warningIdentity = table->item(0, 0);
    QVERIFY(warningIdentity != nullptr);
    QCOMPARE(warningIdentity->data(GeometryRulesPage::DirectionRole).toString(),
             QStringLiteral("front"));
    QCOMPARE(warningIdentity->data(GeometryRulesPage::TemplateIdRole).toString(),
             QStringLiteral("front:front.png"));
    QCOMPARE(warningIdentity->data(GeometryRulesPage::RuleIdRole).toString(), QString());
    QCOMPARE(warningIdentity->data(GeometryRulesPage::ReasonCodeRole).toString(),
             QStringLiteral("keypoint_retention_low"));
}

void TestGeometryRulesPage::leaveOneOutIssueCreatesOneLocatableRowPerTemplate() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-leave-one-out")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray{QJsonObject{
            {QStringLiteral("code"), QStringLiteral("leave_one_out_incomplete")},
            {QStringLiteral("status"), QStringLiteral("incomplete")},
            {QStringLiteral("skipped"), 2},
            {QStringLiteral("templates"), QJsonArray{
                QStringLiteral("front:front.png"),
                QStringLiteral("back:back.png")}}
        }}}
    });

    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *direction = page.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    auto *templates = page.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    QVERIFY(table != nullptr);
    QVERIFY(direction != nullptr);
    QVERIFY(templates != nullptr);
    QCOMPARE(table->rowCount(), 2);
    const QStringList expectedDirections{QStringLiteral("front"), QStringLiteral("back")};
    const QStringList expectedTemplates{QStringLiteral("front:front.png"),
                                        QStringLiteral("back:back.png")};
    for (int row = 0; row < 2; ++row) {
        QTableWidgetItem *identity = table->item(row, 0);
        QVERIFY(identity != nullptr);
        QCOMPARE(identity->data(GeometryRulesPage::DirectionRole).toString(),
                 expectedDirections.at(row));
        QCOMPARE(identity->data(GeometryRulesPage::TemplateIdRole).toString(),
                 expectedTemplates.at(row));
        QCOMPARE(identity->data(GeometryRulesPage::RuleIdRole).toString(), QString());
        QCOMPARE(identity->data(GeometryRulesPage::ReasonCodeRole).toString(),
                 QStringLiteral("leave_one_out_incomplete"));
        QCOMPARE(table->item(row, 2)->text(), QStringLiteral("阻断"));
        QCOMPARE(table->item(row, 3)->text(), QStringLiteral("留一验证未完成"));
        QVERIFY(QMetaObject::invokeMethod(&page, "selectValidationTemplate", Qt::DirectConnection,
                                          Q_ARG(int, row)));
        QCOMPARE(direction->currentData().toString(), expectedDirections.at(row));
        QCOMPARE(templates->currentData(Qt::UserRole + 1).toString(),
                 expectedTemplates.at(row));
    }
}

void TestGeometryRulesPage::unmatchedStandaloneTemplateDoesNotGuessDirectionFromItsName() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-unmatched-template")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray{QJsonObject{
            {QStringLiteral("code"), QStringLiteral("leave_one_out_incomplete")},
            {QStringLiteral("templates"), QJsonArray{
                QStringLiteral("front:not-in-snapshot.png")}}
        }}}
    });

    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    QVERIFY(table != nullptr);
    QCOMPARE(table->rowCount(), 1);
    QTableWidgetItem *identity = table->item(0, 0);
    QVERIFY(identity != nullptr);
    QCOMPARE(identity->data(GeometryRulesPage::DirectionRole).toString(), QString());
    QCOMPARE(identity->data(GeometryRulesPage::TemplateIdRole).toString(),
             QStringLiteral("front:not-in-snapshot.png"));
    QCOMPARE(identity->data(GeometryRulesPage::RuleIdRole).toString(), QString());
    QCOMPARE(identity->data(GeometryRulesPage::ReasonCodeRole).toString(),
             QStringLiteral("leave_one_out_incomplete"));
}

void TestGeometryRulesPage::regressionIssueUsesExpectedDirectionToLocateTemplate() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-regression-direction")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray{QJsonObject{
            {QStringLiteral("code"), QStringLiteral("geometry_fusion_regression")},
            {QStringLiteral("changed_predictions"), QJsonArray{QJsonObject{
                {QStringLiteral("template_id"), QStringLiteral("back:back.png")},
                {QStringLiteral("expected"), QStringLiteral("back")},
                {QStringLiteral("baseline_predicted"), QStringLiteral("back")},
                {QStringLiteral("candidate_predicted"), QStringLiteral("front")}
            }}}
        }}}
    });

    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *direction = page.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    auto *templates = page.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    QVERIFY(table != nullptr);
    QVERIFY(direction != nullptr);
    QVERIFY(templates != nullptr);
    QCOMPARE(table->rowCount(), 1);
    QCOMPARE(table->item(0, 0)->data(GeometryRulesPage::DirectionRole).toString(),
             QStringLiteral("back"));

    QVERIFY(QMetaObject::invokeMethod(&page, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 0)));
    QCOMPARE(direction->currentData().toString(), QStringLiteral("back"));
    QCOMPARE(templates->currentData(Qt::UserRole + 1).toString(),
             QStringLiteral("back:back.png"));
}

void TestGeometryRulesPage::directionIndexIssueResolvesTemplateWithoutGuessing() {
    GeometryRulesPage page;
    QJsonObject snapshot = configuredV2ProfileSnapshot(QString(), QString());
    QJsonArray templates = snapshot.value(QStringLiteral("templates")).toArray();
    templates.prepend(QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:first.png")},
                                  {QStringLiteral("direction"), QStringLiteral("front")},
                                  {QStringLiteral("path"), QString()}});
    snapshot.insert(QStringLiteral("templates"), templates);
    page.setSnapshot(snapshot);
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-index")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("warnings"), QJsonArray{
            QJsonObject{{QStringLiteral("code"), QStringLiteral("effective_area_low")},
                        {QStringLiteral("orientation"), QStringLiteral("front")},
                        {QStringLiteral("index"), 1}}
        }}
    });
    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    QVERIFY(table != nullptr);
    QCOMPARE(table->rowCount(), 1);
    QCOMPARE(table->item(0, 0)->data(GeometryRulesPage::DirectionRole).toString(),
             QStringLiteral("front"));
    QCOMPARE(table->item(0, 0)->data(GeometryRulesPage::TemplateIdRole).toString(),
             QStringLiteral("front:front.png"));
    QCOMPARE(table->item(0, 0)->data(GeometryRulesPage::RuleIdRole).toString(), QString());
}

QJsonObject profileSnapshot(int libraryRevision, int draftRevision, int activeRevision) {
    return QJsonObject{
        {"workpiece_id", "m7"}, {"library_revision", libraryRevision}, {"draft_revision", draftRevision},
        {"active_revision", activeRevision}, {"previous_active_revision", QJsonValue()},
        {"draft", QJsonObject{{"schema_version", 1}, {"directions", QJsonObject{
            {"front", QJsonObject{{"anchor", QJsonValue()}, {"rules", QJsonArray()}}},
            {"back", QJsonObject{{"anchor", QJsonValue()}, {"rules", QJsonArray()}}}}}}}
    };
}

QJsonObject configuredProfileSnapshot(const QString &imagePath, int draftRevision) {
    QJsonObject snapshot = profileSnapshot(1, draftRevision, 0);
    snapshot.insert(QStringLiteral("templates"), QJsonArray{
        QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                    {QStringLiteral("direction"), QStringLiteral("front")},
                    {QStringLiteral("path"), imagePath}}
    });
    QJsonObject draft = snapshot.value(QStringLiteral("draft")).toObject();
    QJsonObject directions = draft.value(QStringLiteral("directions")).toObject();
    const QJsonObject anchor{
        {QStringLiteral("shape"), QStringLiteral("circle")},
        {QStringLiteral("coarse"), QJsonObject{
            {QStringLiteral("cx"), 0.5}, {QStringLiteral("cy"), 0.5},
            {QStringLiteral("r"), 0.45}, {QStringLiteral("angle_deg"), 0.0}}}
    };
    const QJsonObject rule{
        {QStringLiteral("rule_id"), QStringLiteral("inner-glare")},
        {QStringLiteral("name"), QStringLiteral("中心反光")},
        {QStringLiteral("shape"), QStringLiteral("circle")},
        {QStringLiteral("mode"), QStringLiteral("inside")},
        {QStringLiteral("margin_ratio"), 0.0},
        {QStringLiteral("enabled"), true},
        {QStringLiteral("geometry"), QJsonObject{
            {QStringLiteral("cx"), 0.0}, {QStringLiteral("cy"), 0.0},
            {QStringLiteral("r"), 0.70}, {QStringLiteral("angle_deg"), 0.0}}}
    };
    directions.insert(QStringLiteral("front"), QJsonObject{
        {QStringLiteral("anchor"), anchor},
        {QStringLiteral("rules"), QJsonArray{rule}}
    });
    draft.insert(QStringLiteral("directions"), directions);
    snapshot.insert(QStringLiteral("draft"), draft);
    return snapshot;
}

QJsonObject configuredV2ProfileSnapshot(const QString &frontPath, const QString &backPath) {
    QJsonObject snapshot = profileSnapshot(4, 2, 0);
    snapshot.insert(QStringLiteral("templates"), QJsonArray{
        QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                    {QStringLiteral("direction"), QStringLiteral("front")},
                    {QStringLiteral("path"), frontPath}},
        QJsonObject{{QStringLiteral("template_id"), QStringLiteral("back:back.png")},
                    {QStringLiteral("direction"), QStringLiteral("back")},
                    {QStringLiteral("path"), backPath}}
    });
    const QJsonObject anchor{
        {QStringLiteral("shape"), QStringLiteral("circle")},
        {QStringLiteral("mode"), QStringLiteral("auto")},
        {QStringLiteral("coarse"), QJsonObject{
            {QStringLiteral("cx"), 0.5}, {QStringLiteral("cy"), 0.5},
            {QStringLiteral("r"), 0.45}, {QStringLiteral("angle_deg"), 0.0}}}
    };
    const QJsonArray rules{
        QJsonObject{{QStringLiteral("rule_id"), QStringLiteral("glare")},
                    {QStringLiteral("name"), QStringLiteral("中心反光")},
                    {QStringLiteral("shape"), QStringLiteral("circle")},
                    {QStringLiteral("mode"), QStringLiteral("inside")},
                    {QStringLiteral("margin_ratio"), -0.04},
                    {QStringLiteral("margin_semantics"), QStringLiteral("signed_boundary_v2")},
                    {QStringLiteral("enabled"), true}},
        QJsonObject{{QStringLiteral("rule_id"), QStringLiteral("intrusion")},
                    {QStringLiteral("name"), QStringLiteral("四周侵入")},
                    {QStringLiteral("shape"), QStringLiteral("circle")},
                    {QStringLiteral("mode"), QStringLiteral("outside")},
                    {QStringLiteral("margin_ratio"), 0.02},
                    {QStringLiteral("margin_semantics"), QStringLiteral("signed_boundary_v2")},
                    {QStringLiteral("enabled"), true}}
    };
    auto calibration = [](const QString &side, qreal radius) {
        return QJsonObject{
            {QStringLiteral("state"), QStringLiteral("ready")},
            {QStringLiteral("geometry"), QJsonObject{
                {QStringLiteral("cx"), 0.0}, {QStringLiteral("cy"), 0.0},
                {QStringLiteral("r"), radius}, {QStringLiteral("angle_deg"), 0.0}}},
            {QStringLiteral("seed_geometry"), QJsonObject{
                {QStringLiteral("cx"), 0.0}, {QStringLiteral("cy"), 0.0},
                {QStringLiteral("r"), radius + 0.05}, {QStringLiteral("angle_deg"), 0.0}}},
            {QStringLiteral("reference_template"), QJsonObject{
                {QStringLiteral("template_id"), side + QStringLiteral(":")
                    + (side == QStringLiteral("front") ? QStringLiteral("front.png") : QStringLiteral("back.png"))},
                {QStringLiteral("direction"), side}, {QStringLiteral("width"), 120},
                {QStringLiteral("height"), 120}}},
            {QStringLiteral("diagnostics"), QJsonObject()}
        };
    };
    const QJsonObject frontCalibrations{
        {QStringLiteral("glare"), calibration(QStringLiteral("front"), 0.70)},
        {QStringLiteral("intrusion"), calibration(QStringLiteral("front"), 0.94)}
    };
    const QJsonObject backCalibrations{
        {QStringLiteral("glare"), calibration(QStringLiteral("back"), 0.64)},
        {QStringLiteral("intrusion"), calibration(QStringLiteral("back"), 0.91)}
    };
    snapshot.insert(QStringLiteral("draft"), QJsonObject{
        {QStringLiteral("schema_version"), 2},
        {QStringLiteral("rules"), rules},
        {QStringLiteral("directions"), QJsonObject{
            {QStringLiteral("front"), QJsonObject{{QStringLiteral("anchor"), anchor},
                {QStringLiteral("calibrations"), frontCalibrations},
                {QStringLiteral("template_reviews"), QJsonObject()}}},
            {QStringLiteral("back"), QJsonObject{{QStringLiteral("anchor"), anchor},
                {QStringLiteral("calibrations"), backCalibrations},
                {QStringLiteral("template_reviews"), QJsonObject()}}}
        }},
        {QStringLiteral("migration"), QJsonObject{{QStringLiteral("conflicts"), QJsonArray()},
                                                    {QStringLiteral("resolutions"), QJsonArray()}}}
    });
    return snapshot;
}

void TestGeometryRulesPage::publishDisabledUntilCompletedValidationMatchesDraft() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 1));
    QVERIFY(!dialog.findChild<QPushButton *>(QStringLiteral("publishButton"))->isEnabled());
    dialog.setValidationJob(QJsonObject{{"job_id", "job-1"}, {"state", "completed"},
                                        {"base_library_revision", 4}, {"base_draft_revision", 2},
                                        {"progress", QJsonObject{{"completed", 2}, {"total", 2}}}});
    QVERIFY(dialog.findChild<QPushButton *>(QStringLiteral("publishButton"))->isEnabled());
}

void TestGeometryRulesPage::draftEditInvalidatesVisibleValidationResult() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 1));
    const QJsonObject job{
        {QStringLiteral("job_id"), QStringLiteral("job-1")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("progress"), QJsonObject{{QStringLiteral("completed"), 1},
                                                  {QStringLiteral("total"), 1}}},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:00.png")},
                        {QStringLiteral("status"), QStringLiteral("active")}}
        }}}}
    };
    dialog.setValidationJob(job);

    auto *table = dialog.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *progress = dialog.findChild<QProgressBar *>();
    auto *validationHint = dialog.findChild<QLabel *>(QStringLiteral("validationHintLabel"));
    auto *validate = dialog.findChild<QPushButton *>(QStringLiteral("validateButton"));
    auto *publish = dialog.findChild<QPushButton *>(QStringLiteral("publishButton"));
    QVERIFY(table != nullptr);
    QVERIFY(progress != nullptr);
    QVERIFY(validationHint != nullptr);
    QCOMPARE(table->rowCount(), 1);
    QCOMPARE(progress->value(), 1);
    QVERIFY(publish->isEnabled());

    dialog.findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();

    QCOMPARE(table->rowCount(), 0);
    QCOMPARE(progress->value(), 0);
    QVERIFY(validationHint->text().contains(QStringLiteral("旧验证结果已失效")));
    QVERIFY(!validate->isEnabled());
    QVERIFY(!publish->isEnabled());
}

void TestGeometryRulesPage::lateValidationUpdateStaysHiddenAfterDraftEdit() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 1));
    const QJsonObject job{
        {QStringLiteral("job_id"), QStringLiteral("job-1")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("progress"), QJsonObject{{QStringLiteral("completed"), 1},
                                                  {QStringLiteral("total"), 1}}},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("back"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("back:00.png")},
                        {QStringLiteral("status"), QStringLiteral("not_configured")}}
        }}}}
    };
    dialog.setValidationJob(job);
    dialog.findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();

    dialog.setValidationJob(job);

    auto *table = dialog.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *validationHint = dialog.findChild<QLabel *>(QStringLiteral("validationHintLabel"));
    QCOMPARE(table->rowCount(), 0);
    QVERIFY(validationHint->text().contains(QStringLiteral("旧验证结果已失效")));
}

void TestGeometryRulesPage::warningPublishRequiresOverrideReason() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 1));
    dialog.setValidationJob(QJsonObject{{"job_id", "job-1"}, {"state", "completed"},
                                        {"base_library_revision", 4}, {"base_draft_revision", 2},
                                        {"warnings", QJsonArray{QJsonObject{{"code", "effective_area_low"}}}},
                                        {"progress", QJsonObject{{"completed", 2}, {"total", 2}}}});
    QSignalSpy spy(&dialog, &GeometryRulesPage::publishRequested);
    dialog.findChild<QPushButton *>(QStringLiteral("publishButton"))->click();
    QCOMPARE(spy.count(), 0);
}

void TestGeometryRulesPage::warningPublishRequirementIsVisible() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 1));
    dialog.setValidationJob(QJsonObject{{"job_id", "job-1"}, {"state", "completed"},
                                        {"base_library_revision", 4}, {"base_draft_revision", 2},
                                        {"warnings", QJsonArray{QJsonObject{{"code", "effective_area_low"}}}},
                                        {"progress", QJsonObject{{"completed", 2}, {"total", 2}}}});
    auto *label = dialog.findChild<QLabel *>(QStringLiteral("overrideReasonLabel"));
    auto *edit = dialog.findChild<QLineEdit *>(QStringLiteral("overrideReasonEdit"));
    auto *primary = dialog.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    auto *disabledReason = dialog.findChild<QLabel *>(QStringLiteral("publishDisabledReasonLabel"));
    QVERIFY(label != nullptr);
    QVERIFY(edit != nullptr);
    QVERIFY(primary != nullptr);
    QVERIFY(disabledReason != nullptr);
    QCOMPARE(label->text(), QStringLiteral("发布覆盖原因（有告警或回归时必填）"));
    QCOMPARE(edit->placeholderText(), QStringLiteral("请说明已检查告警/回归并确认发布"));
    QVERIFY(!dialog.findChild<QPushButton *>(QStringLiteral("publishButton"))->isEnabled());
    QVERIFY(!primary->isEnabled());
    QVERIFY(disabledReason->text().contains(QStringLiteral("发布覆盖原因")));
    edit->setText(QStringLiteral("已检查有效区域告警，确认继续发布"));
    QVERIFY(dialog.findChild<QPushButton *>(QStringLiteral("publishButton"))->isEnabled());
    QVERIFY(primary->isEnabled());
}

void TestGeometryRulesPage::warningContinuationUsesPrimaryWithoutResaving() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("warning-job")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray()},
        {QStringLiteral("warnings"), QJsonArray{
            QJsonObject{{QStringLiteral("code"), QStringLiteral("effective_area_low")}}}}
    });
    auto *primary = page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    auto *reason = page.findChild<QLineEdit *>(QStringLiteral("overrideReasonEdit"));
    QVERIFY(primary != nullptr);
    QVERIFY(reason != nullptr);
    QSignalSpy workflowSpy(&page, &GeometryRulesPage::publishWorkflowRequested);
    QSignalSpy publishSpy(&page, &GeometryRulesPage::publishRequested);

    reason->setText(QStringLiteral("已人工复核"));
    primary->click();

    QCOMPARE(workflowSpy.count(), 0);
    QCOMPARE(publishSpy.count(), 1);
    const QList<QVariant> arguments = publishSpy.takeFirst();
    QCOMPARE(arguments.at(0).toString(), QStringLiteral("warning-job"));
    QCOMPARE(arguments.at(3).toString(), QStringLiteral("已人工复核"));
}

void TestGeometryRulesPage::warningWithoutJobIdentityStartsANewWorkflow() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray()},
        {QStringLiteral("warnings"), QJsonArray{
            QJsonObject{{QStringLiteral("code"), QStringLiteral("effective_area_low")}}}}
    });
    auto *primary = page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    auto *reason = page.findChild<QLineEdit *>(QStringLiteral("overrideReasonEdit"));
    QVERIFY(primary != nullptr);
    QVERIFY(reason != nullptr);
    QSignalSpy workflowSpy(&page, &GeometryRulesPage::publishWorkflowRequested);
    QSignalSpy publishSpy(&page, &GeometryRulesPage::publishRequested);

    reason->setText(QStringLiteral("没有可恢复的验证任务"));
    primary->click();

    QCOMPARE(publishSpy.count(), 0);
    QCOMPARE(workflowSpy.count(), 1);
}

void TestGeometryRulesPage::newWarningJobClearsPreviousOverrideReason() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    const auto warningJob = [](const QString &jobId) {
        return QJsonObject{
            {QStringLiteral("job_id"), jobId},
            {QStringLiteral("state"), QStringLiteral("completed")},
            {QStringLiteral("base_library_revision"), 4},
            {QStringLiteral("base_draft_revision"), 2},
            {QStringLiteral("blocking_issues"), QJsonArray()},
            {QStringLiteral("warnings"), QJsonArray{QJsonObject{
                {QStringLiteral("code"), QStringLiteral("effective_area_low")}}}}
        };
    };
    auto *reason = page.findChild<QLineEdit *>(QStringLiteral("overrideReasonEdit"));
    auto *primary = page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    QVERIFY(reason != nullptr);
    QVERIFY(primary != nullptr);

    page.setValidationJob(warningJob(QStringLiteral("warning-job-1")));
    reason->setText(QStringLiteral("旧任务人工复核"));
    QVERIFY(primary->isEnabled());
    page.setValidationJob(warningJob(QStringLiteral("warning-job-2")));

    QVERIFY(reason->text().isEmpty());
    QVERIFY(!primary->isEnabled());
}

void TestGeometryRulesPage::draftMutationClearsWarningOverrideReason() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("warning-job")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray()},
        {QStringLiteral("warnings"), QJsonArray{QJsonObject{
            {QStringLiteral("code"), QStringLiteral("effective_area_low")}}}}
    });
    auto *reason = page.findChild<QLineEdit *>(QStringLiteral("overrideReasonEdit"));
    auto *addRule = page.findChild<QPushButton *>(QStringLiteral("addRuleButton"));
    QVERIFY(reason != nullptr);
    QVERIFY(addRule != nullptr);
    reason->setText(QStringLiteral("仅适用于旧草稿"));

    addRule->click();

    QVERIFY(reason->text().isEmpty());
}

void TestGeometryRulesPage::newSnapshotClearsWarningOverrideReason() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("warning-job")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray()},
        {QStringLiteral("warnings"), QJsonArray{QJsonObject{
            {QStringLiteral("code"), QStringLiteral("effective_area_low")}}}}
    });
    auto *reason = page.findChild<QLineEdit *>(QStringLiteral("overrideReasonEdit"));
    QVERIFY(reason != nullptr);
    reason->setText(QStringLiteral("仅适用于旧快照"));

    QJsonObject next = configuredV2ProfileSnapshot(QString(), QString());
    next.insert(QStringLiteral("workpiece_id"), QStringLiteral("m2"));
    page.setSnapshot(next);

    QVERIFY(reason->text().isEmpty());
}

void TestGeometryRulesPage::clearingContinuationClearsWarningOverrideReason() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    page.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("warning-job")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray()},
        {QStringLiteral("warnings"), QJsonArray{QJsonObject{
            {QStringLiteral("code"), QStringLiteral("effective_area_low")}}}}
    });
    auto *reason = page.findChild<QLineEdit *>(QStringLiteral("overrideReasonEdit"));
    QVERIFY(reason != nullptr);
    reason->setText(QStringLiteral("即将失效"));

    page.clearPublishContinuation();

    QVERIFY(reason->text().isEmpty());
}

void TestGeometryRulesPage::ordinaryWorkflowNeverEmitsAStaleOverrideReason() {
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    auto *reason = page.findChild<QLineEdit *>(QStringLiteral("overrideReasonEdit"));
    auto *primary = page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    QVERIFY(reason != nullptr);
    QVERIFY(primary != nullptr);
    reason->setText(QStringLiteral("没有当前告警上下文的旧原因"));
    QSignalSpy workflowSpy(&page, &GeometryRulesPage::publishWorkflowRequested);

    primary->click();

    QCOMPARE(workflowSpy.count(), 1);
    QCOMPARE(workflowSpy.takeFirst().at(3).toString(), QString());
}

void TestGeometryRulesPage::saveAndDeleteRulesUpdateDraft() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(1, 0, 0));
    auto *add = dialog.findChild<QPushButton *>(QStringLiteral("addRuleButton"));
    auto *remove = dialog.findChild<QPushButton *>(QStringLiteral("deleteRuleButton"));
    QVERIFY(add != nullptr);
    QVERIFY(remove != nullptr);
    add->click();
    QVERIFY(!dialog.draft().value("directions").toObject().value("front").toObject()
                .value("rules").toArray().isEmpty());
    remove->click();
    QVERIFY(dialog.draft().value("directions").toObject().value("front").toObject()
                .value("rules").toArray().isEmpty());
}

void TestGeometryRulesPage::templatePreviewLoadsRepresentativeImage() {
    QTemporaryDir directory;
    const QString imagePath = directory.filePath(QStringLiteral("front.png"));
    QImage image(80, 60, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(imagePath));
    QJsonObject snapshot = profileSnapshot(1, 0, 0);
    snapshot.insert(QStringLiteral("templates"), QJsonArray{
        QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                    {QStringLiteral("direction"), QStringLiteral("front")},
                    {QStringLiteral("path"), imagePath}}
    });

    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    auto *combo = dialog.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(combo != nullptr);
    QVERIFY(canvas != nullptr);
    QCOMPARE(combo->count(), 1);
    QCOMPARE(canvas->image().size(), QSize(80, 60));
}

void TestGeometryRulesPage::validationRowSelectsTemplateForReview() {
    QTemporaryDir directory;
    const QString firstPath = directory.filePath(QStringLiteral("front-00.png"));
    const QString secondPath = directory.filePath(QStringLiteral("front-01.png"));
    QImage image(80, 60, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(firstPath));
    QVERIFY(image.save(secondPath));

    QJsonObject snapshot = profileSnapshot(4, 2, 0);
    snapshot.insert(QStringLiteral("templates"), QJsonArray{
        QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front-00.png")},
                    {QStringLiteral("direction"), QStringLiteral("front")},
                    {QStringLiteral("path"), firstPath}},
        QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front-01.png")},
                    {QStringLiteral("direction"), QStringLiteral("front")},
                    {QStringLiteral("path"), secondPath}}
    });

    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    dialog.setValidationJob(QJsonObject{
        {QStringLiteral("job_id"), QStringLiteral("job-1")},
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("progress"), QJsonObject{{QStringLiteral("completed"), 2}, {QStringLiteral("total"), 2}}},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front-00.png")},
                        {QStringLiteral("status"), QStringLiteral("active")},
                        {QStringLiteral("review_state"), QStringLiteral("included")}},
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front-01.png")},
                        {QStringLiteral("status"), QStringLiteral("low_confidence")},
                        {QStringLiteral("review_state"), QStringLiteral("review")}}
        }}}}
    });

    auto *table = dialog.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *templates = dialog.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    QVERIFY(table != nullptr);
    QVERIFY(templates != nullptr);
    QCOMPARE(table->rowCount(), 2);
    QCOMPARE(templates->count(), 2);
    QCOMPARE(templates->findData(QStringLiteral("front:front-01.png"), Qt::UserRole + 1), 1);
    QVERIFY(QMetaObject::invokeMethod(&dialog, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 1)));
    QCoreApplication::processEvents();
    QCOMPARE(templates->currentData(Qt::UserRole + 1).toString(), QStringLiteral("front:front-01.png"));
}

void TestGeometryRulesPage::publishWorkflowButtonEmitsDraft() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 0));
    auto *button = dialog.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    QVERIFY(button != nullptr);
    QVERIFY(button->isEnabled());
    QSignalSpy spy(&dialog, &GeometryRulesPage::publishWorkflowRequested);
    button->click();
    QCOMPARE(spy.count(), 1);
    const QList<QVariant> arguments = spy.takeFirst();
    QCOMPARE(arguments.at(1).toInt(), 4);
    QCOMPARE(arguments.at(2).toInt(), 2);
}

void TestGeometryRulesPage::validationRowSelectsTemplateAcrossDirections() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(80, 60, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    QJsonObject snapshot = profileSnapshot(4, 2, 0);
    snapshot.insert(QStringLiteral("templates"), QJsonArray{
        QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                    {QStringLiteral("direction"), QStringLiteral("front")},
                    {QStringLiteral("path"), frontPath}},
        QJsonObject{{QStringLiteral("template_id"), QStringLiteral("back:back.png")},
                    {QStringLiteral("direction"), QStringLiteral("back")},
                    {QStringLiteral("path"), backPath}}
    });

    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    QSignalSpy previewSpy(&dialog, &GeometryRulesPage::previewRequested);
    dialog.setValidationJob(QJsonObject{
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("report"), QJsonObject{
            {QStringLiteral("front"), QJsonArray{QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")}}}},
            {QStringLiteral("back"), QJsonArray{QJsonObject{{QStringLiteral("template_id"), QStringLiteral("back:back.png")}}}}
        }}
    });

    QVERIFY(QMetaObject::invokeMethod(&dialog, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 1)));
    auto *direction = dialog.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    auto *templates = dialog.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    QVERIFY(direction != nullptr);
    QVERIFY(templates != nullptr);
    QCOMPARE(direction->currentData().toString(), QStringLiteral("back"));
    QCOMPARE(templates->currentData(Qt::UserRole + 1).toString(), QStringLiteral("back:back.png"));
    QCOMPARE(previewSpy.count(), 0);
}

void TestGeometryRulesPage::ruleInventoryDoesNotChangeAcrossValidationDirections() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    GeometryRulesPage dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    dialog.setValidationJob(QJsonObject{
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("report"), QJsonObject{
            {QStringLiteral("front"), QJsonArray{QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")}}}},
            {QStringLiteral("back"), QJsonArray{QJsonObject{{QStringLiteral("template_id"), QStringLiteral("back:back.png")}}}}
        }}
    });
    auto *rules = dialog.findChild<QListWidget *>(QStringLiteral("ruleList"));
    QVERIFY(rules != nullptr);
    QCOMPARE(rules->count(), 2);
    const QString firstId = rules->item(0)->data(Qt::UserRole).toString();
    QCOMPARE(firstId, QStringLiteral("glare"));

    QVERIFY(QMetaObject::invokeMethod(&dialog, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 1)));
    QCOMPARE(rules->count(), 2);
    QCOMPARE(rules->item(0)->data(Qt::UserRole).toString(), firstId);
}

void TestGeometryRulesPage::logicalRuleWithOnlyFrontCalibrationRemainsVisibleOnBack() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    QJsonObject snapshot = configuredV2ProfileSnapshot(frontPath, backPath);
    QJsonObject draft = snapshot.value(QStringLiteral("draft")).toObject();
    QJsonArray rules = draft.value(QStringLiteral("rules")).toArray();
    rules = QJsonArray{rules.at(0)};
    draft.insert(QStringLiteral("rules"), rules);
    QJsonObject directions = draft.value(QStringLiteral("directions")).toObject();
    QJsonObject front = directions.value(QStringLiteral("front")).toObject();
    QJsonObject frontCalibrations = front.value(QStringLiteral("calibrations")).toObject();
    frontCalibrations.remove(QStringLiteral("intrusion"));
    front.insert(QStringLiteral("calibrations"), frontCalibrations);
    directions.insert(QStringLiteral("front"), front);
    QJsonObject back = directions.value(QStringLiteral("back")).toObject();
    back.insert(QStringLiteral("calibrations"), QJsonObject());
    directions.insert(QStringLiteral("back"), back);
    draft.insert(QStringLiteral("directions"), directions);
    snapshot.insert(QStringLiteral("draft"), draft);

    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    auto *rulesWidget = dialog.findChild<QListWidget *>(QStringLiteral("ruleList"));
    auto *direction = dialog.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    QVERIFY(rulesWidget != nullptr);
    QVERIFY(direction != nullptr);
    QCOMPARE(rulesWidget->count(), 1);
    QVERIFY(rulesWidget->item(0)->text().contains(QStringLiteral("正面:已标定")));
    QVERIFY(rulesWidget->item(0)->text().contains(QStringLiteral("反面:缺失")));
    const QString ruleId = rulesWidget->item(0)->data(Qt::UserRole).toString();

    direction->setCurrentIndex(direction->findData(QStringLiteral("back")));

    QCOMPARE(rulesWidget->count(), 1);
    QCOMPARE(rulesWidget->item(0)->data(Qt::UserRole).toString(), ruleId);
    QVERIFY(rulesWidget->item(0)->text().contains(QStringLiteral("正面:已标定")));
    QVERIFY(rulesWidget->item(0)->text().contains(QStringLiteral("反面:缺失")));
}

void TestGeometryRulesPage::deleteLogicalRuleRemovesBothCalibrations() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    GeometryRulesPage dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    auto *rules = dialog.findChild<QListWidget *>(QStringLiteral("ruleList"));
    QVERIFY(rules != nullptr);
    rules->setCurrentRow(0);
    dialog.findChild<QPushButton *>(QStringLiteral("deleteRuleButton"))->click();

    const QJsonObject draft = dialog.draft();
    QCOMPARE(draft.value(QStringLiteral("rules")).toArray().size(), 1);
    QVERIFY(!draft.value(QStringLiteral("directions")).toObject().value(QStringLiteral("front")).toObject()
                 .value(QStringLiteral("calibrations")).toObject().contains(QStringLiteral("glare")));
    QVERIFY(!draft.value(QStringLiteral("directions")).toObject().value(QStringLiteral("back")).toObject()
                 .value(QStringLiteral("calibrations")).toObject().contains(QStringLiteral("glare")));
}

void TestGeometryRulesPage::signedMarginSpinBoxAcceptsNegativeZeroAndPositive() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(1, 0, 0));
    auto *spin = dialog.findChild<QSpinBox *>(QStringLiteral("marginSpinBox"));
    QVERIFY(spin != nullptr);
    QCOMPARE(spin->minimum(), -94);
    QCOMPARE(spin->maximum(), 94);
    spin->setValue(-2);
    QCOMPARE(spin->value(), -2);
    spin->setValue(0);
    QCOMPARE(spin->value(), 0);
    spin->setValue(2);
    QCOMPARE(spin->value(), 2);
}

void TestGeometryRulesPage::newRuleUsesSignedMarkerAndZeroDefault() {
    GeometryRulesPage dialog;
    dialog.setSnapshot(profileSnapshot(1, 0, 0));
    dialog.findChild<QPushButton *>(QStringLiteral("addRuleButton"))->click();
    const QJsonObject rule = dialog.draft().value("directions").toObject()
        .value("front").toObject().value("rules").toArray().first().toObject();
    QCOMPARE(rule.value("margin_ratio").toDouble(), 0.0);
    QCOMPARE(rule.value("margin_semantics").toString(), QStringLiteral("signed_boundary_v2"));
}

void TestGeometryRulesPage::savedDraftHidesGuideAndShowsFittedBoundary() {
    QTemporaryDir directory;
    const QString imagePath = directory.filePath(QStringLiteral("front.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(imagePath));

    QJsonObject snapshot = configuredProfileSnapshot(imagePath);
    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(canvas != nullptr);

    const QJsonObject coarse{{QStringLiteral("shape"), QStringLiteral("circle")},
                             {QStringLiteral("cx"), 60.0},
                             {QStringLiteral("cy"), 60.0},
                             {QStringLiteral("r"), 38.0}};
    const QJsonObject effective{{QStringLiteral("shape"), QStringLiteral("circle")},
                                {QStringLiteral("cx"), 60.0},
                                {QStringLiteral("cy"), 60.0},
                                {QStringLiteral("r"), 31.0}};
    canvas->setCoarseShape(coarse);
    dialog.setRulePreview(QJsonObject{
        {QStringLiteral("template_id"), QStringLiteral("front:front.png")},
        {QStringLiteral("direction"), QStringLiteral("front")},
        {QStringLiteral("profile_patch"), QJsonObject{
            {QStringLiteral("geometry"), QJsonObject{
                {QStringLiteral("cx"), 0.0}, {QStringLiteral("cy"), 0.0},
                {QStringLiteral("r"), 31.0 / 54.0}, {QStringLiteral("angle_deg"), 0.0}}},
            {QStringLiteral("seed_geometry"), QJsonObject{
                {QStringLiteral("cx"), 0.0}, {QStringLiteral("cy"), 0.0},
                {QStringLiteral("r"), 38.0 / 54.0}, {QStringLiteral("angle_deg"), 0.0}}},
            {QStringLiteral("reference_template"), QJsonObject{
                {QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                {QStringLiteral("direction"), QStringLiteral("front")},
                {QStringLiteral("width"), 120}, {QStringLiteral("height"), 120}}}}},
        {QStringLiteral("rule_fit"), QJsonObject{
            {QStringLiteral("fitted_shape"), effective},
            {QStringLiteral("effective_shape"), effective}}}
    });
    QCOMPARE(canvas->fitShape().value(QStringLiteral("r")).toDouble(), 31.0);

    QJsonObject savedSnapshot = snapshot;
    savedSnapshot.insert(QStringLiteral("draft_revision"), 1);
    savedSnapshot.insert(QStringLiteral("draft"), dialog.draft());
    dialog.setSnapshot(savedSnapshot);

    QVERIFY(!canvas->coarseShape().isEmpty());
    QVERIFY(!canvas->guideVisible());
    QVERIFY(canvas->fittedBoundaryVisible());
    QCOMPARE(canvas->fitShape().value(QStringLiteral("r")).toDouble(), 31.0);
}

void TestGeometryRulesPage::validationTemplateShowsFittedBoundaryInsteadOfCoarseGuide() {
    QTemporaryDir directory;
    const QString imagePath = directory.filePath(QStringLiteral("front.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(imagePath));

    GeometryRulesPage dialog;
    dialog.setSnapshot(configuredProfileSnapshot(imagePath, 1));
    const QJsonObject effective{{QStringLiteral("shape"), QStringLiteral("ellipse")},
                                {QStringLiteral("cx"), 60.0}, {QStringLiteral("cy"), 60.0},
                                {QStringLiteral("rx"), 31.0}, {QStringLiteral("ry"), 29.0},
                                {QStringLiteral("angle_deg"), 4.0}};
    dialog.setValidationJob(QJsonObject{
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 1},
        {QStringLiteral("base_draft_revision"), 1},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                        {QStringLiteral("status"), QStringLiteral("active")},
                        {QStringLiteral("rules"), QJsonArray{QJsonObject{
                            {QStringLiteral("rule_id"), QStringLiteral("inner-glare")},
                            {QStringLiteral("status"), QStringLiteral("active")},
                            {QStringLiteral("effective_shape"), effective}}}}}
        }}}}
    });
    QVERIFY(QMetaObject::invokeMethod(&dialog, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 0)));
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(canvas != nullptr);
    QVERIFY(!canvas->coarseShape().isEmpty());
    QVERIFY(!canvas->guideVisible());
    QVERIFY(canvas->fittedBoundaryVisible());
    QCOMPARE(canvas->fitShape().value(QStringLiteral("rx")).toDouble(), 31.0);
}

void TestGeometryRulesPage::layerVisibilityChangesDoNotDirtyDraft() {
    GeometryRulesPage page;
    page.setSnapshot(configuredProfileSnapshot(QString()));
    auto *canvas = page.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    auto *guide = page.findChild<QCheckBox *>(QStringLiteral("guideLayerToggle"));
    auto *fitted = page.findChild<QCheckBox *>(QStringLiteral("fittedLayerToggle"));
    auto *effective = page.findChild<QCheckBox *>(QStringLiteral("effectiveLayerToggle"));
    auto *mask = page.findChild<QCheckBox *>(QStringLiteral("maskLayerToggle"));
    QVERIFY(canvas != nullptr);
    QVERIFY(guide != nullptr);
    QVERIFY(fitted != nullptr);
    QVERIFY(effective != nullptr);
    QVERIFY(mask != nullptr);
    QVERIFY(!page.hasUnsavedChanges());

    guide->setChecked(true);
    fitted->setChecked(false);
    effective->setChecked(false);
    mask->setChecked(false);

    QVERIFY(canvas->guideVisible());
    QVERIFY(!canvas->fittedBoundaryVisible());
    QVERIFY(!canvas->effectiveBoundaryVisible());
    QVERIFY(!canvas->maskOverlayVisible());
    QVERIFY(!page.hasUnsavedChanges());
}

void TestGeometryRulesPage::layerVisibilityPersistsAcrossTemplateRefresh() {
    QTemporaryDir directory;
    const QString firstPath = directory.filePath(QStringLiteral("front.png"));
    const QString secondPath = directory.filePath(QStringLiteral("front-01.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(firstPath));
    QVERIFY(image.save(secondPath));
    QJsonObject snapshot = configuredProfileSnapshot(firstPath);
    QJsonArray templates = snapshot.value(QStringLiteral("templates")).toArray();
    templates.append(QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front-01.png")},
                                 {QStringLiteral("direction"), QStringLiteral("front")},
                                 {QStringLiteral("path"), secondPath}});
    snapshot.insert(QStringLiteral("templates"), templates);
    GeometryRulesPage page;
    page.setSnapshot(snapshot);
    auto *canvas = page.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    auto *combo = page.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    auto *effective = page.findChild<QCheckBox *>(QStringLiteral("effectiveLayerToggle"));
    QVERIFY(canvas != nullptr);
    QVERIFY(combo != nullptr);
    QVERIFY(effective != nullptr);
    effective->setChecked(false);

    combo->setCurrentIndex(1);

    QVERIFY(!effective->isChecked());
    QVERIFY(!canvas->effectiveBoundaryVisible());
    QVERIFY(!page.hasUnsavedChanges());
}

void TestGeometryRulesPage::directionAndTemplateSelectorsStayWithCanvas() {
    GeometryRulesPage page;
    auto *canvas = page.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    auto *direction = page.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    auto *templates = page.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    QVERIFY(canvas != nullptr);
    QVERIFY(direction != nullptr);
    QVERIFY(templates != nullptr);
    QCOMPARE(direction->parentWidget(), canvas->parentWidget());
    QCOMPARE(templates->parentWidget(), canvas->parentWidget());
}

void TestGeometryRulesPage::validationSummaryStaysInRightColumn() {
    GeometryRulesPage page;
    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *status = page.findChild<QLabel *>(QStringLiteral("geometryStatusLabel"));
    auto *progress = page.findChild<QProgressBar *>();
    QVERIFY(table != nullptr);
    QWidget *right = table->parentWidget();
    QVERIFY(right != nullptr);
    QVERIFY(status != nullptr);
    QVERIFY(progress != nullptr);
    QVERIFY(right->isAncestorOf(status));
    QVERIFY(right->isAncestorOf(progress));
}

void TestGeometryRulesPage::publishDisabledReasonStaysInRightColumn() {
    GeometryRulesPage page;
    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *disabledReason = page.findChild<QLabel *>(
        QStringLiteral("publishDisabledReasonLabel"));
    QVERIFY(table != nullptr);
    QWidget *right = table->parentWidget();
    QVERIFY(right != nullptr);
    QVERIFY(disabledReason != nullptr);
    QVERIFY(right->isAncestorOf(disabledReason));
}

void TestGeometryRulesPage::primaryWorkflowStaysInRightColumn() {
    GeometryRulesPage page;
    auto *table = page.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *primary = page.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"));
    QVERIFY(table != nullptr);
    QWidget *right = table->parentWidget();
    QVERIFY(right != nullptr);
    QVERIFY(primary != nullptr);
    QVERIFY(right->isAncestorOf(primary));
}

void TestGeometryRulesPage::failedValidationTemplateDoesNotFallBackToCoarseGuide() {
    QTemporaryDir directory;
    const QString imagePath = directory.filePath(QStringLiteral("front.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(imagePath));

    GeometryRulesPage dialog;
    dialog.setSnapshot(configuredProfileSnapshot(imagePath, 1));
    dialog.setValidationJob(QJsonObject{
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 1},
        {QStringLiteral("base_draft_revision"), 1},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                        {QStringLiteral("status"), QStringLiteral("low_confidence")},
                        {QStringLiteral("reason_code"), QStringLiteral("boundary_not_found")},
                        {QStringLiteral("rules"), QJsonArray{QJsonObject{
                            {QStringLiteral("rule_id"), QStringLiteral("inner-glare")},
                            {QStringLiteral("status"), QStringLiteral("low_confidence")},
                            {QStringLiteral("reason_code"), QStringLiteral("topology_constraint_failed")}}}}}
        }}}}
    });
    QVERIFY(QMetaObject::invokeMethod(&dialog, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 0)));
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    auto *diagnostics = dialog.findChild<QTextEdit *>();
    QVERIFY(canvas != nullptr);
    QVERIFY(diagnostics != nullptr);
    QVERIFY(!canvas->coarseShape().isEmpty());
    QVERIFY(!canvas->guideVisible());
    QVERIFY(canvas->fitShape().isEmpty());
    QVERIFY(diagnostics->toPlainText().contains(QStringLiteral("topology_constraint_failed")));
}

void TestGeometryRulesPage::selectingTemplateRequestsPreviewWhileSavedGuideStaysHidden() {
    QTemporaryDir directory;
    const QString firstPath = directory.filePath(QStringLiteral("front.png"));
    const QString secondPath = directory.filePath(QStringLiteral("front-01.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(firstPath));
    QVERIFY(image.save(secondPath));

    QJsonObject snapshot = configuredProfileSnapshot(firstPath, 1);
    QJsonArray templates = snapshot.value(QStringLiteral("templates")).toArray();
    templates.append(QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front-01.png")},
                                 {QStringLiteral("direction"), QStringLiteral("front")},
                                 {QStringLiteral("path"), secondPath}});
    snapshot.insert(QStringLiteral("templates"), templates);
    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    auto *combo = dialog.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    QVERIFY(canvas != nullptr);
    QVERIFY(combo != nullptr);
    QVERIFY(!canvas->coarseShape().isEmpty());
    QVERIFY(!canvas->guideVisible());
    QSignalSpy previewSpy(&dialog, &GeometryRulesPage::previewRequested);

    combo->setCurrentIndex(1);

    QCOMPARE(previewSpy.count(), 1);
    const QJsonObject request = previewSpy.takeFirst().at(0).toJsonObject();
    QCOMPARE(request.value(QStringLiteral("template_id")).toString(),
             QStringLiteral("front:front-01.png"));
    QVERIFY(!request.value(QStringLiteral("seed_shape")).toObject().isEmpty());
    QVERIFY(!canvas->coarseShape().isEmpty());
    QVERIFY(!canvas->guideVisible());
}

void TestGeometryRulesPage::lowConfidenceBackPreviewExplainsHowToRetry() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    QJsonObject snapshot = configuredV2ProfileSnapshot(frontPath, backPath);
    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    auto *direction = dialog.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    QVERIFY(direction != nullptr);
    direction->setCurrentIndex(direction->findData(QStringLiteral("back")));

    dialog.setRulePreview(QJsonObject{
        {QStringLiteral("status"), QStringLiteral("low_confidence")},
        {QStringLiteral("rule_id"), QStringLiteral("glare")},
        {QStringLiteral("direction"), QStringLiteral("back")},
        {QStringLiteral("template_id"), QStringLiteral("back:back.png")},
        {QStringLiteral("anchor_fit"), QJsonObject{
            {QStringLiteral("candidates"), QJsonArray()},
            {QStringLiteral("selected_candidate_index"), QJsonValue()}}},
        {QStringLiteral("rule_fit"), QJsonObject{
            {QStringLiteral("candidates"), QJsonArray()},
            {QStringLiteral("selected_candidate_index"), QJsonValue()}}}
    });

    bool foundRetryGuidance = false;
    for (QLabel *label : dialog.findChildren<QLabel *>()) {
        if (label->text().contains(QStringLiteral("重新粗画"))) {
            foundRetryGuidance = true;
            break;
        }
    }
    QVERIFY(foundRetryGuidance);
}

void TestGeometryRulesPage::noSelectedRuleClearsEveryRuleOverlay() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    GeometryRulesPage dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    const QJsonObject effective{{QStringLiteral("shape"), QStringLiteral("circle")},
                                {QStringLiteral("cx"), 60.0}, {QStringLiteral("cy"), 60.0},
                                {QStringLiteral("r"), 31.0}};
    dialog.setValidationJob(QJsonObject{
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                        {QStringLiteral("status"), QStringLiteral("active")},
                        {QStringLiteral("rules"), QJsonArray{QJsonObject{
                            {QStringLiteral("rule_id"), QStringLiteral("glare")},
                            {QStringLiteral("status"), QStringLiteral("active")},
                            {QStringLiteral("effective_shape"), effective}}}}}
        }}}}
    });
    auto *rules = dialog.findChild<QListWidget *>(QStringLiteral("ruleList"));
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(rules != nullptr);
    QVERIFY(canvas != nullptr);
    rules->clearSelection();
    rules->setCurrentRow(-1);
    QVERIFY(QMetaObject::invokeMethod(&dialog, "refreshTemplatePreview", Qt::DirectConnection));
    QVERIFY(canvas->coarseShape().isEmpty());
    QVERIFY(canvas->fitShape().isEmpty());
}

void TestGeometryRulesPage::savedCalibrationShowsFittedBoundaryOnlyForReferenceTemplate() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString secondPath = directory.filePath(QStringLiteral("front-01.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(secondPath));
    QVERIFY(image.save(backPath));

    QJsonObject snapshot = configuredV2ProfileSnapshot(frontPath, backPath);
    QJsonArray templates = snapshot.value(QStringLiteral("templates")).toArray();
    templates.append(QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front-01.png")},
                                 {QStringLiteral("direction"), QStringLiteral("front")},
                                 {QStringLiteral("path"), secondPath}});
    snapshot.insert(QStringLiteral("templates"), templates);
    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    auto *templatesCombo = dialog.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    QVERIFY(canvas != nullptr);
    QVERIFY(templatesCombo != nullptr);
    QVERIFY(!canvas->coarseShape().isEmpty());
    QVERIFY(!canvas->guideVisible());
    QVERIFY(qAbs(canvas->fitShape().value(QStringLiteral("r")).toDouble() - 37.8) < 0.001);

    templatesCombo->setCurrentIndex(templatesCombo->findText(QStringLiteral("front:front-01.png")));
    QVERIFY(!canvas->coarseShape().isEmpty());
    QVERIFY(!canvas->guideVisible());
    QVERIFY(canvas->fitShape().isEmpty());
}

void TestGeometryRulesPage::validationDiagnosticsComeFromSelectedNestedRule() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    GeometryRulesPage dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    dialog.setValidationJob(QJsonObject{
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                        {QStringLiteral("status"), QStringLiteral("active")},
                        {QStringLiteral("rules"), QJsonArray{QJsonObject{
                            {QStringLiteral("rule_id"), QStringLiteral("glare")},
                            {QStringLiteral("status"), QStringLiteral("active")},
                            {QStringLiteral("selected_candidate_index"), 2},
                            {QStringLiteral("edge_support"), 0.81},
                            {QStringLiteral("fit_residual"), 0.013},
                            {QStringLiteral("effective_shape"), QJsonObject{
                                {QStringLiteral("shape"), QStringLiteral("circle")},
                                {QStringLiteral("cx"), 60.0}, {QStringLiteral("cy"), 60.0},
                                {QStringLiteral("r"), 31.0}}}}}}}
        }}}}
    });
    QVERIFY(QMetaObject::invokeMethod(&dialog, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 0)));
    auto *diagnostics = dialog.findChild<QTextEdit *>();
    QVERIFY(diagnostics != nullptr);
    const QString text = diagnostics->toPlainText();
    QVERIFY(text.contains(QStringLiteral("候选: 2")));
    QVERIFY(text.contains(QStringLiteral("边缘支持: 0.810")));
    QVERIFY(text.contains(QStringLiteral("残差: 0.013")));
}

void TestGeometryRulesPage::fusionRegressionRemainsVisibleWhileInspectingTemplateFit() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    const QJsonObject change{
        {QStringLiteral("template_id"), QStringLiteral("front:front.png")},
        {QStringLiteral("expected"), QStringLiteral("front")},
        {QStringLiteral("baseline_predicted"), QStringLiteral("front")},
        {QStringLiteral("candidate_predicted"), QStringLiteral("back")},
        {QStringLiteral("candidate_global_prediction"), QStringLiteral("front")},
        {QStringLiteral("candidate_local_prediction"), QStringLiteral("back")},
        {QStringLiteral("candidate_decision_source"), QStringLiteral("local_override")},
        {QStringLiteral("cause"), QStringLiteral("geometry_local_override")}
    };
    GeometryRulesPage dialog;
    dialog.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    dialog.setValidationJob(QJsonObject{
        {QStringLiteral("state"), QStringLiteral("completed")},
        {QStringLiteral("base_library_revision"), 4},
        {QStringLiteral("base_draft_revision"), 2},
        {QStringLiteral("blocking_issues"), QJsonArray{QJsonObject{
            {QStringLiteral("code"), QStringLiteral("geometry_fusion_regression")},
            {QStringLiteral("correct_to_wrong"), 1},
            {QStringLiteral("templates"), QJsonArray{QStringLiteral("front:front.png")}},
            {QStringLiteral("changed_predictions"), QJsonArray{change}}
        }}},
        {QStringLiteral("report"), QJsonObject{{QStringLiteral("front"), QJsonArray{
            QJsonObject{{QStringLiteral("template_id"), QStringLiteral("front:front.png")},
                        {QStringLiteral("status"), QStringLiteral("active")},
                        {QStringLiteral("rules"), QJsonArray{QJsonObject{
                            {QStringLiteral("rule_id"), QStringLiteral("glare")},
                            {QStringLiteral("status"), QStringLiteral("active")},
                            {QStringLiteral("edge_support"), 0.91},
                            {QStringLiteral("effective_shape"), QJsonObject{
                                {QStringLiteral("shape"), QStringLiteral("circle")},
                                {QStringLiteral("cx"), 60.0}, {QStringLiteral("cy"), 60.0},
                                {QStringLiteral("r"), 31.0}}}}}}}
        }}}}
    });

    auto *table = dialog.findChild<QTableWidget *>(QStringLiteral("validationTable"));
    auto *diagnostics = dialog.findChild<QTextEdit *>();
    QVERIFY(table != nullptr);
    QVERIFY(diagnostics != nullptr);
    QVERIFY(table->item(0, 2)->toolTip().contains(QStringLiteral("全局")));
    QVERIFY(QMetaObject::invokeMethod(&dialog, "selectValidationTemplate", Qt::DirectConnection,
                                      Q_ARG(int, 0)));
    const QString text = diagnostics->toPlainText();
    QVERIFY(text.contains(QStringLiteral("发布阻断")));
    QVERIFY(text.contains(QStringLiteral("基线: 正面")));
    QVERIFY(text.contains(QStringLiteral("候选全局: 正面")));
    QVERIFY(text.contains(QStringLiteral("候选局部: 反面")));
    QVERIFY(text.contains(QStringLiteral("当前规则的验证拟合结果")));
}

void TestGeometryRulesPage::publishWorkflowNavigatesToMissingDirectionCalibration() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));
    QJsonObject snapshot = configuredV2ProfileSnapshot(frontPath, backPath);
    QJsonObject draft = snapshot.value(QStringLiteral("draft")).toObject();
    QJsonObject directions = draft.value(QStringLiteral("directions")).toObject();
    QJsonObject back = directions.value(QStringLiteral("back")).toObject();
    QJsonObject calibrations = back.value(QStringLiteral("calibrations")).toObject();
    calibrations.remove(QStringLiteral("glare"));
    back.insert(QStringLiteral("calibrations"), calibrations);
    directions.insert(QStringLiteral("back"), back);
    draft.insert(QStringLiteral("directions"), directions);
    snapshot.insert(QStringLiteral("draft"), draft);

    GeometryRulesPage dialog;
    dialog.setSnapshot(snapshot);
    dialog.findChild<QPushButton *>(QStringLiteral("publishWorkflowButton"))->click();
    auto *direction = dialog.findChild<QComboBox *>(QStringLiteral("directionCombo"));
    auto *status = dialog.findChild<QLabel *>(QStringLiteral("geometryStatusLabel"));
    QVERIFY(direction != nullptr);
    QVERIFY(status != nullptr);
    QCOMPARE(direction->currentData().toString(), QStringLiteral("back"));
    QVERIFY(status->text().contains(QStringLiteral("请先完成反面标定")));
}

void TestGeometryRulesPage::migrationConflictKeepOnlyEmitsExplicitResolution() {
    QTemporaryDir directory;
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::black);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));
    QJsonObject snapshot = configuredV2ProfileSnapshot(frontPath, backPath);
    QJsonObject draft = snapshot.value(QStringLiteral("draft")).toObject();
    draft.insert(QStringLiteral("migration"), QJsonObject{
        {QStringLiteral("source_schema_version"), 1},
        {QStringLiteral("resolutions"), QJsonArray()},
        {QStringLiteral("conflicts"), QJsonArray{QJsonObject{
            {QStringLiteral("conflict_id"), QStringLiteral("duplicate-back-glare")},
            {QStringLiteral("type"), QStringLiteral("same_direction_duplicate")},
            {QStringLiteral("sources"), QJsonArray{
                QJsonObject{{QStringLiteral("direction"), QStringLiteral("back")},
                            {QStringLiteral("source_rule_id"), QStringLiteral("back-glare-a")},
                            {QStringLiteral("logical_rule_id"), QStringLiteral("glare")},
                            {QStringLiteral("name"), QStringLiteral("中心反光 A")}},
                QJsonObject{{QStringLiteral("direction"), QStringLiteral("back")},
                            {QStringLiteral("source_rule_id"), QStringLiteral("back-glare-b")},
                            {QStringLiteral("logical_rule_id"), QStringLiteral("intrusion")},
                            {QStringLiteral("name"), QStringLiteral("中心反光 B")}}
            }}}
        }}
    });
    snapshot.insert(QStringLiteral("draft"), draft);

    GeometryRulesPage dialog;
    QSignalSpy spy(&dialog, &GeometryRulesPage::migrationResolutionRequested);
    dialog.setSnapshot(snapshot);
    auto *conflicts = dialog.findChild<QComboBox *>(QStringLiteral("migrationConflictCombo"));
    auto *actions = dialog.findChild<QComboBox *>(QStringLiteral("migrationActionCombo"));
    auto *survivors = dialog.findChild<QComboBox *>(QStringLiteral("migrationSurvivorCombo"));
    auto *resolve = dialog.findChild<QPushButton *>(QStringLiteral("resolveMigrationButton"));
    QVERIFY(conflicts != nullptr);
    QVERIFY(actions != nullptr);
    QVERIFY(survivors != nullptr);
    QVERIFY(resolve != nullptr);
    conflicts->setCurrentIndex(conflicts->findData(QStringLiteral("duplicate-back-glare")));
    actions->setCurrentIndex(actions->findData(QStringLiteral("keep_only")));
    survivors->setCurrentIndex(survivors->findData(QStringLiteral("back-glare-a")));
    resolve->click();

    QCOMPARE(spy.count(), 1);
    const QList<QVariant> arguments = spy.takeFirst();
    QCOMPARE(arguments.at(0).toString(), QStringLiteral("duplicate-back-glare"));
    const QJsonObject resolution = arguments.at(1).toJsonObject();
    QCOMPARE(resolution.value(QStringLiteral("action")).toString(), QStringLiteral("keep_only"));
    QCOMPARE(resolution.value(QStringLiteral("survivor_rule_id")).toString(), QStringLiteral("back-glare-a"));
}

void TestGeometryRulesPage::allSupportedShapesUseTheSavePath() {
    const QStringList shapes{QStringLiteral("circle"), QStringLiteral("ellipse"),
                             QStringLiteral("rotated_rectangle")};
    for (const QString &shapeName : shapes) {
        GeometryRulesPage page;
        page.setSnapshot(configuredProfileSnapshot(QString()));
        auto *shape = page.findChild<QComboBox *>(QStringLiteral("shapeCombo"));
        QVERIFY(shape != nullptr);
        const int index = shape->findData(shapeName);
        QVERIFY(index >= 0);
        shape->setCurrentIndex(index);
        QSignalSpy saveSpy(&page, &GeometryRulesPage::saveDraftRequested);

        page.requestSaveDraft();

        QCOMPARE(saveSpy.count(), 1);
        const QJsonObject savedDraft = saveSpy.takeFirst().at(0).toJsonObject();
        const QJsonObject savedRule = savedDraft.value(QStringLiteral("directions")).toObject()
            .value(QStringLiteral("front")).toObject().value(QStringLiteral("rules"))
            .toArray().first().toObject();
        QCOMPARE(savedRule.value(QStringLiteral("shape")).toString(), shapeName);
    }
}

namespace {
bool geometryMatchesBackendShapeContract(const QJsonObject &geometry,
                                         const QString &shape) {
    if (geometry.isEmpty()) return false;
    if (shape == QStringLiteral("circle")) {
        return geometry.value(QStringLiteral("r")).toDouble() > 0.0;
    }
    if (shape == QStringLiteral("ellipse")) {
        return geometry.value(QStringLiteral("rx")).toDouble() > 0.0
            && geometry.value(QStringLiteral("ry")).toDouble() > 0.0;
    }
    if (shape == QStringLiteral("rotated_rectangle")) {
        return geometry.value(QStringLiteral("half_width")).toDouble() > 0.0
            && geometry.value(QStringLiteral("half_height")).toDouble() > 0.0;
    }
    return false;
}

bool draftGeometryMatchesBackendContract(const QJsonObject &draft) {
    const QJsonArray rules = draft.value(QStringLiteral("rules")).toArray();
    const QJsonObject directions = draft.value(QStringLiteral("directions")).toObject();
    for (const QJsonValue &ruleValue : rules) {
        const QJsonObject rule = ruleValue.toObject();
        const QString ruleId = rule.value(QStringLiteral("rule_id")).toString();
        const QString shape = rule.value(QStringLiteral("shape")).toString();
        for (const QString &sideName : {QStringLiteral("front"), QStringLiteral("back")}) {
            const QJsonObject calibration = directions.value(sideName).toObject()
                .value(QStringLiteral("calibrations")).toObject().value(ruleId).toObject();
            if (calibration.isEmpty()) continue;
            const QJsonObject geometry = calibration.value(QStringLiteral("geometry")).toObject();
            const QJsonObject seed = calibration.value(QStringLiteral("seed_geometry")).toObject();
            if (calibration.value(QStringLiteral("state")).toString() == QStringLiteral("ready")
                && geometry.isEmpty()) {
                return false;
            }
            if (!geometry.isEmpty() && !geometryMatchesBackendShapeContract(geometry, shape)) {
                return false;
            }
            if (!seed.isEmpty() && !geometryMatchesBackendShapeContract(seed, shape)) {
                return false;
            }
        }
    }
    return true;
}
}

void TestGeometryRulesPage::shapeChangeDoesNotSaveStaleGeometry_data() {
    QTest::addColumn<QString>("shape");
    QTest::newRow("ellipse") << QStringLiteral("ellipse");
    QTest::newRow("rotated rectangle") << QStringLiteral("rotated_rectangle");
}

void TestGeometryRulesPage::shapeChangeDoesNotSaveStaleGeometry() {
    QFETCH(QString, shape);
    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(QString(), QString()));
    const QJsonObject originalDraft = page.draft();
    QVERIFY(draftGeometryMatchesBackendContract(page.draft()));
    auto *shapeCombo = page.findChild<QComboBox *>(QStringLiteral("shapeCombo"));
    QVERIFY(shapeCombo != nullptr);
    shapeCombo->setCurrentIndex(shapeCombo->findData(shape));
    QSignalSpy saveSpy(&page, &GeometryRulesPage::saveDraftRequested);

    page.requestSaveDraft();

    QCOMPARE(saveSpy.count(), 1);
    const QJsonObject savedDraft = saveSpy.takeFirst().at(0).toJsonObject();
    QVERIFY2(draftGeometryMatchesBackendContract(savedDraft),
             "saved geometry fields must match the selected shape as required by the backend");
    const QJsonObject directions = savedDraft.value(QStringLiteral("directions")).toObject();
    for (const QString &sideName : {QStringLiteral("front"), QStringLiteral("back")}) {
        const QJsonObject calibration = directions.value(sideName).toObject()
            .value(QStringLiteral("calibrations")).toObject()
            .value(QStringLiteral("glare")).toObject();
        QVERIFY2(calibration.isEmpty(),
                 "changing a logical shape must require both directions to be recalibrated");
    }

    auto *undoButton = page.findChild<QPushButton *>(QStringLiteral("undoButton"));
    QVERIFY(undoButton != nullptr);
    QVERIFY(undoButton->isEnabled());
    undoButton->click();
    QCOMPARE(page.draft(), originalDraft);

    auto *redoButton = page.findChild<QPushButton *>(QStringLiteral("redoButton"));
    QVERIFY(redoButton != nullptr);
    QVERIFY(redoButton->isEnabled());
    redoButton->click();
    QVERIFY(draftGeometryMatchesBackendContract(page.draft()));
    undoButton->click();
    QCOMPARE(page.draft(), originalDraft);

    shapeCombo->setCurrentIndex(shapeCombo->findData(shape));
    page.discardUnsavedChanges();
    QCOMPARE(page.draft(), originalDraft);
}

void TestGeometryRulesPage::numericShapeProducesBackendCompatiblePayload_data() {
    QTest::addColumn<QString>("shape");
    QTest::addColumn<QJsonObject>("numericShape");
    QTest::newRow("ellipse")
        << QStringLiteral("ellipse")
        << QJsonObject{{QStringLiteral("shape"), QStringLiteral("ellipse")},
                       {QStringLiteral("cx"), 60.0}, {QStringLiteral("cy"), 60.0},
                       {QStringLiteral("rx"), 30.0}, {QStringLiteral("ry"), 20.0},
                       {QStringLiteral("angle_deg"), 0.0}};
    QTest::newRow("rotated rectangle")
        << QStringLiteral("rotated_rectangle")
        << QJsonObject{{QStringLiteral("shape"), QStringLiteral("rotated_rectangle")},
                       {QStringLiteral("cx"), 60.0}, {QStringLiteral("cy"), 60.0},
                       {QStringLiteral("half_width"), 30.0},
                       {QStringLiteral("half_height"), 20.0},
                       {QStringLiteral("angle_deg"), 15.0}};
}

void TestGeometryRulesPage::numericShapeProducesBackendCompatiblePayload() {
    QFETCH(QString, shape);
    QFETCH(QJsonObject, numericShape);
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString frontPath = directory.filePath(QStringLiteral("front.png"));
    const QString backPath = directory.filePath(QStringLiteral("back.png"));
    QImage image(120, 120, QImage::Format_RGB32);
    image.fill(Qt::white);
    QVERIFY(image.save(frontPath));
    QVERIFY(image.save(backPath));

    GeometryRulesPage page;
    page.setSnapshot(configuredV2ProfileSnapshot(frontPath, backPath));
    auto *shapeCombo = page.findChild<QComboBox *>(QStringLiteral("shapeCombo"));
    auto *canvas = page.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(shapeCombo != nullptr);
    QVERIFY(canvas != nullptr);
    shapeCombo->setCurrentIndex(shapeCombo->findData(shape));
    canvas->setNumericShape(numericShape);
    QSignalSpy saveSpy(&page, &GeometryRulesPage::saveDraftRequested);

    page.requestSaveDraft();

    QCOMPARE(saveSpy.count(), 1);
    const QJsonObject savedDraft = saveSpy.takeFirst().at(0).toJsonObject();
    QVERIFY(draftGeometryMatchesBackendContract(savedDraft));
    const QJsonObject frontCalibration = savedDraft.value(QStringLiteral("directions")).toObject()
        .value(QStringLiteral("front")).toObject()
        .value(QStringLiteral("calibrations")).toObject()
        .value(QStringLiteral("glare")).toObject();
    QCOMPARE(frontCalibration.value(QStringLiteral("state")).toString(),
             QStringLiteral("needs_review"));
    QVERIFY(geometryMatchesBackendShapeContract(
        frontCalibration.value(QStringLiteral("seed_geometry")).toObject(), shape));
}

void TestGeometryRulesPage::emptyDraftHasNoDefaultDrawingTool() {
    GeometryRulesPage page;
    page.setSnapshot(profileSnapshot(1, 0, 0));
    auto *shape = page.findChild<QComboBox *>(QStringLiteral("shapeCombo"));
    auto *canvas = page.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(shape != nullptr);
    QVERIFY(canvas != nullptr);
    QVERIFY(shape->currentData().toString().isEmpty());
    canvas->setImage(QImage(120, 120, QImage::Format_RGB32));
    page.resize(800, 600);
    page.show();
    QCoreApplication::processEvents();
    QSignalSpy shapeSpy(canvas, &GeometryRuleCanvas::shapeChanged);

    QTest::mousePress(canvas, Qt::LeftButton, Qt::NoModifier, QPoint(20, 20));
    QTest::mouseMove(canvas, QPoint(80, 80));
    QTest::mouseRelease(canvas, Qt::LeftButton, Qt::NoModifier, QPoint(80, 80));

    QCOMPARE(shapeSpy.count(), 0);
}

void TestGeometryRulesPage::signedMarginKeepsItsSignForInsideAndOutside() {
    const QStringList modes{QStringLiteral("inside"), QStringLiteral("outside")};
    const QList<int> margins{-2, 2};
    for (const QString &modeName : modes) {
        for (int marginPercent : margins) {
            GeometryRulesPage page;
            page.setSnapshot(configuredProfileSnapshot(QString()));
            QComboBox *mode = nullptr;
            for (QComboBox *candidate : page.findChildren<QComboBox *>()) {
                if (candidate->findData(QStringLiteral("inside")) >= 0
                    && candidate->findData(QStringLiteral("outside")) >= 0) {
                    mode = candidate;
                }
            }
            auto *margin = page.findChild<QSpinBox *>(QStringLiteral("marginSpinBox"));
            QVERIFY(mode != nullptr);
            QVERIFY(margin != nullptr);
            QVERIFY(margin->toolTip().contains(QStringLiteral("负值向内收缩")));
            QVERIFY(margin->toolTip().contains(QStringLiteral("正值向外扩张")));
            mode->setCurrentIndex(mode->findData(modeName));
            margin->setValue(marginPercent);
            QSignalSpy saveSpy(&page, &GeometryRulesPage::saveDraftRequested);

            page.requestSaveDraft();

            QCOMPARE(saveSpy.count(), 1);
            const QJsonObject rule = saveSpy.takeFirst().at(0).toJsonObject()
                .value(QStringLiteral("directions")).toObject()
                .value(QStringLiteral("front")).toObject()
                .value(QStringLiteral("rules")).toArray().first().toObject();
            QCOMPARE(rule.value(QStringLiteral("mode")).toString(), modeName);
            QCOMPARE(rule.value(QStringLiteral("margin_ratio")).toDouble(),
                     marginPercent / 100.0);
            QCOMPARE(rule.value(QStringLiteral("margin_semantics")).toString(),
                     QStringLiteral("signed_boundary_v2"));
        }
    }
}

QTEST_MAIN(TestGeometryRulesPage)
#include "test_geometryrulespage.moc"
