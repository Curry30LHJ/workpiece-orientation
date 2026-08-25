#include <QtTest/QtTest>

#include <QDialogButtonBox>
#include <QComboBox>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QListWidget>
#include <QSignalSpy>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTextEdit>

#include "../annotationcanvas.h"
#include "../annotationeditor.h"
#include "../annotationmanager.h"

class TestAnnotationManager : public QObject {
    Q_OBJECT

private:
    static QJsonObject region(double x, double y, double width, double height) {
        return {{QStringLiteral("x"), x}, {QStringLiteral("y"), y},
                {QStringLiteral("width"), width}, {QStringLiteral("height"), height}};
    }

    static QJsonObject snapshot() {
        const QJsonArray templates{
            QJsonObject{{"template_id", "front:00.png"}, {"orientation", "front"}, {"index", 0},
                        {"preview_path", ""}, {"width", 200}, {"height", 100}, {"readable", true}},
            QJsonObject{{"template_id", "front:01.png"}, {"orientation", "front"}, {"index", 1},
                        {"preview_path", ""}, {"width", 200}, {"height", 100}, {"readable", true}},
        };
        const QJsonArray targets{
            QJsonObject{{"template_id", "front:00.png"}, {"orientation", "front"}, {"index", 0},
                        {"state", "active"}, {"provenance", "manual"},
                        {"regions", QJsonArray{region(1, 2, 20, 10)}},
                        {"diagnostics", QJsonObject{{"reason_code", "manual"}}}},
            QJsonObject{{"template_id", "front:01.png"}, {"orientation", "front"}, {"index", 1},
                        {"state", "needs_review"}, {"provenance", "automatic"},
                        {"regions", QJsonArray{region(30, 20, 15, 8)}},
                        {"diagnostics", QJsonObject{{"reason_code", "source_disagreement"},
                                                    {"source_template_ids", QJsonArray{"front:00.png"}},
                                                    {"confidence", 0.75}, {"max_spread_px", 4.2},
                                                    {"area_ratio", 0.08}, {"keypoints_before", 10},
                                                    {"keypoints_after", 7}, {"remaining_ratio", 0.7}}}},
        };
        const QJsonObject group{
            {"group_id", "glare"}, {"name", "边缘反光"}, {"enabled", true},
            {"draft_state", "needs_review"}, {"active_state", "pending"},
            {"summary", QJsonObject{{"manual_count", 1}, {"automatic_count", 1},
                                     {"needs_review_count", 1}, {"unresolved_count", 0}}},
            {"targets", targets},
        };
        return {{"workpiece_id", "m7"}, {"revision", 7}, {"annotation_revision", 5},
                {"active_annotation_revision", 4}, {"templates", templates},
                {"groups", QJsonArray{group}}};
    }

private slots:
    void canvasUsesNativeAspectFitCoordinates() {
        AnnotationCanvas canvas;
        canvas.resize(400, 400);
        canvas.show();
        QTest::qWait(10);
        canvas.setImage(QImage(200, 100, QImage::Format_RGB32));
        QSignalSpy changed(&canvas, &AnnotationCanvas::regionsChanged);
        QTest::mousePress(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(100, 100));
        QTest::mouseMove(&canvas, QPoint(300, 300));
        QTest::mouseRelease(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(300, 300));
        QCOMPARE(changed.count(), 1);
        const QList<AnnotationRegionView> regions = canvas.regions();
        QCOMPARE(regions.size(), 1);
        QCOMPARE(regions.first().rect, QRectF(50, 0, 100, 100));
    }

    void canvasStoresMultipleRegionsAndDeletesSelection() {
        AnnotationCanvas canvas;
        canvas.resize(400, 400);
        canvas.show();
        QTest::qWait(10);
        canvas.setImage(QImage(200, 100, QImage::Format_RGB32));
        canvas.setRegions({
            {QRectF(10, 10, 20, 20), QStringLiteral("manual"), QStringLiteral("active"), false},
            {QRectF(50, 20, 30, 10), QStringLiteral("automatic"), QStringLiteral("needs_review"), false},
        });
        QCOMPARE(canvas.regions().size(), 2);
        QTest::mouseClick(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(40, 140));
        QVERIFY(canvas.selectedRegionIndex() >= 0);
        canvas.deleteSelectedRegion();
        QCOMPARE(canvas.regions().size(), 1);
        QCOMPARE(canvas.regions().first().provenance, QStringLiteral("automatic"));
    }

    void editorSwitchKeepsSavedTemplateAndStaysOpen() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString firstPath = temporary.filePath(QStringLiteral("front-0.png"));
        const QString secondPath = temporary.filePath(QStringLiteral("front-1.png"));
        QVERIFY(QImage(80, 60, QImage::Format_RGB32).save(firstPath));
        QVERIFY(QImage(80, 60, QImage::Format_RGB32).save(secondPath));

        AnnotationEditorDialog editor(QStringList{firstPath, secondPath}, {});
        editor.setUnsavedPromptHandler([]() { return QMessageBox::Save; });
        editor.show();
        QTest::qWait(10);
        auto *canvas = editor.findChild<AnnotationCanvas *>();
        auto *templates = editor.findChild<QComboBox *>();
        QVERIFY(canvas != nullptr);
        QVERIFY(templates != nullptr);
        canvas->setRegions({{QRectF(5, 6, 20, 15), QStringLiteral("manual"), QStringLiteral("active"), false}});
        QVERIFY(QMetaObject::invokeMethod(&editor, "markDirty", Qt::DirectConnection));

        templates->setCurrentIndex(1);

        QVERIFY(editor.isVisible());
        QCOMPARE(editor.result(), 0);
        QCOMPARE(templates->currentIndex(), 1);
        const QJsonArray annotations = editor.annotationPatch().value(QStringLiteral("annotations")).toArray();
        QCOMPARE(annotations.size(), 1);
        QCOMPARE(annotations.first().toObject().value(QStringLiteral("template_id")).toString(),
                 QStringLiteral("front:0"));
    }

    void managerRendersSnapshotAndDiagnostics() {
        AnnotationManagerDialog dialog;
        dialog.setSnapshot(snapshot());
        QCOMPARE(dialog.workpieceId(), QStringLiteral("m7"));
        QCOMPARE(dialog.baseRevision(), 7);
        QVERIFY(dialog.findChild<QListWidget *>(QStringLiteral("annotationGroupList"))->count() == 1);
        QVERIFY(dialog.findChild<QTableWidget *>(QStringLiteral("annotationTemplateTable"))->rowCount() == 2);
        QVERIFY(dialog.findChild<QLabel *>(QStringLiteral("annotationVersionLabel"))->text().contains(QStringLiteral("草稿修订 5")));
        dialog.findChild<QTableWidget *>(QStringLiteral("annotationTemplateTable"))->selectRow(1);
        QVERIFY(dialog.findChild<QTextEdit *>(QStringLiteral("annotationDiagnosticsText"))->toPlainText()
                    .contains(QStringLiteral("source_disagreement")));
        QVERIFY(dialog.findChild<AnnotationCanvas *>(QStringLiteral("annotationPreviewCanvas")) != nullptr);
    }

    void managerShowsPropagationProgressWhileBusy() {
        AnnotationManagerDialog dialog;
        dialog.setSnapshot(snapshot());
        dialog.setBusy(true);
        dialog.setProgress(QStringLiteral("propagating_annotations"), 17, 70);

        const auto *progress = dialog.findChild<QLabel *>(QStringLiteral("annotationProgressLabel"));
        QVERIFY(progress != nullptr);
        QCOMPARE(progress->text(), QStringLiteral("正在递推干扰区域：17 / 70，完成后将刷新模板列表"));
        dialog.setBusy(false);
        QVERIFY(progress->text().isEmpty());
    }

    void managerShowsLocalizedProjectionFailureDiagnostics() {
        QJsonObject value = snapshot();
        QJsonObject group = value.value(QStringLiteral("groups")).toArray().first().toObject();
        QJsonArray targets = group.value(QStringLiteral("targets")).toArray();
        QJsonObject failed = targets.at(1).toObject();
        failed.insert(QStringLiteral("state"), QStringLiteral("unresolved"));
        failed.insert(QStringLiteral("provenance"), QStringLiteral("none"));
        failed.insert(QStringLiteral("regions"), QJsonArray{});
        failed.insert(QStringLiteral("diagnostics"), QJsonObject{
            {QStringLiteral("reason_code"), QStringLiteral("projection_failed")},
            {QStringLiteral("projection_failure_count"), 2},
        });
        targets.replace(1, failed);
        group.insert(QStringLiteral("targets"), targets);
        value.insert(QStringLiteral("groups"), QJsonArray{group});

        AnnotationManagerDialog dialog;
        dialog.setSnapshot(value);
        dialog.findChild<QTableWidget *>(QStringLiteral("annotationTemplateTable"))->selectRow(1);

        QVERIFY(dialog.findChild<QTextEdit *>(QStringLiteral("annotationDiagnosticsText"))
                    ->toPlainText().contains(QStringLiteral("局部特征投影失败")));
    }

    void managerKeepsOperationErrorVisibleAfterBusyClears() {
        AnnotationManagerDialog dialog;
        dialog.setBusy(true);
        dialog.setOperationError(QStringLiteral("局部特征模型不可用，请检查后端日志后重试"));
        dialog.setBusy(false);

        const auto *status = dialog.findChild<QLabel *>(QStringLiteral("annotationOperationStatusLabel"));
        QVERIFY(status != nullptr);
        QVERIFY(status->text().contains(QStringLiteral("局部特征模型不可用")));
        dialog.setSnapshot(snapshot());
        QVERIFY(status->text().isEmpty());
    }

    void managerEmitsDeleteAndReviewWithRevision() {
        AnnotationManagerDialog dialog;
        dialog.setSnapshot(snapshot());
        dialog.setDeleteConfirmationHandler([](const QString &, int) { return true; });
        QSignalSpy deleted(&dialog, &AnnotationManagerDialog::deleteRequested);
        QSignalSpy review(&dialog, &AnnotationManagerDialog::reviewRequested);
        QVERIFY(QMetaObject::invokeMethod(&dialog, "deleteSelectedGroup", Qt::DirectConnection));
        QCOMPARE(deleted.count(), 1);
        QCOMPARE(deleted.at(0).at(0).toString(), QStringLiteral("glare"));
        QCOMPARE(deleted.at(0).at(1).toInt(), 7);
        dialog.findChild<QTableWidget *>(QStringLiteral("annotationTemplateTable"))->selectRow(1);
        QVERIFY(QMetaObject::invokeMethod(&dialog, "acceptSelectedAnnotation", Qt::DirectConnection));
        QCOMPARE(review.count(), 1);
        QCOMPARE(review.at(0).at(0).toString(), QStringLiteral("glare"));
        QCOMPARE(review.at(0).at(1).toString(), QStringLiteral("front:01.png"));
        QCOMPARE(review.at(0).at(2).toString(), QStringLiteral("accept"));
        QCOMPARE(review.at(0).at(4).toInt(), 7);
    }
};

QTEST_MAIN(TestAnnotationManager)
#include "test_annotationmanager.moc"
