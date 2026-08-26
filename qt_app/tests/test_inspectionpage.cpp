#include <QtTest/QtTest>

#include <QApplication>
#include <QDateTime>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QGraphicsScene>
#include <QGroupBox>
#include <QImage>
#include <QLabel>
#include <QListWidget>
#include <QMimeData>
#include <QPushButton>
#include <QSignalSpy>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTextEdit>
#include <QUrl>
#include <QVBoxLayout>
#include <QWheelEvent>

#include "../inspectionimageview.h"
#include "../inspectionpage.h"
#include "../inspectiontypes.h"
#include "../apptheme.h"

static QString writeImage(QTemporaryDir &directory, const QString &name) {
    const QString path = directory.filePath(name);
    QImage image(48, 48, QImage::Format_RGB32);
    image.fill(Qt::darkGray);
    return image.save(path) ? path : QString();
}

static InspectionRecord resultRecord(const QString &id,
                                     const QString &path) {
    InspectionRecord record;
    record.id = id;
    record.imagePath = path;
    record.workpieceId = QStringLiteral("m1");
    record.response = QJsonObject{
        {QStringLiteral("label"), QStringLiteral("front")},
        {QStringLiteral("global_margin"), 0.03},
        {QStringLiteral("local_margin"), 7.3},
        {QStringLiteral("needs_review"), false}
    };
    record.label = QStringLiteral("front");
    record.elapsedMs = 12.0;
    record.completedAt = QDateTime::currentDateTime();
    return record;
}

static QJsonObject predictionResponse(const QString &label,
                                      bool needsReview = false) {
    return QJsonObject{
        {QStringLiteral("label"), label},
        {QStringLiteral("needs_review"), needsReview},
        {QStringLiteral("global_margin"), 0.04},
        {QStringLiteral("local_margin"), 6.0},
        {QStringLiteral("elapsed_ms"), 11.0}
    };
}

class TestInspectionPage : public QObject {
    Q_OBJECT

private slots:
    void initTestCase() { AppTheme::apply(qApp); }
    void batchIdsAreNonEmptyAndUnique() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("batch-id-0.png")),
            writeImage(directory, QStringLiteral("batch-id-1.png")),
            writeImage(directory, QStringLiteral("batch-id-2.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;

        page.beginBatch(paths, QStringLiteral("m1"));

        const QStringList ids = page.batchRecordIds();
        QCOMPARE(ids.size(), 3);
        QVERIFY(!ids.contains(QString()));
        QCOMPARE(QSet<QString>(ids.begin(), ids.end()).size(), 3);
    }

    void filteredSelectionStillTargetsTheSameRecord() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("filter-0.png")),
            writeImage(directory, QStringLiteral("filter-1.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;
        page.beginBatch(paths, QStringLiteral("m1"));
        const QStringList ids = page.batchRecordIds();

        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("back"), true));
        page.setBatchFilter(BatchFilter::NeedsReview);
        auto *table = page.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        QVERIFY(table != nullptr);
        QCOMPARE(table->rowCount(), 1);
        table->setCurrentCell(0, 0);

        QCOMPARE(page.selectedRecordId(), ids.at(1));
        QCOMPARE(table->item(0, 0)->data(Qt::UserRole).toString(), ids.at(1));
    }

    void filteredSubmissionFailureRestoresOriginalSelection() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("filtered-retry-0.png")),
            writeImage(directory, QStringLiteral("filtered-retry-1.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;
        page.setCurrentWorkpiece(QStringLiteral("m1"), QStringLiteral("M1"));
        page.setBackendAvailable(true, false, QString());
        page.beginBatch(paths, QStringLiteral("m1"));
        const QStringList ids = page.batchRecordIds();
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("back")));
        page.setBatchFilter(BatchFilter::Unprocessed);
        auto *table = page.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        auto *front = page.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
        QVERIFY(table != nullptr);
        QVERIFY(front != nullptr);
        table->setCurrentCell(1, 0);
        QCOMPARE(page.selectedRecordId(), ids.at(1));
        QSignalSpy confirmationSpy(&page, &InspectionPage::confirmationRequested);

        front->click();
        QCOMPARE(confirmationSpy.count(), 1);
        page.handleBackendFailure(QStringLiteral("submit_confirmation"),
                                  QStringLiteral("MODEL_ERROR"),
                                  QStringLiteral("submission failed"));

        QCOMPARE(page.selectedRecordId(), ids.at(1));
        QCOMPARE(table->item(table->currentRow(), 0)->data(Qt::UserRole).toString(), ids.at(1));
        QVERIFY(front->isEnabled());
    }

    void failedFilterIncludesPredictionAndConfirmationFailures() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("failed-filter-0.png")),
            writeImage(directory, QStringLiteral("failed-filter-1.png")),
            writeImage(directory, QStringLiteral("failed-filter-2.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;
        page.setBackendAvailable(true, false, QString());
        page.beginBatch(paths, QStringLiteral("m1"));
        const QStringList ids = page.batchRecordIds();

        page.handleBackendFailure(QStringLiteral("predict"), QStringLiteral("MODEL_ERROR"),
                                  QStringLiteral("prediction failed"));
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("back")));
        page.setRecordDisposition(ids.at(1), BatchDisposition::SubmitFailed,
                                  QString(), QStringLiteral("submit failed"));
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));
        page.setBatchFilter(BatchFilter::Failed);

        auto *table = page.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        QVERIFY(table != nullptr);
        QCOMPARE(table->rowCount(), 2);
        QSet<QString> visibleIds;
        for (int row = 0; row < table->rowCount(); ++row) {
            visibleIds.insert(table->item(row, 0)->data(Qt::UserRole).toString());
        }
        QCOMPARE(visibleIds, QSet<QString>({ids.at(0), ids.at(1)}));
        QCOMPARE(page.failedBatchCount(), 2);
    }

    void batchSubmissionFailurePreservesPredictionErrorChannel() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("submission-error.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;
        page.beginBatch({path}, QStringLiteral("m1"));
        const QString recordId = page.batchRecordIds().constFirst();
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));

        page.setRecordDisposition(recordId, BatchDisposition::SubmitFailed,
                                  QStringLiteral("job-failed"),
                                  QStringLiteral("submission failed"));

        const InspectionRecord record = page.recordForId(recordId);
        QCOMPARE(record.label, QStringLiteral("front"));
        QVERIFY(record.error.isEmpty());
        QCOMPARE(record.submissionError, QStringLiteral("submission failed"));
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("resultLabel"))->text()
                    .contains(QStringLiteral("正面")));
        QVERIFY(!page.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))
                     ->toPlainText().isEmpty());
    }

    void evolutionJobUpdateTargetsEveryMatchingRecord() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("job-0.png")),
            writeImage(directory, QStringLiteral("job-1.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;
        page.beginBatch(paths, QStringLiteral("m1"));
        const QStringList ids = page.batchRecordIds();
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("back")));
        page.setRecordDisposition(ids.at(0), BatchDisposition::QueuedFront,
                                  QStringLiteral("coalesced-job"));
        page.setRecordDisposition(ids.at(1), BatchDisposition::QueuedBack,
                                  QStringLiteral("coalesced-job"));

        page.handleBackendResponse(QStringLiteral("list_evolution_jobs"), QJsonObject{
            {QStringLiteral("jobs"), QJsonArray{QJsonObject{
                {QStringLiteral("job_id"), QStringLiteral("coalesced-job")},
                {QStringLiteral("state"), QStringLiteral("failed")},
                {QStringLiteral("error"), QStringLiteral("evolution failed")}}}}});
        page.setBatchFilter(BatchFilter::Failed);

        auto *table = page.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        QVERIFY(table != nullptr);
        QCOMPARE(table->rowCount(), 2);
        QCOMPARE(page.failedBatchCount(), 2);
    }

    void evolutionJobRunningAndCompletedUpdateEveryMatchingRow() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("job-state-0.png")),
            writeImage(directory, QStringLiteral("job-state-1.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;
        page.beginBatch(paths, QStringLiteral("m1"));
        const QStringList ids = page.batchRecordIds();
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("back")));
        page.setRecordDisposition(ids.at(0), BatchDisposition::QueuedFront,
                                  QStringLiteral("shared-job"));
        page.setRecordDisposition(ids.at(1), BatchDisposition::QueuedBack,
                                  QStringLiteral("shared-job"));
        auto *table = page.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        QVERIFY(table != nullptr);

        page.handleBackendResponse(QStringLiteral("list_evolution_jobs"), QJsonObject{
            {QStringLiteral("jobs"), QJsonArray{QJsonObject{
                {QStringLiteral("job_id"), QStringLiteral("shared-job")},
                {QStringLiteral("state"), QStringLiteral("running")}}}}});
        QCOMPARE(table->item(0, 4)->text(), QStringLiteral("正在更新缓存"));
        QCOMPARE(table->item(1, 4)->text(), QStringLiteral("正在更新缓存"));

        page.handleBackendResponse(QStringLiteral("list_evolution_jobs"), QJsonObject{
            {QStringLiteral("jobs"), QJsonArray{QJsonObject{
                {QStringLiteral("job_id"), QStringLiteral("shared-job")},
                {QStringLiteral("state"), QStringLiteral("completed")}}}}});
        QCOMPARE(table->item(0, 4)->text(), QStringLiteral("已参与预测"));
        QCOMPARE(table->item(1, 4)->text(), QStringLiteral("已参与预测"));
    }

    void stopAfterCurrentLeavesUnsentRecordsPending() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("stop-0.png")),
            writeImage(directory, QStringLiteral("stop-1.png")),
            writeImage(directory, QStringLiteral("stop-2.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;
        QSignalSpy commandSpy(&page, &InspectionPage::commandRequested);

        page.beginBatch(paths, QStringLiteral("m1"));
        const QStringList ids = page.batchRecordIds();
        QCOMPARE(commandSpy.count(), 1);
        page.requestBatchStop();
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));

        QCOMPARE(commandSpy.count(), 1);
        QCOMPARE(page.completedBatchCount(), 1);
        QCOMPARE(page.failedBatchCount(), 0);
        page.setBatchFilter(BatchFilter::Unprocessed);
        auto *table = page.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        QVERIFY(table != nullptr);
        QSet<QString> visibleIds;
        for (int row = 0; row < table->rowCount(); ++row) {
            visibleIds.insert(table->item(row, 0)->data(Qt::UserRole).toString());
        }
        QVERIFY(visibleIds.contains(ids.at(1)));
        QVERIFY(visibleIds.contains(ids.at(2)));
    }

    void inFlightBatchIgnoresDroppedSingleImage() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("drop-batch-0.png")),
            writeImage(directory, QStringLiteral("drop-batch-1.png"))};
        const QString droppedPath = writeImage(directory, QStringLiteral("drop-single.png"));
        QVERIFY(!paths.contains(QString()));
        QVERIFY(!droppedPath.isEmpty());
        InspectionPage page;
        QSignalSpy commandSpy(&page, &InspectionPage::commandRequested);
        page.beginBatch(paths, QStringLiteral("m1"));
        auto *view = page.findChild<InspectionImageView *>(
            QStringLiteral("inspectionImageView"));
        QVERIFY(view != nullptr);
        QMimeData mimeData;
        mimeData.setUrls({QUrl::fromLocalFile(droppedPath)});
        QDragEnterEvent dragEvent(QPoint(12, 12), Qt::CopyAction, &mimeData,
                                  Qt::LeftButton, Qt::NoModifier);
        QDropEvent dropEvent(QPointF(12.0, 12.0), Qt::CopyAction, &mimeData,
                             Qt::LeftButton, Qt::NoModifier);

        QApplication::sendEvent(view->viewport(), &dragEvent);
        QApplication::sendEvent(view->viewport(), &dropEvent);
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));

        QCOMPARE(page.mode(), InspectionMode::Batch);
        QCOMPARE(page.completedBatchCount(), 1);
        QCOMPARE(commandSpy.count(), 2);
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("back")));
        QCOMPARE(page.completedBatchCount(), 2);
    }

    void inFlightBatchRejectsModeSwitch() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("mode-batch-0.png")),
            writeImage(directory, QStringLiteral("mode-batch-1.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;
        QSignalSpy commandSpy(&page, &InspectionPage::commandRequested);
        page.beginBatch(paths, QStringLiteral("m1"));

        page.setMode(InspectionMode::Single);
        page.handleBackendResponse(QStringLiteral("predict"),
                                   predictionResponse(QStringLiteral("front")));

        QCOMPARE(page.mode(), InspectionMode::Batch);
        QCOMPARE(page.completedBatchCount(), 1);
        QCOMPARE(commandSpy.count(), 2);
    }

    void transportFailureMarksCurrentUnknownAndNeverResumesOldBatch() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList paths{
            writeImage(directory, QStringLiteral("transport-0.png")),
            writeImage(directory, QStringLiteral("transport-1.png")),
            writeImage(directory, QStringLiteral("transport-2.png"))};
        QVERIFY(!paths.contains(QString()));
        InspectionPage page;
        QSignalSpy commandSpy(&page, &InspectionPage::commandRequested);
        page.beginBatch(paths, QStringLiteral("m1"));

        page.handleBackendFailure(QStringLiteral("predict"),
                                  QStringLiteral("CONNECTION_LOST"),
                                  QStringLiteral("connection lost"));
        page.setBackendAvailable(true, false, QString());

        QCOMPARE(commandSpy.count(), 1);
        QCOMPARE(page.completedBatchCount(), 1);
        QCOMPARE(page.failedBatchCount(), 1);
        page.setBatchFilter(BatchFilter::Failed);
        auto *table = page.findChild<QTableWidget *>(
            QStringLiteral("batchResultsTableWidget"));
        QVERIFY(table != nullptr);
        QCOMPARE(table->rowCount(), 1);
        QCOMPARE(table->item(0, 4)->text(), QStringLiteral("结果未知"));
    }

    void resultShowsDecisionStateWithoutPercentConfidence() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("front.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;

        page.showSingleResult(resultRecord(QStringLiteral("front-1"), path));

        QCOMPARE(page.uiState(), InspectionUiState::Completed);
        QCOMPARE(page.singleImagePath(), path);
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("resultLabel"))->text()
                    .contains(QStringLiteral("正面")));
        const QString evidence = page.findChild<QTextEdit *>(
            QStringLiteral("evidenceTextEdit"))->toPlainText();
        QVERIFY(evidence.contains(QStringLiteral("全局特征")));
        QVERIFY(evidence.contains(QStringLiteral("局部匹配")));
        QVERIFY(evidence.contains(QStringLiteral("几何规则")));
        QVERIFY(!evidence.contains(QStringLiteral("0.03")));
        QVERIFY(!evidence.contains(QStringLiteral("7.3")));
        QVERIFY(!evidence.contains(QLatin1Char('%')));
        const QString rawEvidence = page.findChild<QTextEdit *>(
            QStringLiteral("rawEvidenceTextEdit"))->toPlainText();
        QVERIFY(rawEvidence.contains(QStringLiteral("0.03")));
        QVERIFY(rawEvidence.contains(QStringLiteral("7.3")));
    }

    void backendFailurePreservesVisibleImageAndEvidence() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("preserved.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;
        page.showSingleResult(resultRecord(QStringLiteral("preserved-1"), path));
        const QString evidence = page.findChild<QTextEdit *>(
            QStringLiteral("evidenceTextEdit"))->toPlainText();

        page.handleBackendFailure(QStringLiteral("predict"),
                                  QStringLiteral("CONNECTION_LOST"),
                                  QStringLiteral("连接中断"));

        QCOMPARE(page.singleImagePath(), path);
        QCOMPARE(page.findChild<QTextEdit *>(QStringLiteral("evidenceTextEdit"))->toPlainText(),
                 evidence);
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("resultMessageLabel"))->text()
                    .contains(QStringLiteral("连接中断")));
    }

    void rawEvidenceStartsCollapsed() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("raw.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;
        page.showSingleResult(resultRecord(QStringLiteral("raw-1"), path));

        auto *container = page.findChild<QGroupBox *>(QStringLiteral("rawEvidenceContainer"));
        QVERIFY(container != nullptr);
        QVERIFY(container->isCheckable());
        QVERIFY(!container->isChecked());
    }

    void selectingRecentRecordDoesNotResubmitPrediction() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString firstPath = writeImage(directory, QStringLiteral("first.png"));
        const QString secondPath = writeImage(directory, QStringLiteral("second.png"));
        QVERIFY(!firstPath.isEmpty());
        QVERIFY(!secondPath.isEmpty());
        InspectionPage page;
        InspectionRecord first = resultRecord(QStringLiteral("one"), firstPath);
        first.response.insert(QStringLiteral("global_margin"), 0.11);
        InspectionRecord second = resultRecord(QStringLiteral("two"), secondPath);
        second.response.insert(QStringLiteral("global_margin"), 0.22);
        page.showSingleResult(first);
        page.showSingleResult(second);
        QSignalSpy commandSpy(&page, &InspectionPage::commandRequested);

        page.selectRecentRecord(QStringLiteral("one"));

        QCOMPARE(page.singleImagePath(), firstPath);
        QVERIFY(page.findChild<QTextEdit *>(QStringLiteral("rawEvidenceTextEdit"))->toPlainText()
                    .contains(QStringLiteral("0.11")));
        QCOMPARE(commandSpy.count(), 0);
        auto *list = page.findChild<QListWidget *>(QStringLiteral("recentInspectionList"));
        QVERIFY(list != nullptr);
        bool foundStableId = false;
        for (int row = 0; row < list->count(); ++row) {
            foundStableId = foundStableId
                || list->item(row)->data(Qt::UserRole).toString() == QStringLiteral("one");
        }
        QVERIFY(foundStableId);
    }

    void droppedImageSelectsButDoesNotPredict() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("dropped.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;
        auto *view = page.findChild<InspectionImageView *>(
            QStringLiteral("inspectionImageView"));
        QVERIFY(view != nullptr);
        QSignalSpy commandSpy(&page, &InspectionPage::commandRequested);
        QMimeData mimeData;
        mimeData.setUrls({QUrl::fromLocalFile(path)});
        QDragEnterEvent dragEvent(QPoint(12, 12), Qt::CopyAction, &mimeData,
                                  Qt::LeftButton, Qt::NoModifier);
        QDropEvent event(QPointF(12.0, 12.0), Qt::CopyAction, &mimeData,
                         Qt::LeftButton, Qt::NoModifier);

        QApplication::sendEvent(view->viewport(), &dragEvent);
        QApplication::sendEvent(view->viewport(), &event);

        QCOMPARE(page.singleImagePath(), path);
        QCOMPARE(commandSpy.count(), 0);
    }

    void zoomAndResetDoNotChangeImagePath() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("zoom.png"));
        QVERIFY(!path.isEmpty());
        InspectionImageView view;
        QVERIFY(view.setImagePath(path));
        const qreal initialZoom = view.zoomFactor();
        QWheelEvent event(QPointF(12.0, 12.0), QPointF(12.0, 12.0),
                          QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier,
                          Qt::NoScrollPhase, false);

        QApplication::sendEvent(view.viewport(), &event);

        QVERIFY(view.zoomFactor() > initialZoom);
        QCOMPARE(view.imagePath(), path);
        view.resetView();
        QCOMPARE(view.imagePath(), path);
        QCOMPARE(view.zoomFactor(), 1.0);
    }

    void predictedOrientationOrdersConfirmCorrectRejectActions() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString frontPath = writeImage(directory, QStringLiteral("order-front.png"));
        const QString backPath = writeImage(directory, QStringLiteral("order-back.png"));
        QVERIFY(!frontPath.isEmpty());
        QVERIFY(!backPath.isEmpty());
        InspectionPage page;
        auto *front = page.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
        auto *back = page.findChild<QPushButton *>(QStringLiteral("confirmBackButton"));
        auto *reject = page.findChild<QPushButton *>(QStringLiteral("rejectConfirmationButton"));
        auto *layout = page.findChild<QVBoxLayout *>(QStringLiteral("confirmationActionsLayout"));
        QVERIFY(front != nullptr);
        QVERIFY(back != nullptr);
        QVERIFY(reject != nullptr);
        QVERIFY(layout != nullptr);

        page.showSingleResult(resultRecord(QStringLiteral("front-order"), frontPath));
        QCOMPARE(layout->indexOf(front), 0);
        QCOMPARE(layout->indexOf(back), 1);
        QCOMPARE(layout->indexOf(reject), 2);
        QCOMPARE(front->text(), QStringLiteral("确认正面"));
        QCOMPARE(back->text(), QStringLiteral("修正为反面"));
        QCOMPARE(reject->text(), QStringLiteral("不入库"));
        QCOMPARE(reject->property("role").toString(), QStringLiteral("secondary"));

        InspectionRecord backRecord = resultRecord(QStringLiteral("back-order"), backPath);
        backRecord.label = QStringLiteral("back");
        backRecord.response.insert(QStringLiteral("label"), QStringLiteral("back"));
        page.showSingleResult(backRecord);
        QCOMPARE(layout->indexOf(back), 0);
        QCOMPARE(layout->indexOf(front), 1);
        QCOMPARE(layout->indexOf(reject), 2);
        QCOMPARE(back->text(), QStringLiteral("确认反面"));
        QCOMPARE(front->text(), QStringLiteral("修正为正面"));
    }

    void predictedOrientationKeepsKeyboardOrderAlignedWithVisualActions() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString frontPath = writeImage(directory, QStringLiteral("focus-front.png"));
        const QString backPath = writeImage(directory, QStringLiteral("focus-back.png"));
        QVERIFY(!frontPath.isEmpty());
        QVERIFY(!backPath.isEmpty());
        InspectionPage page;
        auto *predict = page.findChild<QPushButton *>(QStringLiteral("predictButton"));
        auto *front = page.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
        auto *back = page.findChild<QPushButton *>(QStringLiteral("confirmBackButton"));
        auto *reject = page.findChild<QPushButton *>(QStringLiteral("rejectConfirmationButton"));
        QVERIFY(predict != nullptr);
        QVERIFY(front != nullptr);
        QVERIFY(back != nullptr);
        QVERIFY(reject != nullptr);

        InspectionRecord backRecord = resultRecord(QStringLiteral("focus-back"), backPath);
        backRecord.label = QStringLiteral("back");
        backRecord.response.insert(QStringLiteral("label"), QStringLiteral("back"));
        page.showSingleResult(backRecord);
        QCOMPARE(predict->nextInFocusChain(), back);
        QCOMPARE(back->nextInFocusChain(), front);
        QCOMPARE(front->nextInFocusChain(), reject);

        page.showSingleResult(resultRecord(QStringLiteral("focus-front"), frontPath));
        QCOMPARE(predict->nextInFocusChain(), front);
        QCOMPARE(front->nextInFocusChain(), back);
        QCOMPARE(back->nextInFocusChain(), reject);
    }

    void submitFailureKeepsPredictionAndCanRetry() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("submit-retry.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;
        page.setCurrentWorkpiece(QStringLiteral("m1"), QStringLiteral("模型一"));
        page.setBackendAvailable(true, false, QString());
        page.showSingleResult(resultRecord(QStringLiteral("submit-retry"), path));
        auto *front = page.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
        auto *list = page.findChild<QListWidget *>(QStringLiteral("recentInspectionList"));
        QVERIFY(front != nullptr);
        QVERIFY(list != nullptr);
        QSignalSpy confirmationSpy(&page, &InspectionPage::confirmationRequested);

        front->click();
        QCOMPARE(confirmationSpy.count(), 1);
        page.handleBackendResponse(QStringLiteral("submit_confirmation"), QJsonObject{
            {QStringLiteral("job"), QJsonObject{
                {QStringLiteral("job_id"), QStringLiteral("job-failed")},
                {QStringLiteral("state"), QStringLiteral("failed")},
                {QStringLiteral("error"), QStringLiteral("写入队列失败")}}},
        });

        QCOMPARE(page.uiState(), InspectionUiState::Completed);
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("resultLabel"))->text()
                    .contains(QStringLiteral("正面")));
        const QString resultLine = list->item(0)->text().section(QLatin1Char('\n'), 0, 0);
        QVERIFY(resultLine.contains(QStringLiteral("正面")));
        QVERIFY(!resultLine.contains(QStringLiteral(" · 失败")));
        QVERIFY(list->item(0)->text().contains(QStringLiteral("写入失败")));
        QVERIFY(front->isEnabled());
        front->click();
        QCOMPARE(confirmationSpy.count(), 2);

        page.handleBackendFailure(QStringLiteral("submit_confirmation"),
                                  QStringLiteral("MODEL_ERROR"),
                                  QStringLiteral("再次失败"));
        page.selectRecentRecord(QStringLiteral("submit-retry"));
        QCOMPARE(page.uiState(), InspectionUiState::Completed);
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("resultLabel"))->text()
                    .contains(QStringLiteral("正面")));
    }

    void summarySeparatesGlobalAndLocalDecisionEvidence() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("local-override.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;
        InspectionRecord record = resultRecord(QStringLiteral("local-override"), path);
        record.label = QStringLiteral("back");
        record.response.insert(QStringLiteral("label"), QStringLiteral("back"));
        record.response.insert(QStringLiteral("global_prediction"), QStringLiteral("front"));
        record.response.insert(QStringLiteral("local_prediction"), QStringLiteral("back"));
        record.response.insert(QStringLiteral("decision_source"),
                               QStringLiteral("local_override"));

        page.showSingleResult(record);

        const QString summary = page.findChild<QTextEdit *>(
            QStringLiteral("evidenceTextEdit"))->toPlainText();
        QVERIFY(summary.contains(QStringLiteral("全局特征：支持正面")));
        QVERIFY(summary.contains(QStringLiteral("局部匹配：支持反面（采用此结果）")));
        QVERIFY(!summary.contains(QStringLiteral("全局特征：支持反面")));
    }

    void backendRecoveryRestoresFailuresForBothModes() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString singlePath = writeImage(directory, QStringLiteral("single-failed.png"));
        const QString batchPath = writeImage(directory, QStringLiteral("batch-failed.png"));
        QVERIFY(!singlePath.isEmpty());
        QVERIFY(!batchPath.isEmpty());
        InspectionPage page;
        page.setMode(InspectionMode::Single);
        page.setSingleImagePath(singlePath);
        page.showSingleFailure(QStringLiteral("单图预测失败"));
        page.setMode(InspectionMode::Batch);
        page.setSingleImagePath(batchPath);
        page.showSingleFailure(QStringLiteral("批量预测失败"));

        page.setBackendAvailable(false, false, QStringLiteral("连接中断"));
        QCOMPARE(page.uiState(), InspectionUiState::BackendUnavailable);
        page.setMode(InspectionMode::Single);
        QCOMPARE(page.uiState(), InspectionUiState::BackendUnavailable);

        page.setBackendAvailable(true, false, QString());
        QCOMPARE(page.uiState(), InspectionUiState::Failed);
        QCOMPARE(page.singleImagePath(), singlePath);
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("resultMessageLabel"))->text()
                    .contains(QStringLiteral("单图预测失败")));
        page.setMode(InspectionMode::Batch);
        QCOMPARE(page.uiState(), InspectionUiState::Failed);
        QCOMPARE(page.singleImagePath(), batchPath);
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("resultMessageLabel"))->text()
                    .contains(QStringLiteral("批量预测失败")));
    }

    void visibleModeToggleChangesRealInspectionMode() {
        InspectionPage page;
        page.show();
        auto *single = page.findChild<QPushButton *>(QStringLiteral("singleModeButton"));
        auto *batch = page.findChild<QPushButton *>(QStringLiteral("batchModeButton"));
        QVERIFY(single != nullptr);
        QVERIFY(batch != nullptr);
        QVERIFY(single->isVisible());
        QVERIFY(batch->isVisible());
        QVERIFY(single->isChecked());

        QTest::mouseClick(batch, Qt::LeftButton);
        QCOMPARE(page.mode(), InspectionMode::Batch);
        QVERIFY(batch->isChecked());
        QTest::mouseClick(single, Qt::LeftButton);
        QCOMPARE(page.mode(), InspectionMode::Single);
    }

    void runningBatchPreventsModeSwitch() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString first = writeImage(directory, QStringLiteral("batch-one.png"));
        const QString second = writeImage(directory, QStringLiteral("batch-two.png"));
        QVERIFY(!first.isEmpty());
        QVERIFY(!second.isEmpty());
        InspectionPage page;
        page.setCurrentWorkpiece(QStringLiteral("m1"), QStringLiteral("M1"));
        page.setBackendAvailable(true, false, QString());
        page.beginBatch({first, second}, QStringLiteral("m1"));
        QCOMPARE(page.mode(), InspectionMode::Batch);
        auto *single = page.findChild<QPushButton *>(QStringLiteral("singleModeButton"));
        auto *batch = page.findChild<QPushButton *>(QStringLiteral("batchModeButton"));
        QVERIFY(single != nullptr);
        QVERIFY(batch != nullptr);
        QVERIFY(!single->isEnabled());
        QTest::mouseClick(single, Qt::LeftButton);
        QCOMPARE(page.mode(), InspectionMode::Batch);
    }

    void longImagePathIsElidedAndRetainedInTooltip() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(
            directory, QStringLiteral("这是一个非常非常长的中文待检测工件图片文件名用于验证省略显示.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;
        page.resize(720, 520);
        page.setSingleImagePath(path);
        page.show();
        QCoreApplication::processEvents();

        auto *label = page.findChild<QLabel *>(QStringLiteral("currentImageLabel"));
        QVERIFY(label != nullptr);
        QCOMPARE(label->toolTip(), QFileInfo(path).absoluteFilePath());
        QVERIFY(label->text().size() < label->toolTip().size());
    }

    void normalModeExposesExactlyOneVisiblePrimaryAction() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = writeImage(directory, QStringLiteral("primary.png"));
        QVERIFY(!path.isEmpty());
        InspectionPage page;
        page.resize(1280, 720);
        page.setCurrentWorkpiece(QStringLiteral("m1"), QStringLiteral("M1"));
        page.setBackendAvailable(true, false, QString());
        page.setSingleImagePath(path);
        page.show();
        QCoreApplication::processEvents();

        int visiblePrimary = 0;
        for (QPushButton *button : page.findChildren<QPushButton *>()) {
            if (button->property("role").toString() == QStringLiteral("primary")
                && button->isVisibleTo(&page)) {
                ++visiblePrimary;
            }
        }
        QCOMPARE(visiblePrimary, 1);
        auto *predict = page.findChild<QPushButton *>(QStringLiteral("predictButton"));
        QVERIFY(predict != nullptr);
        QVERIFY(predict->isVisibleTo(&page));
        const QRect buttonRect(predict->mapTo(&page, QPoint(0, 0)), predict->size());
        QVERIFY(page.rect().contains(buttonRect));
    }

    void inspectionTabOrderReachesModeInputResultAndConfirmation() {
        InspectionPage page;
        auto *single = page.findChild<QPushButton *>(QStringLiteral("singleModeButton"));
        auto *batch = page.findChild<QPushButton *>(QStringLiteral("batchModeButton"));
        auto *choose = page.findChild<QPushButton *>(QStringLiteral("chooseImageButton"));
        auto *predict = page.findChild<QPushButton *>(QStringLiteral("predictButton"));
        auto *confirm = page.findChild<QPushButton *>(QStringLiteral("confirmFrontButton"));
        QVERIFY(single != nullptr);
        QVERIFY(batch != nullptr);
        QVERIFY(choose != nullptr);
        QVERIFY(predict != nullptr);
        QVERIFY(confirm != nullptr);
        QCOMPARE(single->nextInFocusChain(), batch);
        QCOMPARE(batch->nextInFocusChain(), choose);
        QCOMPARE(choose->nextInFocusChain(), predict);
        QCOMPARE(predict->nextInFocusChain(), confirm);
    }

    void modeSelectorDoesNotConsumeVerticalWorkspace() {
        InspectionPage page;
        QWidget *modeBar = page.findChild<QWidget *>(QStringLiteral("inspectionModeBar"));
        QVERIFY(modeBar != nullptr);
        QCOMPARE(modeBar->sizePolicy().verticalPolicy(), QSizePolicy::Maximum);
    }

    void imageSelectedBeforeFirstLayoutFitsAfterShowAndResize() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString imagePath = writeImage(directory, QStringLiteral("before-layout.png"));
        QVERIFY(!imagePath.isEmpty());
        InspectionPage page;
        page.setSingleImagePath(imagePath);
        page.resize(1280, 720);
        page.show();
        QCoreApplication::processEvents();

        auto *view = page.findChild<InspectionImageView *>(
            QStringLiteral("inspectionImageView"));
        QVERIFY(view != nullptr);
        const QRect rendered = view->mapFromScene(view->scene()->sceneRect()).boundingRect();
        const int limitingViewportSide = qMin(view->viewport()->width(),
                                              view->viewport()->height());
        QVERIFY2(qMin(rendered.width(), rendered.height())
                     >= limitingViewportSide * 0.45,
                 qPrintable(QStringLiteral("rendered=%1x%2 viewport=%3x%4")
                                .arg(rendered.width()).arg(rendered.height())
                                .arg(view->viewport()->width())
                                .arg(view->viewport()->height())));
    }
};

QTEST_MAIN(TestInspectionPage)
#include "test_inspectionpage.moc"
