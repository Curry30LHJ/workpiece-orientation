#include <QtTest/QtTest>

#include <QApplication>
#include <QDateTime>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QGroupBox>
#include <QImage>
#include <QLabel>
#include <QListWidget>
#include <QMimeData>
#include <QPushButton>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTextEdit>
#include <QUrl>
#include <QVBoxLayout>
#include <QWheelEvent>

#include "../inspectionimageview.h"
#include "../inspectionpage.h"
#include "../inspectiontypes.h"

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

class TestInspectionPage : public QObject {
    Q_OBJECT

private slots:
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
};

QTEST_MAIN(TestInspectionPage)
#include "test_inspectionpage.moc"
