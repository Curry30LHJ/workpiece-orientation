#include <QtTest/QtTest>

#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QSignalSpy>
#include <QTableWidget>
#include <QTemporaryDir>

#include "../workpiecelibrarypage.h"

namespace {

QStringList writeImages(const QString &directory, const QString &prefix, int count,
                        int markerBase) {
    QStringList paths;
    for (int index = 0; index < count; ++index) {
        const QString path = QStringLiteral("%1/%2-%3.png")
                                 .arg(directory, prefix)
                                 .arg(index);
        QImage image(32, 32, QImage::Format_RGB32);
        image.fill(qRgb((markerBase + index) & 0xff,
                        (markerBase + index * 3) & 0xff,
                        (markerBase + index * 7) & 0xff));
        image.setPixel(index % 32, (index * 5) % 32,
                       qRgb((markerBase + index * 11) & 0xff, 17, 193));
        if (!image.save(path)) return {};
        paths.append(path);
    }
    return paths;
}

QString copyImage(const QString &source, const QString &target) {
    QFile::remove(target);
    return QFile::copy(source, target) ? target : QString();
}

QJsonObject summary(const QString &id, const QString &name, int front, int back,
                    bool detectable = true) {
    return QJsonObject{
        {QStringLiteral("id"), id},
        {QStringLiteral("name"), name},
        {QStringLiteral("template_counts"),
         QJsonObject{{QStringLiteral("front"), front},
                     {QStringLiteral("back"), back}}},
        {QStringLiteral("geometry_rule_count"), 2},
        {QStringLiteral("detectable"), detectable},
        {QStringLiteral("updated_at"), QStringLiteral("2026-08-25T09:30:00+00:00")},
    };
}

} // namespace

class TestWorkpieceLibraryPage : public QObject {
    Q_OBJECT

private slots:
    void unequalCountsAndWarningsDoNotBlockRegistration() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieceName(QStringLiteral("M1"));
        page.setTemplatePaths(writeImages(directory.path(), QStringLiteral("front"), 1, 10),
                              writeImages(directory.path(), QStringLiteral("back"), 12, 80));

        QCOMPARE(page.findChild<QLabel *>(QStringLiteral("frontTemplatesLabel"))->text(),
                 QStringLiteral("正面已选择 1 张"));
        QCOMPARE(page.findChild<QLabel *>(QStringLiteral("backTemplatesLabel"))->text(),
                 QStringLiteral("反面已选择 12 张"));
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("templateWarningLabel"))->text()
                    .contains(QStringLiteral("模板较少")));
        QVERIFY(page.findChild<QPushButton *>(QStringLiteral("registerButton"))->isEnabled());
    }

    void everySelectedImageIsEmittedInOriginalOrder() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        const QStringList frontPaths = writeImages(
            directory.path(), QStringLiteral("front"), 5, 10);
        const QStringList backPaths = writeImages(
            directory.path(), QStringLiteral("back"), 10, 80);
        QSignalSpy spy(&page, &WorkpieceLibraryPage::commandRequested);
        page.setTemplatePaths(frontPaths, backPaths);
        page.setWorkpieceName(QStringLiteral("M1"));
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                          Qt::DirectConnection));
        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy.first().at(0).toString(), QStringLiteral("register"));
        const QJsonObject fields = spy.first().at(1).toJsonObject();
        QCOMPARE(fields.value("replace").toBool(), false);
        QCOMPARE(fields.value("progress_events").toBool(), true);
        QCOMPARE(fields.value("front_images").toArray(), QJsonArray::fromStringList(frontPaths));
        QCOMPARE(fields.value("back_images").toArray(), QJsonArray::fromStringList(backPaths));
    }

    void emptyUnreadableAndDuplicateTemplatesBlockRegistration() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QStringList front = writeImages(
            directory.path(), QStringLiteral("front"), 2, 10);
        const QStringList back = writeImages(
            directory.path(), QStringLiteral("back"), 2, 80);
        const QString unreadable = directory.filePath(QStringLiteral("broken.png"));
        QFile unreadableFile(unreadable);
        QVERIFY(unreadableFile.open(QIODevice::WriteOnly));
        unreadableFile.write("not an image");
        unreadableFile.close();
        const QString copied = copyImage(
            front.first(), directory.filePath(QStringLiteral("same-content.png")));
        QVERIFY(!copied.isEmpty());

        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieceName(QStringLiteral("M1"));
        QSignalSpy spy(&page, &WorkpieceLibraryPage::commandRequested);

        page.setTemplatePaths({}, back);
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration", Qt::DirectConnection));
        QCOMPARE(spy.count(), 0);

        page.setTemplatePaths(QStringList{unreadable}, back);
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration", Qt::DirectConnection));
        QCOMPARE(spy.count(), 0);

        page.setTemplatePaths(front, QStringList{front.first()});
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration", Qt::DirectConnection));
        QCOMPARE(spy.count(), 0);

        page.setTemplatePaths(front, QStringList{copied});
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration", Qt::DirectConnection));
        QCOMPARE(spy.count(), 0);
        QVERIFY(page.findChild<QLabel *>(QStringLiteral("libraryMessageLabel"))->text()
                    .contains(QStringLiteral("重复")));
    }

    void moreThanThirtyTemplatesOnlyWarns() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieceName(QStringLiteral("M-many"));
        page.setTemplatePaths(writeImages(directory.path(), QStringLiteral("front"), 31, 10),
                              writeImages(directory.path(), QStringLiteral("back"), 1, 180));
        QSignalSpy spy(&page, &WorkpieceLibraryPage::commandRequested);

        QVERIFY(page.findChild<QLabel *>(QStringLiteral("templateWarningLabel"))->text()
                    .contains(QStringLiteral("耗时")));
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration", Qt::DirectConnection));
        QCOMPARE(spy.count(), 1);
    }

    void searchFiltersLibraryWithoutChangingDetectionTarget() {
        WorkpieceLibraryPage page;
        QSignalSpy targetSpy(&page, &WorkpieceLibraryPage::setCurrentWorkpieceRequested);
        QSignalSpy commandSpy(&page, &WorkpieceLibraryPage::commandRequested);
        page.setWorkpieces(
            QJsonArray{summary(QStringLiteral("m1"), QStringLiteral("泵体 A"), 5, 6),
                       summary(QStringLiteral("m2"), QStringLiteral("法兰 B"), 7, 8)},
            QStringLiteral("m2"));

        auto *search = page.findChild<QLineEdit *>(QStringLiteral("librarySearchEdit"));
        auto *list = page.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        QVERIFY(search != nullptr);
        QVERIFY(list != nullptr);
        search->setText(QStringLiteral("泵体"));
        QCOMPARE(list->count(), 1);
        list->setCurrentRow(0);
        QTRY_COMPARE(commandSpy.count(), 1);
        QCOMPARE(commandSpy.first().at(0).toString(), QStringLiteral("get_workpiece_details"));
        QCOMPARE(targetSpy.count(), 0);
        QCOMPARE(page.browsedWorkpieceId(), QStringLiteral("m1"));

        auto *confirmation = page.findChild<QLineEdit *>(
            QStringLiteral("recycleNameConfirmationEdit"));
        auto *recycle = page.findChild<QPushButton *>(
            QStringLiteral("deleteWorkpieceButton"));
        QVERIFY(confirmation != nullptr);
        QVERIFY(recycle != nullptr);
        page.setBackendState(BackendUiState::Ready, QString());
        confirmation->setText(QStringLiteral("泵体 A"));
        QVERIFY(recycle->isEnabled());

        search->setText(QStringLiteral("法兰"));
        QCOMPARE(list->count(), 1);
        QCOMPARE(page.browsedWorkpieceId(), QString());
        QVERIFY(!recycle->isEnabled());
        QCOMPARE(targetSpy.count(), 0);
        QVERIFY(list->item(0)->text().contains(QStringLiteral("[当前检测]")));
    }

    void exactNameConfirmationIsRequiredForRecycle() {
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieces(QJsonArray{summary(QStringLiteral("m1"), QStringLiteral("泵体 A"), 5, 6)},
                           QStringLiteral("m1"));
        auto *list = page.findChild<QListWidget *>(QStringLiteral("libraryWorkpieceList"));
        auto *confirmation = page.findChild<QLineEdit *>(
            QStringLiteral("recycleNameConfirmationEdit"));
        auto *recycle = page.findChild<QPushButton *>(QStringLiteral("deleteWorkpieceButton"));
        QVERIFY(list != nullptr);
        QVERIFY(confirmation != nullptr);
        QVERIFY(recycle != nullptr);
        list->setCurrentRow(0);
        confirmation->setText(QStringLiteral("泵体"));
        QVERIFY(!recycle->isEnabled());
        confirmation->setText(QStringLiteral("泵体 A"));
        QVERIFY(recycle->isEnabled());
        QSignalSpy spy(&page, &WorkpieceLibraryPage::commandRequested);
        recycle->click();
        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy.first().at(0).toString(), QStringLiteral("recycle_workpiece"));
        const QJsonObject fields = spy.first().at(1).toJsonObject();
        QCOMPARE(fields.value(QStringLiteral("workpiece_id")).toString(), QStringLiteral("m1"));
        QVERIFY(!fields.value(QStringLiteral("operation_id")).toString().isEmpty());
    }

    void detailsRenderEveryTemplateWithoutTruncation() {
        WorkpieceLibraryPage page;
        QJsonArray templates;
        for (int index = 0; index < 37; ++index) {
            templates.append(QJsonObject{
                {QStringLiteral("template_id"), QStringLiteral("front:%1.png").arg(index)},
                {QStringLiteral("direction"), index < 31 ? QStringLiteral("front")
                                                         : QStringLiteral("back")},
                {QStringLiteral("preview_path"), QStringLiteral("C:/templates/%1.png").arg(index)},
                {QStringLiteral("source"), QStringLiteral("initial_registration")},
                {QStringLiteral("added_at"), QStringLiteral("2026-08-25T10:00:00+00:00")},
                {QStringLiteral("readable"), index != 36},
            });
        }
        page.setWorkpieceDetails(QJsonObject{
            {QStringLiteral("id"), QStringLiteral("m1")},
            {QStringLiteral("name"), QStringLiteral("M1")},
            {QStringLiteral("templates"), templates},
        });

        auto *table = page.findChild<QTableWidget *>(QStringLiteral("templateDetailsTable"));
        QVERIFY(table != nullptr);
        QCOMPARE(table->rowCount(), 37);
        QCOMPARE(table->item(36, 4)->text(), QStringLiteral("无法读取"));
        QCOMPARE(table->item(36, 0)->data(Qt::UserRole).toString(),
                 QStringLiteral("front:36.png"));
    }

    void evolutionRowsUseJobIdAndPreserveFailures() {
        WorkpieceLibraryPage page;
        QSignalSpy taskSpy(&page, &WorkpieceLibraryPage::taskStatusChanged);
        page.setEvolutionJobs(QJsonArray{
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("job-1")},
                        {QStringLiteral("workpiece_id"), QStringLiteral("m1")},
                        {QStringLiteral("state"), QStringLiteral("failed")},
                        {QStringLiteral("phase"), QStringLiteral("features")},
                        {QStringLiteral("completed"), 3},
                        {QStringLiteral("total"), 5},
                        {QStringLiteral("elapsed_ms"), QJsonValue()},
                        {QStringLiteral("error"), QStringLiteral("feature extraction failed")}},
        });
        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> failedTask = taskSpy.takeFirst();
        QCOMPARE(failedTask.at(1).toString(), QStringLiteral("failed"));
        QCOMPARE(failedTask.at(4).toLongLong(), qint64(-1));
        page.setEvolutionJobs(QJsonArray{
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("job-1")},
                        {QStringLiteral("workpiece_id"), QStringLiteral("m1")},
                        {QStringLiteral("state"), QStringLiteral("failed")},
                        {QStringLiteral("phase"), QStringLiteral("features")},
                        {QStringLiteral("completed"), 3},
                        {QStringLiteral("total"), 5}},
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("job-2")},
                        {QStringLiteral("workpiece_id"), QStringLiteral("m2")},
                        {QStringLiteral("state"), QStringLiteral("completed")},
                        {QStringLiteral("phase"), QStringLiteral("active")},
                        {QStringLiteral("completed"), 1},
                        {QStringLiteral("total"), 1}},
        });
        QCOMPARE(taskSpy.count(), 1);
        QCOMPARE(taskSpy.takeFirst().at(1).toString(), QStringLiteral("active"));

        auto *table = page.findChild<QTableWidget *>(QStringLiteral("evolutionJobsTable"));
        QVERIFY(table != nullptr);
        QCOMPARE(table->rowCount(), 2);
        QCOMPARE(table->item(0, 0)->text(), QStringLiteral("job-1"));
        QCOMPARE(table->item(0, 5)->text(), QStringLiteral("feature extraction failed"));
    }

    void evolutionTaskChoosesNewestActiveBeforeTerminal() {
        WorkpieceLibraryPage page;
        QSignalSpy taskSpy(&page, &WorkpieceLibraryPage::taskStatusChanged);
        page.setEvolutionJobs(QJsonArray{
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("building-new")},
                        {QStringLiteral("state"), QStringLiteral("building")},
                        {QStringLiteral("phase"), QStringLiteral("features")},
                        {QStringLiteral("completed"), 4},
                        {QStringLiteral("total"), 9},
                        {QStringLiteral("last_submitted_at"), 2000.0}},
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("queued-old")},
                        {QStringLiteral("state"), QStringLiteral("queued")},
                        {QStringLiteral("phase"), QStringLiteral("queued")},
                        {QStringLiteral("completed"), 0},
                        {QStringLiteral("total"), 3},
                        {QStringLiteral("last_submitted_at"), 1000.0}},
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("failed-newest")},
                        {QStringLiteral("state"), QStringLiteral("failed")},
                        {QStringLiteral("completed"), 8},
                        {QStringLiteral("total"), 8},
                        {QStringLiteral("last_submitted_at"), 3000.0}},
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("completed-mid")},
                        {QStringLiteral("state"), QStringLiteral("completed")},
                        {QStringLiteral("phase"), QStringLiteral("active")},
                        {QStringLiteral("completed"), 7},
                        {QStringLiteral("total"), 7},
                        {QStringLiteral("last_submitted_at"), 1500.0}},
        });

        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> activeTask = taskSpy.takeFirst();
        QCOMPARE(activeTask.at(1).toString(), QStringLiteral("features"));
        QCOMPARE(activeTask.at(2).toInt(), 4);
        QCOMPARE(activeTask.at(3).toInt(), 9);

        page.setEvolutionJobs(QJsonArray{
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("completed-latest")},
                        {QStringLiteral("state"), QStringLiteral("completed")},
                        {QStringLiteral("phase"), QStringLiteral("active")},
                        {QStringLiteral("completed"), 6},
                        {QStringLiteral("total"), 6},
                        {QStringLiteral("last_submitted_at"), 5000.0}},
            QJsonObject{{QStringLiteral("job_id"), QStringLiteral("failed-older")},
                        {QStringLiteral("state"), QStringLiteral("failed")},
                        {QStringLiteral("completed"), 2},
                        {QStringLiteral("total"), 5},
                        {QStringLiteral("last_submitted_at"), 4000.0}},
        });

        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> terminalTask = taskSpy.takeFirst();
        QCOMPARE(terminalTask.at(1).toString(), QStringLiteral("active"));
        QCOMPARE(terminalTask.at(2).toInt(), 6);
        QCOMPARE(terminalTask.at(3).toInt(), 6);
    }

    void evolutionRetryClearsErrorOnlyWhenFieldIsPresent() {
        WorkpieceLibraryPage page;
        auto *table = page.findChild<QTableWidget *>(QStringLiteral("evolutionJobsTable"));
        QVERIFY(table != nullptr);
        page.setEvolutionJobs(QJsonArray{QJsonObject{
            {QStringLiteral("job_id"), QStringLiteral("job-retry")},
            {QStringLiteral("state"), QStringLiteral("failed")},
            {QStringLiteral("error"), QStringLiteral("old failure")},
        }});
        QCOMPARE(table->item(0, 5)->text(), QStringLiteral("old failure"));

        page.setEvolutionJobs(QJsonArray{QJsonObject{
            {QStringLiteral("job_id"), QStringLiteral("job-retry")},
            {QStringLiteral("state"), QStringLiteral("queued")},
            {QStringLiteral("error"), QJsonValue(QJsonValue::Null)},
        }});
        QCOMPARE(table->item(0, 5)->text(), QString());

        page.setEvolutionJobs(QJsonArray{QJsonObject{
            {QStringLiteral("job_id"), QStringLiteral("job-retry")},
            {QStringLiteral("state"), QStringLiteral("building")},
            {QStringLiteral("error"), QStringLiteral("retry warning")},
        }});
        QCOMPARE(table->item(0, 5)->text(), QStringLiteral("retry warning"));

        page.setEvolutionJobs(QJsonArray{QJsonObject{
            {QStringLiteral("job_id"), QStringLiteral("job-retry")},
            {QStringLiteral("state"), QStringLiteral("building")},
        }});
        QCOMPARE(table->item(0, 5)->text(), QStringLiteral("retry warning"));

        page.setEvolutionJobs(QJsonArray{QJsonObject{
            {QStringLiteral("job_id"), QStringLiteral("job-retry")},
            {QStringLiteral("state"), QStringLiteral("completed")},
            {QStringLiteral("error"), QString()},
        }});
        QCOMPARE(table->item(0, 5)->text(), QString());
    }

    void backendUnavailableKeepsTheEditingDraft() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        const QStringList front = writeImages(directory.path(), QStringLiteral("front"), 2, 10);
        const QStringList back = writeImages(directory.path(), QStringLiteral("back"), 3, 80);
        page.setWorkpieceName(QStringLiteral("M-draft"));
        page.setTemplatePaths(front, back);
        QVERIFY(page.hasUnsavedChanges());
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                          Qt::DirectConnection));
        auto *registerButton = page.findChild<QPushButton *>(
            QStringLiteral("registerButton"));
        QVERIFY(registerButton != nullptr);
        QVERIFY(!registerButton->isEnabled());

        page.setBackendState(BackendUiState::Error, QStringLiteral("连接中断"));
        page.setBackendState(BackendUiState::Ready, QString());

        QVERIFY(page.hasUnsavedChanges());
        QCOMPARE(page.findChild<QLineEdit *>(QStringLiteral("workpieceNameEdit"))->text(),
                 QStringLiteral("M-draft"));
        QCOMPARE(page.findChild<QLabel *>(QStringLiteral("frontTemplatesLabel"))->text(),
                 QStringLiteral("正面已选择 2 张"));
        QCOMPARE(page.findChild<QLabel *>(QStringLiteral("backTemplatesLabel"))->text(),
                 QStringLiteral("反面已选择 3 张"));
        QVERIFY(registerButton->isEnabled());
    }

    void registrationTaskStatusKeepsLastProgressOnOrdinaryFailure() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieceName(QStringLiteral("M-status"));
        page.setTemplatePaths(
            writeImages(directory.path(), QStringLiteral("status-front"), 2, 10),
            writeImages(directory.path(), QStringLiteral("status-back"), 3, 80));
        QSignalSpy taskSpy(&page, &WorkpieceLibraryPage::taskStatusChanged);

        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                          Qt::DirectConnection));
        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> queued = taskSpy.takeFirst();
        QCOMPARE(queued.at(1).toString(), QStringLiteral("queued"));
        QCOMPARE(queued.at(2).toInt(), 0);
        QCOMPARE(queued.at(3).toInt(), 5);
        QCOMPARE(queued.at(4).toLongLong(), qint64(0));

        page.setRegistrationProgress(QJsonObject{
            {QStringLiteral("phase"), QStringLiteral("features")},
            {QStringLiteral("completed"), 3},
            {QStringLiteral("total"), 5},
        }, 246);
        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> progress = taskSpy.takeFirst();
        QCOMPARE(progress.at(1).toString(), QStringLiteral("features"));
        QCOMPARE(progress.at(2).toInt(), 3);
        QCOMPARE(progress.at(3).toInt(), 5);
        QCOMPARE(progress.at(4).toLongLong(), qint64(246));

        page.handleBackendFailure(QStringLiteral("register"),
                                  QStringLiteral("MODEL_ERROR"),
                                  QStringLiteral("feature failed"));
        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> failed = taskSpy.takeFirst();
        QCOMPARE(failed.at(1).toString(), QStringLiteral("failed"));
        QCOMPARE(failed.at(2).toInt(), 3);
        QCOMPARE(failed.at(3).toInt(), 5);
        QCOMPARE(failed.at(4).toLongLong(), qint64(246));
    }

    void registrationTransportInterruptionEmitsFailedWithLastProgress() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieceName(QStringLiteral("M-interrupted"));
        page.setTemplatePaths(
            writeImages(directory.path(), QStringLiteral("interrupted-front"), 1, 10),
            writeImages(directory.path(), QStringLiteral("interrupted-back"), 3, 80));
        QSignalSpy taskSpy(&page, &WorkpieceLibraryPage::taskStatusChanged);
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                          Qt::DirectConnection));
        taskSpy.clear();
        page.setRegistrationProgress(QJsonObject{
            {QStringLiteral("phase"), QStringLiteral("copying")},
            {QStringLiteral("completed"), 1},
            {QStringLiteral("total"), 4},
        }, 135);
        taskSpy.clear();

        page.setBackendState(BackendUiState::Error, QStringLiteral("connection lost"));

        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> failed = taskSpy.takeFirst();
        QCOMPARE(failed.at(1).toString(), QStringLiteral("failed"));
        QCOMPARE(failed.at(2).toInt(), 1);
        QCOMPARE(failed.at(3).toInt(), 4);
        QCOMPARE(failed.at(4).toLongLong(), qint64(135));
    }

    void decliningOverwritePublishesCancelledTerminalStatus() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieceName(QStringLiteral("M-decline"));
        page.setTemplatePaths(
            writeImages(directory.path(), QStringLiteral("decline-front"), 1, 10),
            writeImages(directory.path(), QStringLiteral("decline-back"), 2, 80));
        page.setReplaceConfirmationHandler([](const QString &) { return false; });
        QSignalSpy taskSpy(&page, &WorkpieceLibraryPage::taskStatusChanged);

        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                          Qt::DirectConnection));
        taskSpy.clear();
        QTest::qWait(25);
        page.handleBackendFailure(QStringLiteral("register"),
                                  QStringLiteral("WORKPIECE_EXISTS"),
                                  QStringLiteral("exists"));

        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> cancelled = taskSpy.takeFirst();
        QCOMPARE(cancelled.at(1).toString(), QStringLiteral("cancelled"));
        QCOMPARE(cancelled.at(2).toInt(), 0);
        QCOMPARE(cancelled.at(3).toInt(), 3);
        QVERIFY(cancelled.at(4).toLongLong() >= 20);
    }

    void registrationFailureUsesLiveElapsedWithoutProgress() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieceName(QStringLiteral("M-no-progress"));
        page.setTemplatePaths(
            writeImages(directory.path(), QStringLiteral("no-progress-front"), 1, 10),
            writeImages(directory.path(), QStringLiteral("no-progress-back"), 1, 80));
        QSignalSpy taskSpy(&page, &WorkpieceLibraryPage::taskStatusChanged);

        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                          Qt::DirectConnection));
        taskSpy.clear();
        QTest::qWait(25);
        page.handleBackendFailure(QStringLiteral("register"),
                                  QStringLiteral("MODEL_ERROR"),
                                  QStringLiteral("failed without progress"));

        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> failed = taskSpy.takeFirst();
        QCOMPARE(failed.at(1).toString(), QStringLiteral("failed"));
        QCOMPARE(failed.at(2).toInt(), 0);
        QCOMPARE(failed.at(3).toInt(), 2);
        QVERIFY(failed.at(4).toLongLong() >= 20);
    }

    void registrationFailureElapsedContinuesAfterSparseProgress() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        page.setWorkpieceName(QStringLiteral("M-sparse-progress"));
        page.setTemplatePaths(
            writeImages(directory.path(), QStringLiteral("sparse-front"), 2, 10),
            writeImages(directory.path(), QStringLiteral("sparse-back"), 2, 80));
        QSignalSpy taskSpy(&page, &WorkpieceLibraryPage::taskStatusChanged);

        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                          Qt::DirectConnection));
        taskSpy.clear();
        QTest::qWait(10);
        page.setRegistrationProgress(QJsonObject{
            {QStringLiteral("phase"), QStringLiteral("features")},
            {QStringLiteral("completed"), 1},
            {QStringLiteral("total"), 4},
        }, -1);
        QCOMPARE(taskSpy.count(), 1);
        const qint64 progressElapsed = taskSpy.takeFirst().at(4).toLongLong();
        QTest::qWait(25);
        page.handleBackendFailure(QStringLiteral("register"),
                                  QStringLiteral("MODEL_ERROR"),
                                  QStringLiteral("failed after sparse progress"));

        QCOMPARE(taskSpy.count(), 1);
        const QList<QVariant> failed = taskSpy.takeFirst();
        QCOMPARE(failed.at(1).toString(), QStringLiteral("failed"));
        QCOMPARE(failed.at(2).toInt(), 1);
        QCOMPARE(failed.at(3).toInt(), 4);
        QVERIFY(failed.at(4).toLongLong() >= progressElapsed + 20);
    }

    void overwriteConfirmationResendsAllPathsWithReplaceTrue() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        const QStringList front = writeImages(directory.path(), QStringLiteral("front"), 5, 10);
        const QStringList back = writeImages(directory.path(), QStringLiteral("back"), 10, 80);
        page.setWorkpieceName(QStringLiteral("M1"));
        page.setTemplatePaths(front, back);
        page.setReplaceConfirmationHandler([](const QString &) { return true; });
        QSignalSpy spy(&page, &WorkpieceLibraryPage::commandRequested);
        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration", Qt::DirectConnection));
        QCOMPARE(spy.count(), 1);

        page.handleBackendFailure(QStringLiteral("register"),
                                  QStringLiteral("WORKPIECE_EXISTS"),
                                  QStringLiteral("exists"));

        QCOMPARE(spy.count(), 2);
        const QJsonObject original = spy.at(0).at(1).toJsonObject();
        const QJsonObject replacement = spy.at(1).at(1).toJsonObject();
        QCOMPARE(replacement.value(QStringLiteral("replace")).toBool(), true);
        QCOMPARE(replacement.value(QStringLiteral("front_images")),
                 original.value(QStringLiteral("front_images")));
        QCOMPARE(replacement.value(QStringLiteral("back_images")),
                 original.value(QStringLiteral("back_images")));
        QCOMPARE(replacement.value(QStringLiteral("name")),
                 original.value(QStringLiteral("name")));
    }

    void discardingEditableDraftKeepsSubmittedOverwriteSnapshot() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        WorkpieceLibraryPage page;
        page.setBackendState(BackendUiState::Ready, QString());
        const QStringList submittedFront = writeImages(
            directory.path(), QStringLiteral("submitted-front"), 2, 10);
        const QStringList submittedBack = writeImages(
            directory.path(), QStringLiteral("submitted-back"), 3, 80);
        page.setWorkpieceName(QStringLiteral("M-submitted"));
        page.setTemplatePaths(submittedFront, submittedBack);
        page.setReplaceConfirmationHandler([](const QString &) { return true; });
        QSignalSpy spy(&page, &WorkpieceLibraryPage::commandRequested);

        QVERIFY(QMetaObject::invokeMethod(&page, "submitRegistration",
                                          Qt::DirectConnection));
        QCOMPARE(spy.count(), 1);
        page.discardEditingDraft();
        QVERIFY(!page.hasUnsavedChanges());

        page.handleBackendFailure(QStringLiteral("register"),
                                  QStringLiteral("WORKPIECE_EXISTS"),
                                  QStringLiteral("exists"));

        QCOMPARE(spy.count(), 2);
        const QJsonObject replacement = spy.at(1).at(1).toJsonObject();
        QCOMPARE(replacement.value(QStringLiteral("replace")).toBool(), true);
        QCOMPARE(replacement.value(QStringLiteral("name")).toString(),
                 QStringLiteral("M-submitted"));
        QCOMPARE(replacement.value(QStringLiteral("front_images")).toArray(),
                 QJsonArray::fromStringList(submittedFront));
        QCOMPARE(replacement.value(QStringLiteral("back_images")).toArray(),
                 QJsonArray::fromStringList(submittedBack));
    }
};

QTEST_MAIN(TestWorkpieceLibraryPage)
#include "test_workpiecelibrarypage.moc"
