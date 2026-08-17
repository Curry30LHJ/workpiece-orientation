#include <QtTest/QtTest>

#include <QPushButton>
#include <QComboBox>
#include <QImage>
#include <QSignalSpy>
#include <QTemporaryDir>

#include "../geometrymaskmanager.h"
#include "../geometryrulecanvas.h"

class TestGeometryMaskManager : public QObject {
    Q_OBJECT

private slots:
    void publishDisabledUntilCompletedValidationMatchesDraft();
    void warningPublishRequiresOverrideReason();
    void saveAndDeleteRulesUpdateDraft();
    void templatePreviewLoadsRepresentativeImage();
};

QJsonObject profileSnapshot(int libraryRevision, int draftRevision, int activeRevision) {
    return QJsonObject{
        {"workpiece_id", "m7"}, {"library_revision", libraryRevision}, {"draft_revision", draftRevision},
        {"active_revision", activeRevision}, {"previous_active_revision", QJsonValue()},
        {"draft", QJsonObject{{"schema_version", 1}, {"directions", QJsonObject{
            {"front", QJsonObject{{"anchor", QJsonValue()}, {"rules", QJsonArray()}}},
            {"back", QJsonObject{{"anchor", QJsonValue()}, {"rules", QJsonArray()}}}}}}}
    };
}

void TestGeometryMaskManager::publishDisabledUntilCompletedValidationMatchesDraft() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 1));
    QVERIFY(!dialog.findChild<QPushButton *>(QStringLiteral("publishButton"))->isEnabled());
    dialog.setValidationJob(QJsonObject{{"job_id", "job-1"}, {"state", "completed"},
                                        {"base_library_revision", 4}, {"base_draft_revision", 2},
                                        {"progress", QJsonObject{{"completed", 2}, {"total", 2}}}});
    QVERIFY(dialog.findChild<QPushButton *>(QStringLiteral("publishButton"))->isEnabled());
}

void TestGeometryMaskManager::warningPublishRequiresOverrideReason() {
    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(profileSnapshot(4, 2, 1));
    dialog.setValidationJob(QJsonObject{{"job_id", "job-1"}, {"state", "completed"},
                                        {"base_library_revision", 4}, {"base_draft_revision", 2},
                                        {"warnings", QJsonArray{QJsonObject{{"code", "effective_area_low"}}}},
                                        {"progress", QJsonObject{{"completed", 2}, {"total", 2}}}});
    QSignalSpy spy(&dialog, &GeometryMaskManagerDialog::publishRequested);
    dialog.findChild<QPushButton *>(QStringLiteral("publishButton"))->click();
    QCOMPARE(spy.count(), 0);
}

void TestGeometryMaskManager::saveAndDeleteRulesUpdateDraft() {
    GeometryMaskManagerDialog dialog;
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

void TestGeometryMaskManager::templatePreviewLoadsRepresentativeImage() {
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

    GeometryMaskManagerDialog dialog;
    dialog.setSnapshot(snapshot);
    auto *combo = dialog.findChild<QComboBox *>(QStringLiteral("templatePreviewCombo"));
    auto *canvas = dialog.findChild<GeometryRuleCanvas *>(QStringLiteral("geometryRuleCanvas"));
    QVERIFY(combo != nullptr);
    QVERIFY(canvas != nullptr);
    QCOMPARE(combo->count(), 1);
    QCOMPARE(canvas->image().size(), QSize(80, 60));
}

QTEST_MAIN(TestGeometryMaskManager)
#include "test_geometrymaskmanager.moc"
