#include <QtTest/QtTest>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include "../appconfig.h"

class TestAppConfig : public QObject {
    Q_OBJECT

private:
    static QString writeConfig(const QTemporaryDir &temporary, QJsonObject object) {
        const QString path = temporary.filePath(QStringLiteral("配置.json"));
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            return QString();
        }
        file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
        return path;
    }

    static QJsonObject validConfig(const QTemporaryDir &temporary) {
        const QString root = temporary.path() + QStringLiteral("/模型库");
        QDir().mkpath(root);
        const QString python = temporary.filePath(QStringLiteral("python.exe"));
        const QString backend = temporary.filePath(QStringLiteral("服务.py"));
        QFile(python).open(QIODevice::WriteOnly);
        QFile(backend).open(QIODevice::WriteOnly);
        return {
            {QStringLiteral("python_executable"), python},
            {QStringLiteral("backend_script"), backend},
            {QStringLiteral("project_root"), temporary.path()},
            {QStringLiteral("model_dir"), root},
            {QStringLiteral("library_dir"), temporary.filePath(QStringLiteral("运行库"))},
            {QStringLiteral("host"), QStringLiteral("127.0.0.1")},
            {QStringLiteral("port"), 37651},
            {QStringLiteral("startup_timeout_ms"), 120000},
            {QStringLiteral("request_timeout_ms"), 120000},
        };
    }

private slots:
    void rejectsNonLoopbackHost() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validConfig(temporary);
        object[QStringLiteral("host")] = QStringLiteral("0.0.0.0");
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("127.0.0.1")));
    }

    void acceptsChinesePathsAndDefaults() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QJsonObject object = validConfig(temporary);
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY2(config.has_value(), qPrintable(error));
        QCOMPARE(config->host, QHostAddress(QStringLiteral("127.0.0.1")));
        QCOMPARE(config->port, quint16(37651));
        QCOMPARE(config->startupTimeoutMs, 120000);
        QCOMPARE(config->requestTimeoutMs, 120000);
        QCOMPARE(config->libraryDir, temporary.filePath(QStringLiteral("运行库")));
        QCOMPARE(config->localSearchMode, QStringLiteral("adaptive"));
    }

    void acceptsLocalSearchModes_data() {
        QTest::addColumn<QString>("mode");
        QTest::newRow("adaptive") << QStringLiteral("adaptive");
        QTest::newRow("exhaustive") << QStringLiteral("exhaustive");
    }

    void acceptsLocalSearchModes() {
        QFETCH(QString, mode);
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validConfig(temporary);
        object[QStringLiteral("local_search_mode")] = mode;
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY2(config.has_value(), qPrintable(error));
        QCOMPARE(config->localSearchMode, mode);
    }

    void rejectsUnknownLocalSearchMode() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validConfig(temporary);
        object[QStringLiteral("local_search_mode")] = QStringLiteral("fast");
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("local_search_mode")));
    }

    void rejectsOutOfRangePort() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validConfig(temporary);
        object[QStringLiteral("port")] = 70000;
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("port")));
    }
};

QTEST_MAIN(TestAppConfig)
#include "test_appconfig.moc"
