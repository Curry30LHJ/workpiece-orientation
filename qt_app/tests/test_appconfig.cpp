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

    static QJsonObject validPackagedConfig(const QTemporaryDir &temporary,
                                           const QString &edition) {
        QDir root(temporary.path());
        root.mkpath(QStringLiteral("backend/resources"));
        root.mkpath(QStringLiteral("models/shitu_rec"));
        QFile(root.filePath(QStringLiteral("backend/orientation_backend.exe"))).open(
            QIODevice::WriteOnly
        );
        QFile(root.filePath(QStringLiteral("backend/resources/inference_general.yaml"))).open(
            QIODevice::WriteOnly
        );
        return {
            {QStringLiteral("launch_mode"), QStringLiteral("packaged_executable")},
            {QStringLiteral("backend_executable"),
             QStringLiteral("backend/orientation_backend.exe")},
            {QStringLiteral("project_root"), QStringLiteral(".")},
            {QStringLiteral("paddle_config"),
             QStringLiteral("backend/resources/inference_general.yaml")},
            {QStringLiteral("model_dir"), QStringLiteral("models/shitu_rec")},
            {QStringLiteral("data_root"), QStringLiteral("data")},
            {QStringLiteral("model_sha256"), QString(64, QLatin1Char('a'))},
            {QStringLiteral("compute_device"), edition},
            {QStringLiteral("edition"), edition},
            {QStringLiteral("package_version"), QStringLiteral("1.0.0")},
            {QStringLiteral("inference_mode"), QStringLiteral("fast_geometry")},
            {QStringLiteral("host"), QStringLiteral("127.0.0.1")},
            {QStringLiteral("port"), 37651},
            // The frozen Paddle runtime can take well over one minute to
            // initialize on an offline customer machine.  Packaged configs
            // therefore use the same ten-minute watchdog as the release
            // builder; this is a ceiling, not an intentional startup delay.
            {QStringLiteral("startup_timeout_ms"), 600000},
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
        QCOMPARE(config->inferenceMode, QStringLiteral("legacy"));
        QCOMPARE(config->launchMode, BackendLaunchMode::PythonScript);
    }

    void resolvesDevelopmentFilesystemFieldsRelativeToConfig() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QDir root(temporary.path());
        QVERIFY(root.mkpath(QStringLiteral("runtime")));
        QVERIFY(root.mkpath(QStringLiteral("models")));
        QFile(root.filePath(QStringLiteral("runtime/python.exe"))).open(QIODevice::WriteOnly);
        QFile(root.filePath(QStringLiteral("runtime/service.py"))).open(QIODevice::WriteOnly);
        QJsonObject object = validConfig(temporary);
        object[QStringLiteral("python_executable")] = QStringLiteral("runtime/python.exe");
        object[QStringLiteral("backend_script")] = QStringLiteral("runtime/service.py");
        object[QStringLiteral("project_root")] = QStringLiteral(".");
        object[QStringLiteral("model_dir")] = QStringLiteral("models");
        object[QStringLiteral("library_dir")] = QStringLiteral("runtime_library");
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY2(config.has_value(), qPrintable(error));
        QCOMPARE(config->pythonExecutable,
                 temporary.filePath(QStringLiteral("runtime/python.exe")));
        QCOMPARE(config->backendScript, temporary.filePath(QStringLiteral("runtime/service.py")));
        QCOMPARE(config->projectRoot, temporary.path());
        QCOMPARE(config->modelDir, temporary.filePath(QStringLiteral("models")));
        QCOMPARE(config->libraryDir, temporary.filePath(QStringLiteral("runtime_library")));
    }

    void acceptsPackagedConfigFromLongChineseRelativeRoot() {
        QTemporaryDir scaffold;
        QVERIFY(scaffold.isValid());
        QString packageParent = scaffold.path();
        const QString component = QStringLiteral("中文 空格便携包目录");
        while (packageParent.size() < 180) {
            packageParent += QLatin1Char('/') + component;
        }
        QVERIFY(QDir().mkpath(packageParent));
        QTemporaryDir temporary(packageParent + QStringLiteral("/配置 根 XXXXXX"));
        QVERIFY(temporary.isValid());
        QVERIFY(temporary.path().size() >= 180);
        const QJsonObject object = validPackagedConfig(temporary, QStringLiteral("gpu"));
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY2(config.has_value(), qPrintable(error));
        QCOMPARE(config->launchMode, BackendLaunchMode::PackagedExecutable);
        QCOMPARE(config->backendExecutable,
                 temporary.filePath(QStringLiteral("backend/orientation_backend.exe")));
        QCOMPARE(config->paddleConfigPath,
                 temporary.filePath(QStringLiteral("backend/resources/inference_general.yaml")));
        QCOMPARE(config->projectRoot, temporary.path());
        QCOMPARE(config->modelDir, temporary.filePath(QStringLiteral("models/shitu_rec")));
        QCOMPARE(config->dataRoot, temporary.filePath(QStringLiteral("data")));
        QCOMPARE(config->modelSha256, QString(64, QLatin1Char('a')));
        QCOMPARE(config->computeDevice, QStringLiteral("gpu"));
        QCOMPARE(config->edition, QStringLiteral("gpu"));
        QCOMPARE(config->packageVersion, QStringLiteral("1.0.0"));
    }

    void rejectsPackagedLegacyInferenceMode() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validPackagedConfig(temporary, QStringLiteral("cpu"));
        object[QStringLiteral("inference_mode")] = QStringLiteral("legacy");
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("fast_geometry")));
    }

    void rejectsPackagedEditionDeviceMismatch() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validPackagedConfig(temporary, QStringLiteral("cpu"));
        object[QStringLiteral("compute_device")] = QStringLiteral("gpu");
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("edition")));
        QVERIFY(error.contains(QStringLiteral("compute_device")));
    }

    void rejectsPackagedMissingRequiredFields_data() {
        QTest::addColumn<QString>("field");
        QTest::newRow("compute device") << QStringLiteral("compute_device");
        QTest::newRow("edition") << QStringLiteral("edition");
        QTest::newRow("package version") << QStringLiteral("package_version");
    }

    void rejectsPackagedMissingRequiredFields() {
        QFETCH(QString, field);
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validPackagedConfig(temporary, QStringLiteral("gpu"));
        object.remove(field);
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(field));
    }

    void rejectsPackagedUppercaseModelSha256() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validPackagedConfig(temporary, QStringLiteral("cpu"));
        object[QStringLiteral("model_sha256")] = QString(64, QLatin1Char('A'));
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("model_sha256")));
    }

    void rejectsPackagedDataRootWithoutExistingParent() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validPackagedConfig(temporary, QStringLiteral("cpu"));
        object[QStringLiteral("data_root")] = QStringLiteral("not-created/data");
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("data_root")));
    }

    void rejectsPackagedDataRootThatIsAFile() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QFile(temporary.filePath(QStringLiteral("data"))).open(QIODevice::WriteOnly);
        const QJsonObject object = validPackagedConfig(temporary, QStringLiteral("cpu"));
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("data_root")));
    }

    void acceptsInferenceModes_data() {
        QTest::addColumn<QString>("mode");
        QTest::newRow("legacy") << QStringLiteral("legacy");
        QTest::newRow("fast") << QStringLiteral("fast_geometry");
        QTest::newRow("compare") << QStringLiteral("compare");
    }

    void acceptsInferenceModes() {
        QFETCH(QString, mode);
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validConfig(temporary);
        object[QStringLiteral("inference_mode")] = mode;
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY2(config.has_value(), qPrintable(error));
        QCOMPARE(config->inferenceMode, mode);
    }

    void rejectsUnknownInferenceMode() {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QJsonObject object = validConfig(temporary);
        object[QStringLiteral("inference_mode")] = QStringLiteral("automatic");
        QString error;

        const auto config = AppConfig::load(writeConfig(temporary, object), &error);

        QVERIFY(!config.has_value());
        QVERIFY(error.contains(QStringLiteral("inference_mode")));
        QVERIFY(error.contains(QStringLiteral("legacy")));
        QVERIFY(error.contains(QStringLiteral("fast_geometry")));
        QVERIFY(error.contains(QStringLiteral("compare")));
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
