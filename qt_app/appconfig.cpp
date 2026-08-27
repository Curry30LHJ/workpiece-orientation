#include "appconfig.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>

namespace {

void setError(QString *error, const QString &message) {
    if (error != nullptr) {
        *error = message;
    }
}

bool readString(const QJsonObject &object, const QString &key, QString *value, QString *error) {
    const QJsonValue jsonValue = object.value(key);
    if (!jsonValue.isString() || jsonValue.toString().trimmed().isEmpty()) {
        setError(error, QStringLiteral("Missing or empty configuration field: %1").arg(key));
        return false;
    }
    *value = jsonValue.toString();
    return true;
}

bool requireExistingFile(const QString &path, const QString &key, QString *error) {
    if (!QFileInfo::exists(path) || !QFileInfo(path).isFile()) {
        setError(error, QStringLiteral("Configuration path does not exist: %1 (%2)").arg(key, path));
        return false;
    }
    return true;
}

bool requireExistingDirectory(const QString &path, const QString &key, QString *error) {
    if (!QFileInfo::exists(path) || !QFileInfo(path).isDir()) {
        setError(error, QStringLiteral("Configuration path does not exist: %1 (%2)").arg(key, path));
        return false;
    }
    return true;
}

} // namespace

std::optional<AppConfig> AppConfig::load(const QString &path, QString *error) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        setError(error, QStringLiteral("Cannot open configuration file: %1").arg(path));
        return std::nullopt;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        setError(error, QStringLiteral("Invalid JSON configuration: %1").arg(parseError.errorString()));
        return std::nullopt;
    }
    const QJsonObject object = document.object();
    AppConfig config;
    if (!readString(object, QStringLiteral("python_executable"), &config.pythonExecutable, error)
        || !readString(object, QStringLiteral("backend_script"), &config.backendScript, error)
        || !readString(object, QStringLiteral("project_root"), &config.projectRoot, error)
        || !readString(object, QStringLiteral("model_dir"), &config.modelDir, error)
        || !readString(object, QStringLiteral("library_dir"), &config.libraryDir, error)) {
        return std::nullopt;
    }
    const QJsonValue searchModeValue = object.value(QStringLiteral("local_search_mode"));
    if (!searchModeValue.isUndefined()) {
        if (!searchModeValue.isString()) {
            setError(error, QStringLiteral(
                "local_search_mode must be adaptive or exhaustive"
            ));
            return std::nullopt;
        }
        const QString mode = searchModeValue.toString().trimmed().toLower();
        if (mode != QStringLiteral("adaptive")
            && mode != QStringLiteral("exhaustive")) {
            setError(error, QStringLiteral(
                "local_search_mode must be adaptive or exhaustive"
            ));
            return std::nullopt;
        }
        config.localSearchMode = mode;
    }
    const QJsonValue inferenceModeValue = object.value(QStringLiteral("inference_mode"));
    if (!inferenceModeValue.isUndefined()) {
        if (!inferenceModeValue.isString()) {
            setError(error, QStringLiteral(
                "inference_mode must be legacy, fast_geometry, or compare"
            ));
            return std::nullopt;
        }
        const QString mode = inferenceModeValue.toString().trimmed().toLower();
        if (mode != QStringLiteral("legacy")
            && mode != QStringLiteral("fast_geometry")
            && mode != QStringLiteral("compare")) {
            setError(error, QStringLiteral(
                "inference_mode must be legacy, fast_geometry, or compare"
            ));
            return std::nullopt;
        }
        config.inferenceMode = mode;
    }
    const QString hostText = object.value(QStringLiteral("host")).toString();
    if (hostText != QStringLiteral("127.0.0.1")) {
        setError(error, QStringLiteral("host must be 127.0.0.1"));
        return std::nullopt;
    }
    config.host = QHostAddress(hostText);
    const QJsonValue portValue = object.value(QStringLiteral("port"));
    const QJsonValue startupValue = object.value(QStringLiteral("startup_timeout_ms"));
    const QJsonValue requestValue = object.value(QStringLiteral("request_timeout_ms"));
    if (!portValue.isDouble() || !startupValue.isDouble() || !requestValue.isDouble()) {
        setError(error, QStringLiteral("port and timeout fields must be numbers"));
        return std::nullopt;
    }
    const int port = portValue.toInt(-1);
    const int startupTimeout = startupValue.toInt(-1);
    const int requestTimeout = requestValue.toInt(-1);
    if (port < 1 || port > 65535) {
        setError(error, QStringLiteral("port must be between 1 and 65535"));
        return std::nullopt;
    }
    if (startupTimeout <= 0 || requestTimeout <= 0) {
        setError(error, QStringLiteral("timeouts must be positive"));
        return std::nullopt;
    }
    if (!requireExistingFile(config.pythonExecutable, QStringLiteral("python_executable"), error)
        || !requireExistingFile(config.backendScript, QStringLiteral("backend_script"), error)
        || !requireExistingDirectory(config.projectRoot, QStringLiteral("project_root"), error)
        || !requireExistingDirectory(config.modelDir, QStringLiteral("model_dir"), error)) {
        return std::nullopt;
    }
    config.port = static_cast<quint16>(port);
    config.startupTimeoutMs = startupTimeout;
    config.requestTimeoutMs = requestTimeout;
    return config;
}
