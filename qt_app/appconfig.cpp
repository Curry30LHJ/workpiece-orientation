#include "appconfig.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>

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

bool readOptionalString(const QJsonObject &object, const QString &key, QString *value,
                        QString *error) {
    const QJsonValue jsonValue = object.value(key);
    if (jsonValue.isUndefined()) {
        return true;
    }
    return readString(object, key, value, error);
}

bool readOptionalStringAllowEmpty(const QJsonObject &object, const QString &key,
                                  QString *value, QString *error) {
    const QJsonValue jsonValue = object.value(key);
    if (jsonValue.isUndefined()) {
        return true;
    }
    if (!jsonValue.isString()) {
        setError(error, QStringLiteral("Configuration field must be a string: %1").arg(key));
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

bool requireNativeExecutable(const QString &path, QString *error) {
    const QFileInfo info(path);
    if (!info.exists() || !info.isFile() || info.isSymLink()) {
        setError(error,
                 QStringLiteral("Native PP-ShiTu executable must be a regular file: %1")
                     .arg(path));
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

bool requireDataRootParent(const QString &path, QString *error) {
    const QFileInfo dataRootInfo(path);
    if (dataRootInfo.exists()) {
        if (!dataRootInfo.isDir()) {
            setError(error, QStringLiteral("Configuration path is not a directory: data_root (%1)")
                                .arg(path));
            return false;
        }
        return true;
    }
    const QString parentPath = dataRootInfo.absolutePath();
    if (!QFileInfo::exists(parentPath) || !QFileInfo(parentPath).isDir()) {
        setError(error, QStringLiteral("Configuration data_root parent does not exist: %1 (%2)")
                            .arg(parentPath, path));
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
    const QJsonValue launchModeValue = object.value(QStringLiteral("launch_mode"));
    if (!launchModeValue.isUndefined()) {
        if (!launchModeValue.isString()) {
            setError(error, QStringLiteral("launch_mode must be python_script or packaged_executable"));
            return std::nullopt;
        }
        const QString launchMode = launchModeValue.toString().trimmed().toLower();
        if (launchMode == QStringLiteral("packaged_executable")) {
            config.launchMode = BackendLaunchMode::PackagedExecutable;
        } else if (launchMode != QStringLiteral("python_script")) {
            setError(error, QStringLiteral("launch_mode must be python_script or packaged_executable"));
            return std::nullopt;
        }
    }

    if (!readString(object, QStringLiteral("project_root"), &config.projectRoot, error)
        || !readString(object, QStringLiteral("model_dir"), &config.modelDir, error)
        || !readOptionalString(object, QStringLiteral("python_executable"), &config.pythonExecutable,
                               error)
        || !readOptionalString(object, QStringLiteral("backend_script"), &config.backendScript, error)
        || !readOptionalString(object, QStringLiteral("backend_executable"),
                               &config.backendExecutable, error)
        || !readOptionalString(object, QStringLiteral("paddle_config"), &config.paddleConfigPath,
                               error)
        || !readOptionalString(object, QStringLiteral("library_dir"), &config.libraryDir, error)
        || !readOptionalString(object, QStringLiteral("data_root"), &config.dataRoot, error)
        || !readOptionalString(object, QStringLiteral("model_sha256"), &config.modelSha256,
                               error)
        || !readOptionalString(object, QStringLiteral("compute_device"), &config.computeDevice,
                               error)
        || !readOptionalString(object, QStringLiteral("edition"), &config.edition, error)
        || !readOptionalString(object, QStringLiteral("package_version"), &config.packageVersion,
                               error)) {
        return std::nullopt;
    }

    const QJsonValue ppBackendValue = object.value(QStringLiteral("pp_backend"));
    if (!ppBackendValue.isUndefined()) {
        if (!ppBackendValue.isString()) {
            setError(error, QStringLiteral("pp_backend must be python or native_cpp"));
            return std::nullopt;
        }
        config.ppBackend = ppBackendValue.toString().trimmed().toLower();
    }
    if (config.ppBackend != QStringLiteral("python")
        && config.ppBackend != QStringLiteral("native_cpp")) {
        setError(error, QStringLiteral("pp_backend must be python or native_cpp"));
        return std::nullopt;
    }
    if (!readOptionalStringAllowEmpty(object, QStringLiteral("native_pp_executable"),
                                      &config.nativePpExecutable, error)) {
        return std::nullopt;
    }
    if (config.launchMode == BackendLaunchMode::PackagedExecutable) {
        for (const QString &key : {QStringLiteral("backend_executable"),
                                  QStringLiteral("paddle_config"),
                                  QStringLiteral("data_root"),
                                  QStringLiteral("model_sha256"),
                                  QStringLiteral("compute_device"),
                                  QStringLiteral("edition"),
                                  QStringLiteral("package_version")}) {
            if (!object.contains(key)) {
                setError(error, QStringLiteral("Missing configuration field: %1").arg(key));
                return std::nullopt;
            }
        }
    }
    const QDir configDir = QFileInfo(path).absoluteDir();
    const auto resolvedPath = [&configDir](const QString &value) {
        if (value.isEmpty()) {
            return QString();
        }
        return QDir::cleanPath(QFileInfo(value).isAbsolute()
                                   ? value
                                   : configDir.absoluteFilePath(value));
    };
    config.pythonExecutable = resolvedPath(config.pythonExecutable);
    config.backendScript = resolvedPath(config.backendScript);
    config.backendExecutable = resolvedPath(config.backendExecutable);
    config.projectRoot = resolvedPath(config.projectRoot);
    config.paddleConfigPath = resolvedPath(config.paddleConfigPath);
    config.modelDir = resolvedPath(config.modelDir);
    config.libraryDir = resolvedPath(config.libraryDir);
    config.dataRoot = resolvedPath(config.dataRoot);
    config.nativePpExecutable = resolvedPath(config.nativePpExecutable);
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
    if (config.ppBackend == QStringLiteral("native_cpp")
        && (config.computeDevice != QStringLiteral("cpu")
            || config.inferenceMode != QStringLiteral("fast_geometry"))) {
        setError(error, QStringLiteral(
            "native_cpp requires compute_device=cpu and inference_mode=fast_geometry"));
        return std::nullopt;
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
    if (config.ppBackend == QStringLiteral("native_cpp")) {
        if (config.nativePpExecutable.isEmpty()) {
            setError(error, QStringLiteral(
                "native_pp_executable is required when pp_backend is native_cpp"));
            return std::nullopt;
        }
        if (!requireNativeExecutable(config.nativePpExecutable, error)) {
            return std::nullopt;
        }
    }
    if (config.launchMode == BackendLaunchMode::PythonScript) {
        if (config.pythonExecutable.isEmpty() || config.backendScript.isEmpty()
            || config.libraryDir.isEmpty()
            || !requireExistingFile(config.pythonExecutable, QStringLiteral("python_executable"), error)
            || !requireExistingFile(config.backendScript, QStringLiteral("backend_script"), error)
            || !requireExistingDirectory(config.projectRoot, QStringLiteral("project_root"), error)
            || !requireExistingDirectory(config.modelDir, QStringLiteral("model_dir"), error)) {
            if (error != nullptr && error->isEmpty()) {
                setError(error, QStringLiteral("Missing required development configuration field"));
            }
            return std::nullopt;
        }
    } else if (config.backendExecutable.isEmpty() || config.paddleConfigPath.isEmpty()
               || config.dataRoot.isEmpty() || config.modelSha256.isEmpty()
               || config.computeDevice.isEmpty() || config.edition.isEmpty()
               || config.packageVersion.isEmpty()
               || !requireExistingFile(config.backendExecutable,
                                       QStringLiteral("backend_executable"), error)
               || !requireExistingFile(config.paddleConfigPath, QStringLiteral("paddle_config"), error)
               || !requireExistingDirectory(config.projectRoot, QStringLiteral("project_root"), error)
               || !requireExistingDirectory(config.modelDir, QStringLiteral("model_dir"), error)
               || !requireDataRootParent(config.dataRoot, error)) {
        return std::nullopt;
    }
    if (config.launchMode == BackendLaunchMode::PackagedExecutable) {
        static const QRegularExpression lowercaseSha256(
            QStringLiteral("^[0-9a-f]{64}$")
        );
        if (!lowercaseSha256.match(config.modelSha256).hasMatch()) {
            setError(error, QStringLiteral("model_sha256 must be a lowercase 64-character SHA-256"));
            return std::nullopt;
        }
        if (config.computeDevice != config.edition) {
            setError(error, QStringLiteral("edition and compute_device must match"));
            return std::nullopt;
        }
        if (config.inferenceMode != QStringLiteral("fast_geometry")) {
            setError(error, QStringLiteral("packaged inference_mode must be fast_geometry"));
            return std::nullopt;
        }
    }
    config.port = static_cast<quint16>(port);
    config.startupTimeoutMs = startupTimeout;
    config.requestTimeoutMs = requestTimeout;
    return config;
}
