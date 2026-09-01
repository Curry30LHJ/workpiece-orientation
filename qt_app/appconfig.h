#pragma once

#include <QHostAddress>
#include <QString>

#include <optional>

enum class BackendLaunchMode { PythonScript, PackagedExecutable };

struct AppConfig {
    BackendLaunchMode launchMode = BackendLaunchMode::PythonScript;
    QString pythonExecutable;
    QString backendScript;
    QString backendExecutable;
    QString projectRoot;
    QString paddleConfigPath;
    QString modelDir;
    QString libraryDir;
    QString dataRoot;
    QString modelSha256;
    // PP-ShiTu implementation selected by the TCP backend.  Python remains
    // the compatibility default for existing configuration files.
    QString ppBackend = QStringLiteral("python");
    // Required only when ppBackend is native_cpp; resolved to an absolute
    // regular executable path by AppConfig::load().
    QString nativePpExecutable;
    QString computeDevice = QStringLiteral("gpu");
    QString edition = QStringLiteral("dev");
    QString packageVersion = QStringLiteral("dev");
    QString localSearchMode = QStringLiteral("adaptive");
    QString inferenceMode = QStringLiteral("legacy");
    QHostAddress host;
    quint16 port = 37651;
    int startupTimeoutMs = 600000;
    int requestTimeoutMs = 120000;

    static std::optional<AppConfig> load(const QString &path, QString *error = nullptr);
};
