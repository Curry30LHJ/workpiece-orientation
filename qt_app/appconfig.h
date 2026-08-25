#pragma once

#include <QHostAddress>
#include <QString>

#include <optional>

struct AppConfig {
    QString pythonExecutable;
    QString backendScript;
    QString projectRoot;
    QString modelDir;
    QString libraryDir;
    QString localSearchMode = QStringLiteral("adaptive");
    QHostAddress host;
    quint16 port = 37651;
    int startupTimeoutMs = 120000;
    int requestTimeoutMs = 120000;

    static std::optional<AppConfig> load(const QString &path, QString *error = nullptr);
};
