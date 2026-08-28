#include <QApplication>
#include <QCoreApplication>

#include "appconfig.h"
#include "apptheme.h"
#include "backendclient.h"
#include "backendprocessmanager.h"
#include "mainwindow.h"
#include "startupsmokecontroller.h"

int main(int argc, char *argv[]) {
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    QApplication application(argc, argv);
    AppTheme::apply(&application);
    application.setApplicationName(QStringLiteral("工件正反面检测"));
    const QString configPath = QCoreApplication::applicationDirPath() + QStringLiteral("/app_config.json");
    QString configError;
    const std::optional<AppConfig> config = AppConfig::load(configPath, &configError);
    if (!config.has_value()) {
        if (application.arguments().contains(QStringLiteral("--package-smoke-test"))) {
            return 2;
        }
        MainWindow window;
        window.setBackendError(configError);
        window.show();
        return application.exec();
    }
    BackendClient client;
    BackendProcessManager manager(*config, &client, nullptr, &application);
    if (application.arguments().contains(QStringLiteral("--package-smoke-test"))) {
        StartupSmokeController controller(&manager, config->startupTimeoutMs + 5000, &application);
        QObject::connect(&controller, &StartupSmokeController::finished,
                         &application, [&application](int code) { application.exit(code); });
        manager.start();
        return application.exec();
    }
    MainWindow window(&client, &manager);
    manager.start();
    window.show();
    return application.exec();
}
