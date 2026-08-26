#include <QApplication>
#include <QCoreApplication>

#include "appconfig.h"
#include "apptheme.h"
#include "backendclient.h"
#include "backendprocessmanager.h"
#include "mainwindow.h"

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
        MainWindow window;
        window.setBackendError(configError);
        window.show();
        return application.exec();
    }
    BackendClient client;
    BackendProcessManager manager(*config, &client, nullptr, &application);
    MainWindow window(&client, &manager);
    manager.start();
    window.show();
    return application.exec();
}
