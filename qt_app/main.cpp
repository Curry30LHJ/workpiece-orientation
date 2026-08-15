#include <QApplication>

#include "mainwindow.h"

int main(int argc, char *argv[]) {
    QApplication application(argc, argv);
    application.setApplicationName(QStringLiteral("工件正反面检测"));
    MainWindow window;
    window.show();
    return application.exec();
}
