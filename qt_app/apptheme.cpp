#include "apptheme.h"

#include <QApplication>
#include <QFile>

namespace AppTheme {

QString styleSheet() {
    static const QString sheet = [] {
        QFile file(QStringLiteral(":/theme/theme.qss"));
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return QString();
        }
        return QString::fromUtf8(file.readAll());
    }();
    return sheet;
}

void apply(QApplication *application) {
    if (application) {
        application->setStyleSheet(styleSheet());
    }
}

} // namespace AppTheme
