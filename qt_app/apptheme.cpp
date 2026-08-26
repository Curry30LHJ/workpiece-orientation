#include "apptheme.h"

#include <QApplication>
#include <QColor>
#include <QFile>
#include <QPalette>

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
    if (application == nullptr || application->property("appThemeApplied").toBool()) return;
    QPalette palette = application->palette();
    palette.setColor(QPalette::Window, QColor(QStringLiteral("#F4F6F8")));
    palette.setColor(QPalette::WindowText, QColor(QStringLiteral("#17212B")));
    palette.setColor(QPalette::Base, QColor(QStringLiteral("#FFFFFF")));
    palette.setColor(QPalette::Text, QColor(QStringLiteral("#17212B")));
    palette.setColor(QPalette::Highlight, QColor(QStringLiteral("#DCE9FF")));
    palette.setColor(QPalette::HighlightedText, QColor(QStringLiteral("#17212B")));
    application->setPalette(palette);
    application->setStyleSheet(styleSheet());
    application->setProperty("appThemeApplied", true);
}

} // namespace AppTheme
