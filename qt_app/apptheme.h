#pragma once

#include <QString>

class QApplication;

namespace AppTheme {
QString styleSheet();
void apply(QApplication *application);
}
