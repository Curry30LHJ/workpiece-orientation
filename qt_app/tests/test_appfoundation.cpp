#include <QtTest>
#include <QApplication>
#include <QColor>
#include <QFile>
#include <QLabel>
#include <QPalette>
#include <QProgressBar>
#include "apptheme.h"
#include "taskstatuswidget.h"

class TestAppFoundation : public QObject {
    Q_OBJECT
private slots:
    void themeContainsApprovedTokens() {
        const QString qss = AppTheme::styleSheet();
        QVERIFY(qss.contains(QStringLiteral("#F4F6F8"), Qt::CaseInsensitive));
        QVERIFY(qss.contains(QStringLiteral("#2563EB"), Qt::CaseInsensitive));
        QVERIFY(qss.contains(QStringLiteral("#15803D"), Qt::CaseInsensitive));
        QVERIFY(qss.contains(QStringLiteral("#C9362B"), Qt::CaseInsensitive));
    }

    void themeResourceCanBeReadAtRegisteredPath() {
        QFile themeFile(QStringLiteral(":/theme/theme.qss"));

        QVERIFY(themeFile.open(QIODevice::ReadOnly | QIODevice::Text));
        QVERIFY(!themeFile.readAll().isEmpty());
    }

    void taskStatusKeepsActionableState() {
        TaskStatusWidget widget;
        widget.setRunning(QStringLiteral("建立工件库"),
                          QStringLiteral("提取特征"), 7, 20, 1250);
        QCOMPARE(widget.findChild<QProgressBar *>(
                     QStringLiteral("globalTaskProgressBar"))->value(), 7);
        QVERIFY(widget.findChild<QLabel *>(
                    QStringLiteral("globalTaskTitleLabel"))->text()
                    .contains(QStringLiteral("建立工件库")));
        widget.setMessage(TaskStatusWidget::MessageKind::Error,
                          QStringLiteral("后端断开"), QStringLiteral("重试"));
        QCOMPARE(widget.property("messageKind").toString(),
                 QStringLiteral("error"));
    }

    void taskStatusMessageColors_data() {
        QTest::addColumn<TaskStatusWidget::MessageKind>("kind");
        QTest::addColumn<QColor>("expectedColor");
        QTest::newRow("success") << TaskStatusWidget::MessageKind::Success
                                  << QColor(QStringLiteral("#15803D"));
        QTest::newRow("warning") << TaskStatusWidget::MessageKind::Warning
                                  << QColor(QStringLiteral("#B7791F"));
        QTest::newRow("error") << TaskStatusWidget::MessageKind::Error
                                << QColor(QStringLiteral("#C9362B"));
    }

    void taskStatusMessageColors() {
        QFETCH(TaskStatusWidget::MessageKind, kind);
        QFETCH(QColor, expectedColor);
        AppTheme::apply(qApp);
        TaskStatusWidget widget;
        widget.show();
        QLabel *title = widget.findChild<QLabel *>(QStringLiteral("globalTaskTitleLabel"));
        QLabel *detail = widget.findChild<QLabel *>(QStringLiteral("globalTaskDetailLabel"));
        QVERIFY(title != nullptr);
        QVERIFY(detail != nullptr);

        widget.setMessage(kind, QStringLiteral("状态文案"));
        QCoreApplication::processEvents();

        QCOMPARE(title->palette().color(QPalette::WindowText), expectedColor);
        QCOMPARE(detail->palette().color(QPalette::WindowText), expectedColor);
    }
};

QTEST_MAIN(TestAppFoundation)
#include "test_appfoundation.moc"
