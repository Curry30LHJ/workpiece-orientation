#include <QtTest>
#include <QApplication>
#include <QLabel>
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
};

QTEST_MAIN(TestAppFoundation)
#include "test_appfoundation.moc"
