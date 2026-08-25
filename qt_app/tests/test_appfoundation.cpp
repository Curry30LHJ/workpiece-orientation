#include <QtTest>
#include <QApplication>
#include <QColor>
#include <QComboBox>
#include <QFile>
#include <QFrame>
#include <QLabel>
#include <QPalette>
#include <QProgressBar>
#include <QPushButton>
#include <QSignalSpy>
#include "apptheme.h"
#include "appheader.h"
#include "taskstatuswidget.h"

class TestAppFoundation : public QObject {
    Q_OBJECT
private slots:
    void headerDoesNotChangeDetectionTargetWhileBrowsing() {
        AppHeader header;
        QSignalSpy spy(&header, &AppHeader::currentWorkpieceRequested);
        header.setWorkpieces({{QStringLiteral("m1"), QStringLiteral("M1")},
                              {QStringLiteral("m2"), QStringLiteral("M2")}},
                             QStringLiteral("m1"));
        header.setCurrentWorkpieceId(QStringLiteral("m1"));
        QCOMPARE(header.currentWorkpieceId(), QStringLiteral("m1"));
        QCOMPARE(spy.count(), 0);
    }

    void backendDetailsPanelShowsRecoveryContext() {
        AppHeader header;
        header.resize(900, 300);
        BackendStatusDetails details;
        details.state = BackendUiState::Error;
        details.connectionDetail = QStringLiteral("连接已断开");
        details.modelDetail = QStringLiteral("模型未加载");
        details.currentTask = QStringLiteral("批量检测 4/10");
        details.recentError = QStringLiteral("连接超时");
        details.canRestart = true;
        header.setBackendDetails(details);
        header.show();

        auto *panel = header.findChild<QFrame *>(QStringLiteral("backendDetailsPanel"));
        auto *detailsButton = header.findChild<QPushButton *>(
            QStringLiteral("backendDetailsButton"));
        auto *restartButton = header.findChild<QPushButton *>(
            QStringLiteral("restartBackendButton"));
        QVERIFY(panel != nullptr);
        QVERIFY(detailsButton != nullptr);
        QVERIFY(restartButton != nullptr);
        QVERIFY(!panel->isVisible());

        QTest::mouseClick(detailsButton, Qt::LeftButton);
        QVERIFY(panel->isVisible());
        const QList<QLabel *> labels = panel->findChildren<QLabel *>();
        QStringList renderedText;
        for (const QLabel *label : labels) {
            renderedText.append(label->text());
        }
        const QString combined = renderedText.join(QLatin1Char('\n'));
        QVERIFY(combined.contains(QStringLiteral("连接已断开")));
        QVERIFY(combined.contains(QStringLiteral("模型未加载")));
        QVERIFY(combined.contains(QStringLiteral("批量检测 4/10")));
        QVERIFY(combined.contains(QStringLiteral("连接超时")));
        QVERIFY(restartButton->isVisible());

        details.canRestart = false;
        header.setBackendDetails(details);
        QVERIFY(!restartButton->isVisible());
    }

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
