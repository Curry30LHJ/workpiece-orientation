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
#include <QToolButton>
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

    void backendLifecycleLabelsDistinguishStartupLoadingAndRecovery() {
        AppHeader header;
        auto *label = header.findChild<QLabel *>(QStringLiteral("backendStatusLabel"));
        QVERIFY(label != nullptr);

        header.setBackendState(BackendUiState::Starting,
                               QStringLiteral("正在启动服务"));
        QVERIFY(label->text().contains(QStringLiteral("正在启动")));
        QVERIFY(!label->text().contains(QStringLiteral("模型加载中")));

        header.setBackendState(BackendUiState::Loading,
                               QStringLiteral("正在加载权重"));
        QVERIFY(label->text().contains(QStringLiteral("模型加载中")));

        header.setBackendState(BackendUiState::Recovering,
                               QStringLiteral("连接暂时中断"));
        QVERIFY(label->text().contains(QStringLiteral("正在重连")));
        QCOMPARE(label->property("messageKind").toString(),
                 QStringLiteral("warning"));
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

    void navigationAndStatusUseSemanticIcons() {
        AppHeader header;
        for (const QString &name : {QStringLiteral("inspectionNavButton"),
                                    QStringLiteral("workpieceLibraryNavButton"),
                                    QStringLiteral("geometryRulesNavButton")}) {
            auto *button = header.findChild<QPushButton *>(name);
            QVERIFY(button != nullptr);
            QVERIFY2(!button->icon().isNull(), qPrintable(name));
        }

        TaskStatusWidget status;
        auto *icon = status.findChild<QLabel *>(QStringLiteral("globalTaskIconLabel"));
        QVERIFY(icon != nullptr);
        status.setMessage(TaskStatusWidget::MessageKind::Warning,
                          QStringLiteral("请检查模板"));
        QVERIFY(icon->pixmap() != nullptr && !icon->pixmap()->isNull());
    }

    void warningAndErrorMessagesNeverBecomeTextless() {
        TaskStatusWidget status;
        auto *detail = status.findChild<QLabel *>(QStringLiteral("globalTaskDetailLabel"));
        QVERIFY(detail != nullptr);

        status.setMessage(TaskStatusWidget::MessageKind::Warning, QString());
        QVERIFY(!detail->text().trimmed().isEmpty());
        status.setMessage(TaskStatusWidget::MessageKind::Error, QString());
        QVERIFY(!detail->text().trimmed().isEmpty());
    }

    void longBackendStatusKeepsFullTextInTooltip() {
        AppHeader header;
        header.resize(760, 160);
        const QString detail = QStringLiteral(
            "连接到 E:/一个非常长的中文目录/模型目录/推理服务，并等待模型初始化完成");
        header.setBackendState(BackendUiState::Loading, detail);
        header.show();
        QCoreApplication::processEvents();

        auto *label = header.findChild<QLabel *>(QStringLiteral("backendStatusLabel"));
        QVERIFY(label != nullptr);
        QVERIFY(label->toolTip().contains(detail));
        QVERIFY(label->text().size() < label->toolTip().size());
    }

    void headerTabOrderReachesNavigationAndTargetSelection() {
        AppHeader header;
        auto *inspection = header.findChild<QPushButton *>(
            QStringLiteral("inspectionNavButton"));
        auto *library = header.findChild<QPushButton *>(
            QStringLiteral("workpieceLibraryNavButton"));
        auto *geometry = header.findChild<QPushButton *>(
            QStringLiteral("geometryRulesNavButton"));
        auto *workpiece = header.findChild<QComboBox *>(QStringLiteral("workpieceComboBox"));
        QVERIFY(inspection != nullptr);
        QVERIFY(library != nullptr);
        QVERIFY(geometry != nullptr);
        QVERIFY(workpiece != nullptr);
        QCOMPARE(inspection->nextInFocusChain(), library);
        QCOMPARE(library->nextInFocusChain(), geometry);
        QCOMPARE(geometry->nextInFocusChain(), workpiece);
    }

    void shellChromeKeepsPageContentAsTheVerticalExpansionTarget() {
        AppHeader header;
        TaskStatusWidget status;
        QCOMPARE(header.sizePolicy().verticalPolicy(), QSizePolicy::Maximum);
        QCOMPARE(status.sizePolicy().verticalPolicy(), QSizePolicy::Maximum);
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
