#include "appheader.h"

#include <QButtonGroup>
#include <QComboBox>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QVBoxLayout>

namespace {

QString backendStateText(BackendUiState state) {
    switch (state) {
    case BackendUiState::Loading:
        return QStringLiteral("后端：模型加载中");
    case BackendUiState::Ready:
        return QStringLiteral("后端：已连接");
    case BackendUiState::Busy:
        return QStringLiteral("后端：处理中");
    case BackendUiState::Error:
        return QStringLiteral("后端：不可用");
    case BackendUiState::Disconnected:
    default:
        return QStringLiteral("后端：未连接");
    }
}

} // namespace

AppHeader::AppHeader(QWidget *parent)
    : QWidget(parent),
      inspectionButton_(new QPushButton(QStringLiteral("检测工作台"), this)),
      workpieceLibraryButton_(new QPushButton(QStringLiteral("工件库"), this)),
      geometryRulesButton_(new QPushButton(QStringLiteral("几何规则"), this)),
      workpieceComboBox_(new QComboBox(this)),
      backendStatusLabel_(new QLabel(this)),
      backendDetailsButton_(new QPushButton(QStringLiteral("详情"), this)),
      backendDetailsPanel_(new QFrame(this)),
      connectionDetailLabel_(new QLabel(backendDetailsPanel_)),
      modelDetailLabel_(new QLabel(backendDetailsPanel_)),
      currentTaskLabel_(new QLabel(backendDetailsPanel_)),
      recentErrorLabel_(new QLabel(backendDetailsPanel_)),
      restartBackendButton_(new QPushButton(QStringLiteral("重启后端"), backendDetailsPanel_)) {
    setObjectName(QStringLiteral("appHeader"));
    inspectionButton_->setObjectName(QStringLiteral("inspectionNavButton"));
    workpieceLibraryButton_->setObjectName(QStringLiteral("workpieceLibraryNavButton"));
    geometryRulesButton_->setObjectName(QStringLiteral("geometryRulesNavButton"));
    workpieceComboBox_->setObjectName(QStringLiteral("workpieceComboBox"));
    backendStatusLabel_->setObjectName(QStringLiteral("backendStatusLabel"));
    backendDetailsButton_->setObjectName(QStringLiteral("backendDetailsButton"));
    backendDetailsPanel_->setObjectName(QStringLiteral("backendDetailsPanel"));
    connectionDetailLabel_->setObjectName(QStringLiteral("backendConnectionDetailLabel"));
    modelDetailLabel_->setObjectName(QStringLiteral("backendModelDetailLabel"));
    currentTaskLabel_->setObjectName(QStringLiteral("backendCurrentTaskLabel"));
    recentErrorLabel_->setObjectName(QStringLiteral("backendRecentErrorLabel"));
    restartBackendButton_->setObjectName(QStringLiteral("restartBackendButton"));

    QButtonGroup *navigationGroup = new QButtonGroup(this);
    navigationGroup->setExclusive(true);
    for (QPushButton *button : {inspectionButton_, workpieceLibraryButton_, geometryRulesButton_}) {
        button->setCheckable(true);
        navigationGroup->addButton(button);
    }

    QHBoxLayout *headerLayout = new QHBoxLayout;
    headerLayout->setContentsMargins(0, 0, 0, 0);
    headerLayout->addWidget(inspectionButton_);
    headerLayout->addWidget(workpieceLibraryButton_);
    headerLayout->addWidget(geometryRulesButton_);
    headerLayout->addSpacing(16);
    headerLayout->addWidget(new QLabel(QStringLiteral("当前检测工件："), this));
    headerLayout->addWidget(workpieceComboBox_, 1);
    headerLayout->addSpacing(16);
    headerLayout->addWidget(backendStatusLabel_);
    headerLayout->addWidget(backendDetailsButton_);

    QVBoxLayout *detailsLayout = new QVBoxLayout(backendDetailsPanel_);
    detailsLayout->setContentsMargins(12, 8, 12, 8);
    detailsLayout->addWidget(connectionDetailLabel_);
    detailsLayout->addWidget(modelDetailLabel_);
    detailsLayout->addWidget(currentTaskLabel_);
    detailsLayout->addWidget(recentErrorLabel_);
    detailsLayout->addWidget(restartBackendButton_, 0, Qt::AlignRight);

    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addLayout(headerLayout);
    layout->addWidget(backendDetailsPanel_);
    backendDetailsPanel_->hide();

    connect(inspectionButton_, &QPushButton::clicked, this,
            [this]() { emit pageRequested(AppPage::Inspection); });
    connect(workpieceLibraryButton_, &QPushButton::clicked, this,
            [this]() { emit pageRequested(AppPage::WorkpieceLibrary); });
    connect(geometryRulesButton_, &QPushButton::clicked, this,
            [this]() { emit pageRequested(AppPage::GeometryRules); });
    connect(workpieceComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { emit currentWorkpieceRequested(currentWorkpieceId()); });
    connect(backendDetailsButton_, &QPushButton::clicked, this, [this]() {
        backendDetailsPanel_->setVisible(!backendDetailsPanel_->isVisible());
        emit backendDetailsRequested();
    });
    connect(restartBackendButton_, &QPushButton::clicked,
            this, &AppHeader::restartBackendRequested);

    setCurrentPage(AppPage::Inspection);
    setBackendState(BackendUiState::Disconnected, QString());
    restartBackendButton_->hide();
}

void AppHeader::setCurrentPage(AppPage page) {
    inspectionButton_->setChecked(page == AppPage::Inspection);
    workpieceLibraryButton_->setChecked(page == AppPage::WorkpieceLibrary);
    geometryRulesButton_->setChecked(page == AppPage::GeometryRules);
}

void AppHeader::setWorkpieces(const QList<QPair<QString, QString>> &items,
                               const QString &currentWorkpieceId) {
    const QSignalBlocker blocker(workpieceComboBox_);
    workpieceComboBox_->clear();
    for (const auto &item : items) {
        workpieceComboBox_->addItem(item.second, item.first);
    }
    const int index = workpieceComboBox_->findData(currentWorkpieceId);
    workpieceComboBox_->setCurrentIndex(index >= 0 ? index : (items.isEmpty() ? -1 : 0));
}

void AppHeader::setCurrentWorkpieceId(const QString &workpieceId) {
    const int index = workpieceComboBox_->findData(workpieceId);
    if (index < 0) {
        return;
    }
    const QSignalBlocker blocker(workpieceComboBox_);
    workpieceComboBox_->setCurrentIndex(index);
}

QString AppHeader::currentWorkpieceId() const {
    return workpieceComboBox_->currentData().toString();
}

QString AppHeader::currentWorkpieceName() const {
    return workpieceComboBox_->currentText();
}

void AppHeader::setBackendState(BackendUiState state, const QString &detail) {
    const QString baseText = backendStateText(state);
    backendStatusLabel_->setText(detail.isEmpty()
                                     ? baseText
                                     : QStringLiteral("%1 · %2").arg(baseText, detail));
}

void AppHeader::setBackendDetails(const BackendStatusDetails &details) {
    connectionDetailLabel_->setText(
        QStringLiteral("连接：%1").arg(details.connectionDetail));
    modelDetailLabel_->setText(QStringLiteral("模型：%1").arg(details.modelDetail));
    currentTaskLabel_->setText(QStringLiteral("当前任务：%1").arg(details.currentTask));
    recentErrorLabel_->setText(QStringLiteral("最近错误：%1").arg(details.recentError));
    restartBackendButton_->setEnabled(details.canRestart);
    restartBackendButton_->setVisible(details.canRestart);

    QString summaryDetail;
    if (details.state == BackendUiState::Error) {
        summaryDetail = details.recentError;
    } else if (details.state == BackendUiState::Loading) {
        summaryDetail = details.modelDetail;
    } else {
        summaryDetail = details.connectionDetail;
    }
    setBackendState(details.state, summaryDetail);
}
