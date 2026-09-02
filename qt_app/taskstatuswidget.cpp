#include "taskstatuswidget.h"

#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QSizePolicy>
#include <QStyle>
#include <QVariant>

namespace {

QString messageKindName(TaskStatusWidget::MessageKind kind) {
    switch (kind) {
    case TaskStatusWidget::MessageKind::Success:
        return QStringLiteral("success");
    case TaskStatusWidget::MessageKind::Warning:
        return QStringLiteral("warning");
    case TaskStatusWidget::MessageKind::Error:
        return QStringLiteral("error");
    case TaskStatusWidget::MessageKind::Neutral:
    default:
        return QStringLiteral("neutral");
    }
}

void setMessageKind(TaskStatusWidget *widget, TaskStatusWidget::MessageKind kind) {
    widget->setProperty("messageKind", messageKindName(kind));
    widget->style()->unpolish(widget);
    widget->style()->polish(widget);
    for (QLabel *label : widget->findChildren<QLabel *>()) {
        label->style()->unpolish(label);
        label->style()->polish(label);
        label->update();
    }
    widget->update();
}

} // namespace

TaskStatusWidget::TaskStatusWidget(QWidget *parent)
    : QWidget(parent),
      iconLabel_(new QLabel(this)),
      titleLabel_(new QLabel(this)),
      detailLabel_(new QLabel(this)),
      progressBar_(new QProgressBar(this)),
      actionButton_(new QPushButton(this)) {
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    iconLabel_->setObjectName(QStringLiteral("globalTaskIconLabel"));
    iconLabel_->setFixedSize(24, 24);
    iconLabel_->setAlignment(Qt::AlignCenter);
    titleLabel_->setObjectName(QStringLiteral("globalTaskTitleLabel"));
    detailLabel_->setObjectName(QStringLiteral("globalTaskDetailLabel"));
    progressBar_->setObjectName(QStringLiteral("globalTaskProgressBar"));
    actionButton_->setObjectName(QStringLiteral("globalTaskActionButton"));

    QHBoxLayout *layout = new QHBoxLayout(this);
    layout->setContentsMargins(16, 8, 16, 8);
    layout->setSpacing(8);
    layout->addWidget(iconLabel_);
    layout->addWidget(titleLabel_);
    layout->addWidget(detailLabel_, 1);
    layout->addWidget(progressBar_);
    layout->addWidget(actionButton_);

    connect(actionButton_, &QPushButton::clicked, this, &TaskStatusWidget::actionRequested);
    actionButton_->setProperty("role", QStringLiteral("secondary"));
    setProperty("panel", true);
    setIdle();
}

void TaskStatusWidget::setIdle() {
    titleLabel_->setText(QStringLiteral("当前无执行任务"));
    detailLabel_->clear();
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    progressBar_->hide();
    iconLabel_->clear();
    actionButton_->setText(QString());
    actionButton_->hide();
    setMessageKind(this, MessageKind::Neutral);
}

void TaskStatusWidget::setRunning(const QString &title, const QString &phase,
                                  int completed, int total, qint64 elapsedMs) {
    titleLabel_->setText(title);
    detailLabel_->setText(QStringLiteral("%1 · %2/%3 · 已用 %4 秒")
                              .arg(phase)
                              .arg(completed)
                              .arg(total)
                              .arg(QString::number(elapsedMs / 1000.0, 'f', 1)));
    progressBar_->setRange(0, qMax(1, total));
    progressBar_->setValue(completed);
    progressBar_->show();
    iconLabel_->clear();
    actionButton_->hide();
    setMessageKind(this, MessageKind::Neutral);
}

void TaskStatusWidget::setMessage(MessageKind kind, const QString &text,
                                  const QString &actionText) {
    QString effectiveText = text.trimmed();
    if (effectiveText.isEmpty()) {
        effectiveText = kind == MessageKind::Error
            ? QStringLiteral("操作未完成，请查看详情后重试")
            : (kind == MessageKind::Warning
                   ? QStringLiteral("需要检查后再继续")
                   : QStringLiteral("状态已更新"));
    }
    QString title = QStringLiteral("状态提示");
    QString iconPath;
    if (kind == MessageKind::Success) {
        title = QStringLiteral("操作完成");
        iconPath = QStringLiteral(":/icons/status-success.svg");
    } else if (kind == MessageKind::Warning) {
        title = QStringLiteral("需要注意");
        iconPath = QStringLiteral(":/icons/status-warning.svg");
    } else if (kind == MessageKind::Error) {
        title = QStringLiteral("操作失败");
        iconPath = QStringLiteral(":/icons/status-error.svg");
    }
    titleLabel_->setText(title);
    detailLabel_->setText(effectiveText);
    progressBar_->hide();
    if (iconPath.isEmpty()) {
        iconLabel_->clear();
    } else {
        iconLabel_->setPixmap(QIcon(iconPath).pixmap(20, 20));
    }
    actionButton_->setText(actionText);
    actionButton_->setVisible(!actionText.isEmpty());
    setMessageKind(this, kind);
}
