#include "taskstatuswidget.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
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
      titleLabel_(new QLabel(this)),
      detailLabel_(new QLabel(this)),
      progressBar_(new QProgressBar(this)),
      actionButton_(new QPushButton(this)) {
    titleLabel_->setObjectName(QStringLiteral("globalTaskTitleLabel"));
    detailLabel_->setObjectName(QStringLiteral("globalTaskDetailLabel"));
    progressBar_->setObjectName(QStringLiteral("globalTaskProgressBar"));
    actionButton_->setObjectName(QStringLiteral("globalTaskActionButton"));

    QHBoxLayout *layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(titleLabel_);
    layout->addWidget(detailLabel_, 1);
    layout->addWidget(progressBar_);
    layout->addWidget(actionButton_);

    connect(actionButton_, &QPushButton::clicked, this, &TaskStatusWidget::actionRequested);
    setIdle();
}

void TaskStatusWidget::setIdle() {
    titleLabel_->setText(QStringLiteral("当前无执行任务"));
    detailLabel_->clear();
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
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
    actionButton_->hide();
    setMessageKind(this, MessageKind::Neutral);
}

void TaskStatusWidget::setMessage(MessageKind kind, const QString &text,
                                  const QString &actionText) {
    detailLabel_->setText(text);
    actionButton_->setText(actionText);
    actionButton_->setVisible(!actionText.isEmpty());
    setMessageKind(this, kind);
}
