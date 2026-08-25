#pragma once

#include <QWidget>

class QLabel;
class QProgressBar;
class QPushButton;

class TaskStatusWidget : public QWidget {
    Q_OBJECT
public:
    enum class MessageKind { Neutral, Success, Warning, Error };
    Q_ENUM(MessageKind)

    explicit TaskStatusWidget(QWidget *parent = nullptr);
    void setIdle();
    void setRunning(const QString &title, const QString &phase,
                    int completed, int total, qint64 elapsedMs);
    void setMessage(MessageKind kind, const QString &text,
                    const QString &actionText = QString());

signals:
    void actionRequested();

private:
    QLabel *titleLabel_;
    QLabel *detailLabel_;
    QProgressBar *progressBar_;
    QPushButton *actionButton_;
};
