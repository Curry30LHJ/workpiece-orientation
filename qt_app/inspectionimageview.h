#pragma once

#include <QGraphicsView>

class QDragEnterEvent;
class QDropEvent;
class QGraphicsPixmapItem;
class QGraphicsScene;
class QMimeData;
class QWheelEvent;

class InspectionImageView : public QGraphicsView {
    Q_OBJECT

public:
    explicit InspectionImageView(QWidget *parent = nullptr);
    bool setImagePath(const QString &path);
    QString imagePath() const;
    qreal zoomFactor() const;

public slots:
    void resetView();

signals:
    void imageDropped(const QString &path);

protected:
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    QString droppedImagePath(const QMimeData *mimeData) const;

    QGraphicsScene *scene_ = nullptr;
    QGraphicsPixmapItem *pixmapItem_ = nullptr;
    QString imagePath_;
    qreal zoomFactor_ = 1.0;
};
