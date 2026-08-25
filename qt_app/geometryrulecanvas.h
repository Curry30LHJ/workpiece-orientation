#pragma once

#include <QImage>
#include <QJsonObject>
#include <QPointF>
#include <QRectF>
#include <QWidget>

class QPainter;
class QPen;
class QWheelEvent;
class QKeyEvent;

class GeometryRuleCanvas : public QWidget {
    Q_OBJECT

public:
    enum Tool {
        None,
        Circle,
        Ellipse,
        RotatedRectangle,
    };
    Q_ENUM(Tool)

    explicit GeometryRuleCanvas(QWidget *parent = nullptr);

    void setImage(const QImage &image);
    void setTool(Tool tool);
    void setCoarseShape(const QJsonObject &shape);
    void setFitOverlay(const QJsonObject &fit, const QImage &maskOverlay = QImage());
    void setRotationDegrees(qreal degrees);
    void resetView();
    void cancelGesture();
    void setNumericShape(const QJsonObject &shape);

    QImage image() const { return image_; }
    QJsonObject coarseShape() const { return coarseShape_; }
    QJsonObject fitShape() const { return fitShape_; }
    QRectF imageTarget() const;

signals:
    void shapeChanged(const QJsonObject &shape);
    void shapeCommitted(const QJsonObject &shape);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    QPointF imagePoint(const QPoint &point) const;
    QJsonObject shapeFromDrag(const QPointF &start, const QPointF &end) const;
    void updateShapeFromDrag(const QPointF &end);
    void drawShape(QPainter *painter, const QJsonObject &shape, const QPen &pen) const;
    void drawHandles(QPainter *painter, const QJsonObject &shape) const;

    QImage image_;
    QImage maskOverlay_;
    QJsonObject coarseShape_;
    QJsonObject fitShape_;
    Tool tool_ = None;
    qreal rotationDegrees_ = 0.0;
    QPointF dragStart_;
    bool dragging_ = false;
    bool panning_ = false;
    QPoint lastPanPoint_;
    qreal zoom_ = 1.0;
    QPointF pan_;
};
