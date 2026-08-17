#include "geometryrulecanvas.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPen>

#include <QtMath>

namespace {
QPointF jsonPoint(const QJsonObject &shape, const char *x, const char *y) {
    return QPointF(shape.value(QLatin1String(x)).toDouble(), shape.value(QLatin1String(y)).toDouble());
}

QImage orangeMask(const QImage &mask) {
    if (mask.isNull()) {
        return {};
    }
    const QImage gray = mask.convertToFormat(QImage::Format_Grayscale8);
    QImage result(gray.size(), QImage::Format_ARGB32_Premultiplied);
    for (int y = 0; y < gray.height(); ++y) {
        const uchar *source = gray.constScanLine(y);
        QRgb *target = reinterpret_cast<QRgb *>(result.scanLine(y));
        for (int x = 0; x < gray.width(); ++x) {
            target[x] = qRgba(245, 145, 35, static_cast<int>(source[x] * 0.48));
        }
    }
    return result;
}
}

GeometryRuleCanvas::GeometryRuleCanvas(QWidget *parent) : QWidget(parent) {
    setMinimumSize(240, 160);
    setMouseTracking(true);
    setAutoFillBackground(true);
}

void GeometryRuleCanvas::setImage(const QImage &image) {
    image_ = image;
    dragging_ = false;
    update();
}

void GeometryRuleCanvas::setTool(Tool tool) {
    tool_ = tool;
    if (!coarseShape_.isEmpty()) {
        coarseShape_.insert(QStringLiteral("shape"),
                            tool_ == Circle ? QStringLiteral("circle")
                                            : tool_ == Ellipse ? QStringLiteral("ellipse")
                                                               : QStringLiteral("rotated_rectangle"));
        update();
    }
}

void GeometryRuleCanvas::setCoarseShape(const QJsonObject &shape) {
    coarseShape_ = shape;
    const QString value = shape.value(QStringLiteral("shape")).toString();
    if (value == QStringLiteral("ellipse")) {
        tool_ = Ellipse;
    } else if (value == QStringLiteral("rotated_rectangle")) {
        tool_ = RotatedRectangle;
    } else if (value == QStringLiteral("circle")) {
        tool_ = Circle;
    }
    rotationDegrees_ = shape.value(QStringLiteral("angle_deg")).toDouble(rotationDegrees_);
    update();
}

void GeometryRuleCanvas::setFitOverlay(const QJsonObject &fit, const QImage &maskOverlay) {
    fitShape_ = fit.value(QStringLiteral("fitted_shape")).toObject();
    if (fitShape_.isEmpty() && fit.contains(QStringLiteral("shape"))) {
        fitShape_ = fit;
    }
    maskOverlay_ = maskOverlay;
    update();
}

void GeometryRuleCanvas::setRotationDegrees(qreal degrees) {
    rotationDegrees_ = degrees;
    if (!coarseShape_.isEmpty() && coarseShape_.value(QStringLiteral("shape")).toString()
            == QStringLiteral("rotated_rectangle")) {
        coarseShape_.insert(QStringLiteral("angle_deg"), rotationDegrees_);
        emit shapeChanged(coarseShape_);
    }
    update();
}

QRectF GeometryRuleCanvas::imageTarget() const {
    if (image_.isNull() || width() <= 0 || height() <= 0) {
        return {};
    }
    const qreal scale = qMin(width() / static_cast<qreal>(image_.width()),
                             height() / static_cast<qreal>(image_.height()));
    const QSizeF rendered = QSizeF(image_.size()) * scale;
    return QRectF((width() - rendered.width()) / 2.0,
                  (height() - rendered.height()) / 2.0,
                  rendered.width(), rendered.height());
}

QPointF GeometryRuleCanvas::imagePoint(const QPoint &point) const {
    const QRectF target = imageTarget();
    if (target.isNull()) {
        return {};
    }
    const qreal scale = target.width() / static_cast<qreal>(image_.width());
    return QPointF(qBound(0.0, (point.x() - target.left()) / scale, static_cast<qreal>(image_.width())),
                   qBound(0.0, (point.y() - target.top()) / scale, static_cast<qreal>(image_.height())));
}

QJsonObject GeometryRuleCanvas::shapeFromDrag(const QPointF &start, const QPointF &end) const {
    const QRectF rect(start, end);
    const QRectF normalized = rect.normalized();
    const QPointF center = normalized.center();
    QJsonObject result;
    result.insert(QStringLiteral("cx"), center.x());
    result.insert(QStringLiteral("cy"), center.y());
    if (tool_ == Circle) {
        result.insert(QStringLiteral("shape"), QStringLiteral("circle"));
        result.insert(QStringLiteral("r"), qMin(normalized.width(), normalized.height()) / 2.0);
    } else if (tool_ == Ellipse) {
        result.insert(QStringLiteral("shape"), QStringLiteral("ellipse"));
        result.insert(QStringLiteral("rx"), normalized.width() / 2.0);
        result.insert(QStringLiteral("ry"), normalized.height() / 2.0);
        result.insert(QStringLiteral("angle_deg"), 0.0);
    } else {
        result.insert(QStringLiteral("shape"), QStringLiteral("rotated_rectangle"));
        result.insert(QStringLiteral("half_width"), normalized.width() / 2.0);
        result.insert(QStringLiteral("half_height"), normalized.height() / 2.0);
        result.insert(QStringLiteral("angle_deg"), rotationDegrees_);
    }
    return result;
}

void GeometryRuleCanvas::updateShapeFromDrag(const QPointF &end) {
    coarseShape_ = shapeFromDrag(dragStart_, end);
    emit shapeChanged(coarseShape_);
    update();
}

void GeometryRuleCanvas::drawShape(QPainter *painter, const QJsonObject &shape, const QPen &pen) const {
    if (painter == nullptr || shape.isEmpty() || image_.isNull()) {
        return;
    }
    const QString type = shape.value(QStringLiteral("shape")).toString();
    const QPointF center = jsonPoint(shape, "cx", "cy");
    const QRectF target = imageTarget();
    const qreal scale = target.width() / static_cast<qreal>(image_.width());
    painter->save();
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->translate(target.left(), target.top());
    painter->scale(scale, scale);
    if (type == QStringLiteral("circle")) {
        const qreal radius = shape.value(QStringLiteral("r")).toDouble();
        painter->drawEllipse(center, radius, radius);
    } else if (type == QStringLiteral("ellipse")) {
        const qreal rx = shape.value(QStringLiteral("rx")).toDouble();
        const qreal ry = shape.value(QStringLiteral("ry")).toDouble();
        painter->save();
        painter->translate(center);
        painter->rotate(shape.value(QStringLiteral("angle_deg")).toDouble());
        painter->drawEllipse(QPointF(), rx, ry);
        painter->restore();
    } else if (type == QStringLiteral("rotated_rectangle")) {
        const qreal halfWidth = shape.value(QStringLiteral("half_width")).toDouble();
        const qreal halfHeight = shape.value(QStringLiteral("half_height")).toDouble();
        painter->save();
        painter->translate(center);
        painter->rotate(shape.value(QStringLiteral("angle_deg")).toDouble());
        painter->drawRect(QRectF(-halfWidth, -halfHeight, 2.0 * halfWidth, 2.0 * halfHeight));
        painter->restore();
    }
    painter->restore();
}

void GeometryRuleCanvas::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event)
    QPainter painter(this);
    painter.fillRect(rect(), Qt::white);
    if (image_.isNull()) {
        painter.setPen(Qt::darkGray);
        painter.drawText(rect(), Qt::AlignCenter, QStringLiteral("模板图片无法读取"));
        return;
    }
    const QRectF target = imageTarget();
    painter.drawImage(target, image_);
    if (!maskOverlay_.isNull() && maskOverlay_.size() == image_.size()) {
        painter.drawImage(target, orangeMask(maskOverlay_));
    }
    QPen fitPen(QColor(35, 165, 75), 2.0);
    drawShape(&painter, fitShape_, fitPen);
    QPen coarsePen(QColor(0, 170, 200), 2.0, Qt::DashLine);
    drawShape(&painter, coarseShape_, coarsePen);
}

void GeometryRuleCanvas::mousePressEvent(QMouseEvent *event) {
    if (event->button() != Qt::LeftButton || image_.isNull()) {
        return;
    }
    dragStart_ = imagePoint(event->pos());
    dragging_ = true;
    updateShapeFromDrag(dragStart_);
}

void GeometryRuleCanvas::mouseMoveEvent(QMouseEvent *event) {
    if (!dragging_ || image_.isNull()) {
        return;
    }
    updateShapeFromDrag(imagePoint(event->pos()));
}

void GeometryRuleCanvas::mouseReleaseEvent(QMouseEvent *event) {
    if (!dragging_ || event->button() != Qt::LeftButton) {
        return;
    }
    dragging_ = false;
    updateShapeFromDrag(imagePoint(event->pos()));
}
