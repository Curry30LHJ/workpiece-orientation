#include "geometryrulecanvas.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QWheelEvent>
#include <QKeyEvent>

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
    setFocusPolicy(Qt::StrongFocus);
}

void GeometryRuleCanvas::setImage(const QImage &image) {
    image_ = image;
    // A fitted contour belongs to the previous image and must never be
    // displayed on a newly selected template.
    fitShape_ = QJsonObject();
    maskOverlay_ = QImage();
    dragging_ = false;
    panning_ = false;
    resetView();
    update();
}

void GeometryRuleCanvas::setTool(Tool tool) {
    tool_ = tool;
    if (!coarseShape_.isEmpty()) {
        if (tool_ != None) {
            coarseShape_.insert(QStringLiteral("shape"),
                                tool_ == Circle ? QStringLiteral("circle")
                                                : tool_ == Ellipse ? QStringLiteral("ellipse")
                                                                   : QStringLiteral("rotated_rectangle"));
        }
        update();
    }
}

void GeometryRuleCanvas::setCoarseShape(const QJsonObject &shape) {
    coarseShape_ = shape;
    if (shape.isEmpty()) {
        tool_ = None;
        dragging_ = false;
        update();
        return;
    }
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
    // The effective shape is the boundary actually used by the mask after
    // applying the signed offset. Prefer it over the raw fitted contour so
    // the preview matches the production mask semantics.
    fitShape_ = fit.value(QStringLiteral("effective_shape")).toObject();
    if (fitShape_.isEmpty()) {
        fitShape_ = fit.value(QStringLiteral("fitted_shape")).toObject();
    }
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

void GeometryRuleCanvas::resetView() {
    zoom_ = 1.0;
    pan_ = QPointF();
    update();
}

void GeometryRuleCanvas::cancelGesture() {
    dragging_ = false;
    panning_ = false;
    update();
}

void GeometryRuleCanvas::setNumericShape(const QJsonObject &shape) {
    const QString type = shape.value(QStringLiteral("shape")).toString();
    bool valid = !type.isEmpty() && shape.contains(QStringLiteral("cx")) && shape.contains(QStringLiteral("cy"));
    if (type == QStringLiteral("circle")) {
        valid = valid && shape.value(QStringLiteral("r")).toDouble() > 0.0;
        tool_ = Circle;
    } else if (type == QStringLiteral("ellipse")) {
        valid = valid && shape.value(QStringLiteral("rx")).toDouble() > 0.0
                && shape.value(QStringLiteral("ry")).toDouble() > 0.0;
        tool_ = Ellipse;
    } else if (type == QStringLiteral("rotated_rectangle")) {
        valid = valid && shape.value(QStringLiteral("half_width")).toDouble() > 0.0
                && shape.value(QStringLiteral("half_height")).toDouble() > 0.0;
        tool_ = RotatedRectangle;
    } else {
        valid = false;
    }
    if (!valid) {
        return;
    }
    coarseShape_ = shape;
    rotationDegrees_ = shape.value(QStringLiteral("angle_deg")).toDouble(rotationDegrees_);
    emit shapeChanged(coarseShape_);
    update();
}

QRectF GeometryRuleCanvas::imageTarget() const {
    if (image_.isNull() || width() <= 0 || height() <= 0) {
        return {};
    }
    const qreal scale = qMin(width() / static_cast<qreal>(image_.width()),
                             height() / static_cast<qreal>(image_.height())) * zoom_;
    const QSizeF rendered = QSizeF(image_.size()) * scale;
    const qreal baseScale = qMin(width() / static_cast<qreal>(image_.width()),
                                 height() / static_cast<qreal>(image_.height()));
    const QSizeF baseRendered = QSizeF(image_.size()) * baseScale;
    const QPointF center((width() - baseRendered.width()) / 2.0 + baseRendered.width() / 2.0,
                         (height() - baseRendered.height()) / 2.0 + baseRendered.height() / 2.0);
    return QRectF(center.x() - rendered.width() / 2.0 + pan_.x(),
                  center.y() - rendered.height() / 2.0 + pan_.y(),
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
    const qreal dx = end.x() - start.x();
    const qreal dy = end.y() - start.y();
    QJsonObject result;
    result.insert(QStringLiteral("cx"), start.x());
    result.insert(QStringLiteral("cy"), start.y());
    if (tool_ == Circle) {
        result.insert(QStringLiteral("shape"), QStringLiteral("circle"));
        result.insert(QStringLiteral("r"), qSqrt(dx * dx + dy * dy));
    } else if (tool_ == Ellipse) {
        result.insert(QStringLiteral("shape"), QStringLiteral("ellipse"));
        result.insert(QStringLiteral("rx"), qAbs(dx));
        result.insert(QStringLiteral("ry"), qAbs(dy));
        result.insert(QStringLiteral("angle_deg"), 0.0);
    } else {
        result.insert(QStringLiteral("shape"), QStringLiteral("rotated_rectangle"));
        result.insert(QStringLiteral("half_width"), qAbs(dx));
        result.insert(QStringLiteral("half_height"), qAbs(dy));
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
        const qreal radius = shape.contains(QStringLiteral("r"))
            ? shape.value(QStringLiteral("r")).toDouble()
            : qMin(shape.value(QStringLiteral("rx")).toDouble(),
                   shape.value(QStringLiteral("ry")).toDouble());
        if (radius <= 0.0) {
            painter->restore();
            return;
        }
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

void GeometryRuleCanvas::drawHandles(QPainter *painter, const QJsonObject &shape) const {
    if (painter == nullptr || shape.isEmpty()) {
        return;
    }
    const QRectF target = imageTarget();
    if (target.isNull()) {
        return;
    }
    const qreal scale = target.width() / static_cast<qreal>(image_.width());
    const QPointF center = jsonPoint(shape, "cx", "cy");
    painter->save();
    painter->setPen(QPen(QColor(250, 190, 40), 1.5));
    painter->setBrush(QColor(250, 190, 40));
    painter->translate(target.left(), target.top());
    painter->scale(scale, scale);
    const qreal handleRadius = 4.0 / qMax(scale, 0.1);
    painter->drawEllipse(center, handleRadius, handleRadius);
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
    drawHandles(&painter, coarseShape_);
}

void GeometryRuleCanvas::mousePressEvent(QMouseEvent *event) {
    if (image_.isNull()) {
        return;
    }
    if (event->button() == Qt::MiddleButton) {
        panning_ = true;
        lastPanPoint_ = event->pos();
        return;
    }
    if (event->button() != Qt::LeftButton || tool_ == None) {
        return;
    }
    setFocus();
    fitShape_ = QJsonObject();
    maskOverlay_ = QImage();
    dragStart_ = imagePoint(event->pos());
    dragging_ = true;
    updateShapeFromDrag(dragStart_);
}

void GeometryRuleCanvas::mouseMoveEvent(QMouseEvent *event) {
    if (panning_) {
        const QPoint delta = event->pos() - lastPanPoint_;
        pan_ += QPointF(delta);
        lastPanPoint_ = event->pos();
        update();
        return;
    }
    if (!dragging_ || image_.isNull()) {
        return;
    }
    updateShapeFromDrag(imagePoint(event->pos()));
}

void GeometryRuleCanvas::mouseReleaseEvent(QMouseEvent *event) {
    if (panning_ && event->button() == Qt::MiddleButton) {
        panning_ = false;
        return;
    }
    if (!dragging_ || event->button() != Qt::LeftButton) {
        return;
    }
    dragging_ = false;
    updateShapeFromDrag(imagePoint(event->pos()));
    const QString type = coarseShape_.value(QStringLiteral("shape")).toString();
    const bool valid = (type == QStringLiteral("circle") && coarseShape_.value(QStringLiteral("r")).toDouble() > 0.0)
            || (type == QStringLiteral("ellipse") && coarseShape_.value(QStringLiteral("rx")).toDouble() > 0.0
                && coarseShape_.value(QStringLiteral("ry")).toDouble() > 0.0)
            || (type == QStringLiteral("rotated_rectangle")
                && coarseShape_.value(QStringLiteral("half_width")).toDouble() > 0.0
                && coarseShape_.value(QStringLiteral("half_height")).toDouble() > 0.0);
    if (valid) {
        emit shapeCommitted(coarseShape_);
    }
}

void GeometryRuleCanvas::wheelEvent(QWheelEvent *event) {
    if (image_.isNull() || event == nullptr) {
        return;
    }
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    const QPointF eventPosition = event->position();
#else
    const QPointF eventPosition = event->posF();
#endif
    const QPointF before = imagePoint(eventPosition.toPoint());
    const qreal factor = event->angleDelta().y() >= 0 ? 1.15 : (1.0 / 1.15);
    zoom_ = qBound(0.25, zoom_ * factor, 8.0);
    const QRectF target = imageTarget();
    if (!target.isNull()) {
        const qreal scale = target.width() / static_cast<qreal>(image_.width());
        const QPointF after(target.left() + before.x() * scale, target.top() + before.y() * scale);
        pan_ += eventPosition - after;
    }
    event->accept();
    update();
}

void GeometryRuleCanvas::keyPressEvent(QKeyEvent *event) {
    if (event == nullptr) {
        return;
    }
    if (event->key() == Qt::Key_Escape) {
        cancelGesture();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_R) {
        resetView();
        event->accept();
        return;
    }
    QWidget::keyPressEvent(event);
}
