#include "annotationcanvas.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPen>

namespace {
QColor regionColor(const AnnotationRegionView &region) {
    if (region.status == QStringLiteral("needs_review")) {
        return QColor(220, 130, 20);
    }
    if (region.provenance == QStringLiteral("automatic")) {
        return QColor(35, 110, 210);
    }
    return QColor(35, 160, 70);
}
}

AnnotationCanvas::AnnotationCanvas(QWidget *parent) : QWidget(parent) {
    setMinimumSize(200, 100);
    setMouseTracking(true);
    setAutoFillBackground(true);
}

void AnnotationCanvas::setImage(const QImage &image) {
    image_ = image;
    selectedRegion_ = -1;
    dragging_ = false;
    moving_ = false;
    update();
}

void AnnotationCanvas::setRegions(const QList<AnnotationRegionView> &regions) {
    regions_ = regions;
    selectedRegion_ = -1;
    for (int index = 0; index < regions_.size(); ++index) {
        regions_[index].selected = false;
    }
    update();
}

void AnnotationCanvas::setEditable(bool editable) {
    editable_ = editable;
    if (!editable_) {
        setSelectedRegion(-1);
    }
    update();
}

QRectF AnnotationCanvas::nativeRegion() const {
    return selectedRegion_ >= 0 && selectedRegion_ < regions_.size()
        ? regions_.at(selectedRegion_).rect.normalized() : QRectF();
}

QRectF AnnotationCanvas::imageTarget() const {
    if (image_.isNull()) {
        return QRectF();
    }
    const qreal scale = qMin(width() / static_cast<qreal>(image_.width()),
                             height() / static_cast<qreal>(image_.height()));
    const QSizeF rendered = QSizeF(image_.size()) * scale;
    return QRectF((width() - rendered.width()) / 2.0, (height() - rendered.height()) / 2.0,
                  rendered.width(), rendered.height());
}

QPointF AnnotationCanvas::imagePoint(const QPoint &point) const {
    const QRectF target = imageTarget();
    if (target.isNull()) {
        return {};
    }
    const qreal scale = target.width() / static_cast<qreal>(image_.width());
    return QPointF(qBound(0.0, (point.x() - target.left()) / scale, static_cast<qreal>(image_.width())),
                   qBound(0.0, (point.y() - target.top()) / scale, static_cast<qreal>(image_.height())));
}

int AnnotationCanvas::regionAt(const QPointF &point) const {
    for (int index = regions_.size() - 1; index >= 0; --index) {
        if (regions_.at(index).rect.contains(point)) {
            return index;
        }
    }
    return -1;
}

void AnnotationCanvas::setSelectedRegion(int index) {
    index = index >= 0 && index < regions_.size() ? index : -1;
    if (selectedRegion_ == index) {
        return;
    }
    selectedRegion_ = index;
    for (int i = 0; i < regions_.size(); ++i) {
        regions_[i].selected = i == selectedRegion_;
    }
    emit selectionChanged(selectedRegion_);
    update();
}

void AnnotationCanvas::clampRegion(QRectF *region) const {
    if (region == nullptr || image_.isNull()) {
        return;
    }
    *region = region->normalized();
    region->setLeft(qBound(0.0, region->left(), static_cast<qreal>(image_.width())));
    region->setTop(qBound(0.0, region->top(), static_cast<qreal>(image_.height())));
    region->setRight(qBound(0.0, region->right(), static_cast<qreal>(image_.width())));
    region->setBottom(qBound(0.0, region->bottom(), static_cast<qreal>(image_.height())));
}

void AnnotationCanvas::paintEvent(QPaintEvent *event) {
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
    const qreal scale = target.width() / static_cast<qreal>(image_.width());
    for (int index = 0; index < regions_.size(); ++index) {
        const AnnotationRegionView &region = regions_.at(index);
        const QRectF display(target.left() + region.rect.left() * scale,
                             target.top() + region.rect.top() * scale,
                             region.rect.width() * scale, region.rect.height() * scale);
        QPen pen(regionColor(region), region.selected ? 4.0 : 2.0);
        if (region.status == QStringLiteral("needs_review")) {
            pen.setStyle(Qt::DashLine);
        }
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(display);
        painter.setPen(regionColor(region));
        painter.drawText(display.topLeft() + QPointF(3, 14), QString::number(index + 1));
    }
}

void AnnotationCanvas::mousePressEvent(QMouseEvent *event) {
    if (event->button() != Qt::LeftButton || !editable_ || image_.isNull()) {
        return;
    }
    const QPointF point = imagePoint(event->pos());
    const int hit = regionAt(point);
    if (hit >= 0) {
        setSelectedRegion(hit);
        dragStart_ = point;
        dragRegionStart_ = regions_.at(hit).rect;
        moving_ = true;
        dragging_ = true;
        return;
    }
    setSelectedRegion(-1);
    dragStart_ = point;
    moving_ = false;
    dragging_ = true;
    regions_.append({QRectF(point, point), QStringLiteral("manual"), QStringLiteral("active"), true});
    selectedRegion_ = regions_.size() - 1;
    emit selectionChanged(selectedRegion_);
    update();
}

void AnnotationCanvas::mouseMoveEvent(QMouseEvent *event) {
    if (!dragging_ || !editable_ || image_.isNull()) {
        return;
    }
    const QPointF point = imagePoint(event->pos());
    if (moving_ && selectedRegion_ >= 0) {
        QRectF moved = dragRegionStart_.translated(point - dragStart_);
        clampRegion(&moved);
        regions_[selectedRegion_].rect = moved;
    } else if (selectedRegion_ >= 0) {
        regions_[selectedRegion_].rect = QRectF(dragStart_, point).normalized();
        clampRegion(&regions_[selectedRegion_].rect);
    }
    update();
}

void AnnotationCanvas::mouseReleaseEvent(QMouseEvent *event) {
    if (!dragging_ || event->button() != Qt::LeftButton) {
        return;
    }
    dragging_ = false;
    moving_ = false;
    if (selectedRegion_ >= 0 && selectedRegion_ < regions_.size()) {
        if (regions_[selectedRegion_].rect.width() < 1.0 && regions_[selectedRegion_].rect.height() < 1.0) {
            regions_[selectedRegion_].rect = QRectF(dragStart_, imagePoint(event->pos())).normalized();
        }
        clampRegion(&regions_[selectedRegion_].rect);
        if (regions_[selectedRegion_].rect.width() < 1.0 || regions_[selectedRegion_].rect.height() < 1.0) {
            regions_.removeAt(selectedRegion_);
            setSelectedRegion(-1);
        }
    }
    emit regionsChanged();
    update();
}

void AnnotationCanvas::deleteSelectedRegion() {
    if (!editable_ || selectedRegion_ < 0 || selectedRegion_ >= regions_.size()) {
        return;
    }
    regions_.removeAt(selectedRegion_);
    setSelectedRegion(-1);
    emit regionsChanged();
    update();
}
