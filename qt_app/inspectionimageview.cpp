#include "inspectionimageview.h"

#include <QDragEnterEvent>
#include <QFileInfo>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QImageReader>
#include <QMimeData>
#include <QPixmap>
#include <QUrl>
#include <QWheelEvent>

namespace {
bool hasSupportedSuffix(const QString &path) {
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == QStringLiteral("png") || suffix == QStringLiteral("jpg")
        || suffix == QStringLiteral("jpeg") || suffix == QStringLiteral("bmp")
        || suffix == QStringLiteral("tif") || suffix == QStringLiteral("tiff");
}

bool isReadableImage(const QString &path) {
    if (!QFileInfo(path).isFile() || !hasSupportedSuffix(path)) {
        return false;
    }
    QImageReader reader(path);
    return reader.canRead();
}
}

InspectionImageView::InspectionImageView(QWidget *parent)
    : QGraphicsView(parent), scene_(new QGraphicsScene(this)) {
    setScene(scene_);
    setAcceptDrops(true);
    viewport()->setAcceptDrops(true);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setBackgroundBrush(QColor(246, 247, 249));
}

bool InspectionImageView::setImagePath(const QString &path) {
    const QString absolutePath = QFileInfo(path).absoluteFilePath();
    QImageReader reader(absolutePath);
    const QImage image = isReadableImage(absolutePath) ? reader.read() : QImage();
    if (image.isNull()) {
        scene_->clear();
        pixmapItem_ = nullptr;
        imagePath_.clear();
        resetTransform();
        zoomFactor_ = 1.0;
        return false;
    }

    scene_->clear();
    pixmapItem_ = scene_->addPixmap(QPixmap::fromImage(image));
    scene_->setSceneRect(pixmapItem_->boundingRect());
    imagePath_ = absolutePath;
    resetView();
    return true;
}

QString InspectionImageView::imagePath() const {
    return imagePath_;
}

qreal InspectionImageView::zoomFactor() const {
    return zoomFactor_;
}

void InspectionImageView::resetView() {
    resetTransform();
    zoomFactor_ = 1.0;
    if (pixmapItem_ != nullptr && !pixmapItem_->pixmap().isNull()) {
        fitInView(pixmapItem_, Qt::KeepAspectRatio);
    }
}

QString InspectionImageView::droppedImagePath(const QMimeData *mimeData) const {
    if (mimeData == nullptr || !mimeData->hasUrls()) {
        return QString();
    }
    const QList<QUrl> urls = mimeData->urls();
    if (urls.size() != 1 || !urls.constFirst().isLocalFile()) {
        return QString();
    }
    const QString path = QFileInfo(urls.constFirst().toLocalFile()).absoluteFilePath();
    return isReadableImage(path) ? path : QString();
}

void InspectionImageView::dragEnterEvent(QDragEnterEvent *event) {
    if (!droppedImagePath(event->mimeData()).isEmpty()) {
        event->acceptProposedAction();
        return;
    }
    event->ignore();
}

void InspectionImageView::dropEvent(QDropEvent *event) {
    const QString path = droppedImagePath(event->mimeData());
    if (path.isEmpty()) {
        event->ignore();
        return;
    }
    event->acceptProposedAction();
    emit imageDropped(path);
}

void InspectionImageView::wheelEvent(QWheelEvent *event) {
    if (pixmapItem_ == nullptr || event->angleDelta().y() == 0) {
        QGraphicsView::wheelEvent(event);
        return;
    }
    const qreal requested = event->angleDelta().y() > 0
        ? zoomFactor_ * 1.2 : zoomFactor_ / 1.2;
    const qreal bounded = qBound<qreal>(0.1, requested, 8.0);
    const qreal scaleChange = bounded / zoomFactor_;
    if (!qFuzzyCompare(scaleChange, 1.0)) {
        scale(scaleChange, scaleChange);
        zoomFactor_ = bounded;
    }
    event->accept();
}
