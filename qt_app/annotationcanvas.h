#pragma once

#include <QImage>
#include <QList>
#include <QRectF>
#include <QWidget>

struct AnnotationRegionView {
    QRectF rect;
    QString provenance;
    QString status;
    bool selected = false;
};

class AnnotationCanvas : public QWidget {
    Q_OBJECT

public:
    explicit AnnotationCanvas(QWidget *parent = nullptr);

    void setImage(const QImage &image);
    QImage image() const { return image_; }
    void setRegions(const QList<AnnotationRegionView> &regions);
    QList<AnnotationRegionView> regions() const { return regions_; }
    void setEditable(bool editable);
    bool isEditable() const { return editable_; }
    int selectedRegionIndex() const { return selectedRegion_; }
    QRectF nativeRegion() const;

public slots:
    void deleteSelectedRegion();

signals:
    void regionsChanged();
    void selectionChanged(int index);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    QPointF imagePoint(const QPoint &point) const;
    QRectF imageTarget() const;
    int regionAt(const QPointF &point) const;
    void setSelectedRegion(int index);
    void clampRegion(QRectF *region) const;

    QImage image_;
    QList<AnnotationRegionView> regions_;
    QPointF dragStart_;
    QRectF dragRegionStart_;
    bool dragging_ = false;
    bool moving_ = false;
    bool editable_ = true;
    int selectedRegion_ = -1;
};
