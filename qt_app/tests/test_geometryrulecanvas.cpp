#include <QtTest/QtTest>

#include <QImage>
#include <QJsonObject>
#include <QSignalSpy>

#include "../geometryrulecanvas.h"

namespace {
QImage renderCanvas(GeometryRuleCanvas *canvas) {
    QImage rendered(canvas->size(), QImage::Format_ARGB32_Premultiplied);
    rendered.fill(Qt::transparent);
    canvas->render(&rendered);
    return rendered;
}

int countPixelsNear(const QImage &image, const QColor &color, int tolerance = 4) {
    int count = 0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const QColor pixel = image.pixelColor(x, y);
            if (qAbs(pixel.red() - color.red()) <= tolerance
                    && qAbs(pixel.green() - color.green()) <= tolerance
                    && qAbs(pixel.blue() - color.blue()) <= tolerance) {
                ++count;
            }
        }
    }
    return count;
}
}

class TestGeometryRuleCanvas : public QObject {
    Q_OBJECT

private slots:
    void draggedCircleUsesNativeImageCoordinates();
    void draggedEllipseUsesNativeImageCoordinates();
    void rotatedRectangleUsesConfiguredAngle();
    void noToolDoesNotDraw();
    void escapeCancelsIncompleteGesture();
    void numericShapeAndResetViewRemainStable();
    void overlaysAndResizeRemainSafe();
    void changingImageClearsPreviousFitOverlay();
    void effectiveShapeIsPreferredOverRawShape();
    void hidingGuideDoesNotDeleteItsGeometry();
    void fittedAndEffectiveBoundariesRemainDistinct();
    void layerVisibilityControlsRemainIndependent();
    void fitOverlayInputFormsAndClearingRemainCompatible();
    void visibilityPersistsWithoutMutatingCanvasState();
    void paintLayersUseFixedOrderAndVisibility();
    void startingNewGuideClearsFitLayersAndMask();
};

void TestGeometryRuleCanvas::draggedCircleUsesNativeImageCoordinates() {
    GeometryRuleCanvas canvas;
    canvas.resize(600, 400);
    canvas.setImage(QImage(300, 300, QImage::Format_RGB32));
    canvas.setTool(GeometryRuleCanvas::Circle);
    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(300, 200));
    QTest::mouseMove(&canvas, QPoint(367, 200));
    QTest::mouseRelease(&canvas, Qt::LeftButton, {}, QPoint(367, 200));
    const QJsonObject shape = canvas.coarseShape();
    QCOMPARE(shape.value("shape").toString(), QStringLiteral("circle"));
    QVERIFY(qAbs(shape.value("cx").toDouble() - 150.0) < 1.0);
    QVERIFY(qAbs(shape.value("cy").toDouble() - 150.0) < 1.0);
    QVERIFY(qAbs(shape.value("r").toDouble() - 50.0) < 1.0);
}

void TestGeometryRuleCanvas::draggedEllipseUsesNativeImageCoordinates() {
    GeometryRuleCanvas canvas;
    canvas.resize(600, 400);
    canvas.setImage(QImage(300, 200, QImage::Format_RGB32));
    canvas.setTool(GeometryRuleCanvas::Ellipse);
    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(300, 200));
    QTest::mouseRelease(&canvas, Qt::LeftButton, {}, QPoint(450, 300));
    const QJsonObject shape = canvas.coarseShape();
    QCOMPARE(shape.value("shape").toString(), QStringLiteral("ellipse"));
    QVERIFY(qAbs(shape.value("cx").toDouble() - 150.0) < 1.0);
    QVERIFY(qAbs(shape.value("cy").toDouble() - 100.0) < 1.0);
    QVERIFY(qAbs(shape.value("rx").toDouble() - 75.0) < 1.0);
    QVERIFY(qAbs(shape.value("ry").toDouble() - 50.0) < 1.0);
}

void TestGeometryRuleCanvas::rotatedRectangleUsesConfiguredAngle() {
    GeometryRuleCanvas canvas;
    canvas.resize(400, 400);
    canvas.setImage(QImage(200, 200, QImage::Format_RGB32));
    canvas.setTool(GeometryRuleCanvas::RotatedRectangle);
    canvas.setRotationDegrees(27.5);
    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(100, 100));
    QTest::mouseRelease(&canvas, Qt::LeftButton, {}, QPoint(300, 300));
    const QJsonObject shape = canvas.coarseShape();
    QCOMPARE(shape.value("shape").toString(), QStringLiteral("rotated_rectangle"));
    QVERIFY(qAbs(shape.value("angle_deg").toDouble() - 27.5) < 0.01);
    QVERIFY(shape.value("half_width").toDouble() > 45.0);
}

void TestGeometryRuleCanvas::noToolDoesNotDraw() {
    GeometryRuleCanvas canvas;
    canvas.resize(400, 400);
    canvas.setImage(QImage(200, 200, QImage::Format_RGB32));
    canvas.setTool(GeometryRuleCanvas::None);
    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(200, 200));
    QTest::mouseRelease(&canvas, Qt::LeftButton, {}, QPoint(300, 200));
    QVERIFY(canvas.coarseShape().isEmpty());
}

void TestGeometryRuleCanvas::escapeCancelsIncompleteGesture() {
    GeometryRuleCanvas canvas;
    canvas.resize(400, 400);
    canvas.setImage(QImage(200, 200, QImage::Format_RGB32));
    canvas.setTool(GeometryRuleCanvas::Circle);
    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(200, 200));
    QTest::keyClick(&canvas, Qt::Key_Escape);
    QVERIFY(canvas.coarseShape().value("r").toDouble() == 0.0);
}

void TestGeometryRuleCanvas::numericShapeAndResetViewRemainStable() {
    GeometryRuleCanvas canvas;
    canvas.resize(400, 400);
    canvas.setImage(QImage(200, 200, QImage::Format_RGB32));
    canvas.setNumericShape(QJsonObject{{"shape", "circle"}, {"cx", 100}, {"cy", 100}, {"r", 30}});
    QCOMPARE(canvas.coarseShape().value("cx").toDouble(), 100.0);
    canvas.resetView();
    QVERIFY(qAbs(canvas.imageTarget().size().width() - 400.0) < 0.01);
}

void TestGeometryRuleCanvas::overlaysAndResizeRemainSafe() {
    GeometryRuleCanvas canvas;
    canvas.resize(320, 240);
    canvas.setImage(QImage(640, 480, QImage::Format_RGB32));
    canvas.setFitOverlay(QJsonObject{{"shape", "ellipse"}, {"cx", 320}, {"cy", 240},
                                     {"rx", 200}, {"ry", 150}, {"angle_deg", 0}},
                         QImage(640, 480, QImage::Format_ARGB32));
    canvas.resize(800, 300);
    QImage rendered(canvas.size(), QImage::Format_ARGB32_Premultiplied);
    rendered.fill(Qt::transparent);
    canvas.render(&rendered);
    QVERIFY(canvas.coarseShape().isEmpty());
}

void TestGeometryRuleCanvas::changingImageClearsPreviousFitOverlay() {
    GeometryRuleCanvas canvas;
    canvas.setImage(QImage(120, 120, QImage::Format_RGB32));
    canvas.setFitOverlay(QJsonObject{
        {"fitted_shape", QJsonObject{{"shape", "circle"}, {"cx", 60}, {"cy", 60}, {"r", 45}}},
        {"effective_shape", QJsonObject{{"shape", "circle"}, {"cx", 60}, {"cy", 60}, {"r", 40}}},
    });
    QVERIFY(!canvas.fitShape().isEmpty());

    canvas.setImage(QImage(120, 120, QImage::Format_RGB32));

    QVERIFY(canvas.fitShape().isEmpty());
    QVERIFY(canvas.fittedShape().isEmpty());
    QVERIFY(canvas.effectiveShape().isEmpty());
}

void TestGeometryRuleCanvas::effectiveShapeIsPreferredOverRawShape() {
    GeometryRuleCanvas canvas;
    canvas.setImage(QImage(120, 120, QImage::Format_RGB32));
    canvas.setFitOverlay(QJsonObject{
        {"fitted_shape", QJsonObject{{"shape", "circle"}, {"cx", 60}, {"cy", 60}, {"r", 45}}},
        {"effective_shape", QJsonObject{{"shape", "circle"}, {"cx", 60}, {"cy", 60}, {"r", 40}}},
    });

    QCOMPARE(canvas.fitShape().value("r").toDouble(), 40.0);
}

void TestGeometryRuleCanvas::hidingGuideDoesNotDeleteItsGeometry() {
    GeometryRuleCanvas canvas;
    const QJsonObject guide{{"shape", "circle"}, {"cx", 100.0},
                            {"cy", 100.0}, {"r", 60.0}};
    canvas.setCoarseShape(guide);

    canvas.setGuideVisible(false);

    QCOMPARE(canvas.coarseShape(), guide);
    QVERIFY(!canvas.guideVisible());
}

void TestGeometryRuleCanvas::fittedAndEffectiveBoundariesRemainDistinct() {
    GeometryRuleCanvas canvas;
    canvas.setFitOverlay({
        {"fitted_shape", QJsonObject{{"shape", "circle"}, {"r", 50.0}}},
        {"effective_shape", QJsonObject{{"shape", "circle"}, {"r", 48.0}}},
    });

    QCOMPARE(canvas.fittedShape().value("r").toDouble(), 50.0);
    QCOMPARE(canvas.effectiveShape().value("r").toDouble(), 48.0);
    canvas.setFittedBoundaryVisible(false);

    QVERIFY(!canvas.fittedBoundaryVisible());
    QCOMPARE(canvas.effectiveShape().value("r").toDouble(), 48.0);
}

void TestGeometryRuleCanvas::layerVisibilityControlsRemainIndependent() {
    GeometryRuleCanvas canvas;

    QVERIFY(canvas.guideVisible());
    QVERIFY(canvas.fittedBoundaryVisible());
    QVERIFY(canvas.effectiveBoundaryVisible());
    QVERIFY(canvas.maskOverlayVisible());

    canvas.setGuideVisible(false);
    canvas.setFittedBoundaryVisible(false);
    canvas.setEffectiveBoundaryVisible(false);
    canvas.setMaskOverlayVisible(false);

    QVERIFY(!canvas.guideVisible());
    QVERIFY(!canvas.fittedBoundaryVisible());
    QVERIFY(!canvas.effectiveBoundaryVisible());
    QVERIFY(!canvas.maskOverlayVisible());

    canvas.setEffectiveBoundaryVisible(true);
    QVERIFY(!canvas.guideVisible());
    QVERIFY(!canvas.fittedBoundaryVisible());
    QVERIFY(canvas.effectiveBoundaryVisible());
    QVERIFY(!canvas.maskOverlayVisible());
}

void TestGeometryRuleCanvas::fitOverlayInputFormsAndClearingRemainCompatible() {
    GeometryRuleCanvas canvas;
    const QJsonObject fitted{{"shape", "circle"}, {"cx", 50.0},
                             {"cy", 50.0}, {"r", 40.0}};
    const QJsonObject effective{{"shape", "circle"}, {"cx", 50.0},
                                {"cy", 50.0}, {"r", 36.0}};

    canvas.setFitOverlay(QJsonObject{{"fitted_shape", fitted}});
    QCOMPARE(canvas.fittedShape(), fitted);
    QVERIFY(canvas.effectiveShape().isEmpty());
    QCOMPARE(canvas.fitShape(), fitted);

    canvas.setFitOverlay(QJsonObject{{"effective_shape", effective}});
    QVERIFY(canvas.fittedShape().isEmpty());
    QCOMPARE(canvas.effectiveShape(), effective);
    QCOMPARE(canvas.fitShape(), effective);

    canvas.setFitOverlay(fitted);
    QCOMPARE(canvas.fittedShape(), fitted);
    QVERIFY(canvas.effectiveShape().isEmpty());
    QCOMPARE(canvas.fitShape(), fitted);

    canvas.setFitOverlay(QJsonObject());
    QVERIFY(canvas.fittedShape().isEmpty());
    QVERIFY(canvas.effectiveShape().isEmpty());
    QVERIFY(canvas.fitShape().isEmpty());
}

void TestGeometryRuleCanvas::visibilityPersistsWithoutMutatingCanvasState() {
    GeometryRuleCanvas canvas;
    canvas.resize(300, 200);
    QImage image(300, 200, QImage::Format_RGB32);
    image.fill(Qt::white);
    canvas.setImage(image);
    const QJsonObject guide{{"shape", "circle"}, {"cx", 150.0},
                            {"cy", 100.0}, {"r", 60.0}};
    canvas.setCoarseShape(guide);
    QSignalSpy shapeChanged(&canvas, &GeometryRuleCanvas::shapeChanged);

    canvas.setGuideVisible(false);
    canvas.setFittedBoundaryVisible(false);
    canvas.setEffectiveBoundaryVisible(false);
    canvas.setMaskOverlayVisible(false);
    canvas.setFitOverlay(QJsonObject{
        {"fitted_shape", QJsonObject{{"shape", "circle"}, {"r", 50.0}}},
        {"effective_shape", QJsonObject{{"shape", "circle"}, {"r", 48.0}}},
    });

    QVERIFY(!canvas.guideVisible());
    QVERIFY(!canvas.fittedBoundaryVisible());
    QVERIFY(!canvas.effectiveBoundaryVisible());
    QVERIFY(!canvas.maskOverlayVisible());
    QCOMPARE(canvas.coarseShape(), guide);

    canvas.setImage(image);
    QVERIFY(!canvas.guideVisible());
    QVERIFY(!canvas.fittedBoundaryVisible());
    QVERIFY(!canvas.effectiveBoundaryVisible());
    QVERIFY(!canvas.maskOverlayVisible());
    QCOMPARE(canvas.coarseShape(), guide);
    QVERIFY(canvas.fittedShape().isEmpty());
    QVERIFY(canvas.effectiveShape().isEmpty());

    const QRectF targetBeforeToggles = canvas.imageTarget();
    canvas.setGuideVisible(true);
    canvas.setFittedBoundaryVisible(true);
    canvas.setEffectiveBoundaryVisible(true);
    canvas.setMaskOverlayVisible(true);
    QCOMPARE(canvas.imageTarget(), targetBeforeToggles);
    QCOMPARE(canvas.coarseShape(), guide);
    QCOMPARE(shapeChanged.count(), 0);
}

void TestGeometryRuleCanvas::paintLayersUseFixedOrderAndVisibility() {
    GeometryRuleCanvas canvas;
    canvas.resize(240, 240);
    QImage image(240, 240, QImage::Format_RGB32);
    image.fill(Qt::white);
    canvas.setImage(image);
    const QJsonObject boundary{{"shape", "circle"}, {"cx", 120.0},
                               {"cy", 120.0}, {"r", 70.0}};
    canvas.setCoarseShape(boundary);
    canvas.setFitOverlay(QJsonObject{{"fitted_shape", boundary},
                                     {"effective_shape", boundary}});
    const QRectF targetBeforePaint = canvas.imageTarget();

    const QImage layered = renderCanvas(&canvas);
    QCOMPARE(countPixelsNear(layered, QColor(0, 170, 200)), 0);
    QVERIFY(countPixelsNear(layered, QColor(35, 165, 75)) > 0);
    QVERIFY(countPixelsNear(layered, QColor(245, 145, 35)) > 0);
    QCOMPARE(canvas.imageTarget(), targetBeforePaint);
    QCOMPARE(canvas.coarseShape(), boundary);
    QCOMPARE(canvas.fittedShape(), boundary);
    QCOMPARE(canvas.effectiveShape(), boundary);

    canvas.setFittedBoundaryVisible(false);
    canvas.setEffectiveBoundaryVisible(false);
    const QImage guideAndHandles = renderCanvas(&canvas);
    QVERIFY(countPixelsNear(guideAndHandles, QColor(0, 170, 200)) > 0);
    QVERIFY(countPixelsNear(guideAndHandles, QColor(250, 190, 40)) > 0);

    canvas.setGuideVisible(false);
    const QImage guideHidden = renderCanvas(&canvas);
    QCOMPARE(countPixelsNear(guideHidden, QColor(0, 170, 200)), 0);
    QCOMPARE(countPixelsNear(guideHidden, QColor(250, 190, 40)), 0);
    QCOMPARE(canvas.coarseShape(), boundary);

    QImage mask(240, 240, QImage::Format_Grayscale8);
    mask.fill(255);
    canvas.setFitOverlay(QJsonObject(), mask);
    canvas.setMaskOverlayVisible(true);
    const QColor maskVisibleCenter = renderCanvas(&canvas).pixelColor(120, 120);
    QVERIFY(maskVisibleCenter != QColor(Qt::white));
    canvas.setMaskOverlayVisible(false);
    QCOMPARE(renderCanvas(&canvas).pixelColor(120, 120), QColor(Qt::white));
}

void TestGeometryRuleCanvas::startingNewGuideClearsFitLayersAndMask() {
    GeometryRuleCanvas canvas;
    canvas.resize(240, 240);
    QImage image(240, 240, QImage::Format_RGB32);
    image.fill(Qt::white);
    canvas.setImage(image);
    const QJsonObject fitted{{"shape", "circle"}, {"cx", 120.0},
                             {"cy", 120.0}, {"r", 70.0}};
    const QJsonObject effective{{"shape", "circle"}, {"cx", 120.0},
                                {"cy", 120.0}, {"r", 66.0}};
    QImage mask(240, 240, QImage::Format_Grayscale8);
    mask.fill(255);
    canvas.setFitOverlay(QJsonObject{{"fitted_shape", fitted},
                                     {"effective_shape", effective}},
                         mask);
    canvas.setGuideVisible(false);
    canvas.setTool(GeometryRuleCanvas::Circle);
    QVERIFY(renderCanvas(&canvas).pixelColor(120, 120) != QColor(Qt::white));

    QTest::mousePress(&canvas, Qt::LeftButton, {}, QPoint(120, 120));

    QVERIFY(canvas.fittedShape().isEmpty());
    QVERIFY(canvas.effectiveShape().isEmpty());
    QVERIFY(canvas.fitShape().isEmpty());
    QVERIFY(!canvas.guideVisible());
    QVERIFY(canvas.maskOverlayVisible());
    QCOMPARE(renderCanvas(&canvas).pixelColor(120, 120), QColor(Qt::white));
    QTest::mouseRelease(&canvas, Qt::LeftButton, {}, QPoint(120, 120));
}

QTEST_MAIN(TestGeometryRuleCanvas)
#include "test_geometryrulecanvas.moc"
