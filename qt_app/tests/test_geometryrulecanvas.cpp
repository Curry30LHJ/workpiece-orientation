#include <QtTest/QtTest>

#include <QImage>
#include <QJsonObject>

#include "../geometryrulecanvas.h"

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
    canvas.setFitOverlay(QJsonObject{{"shape", "circle"}, {"cx", 60}, {"cy", 60}, {"r", 45}});
    QVERIFY(!canvas.fitShape().isEmpty());

    canvas.setImage(QImage(120, 120, QImage::Format_RGB32));

    QVERIFY(canvas.fitShape().isEmpty());
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

QTEST_MAIN(TestGeometryRuleCanvas)
#include "test_geometryrulecanvas.moc"
