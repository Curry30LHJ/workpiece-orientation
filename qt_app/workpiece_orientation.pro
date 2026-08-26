QT += widgets network
CONFIG += c++17
TEMPLATE = app
TARGET = workpiece_orientation
QMAKE_CXXFLAGS += /utf-8

SOURCES += \
    main.cpp \
    appheader.cpp \
    apptheme.cpp \
    appconfig.cpp \
    backendclient.cpp \
    backendprocessmanager.cpp \
    processlauncher.cpp \
    mainwindow.cpp \
    inspectionimageview.cpp \
    inspectionpage.cpp \
    workpiecelibrarypage.cpp \
    geometryrulecanvas.cpp \
    geometryrulespage.cpp \
    annotationcanvas.cpp \
    annotationmanager.cpp \
    annotationeditor.cpp \
    taskstatuswidget.cpp

HEADERS += \
    appconfig.h \
    appheader.h \
    apptheme.h \
    backendclient.h \
    backendprocessmanager.h \
    processlauncher.h \
    mainwindow.h \
    inspectiontypes.h \
    inspectionimageview.h \
    inspectionpage.h \
    workpiecelibrarypage.h \
    geometryrulecanvas.h \
    geometryrulespage.h \
    annotationcanvas.h \
    annotationmanager.h \
    annotationeditor.h \
    taskstatuswidget.h

FORMS += mainwindow.ui \
         inspectionpage.ui \
         workpiecelibrarypage.ui

RESOURCES += resources.qrc
