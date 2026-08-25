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
    geometryrulecanvas.cpp \
    geometrymaskmanager.cpp \
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
    geometryrulecanvas.h \
    geometrymaskmanager.h \
    annotationcanvas.h \
    annotationmanager.h \
    annotationeditor.h \
    taskstatuswidget.h

FORMS += mainwindow.ui

RESOURCES += resources.qrc
