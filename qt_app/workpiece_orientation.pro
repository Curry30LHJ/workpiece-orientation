QT += widgets network
CONFIG += c++17
TEMPLATE = app
TARGET = workpiece_orientation
QMAKE_CXXFLAGS += /utf-8

SOURCES += \
    main.cpp \
    appconfig.cpp \
    backendclient.cpp \
    backendprocessmanager.cpp \
    processlauncher.cpp \
    mainwindow.cpp \
    geometryrulecanvas.cpp \
    geometrymaskmanager.cpp \
    annotationcanvas.cpp \
    annotationmanager.cpp \
    annotationeditor.cpp

HEADERS += \
    appconfig.h \
    backendclient.h \
    backendprocessmanager.h \
    processlauncher.h \
    mainwindow.h \
    geometryrulecanvas.h \
    geometrymaskmanager.h \
    annotationcanvas.h \
    annotationmanager.h \
    annotationeditor.h

FORMS += mainwindow.ui
