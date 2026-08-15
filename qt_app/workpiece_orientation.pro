QT += widgets network
CONFIG += c++17
TEMPLATE = app
TARGET = workpiece_orientation
QMAKE_CXXFLAGS += /utf-8

SOURCES += \
    main.cpp \
    appconfig.cpp \
    backendclient.cpp \
    mainwindow.cpp

HEADERS += \
    appconfig.h \
    backendclient.h \
    mainwindow.h

FORMS += mainwindow.ui
