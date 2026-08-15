QT += widgets network
CONFIG += c++17
TEMPLATE = app
TARGET = workpiece_orientation
QMAKE_CXXFLAGS += /utf-8

SOURCES += \
    main.cpp \
    appconfig.cpp \
    mainwindow.cpp

HEADERS += \
    appconfig.h \
    mainwindow.h

FORMS += mainwindow.ui
