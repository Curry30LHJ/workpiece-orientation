QT += widgets testlib
CONFIG += c++17 testcase
TEMPLATE = app
TARGET = test_annotationmanager
QMAKE_CXXFLAGS += /utf-8

SOURCES += \
    $$PWD/../annotationcanvas.cpp \
    $$PWD/../annotationeditor.cpp \
    $$PWD/../annotationmanager.cpp \
    test_annotationmanager.cpp

HEADERS += \
    $$PWD/../annotationcanvas.h \
    $$PWD/../annotationeditor.h \
    $$PWD/../annotationmanager.h
