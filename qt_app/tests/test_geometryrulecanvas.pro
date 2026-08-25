QT += widgets testlib
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_geometryrulecanvas
SOURCES += test_geometryrulecanvas.cpp ../geometryrulecanvas.cpp
HEADERS += ../geometryrulecanvas.h
