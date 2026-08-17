QT += widgets testlib
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_geometrymaskmanager
SOURCES += test_geometrymaskmanager.cpp ../geometrymaskmanager.cpp ../geometryrulecanvas.cpp
HEADERS += ../geometrymaskmanager.h ../geometryrulecanvas.h
