QT += widgets testlib
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_geometryrulespage
SOURCES += test_geometryrulespage.cpp ../geometryrulespage.cpp ../geometryrulecanvas.cpp
HEADERS += ../geometryrulespage.h ../geometryrulecanvas.h
