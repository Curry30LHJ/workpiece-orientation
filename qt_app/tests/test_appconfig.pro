QT += testlib network
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_appconfig
SOURCES += test_appconfig.cpp \
           ../appconfig.cpp
HEADERS += ../appconfig.h
