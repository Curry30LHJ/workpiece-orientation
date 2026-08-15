QT += testlib network
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_backendclient
SOURCES += test_backendclient.cpp \
           ../backendclient.cpp
HEADERS += ../backendclient.h
