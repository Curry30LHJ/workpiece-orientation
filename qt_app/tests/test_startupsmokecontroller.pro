QT += testlib network
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8
TARGET = test_startupsmokecontroller
SOURCES += test_startupsmokecontroller.cpp ../startupsmokecontroller.cpp ../backendprocessmanager.cpp ../backendclient.cpp ../processlauncher.cpp ../appconfig.cpp
HEADERS += ../startupsmokecontroller.h ../backendprocessmanager.h ../backendclient.h ../processlauncher.h ../appconfig.h
