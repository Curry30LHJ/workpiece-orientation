QT += testlib network
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_backendprocessmanager
SOURCES += test_backendprocessmanager.cpp \
           ../appconfig.cpp \
           ../backendclient.cpp \
           ../backendprocessmanager.cpp \
           ../processlauncher.cpp
HEADERS += ../appconfig.h \
           ../backendclient.h \
           ../backendprocessmanager.h \
           ../processlauncher.h
