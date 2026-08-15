QT += widgets network testlib
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_mainwindow
SOURCES += test_mainwindow.cpp \
           ../appconfig.cpp \
           ../backendclient.cpp \
           ../backendprocessmanager.cpp \
           ../processlauncher.cpp \
           ../mainwindow.cpp
HEADERS += ../appconfig.h \
           ../backendclient.h \
           ../backendprocessmanager.h \
           ../processlauncher.h \
           ../mainwindow.h
FORMS += ../mainwindow.ui
