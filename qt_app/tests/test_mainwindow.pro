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
           ../mainwindow.cpp \
           ../geometryrulecanvas.cpp \
           ../geometrymaskmanager.cpp \
           ../annotationcanvas.cpp \
           ../annotationmanager.cpp \
           ../annotationeditor.cpp
HEADERS += ../appconfig.h \
           ../backendclient.h \
           ../backendprocessmanager.h \
           ../processlauncher.h \
           ../mainwindow.h \
           ../geometryrulecanvas.h \
           ../geometrymaskmanager.h \
           ../annotationcanvas.h \
           ../annotationmanager.h \
           ../annotationeditor.h
FORMS += ../mainwindow.ui
