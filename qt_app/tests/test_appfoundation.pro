QT += widgets testlib
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_appfoundation
INCLUDEPATH += ..
SOURCES += test_appfoundation.cpp \
           ../appheader.cpp \
           ../apptheme.cpp \
           ../taskstatuswidget.cpp
HEADERS += ../appheader.h \
           ../apptheme.h \
           ../taskstatuswidget.h
RESOURCES += ../resources.qrc
