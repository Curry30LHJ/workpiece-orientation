QT += widgets testlib
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_workpiecelibrarypage
SOURCES += test_workpiecelibrarypage.cpp \
           ../workpiecelibrarypage.cpp \
           ../taskstatuswidget.cpp
HEADERS += ../workpiecelibrarypage.h \
           ../appheader.h \
           ../taskstatuswidget.h
FORMS += ../workpiecelibrarypage.ui
