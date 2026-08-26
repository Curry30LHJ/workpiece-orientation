QT += widgets testlib
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_inspectionpage
SOURCES += test_inspectionpage.cpp \
           ../inspectionimageview.cpp \
           ../inspectionpage.cpp \
           ../apptheme.cpp
HEADERS += ../inspectiontypes.h \
           ../inspectionimageview.h \
           ../inspectionpage.h \
           ../apptheme.h
FORMS += ../inspectionpage.ui
RESOURCES += ../resources.qrc
