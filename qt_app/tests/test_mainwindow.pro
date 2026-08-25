QT += widgets network testlib
CONFIG += console c++17 testcase
TEMPLATE = app
QMAKE_CXXFLAGS += /utf-8

TARGET = test_mainwindow
SOURCES += test_mainwindow.cpp \
           ../appheader.cpp \
           ../appconfig.cpp \
           ../backendclient.cpp \
           ../backendprocessmanager.cpp \
           ../processlauncher.cpp \
           ../mainwindow.cpp \
           ../inspectionimageview.cpp \
           ../inspectionpage.cpp \
           ../workpiecelibrarypage.cpp \
           ../geometryrulecanvas.cpp \
           ../geometrymaskmanager.cpp \
           ../annotationcanvas.cpp \
           ../annotationmanager.cpp \
           ../annotationeditor.cpp \
           ../taskstatuswidget.cpp
HEADERS += ../appheader.h \
           ../appconfig.h \
           ../backendclient.h \
           ../backendprocessmanager.h \
           ../processlauncher.h \
           ../mainwindow.h \
           ../inspectiontypes.h \
           ../inspectionimageview.h \
           ../inspectionpage.h \
           ../workpiecelibrarypage.h \
           ../geometryrulecanvas.h \
           ../geometrymaskmanager.h \
           ../annotationcanvas.h \
           ../annotationmanager.h \
           ../annotationeditor.h \
           ../taskstatuswidget.h
FORMS += ../mainwindow.ui \
         ../inspectionpage.ui \
         ../workpiecelibrarypage.ui
