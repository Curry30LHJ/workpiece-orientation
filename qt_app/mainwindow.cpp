#include "mainwindow.h"

#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    ui->registerButton->setEnabled(false);
    ui->predictButton->setEnabled(false);
    ui->restartBackendButton->setEnabled(false);
    ui->resultLabel->setText(QStringLiteral("尚未检测"));
    ui->reviewLabel->clear();
}

MainWindow::~MainWindow() {
    delete ui;
}
