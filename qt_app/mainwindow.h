#pragma once

#include <QMainWindow>
#include <QJsonObject>
#include <QStringList>

#include <functional>

#include "backendclient.h"

class BackendProcessManager;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    MainWindow(BackendClient *client, BackendProcessManager *manager, QWidget *parent = nullptr);
    ~MainWindow() override;

    void setTemplatePaths(const QStringList &frontPaths, const QStringList &backPaths);
    void setWorkpieceName(const QString &name);
    void setInspectionImagePath(const QString &path);
    void setReplaceConfirmationHandler(std::function<bool(const QString &)> handler);
    void setBackendError(const QString &message);

private slots:
    void chooseFrontTemplates();
    void chooseBackTemplates();
    void chooseInspectionImage();
    void refreshWorkpieces();
    void submitRegistration();
    void submitPrediction();
    void restartBackend();

    void onBackendReady();
    void onBackendUnavailable(const QString &reason);
    void onClientStateChanged(BackendClient::State state, const QString &detail);
    void onClientResponse(const QString &command, const QJsonObject &response);
    void onClientRequestFailed(const QString &code, const QString &message);
    void onConnectionLost(const QString &reason);

private:
    void initializeUi();
    void connectBackendSignals();
    void updateButtonStates();
    void updateTemplateLabels();
    void updatePreview();
    void clearInspectionState();
    void sendRegistration(bool replace);
    bool validateRegistration(QString *error) const;
    bool validateImagePath(const QString &path, QString *error) const;
    QStringList normalizedPaths(const QStringList &paths) const;
    QString selectedWorkpieceId() const;
    QString orientationText(const QString &label) const;
    QString decisionSourceText(const QString &source) const;
    QString formatScore(const QJsonObject &scores, const QString &key) const;
    void showLibraryMessage(const QString &message, bool error = false);

    Ui::MainWindow *ui;
    BackendClient *client_ = nullptr;
    BackendProcessManager *manager_ = nullptr;
    QStringList frontTemplatePaths_;
    QStringList backTemplatePaths_;
    QString inspectionImagePath_;
    QString pendingCommand_;
    QString pendingWorkpieceName_;
    bool pendingReplace_ = false;
    bool registrationInFlight_ = false;
    bool backendReady_ = false;
    bool clientBusy_ = false;
    std::function<bool(const QString &)> replaceConfirmationHandler_;
};
