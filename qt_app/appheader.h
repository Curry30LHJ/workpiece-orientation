#pragma once

#include <QList>
#include <QMetaType>
#include <QPair>
#include <QString>
#include <QWidget>

class QLabel;
class QComboBox;
class QFrame;
class QPushButton;

enum class AppPage { Inspection = 0, WorkpieceLibrary = 1, GeometryRules = 2 };
Q_DECLARE_METATYPE(AppPage)

enum class BackendUiState { Disconnected, Loading, Ready, Busy, Error };

struct BackendStatusDetails {
    BackendUiState state = BackendUiState::Disconnected;
    QString connectionDetail;
    QString modelDetail;
    QString currentTask;
    QString recentError;
    bool canRestart = false;
};

class AppHeader : public QWidget {
    Q_OBJECT

public:
    explicit AppHeader(QWidget *parent = nullptr);
    void setCurrentPage(AppPage page);
    void setWorkpieces(const QList<QPair<QString, QString>> &items,
                       const QString &currentWorkpieceId);
    void setCurrentWorkpieceId(const QString &workpieceId);
    QString currentWorkpieceId() const;
    QString currentWorkpieceName() const;
    void setBackendState(BackendUiState state, const QString &detail);
    void setBackendDetails(const BackendStatusDetails &details);

signals:
    void pageRequested(AppPage page);
    void currentWorkpieceRequested(const QString &workpieceId);
    void backendDetailsRequested();
    void restartBackendRequested();

private:
    QPushButton *inspectionButton_;
    QPushButton *workpieceLibraryButton_;
    QPushButton *geometryRulesButton_;
    QComboBox *workpieceComboBox_;
    QLabel *backendStatusLabel_;
    QPushButton *backendDetailsButton_;
    QFrame *backendDetailsPanel_;
    QLabel *connectionDetailLabel_;
    QLabel *modelDetailLabel_;
    QLabel *currentTaskLabel_;
    QLabel *recentErrorLabel_;
    QPushButton *restartBackendButton_;
};
