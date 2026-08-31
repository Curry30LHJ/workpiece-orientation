#pragma once

#include <QHash>
#include <QJsonObject>
#include <QStringList>
#include <QWidget>

#include "inspectiontypes.h"

class QEvent;

namespace Ui {
class InspectionPage;
}

class InspectionPage : public QWidget {
    Q_OBJECT

public:
    explicit InspectionPage(QWidget *parent = nullptr);
    ~InspectionPage() override;

    void setMode(InspectionMode mode);
    InspectionMode mode() const;
    InspectionUiState uiState() const;
    void setCurrentWorkpiece(const QString &id, const QString &name);
    void setBackendAvailable(bool available, bool busy, const QString &reason);
    void setBatchPredictionCapabilities(bool supported, bool ready, int workers);
    void setSingleImagePath(const QString &path);
    void clearBatchState();
    void beginBatch(const QStringList &paths, const QString &workpieceId);
    void setBatchFilter(BatchFilter filter);
    BatchFilter batchFilter() const;
    QString selectedRecordId() const;
    QStringList batchRecordIds() const;
    InspectionRecord recordForId(const QString &recordId) const;
    void requestBatchStop();
    int completedBatchCount() const;
    int failedBatchCount() const;
    bool batchRunning() const;
    void setRecordDisposition(const QString &recordId,
                              BatchDisposition disposition,
                              const QString &evolutionJobId = QString(),
                              const QString &error = QString());
    QString singleImagePath() const;
    void showSingleResult(const InspectionRecord &record);
    void showSingleFailure(const QString &message);
    void selectRecentRecord(const QString &recordId);
    void handleBackendResponse(const QString &command, const QJsonObject &response);
    void handleBackendFailure(const QString &command, const QString &code,
                              const QString &message);

signals:
    void commandRequested(const QString &command, const QJsonObject &fields);
    void confirmationRequested(const QString &recordId,
                               const QString &workpieceId,
                               const QString &imagePath,
                               const QString &orientation);
    void rejectionRequested(const QString &recordId);
    void batchFinished(bool stopped);

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    struct ModeState {
        QString imagePath;
        InspectionUiState uiState = InspectionUiState::Idle;
        InspectionRecord visibleRecord;
        bool hasRecord = false;
        QString message;
    };

    ModeState &activeState();
    const ModeState &activeState() const;
    void requestPrediction();
    void requestNextBatchPrediction();
    void requestBatchPrediction();
    void failBatchProtocol(const QString &message);
    void finishBatch(bool stopped, bool tableAlreadyRebuilt = false);
    void requestConfirmation(const QString &orientation);
    void rejectCurrentRecord();
    void storeRecentRecord(const InspectionRecord &record);
    void rebuildRecentList();
    void rebuildBatchTable();
    void selectBatchRecord(const QString &recordId, bool userInitiated);
    void selectPreferredBatchRecord(const QString &afterRecordId = QString(),
                                    bool rebuildTable = true);
    InspectionRecord *batchRecord(const QString &recordId);
    const InspectionRecord *batchRecord(const QString &recordId) const;
    bool matchesFilter(const InspectionRecord &record) const;
    void updateBatchSummary();
    void renderActiveState();
    void renderRecord(const InspectionRecord &record);
    void updateResponsivePresentation();
    void setCurrentResultTargetText(const QString &text);
    void updateActionOrder(const QString &predictedOrientation);
    void updateActionAvailability();
    void updateRecordDisposition(const QString &recordId, BatchDisposition disposition,
                                 const QString &error = QString());
    QString orientationText(const QString &label) const;
    QString dispositionText(const InspectionRecord &record) const;
    QString evidenceSummary(const InspectionRecord &record) const;
    QString rawEvidence(const InspectionRecord &record) const;

    Ui::InspectionPage *ui;
    InspectionMode mode_ = InspectionMode::Single;
    ModeState singleState_;
    ModeState batchState_;
    QString currentWorkpieceId_;
    QString currentWorkpieceName_;
    bool backendAvailable_ = false;
    bool backendBusy_ = false;
    bool backendStatusKnown_ = false;
    QString backendReason_;
    QHash<QString, InspectionRecord> recentRecords_;
    QStringList recentRecordOrder_;
    QString pendingPredictionImagePath_;
    QString pendingPredictionWorkpieceId_;
    QString pendingConfirmationRecordId_;
    QString pendingConfirmationOrientation_;
    QHash<QString, InspectionRecord> batchRecords_;
    QStringList batchRecordOrder_;
    QString currentBatchRequestId_;
    bool batchRequestInFlight_ = false;
    bool batchFallbackToScalar_ = false;
    bool batchItemsAccepted_ = false;
    QString batchFallbackReason_;
    double batchElapsedMs_ = 0.0;
    int batchWorkerCount_ = 0;
    bool batchWorkerCountKnown_ = false;
    bool batchPredictionSupported_ = false;
    bool batchPredictionReady_ = false;
    int batchPredictionWorkers_ = 0;
    QString selectedRecordId_;
    QString batchWorkpieceId_;
    BatchFilter batchFilter_ = BatchFilter::All;
    int batchCursor_ = 0;
    bool batchSelectionPinned_ = false;
    bool changingBatchSelection_ = false;
    bool batchRunning_ = false;
    bool stopRequested_ = false;
    bool compactResultLayout_ = false;
};
