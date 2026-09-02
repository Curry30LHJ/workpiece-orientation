#pragma once

#include <QDateTime>
#include <QJsonObject>
#include <QString>

enum class InspectionMode { Single, Batch };
enum class InspectionUiState {
    Idle, Running, Completed, NeedsReview, Failed, BackendUnavailable
};
enum class BatchFilter { All, NeedsReview, Unprocessed, Failed };
enum class BatchDisposition {
    Pending, Submitting, PredictionFailed,
    QueuedFront, QueuedBack, Rejected, SubmitFailed, AlreadyStored
};

inline bool isAlreadyStoredErrorCode(const QString &code) {
    return code.trimmed() == QStringLiteral("DUPLICATE_TEMPLATE");
}

inline BatchDisposition confirmationFailureDisposition(const QString &code) {
    return isAlreadyStoredErrorCode(code)
        ? BatchDisposition::AlreadyStored : BatchDisposition::SubmitFailed;
}

struct InspectionRecord {
    QString id;
    QString imagePath;
    QString workpieceId;
    QJsonObject response;
    QString label;
    bool needsReview = false;
    double elapsedMs = 0.0;
    QDateTime completedAt;
    BatchDisposition disposition = BatchDisposition::Pending;
    QString evolutionJobId;
    QString error;
    QString submissionError;
};
