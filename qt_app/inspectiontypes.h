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
    QueuedFront, QueuedBack, Rejected, SubmitFailed
};

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
