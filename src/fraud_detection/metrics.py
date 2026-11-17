import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def classification_metrics(labels, scores) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    return {
        "rows": len(labels),
        "fraud_count": int(labels.sum()),
        "fraud_rate": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, scores)),
    }
