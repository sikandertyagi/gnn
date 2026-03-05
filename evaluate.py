import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def anomaly_scores(model, X, batch_size=256):
    """Return per-sequence reconstruction MSE for all sequences in X."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    loader = DataLoader(
        TensorDataset(torch.tensor(X).float()),
        batch_size=batch_size, shuffle=False
    )

    scores = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            recon = model(x)
            # MSE per sequence: mean over time-steps and features
            mse = torch.mean((x - recon) ** 2, dim=(1, 2))
            scores.extend(mse.cpu().numpy())

    return np.array(scores)


def evaluate_metrics(scores, labels):
    """
    Compute full anomaly-detection metrics.

    Training was on label 0 (normal).
    Anomalous = labels 1 or 2.
    Higher reconstruction score => more anomalous.
    """

    binary_labels = (labels != 0).astype(int)   # 0 = normal, 1 = anomaly

    auroc = roc_auc_score(binary_labels, scores)
    auprc = average_precision_score(binary_labels, scores)

    # Best threshold via Youden's J  (max TPR - FPR on the ROC curve)
    fpr, tpr, thresholds = roc_curve(binary_labels, scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    preds = (scores >= best_thresh).astype(int)

    precision = precision_score(binary_labels, preds, zero_division=0)
    recall    = recall_score(binary_labels, preds, zero_division=0)
    f1        = f1_score(binary_labels, preds, zero_division=0)
    cm        = confusion_matrix(binary_labels, preds)

    print("\n========== Anomaly Detection Metrics ==========")
    print(f"AUROC            : {auroc:.4f}")
    print(f"AUPRC            : {auprc:.4f}")
    print(f"Best Threshold   : {best_thresh:.6f}  (Youden's J)")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':15s}  Pred Normal  Pred Anomaly")
    print(f"{'Actual Normal':15s}  {cm[0, 0]:11d}  {cm[0, 1]:12d}")
    print(f"{'Actual Anomaly':15s}  {cm[1, 0]:11d}  {cm[1, 1]:12d}")
    print(f"\nClassification Report:")
    print(classification_report(binary_labels, preds, target_names=["Normal", "Anomaly"], zero_division=0))
    print("================================================\n")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "threshold": best_thresh,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
