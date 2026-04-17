import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import mlflow


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_precision", precision)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    mlflow.log_metric("test_tn", int(tn))
    mlflow.log_metric("test_fp", int(fp))
    mlflow.log_metric("test_fn", int(fn))
    mlflow.log_metric("test_tp", int(tp))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    return acc, f1, recall, precision
